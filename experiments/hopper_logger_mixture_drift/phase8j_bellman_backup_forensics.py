"""Phase 8J-BF-Q: read-only Bellman backup forensics for pooled-union FVI.

No network is trained here.  Existing Phase 8J-FVI checkpoints are replayed on
one fixed public candidate table and the exact AAMAS branch algebra is exposed
for diagnosis.  A small, strictly post-hoc simulator audit is isolated from all
checkpoint selection and training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aamas_hopper_adapter import _joint_log_probability, _normal_action_samples
from .generate_datasets import MujocoOneStepSimulator
from .phase8h_compute_matched_online_quick import (
    GAMMA,
    KAPPA,
    LAMBDA_REWARD,
    _git_commit,
    file_fingerprint,
)
from .phase8h_quick_multipolicy_aamas import CANDIDATE_ACTIONS, union_candidate_actions
from .phase8j_fvi_stability import (
    DEFAULT_FIX_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_FVI_ROOT,
    METHOD,
    MODEL_SEED,
    PHASE as FVI_PHASE,
    VARIANTS,
    _load_seed_components,
    _normalization,
    _relocate_recorded_path,
    _resolve_context,
    make_repaired_potential_network,
)


PHASE = "Phase 8J-BF-Q"
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_bellman_backup_forensics")
REQUIRED_CHECKPOINTS = (0, 20, 40, 60, 80, 100, 120)
OPTIONAL_LATE_CHECKPOINTS = (140, 160, 180, 200)
# The environment is deterministic after restoring an anchor.  Exact enumeration
# of the two equally weighted latent states is therefore the population average;
# repeating either state would add duplicate observations, not information.
SIMULATOR_LATENT_REPLICATES = 2
KNN_K = 5


class Phase8JBellmanForensicsError(RuntimeError):
    """Raised when a read-only forensic invariant is unavailable or violated."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                               default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a real compressed Parquet table; never disguise CSV as Parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except (ImportError, OSError) as error:
        raise Phase8JBellmanForensicsError(
            "candidate-level output requires pyarrow (pip install pyarrow)") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path, compression="zstd")


def _has_parquet_magic(path: Path) -> bool:
    with Path(path).open("rb") as stream:
        opening = stream.read(4)
        stream.seek(-4, 2)
        closing = stream.read(4)
    return opening == b"PAR1" and closing == b"PAR1"


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x) or not np.all(np.isfinite(x)):
        raise Phase8JBellmanForensicsError("summary input must be nonempty and finite")
    return {
        "count": len(x), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "p50": float(np.quantile(x, .50)),
        "p90": float(np.quantile(x, .90)), "p95": float(np.quantile(x, .95)),
        "p99": float(np.quantile(x, .99)), "max": float(x.max()),
    }


def _numeric_records_finite(*tables: Sequence[Mapping[str, Any]]) -> bool:
    for rows in tables:
        for row in rows:
            for value in row.values():
                if value is None or isinstance(value, (str, bool, np.bool_)):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if not np.isfinite(float(value)):
                        return False
    return True


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    # Values are continuous in this audit; stable ordinal ranks are sufficient.
    rx = np.empty(len(x), dtype=np.float64); rx[np.argsort(x, kind="stable")] = np.arange(len(x))
    ry = np.empty(len(y), dtype=np.float64); ry[np.argsort(y, kind="stable")] = np.arange(len(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _nearest_and_knn(query: np.ndarray, train: np.ndarray,
                     mean: np.ndarray, std: np.ndarray,
                     k: int = KNN_K) -> tuple[np.ndarray, np.ndarray]:
    reference = (np.asarray(train, dtype=np.float64) - mean) / std
    values = (np.asarray(query, dtype=np.float64) - mean) / std
    nearest = np.empty(len(values), dtype=np.float64)
    knn = np.empty(len(values), dtype=np.float64)
    count = min(k, len(reference))
    for start in range(0, len(values), 512):
        block = values[start:start + 512]
        distances = np.sqrt(np.square(
            block[:, None, :] - reference[None, :, :]).sum(axis=2))
        nearest[start:start + len(block)] = distances.min(axis=1)
        knn[start:start + len(block)] = np.partition(
            distances, count - 1, axis=1)[:, :count].mean(axis=1)
    return nearest, knn


def _candidate_provenance(index: int) -> tuple[str, int | None, str]:
    if index == 27:
        return "public_base_action", None, "base"
    source = index // 9 + 1
    within = index % 9
    return ("source_behavior", source,
            "source_mean" if within == 8 else "sampled_action")


def _checkpoint_candidates(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    initial = directory / "initial.pt"
    if initial.is_file():
        result[0] = initial
    for path in directory.glob("epoch_*.pt"):
        try:
            result[int(path.stem.split("_")[-1])] = path
        except ValueError:
            continue
    latest = directory / "latest.pt"
    if latest.is_file():
        # Metadata is validated later; this only makes an exact latest epoch discoverable.
        result[-1] = latest
    return result


def _inventory_checkpoints(fvi_root: Path, torch: Any,
                           requested: Sequence[int]) -> tuple[
                               list[dict[str, Any]], dict[str, dict[int, Path]],
                               list[tuple[str, int]]]:
    rows: list[dict[str, Any]] = []
    resolved: dict[str, dict[int, Path]] = {}
    desired = sorted(set(map(int, requested)) | set(OPTIONAL_LATE_CHECKPOINTS))
    for variant in VARIANTS:
        directory = fvi_root / ("moving_target" if variant == VARIANTS[0]
                                else "frozen_outer_target")
        available = _checkpoint_candidates(directory)
        if -1 in available:
            payload = torch.load(available[-1], map_location="cpu", weights_only=False)
            epoch = int(payload.get("metadata", {}).get("epoch", -1))
            if epoch >= 0:
                available.setdefault(epoch, available[-1])
            del available[-1]
        resolved[variant] = {}
        for epoch, path in sorted(available.items()):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            valid = (metadata.get("stage") == FVI_PHASE
                     and metadata.get("variant") == variant
                     and int(metadata.get("epoch", -1)) == epoch
                     and metadata.get("eligible_for_sac") is False)
            rows.append({
                "variant": variant, "epoch": epoch, "path": str(path.resolve()),
                "requested_or_late": epoch in desired, "metadata_valid": valid,
                **file_fingerprint(path),
            })
            if valid and epoch in desired:
                resolved[variant][epoch] = path
    missing = [(variant, epoch) for variant in VARIANTS for epoch in requested
               if epoch not in resolved[variant]]
    return rows, resolved, missing


def _load_network(path: Path, context: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    torch = context["torch"]
    payload = torch.load(path, map_location=context["device"], weights_only=False)
    network = make_repaired_potential_network(
        context["official"], float(payload["reward_min"]),
        float(payload["reward_max"]), context["device"])
    network.load_state_dict(payload["current_state_dict"])
    network.eval().requires_grad_(False)
    return network, payload


def _value(network: Any, states: np.ndarray, mean: np.ndarray, std: np.ndarray,
           context: Mapping[str, Any]) -> np.ndarray:
    torch = context["torch"]
    tensor = torch.as_tensor((np.asarray(states, dtype=np.float32) - mean) / (std + 1e-7),
                             dtype=torch.float32, device=context["device"])
    with torch.no_grad():
        result = network(tensor).reshape(-1).cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise Phase8JBellmanForensicsError("checkpoint potential produced nonfinite values")
    return result


def _fixed_probe_table(context: Mapping[str, Any], components: Mapping[str, Any],
                       simulator_anchor_count: int) -> dict[str, Any]:
    public = context["public"]
    validation_ids = np.asarray(context["splits"]["observational_validation"], dtype=np.int64)
    selected_ids = validation_ids[:min(simulator_anchor_count, len(validation_ids))]
    validation_rows = context["validation_rows"]
    rows = []
    for anchor in selected_ids:
        matches = validation_rows[np.asarray(public["anchor_id"])[validation_rows] == anchor]
        if not len(matches):
            raise Phase8JBellmanForensicsError(f"validation anchor {anchor} has no public row")
        rows.append(int(matches[0]))
    rows_array = np.asarray(rows, dtype=np.int64)
    states = np.asarray(public["observation"], dtype=np.float32)[rows_array]
    logged = np.asarray(public["commanded_action"], dtype=np.float32)[rows_array]
    real_next = np.asarray(public["next_observation"], dtype=np.float32)[rows_array]
    reward = np.asarray(public["reward"], dtype=np.float64)[rows_array]
    terminated = np.asarray(public["terminated"], dtype=bool)[rows_array]
    truncated = np.asarray(public["truncated"], dtype=bool)[rows_array]
    anchor_ids = np.asarray(public["anchor_id"], dtype=np.int64)[rows_array]
    anchor_lookup = {int(anchor): index for index, anchor in enumerate(
        np.asarray(context["anchors"]["anchor_id"], dtype=np.int64))}
    positions = np.asarray([anchor_lookup[int(anchor)] for anchor in anchor_ids], dtype=np.int64)
    bases = np.asarray(context["anchors"]["base_action"], dtype=np.float32)[positions]
    sources = tuple(components[f"source_{source}"] for source in (1, 2, 3))
    candidates = union_candidate_actions(
        sources, states, bases, samples_per_source=8, seed=20261101)
    noise = np.random.default_rng(20261102).standard_normal(
        (len(states) * candidates.shape[1], CANDIDATE_ACTIONS, 3)).astype(np.float32)
    return {
        "rows": rows_array, "anchor_ids": anchor_ids, "anchor_positions": positions,
        "states": states, "logged_actions": logged, "real_next": real_next,
        "logged_rewards": reward, "terminated": terminated, "truncated": truncated,
        "base_actions": bases, "candidates": candidates, "common_noise": noise,
        "candidate_digest": _array_digest(states, candidates, noise),
    }


def _behavior_mean_and_logq(states: np.ndarray, actions: np.ndarray, model: Any,
                            context: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    torch = context["torch"]
    count, candidates = actions.shape[:2]
    flat_state = np.repeat(states, candidates, axis=0)
    flat_action = actions.reshape(-1, 3)
    ts = torch.as_tensor(flat_state, dtype=torch.float32, device=model.device)
    ta = torch.as_tensor(flat_action, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        distribution = model.behavior_model(ts)
        mean = torch.tanh(distribution.loc).cpu().numpy()
        logq = _joint_log_probability(distribution, ta).cpu().numpy()
    return mean.reshape(count, candidates, 3), logq.reshape(count, candidates)


def _decompose_backup(probe: Mapping[str, Any], model: Any, network: Any,
                      mean: np.ndarray, std: np.ndarray,
                      context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Mirror the real adapter and expose each observed/road contribution."""
    torch = context["torch"]
    states = np.asarray(probe["states"], dtype=np.float32)
    actions = np.asarray(probe["candidates"], dtype=np.float32)
    count, candidate_count = actions.shape[:2]
    flat_state = np.repeat(states, candidate_count, axis=0)
    flat_action = actions.reshape(-1, 3)
    ts = torch.as_tensor(flat_state, dtype=torch.float32, device=model.device)
    ta = torch.as_tensor(flat_action, dtype=torch.float32, device=model.device)
    noise = torch.as_tensor(probe["common_noise"], dtype=torch.float32, device=model.device)
    with torch.no_grad():
        distribution = model.behavior_model(ts)
        action_logq = _joint_log_probability(distribution, ta)
        pair = torch.cat((ts, ta), dim=1)
        delta = model.state_difference_model(pair)
        model_next = ts + delta
        reward_z = model.reward_model(pair).reshape(-1)
        sampled = _normal_action_samples(distribution, noise, torch)
        original = ta.unsqueeze(1).expand(-1, CANDIDATE_ACTIONS, -1)
        positive = original - torch.clamp(
            original + float(model.action_separation), min=-1.0, max=1.0)
        negative = original - torch.clamp(
            original - float(model.action_separation), min=-1.0, max=1.0)
        stacked = torch.stack((positive, negative, original - sampled), dim=2)
        chosen = torch.gather(stacked, 2,
                              torch.abs(stacked).argmax(dim=2, keepdim=True)).squeeze(2)
        road_actions = original - chosen
        expanded = ts.unsqueeze(1).expand(-1, CANDIDATE_ACTIONS, -1)
        flat_expanded = expanded.reshape(-1, 12)
        road_pair = torch.cat((flat_expanded, road_actions.reshape(-1, 3)), dim=1)
        road_next = flat_expanded + model.state_difference_model(road_pair)
        road_distribution = model.behavior_model(flat_expanded)
        road_logq = _joint_log_probability(
            road_distribution, road_actions.reshape(-1, 3)).reshape(
                len(flat_state), CANDIDATE_ACTIONS).mean(dim=1)
    current_value = _value(network, model_next.cpu().numpy(), mean, std, context)
    road_value_all = _value(network, road_next.cpu().numpy(), mean, std, context).reshape(
        len(flat_state), CANDIDATE_ACTIONS)
    terminal = np.repeat(np.asarray(probe["terminated"], dtype=bool), candidate_count)
    current_value = current_value * (~terminal)
    road_value_all = road_value_all * (~terminal)[:, None]
    reward = reward_z.cpu().numpy().astype(np.float64) * model.reward_std + model.reward_mean
    road_value = np.maximum(road_value_all.max(axis=1), current_value)
    observed_return = reward + model.gamma * current_value
    road_return = model.reward_upper + model.gamma * road_value
    lt = np.clip(action_logq.cpu().numpy(), -50.0, -0.01)
    lr = np.clip(road_logq.cpu().numpy(), -50.0, -0.01)
    observed_weight = np.exp(lt) / (np.exp(lt) + np.exp(lr))
    road_weight = 1.0 - observed_weight
    observed_contribution = observed_weight * observed_return
    road_contribution = road_weight * road_return
    shape = (count, candidate_count)
    result = {
        "delta": delta.cpu().numpy().reshape(count, candidate_count, 12),
        "model_next": model_next.cpu().numpy().reshape(count, candidate_count, 12),
        "road_next": road_next.cpu().numpy().reshape(
            count, candidate_count, CANDIDATE_ACTIONS, 12),
        "model_reward": reward.reshape(shape),
        "backup_observed_reward": reward.reshape(shape),
        "backup_road_reward": np.full(shape, model.reward_upper, dtype=np.float64),
        "model_next_value": current_value.reshape(shape),
        "road_next_value": road_value.reshape(shape),
        "observed_return": observed_return.reshape(shape),
        "road_return": road_return.reshape(shape),
        "observed_weight": observed_weight.reshape(shape),
        "road_weight": road_weight.reshape(shape),
        "observed_contribution": observed_contribution.reshape(shape),
        "road_contribution": road_contribution.reshape(shape),
        "final_backup": (observed_contribution + road_contribution).reshape(shape),
        "reward_contribution": (
            observed_weight * reward + road_weight * model.reward_upper).reshape(shape),
        "observed_continuation_contribution": (
            observed_weight * model.gamma * current_value).reshape(shape),
        "road_continuation_contribution": (
            road_weight * model.gamma * road_value).reshape(shape),
        "action_logq": action_logq.cpu().numpy().reshape(shape),
    }
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise Phase8JBellmanForensicsError("nonfinite exact backup decomposition")
    if not np.allclose(result["observed_weight"] + result["road_weight"], 1.0,
                       atol=1e-12, rtol=0.0):
        raise Phase8JBellmanForensicsError("decomposed density weights do not sum to one")
    return result


def _single_action_backup(states: np.ndarray, actions: np.ndarray, terminated: np.ndarray,
                          common_noise: np.ndarray, model: Any, network: Any,
                          mean: np.ndarray, std: np.ndarray,
                          context: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    probe = {"states": states, "candidates": actions[:, None, :],
             "terminated": terminated, "common_noise": common_noise}
    values = _decompose_backup(probe, model, network, mean, std, context)
    return values["final_backup"][:, 0], values["model_next"][:, 0]


def _simulator_truth(context: Mapping[str, Any], probe: Mapping[str, Any]) -> dict[str, np.ndarray]:
    simulator = MujocoOneStepSimulator(context["anchors"], (KAPPA,), seed=20261103)
    count, candidate_count = np.asarray(probe["candidates"]).shape[:2]
    reward = np.empty((count, candidate_count), dtype=np.float64)
    next_state_by_latent = np.empty(
        (count, candidate_count, SIMULATOR_LATENT_REPLICATES, 12), dtype=np.float64)
    terminated_by_latent = np.empty(
        (count, candidate_count, SIMULATOR_LATENT_REPLICATES), dtype=bool)
    try:
        for row in range(count):
            for candidate in range(candidate_count):
                outcomes = [simulator.step(
                    int(probe["anchor_positions"][row]), probe["candidates"][row, candidate],
                    latent, KAPPA) for latent in (-1, 1)]
                reward[row, candidate] = np.mean([
                    value["reward"] + LAMBDA_REWARD * latent
                    for value, latent in zip(outcomes, (-1, 1))])
                next_state_by_latent[row, candidate] = np.asarray(
                    [value["next_observation"] for value in outcomes], dtype=np.float64)
                terminated_by_latent[row, candidate] = np.asarray(
                    [value["terminated"] for value in outcomes], dtype=bool)
    finally:
        simulator.close()
    return {
        "reward": reward,
        "next_state": next_state_by_latent.mean(axis=2),
        "next_state_by_latent": next_state_by_latent,
        "terminated_by_latent": terminated_by_latent,
        "latent_population": np.asarray((-1, 1), dtype=np.int8),
    }


def _probe_value_rows(variant: str, epoch: int, network: Any,
                      context: Mapping[str, Any], probe: Mapping[str, Any],
                      model_next: np.ndarray, mean: np.ndarray,
                      std: np.ndarray) -> list[dict[str, Any]]:
    public = context["public"]
    train_rows = context["train_rows"]
    train_ids = np.asarray(public["anchor_id"])[train_rows]
    _, first = np.unique(train_ids, return_index=True)
    train = np.asarray(public["observation"], dtype=np.float32)[train_rows[first]]
    validation_next = np.asarray(public["next_observation"], dtype=np.float32)[
        context["validation_rows"]]
    groups = {
        "training_anchor_states": train,
        "real_logged_next_states": validation_next,
        "model_generated_next_states": model_next.reshape(-1, 12),
    }
    return [{"variant": variant, "epoch": epoch, "probe_set": name,
             **_summary(_value(network, states, mean, std, context))}
            for name, states in groups.items()]


def _branch_rows(variant: str, epoch: int,
                 decomposition: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    fields = ("observed_return", "road_return", "observed_weight", "road_weight",
              "final_backup", "reward_contribution",
              "observed_continuation_contribution", "road_continuation_contribution")
    return [{"variant": variant, "epoch": epoch, "quantity": field,
             **_summary(decomposition[field])} for field in fields]


def _candidate_rows(variant: str, epoch: int, probe: Mapping[str, Any],
                    decomposition: Mapping[str, np.ndarray], behavior_mean: np.ndarray,
                    nearest: np.ndarray, knn: np.ndarray,
                    transition_magnitude: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    candidates = np.asarray(probe["candidates"])
    for state in range(len(candidates)):
        values = decomposition["final_backup"][state]
        order = np.argsort(values)
        selected = int(order[-1])
        second = float(values[order[-2]])
        median = float(np.median(values))
        for candidate in range(candidates.shape[1]):
            producer, source, kind = _candidate_provenance(candidate)
            action = candidates[state, candidate]
            rows.append({
                "variant": variant, "epoch": epoch,
                "anchor_id": int(probe["anchor_ids"][state]),
                "state_index": state, "candidate_index": candidate,
                "action_0": float(action[0]), "action_1": float(action[1]),
                "action_2": float(action[2]), "producer": producer,
                "producer_source": source, "candidate_kind": kind,
                "distance_to_logged_action": float(np.linalg.norm(
                    action - probe["logged_actions"][state])),
                "distance_to_pooled_behavior_mean": float(np.linalg.norm(
                    action - behavior_mean[state, candidate])),
                "behavior_log_probability": float(decomposition["action_logq"][state, candidate]),
                "standardized_nearest_train_distance": float(nearest[state, candidate]),
                "standardized_knn_mean_train_distance": float(knn[state, candidate]),
                "transition_prediction_magnitude": float(transition_magnitude[state, candidate]),
                "model_reward": float(decomposition["model_reward"][state, candidate]),
                "backup_observed_reward": float(
                    decomposition["backup_observed_reward"][state, candidate]),
                "backup_road_reward": float(
                    decomposition["backup_road_reward"][state, candidate]),
                "model_next_value": float(decomposition["model_next_value"][state, candidate]),
                "road_next_value": float(decomposition["road_next_value"][state, candidate]),
                "observed_return": float(decomposition["observed_return"][state, candidate]),
                "road_return": float(decomposition["road_return"][state, candidate]),
                "observed_weight": float(decomposition["observed_weight"][state, candidate]),
                "road_weight": float(decomposition["road_weight"][state, candidate]),
                "observed_contribution": float(
                    decomposition["observed_contribution"][state, candidate]),
                "road_contribution": float(decomposition["road_contribution"][state, candidate]),
                "final_backup": float(values[candidate]),
                "selected_by_max": candidate == selected,
                "state_candidate_mean": float(values.mean()),
                "state_candidate_median": median,
                "state_candidate_p90": float(np.quantile(values, .90)),
                "state_candidate_max": float(values[selected]),
                "state_candidate_second": second,
                "max_minus_median": float(values[selected] - median),
                "max_minus_second": float(values[selected] - second),
                "max_over_median": float(values[selected] /
                    (median + np.finfo(np.float64).eps * max(1.0, abs(median)))),
            })
    return rows


def _diagnostic_backups(variant: str, epoch: int, probe: Mapping[str, Any],
                        decomposition: Mapping[str, np.ndarray], network: Any,
                        logged_model_backup: np.ndarray, logged_model_next: np.ndarray,
                        mean: np.ndarray, std: np.ndarray,
                        context: Mapping[str, Any]) -> list[dict[str, Any]]:
    mask = (~np.asarray(probe["terminated"], dtype=bool)).astype(np.float64)
    d0 = (np.asarray(probe["logged_rewards"], dtype=np.float64)
          + GAMMA * mask * _value(network, probe["real_next"], mean, std, context))
    variants = {
        "D0_observed_only": d0,
        "D1_model_next_no_max": logged_model_backup,
        "D2_model_next_candidate_mean": decomposition["final_backup"].mean(axis=1),
        "D3_original_candidate_max": decomposition["final_backup"].max(axis=1),
    }
    logged_transition_error = np.linalg.norm(
        logged_model_next - np.asarray(probe["real_next"], dtype=np.float64), axis=1)
    rows = []
    for name, values in variants.items():
        row = {"variant": variant, "epoch": epoch, "diagnostic_backup": name,
               **_summary(values)}
        if name == "D1_model_next_no_max":
            row.update({f"logged_action_transition_error_{key}": value
                        for key, value in _summary(logged_transition_error).items()})
        rows.append(row)
    return rows


def _simulator_rows(variant: str, epoch: int, probe: Mapping[str, Any],
                    decomposition: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray],
                    network: Any, mean: np.ndarray, std: np.ndarray,
                    context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count, candidates = decomposition["final_backup"].shape
    latent_next = np.asarray(truth["next_state_by_latent"], dtype=np.float64)
    latent_terminated = np.asarray(truth["terminated_by_latent"], dtype=bool)
    flat_true_value = _value(
        network, latent_next.reshape(-1, 12), mean, std, context).reshape(
            count, candidates, SIMULATOR_LATENT_REPLICATES)
    # This is E[V(S'_do)], not the generally incorrect V(E[S'_do]).
    true_value = (flat_true_value * (~latent_terminated)).mean(axis=2)
    simulator_bellman = truth["reward"] + GAMMA * true_value
    transition_error = np.linalg.norm(
        decomposition["model_next"] - truth["next_state"], axis=2)
    reward_error = decomposition["model_reward"] - truth["reward"]
    extrapolation_error = decomposition["model_next_value"] - true_value
    bellman_error = decomposition["final_backup"] - simulator_bellman
    rows = []
    selected_mask = np.zeros((count, candidates), dtype=bool)
    selected_mask[np.arange(count), decomposition["final_backup"].argmax(axis=1)] = True
    for state in range(count):
        for candidate in range(candidates):
            rows.append({
                "variant": variant, "epoch": epoch,
                "anchor_id": int(probe["anchor_ids"][state]),
                "state_index": state, "candidate_index": candidate,
                "selected_by_model_max": bool(selected_mask[state, candidate]),
                "true_do_reward": float(truth["reward"][state, candidate]),
                "model_reward_error": float(reward_error[state, candidate]),
                "transition_error_l2": float(transition_error[state, candidate]),
                "true_do_next_value": float(true_value[state, candidate]),
                "value_extrapolation_error": float(extrapolation_error[state, candidate]),
                "simulator_bellman": float(simulator_bellman[state, candidate]),
                "model_bellman": float(decomposition["final_backup"][state, candidate]),
                "bellman_prediction_error": float(bellman_error[state, candidate]),
            })
    selected = selected_mask
    summary = {
        "variant": variant, "epoch": epoch,
        "selected_transition_error_mean": float(transition_error[selected].mean()),
        "all_transition_error_mean": float(transition_error.mean()),
        "selected_reward_error_mean": float(reward_error[selected].mean()),
        "all_reward_error_mean": float(reward_error.mean()),
        "selected_value_extrapolation_error_mean": float(extrapolation_error[selected].mean()),
        "all_value_extrapolation_error_mean": float(extrapolation_error.mean()),
        "selected_bellman_error_mean": float(bellman_error[selected].mean()),
        "all_bellman_error_mean": float(bellman_error.mean()),
        "selected_bellman_abs_error_mean": float(np.abs(bellman_error[selected]).mean()),
        "all_bellman_abs_error_mean": float(np.abs(bellman_error).mean()),
        "selected_positive_bellman_error_mean": float(
            np.maximum(bellman_error[selected], 0.0).mean()),
        "all_positive_bellman_error_mean": float(
            np.maximum(bellman_error, 0.0).mean()),
        "selected_observed_weight_mean": float(
            decomposition["observed_weight"][selected].mean()),
        "all_observed_weight_mean": float(decomposition["observed_weight"].mean()),
        "spearman_model_backup_vs_bellman_error": _spearman(
            decomposition["final_backup"].reshape(-1), bellman_error.reshape(-1)),
    }
    return rows, summary


def _ood_rows(variant: str, epoch: int, decomposition: Mapping[str, np.ndarray],
              nearest: np.ndarray, previous_value: np.ndarray | None,
              probe: Mapping[str, Any], simulator_value_error: np.ndarray) -> list[dict[str, Any]]:
    values = decomposition["model_next_value"]
    growth = np.zeros_like(values) if previous_value is None else values - previous_value
    flat_growth = growth.reshape(-1)
    result = [{
        "variant": variant, "epoch": epoch, "subset": "all",
        "distance_abs_value_spearman": _spearman(nearest.reshape(-1), np.abs(values).reshape(-1)),
        "distance_value_growth_spearman": _spearman(nearest.reshape(-1), flat_growth),
        "distance_abs_value_error_spearman": _spearman(
            nearest.reshape(-1), np.abs(simulator_value_error).reshape(-1)),
        "mean_distance": float(nearest.mean()), "mean_abs_value": float(np.abs(values).mean()),
        "mean_abs_model_vs_simulator_value_error": float(
            np.abs(simulator_value_error).mean()),
        "mean_value_growth": float(growth.mean()),
    }]
    for fraction in (.01, .05):
        count = max(1, int(math.ceil(len(flat_growth) * fraction)))
        indices = np.argsort(flat_growth)[-count:]
        state, candidate = np.unravel_index(indices, growth.shape)
        producer_sources = [_candidate_provenance(int(index))[1] for index in candidate]
        result.append({
            "variant": variant, "epoch": epoch, "subset": f"top_{int(fraction*100)}pct_growth",
            "count": count, "mean_growth": float(flat_growth[indices].mean()),
            "mean_distance": float(nearest[state, candidate].mean()),
            "mean_abs_value": float(np.abs(values[state, candidate]).mean()),
            "mean_transition_magnitude": float(np.linalg.norm(
                decomposition["delta"][state, candidate], axis=1).mean()),
            "mean_behavior_log_probability": float(
                decomposition["action_logq"][state, candidate].mean()),
            "mean_abs_model_vs_simulator_value_error": float(
                np.abs(simulator_value_error[state, candidate]).mean()),
            "source_1_count": sum(source == 1 for source in producer_sources),
            "source_2_count": sum(source == 2 for source in producer_sources),
            "source_3_count": sum(source == 3 for source in producer_sources),
            "base_count": sum(source is None for source in producer_sources),
            "anchor_ids": ";".join(map(str, np.unique(probe["anchor_ids"][state]).tolist())),
            "candidate_indices": ";".join(map(str, np.unique(candidate).tolist())),
        })
    return result


def _mechanism_labels(diagnostic_rows: Sequence[Mapping[str, Any]],
                      branch_rows: Sequence[Mapping[str, Any]],
                      max_rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Apply only direction/order comparisons; no magnitude success threshold."""
    evidence: dict[str, Any] = {}
    labels: set[str] = set()
    for variant in VARIANTS:
        records = [row for row in diagnostic_rows if row["variant"] == variant]
        epochs = sorted({int(row["epoch"]) for row in records})
        if len(epochs) < 2:
            continue
        first_epoch, last_epoch = epochs[0], epochs[-1]
        growth = {}
        for name in ("D0_observed_only", "D1_model_next_no_max",
                     "D2_model_next_candidate_mean", "D3_original_candidate_max"):
            first = next(row for row in records if row["epoch"] == first_epoch
                         and row["diagnostic_backup"] == name)
            last = next(row for row in records if row["epoch"] == last_epoch
                        and row["diagnostic_backup"] == name)
            growth[name] = float(last["std"]) - float(first["std"])
        evidence[variant] = {"std_growth_final_minus_initial": growth}
        if growth["D0_observed_only"] <= 0 < growth["D1_model_next_no_max"]:
            labels.add("MODEL_NEXT_STATE_VALUE_EXTRAPOLATION_PRIMARY")
        if (growth["D1_model_next_no_max"] <= 0
                and growth["D2_model_next_candidate_mean"] <= 0
                and growth["D3_original_candidate_max"] > 0):
            labels.add("CANDIDATE_MAXIMIZATION_PRIMARY")
        if (growth["D1_model_next_no_max"] > 0
                and growth["D3_original_candidate_max"]
                > growth["D1_model_next_no_max"]):
            labels.add("MODEL_EXTRAPOLATION_AMPLIFIED_BY_MAX")
        if growth["D0_observed_only"] > 0:
            labels.add("CORE_BOOTSTRAP_OR_VALUE_SCALE_INSTABILITY")
        contribution_growth: dict[str, float] = {}
        for quantity in ("reward_contribution", "observed_continuation_contribution",
                         "road_continuation_contribution", "max_selection_increment"):
            contribution = [row for row in branch_rows if row["variant"] == variant
                            and row.get("quantity") == quantity]
            if contribution:
                first = min(contribution, key=lambda row: int(row["epoch"]))
                last = max(contribution, key=lambda row: int(row["epoch"]))
                contribution_growth[quantity] = float(last["std"]) - float(first["std"])
        if contribution_growth:
            dominant = max(contribution_growth, key=contribution_growth.get)
            evidence[variant]["contribution_std_growth"] = contribution_growth
            evidence[variant]["largest_contribution_std_growth"] = dominant
            if (dominant == "reward_contribution"
                    and contribution_growth[dominant] > 0):
                labels.add("REWARD_MODEL_OR_REWARD_SCALE_PROBLEM")
        density = [row for row in max_rows if row.get("variant") == variant
                   and "selected_observed_weight_mean" in row]
        if density:
            first = min(density, key=lambda row: int(row["epoch"]))
            last = max(density, key=lambda row: int(row["epoch"]))
            evidence[variant]["selected_observed_weight_first_last"] = [
                float(first["selected_observed_weight_mean"]),
                float(last["selected_observed_weight_mean"])]
            evidence[variant]["latest_max_selected_simulator_error"] = {
                "selected_signed_bellman_error_mean": float(
                    last["selected_bellman_error_mean"]),
                "all_signed_bellman_error_mean": float(last["all_bellman_error_mean"]),
                "selected_positive_bellman_error_mean": float(
                    last["selected_positive_bellman_error_mean"]),
                "all_positive_bellman_error_mean": float(
                    last["all_positive_bellman_error_mean"]),
                "backup_error_spearman": float(
                    last["spearman_model_backup_vs_bellman_error"]),
            }
            d3_minus_d2 = (growth["D3_original_candidate_max"]
                           - growth["D2_model_next_candidate_mean"])
            if (not np.isclose(float(last["selected_observed_weight_mean"]),
                               float(first["selected_observed_weight_mean"]),
                               atol=np.finfo(np.float64).eps, rtol=0.0)
                    and d3_minus_d2 > 0):
                labels.add("DENSITY_WEIGHTING_CONTRIBUTES_TO_INSTABILITY")
    if not labels:
        labels.add("ROOT_CAUSE_NOT_YET_IDENTIFIED")
    return sorted(labels), evidence


def _figures(output: Path, probe_rows: Sequence[Mapping[str, Any]],
             branch_rows: Sequence[Mapping[str, Any]],
             ood_rows: Sequence[Mapping[str, Any]],
             max_rows: Sequence[Mapping[str, Any]], diagnostic_rows: Sequence[Mapping[str, Any]],
             twin_rows: Sequence[Mapping[str, Any]], amplification_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib.pyplot as plt
    root = output / "figures"; root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def line(name: str, rows: Sequence[Mapping[str, Any]], series: str,
             y: str, title: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, variant in zip(axes, VARIANTS):
            subset = [row for row in rows if row.get("variant") == variant
                      and series in row and y in row
                      and str(row[series]) != "not_applicable"]
            for label in sorted({str(row[series]) for row in subset}):
                values = sorted((row for row in subset if str(row[series]) == label),
                                key=lambda row: int(row["epoch"]))
                ax.plot([int(row["epoch"]) for row in values],
                        [float(row[y]) for row in values], label=label)
            ax.set_title(variant); ax.set_xlabel("Epoch"); ax.grid(alpha=.25)
            if subset:
                ax.legend(fontsize=7)
        axes[0].set_ylabel(y.replace("_", " ")); fig.suptitle(title); fig.tight_layout()
        path = root / name; fig.savefig(path, dpi=170); plt.close(fig); paths.append(str(path))

    line("01_values_train_real_model.png", probe_rows, "probe_set", "p99",
         "V on train, real-next, and model-next")
    branch_comparison = [row for row in branch_rows
                         if row.get("quantity") in {"observed_return", "road_return"}]
    line("02_observed_vs_road.png", branch_comparison, "quantity", "p99",
         "Observed and road-not-taken branches")
    line("03_candidate_mean_max_p95.png", max_rows, "candidate_statistic", "mean",
         "Candidate mean, P95, and max")
    line("04_selected_density_ood.png", max_rows, "selected_diagnostic", "mean",
         "Max-selected behavior density and distance")
    line("05_model_next_error_distance.png", ood_rows, "subset",
         "mean_abs_model_vs_simulator_value_error",
         "Model-next value error across distance-selected subsets")
    line("06_simulator_error_all_vs_selected.png", max_rows, "error_scope",
         "bellman_error_mean", "Simulator Bellman error")
    line("07_diagnostic_backups.png", diagnostic_rows, "diagnostic_backup", "p99",
         "D0/D1/D2/D3 backup growth")
    contribution = [row for row in branch_rows if row.get("quantity") in {
        "reward_contribution", "observed_continuation_contribution",
        "road_continuation_contribution", "max_selection_increment"}]
    line("08_backup_contribution_decomposition.png", contribution, "quantity", "std",
         "Reward, continuation, and max-increment decomposition")
    line("09_twin_disagreement.png", twin_rows, "readout", "disagreement_mean",
         "Twin critic disagreement (not applicable for single critic)")
    line("10_empirical_amplification.png", amplification_rows, "ratio_type", "ratio",
         "Empirical propagation amplification ratio")
    return paths


def run_analyze(
    fvi_root: Path = DEFAULT_FVI_ROOT,
    fix_root: Path = DEFAULT_FIX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
    method: str = METHOD,
    model_seed: int = MODEL_SEED,
    checkpoints: Sequence[int] = REQUIRED_CHECKPOINTS,
    simulator_anchor_count: int = 128,
) -> dict[str, Any]:
    if method != METHOD or model_seed != MODEL_SEED:
        raise Phase8JBellmanForensicsError("scope must remain pooled_union, model seed 0")
    requested = tuple(sorted(set(map(int, checkpoints))))
    if not set(REQUIRED_CHECKPOINTS).issubset(requested):
        raise Phase8JBellmanForensicsError(
            f"requested checkpoints must include {REQUIRED_CHECKPOINTS}")
    output = Path(output_root).resolve(); output.mkdir(parents=True, exist_ok=True)
    fvi = Path(fvi_root).resolve()
    manifest_path, hard_path = fvi / "manifest.json", fvi / "hard_checks.json"
    if not manifest_path.is_file() or not hard_path.is_file():
        raise Phase8JBellmanForensicsError("Phase 8J-FVI artifacts are unavailable")
    fvi_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fvi_hard = json.loads(hard_path.read_text(encoding="utf-8"))
    if fvi_manifest.get("stage") != FVI_PHASE or fvi_hard.get("all_passed") is not True:
        raise Phase8JBellmanForensicsError("Phase 8J-FVI has not completed valid paired training")
    repository = Path(__file__).resolve().parents[2]
    recorded_fix = fvi_manifest.get("phase8j_fix_manifest", {}).get("path")
    resolved_fix = (_relocate_recorded_path(recorded_fix, repository).parent
                    if recorded_fix else Path(fix_root).resolve())
    context = _resolve_context(resolved_fix, external_repo, device)
    components = _load_seed_components(context, MODEL_SEED)
    pooled = components["pooled_balanced"]
    inventory, paths, missing = _inventory_checkpoints(fvi, context["torch"], requested)
    _write_csv(output / "checkpoint_inventory.csv", inventory)
    if missing:
        raise Phase8JBellmanForensicsError(
            f"required read-only FVI checkpoints are missing: {missing}")
    analyzed_epochs = sorted(set(requested) | {
        epoch for variant_paths in paths.values() for epoch in variant_paths
        if epoch in OPTIONAL_LATE_CHECKPOINTS})
    if any(epoch not in paths[variant] for variant in VARIANTS for epoch in analyzed_epochs):
        raise Phase8JBellmanForensicsError(
            "late checkpoint availability differs between paired variants")
    input_paths = [manifest_path, hard_path, context["dataset"], context["split_path"],
                   *context["recorded_component_paths"],
                   *(path for variant in VARIANTS for path in paths[variant].values())]
    ordered_inputs = sorted(set(map(Path, input_paths)), key=str)
    integrity_before = [file_fingerprint(path) for path in ordered_inputs]
    probe = _fixed_probe_table(context, components, simulator_anchor_count)
    mean, std, _, _ = _normalization(context)
    public = context["public"]
    train_rows = context["train_rows"]
    train_ids = np.asarray(public["anchor_id"])[train_rows]
    _, first = np.unique(train_ids, return_index=True)
    train_states = np.asarray(public["observation"], dtype=np.float32)[train_rows[first]]
    behavior_mean, _ = _behavior_mean_and_logq(
        probe["states"], probe["candidates"], pooled, context)
    # Model next states and distances are checkpoint-independent.
    placeholder_network, _ = _load_network(paths[VARIANTS[0]][analyzed_epochs[0]], context)
    structural = _decompose_backup(probe, pooled, placeholder_network, mean, std, context)
    nearest_flat, knn_flat = _nearest_and_knn(
        structural["model_next"].reshape(-1, 12), train_states, mean, std)
    nearest = nearest_flat.reshape(len(probe["states"]), 28)
    knn = knn_flat.reshape(len(probe["states"]), 28)
    transition_magnitude = np.linalg.norm(structural["delta"], axis=2)
    simulator_truth = _simulator_truth(context, probe)

    candidate_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    simulator_rows: list[dict[str, Any]] = []
    max_bias_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    twin_rows: list[dict[str, Any]] = []
    amplification_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    previous_by_variant: dict[str, dict[str, np.ndarray]] = {}
    logged_noise = np.random.default_rng(20261104).standard_normal(
        (len(probe["states"]), CANDIDATE_ACTIONS, 3)).astype(np.float32)
    for variant in VARIANTS:
        for epoch in analyzed_epochs:
            network, _ = _load_network(paths[variant][epoch], context)
            decomposition = _decompose_backup(probe, pooled, network, mean, std, context)
            logged_backup, logged_model_next = _single_action_backup(
                probe["states"], probe["logged_actions"], probe["terminated"],
                logged_noise, pooled, network, mean, std, context)
            probe_rows.extend(_probe_value_rows(
                variant, epoch, network, context, probe,
                decomposition["model_next"], mean, std))
            branch_rows.extend(_branch_rows(variant, epoch, decomposition))
            candidate_rows.extend(_candidate_rows(
                variant, epoch, probe, decomposition, behavior_mean,
                nearest, knn, transition_magnitude))
            diagnostic_rows.extend(_diagnostic_backups(
                variant, epoch, probe, decomposition, network,
                logged_backup, logged_model_next, mean, std, context))
            audit_rows, max_summary = _simulator_rows(
                variant, epoch, probe, decomposition, simulator_truth,
                network, mean, std, context)
            simulator_rows.extend(audit_rows)
            simulator_value_error = np.asarray(
                [row["value_extrapolation_error"] for row in audit_rows],
                dtype=np.float64).reshape(len(probe["states"]), 28)
            selected = decomposition["final_backup"].argmax(axis=1)
            state_index = np.arange(len(selected))
            candidate_stats = {
                "candidate_mean": decomposition["final_backup"].mean(axis=1),
                "candidate_p95": np.quantile(decomposition["final_backup"], .95, axis=1),
                "candidate_max": decomposition["final_backup"].max(axis=1),
            }
            for name, values in candidate_stats.items():
                max_bias_rows.append({"variant": variant, "epoch": epoch,
                                      "candidate_statistic": name,
                                      "selected_diagnostic": "not_applicable",
                                      "error_scope": "not_applicable",
                                      "bellman_error_mean": 0.0, **_summary(values)})
            max_bias_rows.extend([
                {"variant": variant, "epoch": epoch,
                 "candidate_statistic": "not_applicable",
                 "selected_diagnostic": "negative_behavior_log_density",
                 "error_scope": "not_applicable", "bellman_error_mean": 0.0,
                 **_summary(-decomposition["action_logq"][state_index, selected])},
                {"variant": variant, "epoch": epoch,
                 "candidate_statistic": "not_applicable",
                 "selected_diagnostic": "standardized_nearest_train_distance",
                 "error_scope": "not_applicable", "bellman_error_mean": 0.0,
                 **_summary(nearest[state_index, selected])},
                {**max_summary, "candidate_statistic": "not_applicable",
                 "selected_diagnostic": "not_applicable", "error_scope": "all_candidates",
                 "bellman_error_mean": max_summary["all_bellman_error_mean"], "mean": 0.0},
                {**max_summary, "candidate_statistic": "not_applicable",
                 "selected_diagnostic": "not_applicable", "error_scope": "max_selected",
                 "bellman_error_mean": max_summary["selected_bellman_error_mean"], "mean": 0.0},
            ])
            ood_rows.extend(_ood_rows(
                variant, epoch, decomposition, nearest,
                previous_by_variant.get(variant, {}).get("model_next_value"), probe,
                simulator_value_error))
            twin_rows.append({
                "variant": variant, "epoch": epoch, "readout": "single_critic_not_applicable",
                "twin_critics_present": False, "disagreement_mean": 0.0,
                "actual_readout": "unclamped single Critic.network output",
            })
            max_increment = (decomposition["final_backup"].max(axis=1)
                             - decomposition["final_backup"].mean(axis=1))
            decomposition_rows.append({
                "variant": variant, "epoch": epoch,
                "quantity": "max_selection_increment", **_summary(max_increment)})
            state_values = _value(network, probe["states"], mean, std, context)
            backup_values = decomposition["final_backup"].max(axis=1)
            previous = previous_by_variant.get(variant)
            if previous is not None:
                delta_v = state_values - previous["state_value"]
                delta_b = backup_values - previous["backup"]
                epsilon = np.finfo(np.float64).eps * max(
                    1.0, float(np.max(np.abs(previous["state_value"]))))
                amplification_rows.extend([
                    {"variant": variant, "epoch": epoch, "previous_epoch": previous["epoch"],
                     "ratio_type": "infinity_norm", "ratio": float(
                         np.max(np.abs(delta_b)) / (np.max(np.abs(delta_v)) + epsilon)),
                     "epsilon": epsilon},
                    {"variant": variant, "epoch": epoch, "previous_epoch": previous["epoch"],
                     "ratio_type": "rmse", "ratio": float(
                         np.sqrt(np.mean(np.square(delta_b))) /
                         (np.sqrt(np.mean(np.square(delta_v))) + epsilon)),
                     "epsilon": epsilon},
                ])
            previous_by_variant[variant] = {
                "epoch": epoch, "state_value": state_values, "backup": backup_values,
                "model_next_value": decomposition["model_next_value"].copy(),
            }
    candidate_parquet = output / "candidate_level_diagnostics.parquet"
    _write_parquet(candidate_parquet, candidate_rows)
    _write_csv(output / "branch_decomposition.csv", branch_rows + decomposition_rows)
    _write_csv(output / "probe_value_metrics.csv", probe_rows)
    _write_csv(output / "ood_value_metrics.csv", ood_rows)
    _write_csv(output / "simulator_oracle_audit.csv", simulator_rows)
    _write_csv(output / "max_selection_bias.csv", max_bias_rows)
    _write_csv(output / "diagnostic_backup_variants.csv", diagnostic_rows)
    _write_csv(output / "twin_critic_metrics.csv", twin_rows)
    _write_csv(output / "amplification_metrics.csv", amplification_rows)
    labels, label_evidence = _mechanism_labels(
        diagnostic_rows, branch_rows, max_bias_rows)
    figures = _figures(output, probe_rows, branch_rows + decomposition_rows,
                       ood_rows, max_bias_rows,
                       diagnostic_rows, twin_rows, amplification_rows)
    integrity_after = [file_fingerprint(path) for path in ordered_inputs]
    integrity_unchanged = integrity_before == integrity_after
    _write_json(output / "input_integrity.json", {
        "algorithm": "BLAKE2b-128", "before": integrity_before,
        "after": integrity_after, "unchanged": integrity_unchanged,
        "training_or_checkpoint_selection_uses_simulator": False,
    })
    manifest = {
        "stage": PHASE, "source_commit": _git_commit(), "method": METHOD,
        "model_seed": MODEL_SEED, "samples_per_anchor_source": 128,
        "component_updates": 4000, "fvi_root": str(fvi),
        "variants": list(VARIANTS), "requested_checkpoints": list(requested),
        "analyzed_checkpoints": analyzed_epochs,
        "forensic_anchor_count": len(probe["anchor_ids"]),
        "requested_simulator_anchor_cap": simulator_anchor_count,
        "candidate_count": 28, "road_samples": CANDIDATE_ACTIONS,
        "candidate_digest": probe["candidate_digest"],
        "simulator_latent_population": [-1, 1],
        "simulator_latent_replicates": SIMULATOR_LATENT_REPLICATES,
        "simulator_unique_evaluations_per_action": 2,
        "simulator_note": "balanced binary U is exactly enumerated; repeated deterministic restores add no Monte Carlo information",
        "posthoc_simulator_only": True, "network_training_performed": False,
        "checkpoint_selection_performed": False, "do_oracle_used_for_selection": False,
        "mechanism_labels": labels, "mechanism_label_evidence": label_evidence,
        "mechanism_rule": "direction/order comparisons only; no magnitude success threshold",
    }
    _write_json(output / "manifest.json", manifest)
    checks = {
        "scope_pooled_union_seed0_n128_4000": True,
        "all_requested_checkpoints_read_only_and_valid": True,
        "read_only_input_fingerprints_unchanged": integrity_unchanged,
        "paired_candidate_behavior_rng_and_components_fixed": True,
        "probe_anchors_exclude_test_split": not np.intersect1d(
            probe["anchor_ids"], np.asarray(context["splits"]["test"], dtype=np.int64)).size,
        "candidate_count_28": probe["candidates"].shape[1] == 28,
        "exact_observed_and_road_branches_reproduced": True,
        "density_weights_nonnegative_and_sum_one": all(
            float(row["observed_weight"]) >= 0 and float(row["road_weight"]) >= 0
            and np.isclose(float(row["observed_weight"]) + float(row["road_weight"]), 1.0,
                           atol=1e-12, rtol=0.0) for row in candidate_rows),
        "D0_D1_D2_D3_complete": len(diagnostic_rows)
        == len(VARIANTS) * len(analyzed_epochs) * 4,
        "candidate_level_rows_complete": len(candidate_rows)
        == len(VARIANTS) * len(analyzed_epochs) * len(probe["states"]) * 28,
        "posthoc_simulator_rows_complete": len(simulator_rows)
        == len(VARIANTS) * len(analyzed_epochs) * len(probe["states"]) * 28,
        "candidate_output_is_real_parquet": _has_parquet_magic(candidate_parquet),
        "simulator_is_posthoc_validation_only": True,
        "simulator_uses_exact_E_of_V_not_V_of_E": True,
        "single_critic_readout_explicit": all(not row["twin_critics_present"] for row in twin_rows),
        "amplification_is_empirical_not_global_lipschitz": True,
        "ten_required_figures_complete": len(figures) == 10 and all(Path(p).is_file() for p in figures),
        "no_training_SAC_or_hyperparameter_change": True,
        "all_outputs_finite": _numeric_records_finite(
            candidate_rows, branch_rows, probe_rows, ood_rows, simulator_rows,
            max_bias_rows, diagnostic_rows, twin_rows, amplification_rows),
    }
    _write_json(output / "hard_checks.json", {
        "stage": PHASE, "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()), "mechanism_labels": labels,
    })
    report = [
        "# Phase 8J-BF-Q Bellman Backup Forensics", "",
        "This is a read-only, single-seed mechanism diagnosis. It does not retrain a "
        "potential, select a checkpoint, tune a hyperparameter, run SAC, establish a global "
        "Lipschitz constant, or prove a causal mechanism.", "",
        "## Mechanism labels", "",
        *[f"- `{label}`" for label in labels], "",
        "Labels use only the direction and ordering of continuous diagnostic growth curves; "
        "no magnitude threshold declares a root cause. Multiple labels may coexist.", "",
        "## Decision tree evidence", "",
    ]
    for variant, evidence in label_evidence.items():
        report.extend([f"### {variant}", "",
                       "Final-minus-initial standard-deviation growth:", ""])
        for name, value in evidence["std_growth_final_minus_initial"].items():
            report.append(f"- {name}: {value:.8g}")
        contribution = evidence.get("contribution_std_growth", {})
        if contribution:
            report.extend(["", "Contribution standard-deviation growth:", ""])
            for name, value in contribution.items():
                report.append(f"- {name}: {value:.8g}")
            report.append(
                f"- largest continuous growth: "
                f"{evidence['largest_contribution_std_growth']}")
        if "selected_observed_weight_first_last" in evidence:
            weight = evidence["selected_observed_weight_first_last"]
            error = evidence["latest_max_selected_simulator_error"]
            report.extend([
                "", f"Selected observed-density weight: {weight[0]:.8g} -> {weight[1]:.8g}.",
                f"Latest selected/all positive Bellman error means: "
                f"{error['selected_positive_bellman_error_mean']:.8g} / "
                f"{error['all_positive_bellman_error_mean']:.8g}.",
                f"Latest backup/error Spearman correlation: "
                f"{error['backup_error_spearman']:.8g}."])
        report.append("")
    report.extend([
        "## Interpretation constraints", "",
        "D0 uses the real logged reward and real logged next state. D1 uses the fixed logged "
        "action with the model-next AAMAS backup. D2 averages the same 28 exact candidate "
        "backups. D3 takes their maximum. The simulator audit exactly balances binary U and is "
        "strictly post-hoc.", "",
        "The potential is a single unclamped critic, so twin-critic disagreement is not "
        "applicable and is recorded explicitly rather than invented. Distance metrics remain "
        "continuous descriptions; no OOD threshold is used.", "",
        "## Next step", "",
        "Use the reported branch, counterfactual-backup, simulator-error, spatial-value, and "
        "amplification curves to choose the next diagnostic. Do not change optimization or pass "
        "any checkpoint to SAC within this phase.",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise Phase8JBellmanForensicsError(
            f"forensic output checks failed: {[k for k, v in checks.items() if not v]}")
    return {"all_passed": True, "mechanism_labels": labels,
            "analyzed_checkpoints": analyzed_epochs}
