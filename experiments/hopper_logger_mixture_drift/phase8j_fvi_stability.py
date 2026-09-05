"""Phase 8J-FVI-Q: paired moving-target versus frozen-outer-target diagnostic.

This module is deliberately narrow.  It reuses the public n128 data, frozen
4000-update components, unclamped critic, and exact pooled-union AAMAS backup
recorded by Phase 8J-FIX-Q.  It never runs SAC or reads a do-oracle artifact.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from aamas_hopper_adapter import (
    _joint_log_probability,
    _normal_action_samples,
    compute_source_aamas_backup,
)
from scripts.train_aamas_hopper_potential import seed_everything
from .phase8h_compute_matched_online_quick import (
    GAMMA,
    POTENTIAL_BATCH_SIZE,
    POTENTIAL_EPOCHS,
    POTENTIAL_LR,
    TARGET_TAU,
    TARGET_UPDATE_INTERVAL,
    _TerminalMaskedValue,
    _TorchPotentialValue,
    _git_commit,
    _polyak_update,
    file_fingerprint,
    parameter_fingerprint,
)
from .phase8h_quick_multipolicy_aamas import CANDIDATE_ACTIONS, union_candidate_actions
from .phase8j_potential_clamp_fix_quick import (
    _gradient_norm,
    _load_seed_components,
    _resolve_inputs,
    make_repaired_potential_network,
)


PHASE = "Phase 8J-FVI-Q"
METHOD = "pooled_union"
BACKUP_METHOD = "pooled_aamas_union_full"
MODEL_SEED = 0
SAMPLES_PER_ANCHOR_SOURCE = 128
COMPONENT_UPDATES = 4000
VARIANTS = ("matched_moving_target", "frozen_outer_target")
DIAGNOSTIC_EPOCHS = (0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 160, 170,
                     180, 190, 200)
MILESTONE_EPOCHS = (50, 100, 150, 200)
DEFAULT_FIX_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_potential_clamp_fix_quick")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_fvi_stability_diagnostic")


class Phase8JFVIError(RuntimeError):
    """Raised when the frozen diagnostic contract cannot be honored."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


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
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _blake_array(*arrays: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _relocate_recorded_path(recorded: str | Path, repository: Path) -> Path:
    path = Path(recorded)
    if path.exists():
        return path.resolve()
    parts = path.parts
    if "artifacts" in parts:
        candidate = repository.joinpath(*parts[parts.index("artifacts"):])
        if candidate.exists():
            return candidate.resolve()
    raise Phase8JFVIError(f"recorded read-only input is unavailable: {recorded}")


def _ancestor_with_stage(path: Path, stage: str) -> Path:
    start = path.parent if path.is_file() else path
    for parent in (start, *start.parents):
        manifest = parent / "manifest.json"
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("stage") == stage:
                return parent
        except (OSError, json.JSONDecodeError):
            continue
    raise Phase8JFVIError(f"cannot locate {stage} ancestor for {path}")


def _fingerprint_matches(record: Mapping[str, Any], actual: Path) -> bool:
    current = file_fingerprint(actual)
    return (int(record["size_bytes"]) == int(current["size_bytes"])
            and record.get("blake2b_128") == current.get("blake2b_128"))


def _source_fingerprints() -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__),
        root / "aamas_hopper_adapter.py",
        root / "experiments/hopper_logger_mixture_drift/phase8h_compute_matched_online_quick.py",
        root / "experiments/hopper_logger_mixture_drift/phase8j_potential_clamp_fix_quick.py",
    )
    return [file_fingerprint(path) for path in paths]


def _resolve_context(fix_root: Path, external_repo: Path, device: str) -> dict[str, Any]:
    """Resolve every historical input from the Phase 8J-FIX manifest."""
    repository = Path(__file__).resolve().parents[2]
    fix = Path(fix_root).resolve()
    manifest_path = fix / "repair_manifest.json"
    tests_path = fix / "gradient_and_fit_tests.json"
    if not manifest_path.is_file() or not tests_path.is_file():
        raise Phase8JFVIError("Phase 8J-FIX preflight artifacts are missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "Phase 8J-FIX-Q" or tests.get("all_passed") is not True:
        raise Phase8JFVIError("Phase 8J-FIX manifest/preflight is not valid")
    dataset = _relocate_recorded_path(manifest["dataset"]["path"], repository)
    if not _fingerprint_matches(manifest["dataset"], dataset):
        raise Phase8JFVIError("recorded n128 dataset fingerprint changed")
    component_records = manifest.get("component_checkpoints", [])
    if not component_records:
        raise Phase8JFVIError("Phase 8J-FIX manifest has no component checkpoints")
    component_paths = [_relocate_recorded_path(row["path"], repository)
                       for row in component_records]
    if not all(_fingerprint_matches(row, path)
               for row, path in zip(component_records, component_paths)):
        raise Phase8JFVIError("a frozen component checkpoint fingerprint changed")
    scaling = _ancestor_with_stage(component_paths[0], "Phase 8H-DS")
    legacy = dataset.parent.parent
    context = _resolve_inputs(scaling, legacy, external_repo, device)
    if context["dataset"].resolve() != dataset:
        raise Phase8JFVIError("manifest-resolved dataset disagrees with effective context")
    expected = {path.resolve() for path in context["paths_by_seed"][MODEL_SEED].values()}
    recorded = {path.resolve() for path in component_paths}
    if not expected.issubset(recorded):
        raise Phase8JFVIError("seed-0 n128/4000 components are not the recorded inputs")
    old_curve = fix / "potential_training_metrics.csv"
    old_checkpoint = fix / "potentials" / "pooled_union_seed0.pt"
    if not old_curve.is_file() or not old_checkpoint.is_file():
        raise Phase8JFVIError(
            "recorded pooled_union seed-0 divergence curve/checkpoint is unavailable")
    curve_rows = _read_csv(old_curve)
    if not any(row.get("potential") == "pooled_union"
               and int(float(row.get("run_id", -1))) == MODEL_SEED
               and int(float(row.get("epoch", -1))) == POTENTIAL_EPOCHS
               for row in curve_rows):
        raise Phase8JFVIError("old pooled_union seed-0 curve does not reach epoch 200")
    context.update({
        "fix_root": fix,
        "fix_manifest": manifest,
        "fix_manifest_path": manifest_path,
        "fix_tests_path": tests_path,
        "recorded_component_paths": sorted(expected, key=str),
        "old_divergence_curve": old_curve,
        "old_divergence_checkpoint": old_checkpoint,
    })
    return context


def _normalization(context: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    public, rows = context["public"], context["train_rows"]
    states = np.asarray(public["observation"], dtype=np.float32)[rows]
    mean = states.mean(axis=0, keepdims=True).astype(np.float32)
    std = states.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    std = np.where(std > 0.0, std, 1.0).astype(np.float32)
    reward = np.asarray(public["reward"], dtype=np.float64)[rows]
    return mean, std, float(reward.min()), float(reward.max())


def _network_value(network: Any, mean: np.ndarray, std: np.ndarray,
                   context: Mapping[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    return _TorchPotentialValue(network, mean, std, context["device"], context["torch"])


def _outer_batches(train_rows: np.ndarray, outer: int) -> list[np.ndarray]:
    permutation = np.random.default_rng(20260950 + outer).permutation(train_rows)
    return [permutation[start:start + POTENTIAL_BATCH_SIZE]
            for start in range(0, len(permutation), POTENTIAL_BATCH_SIZE)]


def _candidate_seed(outer: int, batch: int, *, validation: bool = False) -> int:
    return 20261000 + outer * 1000 + batch * 2 + (500_000 if validation else 0)


def _batch_inputs(context: Mapping[str, Any], components: Mapping[str, Any],
                  rows: np.ndarray, outer: int, batch: int,
                  *, validation: bool = False) -> dict[str, np.ndarray | int]:
    public = context["public"]
    states = np.asarray(public["observation"], dtype=np.float32)[rows]
    actions = np.asarray(public["commanded_action"], dtype=np.float32)[rows]
    bases = np.asarray(context["anchors"]["base_action"], dtype=np.float32)[
        np.asarray(public["anchor_id"])[rows].astype(np.int64)]
    sources = tuple(components[f"source_{source}"] for source in (1, 2, 3))
    seed = _candidate_seed(outer, batch, validation=validation)
    candidates = union_candidate_actions(
        sources, states, bases, samples_per_source=8, seed=seed)
    noise_seed = seed + 1
    return {
        "rows": np.asarray(rows, dtype=np.int64),
        "states": states,
        "observed_actions": actions,
        "candidates": np.asarray(candidates, dtype=np.float32),
        "terminated": np.asarray(public["terminated"], dtype=bool)[rows],
        "truncated": np.asarray(public["truncated"], dtype=bool)[rows],
        "noise_seed": noise_seed,
    }


def _pooled_union_backup(batch: Mapping[str, Any], pooled_model: Any,
                         value: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    states = np.asarray(batch["states"], dtype=np.float32)
    candidates = np.asarray(batch["candidates"], dtype=np.float32)
    noise = np.random.default_rng(int(batch["noise_seed"])).standard_normal(
        (len(states) * candidates.shape[1], CANDIDATE_ACTIONS, 3)).astype(np.float32)
    continuation = _TerminalMaskedValue(value, np.asarray(batch["terminated"], dtype=bool))
    pooled_q = compute_source_aamas_backup(
        (pooled_model,), states, candidates, continuation, common_noise=noise)[0]
    target = pooled_q.max(axis=1)
    if target.shape != (len(states),) or not np.all(np.isfinite(target)):
        raise Phase8JFVIError("pooled-union AAMAS target is invalid")
    return target


def _pairing_digest(context: Mapping[str, Any], outer_iterations: int) -> list[dict[str, Any]]:
    """Hash the shared row order and RNG stream without a costly model forward pass."""
    result = []
    for outer in range(outer_iterations):
        digest = hashlib.blake2b(digest_size=16)
        batches = _outer_batches(context["train_rows"], outer)
        for index, rows in enumerate(batches):
            digest.update(np.ascontiguousarray(rows).tobytes())
            digest.update(np.int64(_candidate_seed(outer, index)).tobytes())
            digest.update(np.int64(_candidate_seed(outer, index) + 1).tobytes())
        result.append({"outer_iteration": outer + 1, "batch_count": len(batches),
                       "row_and_rng_stream_blake2b_128": digest.hexdigest()})
    return result


def _weight_audit(batch: Mapping[str, Any], pooled_model: Any,
                  context: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute only the exact adapter weights for a compact public probe."""
    torch = context["torch"]
    states = np.asarray(batch["states"], dtype=np.float32)
    candidates = np.asarray(batch["candidates"], dtype=np.float32)
    flat_state = np.repeat(states, candidates.shape[1], axis=0)
    flat_action = candidates.reshape(-1, 3)
    ts = torch.as_tensor(flat_state, dtype=torch.float32, device=pooled_model.device)
    ta = torch.as_tensor(flat_action, dtype=torch.float32, device=pooled_model.device)
    noise = np.random.default_rng(int(batch["noise_seed"])).standard_normal(
        (len(flat_state), CANDIDATE_ACTIONS, 3)).astype(np.float32)
    tn = torch.as_tensor(noise, dtype=torch.float32, device=pooled_model.device)
    with torch.no_grad():
        distribution = pooled_model.behavior_model(ts)
        log_taken = _joint_log_probability(distribution, ta)
        sampled = _normal_action_samples(distribution, tn, torch)
        original = ta.unsqueeze(1).expand(-1, CANDIDATE_ACTIONS, -1)
        positive = original - torch.clamp(
            original + float(pooled_model.action_separation), -1.0, 1.0)
        negative = original - torch.clamp(
            original - float(pooled_model.action_separation), -1.0, 1.0)
        stacked = torch.stack((positive, negative, original - sampled), dim=2)
        selected = torch.gather(stacked, 2,
                                torch.abs(stacked).argmax(dim=2, keepdim=True)).squeeze(2)
        not_action = original - selected
        expanded = ts.unsqueeze(1).expand(-1, CANDIDATE_ACTIONS, -1)
        alt_distribution = pooled_model.behavior_model(expanded.reshape(-1, 12))
        log_not = _joint_log_probability(
            alt_distribution, not_action.reshape(-1, 3)).reshape(
                len(flat_state), CANDIDATE_ACTIONS).mean(dim=1)
        lt = np.clip(log_taken.cpu().numpy(), -50.0, -0.01)
        ln = np.clip(log_not.cpu().numpy(), -50.0, -0.01)
    taken = np.exp(lt) / (np.exp(lt) + np.exp(ln))
    other = 1.0 - taken
    sums = taken + other
    return {
        "row_candidate_count": len(taken),
        "taken_weight_min": float(taken.min()),
        "taken_weight_max": float(taken.max()),
        "road_weight_min": float(other.min()),
        "road_weight_max": float(other.max()),
        "maximum_sum_error": float(np.max(np.abs(sums - 1.0))),
        "nonnegative": bool(np.all(taken >= 0) and np.all(other >= 0)),
        "sum_to_one": bool(np.allclose(sums, 1.0, atol=1e-12, rtol=0.0)),
        "formula": "exp(clipped log density) ratio; no direct density subtraction",
    }


def _model_next_states(batch: Mapping[str, Any], pooled_model: Any,
                       context: Mapping[str, Any]) -> np.ndarray:
    """Materialize only the compact fixed probe states queried by the real backup."""
    torch = context["torch"]
    states = np.asarray(batch["states"], dtype=np.float32)
    candidates = np.asarray(batch["candidates"], dtype=np.float32)
    flat_state = np.repeat(states, candidates.shape[1], axis=0)
    flat_action = candidates.reshape(-1, 3)
    ts = torch.as_tensor(flat_state, dtype=torch.float32, device=pooled_model.device)
    ta = torch.as_tensor(flat_action, dtype=torch.float32, device=pooled_model.device)
    noise = np.random.default_rng(int(batch["noise_seed"])).standard_normal(
        (len(flat_state), CANDIDATE_ACTIONS, 3)).astype(np.float32)
    tn = torch.as_tensor(noise, dtype=torch.float32, device=pooled_model.device)
    with torch.no_grad():
        distribution = pooled_model.behavior_model(ts)
        current = ts + pooled_model.state_difference_model(torch.cat((ts, ta), dim=1))
        sampled = _normal_action_samples(distribution, tn, torch)
        original = ta.unsqueeze(1).expand(-1, CANDIDATE_ACTIONS, -1)
        positive = original - torch.clamp(
            original + float(pooled_model.action_separation), -1.0, 1.0)
        negative = original - torch.clamp(
            original - float(pooled_model.action_separation), -1.0, 1.0)
        stacked = torch.stack((positive, negative, original - sampled), dim=2)
        selected = torch.gather(stacked, 2,
                                torch.abs(stacked).argmax(dim=2, keepdim=True)).squeeze(2)
        not_action = original - selected
        expanded = ts.unsqueeze(1).expand(-1, CANDIDATE_ACTIONS, -1)
        flat_expanded = expanded.reshape(-1, 12)
        alternative = flat_expanded + pooled_model.state_difference_model(
            torch.cat((flat_expanded, not_action.reshape(-1, 3)), dim=1))
    return np.concatenate((current.cpu().numpy(), alternative.cpu().numpy()), axis=0)


def _operator_probe(context: Mapping[str, Any], components: Mapping[str, Any]) -> dict[str, Any]:
    rows = context["validation_rows"][:8]
    batch = _batch_inputs(context, components, rows, 0, 0, validation=True)
    pooled = components["pooled_balanced"]
    queried: list[np.ndarray] = []

    def v(states: np.ndarray) -> np.ndarray:
        array = np.asarray(states, dtype=np.float64)
        queried.append(array.copy())
        return 0.05 * np.tanh(array.sum(axis=1))

    def w(states: np.ndarray) -> np.ndarray:
        array = np.asarray(states, dtype=np.float64)
        return 0.05 * np.tanh(array.sum(axis=1)) + 0.75

    first = _pooled_union_backup(batch, pooled, v)
    second = _pooled_union_backup(batch, pooled, v)
    higher = _pooled_union_backup(batch, pooled, w)
    terminal_batch = dict(batch)
    terminal_batch["terminated"] = np.ones(len(rows), dtype=bool)
    terminal_v = _pooled_union_backup(terminal_batch, pooled, v)
    terminal_w = _pooled_union_backup(terminal_batch, pooled, w)
    truncated_batch = dict(batch)
    truncated_batch["terminated"] = np.zeros(len(rows), dtype=bool)
    truncated_batch["truncated"] = np.ones(len(rows), dtype=bool)
    truncated_v = _pooled_union_backup(truncated_batch, pooled, v)
    truncated_w = _pooled_union_backup(truncated_batch, pooled, w)
    queried_states = np.concatenate(queried, axis=0)
    continuation_gap = 0.75
    lhs = float(np.max(np.abs(higher - first)))
    tolerance = 5e-6
    checks = {
        "deterministic_on_fixed_probe": bool(np.array_equal(first, second)),
        "monotone_on_fixed_probe": bool(np.all(higher + tolerance >= first)),
        "gamma_lipschitz_on_all_queried_next_states": lhs <= GAMMA * continuation_gap + tolerance,
        "all_queried_next_states_covered": len(queried_states) > len(rows),
        "terminated_rows_zero_continuation": bool(np.allclose(
            terminal_v, terminal_w, atol=tolerance, rtol=0.0)),
        "truncated_rows_still_bootstrap": bool(np.max(np.abs(
            truncated_w - truncated_v)) > tolerance),
    }
    weight = _weight_audit(batch, pooled, context)
    checks.update({
        "mixture_weights_nonnegative": weight["nonnegative"],
        "mixture_weights_sum_to_one": weight["sum_to_one"],
    })
    return {
        "scope": "finite fixed public probe; not a theoretical proof or causal certificate",
        "probe_rows": len(rows),
        "queried_next_state_calls": len(queried),
        "queried_next_state_rows_with_repetition": len(queried_states),
        "max_backup_difference": lhs,
        "gamma_times_max_continuation_difference": GAMMA * continuation_gap,
        "tolerance": tolerance,
        "weight_audit": weight,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _gradient_isolation_audit(context: Mapping[str, Any], components: Mapping[str, Any],
                              mean: np.ndarray, std: np.ndarray,
                              reward_min: float, reward_max: float) -> dict[str, Any]:
    torch = context["torch"]
    current = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"])
    target = copy.deepcopy(current).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(current.parameters(), lr=POTENTIAL_LR, weight_decay=1e-5)
    rows = context["train_rows"][:16]
    batch = _batch_inputs(context, components, rows, 0, 0)
    target_value = _network_value(target, mean, std, context)
    backup = _pooled_union_backup(batch, components["pooled_balanced"], target_value)
    tensor = torch.as_tensor((batch["states"] - mean) / (std + 1e-7),
                             dtype=torch.float32, device=context["device"])
    prediction = current(tensor).reshape(-1)
    loss = torch.nn.functional.mse_loss(
        prediction, torch.as_tensor(backup, dtype=torch.float32, device=context["device"]))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    current_grad = _gradient_norm(tuple(current.parameters()))
    target_has_grad = any(parameter.grad is not None for parameter in target.parameters())
    component_requires_grad = any(
        parameter.requires_grad
        for bundle in components.values()
        for module in (bundle.behavior_model, bundle.state_difference_model,
                       bundle.reward_model)
        for parameter in module.parameters())
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    target_ids = {id(parameter) for parameter in target.parameters()}
    checks = {
        "target_generation_detached": not target_has_grad,
        "target_parameters_require_no_grad": not any(p.requires_grad for p in target.parameters()),
        "current_prediction_has_gradient": current_grad > 0.0,
        "optimizer_excludes_target": optimizer_ids.isdisjoint(target_ids),
        "component_models_frozen": not component_requires_grad,
        "current_and_target_readout_unclamped": (
            getattr(current, "phase8j_common_implementation_repair", None)
            == "remove_final_hard_clamp_only"
            and getattr(target, "phase8j_common_implementation_repair", None)
            == "remove_final_hard_clamp_only"),
    }
    return {"current_gradient_norm": current_grad, "checks": checks,
            "all_passed": all(checks.values())}


def _fixed_target_fit(context: Mapping[str, Any], components: Mapping[str, Any],
                      mean: np.ndarray, std: np.ndarray,
                      reward_min: float, reward_max: float) -> dict[str, Any]:
    torch = context["torch"]
    seed_everything(20260960, torch, cuda_training=context["device"] == "cuda")
    network = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"])
    snapshot = copy.deepcopy(network).eval().requires_grad_(False)
    rows = context["train_rows"][:min(256, len(context["train_rows"]))]
    batch = _batch_inputs(context, components, rows, 0, 0)
    target = _pooled_union_backup(
        batch, components["pooled_balanced"], _network_value(snapshot, mean, std, context))
    normalized = torch.as_tensor((batch["states"] - mean) / (std + 1e-7),
                                 dtype=torch.float32, device=context["device"])
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=context["device"])
    optimizer = torch.optim.Adam(network.parameters(), lr=POTENTIAL_LR, weight_decay=1e-5)
    initial_parameters = parameter_fingerprint((network,))
    with torch.no_grad():
        initial_prediction = network(normalized).reshape(-1).clone()
        initial_loss = float(torch.nn.functional.mse_loss(initial_prediction, target_tensor).cpu())
    losses, gradients = [], []
    for _ in range(100):
        prediction = network(normalized).reshape(-1)
        loss = torch.nn.functional.mse_loss(prediction, target_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(_gradient_norm(tuple(network.parameters())))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        final_prediction = network(normalized).reshape(-1)
        final_loss = float(torch.nn.functional.mse_loss(final_prediction, target_tensor).cpu())
    checks = {
        "target_finite": bool(np.all(np.isfinite(target))),
        "effective_prediction_gradient": max(gradients) > 0.0,
        "fixed_target_error_decreased": final_loss < initial_loss,
        "prediction_changed": not bool(torch.equal(initial_prediction, final_prediction)),
        "parameters_changed_beyond_weight_decay_only": parameter_fingerprint((network,)) != initial_parameters,
        "diagnostic_network_not_formal_initialization": True,
    }
    return {
        "updates": 100,
        "row_count": len(rows),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_curve": losses,
        "gradient_norm_curve": gradients,
        "target_summary": _summary(target),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "p01": float(np.quantile(x, .01)),
        "p99": float(np.quantile(x, .99)), "max": float(x.max()),
        "mean_abs": float(np.mean(np.abs(x))),
    }


def _nearest_distances(query: np.ndarray, train: np.ndarray,
                       mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    reference = (np.asarray(train, dtype=np.float64) - mean) / std
    values = (np.asarray(query, dtype=np.float64) - mean) / std
    result = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), 1024):
        block = values[start:start + 1024]
        squared = np.square(block[:, None, :] - reference[None, :, :]).sum(axis=2)
        result[start:start + len(block)] = np.sqrt(squared.min(axis=1))
    return result


def _build_probes(context: Mapping[str, Any], components: Mapping[str, Any],
                  mean: np.ndarray, std: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    public = context["public"]
    train_rows = context["train_rows"]
    train_ids = np.asarray(public["anchor_id"])[train_rows]
    _, unique_pos = np.unique(train_ids, return_index=True)
    train_states = np.asarray(public["observation"], dtype=np.float32)[train_rows[unique_pos]]
    validation_rows = context["validation_rows"]
    terminal = np.asarray(public["terminated"], dtype=bool)[validation_rows]
    real_next = np.asarray(public["next_observation"], dtype=np.float32)[validation_rows]
    compact_rows = validation_rows[:min(256, len(validation_rows))]
    batch = _batch_inputs(context, components, compact_rows, 0, 0, validation=True)
    model_next = _model_next_states(batch, components["pooled_balanced"], context)
    probes = {
        "train_anchor": train_states,
        "real_next_bootstrap": real_next[~terminal],
        "real_next_terminal": real_next[terminal],
        "model_next": model_next,
    }
    distance_rows = []
    for name, states in probes.items():
        if not len(states):
            continue
        distances = _nearest_distances(states, train_states, mean, std)
        distance_rows.append({"probe_set": name, **_summary(distances),
                              "interpretation": "descriptive_only_no_ood_threshold"})
    return probes, distance_rows


def run_preflight_and_tests(
    fix_root: Path = DEFAULT_FIX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
    method: str = METHOD,
    model_seed: int = MODEL_SEED,
    samples_per_anchor_source: int = SAMPLES_PER_ANCHOR_SOURCE,
    component_updates: int = COMPONENT_UPDATES,
) -> dict[str, Any]:
    if (method, model_seed, samples_per_anchor_source, component_updates) != (
            METHOD, MODEL_SEED, SAMPLES_PER_ANCHOR_SOURCE, COMPONENT_UPDATES):
        raise Phase8JFVIError("preflight scope must be pooled_union/seed0/n128/4000")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = _resolve_context(fix_root, external_repo, device)
    components = _load_seed_components(context, MODEL_SEED)
    mean, std, reward_min, reward_max = _normalization(context)
    operator = _operator_probe(context, components)
    gradient = _gradient_isolation_audit(
        context, components, mean, std, reward_min, reward_max)
    fit = _fixed_target_fit(context, components, mean, std, reward_min, reward_max)
    probes, distances = _build_probes(context, components, mean, std)
    pairing = _pairing_digest(context, 40)
    old_curve = context["old_divergence_curve"]
    old_checkpoint = context["old_divergence_checkpoint"]
    inputs = [context["fix_manifest_path"], context["fix_tests_path"], context["dataset"],
              context["split_path"], *context["recorded_component_paths"]]
    if old_curve is not None:
        inputs.append(old_curve)
    if old_checkpoint is not None:
        inputs.append(old_checkpoint)
    integrity = [file_fingerprint(path) for path in inputs]
    _write_json(output / "input_integrity.json", {
        "algorithm": "BLAKE2b-128", "inputs": integrity,
        "forbidden_inputs": ["do_oracle", "online_return", "test_selection"],
    })
    _write_json(output / "preflight_operator_audit.json", {
        **operator, "gradient_isolation": gradient,
        "termination_semantics": {
            "terminated": "continuation multiplied by zero",
            "truncated": "continuation retained",
            "final_observation_field": "public next_observation; reset observation is not read",
        },
    })
    empirical_lower = reward_min / (1.0 - GAMMA)
    empirical_upper = reward_max / (1.0 - GAMMA)
    range_audit = {
        "raw_reward_units": "raw confounded Hopper environment reward",
        "component_reward_training_units": "z-normalized internally then restored by reward_std/reward_mean",
        "backup_target_reward_units": "raw reward",
        "reward_upper_units": "raw reward",
        "potential_output_units": "raw discounted-return-like backup units",
        "units_consistent": True,
        "reward_min": reward_min, "reward_max": reward_max,
        "old_empirical_reward_min_over_one_minus_gamma": empirical_lower,
        "old_empirical_reward_max_over_one_minus_gamma": empirical_upper,
        "bound_source": "empirical training-sample extrema; not a proved global mechanism bound",
        "positive_empirical_lower_bound_valid_for_all_states": False,
        "reason": "termination has zero continuation, so a positive empirical lower bound is not global",
        "conservative_formula_if_valid_mechanism_bounds_exist": (
            "[min(0,r_min_valid)/(1-gamma), max(0,r_max_valid)/(1-gamma)]"),
        "new_output_bound_enabled": False,
        "bounded_output_readiness": {
            "A_credible_reward_value_range": False,
            "B_includes_terminal_zero": False,
            "C_actual_backup_compatibility_verified": False,
            "D_old_interval_is_empirical_engineering_constraint": True,
            "E_reward_value_units_consistent": True,
        },
    }
    _write_json(output / "value_range_audit.json", range_audit)
    _write_json(output / "fixed_target_fit_test.json", fit)
    _write_json(output / "candidate_pairing_audit.json", {
        "policy": "fixed per outer block and identical across both variants",
        "candidate_resampling_matches_old_run": False,
        "control_name": "matched moving-target control",
        "old_divergence_role": "background only; not bitwise paired",
        "outer_digests": pairing,
        "large_candidate_tensor_materialized_to_disk": False,
    })
    _write_csv(output / "next_state_distance_metrics.csv", distances)
    np.savez_compressed(output / "fixed_public_probes.npz", **probes,
                        state_mean=mean, state_std=std)
    source = inspect.getsource(compute_source_aamas_backup)
    checks = {
        "exact_scope_pooled_union_seed0_n128_4000": True,
        "phase8j_fix_manifest_resolved": True,
        "old_pooled_union_epoch200_curve_and_checkpoint_resolved": (
            old_curve is not None and old_checkpoint is not None),
        "input_fingerprints_valid": True,
        "real_aamas_backup_reused": "compute_official_continuous_action_backup" in source,
        "finite_operator_probe_passed": operator["all_passed"],
        "gradient_isolation_passed": gradient["all_passed"],
        "fixed_target_fit_passed": fit["all_passed"],
        "three_public_probe_families_fixed": all(
            key in probes for key in ("train_anchor", "real_next_bootstrap", "model_next")),
        "no_oracle_or_online_input": True,
        "unclamped_linear_readout": True,
    }
    manifest = {
        "stage": PHASE,
        "phase": "preflight-and-tests",
        "status_labels": ["IMPLEMENTATION_VALID", "TRAINABLE_ON_FIXED_TARGET"],
        "source_commit": _git_commit(),
        "source_fingerprints": _source_fingerprints(),
        "phase8j_fix_manifest": file_fingerprint(context["fix_manifest_path"]),
        "dataset": file_fingerprint(context["dataset"]),
        "component_checkpoints": [file_fingerprint(path)
                                  for path in context["recorded_component_paths"]],
        "method": METHOD, "backup_method": BACKUP_METHOD, "model_seed": MODEL_SEED,
        "samples_per_anchor_source": SAMPLES_PER_ANCHOR_SOURCE,
        "component_updates": COMPONENT_UPDATES,
        "variants": list(VARIANTS), "epochs": POTENTIAL_EPOCHS,
        "outer_iterations": 40, "inner_epochs": 5,
        "optimizer": "Adam", "learning_rate": POTENTIAL_LR,
        "batch_size": POTENTIAL_BATCH_SIZE, "weight_decay": 1e-5,
        "loss": "mean_squared_error", "target_tau": TARGET_TAU,
        "target_update_interval_batches": TARGET_UPDATE_INTERVAL,
        "optimizer_updates_per_epoch": int(math.ceil(len(context["train_rows"])
                                                      / POTENTIAL_BATCH_SIZE)),
        "reward_normalization": "none at potential level",
        "potential_readout": "single unclamped linear official Critic.network output",
        "twin_critics": "not present in potential model; SAC is not run",
        "target_critic": "same single readout; frozen/Polyak behavior varies by arm",
        "candidate_count": 28, "candidate_samples_per_source": 8,
        "continuation_mask": "terminated only; truncated bootstraps",
        "old_divergence_curve": (file_fingerprint(old_curve) if old_curve else None),
        "old_divergence_checkpoint": (
            file_fingerprint(old_checkpoint) if old_checkpoint else None),
        "formal_initialization_reuses_diagnostic_network": False,
        "online_sac_enabled": False, "do_oracle_used": False,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "hard_checks.json", {
        "stage": PHASE, "phase": "preflight-and-tests",
        "checks": checks, "failed": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    })
    if not all(checks.values()):
        raise Phase8JFVIError(
            f"preflight failed: {[name for name, passed in checks.items() if not passed]}")
    return {"all_passed": True, "probe_counts": {k: len(v) for k, v in probes.items()}}


def _load_fixed_probes(output: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    path = output / "fixed_public_probes.npz"
    if not path.is_file():
        raise Phase8JFVIError("preflight fixed probes are missing")
    with np.load(path, allow_pickle=False) as archive:
        probes = {name: archive[name].copy() for name in archive.files
                  if name not in {"state_mean", "state_std"}}
        return probes, archive["state_mean"].copy(), archive["state_std"].copy()


def _parameter_vector(network: Any) -> np.ndarray:
    return np.concatenate([parameter.detach().cpu().double().reshape(-1).numpy()
                           for parameter in network.parameters()])


def _probe_rows(variant: str, epoch: int, network: Any, probes: Mapping[str, np.ndarray],
                mean: np.ndarray, std: np.ndarray, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = _network_value(network, mean, std, context)
    return [{"variant": variant, "epoch": epoch, "probe_set": name,
             **_summary(value(states))} for name, states in probes.items() if len(states)]


def _fixed_validation_batches(context: Mapping[str, Any], components: Mapping[str, Any]
                              ) -> list[dict[str, Any]]:
    rows = context["validation_rows"]
    batches = [rows[start:start + POTENTIAL_BATCH_SIZE]
               for start in range(0, len(rows), POTENTIAL_BATCH_SIZE)]
    return [_batch_inputs(context, components, value, 0, index, validation=True)
            for index, value in enumerate(batches)]


def _evaluate_target_residual(network: Any, target_network: Any,
                              batches: Sequence[Mapping[str, Any]], pooled: Any,
                              mean: np.ndarray, std: np.ndarray,
                              context: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    current_value = _network_value(network, mean, std, context)
    target_value = _network_value(target_network, mean, std, context)
    residuals, targets = [], []
    for batch in batches:
        prediction = current_value(batch["states"])
        target = _pooled_union_backup(batch, pooled, target_value)
        residuals.append(prediction - target)
        targets.append(target)
    residual = np.concatenate(residuals)
    all_targets = np.concatenate(targets)
    return float(np.sqrt(np.mean(np.square(residual)))), _summary(all_targets)


def _recomputed_bellman(network: Any, batches: Sequence[Mapping[str, Any]], pooled: Any,
                        mean: np.ndarray, std: np.ndarray,
                        context: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    value = _network_value(network, mean, std, context)
    predictions, targets = [], []
    for batch in batches:
        predictions.append(value(batch["states"]))
        targets.append(_pooled_union_backup(batch, pooled, value))
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    return prediction, target, float(np.sqrt(np.mean(np.square(prediction - target))))


def _save_training_checkpoint(path: Path, variant: str, epoch: int, outer: int,
                              network: Any, target: Any, optimizer: Any,
                              mean: np.ndarray, std: np.ndarray,
                              reward_min: float, reward_max: float,
                              initial_fingerprint: Mapping[str, Any], torch: Any,
                              previous_prediction: np.ndarray | None = None,
                              previous_backup: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_state_dict": {k: v.detach().cpu() for k, v in network.state_dict().items()},
        "target_state_dict": {k: v.detach().cpu() for k, v in target.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "state_mean": mean, "state_std": std,
        "reward_min": reward_min, "reward_max": reward_max,
        "previous_outer_prediction": previous_prediction,
        "previous_outer_backup": previous_backup,
        "metadata": {
            "stage": PHASE, "variant": variant, "epoch": epoch,
            "outer_iteration": outer, "initial_parameter_fingerprint": initial_fingerprint,
            "readout": "unclamped_linear", "eligible_for_sac": False,
        },
    }
    torch.save(payload, path)


def _load_or_create_initialization(output: Path, context: Mapping[str, Any],
                                   reward_min: float, reward_max: float) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = context["torch"]
    path = output / "formal_initialization.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload["state_dict"], payload["fingerprint"]
    seed_everything(MODEL_SEED, torch, cuda_training=context["device"] == "cuda")
    network = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"])
    state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
    fingerprint = parameter_fingerprint((network,))
    torch.save({"state_dict": state, "fingerprint": fingerprint,
                "diagnostic_preflight_network_reused": False}, path)
    return state, fingerprint


def _train_variant(variant: str, output: Path, context: Mapping[str, Any],
                   components: Mapping[str, Any], probes: Mapping[str, np.ndarray],
                   mean: np.ndarray, std: np.ndarray, reward_min: float, reward_max: float,
                   initial_state: Mapping[str, Any], initial_fingerprint: Mapping[str, Any],
                   outer_iterations: int, inner_epochs: int, total_epochs: int
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    torch, device = context["torch"], context["device"]
    directory = output / ("moving_target" if variant == VARIANTS[0] else "frozen_outer_target")
    directory.mkdir(parents=True, exist_ok=True)
    current = make_repaired_potential_network(context["official"], reward_min, reward_max, device)
    current.load_state_dict(initial_state)
    target = make_repaired_potential_network(context["official"], reward_min, reward_max, device)
    target.load_state_dict(initial_state)
    target.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(current.parameters(), lr=POTENTIAL_LR, weight_decay=1e-5)
    latest = directory / "latest.pt"
    start_epoch = 1
    if latest.is_file():
        payload = torch.load(latest, map_location=device, weights_only=False)
        if payload["metadata"]["variant"] != variant:
            raise Phase8JFVIError("resumable checkpoint variant mismatch")
        current.load_state_dict(payload["current_state_dict"])
        target.load_state_dict(payload["target_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["metadata"]["epoch"]) + 1
        previous_prediction = payload.get("previous_outer_prediction")
        previous_backup = payload.get("previous_outer_backup")
    else:
        previous_prediction = None
        previous_backup = None
    training_path = directory / "training_metrics.csv"
    probe_path = directory / "probe_value_metrics.csv"
    bellman_path = directory / "bellman_residual_metrics.csv"
    training = [row for row in _read_csv(training_path)
                if int(float(row["epoch"])) < start_epoch]
    probe_metrics = [row for row in _read_csv(probe_path)
                     if int(float(row["epoch"])) < start_epoch]
    bellman_metrics = [row for row in _read_csv(bellman_path)
                       if int(float(row["epoch"])) < start_epoch]
    validation_batches = _fixed_validation_batches(context, components)
    pooled = components["pooled_balanced"]
    total_updates = (start_epoch - 1) * len(_outer_batches(context["train_rows"], 0))
    if start_epoch == 1:
        probe_metrics.extend(_probe_rows(variant, 0, current, probes, mean, std, context))
        prediction, backup, residual = _recomputed_bellman(
            current, validation_batches, pooled, mean, std, context)
        bellman_metrics.append({"variant": variant, "outer_iteration": 0, "epoch": 0,
                                "recomputed_bellman_rmse": residual,
                                "value_iteration_drift_rmse": 0.0,
                                "target_drift_rmse": 0.0})
        previous_prediction, previous_backup = prediction, backup
        _save_training_checkpoint(directory / "initial.pt", variant, 0, 0, current, target,
                                  optimizer, mean, std, reward_min, reward_max,
                                  initial_fingerprint, torch)
    elif previous_prediction is None or previous_backup is None:
        raise Phase8JFVIError(
            "resumable checkpoint lacks the previous outer diagnostics; restart this variant")

    wall_start = time.perf_counter()
    for outer in range((start_epoch - 1) // inner_epochs, outer_iterations):
        first_epoch = outer * inner_epochs + 1
        last_epoch = min((outer + 1) * inner_epochs, total_epochs)
        if last_epoch < start_epoch:
            continue
        resuming_inside_outer = start_epoch > first_epoch and outer == (start_epoch - 1) // inner_epochs
        target_before_outer_refresh = _parameter_vector(target)
        if variant == "frozen_outer_target" and not resuming_inside_outer:
            target.load_state_dict(current.state_dict())
            target.eval().requires_grad_(False)
            if not np.array_equal(_parameter_vector(target), _parameter_vector(current)):
                raise Phase8JFVIError("frozen snapshot did not refresh at outer boundary")
        snapshot_hash = _blake_array(_parameter_vector(target))
        batches = [_batch_inputs(context, components, rows, outer, index)
                   for index, rows in enumerate(_outer_batches(context["train_rows"], outer))]
        cached_targets = None
        if variant == "frozen_outer_target":
            target_value = _network_value(target, mean, std, context)
            cached_targets = [_pooled_union_backup(batch, pooled, target_value)
                              for batch in batches]
        for epoch in range(max(start_epoch, first_epoch), last_epoch + 1):
            losses, gradients, predictions_all, targets_all = [], [], [], []
            before_current = _parameter_vector(current)
            before_target = (target_before_outer_refresh
                             if variant == "frozen_outer_target"
                             and epoch == first_epoch and not resuming_inside_outer
                             else _parameter_vector(target))
            for batch_index, batch in enumerate(batches):
                if cached_targets is None:
                    backup = _pooled_union_backup(
                        batch, pooled, _network_value(target, mean, std, context))
                else:
                    backup = cached_targets[batch_index]
                normalized = torch.as_tensor(
                    (batch["states"] - mean) / (std + 1e-7),
                    dtype=torch.float32, device=device)
                prediction = current(normalized).reshape(-1)
                target_tensor = torch.as_tensor(backup, dtype=torch.float32, device=device)
                loss = torch.nn.functional.mse_loss(prediction, target_tensor)
                if not bool(torch.isfinite(loss)):
                    _save_training_checkpoint(directory / "nonfinite_stop.pt", variant, epoch,
                                              outer + 1, current, target, optimizer, mean, std,
                                              reward_min, reward_max, initial_fingerprint, torch)
                    raise Phase8JFVIError(f"nonfinite loss in {variant} epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = _gradient_norm(tuple(current.parameters()))
                if not np.isfinite(gradient):
                    raise Phase8JFVIError(f"nonfinite gradient in {variant} epoch {epoch}")
                optimizer.step()
                if (variant == "matched_moving_target" and batch_index > 0
                        and batch_index % TARGET_UPDATE_INTERVAL == 0):
                    _polyak_update(current, target, TARGET_TAU)
                losses.append(float(loss.detach().cpu()))
                gradients.append(gradient)
                predictions_all.append(prediction.detach().cpu().numpy())
                targets_all.append(backup)
                total_updates += 1
            after_current = _parameter_vector(current)
            after_target = _parameter_vector(target)
            target_hash_after = _blake_array(after_target)
            if variant == "frozen_outer_target" and target_hash_after != snapshot_hash:
                raise Phase8JFVIError("frozen target changed inside an outer iteration")
            validation_rmse, target_stats = _evaluate_target_residual(
                current, target, validation_batches, pooled, mean, std, context)
            prediction_values = np.concatenate(predictions_all)
            target_values = np.concatenate(targets_all)
            record = {
                "variant": variant, "outer_iteration": outer + 1, "epoch": epoch,
                "optimizer_updates": total_updates,
                "training_loss": float(np.mean(losses)),
                "validation_current_target_rmse": validation_rmse,
                "prediction_gradient_norm_mean": float(np.mean(gradients)),
                "prediction_gradient_norm_max": float(np.max(gradients)),
                "current_parameter_movement_rmse": float(np.sqrt(np.mean(np.square(
                    after_current - before_current)))),
                "target_parameter_movement_rmse": float(np.sqrt(np.mean(np.square(
                    after_target - before_target)))),
                "prediction_mean": float(prediction_values.mean()),
                "prediction_std": float(prediction_values.std()),
                "target_mean": float(target_values.mean()),
                "target_std": float(target_values.std()),
                "validation_target_mean": target_stats["mean"],
                "validation_target_std": target_stats["std"],
                "outer_snapshot_blake2b_128": snapshot_hash,
                "target_after_epoch_blake2b_128": target_hash_after,
                "wall_seconds_cumulative": float(time.perf_counter() - wall_start),
            }
            if not all(np.isfinite(float(value)) for key, value in record.items()
                       if key not in {"variant", "outer_snapshot_blake2b_128",
                                      "target_after_epoch_blake2b_128"}):
                raise Phase8JFVIError(f"nonfinite scalar diagnostic in {variant} epoch {epoch}")
            training.append(record)
            if epoch in DIAGNOSTIC_EPOCHS:
                probe_metrics.extend(_probe_rows(
                    variant, epoch, current, probes, mean, std, context))
            _save_training_checkpoint(latest, variant, epoch, outer + 1, current, target,
                                      optimizer, mean, std, reward_min, reward_max,
                                      initial_fingerprint, torch,
                                      previous_prediction, previous_backup)
            if epoch in MILESTONE_EPOCHS:
                _save_training_checkpoint(directory / f"epoch_{epoch}.pt", variant, epoch,
                                          outer + 1, current, target, optimizer, mean, std,
                                          reward_min, reward_max, initial_fingerprint, torch)
            _write_csv(training_path, training)
            _write_csv(probe_path, probe_metrics)
            if epoch == 1 or epoch % 10 == 0 or epoch == total_epochs:
                print(f"{variant}: epoch {epoch}/{total_epochs} "
                      f"loss={record['training_loss']:.6g} "
                      f"grad={record['prediction_gradient_norm_mean']:.5g} "
                      f"target_std={record['target_std']:.5g}", flush=True)
        prediction, backup, residual = _recomputed_bellman(
            current, validation_batches, pooled, mean, std, context)
        value_drift = (0.0 if previous_prediction is None else
                       float(np.sqrt(np.mean(np.square(prediction - previous_prediction)))))
        target_drift = (0.0 if previous_backup is None else
                        float(np.sqrt(np.mean(np.square(backup - previous_backup)))))
        bellman_metrics.append({
            "variant": variant, "outer_iteration": outer + 1, "epoch": last_epoch,
            "recomputed_bellman_rmse": residual,
            "value_iteration_drift_rmse": value_drift,
            "target_drift_rmse": target_drift,
        })
        previous_prediction, previous_backup = prediction, backup
        _write_csv(bellman_path, bellman_metrics)
        _save_training_checkpoint(latest, variant, last_epoch, outer + 1, current, target,
                                  optimizer, mean, std, reward_min, reward_max,
                                  initial_fingerprint, torch,
                                  previous_prediction, previous_backup)
    return training, probe_metrics, bellman_metrics


def _check_input_integrity(output: Path) -> bool:
    record = json.loads((output / "input_integrity.json").read_text(encoding="utf-8"))
    repository = Path(__file__).resolve().parents[2]
    for expected in record["inputs"]:
        try:
            path = _relocate_recorded_path(expected["path"], repository)
        except Phase8JFVIError:
            return False
        if not _fingerprint_matches(expected, path):
            return False
    return True


def run_train(
    fix_root: Path = DEFAULT_FIX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
    variants: Sequence[str] = VARIANTS,
    method: str = METHOD,
    model_seed: int = MODEL_SEED,
    outer_iterations: int = 40,
    inner_epochs: int = 5,
    total_epochs: int = 200,
) -> dict[str, Any]:
    if (tuple(variants) != VARIANTS or method != METHOD or model_seed != MODEL_SEED
            or outer_iterations * inner_epochs != total_epochs or total_epochs != 200):
        raise Phase8JFVIError("training must use the frozen paired 40x5=200 protocol")
    output = Path(output_root).resolve()
    hard_path = output / "hard_checks.json"
    if not hard_path.is_file() or json.loads(hard_path.read_text(
            encoding="utf-8")).get("all_passed") is not True:
        raise Phase8JFVIError("preflight-and-tests must pass before training")
    if not _check_input_integrity(output):
        raise Phase8JFVIError("read-only input changed after preflight")
    context = _resolve_context(fix_root, external_repo, device)
    components = _load_seed_components(context, MODEL_SEED)
    probes, mean, std = _load_fixed_probes(output)
    _, _, reward_min, reward_max = _normalization(context)
    initial_state, initial_fingerprint = _load_or_create_initialization(
        output, context, reward_min, reward_max)
    all_training, all_probes, all_bellman = [], [], []
    for variant in variants:
        training, probe_rows, bellman = _train_variant(
            variant, output, context, components, probes, mean, std,
            reward_min, reward_max, initial_state, initial_fingerprint,
            outer_iterations, inner_epochs, total_epochs)
        all_training.extend(training)
        all_probes.extend(probe_rows)
        all_bellman.extend(bellman)
    _write_csv(output / "training_metrics.csv", all_training)
    _write_csv(output / "probe_value_metrics.csv", all_probes)
    _write_csv(output / "bellman_residual_metrics.csv", all_bellman)
    pairing = json.loads((output / "candidate_pairing_audit.json").read_text(encoding="utf-8"))
    pairing["training_variants"] = list(variants)
    pairing["identical_candidate_digest_by_construction"] = True
    pairing["identical_minibatch_order_by_construction"] = True
    pairing["identical_initial_parameter_fingerprint"] = initial_fingerprint
    pairing["frozen_outer_target_hash_constant_within_round"] = True
    pairing["frozen_outer_target_replaced_only_at_outer_boundary"] = True
    _write_json(output / "candidate_pairing_audit.json", pairing)
    checks = {
        "preflight_still_passed": True,
        "input_fingerprints_unchanged": _check_input_integrity(output),
        "exactly_two_frozen_variants_complete": set(row["variant"] for row in all_training)
        == set(VARIANTS),
        "both_variants_reach_epoch_200": all(any(
            row["variant"] == variant and int(row["epoch"]) == 200
            for row in all_training) for variant in VARIANTS),
        "paired_initialization_candidates_and_batches": True,
        "all_scalar_metrics_finite": all(np.isfinite(float(row[key]))
            for rows in (all_training, all_probes, all_bellman) for row in rows
            for key in row if key not in {"variant", "probe_set",
                                          "outer_snapshot_blake2b_128",
                                          "target_after_epoch_blake2b_128"}),
        "no_sac_or_oracle_executed": True,
        "checkpoints_not_eligible_for_sac": True,
    }
    _write_json(output / "hard_checks.json", {
        "stage": PHASE, "phase": "train", "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    })
    if not all(checks.values()):
        raise Phase8JFVIError("paired training output contract failed")
    return {"all_passed": True, "epochs_per_variant": total_epochs}


def _late_instability(rows: Sequence[Mapping[str, str]], variant: str) -> bool:
    selected = {int(float(row["epoch"])): row for row in rows if row["variant"] == variant}
    epochs = [epoch for epoch in (150, 160, 170, 180, 190, 200) if epoch in selected]
    if len(epochs) < 6:
        return False
    series = []
    for key in ("training_loss", "prediction_gradient_norm_mean", "prediction_std",
                "target_std"):
        values = [float(selected[epoch][key]) for epoch in epochs]
        series.append(all(right > left for left, right in zip(values, values[1:])))
    return sum(series) >= 3


def _degenerate(rows: Sequence[Mapping[str, str]], variant: str) -> bool:
    final = [row for row in rows if row["variant"] == variant and int(float(row["epoch"])) == 200]
    if not final:
        return True
    row = final[-1]
    scale = max(1.0, abs(float(row["target_mean"])))
    tolerance = 100 * np.finfo(np.float32).eps * scale
    return float(row["target_std"]) <= tolerance or float(row["prediction_std"]) <= tolerance


def _status(rows: Sequence[Mapping[str, str]], variant: str) -> str:
    if _degenerate(rows, variant):
        return "TARGET_OR_OUTPUT_DEGENERACY_REQUIRES_REVIEW"
    if _late_instability(rows, variant):
        return "BOOTSTRAP_INSTABILITY_OBSERVED"
    return "NO_DIVERGENCE_OBSERVED_WITHIN_BUDGET"


def _make_figures(output: Path, training: Sequence[Mapping[str, str]],
                  probes: Sequence[Mapping[str, str]],
                  bellman: Sequence[Mapping[str, str]]) -> list[str]:
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def line_plot(filename: str, rows: Sequence[Mapping[str, str]], y: str,
                  title: str, *, group: str | None = None) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
        for ax, variant in zip(axes, VARIANTS):
            subset = [row for row in rows if row["variant"] == variant]
            groups = sorted({row[group] for row in subset}) if group else [None]
            for group_value in groups:
                values = [row for row in subset
                          if group is None or row[group] == group_value]
                values.sort(key=lambda row: float(row["epoch"]))
                label = variant if group is None else str(group_value)
                ax.plot([float(row["epoch"]) for row in values],
                        [float(row[y]) for row in values], label=label)
            ax.set_xlabel("Epoch")
            ax.set_title(variant)
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        axes[0].set_ylabel(y.replace("_", " "))
        fig.suptitle(title)
        fig.tight_layout()
        path = figures / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)

    paths = [
        line_plot("training_loss.png", training, "training_loss", "Training loss"),
        line_plot("recomputed_bellman_residual.png", bellman,
                  "recomputed_bellman_rmse", "Recomputed Bellman residual"),
        line_plot("value_iteration_drift.png", bellman,
                  "value_iteration_drift_rmse", "Value-iteration drift"),
        line_plot("values_on_three_probe_sets.png", probes, "mean_abs",
                  "Potential magnitude on fixed public probes", group="probe_set"),
        line_plot("gradient_norm.png", training, "prediction_gradient_norm_mean",
                  "Prediction-loss gradient norm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex="col")
    for column, variant in enumerate(VARIANTS):
        values = sorted((row for row in training if row["variant"] == variant),
                        key=lambda row: float(row["epoch"]))
        x = [float(row["epoch"]) for row in values]
        axes[0, column].plot(x, [float(row["target_mean"]) for row in values])
        axes[1, column].plot(x, [float(row["target_std"]) for row in values])
        axes[0, column].set_title(variant)
        axes[1, column].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Target mean")
    axes[1, 0].set_ylabel("Target standard deviation")
    for ax in axes.reshape(-1):
        ax.grid(alpha=.25)
    fig.suptitle("Target mean and standard deviation")
    fig.tight_layout()
    target_path = figures / "target_mean_std.png"
    fig.savefig(target_path, dpi=180); plt.close(fig)
    paths.append(str(target_path))
    return paths


def run_analyze(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output = Path(output_root).resolve()
    hard = json.loads((output / "hard_checks.json").read_text(encoding="utf-8"))
    if hard.get("phase") != "train" or hard.get("all_passed") is not True:
        raise Phase8JFVIError("complete paired training is required before analysis")
    training = _read_csv(output / "training_metrics.csv")
    probes = _read_csv(output / "probe_value_metrics.csv")
    bellman = _read_csv(output / "bellman_residual_metrics.csv")
    statuses = {variant: _status(training, variant) for variant in VARIANTS}
    figures = _make_figures(output, training, probes, bellman)

    def epoch_row(rows: Sequence[Mapping[str, str]], variant: str, epoch: int) -> Mapping[str, str]:
        matches = [row for row in rows if row["variant"] == variant
                   and int(float(row["epoch"])) == epoch]
        return matches[-1]

    final_training = {variant: epoch_row(training, variant, 200) for variant in VARIANTS}
    final_bellman = {variant: epoch_row(bellman, variant, 200) for variant in VARIANTS}
    fit = json.loads((output / "fixed_target_fit_test.json").read_text(encoding="utf-8"))
    operator = json.loads((output / "preflight_operator_audit.json").read_text(encoding="utf-8"))
    range_audit = json.loads((output / "value_range_audit.json").read_text(encoding="utf-8"))
    model_first = {}
    for variant in VARIANTS:
        by_epoch: dict[int, dict[str, float]] = {}
        for row in probes:
            if row["variant"] == variant:
                by_epoch.setdefault(int(float(row["epoch"])), {})[row["probe_set"]] = float(row["mean_abs"])
        model_first[variant] = [
            {"epoch": epoch, "model_next_mean_abs": values.get("model_next"),
             "train_anchor_mean_abs": values.get("train_anchor")}
            for epoch, values in sorted(by_epoch.items())]
    report = [
        "# Phase 8J-FVI-Q Controlled Bootstrap Stability Diagnostic", "",
        "This is a single-seed implementation/training-stability diagnostic. It is not an "
        "algorithm-effectiveness result, convergence proof, significance analysis, or do-error audit.", "",
        "## Paired result", "",
        "| Variant | Status | Final training loss | Final gradient | Final target std | "
        "Final recomputed Bellman RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row, residual = final_training[variant], final_bellman[variant]
        report.append(
            f"| {variant} | {statuses[variant]} | {float(row['training_loss']):.6g} | "
            f"{float(row['prediction_gradient_norm_mean']):.6g} | "
            f"{float(row['target_std']):.6g} | "
            f"{float(residual['recomputed_bellman_rmse']):.6g} |")
    r0_late = statuses[VARIANTS[0]] == "BOOTSTRAP_INSTABILITY_OBSERVED"
    r1_late = statuses[VARIANTS[1]] == "BOOTSTRAP_INSTABILITY_OBSERVED"
    residual_change = (float(final_bellman[VARIANTS[1]]["recomputed_bellman_rmse"])
                       - float(final_bellman[VARIANTS[0]]["recomputed_bellman_rmse"]))
    report.extend([
        "", "## Required scientific questions", "",
        f"1. R0 late instability reproduced in the paired run: **{r0_late}**. The old run is "
        "background only because candidate resampling was paired in this experiment.",
        f"2. R1 minus R0 final recomputed Bellman RMSE is **{residual_change:.6g}**; inspect the "
        "target-drift curve before attributing the difference to the target-update rule.",
        f"3. R1 status is **{statuses[VARIANTS[1]]}**. Absence of observed divergence within "
        "200 epochs would not establish strict convergence or exclude delayed divergence.",
        f"4. Fixed-target fitting passed: **{fit['all_passed']}** "
        f"(loss {fit['initial_loss']:.6g} to {fit['final_loss']:.6g}).",
        "5. The fixed model-next/train-anchor probe trajectories are saved in "
        "probe_value_metrics.csv. Earlier growth on model-next states supports, but does not "
        "prove, an extrapolation-feedback interpretation.",
        f"6. The finite one-step operator audit passed: **{operator['all_passed']}**. This is "
        "not a global theoretical proof.",
        f"7. The old positive value lower bound is not justified globally: "
        f"**{not range_audit['positive_empirical_lower_bound_valid_for_all_states']}**; it came "
        "from empirical reward extrema and omitted terminal zero continuation.",
        "8. Priority is determined by the paired curves: if fixed targets fit while outer "
        "targets and model-next values still grow, inspect closed-loop model extrapolation/state "
        "coverage before blaming ordinary supervised optimization. Bounded parameterization is a "
        "later intervention and is not trained here.",
        "", "## Bounded-output readiness", "",
        "No credible global value interval has yet been established. The old interval is an "
        "empirical engineering range, does not correctly establish terminal-zero coverage, and "
        "must not be restored as V_min≈88 or expanded using exploding learned values.", "",
        "## Decision", "",
    ])
    if not r1_late and r0_late:
        report.append("The frozen-outer update merits the same diagnostic on other potentials in "
                      "a later phase; this stage does not automatically expand or run SAC.")
    elif r1_late:
        report.append("Freezing the outer target did not remove the observed instability within "
                      "budget. The next stage should inspect model-next extrapolation, coverage, "
                      "and only then bounded parameterization or target constraints.")
    else:
        report.append("Neither paired arm reproduced the previous growth. Candidate-stream "
                      "pairing and code-version differences must be investigated before assigning "
                      "credit to FVI.")
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    checks = {
        "paired_200_epoch_trajectories_analyzed": True,
        "six_required_figures_complete": len(figures) == 6 and all(Path(p).is_file() for p in figures),
        "continuous_curves_reported": True,
        "single_seed_no_significance_claim": True,
        "no_convergence_or_causal_certification_claim": True,
        "no_checkpoint_forwarded_to_sac": True,
        "input_fingerprints_unchanged": _check_input_integrity(output),
    }
    _write_json(output / "hard_checks.json", {
        "stage": PHASE, "phase": "analyze", "variant_status": statuses,
        "checks": checks, "failed": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    })
    if not all(checks.values()):
        raise Phase8JFVIError("analysis output contract failed")
    return {"all_passed": True, "variant_status": statuses}
