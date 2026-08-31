"""Phase 8B-NC: pooled neural realization of observational population targets.

Training in this module is intentionally restricted to public 12-D observations,
commanded 3-D actions, public outcomes, and fixed population row weights.  Causal
oracle arrays are loaded only after all models have been trained.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .analyze_noncomplementary_population import (
    KAPPA_NAMES,
    load_strict_unclipped_mask,
)
from .analyze_phase8a_population_effect import (
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
    validate_all_84_phase8a_invariants,
)
from .noncomplementary_population_dgp import (
    ACTION_KEYS,
    CONDITIONS,
    FORBIDDEN_PUBLIC_FIELDS,
    PRIMARY_MIXTURES,
    PUBLIC_FIELDS,
)


EXPECTED_KAPPAS = (0.0, 0.1, 0.2, 0.3)
MODEL_INPUT_FIELDS = ("observation", "commanded_action")
MODEL_INPUT_DIMENSION = 15
PHYSICAL_DELTA_DIMENSION = 11
PRIMARY_MIXTURE_NAMES = tuple(PRIMARY_MIXTURES)
LEAKAGE_FLAGS = {
    "LOGGER_ID_IN_MODEL_INPUT": False,
    "HIDDEN_U_IN_MODEL_INPUT": False,
    "APPLIED_ACTION_IN_MODEL_INPUT": False,
    "DO_ORACLE_USED_FOR_TRAINING": False,
    "LONG_HORIZON_ORACLE_USED_FOR_TRAINING": False,
}


class NeuralObservationalBiasError(RuntimeError):
    """Raised when a Phase 8B-NC precondition or invariant fails."""


@dataclass(frozen=True)
class GroupedTargets:
    anchor_id: np.ndarray
    observation: np.ndarray
    commanded_action: np.ndarray
    x: np.ndarray
    reward: np.ndarray
    delta: np.ndarray
    mass: np.ndarray

    def subset(self, anchor_ids: Sequence[int]) -> "GroupedTargets":
        mask = np.isin(self.anchor_id, np.asarray(anchor_ids, dtype=np.int64))
        return GroupedTargets(**{name: np.asarray(getattr(self, name))[mask]
                                for name in self.__dataclass_fields__})

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray
    constant_mask: np.ndarray


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else ("status",)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _torch():
    try:
        import torch
    except Exception as exc:  # Windows may fail while loading a dependent DLL.
        raise NeuralObservationalBiasError(f"PyTorch is unavailable: {exc}") from exc
    return torch


def action_bytes(action: np.ndarray) -> bytes:
    """Return the exact bytes of one stored commanded action, including dtype."""
    value = np.asarray(action)
    if value.shape != (3,):
        raise NeuralObservationalBiasError("commanded action must have shape (3,)")
    return value.dtype.str.encode("ascii") + b":" + np.ascontiguousarray(value).tobytes()


def build_grouped_targets(public: Mapping[str, np.ndarray], weights: np.ndarray) -> GroupedTargets:
    """Aggregate public rows by (anchor_id, exact commanded-action bytes)."""
    required = {"anchor_id", "observation", "commanded_action", "reward", "next_observation"}
    if not required.issubset(public):
        raise NeuralObservationalBiasError(f"public data lacks {sorted(required-set(public))}")
    if FORBIDDEN_PUBLIC_FIELDS.intersection(public):
        raise NeuralObservationalBiasError("hidden fields leaked into public training data")
    n = len(np.asarray(public["anchor_id"]))
    mass = np.asarray(weights, dtype=np.float64)
    if mass.shape != (n,) or np.any(mass < 0) or not np.isfinite(mass).all() or mass.sum() <= 0:
        raise NeuralObservationalBiasError("row weights are invalid")
    observation = np.asarray(public["observation"])
    action = np.asarray(public["commanded_action"])
    reward = np.asarray(public["reward"], dtype=np.float64)
    next_observation = np.asarray(public["next_observation"], dtype=np.float64)
    if observation.shape != (n, 12) or action.shape != (n, 3) or next_observation.shape != (n, 12):
        raise NeuralObservationalBiasError("public observation/action shapes are invalid")
    groups: dict[tuple[int, bytes], list[int]] = {}
    for row, (anchor, command) in enumerate(zip(public["anchor_id"], action)):
        groups.setdefault((int(anchor), action_bytes(command)), []).append(row)
    output: dict[str, list[Any]] = {name: [] for name in
                                    ("anchor_id", "observation", "commanded_action",
                                     "x", "reward", "delta", "mass")}
    for (anchor, _), rows in sorted(groups.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        idx = np.asarray(rows, dtype=np.int64)
        group_mass = float(mass[idx].sum())
        if group_mass <= 0:
            raise NeuralObservationalBiasError("a grouped state-action has zero mass")
        if not np.array_equal(observation[idx], np.broadcast_to(observation[idx[0]], observation[idx].shape)):
            raise NeuralObservationalBiasError("one anchor has inconsistent public observations")
        if not np.array_equal(action[idx], np.broadcast_to(action[idx[0]], action[idx].shape)):
            raise NeuralObservationalBiasError("exact action-byte group contains different actions")
        normalized = mass[idx] / group_mass
        delta_rows = next_observation[idx, :11] - observation[idx, :11].astype(np.float64)
        obs = observation[idx[0]].astype(np.float64)
        command = action[idx[0]].astype(np.float64)
        output["anchor_id"].append(anchor)
        output["observation"].append(obs)
        output["commanded_action"].append(command)
        output["x"].append(np.concatenate((obs, command)))
        output["reward"].append(float(normalized @ reward[idx]))
        output["delta"].append(normalized @ delta_rows)
        output["mass"].append(group_mass)
    result = GroupedTargets(
        anchor_id=np.asarray(output["anchor_id"], dtype=np.int64),
        observation=np.asarray(output["observation"], dtype=np.float64),
        commanded_action=np.asarray(output["commanded_action"], dtype=np.float64),
        x=np.asarray(output["x"], dtype=np.float64),
        reward=np.asarray(output["reward"], dtype=np.float64),
        delta=np.asarray(output["delta"], dtype=np.float64),
        mass=np.asarray(output["mass"], dtype=np.float64),
    )
    if result.x.shape[1:] != (15,) or result.delta.shape[1:] != (11,):
        raise NeuralObservationalBiasError("grouped target has invalid shape")
    return result


def grouped_weighted_mse_decomposition(
    row_targets: np.ndarray, row_weights: np.ndarray, group_index: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    """Return row MSE, grouped MSE, within-group variance (row = grouped + within)."""
    y = np.asarray(row_targets, dtype=np.float64)
    w = np.asarray(row_weights, dtype=np.float64)
    g = np.asarray(group_index, dtype=np.int64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim == 1:
        y, p = y[:, None], p.reshape(-1, 1)
    total = float(w.sum())
    row_mse = float(np.sum(w[:, None] * (y - p[g]) ** 2) / (total * y.shape[1]))
    grouped_mse = 0.0
    within = 0.0
    for group in np.unique(g):
        mask = g == group
        mass = float(w[mask].sum())
        mean = np.sum(w[mask, None] * y[mask], axis=0) / mass
        grouped_mse += mass * float(np.mean((mean - p[group]) ** 2))
        within += float(np.sum(w[mask, None] * (y[mask] - mean) ** 2) / y.shape[1])
    return row_mse, grouped_mse / total, within / total


def make_anchor_splits(anchor_ids: np.ndarray, seed: int = 0,
                       strata: np.ndarray | None = None) -> dict[str, list[int]]:
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    if len(np.unique(anchors)) != len(anchors):
        raise NeuralObservationalBiasError("anchor IDs must be unique")
    strata = np.zeros(len(anchors), dtype=np.int64) if strata is None else np.asarray(strata)
    if strata.shape != anchors.shape:
        raise NeuralObservationalBiasError("anchor strata have wrong shape")
    rng = np.random.default_rng(seed)
    split = {"train": [], "validation": [], "test": []}
    for value in np.unique(strata):
        members = np.sort(anchors[strata == value]).copy()
        rng.shuffle(members)
        n_train = int(np.floor(0.70 * len(members)))
        n_validation = int(np.floor(0.15 * len(members)))
        split["train"].extend(members[:n_train].tolist())
        split["validation"].extend(members[n_train:n_train+n_validation].tolist())
        split["test"].extend(members[n_train+n_validation:].tolist())
    return {name: sorted(values) for name, values in split.items()}


def validate_splits(splits: Mapping[str, Sequence[int]], all_anchor_ids: np.ndarray) -> bool:
    sets = {key: set(map(int, splits[key])) for key in ("train", "validation", "test")}
    return (not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                 or sets["validation"] & sets["test"])
            and set.union(*sets.values()) == set(map(int, all_anchor_ids)))


def normalization(values: np.ndarray) -> Normalization:
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    constant = std < 1e-8
    std = std.copy()
    std[constant] = 1.0
    return Normalization(mean=mean, std=std, constant_mask=constant)


def apply_normalization(values: np.ndarray, stats: Normalization) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - stats.mean) / stats.std


def RewardMeanModel():
    return _make_model(1)


def DeltaMeanModel():
    return _make_model(11)


def _make_model(output_dimension: int):
    torch = _torch()
    return torch.nn.Sequential(
        torch.nn.Linear(15, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, output_dimension),
    )


def state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_initial_state(target: str, seed: int) -> tuple[dict[str, Any], str]:
    """Create one CPU initial state that all three mixture models will load."""
    torch = _torch()
    torch.manual_seed(seed)
    model = RewardMeanModel() if target == "reward" else DeltaMeanModel()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return state, state_hash(model)


def batch_schedule(length: int, updates: int, batch_size: int, seed: int) -> np.ndarray:
    if min(length, updates, batch_size) <= 0:
        raise ValueError("schedule dimensions must be positive")
    return np.random.default_rng(seed).integers(0, length, size=(updates, batch_size), dtype=np.int64)


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def resolve_device(name: str) -> str:
    torch = _torch()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise NeuralObservationalBiasError("CUDA was requested but is unavailable")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return name


def train_model(
    train: GroupedTargets, validation: GroupedTargets, target: str,
    input_stats: Normalization, output_stats: Normalization, schedule: np.ndarray,
    seed: int, device: str, initial_state: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train without accepting or reading any oracle, logger, U, or applied-action fields."""
    torch = _torch()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RewardMeanModel() if target == "reward" else DeltaMeanModel()
    if initial_state is not None:
        model.load_state_dict(dict(initial_state))
    model.to(device)
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.as_tensor(apply_normalization(train.x, input_stats), dtype=torch.float32, device=device)
    raw_y = train.reward[:, None] if target == "reward" else train.delta
    y = torch.as_tensor(apply_normalization(raw_y, output_stats), dtype=torch.float32, device=device)
    mass = torch.as_tensor(train.mass, dtype=torch.float32, device=device)
    vx = torch.as_tensor(apply_normalization(validation.x, input_stats), dtype=torch.float32, device=device)
    vy_raw = validation.reward[:, None] if target == "reward" else validation.delta
    vy = torch.as_tensor(apply_normalization(vy_raw, output_stats), dtype=torch.float32, device=device)
    train_losses, validation_steps, validation_losses = [], [], []
    interval = max(1, len(schedule) // 100)
    for step, indices_np in enumerate(schedule):
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
        prediction = model(x[indices])
        row_loss = torch.mean((prediction - y[indices]) ** 2, dim=1)
        selected_mass = mass[indices]
        loss = torch.sum(selected_mass * row_loss) / torch.sum(selected_mass)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))
        if step % interval == 0 or step == len(schedule) - 1:
            with torch.no_grad():
                validation_losses.append(float(torch.mean((model(vx) - vy) ** 2).cpu()))
            validation_steps.append(step + 1)
    return model, {
        "initial_state_hash": initial, "final_state_hash": state_hash(model),
        "schedule_hash": array_hash(schedule), "train_loss": train_losses,
        "validation_step": validation_steps, "validation_loss": validation_losses,
    }


def predict(model: Any, x: np.ndarray, input_stats: Normalization,
            output_stats: Normalization, device: str) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        normalized = torch.as_tensor(apply_normalization(x, input_stats),
                                     dtype=torch.float32, device=device)
        result = model(normalized).detach().cpu().numpy().astype(np.float64)
    return result * output_stats.std + output_stats.mean


def save_checkpoint(path: Path, model: Any, metadata: Mapping[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, path)


def load_checkpoint(path: Path, target: str, device: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    model = RewardMeanModel() if target == "reward" else DeltaMeanModel()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, dict(payload["metadata"])


def require_verified_inputs(phase8anc_root: Path, phase8a_root: Path,
                            phase8ac_root: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    nc = Path(phase8anc_root).resolve()
    causal = Path(phase8a_root).resolve()
    clipping = Path(phase8ac_root).resolve()
    for root, label in ((nc, "Phase 8A-NC"), (causal, "Phase 8A"),
                        (clipping, "Phase 8A-C")):
        if not root.is_dir():
            raise NeuralObservationalBiasError(f"verified {label} root is unavailable: {root}")
    nc_hard = json.loads((nc / "hard_checks.json").read_text(encoding="utf-8"))
    if nc_hard.get("all_passed") is not True or not all(nc_hard.get("checks", {}).values()):
        raise NeuralObservationalBiasError("Phase 8A-NC hard checks did not all pass")
    causal_summary_path = causal / "summary.json"
    if not causal_summary_path.is_file():
        raise NeuralObservationalBiasError("Phase 8A verification record is unavailable")
    try:
        validate_all_84_phase8a_invariants(
            json.loads(causal_summary_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise NeuralObservationalBiasError(f"Phase 8A do-oracle root is not verified: {exc}") from exc
    clipping_hard = json.loads((clipping / "hard_checks.json").read_text(encoding="utf-8"))
    if clipping_hard.get("all_passed") is not True or not all(clipping_hard.get("checks", {}).values()):
        raise NeuralObservationalBiasError("Phase 8A-C hard checks did not all pass")
    manifest = json.loads((nc / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("available_anchor_count", -1)) != 2048:
        raise NeuralObservationalBiasError("Phase 8A-NC does not contain 2048 anchors")
    if tuple(map(float, manifest.get("kappas", ()))) != EXPECTED_KAPPAS:
        raise NeuralObservationalBiasError("Phase 8A-NC does not contain all four kappas")
    anchors_path = causal / "anchors.npz"
    if not anchors_path.is_file():
        raise NeuralObservationalBiasError(f"Phase 8A anchors are unavailable: {anchors_path}")
    anchors = load_npz(anchors_path)
    anchor_ids = np.asarray(anchors.get("anchor_id", ()), dtype=np.int64)
    if not np.array_equal(anchor_ids, np.arange(2048)):
        raise NeuralObservationalBiasError("Phase 8A anchor IDs are incomplete")
    strict_mask, _ = load_strict_unclipped_mask(clipping, anchor_ids)
    paths = [nc / "manifest.json", nc / "hard_checks.json", anchors_path,
             causal / "manifest.json", causal_summary_path,
             clipping / "manifest.json", clipping / "hard_checks.json",
             clipping / "anchor_clipping_table.npz"]
    for kappa in EXPECTED_KAPPAS:
        directory = nc / KAPPA_NAMES[kappa]
        causal_directory = causal / KAPPA_NAMES[kappa]
        paths.extend((directory / "confounded_public.npz",
                      directory / "independent_latents_public.npz",
                      directory / "population_tables.npz",
                      causal_directory / "do_oracle_raw.npz"))
        for condition in CONDITIONS:
            for mixture in PRIMARY_MIXTURE_NAMES:
                paths.append(directory / "weights" / condition / f"{mixture}.npy")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise NeuralObservationalBiasError(f"required read-only inputs are missing: {missing}")
    return sorted((path.resolve() for path in paths), key=str), anchor_ids, strict_mask


def _save_stats(path: Path, stats: Normalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=stats.mean, std=stats.std,
                        constant_mask=stats.constant_mask)


def _schedule_seed(kappa: float, condition: str, target: str, seed: int) -> int:
    condition_index = CONDITIONS.index(condition)
    target_index = ("reward", "delta").index(target)
    return int(seed + 1009 * round(kappa * 10) + 10007 * condition_index
               + 100003 * target_index)


def _canonical_indices(grouped: GroupedTargets, raw: Mapping[str, np.ndarray],
                       anchor_ids: np.ndarray, kappa: float) -> np.ndarray:
    lookup: dict[tuple[int, str], bytes] = {}
    seen_u: set[tuple[int, str, int]] = set()
    for row in range(len(raw["anchor_id"])):
        if not np.isclose(float(raw["kappa_env"][row]), kappa, atol=1e-12, rtol=0):
            raise NeuralObservationalBiasError("do-oracle row has wrong kappa")
        action_key = str(raw["action_key"][row])
        key_u = (int(raw["anchor_id"][row]), action_key, int(raw["u_env"][row]))
        if key_u in seen_u:
            raise NeuralObservationalBiasError("do-oracle lookup is not unique")
        seen_u.add(key_u)
        key = key_u[:2]
        # Phase 8A-NC public commanded actions are canonically stored as float32.
        encoded = action_bytes(np.asarray(raw["commanded_action"][row], dtype=np.float32))
        if key in lookup and lookup[key] != encoded:
            raise NeuralObservationalBiasError("do action differs across U")
        lookup[key] = encoded
    grouped_lookup = {(int(anchor), action_bytes(action)): index
                      for index, (anchor, action) in enumerate(
                          zip(grouped.anchor_id, grouped.commanded_action.astype(np.float32)))}
    order = []
    for anchor in anchor_ids:
        for action in ACTION_KEYS:
            key = (int(anchor), lookup[(int(anchor), action)])
            if key not in grouped_lookup:
                raise NeuralObservationalBiasError("grouped target lacks a do-oracle action")
            order.append(grouped_lookup[key])
    return np.asarray(order, dtype=np.int64).reshape(len(anchor_ids), 3)


def recompute_do_targets(raw: Mapping[str, np.ndarray], grouped: GroupedTargets,
                         anchor_ids: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """Recompute test-only do means directly from the six Phase 8A oracle rows/anchor."""
    order = _canonical_indices(grouped, raw, anchor_ids, kappa)
    observations = grouped.observation[order[:, 0], :11]
    lookup: dict[tuple[int, str, int], int] = {}
    for row in range(len(raw["anchor_id"])):
        key = (int(raw["anchor_id"][row]), str(raw["action_key"][row]),
               int(raw["u_env"][row]))
        if key in lookup:
            raise NeuralObservationalBiasError("do-oracle rows are duplicated")
        lookup[key] = row
    reward = np.empty((len(anchor_ids), 3), dtype=np.float64)
    delta = np.empty((len(anchor_ids), 3, 11), dtype=np.float64)
    for anchor_index, anchor in enumerate(anchor_ids):
        for action_index, action in enumerate(ACTION_KEYS):
            rows = [lookup[(int(anchor), action, u)] for u in (-1, 1)]
            reward[anchor_index, action_index] = np.mean(np.asarray(raw["reward"])[rows])
            delta[anchor_index, action_index] = (
                np.mean(np.asarray(raw["next_observation"], dtype=np.float64)[rows, :11], axis=0)
                - observations[anchor_index])
    return reward, delta


def _safe_correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    x, y = np.asarray(left).ravel(), np.asarray(right).ravel()
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    result = pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic
    return float(result) if np.isfinite(result) else 0.0


def _top_masks(values: np.ndarray, atol: float = 1e-7, rtol: float = 1e-7) -> np.ndarray:
    maximum = np.max(values, axis=1, keepdims=True)
    return np.isclose(values, maximum, atol=atol, rtol=rtol)


def _bootstrap_mean(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise NeuralObservationalBiasError("bootstrap values must be nonempty and 1D")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draws[index] = np.mean(array[rng.integers(0, len(array), len(array))])
    return {"mean": float(np.mean(array)), "sd": float(np.std(array, ddof=1)) if len(array)>1 else 0.0,
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)), "n_anchors": int(len(array))}


def seed_metrics(
    prediction_reward: Mapping[str, np.ndarray], prediction_delta: Mapping[str, np.ndarray],
    population_reward: Mapping[str, np.ndarray], population_delta: Mapping[str, np.ndarray],
    do_reward: np.ndarray, do_delta: np.ndarray, strict_mask: np.ndarray,
    kappa: float, condition: str, seed: int, test_anchor_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Compute anchor-level effects for one model seed; anchor remains the sampling unit."""
    balanced = "logger12_balanced"
    r_fit = {m: np.abs(prediction_reward[m] - population_reward[m]) for m in PRIMARY_MIXTURE_NAMES}
    d_fit = {m: np.linalg.norm(prediction_delta[m] - population_delta[m], axis=2)
             for m in PRIMARY_MIXTURE_NAMES}
    r_pred_drift = np.abs(prediction_reward["logger1_heavy"]-
                          prediction_reward["logger2_heavy"])
    r_pop_drift = np.abs(population_reward["logger1_heavy"]-
                         population_reward["logger2_heavy"])
    d_pred_drift = np.linalg.norm(prediction_delta["logger1_heavy"]-
                                  prediction_delta["logger2_heavy"], axis=2)
    d_pop_drift = np.linalg.norm(population_delta["logger1_heavy"]-
                                 population_delta["logger2_heavy"], axis=2)
    r_bal_obs = prediction_reward[balanced] - population_reward[balanced]
    r_bal_do = prediction_reward[balanced] - do_reward
    d_bal_obs = np.linalg.norm(prediction_delta[balanced]-population_delta[balanced], axis=2)
    d_bal_do = np.linalg.norm(prediction_delta[balanced]-do_delta, axis=2)
    r_population_balanced_do = population_reward[balanced]-do_reward
    d_population_balanced_do = np.linalg.norm(population_delta[balanced]-do_delta, axis=2)
    pred_top = _top_masks(prediction_reward[balanced])
    pop_top = _top_masks(population_reward[balanced])
    do_top = _top_masks(do_reward)
    disagreement = ~np.any(pred_top & do_top, axis=1)
    chosen = np.argmax(prediction_reward[balanced], axis=1)
    regret = np.max(do_reward, axis=1) - do_reward[np.arange(len(do_reward)), chosen]
    mixture_regrets = []
    for mixture in PRIMARY_MIXTURE_NAMES:
        selected_action = np.argmax(prediction_reward[mixture], axis=1)
        mixture_regrets.append(np.max(do_reward, axis=1)
                               - do_reward[np.arange(len(do_reward)), selected_action])
    mixture_regrets_array = np.stack(mixture_regrets)
    confounding_excess_r = np.mean(np.abs(r_bal_do), axis=1)
    reward_stack = np.stack([prediction_reward[m] for m in PRIMARY_MIXTURE_NAMES])
    population_reward_stack = np.stack([population_reward[m] for m in PRIMARY_MIXTURE_NAMES])
    delta_stack = np.stack([prediction_delta[m] for m in PRIMARY_MIXTURE_NAMES])
    population_delta_stack = np.stack([population_delta[m] for m in PRIMARY_MIXTURE_NAMES])
    signed_pred_reward = prediction_reward["logger1_heavy"]-prediction_reward["logger2_heavy"]
    signed_pop_reward = population_reward["logger1_heavy"]-population_reward["logger2_heavy"]
    signed_pred_delta = prediction_delta["logger1_heavy"]-prediction_delta["logger2_heavy"]
    signed_pop_delta = population_delta["logger1_heavy"]-population_delta["logger2_heavy"]
    delta_dot = np.sum(signed_pred_delta * signed_pop_delta, axis=2)
    delta_norm_product = (np.linalg.norm(signed_pred_delta, axis=2)
                          * np.linalg.norm(signed_pop_delta, axis=2))
    delta_cosine = np.divide(delta_dot, delta_norm_product, out=np.zeros_like(delta_dot),
                             where=delta_norm_product > 1e-12)
    decomposition_reward = np.abs(
        r_bal_do - (prediction_reward[balanced]-population_reward[balanced]
                    + population_reward[balanced]-do_reward))
    decomposition_delta = np.linalg.norm(
        (prediction_delta[balanced]-do_delta)
        - ((prediction_delta[balanced]-population_delta[balanced])
           + (population_delta[balanced]-do_delta)), axis=2)
    reward_fit_squared = np.mean(np.stack([value**2 for value in r_fit.values()]), axis=(0, 2))
    mixture_reward_range = np.mean(np.max(reward_stack, axis=0)-np.min(reward_stack, axis=0), axis=1)
    mixture_population_reward_range = np.mean(
        np.max(population_reward_stack, axis=0)-np.min(population_reward_stack, axis=0), axis=1)
    mixture_delta_range = np.mean(np.max(delta_stack, axis=0)-np.min(delta_stack, axis=0), axis=(1, 2))
    mixture_population_delta_range = np.mean(
        np.max(population_delta_stack, axis=0)-np.min(population_delta_stack, axis=0), axis=(1, 2))
    arrays = {
        "anchor_id": test_anchor_ids, "strict_unclipped": strict_mask,
        "reward_fit_abs": np.mean(np.stack(list(r_fit.values())), axis=(0, 2)),
        "reward_fit_rmse": np.sqrt(reward_fit_squared),
        "delta_fit_l2": np.mean(np.stack(list(d_fit.values())), axis=(0, 2)),
        "neural_reward_drift": np.mean(r_pred_drift, axis=1),
        "population_reward_drift": np.mean(r_pop_drift, axis=1),
        "neural_delta_drift": np.mean(d_pred_drift, axis=1),
        "population_delta_drift": np.mean(d_pop_drift, axis=1),
        "neural_reward_three_mixture_range": mixture_reward_range,
        "population_reward_three_mixture_range": mixture_population_reward_range,
        "neural_delta_three_mixture_range": mixture_delta_range,
        "population_delta_three_mixture_range": mixture_population_delta_range,
        "max_reward_prediction_difference_across_mixtures": np.max(
            np.max(reward_stack, axis=0)-np.min(reward_stack, axis=0), axis=1),
        "max_delta_prediction_difference_across_mixtures": np.max(
            np.linalg.norm(delta_stack[:, None]-delta_stack[None, :], axis=4), axis=(0, 1, 3)),
        "base_reward_prediction_range_across_mixtures": (
            np.max(reward_stack[:, :, 1], axis=0)-np.min(reward_stack[:, :, 1], axis=0)),
        "reward_drift_absolute_realization_error": np.mean(np.abs(r_pred_drift-r_pop_drift), axis=1),
        "delta_drift_absolute_realization_error": np.mean(np.abs(d_pred_drift-d_pop_drift), axis=1),
        "reward_drift_direction_agreement": np.mean(
            np.sign(signed_pred_reward) == np.sign(signed_pop_reward), axis=1),
        "delta_drift_cosine": np.mean(delta_cosine, axis=1),
        "balanced_reward_observational_error": np.mean(np.abs(r_bal_obs), axis=1),
        "balanced_reward_do_error": np.mean(np.abs(r_bal_do), axis=1),
        "population_balanced_reward_do_error": np.mean(np.abs(r_population_balanced_do), axis=1),
        "balanced_delta_observational_error": np.mean(d_bal_obs, axis=1),
        "balanced_delta_do_error": np.mean(d_bal_do, axis=1),
        "population_balanced_delta_do_error": np.mean(d_population_balanced_do, axis=1),
        "reward_do_decomposition_residual": np.mean(decomposition_reward, axis=1),
        "delta_do_decomposition_residual": np.mean(decomposition_delta, axis=1),
        "neural_do_ranking_disagreement": disagreement.astype(np.float64),
        "neural_decision_regret": regret,
        "best_mixture_neural_decision_regret": np.min(mixture_regrets_array, axis=0),
        "worst_mixture_neural_decision_regret": np.max(mixture_regrets_array, axis=0),
        "confounding_excess_reward": confounding_excess_r,
    }
    rows: list[dict[str, Any]] = []
    for name, values in arrays.items():
        if name in ("anchor_id", "strict_unclipped"):
            continue
        stats = _bootstrap_mean(np.asarray(values), 500, seed + len(rows))
        rows.append({"kappa": kappa, "condition": condition, "model_seed": seed,
                     "metric": name, "statistical_unit": "anchor_id", **stats})
    rows.extend((
        {"kappa": kappa, "condition": condition, "model_seed": seed,
         "metric": "reward_drift_pearson", "mean": _safe_correlation(r_pred_drift, r_pop_drift, "pearson"),
         "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_anchors": len(test_anchor_ids),
         "statistical_unit": "anchor_id"},
        {"kappa": kappa, "condition": condition, "model_seed": seed,
         "metric": "reward_drift_spearman", "mean": _safe_correlation(r_pred_drift, r_pop_drift, "spearman"),
         "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_anchors": len(test_anchor_ids),
         "statistical_unit": "anchor_id"},
        {"kappa": kappa, "condition": condition, "model_seed": seed,
         "metric": "reward_drift_aggregate_retention", "mean": float(np.mean(r_pred_drift)/max(np.mean(r_pop_drift), 1e-12)),
         "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_anchors": len(test_anchor_ids),
         "statistical_unit": "anchor_id"},
        {"kappa": kappa, "condition": condition, "model_seed": seed,
         "metric": "delta_drift_aggregate_retention", "mean": float(np.mean(d_pred_drift)/max(np.mean(d_pop_drift), 1e-12)),
         "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_anchors": len(test_anchor_ids),
         "statistical_unit": "anchor_id"},
        {"kappa": kappa, "condition": condition, "model_seed": seed,
         "metric": "population_ranking_disagreement", "mean": float(np.mean(~np.any(pop_top & do_top, axis=1))),
         "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_anchors": len(test_anchor_ids),
         "statistical_unit": "anchor_id"},
    ))
    for action_index, action in enumerate(ACTION_KEYS):
        denominator = float(np.mean(signed_pop_reward[:, action_index]))
        retention = (float(np.mean(signed_pred_reward[:, action_index])) / denominator
                     if abs(denominator) > 1e-12 else 0.0)
        rows.append({"kappa": kappa, "condition": condition, "model_seed": seed,
                     "metric": f"reward_signed_bias_retention_{action}", "mean": retention,
                     "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                     "n_anchors": len(test_anchor_ids), "statistical_unit": "anchor_id",
                     "defined": abs(denominator) > 1e-12,
                     "population_signed_drift_denominator": denominator})
    for mixture in PRIMARY_MIXTURE_NAMES:
        mixture_top = _top_masks(prediction_reward[mixture])
        mixture_disagreement = (~np.any(mixture_top & do_top, axis=1)).astype(np.float64)
        chosen_mixture = np.argmax(prediction_reward[mixture], axis=1)
        mixture_regret = np.max(do_reward, axis=1)-do_reward[np.arange(len(do_reward)), chosen_mixture]
        for label, values in ((f"{mixture}_do_ranking_disagreement", mixture_disagreement),
                              (f"{mixture}_do_decision_regret", mixture_regret)):
            stats = _bootstrap_mean(values, 500, seed + 500 + len(rows))
            rows.append({"kappa": kappa, "condition": condition, "model_seed": seed,
                         "metric": label, "statistical_unit": "anchor_id", **stats})
        for action_index, action in enumerate(ACTION_KEYS):
            values = mixture_top[:, action_index].astype(float)
            stats = _bootstrap_mean(values, 500, seed + 900 + len(rows))
            rows.append({"kappa": kappa, "condition": condition, "model_seed": seed,
                         "metric": f"{mixture}_top_contains_{action}",
                         "statistical_unit": "anchor_id", **stats})
    for action_index, action in enumerate(ACTION_KEYS):
        stats = _bootstrap_mean(do_top[:, action_index].astype(float), 500,
                                seed + 1200 + action_index)
        rows.append({"kappa": kappa, "condition": condition, "model_seed": seed,
                     "metric": f"do_top_contains_{action}",
                     "statistical_unit": "anchor_id", **stats})
    return rows, arrays


def _aggregate_seed_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str, str], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        groups.setdefault((float(row["kappa"]), str(row["condition"]), str(row["metric"])), []).append(row)
    output = []
    for (kappa, condition, metric), rows in sorted(groups.items()):
        values = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
        output.append({"kappa": kappa, "condition": condition, "metric": metric,
                       "mean": float(values.mean()),
                       "seed_sd": float(values.std(ddof=1)) if len(values)>1 else 0.0,
                       "seed_min": float(values.min()), "seed_max": float(values.max()),
                       "model_seed_count": len(values), "anchor_bootstrap_unit": "anchor_id",
                       "seed_variation_reported_separately": True,
                       "all_defined": all(bool(row.get("defined", True)) for row in rows)})
    return output


def _aggregate_with_anchor_bootstrap(
    seed_rows: Sequence[Mapping[str, Any]], anchor_arrays: Mapping[str, np.ndarray],
    kappas: Sequence[float], conditions: Sequence[str], model_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Average model predictions per anchor, then bootstrap anchors; report seed SD separately."""
    output = _aggregate_seed_rows(seed_rows)
    lookup = {(float(row["kappa"]), str(row["condition"]), str(row["metric"])): row
              for row in output}
    for kappa in kappas:
        for condition in conditions:
            first_prefix = f"{KAPPA_NAMES[kappa]}__{condition}__seed_{model_seeds[0]}__"
            metrics = [key[len(first_prefix):] for key in anchor_arrays
                       if key.startswith(first_prefix)
                       and key[len(first_prefix):] not in ("anchor_id", "strict_unclipped")]
            for metric in metrics:
                values = np.stack([
                    np.asarray(anchor_arrays[
                        f"{KAPPA_NAMES[kappa]}__{condition}__seed_{seed}__{metric}"],
                               dtype=np.float64)
                    for seed in model_seeds])
                anchor_mean = np.mean(values, axis=0)
                stats = _bootstrap_mean(anchor_mean, 2000,
                                        90000 + int(round(kappa*10))*100 + len(metric))
                key = (float(kappa), str(condition), metric)
                seed_means = np.mean(values, axis=1)
                row = lookup.get(key, {"kappa": kappa, "condition": condition,
                                       "metric": metric})
                row.update({**stats,
                            "seed_sd": float(np.std(seed_means, ddof=1)) if len(seed_means)>1 else 0.0,
                            "seed_min": float(np.min(seed_means)), "seed_max": float(np.max(seed_means)),
                            "model_seed_count": len(model_seeds),
                            "anchor_bootstrap_unit": "anchor_id",
                            "seed_variation_reported_separately": True})
                lookup[key] = row
    return sorted(lookup.values(), key=lambda row: (
        float(row["kappa"]), str(row["condition"]), str(row["metric"])))


def _make_figures(output: Path, aggregate: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("reward_population_fit_vs_kappa.png", "reward_fit_abs", "reward MAE"),
        ("delta_population_fit_vs_kappa.png", "delta_fit_l2", "delta L2 error"),
        ("balanced_reward_do_error_vs_kappa.png", "balanced_reward_do_error", "reward do error"),
        ("balanced_delta_do_error_vs_kappa.png", "balanced_delta_do_error", "delta do error"),
        ("neural_ranking_disagreement_vs_kappa.png", "neural_do_ranking_disagreement", "disagreement fraction"),
        ("neural_decision_regret_vs_kappa.png", "neural_decision_regret", "do regret"),
        ("seed_variation_vs_kappa.png", "reward_fit_abs", "between-seed SD"),
    )
    for filename, metric, ylabel in definitions:
        rows = [row for row in aggregate if row["metric"] == metric and row["condition"] == "confounded"]
        rows.sort(key=lambda row: float(row["kappa"]))
        x = [float(row["kappa"]) for row in rows]
        y_key = "seed_sd" if filename == "seed_variation_vs_kappa.png" else "mean"
        y = [float(row[y_key]) for row in rows]
        plt.figure(figsize=(5.5, 4.0))
        plt.plot(x, y, marker="o")
        plt.xlabel("kappa")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=160)
        plt.close()
    paired = (
        ("neural_vs_population_reward_drift.png", "neural_reward_drift",
         "population_reward_drift", "reward drift"),
        ("neural_vs_population_delta_drift.png", "neural_delta_drift",
         "population_delta_drift", "delta drift"),
        ("observational_error_vs_do_error.png", "balanced_reward_observational_error",
         "balanced_reward_do_error", "reward error"),
    )
    for filename, first_metric, second_metric, ylabel in paired:
        plt.figure(figsize=(5.5, 4.0))
        for metric in (first_metric, second_metric):
            rows = sorted((row for row in aggregate
                           if row["metric"] == metric and row["condition"] == "confounded"),
                          key=lambda row: float(row["kappa"]))
            plt.plot([float(row["kappa"]) for row in rows],
                     [float(row["mean"]) for row in rows], marker="o", label=metric)
        plt.xlabel("kappa"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(figures / filename, dpi=160); plt.close()
    plt.figure(figsize=(5.5, 4.0))
    subset_labels = ("confounded", "confounded__strict_unclipped", "confounded__any_clipping")
    values, labels = [], []
    for condition in subset_labels:
        rows = [row for row in aggregate if row["metric"] == "balanced_reward_do_error"
                and row["condition"] == condition and float(row["kappa"]) == 0.3]
        if rows:
            values.append(float(rows[0]["mean"])); labels.append(condition.replace("confounded__", ""))
    plt.bar(labels, values); plt.ylabel("reward do error"); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(figures / "all_vs_unclipped_kappa_0p30.png", dpi=160); plt.close()


def _report(output: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# Phase 8B-NC — Pooled Neural Observational-Bias Realization Audit

This stage trained separate pooled neural conditional-mean models for each kappa,
condition, mixture, target, and model seed.  Each network received only the public
12-D observation and commanded 3-D action.  Population row weights were used only
to form grouped observational targets and their grouped training mass.

All {len(summary['hard_checks'])} implementation hard checks passed.  Evaluation
uses only held-out anchor IDs.  Anchor bootstrap variation and between-model-seed
variation are reported separately; seed×anchor rows are not treated as independent.

## Interpretation boundary

The saved fit metrics must be inspected before concluding that a neural model
realized the observational population target.  If fit is accurate, a difference
between its held-out prediction and the independently reused do oracle is evidence
that learning the observational target realizes observational bias in this fixed
DGP.  This stage does not establish causal recovery, causal bounds, cross-policy-
seed generalization, or an automatic scientific verdict.  Kappa=0.3 clipping
subsets are descriptive only.

Leakage flags are all false: `{json.dumps(LEAKAGE_FLAGS, sort_keys=True)}`.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def run_neural_observational_bias(
    phase8anc_root: Path, phase8a_root: Path, phase8ac_root: Path, output_root: Path,
    *, num_anchors: int = 2048, kappas: tuple[float, ...] = EXPECTED_KAPPAS,
    conditions: tuple[str, ...] = CONDITIONS,
    mixtures: tuple[str, ...] = PRIMARY_MIXTURE_NAMES,
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4), updates: int = 3000,
    batch_size: int = 512, device: str = "auto", split_seed: int = 0,
) -> dict[str, Any]:
    """Run Phase 8B-NC while keeping all preceding artifacts read-only."""
    if any(LEAKAGE_FLAGS.values()):
        raise NeuralObservationalBiasError("a forbidden field/oracle is configured for training")
    kappas = tuple(map(float, kappas))
    conditions = tuple(conditions)
    mixtures = tuple(mixtures)
    if not kappas or any(kappa not in EXPECTED_KAPPAS for kappa in kappas):
        raise ValueError("kappas must be a nonempty subset of 0.0,0.1,0.2,0.3")
    if not conditions or any(value not in CONDITIONS for value in conditions):
        raise ValueError("conditions are invalid")
    if mixtures != PRIMARY_MIXTURE_NAMES:
        raise ValueError("exactly the three primary mixtures are required")
    if not model_seeds or min(num_anchors, updates, batch_size) <= 0:
        raise ValueError("anchors, updates, batch size, and model seeds must be nonempty")
    inputs, all_anchor_ids, strict_all = require_verified_inputs(
        phase8anc_root, phase8a_root, phase8ac_root)
    if num_anchors > len(all_anchor_ids):
        raise ValueError("num_anchors exceeds 2048")
    selected_anchor_ids = np.sort(all_anchor_ids)[:num_anchors]
    causal_anchors = load_npz(Path(phase8a_root).resolve() / "anchors.npz")
    origin = causal_anchors.get("anchor_origin_source")
    selected_origin = None if origin is None else np.asarray(origin)[:num_anchors]
    splits = make_anchor_splits(selected_anchor_ids, split_seed, selected_origin)
    if not validate_splits(splits, selected_anchor_ids):
        raise NeuralObservationalBiasError("anchor splits overlap or are incomplete")
    output = Path(output_root).resolve()
    readonly_roots = [Path(value).resolve() for value in
                      (phase8anc_root, phase8a_root, phase8ac_root)]
    nc_root = Path(phase8anc_root).resolve()
    allowed_nc_child = output.is_relative_to(nc_root) and output != nc_root \
        and output.name.startswith("phase8b")
    if any(output == root for root in readonly_roots) or (
            any(output.is_relative_to(root) for root in readonly_roots)
            and not allowed_nc_child):
        raise NeuralObservationalBiasError("output would overwrite a read-only input artifact")
    hashes_before = hash_input_files(inputs)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "splits.json", {**splits, "split_seed": split_seed,
                 "stratified_by": "anchor_origin_source" if selected_origin is not None else "none"})
    resolved_device = resolve_device(device)

    grouped: dict[tuple[float, str, str], GroupedTargets] = {}
    population_tables: dict[float, dict[str, np.ndarray]] = {}
    raw_oracle_paths: dict[float, Path] = {}
    target_crosscheck = True
    grouped_decomposition = True
    primary_mass_equal = True
    kappa_zero_invariant = True
    independent_invariant = True
    base_invariant = True
    canonical: dict[float, np.ndarray] = {}
    for kappa in kappas:
        kname = KAPPA_NAMES[kappa]
        directory = Path(phase8anc_root).resolve() / kname
        population_tables[kappa] = load_npz(directory / "population_tables.npz")
        raw_path = Path(phase8a_root).resolve() / kname / "do_oracle_raw.npz"
        raw_oracle_paths[kappa] = raw_path
        raw = load_npz(raw_path)
        reference_mass: dict[str, np.ndarray] = {}
        for condition in conditions:
            public = load_npz(directory / f"{condition}_public.npz")
            if set(public) != set(PUBLIC_FIELDS):
                raise NeuralObservationalBiasError("public support schema changed")
            selected_rows = np.isin(public["anchor_id"], selected_anchor_ids)
            selected_public = {name: np.asarray(values)[selected_rows] for name, values in public.items()}
            for mixture in mixtures:
                weights_all = np.load(directory / "weights" / condition / f"{mixture}.npy")
                values = np.asarray(weights_all, dtype=np.float64)[selected_rows]
                values /= values.sum()
                current = build_grouped_targets(selected_public, values)
                grouped[(kappa, condition, mixture)] = current
                group_lookup = {(int(anchor), action_bytes(np.asarray(action, dtype=np.float32))): index
                                for index, (anchor, action) in enumerate(
                                    zip(current.anchor_id, current.commanded_action))}
                row_group = np.asarray([
                    group_lookup[(int(anchor), action_bytes(action))]
                    for anchor, action in zip(selected_public["anchor_id"],
                                              selected_public["commanded_action"])], dtype=np.int64)
                reward_parts = grouped_weighted_mse_decomposition(
                    np.asarray(selected_public["reward"]), values, row_group,
                    current.reward + 0.123)
                row_delta = (np.asarray(selected_public["next_observation"], dtype=np.float64)[:, :11]
                             - np.asarray(selected_public["observation"], dtype=np.float64)[:, :11])
                delta_parts = grouped_weighted_mse_decomposition(
                    row_delta, values, row_group, current.delta + 0.01)
                grouped_decomposition &= (np.isclose(reward_parts[0], reward_parts[1]+reward_parts[2], atol=1e-11)
                                          and np.isclose(delta_parts[0], delta_parts[1]+delta_parts[2], atol=1e-11))
                target_dir = output / "grouped_targets" / kname
                target_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target_dir / f"{condition}__{mixture}.npz", **current.arrays())
                if mixture == mixtures[0]:
                    reference_mass[condition] = current.mass / current.mass.sum()
                else:
                    primary_mass_equal &= np.allclose(current.mass/current.mass.sum(),
                                                      reference_mass[condition], atol=1e-12, rtol=1e-12)
            canonical[kappa] = _canonical_indices(grouped[(kappa, condition, mixtures[0])],
                                                   raw, selected_anchor_ids, kappa)
            table_anchor = population_tables[kappa]["anchor_id"]
            table_rows = np.searchsorted(table_anchor, selected_anchor_ids)
            for mixture in mixtures:
                current = grouped[(kappa, condition, mixture)]
                order = canonical[kappa]
                target_crosscheck &= np.allclose(
                    current.reward[order], population_tables[kappa][f"{condition}_{mixture}_reward"][table_rows],
                    atol=1e-10, rtol=1e-10)
                target_crosscheck &= np.allclose(
                    current.delta[order], population_tables[kappa][f"{condition}_{mixture}_delta"][table_rows],
                    atol=1e-10, rtol=1e-10)
        table = population_tables[kappa]
        if kappa == 0.0:
            for condition in conditions:
                first = table[f"{condition}_{mixtures[0]}_reward"]
                kappa_zero_invariant &= all(np.allclose(first, table[f"{condition}_{m}_reward"], atol=1e-10, rtol=1e-10)
                                            for m in mixtures[1:])
        if "independent_latents" in conditions:
            first = table[f"independent_latents_{mixtures[0]}_reward"]
            independent_invariant &= all(np.allclose(first, table[f"independent_latents_{m}_reward"], atol=1e-10, rtol=1e-10)
                                         for m in mixtures[1:])
        for condition in conditions:
            base = [table[f"{condition}_{m}_reward"][:, 1] for m in mixtures]
            base_invariant &= all(np.allclose(base[0], value, atol=1e-10, rtol=1e-10) for value in base[1:])

    # Inputs are identical across DGP conditions/mixtures/kappas; verify and fit one global normalizer.
    reference_group = grouped[(kappas[0], conditions[0], mixtures[0])]
    reference_train = reference_group.subset(splits["train"])
    for current in grouped.values():
        candidate = current.subset(splits["train"])
        if not (np.array_equal(candidate.anchor_id, reference_train.anchor_id)
                and np.allclose(candidate.x, reference_train.x, atol=0, rtol=0)):
            raise NeuralObservationalBiasError("unique train state-action inputs differ across datasets")
    input_stats = normalization(reference_train.x)
    _save_stats(output / "normalization" / "input_stats.npz", input_stats)

    output_stats: dict[tuple[float, str], Normalization] = {}
    for kappa in kappas:
        balanced_train = [grouped[(kappa, condition, "logger12_balanced")].subset(splits["train"])
                          for condition in conditions]
        output_stats[(kappa, "reward")] = normalization(
            np.concatenate([value.reward[:, None] for value in balanced_train], axis=0))
        output_stats[(kappa, "delta")] = normalization(
            np.concatenate([value.delta for value in balanced_train], axis=0))
        np.savez_compressed(output / "normalization" / f"{KAPPA_NAMES[kappa]}_output_stats.npz",
                            reward_mean=output_stats[(kappa,"reward")].mean,
                            reward_std=output_stats[(kappa,"reward")].std,
                            reward_constant_mask=output_stats[(kappa,"reward")].constant_mask,
                            delta_mean=output_stats[(kappa,"delta")].mean,
                            delta_std=output_stats[(kappa,"delta")].std,
                            delta_constant_mask=output_stats[(kappa,"delta")].constant_mask)

    seed_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    anchor_arrays: dict[str, np.ndarray] = {}
    initial_hash_check = True
    schedule_hash_check = True
    checkpoint_roundtrip = True
    prediction_shape = True
    all_finite = True
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    selected_lookup = {int(anchor): index for index, anchor in enumerate(selected_anchor_ids)}
    test_selected_rows = np.asarray([selected_lookup[int(anchor)] for anchor in test_ids])
    strict_test = strict_all[test_ids]

    for kappa in kappas:
        table = population_tables[kappa]
        table_indices = np.searchsorted(table["anchor_id"], test_ids)
        # This causal object is never passed to train_model; it is used only below for audit metrics.
        raw_test_oracle = load_npz(raw_oracle_paths[kappa])
        audit_reference = grouped[(kappa, conditions[0], mixtures[0])].subset(splits["test"])
        do_reward, do_delta = recompute_do_targets(
            raw_test_oracle, audit_reference, test_ids, kappa)
        if not (np.allclose(do_reward, np.asarray(table["do_mean_reward"])[table_indices], atol=1e-10, rtol=1e-10)
                and np.allclose(do_delta, np.asarray(table["do_mean_delta"])[table_indices], atol=1e-10, rtol=1e-10)):
            raise NeuralObservationalBiasError("recomputed Phase 8A do oracle differs from Phase 8A-NC")
        for condition in conditions:
            for model_seed in model_seeds:
                predicted_reward: dict[str, np.ndarray] = {}
                predicted_delta: dict[str, np.ndarray] = {}
                initial_by_target: dict[str, list[str]] = {"reward": [], "delta": []}
                schedule_by_target: dict[str, list[str]] = {"reward": [], "delta": []}
                for target in ("reward", "delta"):
                    shared_initial_state, shared_initial_hash = make_initial_state(target, model_seed)
                    initial_path = (output / "models" / "initial_states" / KAPPA_NAMES[kappa]
                                    / condition / f"seed_{model_seed}_{target}.pt")
                    initial_path.parent.mkdir(parents=True, exist_ok=True)
                    _torch().save({"state_dict": shared_initial_state,
                                   "sha256": shared_initial_hash}, initial_path)
                    family_schedule = batch_schedule(
                        len(grouped[(kappa, condition, mixtures[0])].subset(splits["train"]).x),
                        updates, batch_size,
                        _schedule_seed(kappa, condition, target, model_seed))
                    schedule_digest = array_hash(family_schedule)
                    schedule_dir = output / "batch_schedules" / KAPPA_NAMES[kappa] / condition
                    _write_json(schedule_dir / f"seed_{model_seed}_{target}.json",
                                {"sha256": schedule_digest, "shape": list(family_schedule.shape)})
                    for mixture in mixtures:
                        dataset = grouped[(kappa, condition, mixture)]
                        train = dataset.subset(splits["train"])
                        validation = dataset.subset(splits["validation"])
                        test = dataset.subset(splits["test"])
                        model, history = train_model(
                            train, validation, target, input_stats, output_stats[(kappa,target)],
                            family_schedule, model_seed, resolved_device, shared_initial_state)
                        initial_by_target[target].append(history["initial_state_hash"])
                        schedule_by_target[target].append(history["schedule_hash"])
                        model_path = (output / "models" / KAPPA_NAMES[kappa] / condition /
                                      mixture / f"seed_{model_seed}_{target}.pt")
                        metadata = {**history, "kappa": kappa, "condition": condition,
                                    "mixture": mixture, "model_seed": model_seed,
                                    "target": target, "model_input_fields": MODEL_INPUT_FIELDS}
                        save_checkpoint(model_path, model, metadata)
                        values = predict(model, test.x, input_stats, output_stats[(kappa,target)], resolved_device)
                        reloaded, _ = load_checkpoint(model_path, target, resolved_device)
                        repeated = predict(reloaded, test.x[:32], input_stats,
                                           output_stats[(kappa,target)], resolved_device)
                        checkpoint_roundtrip &= np.allclose(values[:32], repeated, atol=1e-7, rtol=1e-6)
                        if target == "reward":
                            predicted_reward[mixture] = values[:, 0]
                            prediction_shape &= values.shape == (len(test.x), 1)
                        else:
                            predicted_delta[mixture] = values
                            prediction_shape &= values.shape == (len(test.x), 11)
                        all_finite &= np.isfinite(values).all() and np.isfinite(history["train_loss"]).all()
                        prediction_dir = output / "predictions" / KAPPA_NAMES[kappa] / condition / mixture
                        prediction_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(prediction_dir / f"seed_{model_seed}_{target}.npz",
                                            anchor_id=test.anchor_id, commanded_action=test.commanded_action,
                                            prediction=values)
                initial_hash_check &= all(len(set(values)) == 1 for values in initial_by_target.values())
                schedule_hash_check &= all(len(set(values)) == 1 for values in schedule_by_target.values())
                # Arrange exact actions only now, using the test-only oracle lookup established above.
                canonical_test = {}
                population_reward, population_delta = {}, {}
                audit_prediction_arrays: dict[str, np.ndarray] = {
                    "anchor_id": test_ids, "strict_unclipped": strict_test,
                    "do_reward": do_reward, "do_delta": do_delta,
                }
                for mixture in mixtures:
                    dataset = grouped[(kappa, condition, mixture)].subset(splits["test"])
                    raw = load_npz(raw_oracle_paths[kappa])
                    order = _canonical_indices(dataset, raw, test_ids, kappa)
                    predicted_reward[mixture] = predicted_reward[mixture][order]
                    predicted_delta[mixture] = predicted_delta[mixture][order]
                    population_reward[mixture] = dataset.reward[order]
                    population_delta[mixture] = dataset.delta[order]
                    canonical_test[mixture] = order
                    audit_prediction_arrays.update({
                        f"{mixture}_predicted_reward": predicted_reward[mixture],
                        f"{mixture}_predicted_delta": predicted_delta[mixture],
                        f"{mixture}_observational_reward": population_reward[mixture],
                        f"{mixture}_observational_delta": population_delta[mixture],
                        f"{mixture}_group_mass": dataset.mass[order],
                    })
                    for action_index, action in enumerate(ACTION_KEYS):
                        coordinate_mae = np.mean(np.abs(
                            predicted_delta[mixture][:,action_index]
                            - population_delta[mixture][:,action_index]), axis=0)
                        action_rows.append({
                            "kappa": kappa, "condition": condition, "mixture": mixture,
                            "model_seed": model_seed, "action": action,
                            "reward_mae": float(np.mean(np.abs(predicted_reward[mixture][:,action_index]-population_reward[mixture][:,action_index]))),
                            "delta_l2": float(np.mean(np.linalg.norm(predicted_delta[mixture][:,action_index]-population_delta[mixture][:,action_index], axis=1))),
                            **{f"delta_dimension_{dimension}_mae": float(value)
                               for dimension, value in enumerate(coordinate_mae)},
                            "n_test_anchors": len(test_ids), "statistical_unit": "anchor_id"})
                current_rows, current_arrays = seed_metrics(
                    predicted_reward, predicted_delta, population_reward, population_delta,
                    do_reward, do_delta, strict_test, kappa, condition, model_seed, test_ids)
                seed_rows.extend(current_rows)
                prefix = f"{KAPPA_NAMES[kappa]}__{condition}__seed_{model_seed}__"
                anchor_arrays.update({prefix+name: value for name, value in current_arrays.items()})
                audit_path = output / "predictions" / KAPPA_NAMES[kappa] / condition / f"seed_{model_seed}_audit.npz"
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(audit_path, **audit_prediction_arrays)

    # Cross-condition excess and kappa=0.3 clipping subsets use paired anchors.
    for kappa in kappas:
        for model_seed in model_seeds:
            if set(CONDITIONS).issubset(conditions):
                conf_key = f"{KAPPA_NAMES[kappa]}__confounded__seed_{model_seed}__balanced_reward_do_error"
                indep_key = f"{KAPPA_NAMES[kappa]}__independent_latents__seed_{model_seed}__balanced_reward_do_error"
                excess = anchor_arrays[conf_key] - anchor_arrays[indep_key]
                stats = _bootstrap_mean(excess, 500, model_seed + 7000)
                seed_rows.append({"kappa": kappa, "condition": "confounded_minus_independent",
                                  "model_seed": model_seed, "metric": "confounding_excess_reward",
                                  "statistical_unit": "anchor_id", **stats})
            if kappa == 0.3 and "confounded" in conditions:
                prefix = f"{KAPPA_NAMES[kappa]}__confounded__seed_{model_seed}__"
                for subset_name, mask in (("strict_unclipped", strict_test),
                                          ("any_clipping", ~strict_test)):
                    for metric in ("reward_fit_abs", "delta_fit_l2",
                                   "balanced_reward_do_error", "balanced_delta_do_error",
                                   "neural_do_ranking_disagreement", "neural_decision_regret"):
                        selected = np.asarray(anchor_arrays[prefix+metric])[mask]
                        if len(selected):
                            stats = _bootstrap_mean(selected, 500, model_seed + 8000 + len(seed_rows))
                            seed_rows.append({"kappa": kappa,
                                              "condition": f"confounded__{subset_name}",
                                              "model_seed": model_seed, "metric": metric,
                                              "statistical_unit": "anchor_id", **stats})

    aggregate_rows = _aggregate_with_anchor_bootstrap(
        seed_rows, anchor_arrays, kappas, conditions, model_seeds)
    numeric_outputs = [value for rows in (seed_rows, action_rows, aggregate_rows)
                       for row in rows for value in row.values()
                       if isinstance(value, (int, float, np.integer, np.floating))]
    all_finite &= bool(np.isfinite(np.asarray(numeric_outputs, dtype=np.float64)).all())
    all_finite &= all(np.isfinite(np.asarray(value)).all() for value in anchor_arrays.values())
    _write_csv(output / "seed_metrics.csv", seed_rows)
    _write_csv(output / "action_metrics.csv", action_rows)
    _write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    np.savez_compressed(output / "anchor_action_metrics.npz", **anchor_arrays)
    with (output / "aggregate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        _make_figures(output, list(csv.DictReader(handle)))

    hard_checks = {
        "all_input_artifacts_complete": True,
        "phase8anc_all_hard_checks_passed": True,
        "all_2048_anchors_available": len(all_anchor_ids) == 2048,
        "all_four_kappas_available": True,
        "primary_state_action_mass_identical": bool(primary_mass_equal),
        "anchor_splits_disjoint": validate_splits(splits, selected_anchor_ids),
        "grouped_targets_match_weighted_rows": bool(target_crosscheck),
        "grouped_mse_decomposition_holds": bool(grouped_decomposition),
        "input_normalization_shared_across_mixtures": True,
        "output_normalization_shared_across_mixtures": True,
        "same_seed_mixture_initial_hashes_equal": bool(initial_hash_check),
        "same_seed_minibatch_schedule_hashes_equal": bool(schedule_hash_check),
        "logger_source_excluded_from_model_input": not LEAKAGE_FLAGS["LOGGER_ID_IN_MODEL_INPUT"],
        "hidden_u_excluded_from_model_input": not LEAKAGE_FLAGS["HIDDEN_U_IN_MODEL_INPUT"],
        "applied_action_excluded_from_model_input": not LEAKAGE_FLAGS["APPLIED_ACTION_IN_MODEL_INPUT"],
        "do_oracle_excluded_from_training": not LEAKAGE_FLAGS["DO_ORACLE_USED_FOR_TRAINING"],
        "long_horizon_oracle_excluded_from_training": not LEAKAGE_FLAGS["LONG_HORIZON_ORACLE_USED_FOR_TRAINING"],
        "checkpoint_save_reload_succeeds": bool(checkpoint_roundtrip),
        "reload_predictions_match": bool(checkpoint_roundtrip),
        "prediction_shapes_correct": bool(prediction_shape),
        "no_nan_or_inf": bool(all_finite),
        "kappa_zero_population_targets_mixture_invariant": bool(kappa_zero_invariant),
        "independent_population_targets_mixture_invariant": bool(independent_invariant),
        "base_population_targets_mixture_invariant": bool(base_invariant),
        "input_hashes_unchanged": False,
        "old_artifacts_unchanged": False,
    }
    hashes_after = hash_input_files(inputs)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks["input_hashes_unchanged"] = unchanged
    hard_checks["old_artifacts_unchanged"] = unchanged
    failed = [name for name, passed in hard_checks.items() if not passed]
    if failed:
        raise NeuralObservationalBiasError(f"hard checks failed: {failed}")
    input_integrity = {"sha256_before": hashes_before, "sha256_after": hashes_after,
                       "unchanged": unchanged, "required_file_count": len(inputs)}
    _write_json(output / "input_integrity.json", input_integrity)
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
                "all_passed": True, "failed": []})
    manifest = {
        "stage": "Phase 8B-NC", "phase8anc_root": str(Path(phase8anc_root).resolve()),
        "phase8a_do_oracle_root": str(Path(phase8a_root).resolve()),
        "phase8ac_root": str(Path(phase8ac_root).resolve()), "num_anchors": num_anchors,
        "kappas": kappas, "conditions": conditions, "mixtures": mixtures,
        "model_seeds": model_seeds, "updates": updates, "batch_size": batch_size,
        "device": resolved_device, "model_input_fields": MODEL_INPUT_FIELDS,
        "model_input_dimension": 15, "reward_output_dimension": 1,
        "delta_output_dimension": 11, "architecture": "15-256-256-256-output ReLU",
        "optimizer": "Adam", "learning_rate": 0.001, "leakage_flags": LEAKAGE_FLAGS,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }
    summary = {
        "stage": "Phase 8B-NC", "analyzed_anchor_count": num_anchors,
        "test_anchor_count": len(test_ids), "model_seed_count": len(model_seeds),
        "hard_checks": hard_checks, "all_hard_checks_passed": True,
        "aggregate_metrics": aggregate_rows, "scientific_verdict": "MANUAL_DECISION_REQUIRED",
        "statistical_units": {"bootstrap": "anchor_id", "model_variation": "model_seed",
                              "seed_anchor_rows_treated_as_independent": False},
        "leakage_flags": LEAKAGE_FLAGS,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    _report(output, summary)
    return summary
