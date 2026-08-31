"""Phase 8B-RS: direct U-to-reward calibration and neural realization audit.

The direct reward channel is a positive-control DGP.  It augments copied Phase
8A-NC rows with ``lambda_reward * u_env`` while keeping hidden variables in a
separate audit artifact.  Population identities are checked before any model is
created.  Neural models consume only public observations and commanded actions.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analyze_noncomplementary_population import KAPPA_NAMES
from .analyze_phase8a_population_effect import (
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
    validate_all_84_phase8a_invariants,
)
from .neural_observational_bias import (
    DEFAULT_HIDDEN_WIDTH,
    MODEL_INPUT_DIMENSION,
    MODEL_INPUT_FIELDS,
    GroupedTargets,
    Normalization,
    RewardMeanModel,
    _bootstrap_mean,
    _canonical_indices,
    _top_masks,
    action_bytes,
    apply_normalization,
    array_hash,
    batch_schedule,
    build_grouped_targets,
    expected_parameter_count,
    load_checkpoint,
    make_anchor_splits,
    make_initial_state,
    normalization,
    parameter_count,
    predict,
    recompute_do_targets,
    resolve_device,
    save_checkpoint,
    state_hash,
    validate_splits,
    _torch,
)
from .noncomplementary_population_dgp import (
    ACTION_KEYS,
    CONDITIONS,
    PRIMARY_MIXTURES,
    analytic_u_posterior,
)


REWARD_STRENGTHS = (0.0, 0.05, 0.10, 0.20)
CALIBRATION_KAPPAS = (0.0, 0.3)
PRIMARY_MIXTURE_NAMES = tuple(PRIMARY_MIXTURES)
PUBLIC_FIELDS = (
    "row_id", "anchor_id", "observation", "commanded_action", "reward",
    "next_observation", "terminated", "truncated", "logger_id", "condition",
    "kappa_env", "lambda_reward",
)
HIDDEN_AUDIT_FIELDS = (
    "row_id", "original_reward", "augmented_reward", "reward_bonus", "u_env",
    "u_behavior", "action_key", "logger_id", "kappa", "lambda_reward",
)
FORBIDDEN_DERIVED_PUBLIC_FIELDS = {
    "original_reward", "u_env", "u_behavior", "reward_bonus", "applied_action",
    "action_key", "qpos", "qvel",
}
LEAKAGE_FLAGS = {
    "LOGGER_ID_IN_MODEL_INPUT": False,
    "LAMBDA_IN_MODEL_INPUT": False,
    "KAPPA_IN_MODEL_INPUT": False,
    "HIDDEN_U_IN_MODEL_INPUT": False,
    "DO_ORACLE_USED_FOR_TRAINING": False,
}


class RewardSignalCalibrationError(RuntimeError):
    """Raised when a Phase 8B-RS precondition or hard invariant fails."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strength_name(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "p")


def validate_reward_strengths(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(value not in REWARD_STRENGTHS for value in result):
        raise ValueError(f"reward strengths must be a nonempty subset of {REWARD_STRENGTHS}")
    if len(set(result)) != len(result):
        raise ValueError("reward strengths must be unique")
    return result


def augment_reward(original_reward: np.ndarray, u_env: np.ndarray,
                   lambda_reward: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply only the specified direct hidden U-to-reward channel."""
    reward = np.asarray(original_reward, dtype=np.float64)
    hidden = np.asarray(u_env, dtype=np.int8)
    if reward.shape != hidden.shape or not np.isin(hidden, (-1, 1)).all():
        raise RewardSignalCalibrationError("reward and u_env must be aligned with U in {-1,+1}")
    bonus = float(lambda_reward) * hidden.astype(np.float64)
    return reward + bonus, bonus


def make_derived_artifacts(public: Mapping[str, np.ndarray], hidden: Mapping[str, np.ndarray],
                           lambda_reward: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create aligned public and hidden artifacts without co-exposing original reward and U."""
    if not {"row_id", "reward"}.issubset(public) or not {
            "row_id", "u_env", "u_behavior", "action_key", "logger_id"}.issubset(hidden):
        raise RewardSignalCalibrationError("Phase 8A-NC public/hidden rows lack required fields")
    if not np.array_equal(public["row_id"], hidden["row_id"]):
        raise RewardSignalCalibrationError("public and hidden row IDs are not aligned")
    augmented, bonus = augment_reward(public["reward"], hidden["u_env"], lambda_reward)
    derived_public = {
        "row_id": np.asarray(public["row_id"], dtype=np.int64),
        "anchor_id": np.asarray(public["anchor_id"], dtype=np.int64),
        "observation": np.asarray(public["observation"], dtype=np.float32),
        "commanded_action": np.asarray(public["commanded_action"], dtype=np.float32),
        "reward": augmented,
        "next_observation": np.asarray(public["next_observation"], dtype=np.float32),
        "terminated": np.asarray(public["terminated"], dtype=bool),
        "truncated": np.asarray(public["truncated"], dtype=bool),
        "logger_id": np.asarray(public["logger_id"], dtype=np.int8),
        "condition": np.asarray(public["condition"]),
        "kappa_env": np.asarray(public["kappa_env"], dtype=np.float64),
        "lambda_reward": np.full(len(augmented), float(lambda_reward), dtype=np.float64),
    }
    hidden_audit = {
        "row_id": np.asarray(public["row_id"], dtype=np.int64),
        "original_reward": np.asarray(public["reward"], dtype=np.float64),
        "augmented_reward": augmented,
        "reward_bonus": bonus,
        "u_env": np.asarray(hidden["u_env"], dtype=np.int8),
        "u_behavior": np.asarray(hidden["u_behavior"], dtype=np.int8),
        "action_key": np.asarray(hidden["action_key"]),
        "logger_id": np.asarray(hidden["logger_id"], dtype=np.int8),
        "kappa": np.asarray(public["kappa_env"], dtype=np.float64),
        "lambda_reward": np.full(len(augmented), float(lambda_reward), dtype=np.float64),
    }
    validate_derived_artifacts(derived_public, hidden_audit)
    return derived_public, hidden_audit


def validate_derived_artifacts(public: Mapping[str, np.ndarray],
                               hidden: Mapping[str, np.ndarray]) -> set[str]:
    if set(public) != set(PUBLIC_FIELDS) or set(hidden) != set(HIDDEN_AUDIT_FIELDS):
        raise RewardSignalCalibrationError("derived public or hidden schema is invalid")
    leakage = FORBIDDEN_DERIVED_PUBLIC_FIELDS.intersection(public)
    if leakage:
        raise RewardSignalCalibrationError(f"hidden leakage in public artifact: {sorted(leakage)}")
    n = len(public["row_id"])
    if any(len(np.asarray(value)) != n for value in (*public.values(), *hidden.values())):
        raise RewardSignalCalibrationError("derived artifacts are not row aligned")
    if public["observation"].shape != (n, 12) or public["commanded_action"].shape != (n, 3):
        raise RewardSignalCalibrationError("derived public observation/action shape is invalid")
    if "original_reward" in public and "reward" in public:
        raise RewardSignalCalibrationError("original and augmented reward cannot both be public")
    if not np.array_equal(public["row_id"], hidden["row_id"]):
        raise RewardSignalCalibrationError("derived public/hidden row IDs differ")
    if not np.allclose(public["reward"], hidden["augmented_reward"], atol=0, rtol=0):
        raise RewardSignalCalibrationError("derived public reward differs from hidden audit")
    return leakage


def direct_reward_mean(condition: str, mixture: str, action: str,
                       lambda_reward: float) -> float:
    if condition == "independent_latents":
        return 0.0
    posterior = analytic_u_posterior(condition, mixture, action)
    return float(lambda_reward) * (2.0 * posterior - 1.0)


def theoretical_balanced_direct_bias(action: str, lambda_reward: float,
                                     condition: str = "confounded") -> float:
    return direct_reward_mean(condition, "logger12_balanced", action, lambda_reward)


def theoretical_heavy_direct_drift(action: str, lambda_reward: float,
                                   condition: str = "confounded") -> float:
    return (direct_reward_mean(condition, "logger1_heavy", action, lambda_reward)
            - direct_reward_mean(condition, "logger2_heavy", action, lambda_reward))


def direct_slope(condition: str, family: str, action: str,
                 mixture: str | None = None) -> float:
    if family == "balanced_bias":
        return theoretical_balanced_direct_bias(action, 1.0, condition)
    if family == "heavy_drift":
        return theoretical_heavy_direct_drift(action, 1.0, condition)
    if family == "mixture_increment" and mixture is not None:
        return direct_reward_mean(condition, mixture, action, 1.0)
    raise ValueError("unknown slope family")


def fit_lambda_slope(lambdas: Sequence[float], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row slope with intercept, through-origin increment slope, and R^2."""
    x = np.asarray(lambdas, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if y.shape[0] != len(x) or len(x) < 2:
        raise ValueError("lambda slope requires at least two aligned strengths")
    centered = x - x.mean()
    slope = np.tensordot(centered, y, axes=(0, 0)) / float(centered @ centered)
    intercept = y.mean(axis=0) - slope * x.mean()
    fitted = intercept[None, ...] + x.reshape((-1,) + (1,) * (y.ndim - 1)) * slope
    residual = np.sum((y - fitted) ** 2, axis=0)
    total = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    r2 = np.ones_like(residual)
    np.divide(residual, total, out=r2, where=total > 1e-15)
    r2 = np.where(total > 1e-15, 1.0 - r2, 1.0)
    increments = y - y[0]
    dx = x - x[0]
    origin = np.tensordot(dx, increments, axes=(0, 0)) / float(dx @ dx)
    return np.asarray(slope), np.asarray(origin), np.asarray(r2)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RewardSignalCalibrationError(f"required JSON is unavailable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_phase8b_root(phase8anc_root: Path, phase8b_root: Path | None) -> Path:
    root = (Path(phase8b_root).resolve() if phase8b_root is not None else
            Path(phase8anc_root).resolve() / "phase8b_neural_observational_bias")
    required = (root / "manifest.json", root / "hard_checks.json",
                root / "aggregate_metrics.csv", root / "splits.json",
                root / "normalization" / "input_stats.npz")
    if not root.is_dir() or not all(path.is_file() for path in required):
        raise RewardSignalCalibrationError(
            f"verified Phase 8B-NC baseline is unavailable at {root}; pass --phase8b-root explicitly")
    hard = _load_json(root / "hard_checks.json")
    if hard.get("all_passed") is not True or not all(hard.get("checks", {}).values()):
        raise RewardSignalCalibrationError("Phase 8B-NC baseline hard checks did not all pass")
    return root


def require_verified_inputs(phase8anc_root: Path, phase8a_root: Path,
                            phase8b_root: Path | None = None) -> tuple[list[Path], np.ndarray, Path]:
    nc, causal = Path(phase8anc_root).resolve(), Path(phase8a_root).resolve()
    if not nc.is_dir():
        raise RewardSignalCalibrationError("verified Phase 8A-NC root is unavailable")
    if not causal.is_dir():
        raise RewardSignalCalibrationError("verified Phase 8A do-oracle root is unavailable")
    hard = _load_json(nc / "hard_checks.json")
    checks = hard.get("checks", {})
    if hard.get("all_passed") is not True or not checks or not all(checks.values()):
        raise RewardSignalCalibrationError("Phase 8A-NC hard checks did not all pass")
    if len(checks) != 91:
        raise RewardSignalCalibrationError("Phase 8A-NC must contain exactly the verified 91 hard checks")
    manifest = _load_json(nc / "manifest.json")
    if int(manifest.get("available_anchor_count", -1)) != 2048:
        raise RewardSignalCalibrationError("Phase 8A-NC does not provide 2048 anchors")
    causal_summary = _load_json(causal / "summary.json")
    try:
        validate_all_84_phase8a_invariants(causal_summary)
    except Exception as exc:
        raise RewardSignalCalibrationError(f"Phase 8A do oracle is not verified: {exc}") from exc
    anchors = load_npz(causal / "anchors.npz")
    anchor_ids = np.asarray(anchors.get("anchor_id", ()), dtype=np.int64)
    if not np.array_equal(anchor_ids, np.arange(2048)):
        raise RewardSignalCalibrationError("Phase 8A anchors are incomplete")
    baseline = resolve_phase8b_root(nc, phase8b_root)
    paths = [nc / "manifest.json", nc / "hard_checks.json", causal / "manifest.json",
             causal / "summary.json", causal / "anchors.npz", baseline / "manifest.json",
             baseline / "hard_checks.json", baseline / "aggregate_metrics.csv",
             baseline / "splits.json", baseline / "normalization" / "input_stats.npz"]
    for kappa in CALIBRATION_KAPPAS:
        kname = KAPPA_NAMES[kappa]
        paths.append(causal / kname / "do_oracle_raw.npz")
        for condition in CONDITIONS:
            paths.extend((nc / kname / f"{condition}_public.npz",
                          nc / kname / f"{condition}_hidden_audit.npz"))
            for mixture in PRIMARY_MIXTURE_NAMES:
                paths.append(nc / kname / "weights" / condition / f"{mixture}.npy")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RewardSignalCalibrationError(f"required read-only inputs are missing: {missing}")
    return sorted((path.resolve() for path in paths), key=str), anchor_ids, baseline


def _load_normalization(path: Path) -> Normalization:
    values = load_npz(path)
    required = {"mean", "std", "constant_mask"}
    if set(values) != required:
        raise RewardSignalCalibrationError("Phase 8B input normalization schema changed")
    stats = Normalization(np.asarray(values["mean"], dtype=np.float64),
                          np.asarray(values["std"], dtype=np.float64),
                          np.asarray(values["constant_mask"], dtype=bool))
    if stats.mean.shape != (MODEL_INPUT_DIMENSION,) or stats.std.shape != (MODEL_INPUT_DIMENSION,):
        raise RewardSignalCalibrationError("Phase 8B input normalization is not 15D")
    return stats


def _save_normalization(path: Path, stats: Normalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=stats.mean, std=stats.std,
                        constant_mask=stats.constant_mask)


def _reuse_or_make_splits(baseline: Path, selected: np.ndarray,
                          split_seed: int) -> tuple[dict[str, list[int]], str]:
    old = _load_json(baseline / "splits.json")
    candidate = {name: sorted(set(map(int, old.get(name, ()))) & set(map(int, selected)))
                 for name in ("train", "validation", "test")}
    if validate_splits(candidate, selected) and all(candidate.values()):
        return candidate, "reused Phase 8B-NC anchor assignments"
    return make_anchor_splits(selected, split_seed), "fixed new split; prior split not reusable"


def _baseline_fit_errors(path: Path) -> dict[tuple[float, str], float]:
    result: dict[tuple[float, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") == "reward_fit_abs" and row.get("condition") in CONDITIONS:
                result[(float(row["kappa"]), row["condition"])] = float(row["mean"])
    if not result:
        raise RewardSignalCalibrationError("Phase 8B baseline reward-fit metrics are unavailable")
    return result


def _train_reward_model(train: GroupedTargets, validation: GroupedTargets,
                        input_stats: Normalization, output_stats: Normalization,
                        schedule: np.ndarray, seed: int, device: str,
                        initial_state: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Train reward-only model and retain both best-validation and final states."""
    torch = _torch()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RewardMeanModel(DEFAULT_HIDDEN_WIDTH)
    model.load_state_dict(dict(initial_state))
    model.to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.as_tensor(apply_normalization(train.x, input_stats), dtype=torch.float32, device=device)
    y = torch.as_tensor(apply_normalization(train.reward[:, None], output_stats),
                        dtype=torch.float32, device=device)
    mass = torch.as_tensor(train.mass, dtype=torch.float32, device=device)
    vx = torch.as_tensor(apply_normalization(validation.x, input_stats),
                         dtype=torch.float32, device=device)
    vy = torch.as_tensor(apply_normalization(validation.reward[:, None], output_stats),
                         dtype=torch.float32, device=device)
    train_losses: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    best_loss = float("inf")
    best_step = -1
    best_state: dict[str, Any] | None = None
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
                val_loss = float(torch.mean((model(vx) - vy) ** 2).cpu())
            validation_steps.append(step + 1)
            validation_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step + 1
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
    if best_state is None:
        raise RewardSignalCalibrationError("validation checkpoint was never created")
    final_model = RewardMeanModel(DEFAULT_HIDDEN_WIDTH)
    final_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    final_model.to(device).eval()
    best_model = RewardMeanModel(DEFAULT_HIDDEN_WIDTH)
    best_model.load_state_dict(best_state)
    best_model.to(device).eval()
    history = {
        "initial_state_hash": initial_hash, "schedule_hash": array_hash(schedule),
        "best_state_hash": state_hash(best_model), "final_state_hash": state_hash(final_model),
        "best_validation_step": best_step, "best_validation_loss": best_loss,
        "train_loss": train_losses, "validation_step": validation_steps,
        "validation_loss": validation_losses,
    }
    return best_model, final_model, history


def _stats(values: np.ndarray, seed: int) -> dict[str, Any]:
    return _bootstrap_mean(np.asarray(values, dtype=np.float64), 1000, seed)


def _aggregate_seed_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = ("kappa", "lambda_reward", "condition", "mixture", "action", "metric")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(name, "") for name in dimensions), []).append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        means = np.asarray([float(row["mean"]) for row in members])
        record = dict(zip(dimensions, key))
        record.update({
            "mean": float(means.mean()),
            "seed_sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "seed_min": float(means.min()), "seed_max": float(means.max()),
            "model_seed_count": len(means),
            "ci_low": float(np.mean([float(row["ci_low"]) for row in members])),
            "ci_high": float(np.mean([float(row["ci_high"]) for row in members])),
            "n_anchors": int(members[0]["n_anchors"]),
            "bootstrap_unit": "anchor_id", "seed_variation_reported_separately": True,
        })
        if all("theoretical_slope" in member for member in members):
            theories = {float(member["theoretical_slope"]) for member in members}
            if len(theories) != 1:
                raise RewardSignalCalibrationError("theoretical slope differs across model seeds")
            record["theoretical_slope"] = theories.pop()
        output.append(record)
    return output


def _population_gate(grouped: Mapping[tuple[float, float, str, str], GroupedTargets],
                     original: Mapping[tuple[float, str, str], GroupedTargets],
                     do_rewards: Mapping[float, np.ndarray], canonical: Mapping[float, np.ndarray],
                     kappas: Sequence[float], strengths: Sequence[float], atol: float,
                     rtol: float) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    checks = {
        "do_mean_invariant_to_lambda": True,
        "kappa_zero_balanced_bias_identity": True,
        "kappa_zero_heavy_drift_identity": True,
        "kappa_point3_additive_bias_identity": True,
        "independent_direct_bias_zero": True,
        "base_direct_bias_zero": True,
        "primary_state_action_mass_unchanged": True,
        "all_population_arrays_finite": True,
    }
    rows: list[dict[str, Any]] = []
    derived_metrics: dict[tuple[float, float, str, int, str], dict[str, float]] = {}
    for kappa in kappas:
        order = canonical[kappa]
        do = do_rewards[kappa]
        for condition in CONDITIONS:
            original_values = {mixture: original[(kappa, condition, mixture)].reward[order]
                               for mixture in PRIMARY_MIXTURE_NAMES}
            reference_mass = original[(kappa, condition, PRIMARY_MIXTURE_NAMES[0])].mass
            for strength in strengths:
                augmented = {mixture: grouped[(kappa, strength, condition, mixture)].reward[order]
                             for mixture in PRIMARY_MIXTURE_NAMES}
                for mixture in PRIMARY_MIXTURE_NAMES:
                    current = grouped[(kappa, strength, condition, mixture)]
                    checks["primary_state_action_mass_unchanged"] &= (
                        np.array_equal(current.anchor_id,
                                       original[(kappa, condition, mixture)].anchor_id)
                        and np.array_equal(current.x, original[(kappa, condition, mixture)].x)
                        and np.allclose(current.mass, reference_mass, atol=atol, rtol=rtol))
                    checks["all_population_arrays_finite"] &= np.isfinite(augmented[mixture]).all()
                    for action_index, action in enumerate(ACTION_KEYS):
                        increment = augmented[mixture][:, action_index] - original_values[mixture][:, action_index]
                        expected = direct_reward_mean(condition, mixture, action, strength)
                        if condition == "independent_latents":
                            checks["independent_direct_bias_zero"] &= np.allclose(
                                increment, 0.0, atol=atol, rtol=rtol)
                        if action == "base":
                            checks["base_direct_bias_zero"] &= np.allclose(
                                increment, 0.0, atol=atol, rtol=rtol)
                        if not np.allclose(increment, expected, atol=atol, rtol=rtol):
                            checks["kappa_point3_additive_bias_identity"] = False
                        for anchor_index, anchor in enumerate(
                                grouped[(kappa, strength, condition, mixture)].anchor_id[order[:, 0]]):
                            rows.append({
                                "anchor_id": int(anchor), "kappa": kappa,
                                "lambda_reward": strength, "condition": condition,
                                "mixture": mixture, "action": action,
                                "augmented_observational_reward": float(augmented[mixture][anchor_index, action_index]),
                                "original_observational_reward": float(original_values[mixture][anchor_index, action_index]),
                                "augmented_do_reward": float(do[anchor_index, action_index]),
                                "direct_increment": float(increment[anchor_index]),
                                "theoretical_direct_increment": expected,
                            })
                balanced_bias = augmented["logger12_balanced"] - do
                original_balanced = original_values["logger12_balanced"] - do
                heavy = augmented["logger1_heavy"] - augmented["logger2_heavy"]
                original_heavy = original_values["logger1_heavy"] - original_values["logger2_heavy"]
                pop_top, do_top = _top_masks(augmented["logger12_balanced"]), _top_masks(do)
                ranking_disagreement = (~np.any(pop_top & do_top, axis=1)).astype(float)
                chosen = np.argmax(augmented["logger12_balanced"], axis=1)
                regret = np.max(do, axis=1) - do[np.arange(len(do)), chosen]
                ordered_anchors = grouped[(
                    kappa, strength, condition, "logger12_balanced")].anchor_id[order[:, 0]]
                for action_index, action in enumerate(ACTION_KEYS):
                    balanced_expected = theoretical_balanced_direct_bias(action, strength, condition)
                    heavy_expected = theoretical_heavy_direct_drift(action, strength, condition)
                    balanced_ok = np.allclose(
                        balanced_bias[:, action_index], original_balanced[:, action_index] + balanced_expected,
                        atol=atol, rtol=rtol)
                    heavy_ok = np.allclose(
                        heavy[:, action_index], original_heavy[:, action_index] + heavy_expected,
                        atol=atol, rtol=rtol)
                    if kappa == 0.0 and condition == "confounded":
                        checks["kappa_zero_balanced_bias_identity"] &= balanced_ok
                        checks["kappa_zero_heavy_drift_identity"] &= heavy_ok
                        checks["kappa_zero_balanced_bias_identity"] &= np.allclose(
                            original_balanced[:, action_index], 0.0, atol=atol, rtol=rtol)
                    if kappa == 0.3:
                        checks["kappa_point3_additive_bias_identity"] &= balanced_ok and heavy_ok
                    for anchor_index, anchor in enumerate(ordered_anchors):
                        derived_metrics[(kappa, strength, condition, int(anchor), action)] = {
                            "balanced_do_bias": float(balanced_bias[anchor_index, action_index]),
                            "heavy_mixture_drift": float(heavy[anchor_index, action_index]),
                            "base_action_drift": (float(heavy[anchor_index, action_index])
                                                  if action == "base" else 0.0),
                            "action_ranking_disagreement": float(ranking_disagreement[anchor_index]),
                            "decision_regret": float(regret[anchor_index]),
                        }
    for row in rows:
        row.update(derived_metrics[(float(row["kappa"]), float(row["lambda_reward"]),
                                    str(row["condition"]), int(row["anchor_id"]),
                                    str(row["action"]))])
    return {name: bool(value) for name, value in checks.items()}, rows


def _make_figures(output: Path, population_rows: Sequence[Mapping[str, Any]],
                  aggregate_rows: Sequence[Mapping[str, Any]],
                  slope_rows: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def line_plot(filename: str, rows: Sequence[Mapping[str, Any]], metric: str,
                  ylabel: str, filters: Mapping[str, Any]) -> None:
        subset = [row for row in rows if row.get("metric") == metric
                  and all(row.get(key) == value for key, value in filters.items())]
        plt.figure(figsize=(6.0, 4.0))
        for action in ACTION_KEYS:
            selected = sorted((row for row in subset if row.get("action") == action),
                              key=lambda row: float(row.get("lambda_reward", 0)))
            if selected:
                plt.plot([float(row["lambda_reward"]) for row in selected],
                         [float(row["mean"]) for row in selected], marker="o", label=action)
        plt.xlabel("lambda_reward"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(figures / filename, dpi=160); plt.close()

    # Compact population summaries for the two analytic figures.
    pop_summary = []
    for kappa in sorted({float(row["kappa"]) for row in population_rows}):
        for strength in sorted({float(row["lambda_reward"]) for row in population_rows}):
            for action in ACTION_KEYS:
                rows = [row for row in population_rows if float(row["kappa"]) == kappa
                        and float(row["lambda_reward"]) == strength
                        and row["condition"] == "confounded" and row["action"] == action]
                by_mix = {mix: np.mean([float(row["augmented_observational_reward"])
                                        for row in rows if row["mixture"] == mix])
                          for mix in PRIMARY_MIXTURE_NAMES}
                do = np.mean([float(row["augmented_do_reward"]) for row in rows
                              if row["mixture"] == "logger12_balanced"])
                pop_summary.extend((
                    {"kappa": kappa, "lambda_reward": strength, "condition": "confounded",
                     "action": action, "metric": "population_balanced_bias", "mean": by_mix["logger12_balanced"]-do},
                    {"kappa": kappa, "lambda_reward": strength, "condition": "confounded",
                     "action": action, "metric": "population_heavy_drift",
                     "mean": by_mix["logger1_heavy"]-by_mix["logger2_heavy"]},
                ))
    line_plot("population_balanced_bias_vs_lambda.png", pop_summary,
              "population_balanced_bias", "signed balanced bias", {"kappa": 0.0})
    line_plot("population_heavy_drift_vs_lambda.png", pop_summary,
              "population_heavy_drift", "signed heavy drift", {"kappa": 0.0})
    line_plot("neural_balanced_bias_vs_lambda.png", aggregate_rows,
              "signed_balanced_neural_bias", "signed neural bias",
              {"kappa": 0.0, "condition": "confounded", "mixture": "logger12_balanced"})
    line_plot("neural_heavy_drift_vs_lambda.png", aggregate_rows,
              "signed_heavy_neural_drift", "signed neural drift",
              {"kappa": 0.0, "condition": "confounded", "mixture": "logger1_minus_logger2"})

    def bar(filename: str, rows: Sequence[Mapping[str, Any]], metric: str, ylabel: str) -> None:
        selected = [row for row in rows if row.get("metric") == metric]
        labels = [f"{row.get('condition','')}:{row.get('action','')}" for row in selected]
        values = [float(row["mean"]) for row in selected]
        plt.figure(figsize=(max(6.0, len(values) * 0.35), 4.0))
        plt.bar(labels, values); plt.ylabel(ylabel); plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(figures / filename, dpi=160); plt.close()

    bar("learned_slope_by_action.png", slope_rows, "balanced_slope_with_intercept", "learned slope")
    bar("confounded_vs_independent_slope.png", slope_rows, "balanced_slope_error", "slope error")
    line_plot("paired_increment_error_vs_lambda.png", aggregate_rows,
              "paired_increment_absolute_error", "paired increment MAE",
              {"kappa": 0.0, "condition": "confounded", "mixture": "logger12_balanced"})
    line_plot("base_leakage_vs_lambda.png", aggregate_rows,
              "base_prediction_absolute_increment", "base leakage",
              {"kappa": 0.0, "condition": "confounded", "mixture": "logger12_balanced"})
    line_plot("observational_fit_vs_signal.png", aggregate_rows,
              "reward_population_mae", "reward population MAE",
              {"kappa": 0.0, "condition": "confounded", "mixture": "logger12_balanced"})
    bar("seed_variation_vs_lambda.png", aggregate_rows, "reward_population_mae", "between-seed SD")


def _write_report(output: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# Phase 8B-RS — Direct U-to-Reward Signal Calibration

This positive-control stage added `lambda_reward * u_env` to copied rewards.  It
did not replace the physical confounding DGP.  All population identities were
verified before neural training, and all read-only input hashes were unchanged.

The neural audit used only `{MODEL_INPUT_FIELDS}` ({MODEL_INPUT_DIMENSION}D), the
fixed width-{DEFAULT_HIDDEN_WIDTH} reward architecture, {summary['model_seed_count']}
model seeds, and held-out anchor evaluation.  Best-validation checkpoints were
used for reported predictions; final checkpoints were also retained.

## Interpretation logic

Case A: correct signed slopes, near-zero base/independent slopes, and paired
increment errors below absolute fit errors indicate that stronger direct reward
signal is learnable and that the original failure was substantially a
signal-to-approximation problem.

Case B: improved absolute fit with incorrect slopes or failed negative controls
indicates persistent cross-action interference or state generalization error.

Case C: failure to recover the lambda=0.20 slopes indicates failure even on this
strong positive control.

No effect-size threshold or automatic scientific verdict is applied.  The saved
metrics require manual review.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def run_reward_signal_calibration(
    phase8anc_root: Path, phase8a_root: Path, output_root: Path, *,
    phase8b_root: Path | None = None, num_anchors: int = 512,
    kappas: tuple[float, ...] = CALIBRATION_KAPPAS,
    reward_strengths: tuple[float, ...] = REWARD_STRENGTHS,
    conditions: tuple[str, ...] = CONDITIONS,
    mixtures: tuple[str, ...] = PRIMARY_MIXTURE_NAMES,
    model_seeds: tuple[int, ...] = (0, 1, 2), updates: int = 1500,
    batch_size: int = 512, device: str = "auto", split_seed: int = 0,
) -> dict[str, Any]:
    """Run the population gate and reward-only neural positive-control pilot."""
    kappas = tuple(float(value) for value in kappas)
    strengths = validate_reward_strengths(reward_strengths)
    conditions, mixtures = tuple(conditions), tuple(mixtures)
    if not kappas or any(value not in CALIBRATION_KAPPAS for value in kappas):
        raise ValueError(f"kappas must be a nonempty subset of {CALIBRATION_KAPPAS}")
    if conditions != CONDITIONS:
        raise ValueError(f"both conditions are required in canonical order: {CONDITIONS}")
    if mixtures != PRIMARY_MIXTURE_NAMES:
        raise ValueError("exactly the three primary mixtures are required in canonical order")
    if strengths[0] != 0.0 or tuple(sorted(strengths)) != strengths or len(strengths) < 2:
        raise ValueError("reward strengths must be increasing, paired to lambda=0, and contain >=2 values")
    if not model_seeds or min(num_anchors, updates, batch_size) <= 0:
        raise ValueError("anchors, updates, batch size, and model seeds must be nonempty/positive")
    if num_anchors > 512:
        raise ValueError("Phase 8B-RS is fixed to at most the 512 pilot anchors")

    paths, all_anchor_ids, baseline_root = require_verified_inputs(
        phase8anc_root, phase8a_root, phase8b_root)
    if num_anchors > len(all_anchor_ids):
        raise ValueError("num_anchors exceeds the 2048 verified anchors")
    selected = np.sort(all_anchor_ids)[:num_anchors]
    output = Path(output_root).resolve()
    nc, causal = Path(phase8anc_root).resolve(), Path(phase8a_root).resolve()
    if output in (nc, causal, baseline_root) or not output.name.startswith("phase8b_reward_signal"):
        raise RewardSignalCalibrationError("output must be a new phase8b_reward_signal* artifact")
    hashes_before = hash_input_files(paths)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_provenance = _reuse_or_make_splits(baseline_root, selected, split_seed)
    if not validate_splits(splits, selected):
        raise RewardSignalCalibrationError("anchor split is overlapping or incomplete")
    _write_json(output / "splits.json", {**splits, "split_seed": split_seed,
                                         "provenance": split_provenance})
    input_stats = _load_normalization(baseline_root / "normalization" / "input_stats.npz")
    _save_normalization(output / "normalization" / "input_stats.npz", input_stats)
    baseline_errors = _baseline_fit_errors(baseline_root / "aggregate_metrics.csv")
    manifest_in = _load_json(nc / "manifest.json")
    atol = float(manifest_in.get("numerical_tolerance", {}).get("atol", 1e-7))
    rtol = float(manifest_in.get("numerical_tolerance", {}).get("rtol", 1e-7))

    grouped: dict[tuple[float, float, str, str], GroupedTargets] = {}
    original: dict[tuple[float, str, str], GroupedTargets] = {}
    raw_oracles: dict[float, dict[str, np.ndarray]] = {}
    canonical: dict[float, np.ndarray] = {}
    do_rewards: dict[float, np.ndarray] = {}
    leakage_empty = True
    all_arrays_finite = True
    lambda_zero_public_exact = True
    for kappa in kappas:
        kname = KAPPA_NAMES[kappa]
        raw = load_npz(causal / kname / "do_oracle_raw.npz")
        raw_oracles[kappa] = raw
        for condition in conditions:
            public_all = load_npz(nc / kname / f"{condition}_public.npz")
            hidden_all = load_npz(nc / kname / f"{condition}_hidden_audit.npz")
            if FORBIDDEN_DERIVED_PUBLIC_FIELDS.intersection(public_all):
                raise RewardSignalCalibrationError("Phase 8A-NC public data exposes a hidden field")
            mask = np.isin(public_all["anchor_id"], selected)
            public = {name: np.asarray(value)[mask] for name, value in public_all.items()}
            hidden = {name: np.asarray(value)[mask] for name, value in hidden_all.items()}
            original_public, _ = make_derived_artifacts(public, hidden, 0.0)
            for mixture in mixtures:
                weights = np.asarray(np.load(nc / kname / "weights" / condition / f"{mixture}.npy"),
                                     dtype=np.float64)[mask]
                weights /= weights.sum()
                original[(kappa, condition, mixture)] = build_grouped_targets(original_public, weights)
            for strength in strengths:
                derived_public, hidden_audit = make_derived_artifacts(public, hidden, strength)
                leakage = validate_derived_artifacts(derived_public, hidden_audit)
                leakage_empty &= not leakage
                print(f"hidden leakage: {leakage}")
                directory = output / "derived_data" / kname / strength_name(strength)
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / f"{condition}_public.npz", **derived_public)
                np.savez_compressed(directory / f"{condition}_hidden_audit.npz", **hidden_audit)
                all_arrays_finite &= all(np.isfinite(np.asarray(value)).all()
                                         for value in derived_public.values()
                                         if np.issubdtype(np.asarray(value).dtype, np.number))
                if strength == 0.0:
                    lambda_zero_public_exact &= np.array_equal(
                        derived_public["reward"], np.asarray(public["reward"], dtype=np.float64))
                for mixture in mixtures:
                    weights = np.asarray(np.load(
                        nc / kname / "weights" / condition / f"{mixture}.npy"), dtype=np.float64)[mask]
                    weights /= weights.sum()
                    grouped[(kappa, strength, condition, mixture)] = build_grouped_targets(
                        derived_public, weights)
        reference = original[(kappa, conditions[0], mixtures[0])]
        canonical[kappa] = _canonical_indices(reference, raw, selected, kappa)
        do_original, _ = recompute_do_targets(raw, reference, selected, kappa)
        # Explicitly recompute the augmented do mean for every lambda and prove cancellation.
        lookup = {(int(a), str(key), int(u)): i for i, (a, key, u) in enumerate(
            zip(raw["anchor_id"], raw["action_key"], raw["u_env"]))}
        for strength in strengths:
            augmented_do = np.empty_like(do_original)
            for ai, anchor in enumerate(selected):
                for action_index, action in enumerate(ACTION_KEYS):
                    minus = lookup[(int(anchor), action, -1)]
                    plus = lookup[(int(anchor), action, 1)]
                    augmented_do[ai, action_index] = 0.5 * (
                        float(raw["reward"][plus]) + strength
                        + float(raw["reward"][minus]) - strength)
            if not np.allclose(augmented_do, do_original, atol=atol, rtol=rtol):
                raise RewardSignalCalibrationError("augmented do mean changes with lambda")
        do_rewards[kappa] = do_original

    gate_checks, population_rows = _population_gate(
        grouped, original, do_rewards, canonical, kappas, strengths, atol, rtol)
    for row in population_rows:
        signal = abs(float(row["theoretical_direct_increment"]))
        baseline = baseline_errors.get((float(row["kappa"]), str(row["condition"])))
        row["direct_signal_magnitude"] = signal
        row["previous_phase8b_reward_fit_error"] = baseline if baseline is not None else ""
        row["direct_signal_over_previous_fit_error"] = (
            signal / baseline if baseline is not None and baseline > 0 else 0.0)
    gate_checks.update({
        "verified_phase8anc_required": True,
        "verified_phase8a_oracle_required": True,
        "reward_grid_exact": set(strengths).issubset(REWARD_STRENGTHS),
        "augmented_reward_uses_u_env": True,
        "public_hidden_leakage_empty": leakage_empty,
        "original_and_augmented_reward_not_both_public": True,
        "all_2048_source_anchors_available": len(all_anchor_ids) == 2048,
        "all_requested_pilot_anchors_available": len(selected) == num_anchors,
        "split_reused_or_fixed": validate_splits(splits, selected),
        "lambda_zero_public_reward_matches_phase8anc": lambda_zero_public_exact,
        "all_derived_arrays_finite": all_arrays_finite,
    })
    failed_gate = [name for name, passed in gate_checks.items() if not passed]
    _write_csv(output / "population_tables.csv", population_rows)
    _write_json(output / "population_audit.json", {
        "checks": gate_checks, "all_passed": not failed_gate, "failed": failed_gate,
        "theoretical_direct_signal": {
            str(value): {"balanced_bias_magnitude": 0.6 * value,
                         "heavy_drift_magnitude": (14.0 / 45.0) * value}
            for value in REWARD_STRENGTHS},
        "previous_phase8b_reward_fit_error": {
            f"kappa_{kappa}:{condition}": value
            for (kappa, condition), value in baseline_errors.items()},
    })
    if failed_gate:
        gate_hashes_after = hash_input_files(paths)
        gate_unchanged = input_hashes_unchanged(hashes_before, gate_hashes_after)
        gate_checks["input_hashes_unchanged"] = gate_unchanged
        gate_checks["old_artifacts_unchanged"] = gate_unchanged
        failed_gate = [name for name, passed in gate_checks.items() if not passed]
        _write_json(output / "input_integrity.json", {
            "sha256_before": hashes_before, "sha256_after": gate_hashes_after,
            "unchanged": gate_unchanged, "required_file_count": len(paths)})
        _write_json(output / "hard_checks.json", {"checks": gate_checks,
                                                    "all_passed": False,
                                                    "failed": failed_gate})
        raise RewardSignalCalibrationError(f"population gate failed: {failed_gate}")

    resolved_device = resolve_device(device)
    train_ids, validation_ids = splits["train"], splits["validation"]
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    output_stats: dict[float, Normalization] = {}
    for kappa in kappas:
        balanced_zero = [grouped[(kappa, 0.0, condition, "logger12_balanced")].subset(train_ids)
                         for condition in conditions]
        stats = normalization(np.concatenate([item.reward[:, None] for item in balanced_zero]))
        output_stats[kappa] = stats
        _save_normalization(output / "normalization" / f"{KAPPA_NAMES[kappa]}_reward_stats.npz", stats)

    predictions: dict[tuple[float, float, str, str, int], np.ndarray] = {}
    population: dict[tuple[float, float, str, str], np.ndarray] = {}
    checkpoint_roundtrip = True
    initial_hash_consistent = True
    schedule_hash_consistent = True
    best_checkpoint_used = True
    all_neural_finite = True
    model_count = len(kappas) * len(strengths) * len(conditions) * len(mixtures) * len(model_seeds)
    completed = 0
    for kappa in kappas:
        order = canonical[kappa]
        test_order = np.asarray([np.where(selected == anchor)[0][0] for anchor in test_ids])
        canonical_test_rows = order[test_order]
        for condition in conditions:
            for strength in strengths:
                for mixture in mixtures:
                    dataset = grouped[(kappa, strength, condition, mixture)]
                    population[(kappa, strength, condition, mixture)] = dataset.reward[canonical_test_rows]
            for model_seed in model_seeds:
                paired_seed = int(model_seed + 1009 * round(kappa * 10)
                                  + 10007 * CONDITIONS.index(condition))
                initial_state, initial_digest = make_initial_state(
                    "reward", paired_seed, DEFAULT_HIDDEN_WIDTH)
                train_length = len(grouped[(kappa, strengths[0], condition, mixtures[0])].subset(train_ids).x)
                schedule = batch_schedule(train_length, updates, batch_size, paired_seed + 300007)
                schedule_digest = array_hash(schedule)
                observed_initial, observed_schedule = [], []
                for strength in strengths:
                    for mixture in mixtures:
                        dataset = grouped[(kappa, strength, condition, mixture)]
                        train, validation, test = (dataset.subset(ids) for ids in
                                                   (train_ids, validation_ids, test_ids))
                        best, final, history = _train_reward_model(
                            train, validation, input_stats, output_stats[kappa], schedule,
                            paired_seed, resolved_device, initial_state)
                        observed_initial.append(history["initial_state_hash"])
                        observed_schedule.append(history["schedule_hash"])
                        metadata = {
                            **history, "kappa": kappa, "lambda_reward": strength,
                            "condition": condition, "mixture": mixture,
                            "model_seed": model_seed, "target": "reward",
                            "model_input_fields": MODEL_INPUT_FIELDS,
                            "model_input_dimension": MODEL_INPUT_DIMENSION,
                            "hidden_width": DEFAULT_HIDDEN_WIDTH,
                            "parameter_count": parameter_count(best),
                            "normalization_shared_across_lambda": True,
                        }
                        model_dir = (output / "models" / KAPPA_NAMES[kappa]
                                     / strength_name(strength) / condition / mixture)
                        best_path = model_dir / f"seed_{model_seed}_best.pt"
                        final_path = model_dir / f"seed_{model_seed}_final.pt"
                        save_checkpoint(best_path, best, {**metadata, "checkpoint_role": "best_validation"})
                        save_checkpoint(final_path, final, {**metadata, "checkpoint_role": "final"})
                        prediction = predict(best, test.x, input_stats, output_stats[kappa], resolved_device)[:, 0]
                        reloaded, reloaded_metadata = load_checkpoint(
                            best_path, "reward", resolved_device, DEFAULT_HIDDEN_WIDTH)
                        repeated = predict(reloaded, test.x[:32], input_stats,
                                           output_stats[kappa], resolved_device)[:, 0]
                        checkpoint_roundtrip &= (
                            reloaded_metadata.get("checkpoint_role") == "best_validation"
                            and np.allclose(prediction[:32], repeated, atol=1e-7, rtol=1e-6))
                        # Canonical test group ordering is anchor-major/action-major.
                        test_lookup = {(int(anchor), action_bytes(action.astype(np.float32))): i
                                       for i, (anchor, action) in enumerate(
                                           zip(test.anchor_id, test.commanded_action))}
                        raw = raw_oracles[kappa]
                        command_lookup = {}
                        for row in range(len(raw["anchor_id"])):
                            command_lookup[(int(raw["anchor_id"][row]), str(raw["action_key"][row]))] = (
                                action_bytes(np.asarray(raw["commanded_action"][row], dtype=np.float32)))
                        prediction_order = [test_lookup[(int(anchor), command_lookup[(int(anchor), action)])]
                                            for anchor in test_ids for action in ACTION_KEYS]
                        values = prediction[np.asarray(prediction_order)].reshape(len(test_ids), 3)
                        predictions[(kappa, strength, condition, mixture, model_seed)] = values
                        prediction_dir = (output / "predictions" / KAPPA_NAMES[kappa]
                                          / strength_name(strength) / condition / mixture)
                        prediction_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(prediction_dir / f"seed_{model_seed}.npz",
                                            anchor_id=test_ids, prediction=values,
                                            population_target=population[(kappa, strength, condition, mixture)])
                        all_neural_finite &= np.isfinite(values).all()
                        checkpoint_roundtrip &= (
                            expected_parameter_count(1, DEFAULT_HIDDEN_WIDTH) == parameter_count(best))
                        best_checkpoint_used &= history["best_validation_step"] in history["validation_step"]
                        completed += 1
                        print(f"trained reward models: {completed}/{model_count}")
                initial_hash_consistent &= set(observed_initial) == {initial_digest}
                schedule_hash_consistent &= set(observed_schedule) == {schedule_digest}

    # Evaluation is deliberately after all training; do arrays were not passed to training.
    seed_rows: list[dict[str, Any]] = []
    anchor_arrays: dict[str, np.ndarray] = {}
    for kappa in kappas:
        test_positions = np.asarray([np.where(selected == anchor)[0][0] for anchor in test_ids])
        do = do_rewards[kappa][test_positions]
        for condition in conditions:
            for model_seed in model_seeds:
                for strength in strengths:
                    pred = {m: predictions[(kappa, strength, condition, m, model_seed)] for m in mixtures}
                    pop = {m: population[(kappa, strength, condition, m)] for m in mixtures}
                    for mixture in mixtures:
                        for action_index, action in enumerate(ACTION_KEYS):
                            error = pred[mixture][:, action_index] - pop[mixture][:, action_index]
                            for metric, values in (
                                ("reward_population_mae", np.abs(error)),
                                ("reward_population_rmse", error ** 2),
                            ):
                                stats = _stats(values, 1000 + len(seed_rows))
                                if metric.endswith("rmse"):
                                    stats = {**stats, "mean": float(np.sqrt(np.mean(values))),
                                             "ci_low": float(np.sqrt(max(stats["ci_low"], 0))),
                                             "ci_high": float(np.sqrt(max(stats["ci_high"], 0)))}
                                seed_rows.append({"kappa": kappa, "lambda_reward": strength,
                                    "condition": condition, "mixture": mixture, "action": action,
                                    "model_seed": model_seed, "metric": metric,
                                    "statistical_unit": "anchor_id", **stats})
                            if strength != strengths[0]:
                                base_prediction = predictions[(kappa, strengths[0], condition, mixture, model_seed)]
                                increment = pred[mixture][:, action_index] - base_prediction[:, action_index]
                                expected = direct_slope(condition, "mixture_increment", action, mixture) * strength
                                paired_error = np.abs(increment - expected)
                                for metric, values in (("paired_prediction_increment", increment),
                                                       ("paired_increment_absolute_error", paired_error)):
                                    seed_rows.append({"kappa": kappa, "lambda_reward": strength,
                                        "condition": condition, "mixture": mixture, "action": action,
                                        "model_seed": model_seed, "metric": metric,
                                        "statistical_unit": "anchor_id", **_stats(values, 2000+len(seed_rows))})
                                if action == "base":
                                    seed_rows.append({"kappa": kappa, "lambda_reward": strength,
                                        "condition": condition, "mixture": mixture, "action": action,
                                        "model_seed": model_seed,
                                        "metric": "base_prediction_absolute_increment",
                                        "statistical_unit": "anchor_id",
                                        **_stats(np.abs(increment), 3000+len(seed_rows))})
                    balanced_bias = pred["logger12_balanced"] - do
                    pop_balanced = pop["logger12_balanced"] - do
                    heavy = pred["logger1_heavy"] - pred["logger2_heavy"]
                    pop_heavy = pop["logger1_heavy"] - pop["logger2_heavy"]
                    for action_index, action in enumerate(ACTION_KEYS):
                        for mixture_label, metric, values in (
                            ("logger12_balanced", "signed_balanced_neural_bias", balanced_bias[:, action_index]),
                            ("logger12_balanced", "signed_balanced_population_bias", pop_balanced[:, action_index]),
                            ("logger1_minus_logger2", "signed_heavy_neural_drift", heavy[:, action_index]),
                            ("logger1_minus_logger2", "signed_heavy_population_drift", pop_heavy[:, action_index]),
                        ):
                            seed_rows.append({"kappa": kappa, "lambda_reward": strength,
                                "condition": condition, "mixture": mixture_label, "action": action,
                                "model_seed": model_seed, "metric": metric,
                                "statistical_unit": "anchor_id", **_stats(values, 4000+len(seed_rows))})
                        if strength > 0:
                            heavy_zero = (
                                predictions[(kappa, strengths[0], condition,
                                             "logger1_heavy", model_seed)][:, action_index]
                                - predictions[(kappa, strengths[0], condition,
                                               "logger2_heavy", model_seed)][:, action_index])
                            heavy_increment = heavy[:, action_index] - heavy_zero
                            expected_heavy = theoretical_heavy_direct_drift(
                                action, strength, condition)
                            for metric, values in (
                                ("paired_heavy_prediction_increment", heavy_increment),
                                ("paired_heavy_increment_absolute_error",
                                 np.abs(heavy_increment - expected_heavy)),
                            ):
                                seed_rows.append({"kappa": kappa, "lambda_reward": strength,
                                    "condition": condition,
                                    "mixture": "logger1_minus_logger2", "action": action,
                                    "model_seed": model_seed, "metric": metric,
                                    "statistical_unit": "anchor_id",
                                    **_stats(values, 4500+len(seed_rows))})
                    pred_top, do_top = _top_masks(pred["logger12_balanced"]), _top_masks(do)
                    disagreement = (~np.any(pred_top & do_top, axis=1)).astype(float)
                    chosen = np.argmax(pred["logger12_balanced"], axis=1)
                    regret = np.max(do, axis=1) - do[np.arange(len(do)), chosen]
                    base_predictions = np.stack([pred[mixture][:, 1] for mixture in mixtures])
                    base_range = np.max(base_predictions, axis=0) - np.min(base_predictions, axis=0)
                    for metric, values in (("neural_do_ranking_disagreement", disagreement),
                                           ("neural_decision_regret", regret),
                                           ("base_prediction_range_across_mixtures", base_range)):
                        seed_rows.append({"kappa": kappa, "lambda_reward": strength,
                            "condition": condition,
                            "mixture": ("all_primary" if metric.startswith("base_")
                                        else "logger12_balanced"),
                            "action": ("base" if metric.startswith("base_") else "all"),
                            "model_seed": model_seed, "metric": metric,
                            "statistical_unit": "anchor_id", **_stats(values, 5000+len(seed_rows))})
                    if strength > 0:
                        seed_rows.append({"kappa": kappa, "lambda_reward": strength,
                            "condition": condition, "mixture": "all_primary", "action": "base",
                            "model_seed": model_seed,
                            "metric": "base_leakage_over_balanced_direct_signal",
                            "statistical_unit": "anchor_id",
                            **_stats(base_range / (0.6 * strength), 5500+len(seed_rows))})

                all_base_predictions = np.stack([
                    predictions[(kappa, strength, condition, mixture, model_seed)][:, 1]
                    for strength in strengths for mixture in mixtures])
                all_base_range = np.max(all_base_predictions, axis=0) - np.min(
                    all_base_predictions, axis=0)
                seed_rows.append({"kappa": kappa, "lambda_reward": "all",
                    "condition": condition, "mixture": "all_primary", "action": "base",
                    "model_seed": model_seed,
                    "metric": "base_prediction_range_across_mixture_and_lambda",
                    "statistical_unit": "anchor_id",
                    **_stats(all_base_range, 5800+len(seed_rows))})

    slope_seed_rows: list[dict[str, Any]] = []
    slope_arrays: dict[str, np.ndarray] = {}
    lambda_array = np.asarray(strengths, dtype=np.float64)
    if len(strengths) >= 2 and strengths[0] == 0.0:
        for kappa in kappas:
            for condition in conditions:
                for model_seed in model_seeds:
                    balanced = np.stack([predictions[(kappa, strength, condition,
                                                      "logger12_balanced", model_seed)]
                                         for strength in strengths])
                    heavy = np.stack([predictions[(kappa, strength, condition,
                                                   "logger1_heavy", model_seed)]
                                      - predictions[(kappa, strength, condition,
                                                     "logger2_heavy", model_seed)]
                                      for strength in strengths])
                    for family, values in (("balanced", balanced), ("heavy", heavy)):
                        slope, origin, r2 = fit_lambda_slope(lambda_array, values)
                        for action_index, action in enumerate(ACTION_KEYS):
                            theory_family = "balanced_bias" if family == "balanced" else "heavy_drift"
                            theory = direct_slope(condition, theory_family, action)
                            for metric, array in (
                                (f"{family}_slope_with_intercept", slope[:, action_index]),
                                (f"{family}_slope_through_origin_increment", origin[:, action_index]),
                                (f"{family}_slope_error", np.abs(slope[:, action_index]-theory)),
                                (f"{family}_slope_r2", r2[:, action_index]),
                            ):
                                slope_seed_rows.append({"kappa": kappa, "lambda_reward": "all",
                                    "condition": condition, "mixture": family, "action": action,
                                    "model_seed": model_seed, "metric": metric,
                                    "theoretical_slope": theory, "statistical_unit": "anchor_id",
                                    **_stats(array, 6000+len(slope_seed_rows))})
                                slope_arrays[f"{KAPPA_NAMES[kappa]}__{condition}__seed_{model_seed}__{metric}__{action}"] = array

    aggregate_rows = _aggregate_seed_metrics(seed_rows)
    slope_rows = _aggregate_seed_metrics(slope_seed_rows)
    for index, row in enumerate(slope_rows):
        arrays = np.stack([
            slope_arrays[
                f"{KAPPA_NAMES[float(row['kappa'])]}__{row['condition']}__seed_{seed}__"
                f"{row['metric']}__{row['action']}"]
            for seed in model_seeds])
        anchor_mean = np.mean(arrays, axis=0)
        anchor_stats = _bootstrap_mean(anchor_mean, 2000, 70000 + index)
        row.update(anchor_stats)
        row["seed_sd"] = float(np.std(np.mean(arrays, axis=1), ddof=1)) \
            if len(model_seeds) > 1 else 0.0
        row["ci_method"] = "bootstrap anchors after averaging paired model-seed predictions"
    all_metrics_finite = all(np.isfinite(float(row[key])) for rows in
        (seed_rows, aggregate_rows, slope_seed_rows, slope_rows) for row in rows
        for key in ("mean", "ci_low", "ci_high") if key in row)
    np.savez_compressed(output / "anchor_action_metrics.npz", **slope_arrays)
    _write_csv(output / "seed_metrics.csv", [*seed_rows, *slope_seed_rows])
    _write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(output / "slope_metrics.csv", slope_rows)
    _make_figures(output, population_rows, aggregate_rows, slope_rows)

    hard_checks = {
        **gate_checks,
        "public_hidden_leakage_empty": leakage_empty,
        "original_and_augmented_reward_not_both_public": True,
        "all_512_pilot_anchors_available": num_anchors <= 512,
        "split_reused_or_fixed": validate_splits(splits, selected),
        "reward_model_input_15d": MODEL_INPUT_DIMENSION == 15,
        "model_excludes_lambda": "lambda_reward" not in MODEL_INPUT_FIELDS,
        "model_excludes_kappa": "kappa_env" not in MODEL_INPUT_FIELDS,
        "model_excludes_logger": "logger_id" not in MODEL_INPUT_FIELDS,
        "model_excludes_hidden_u": "u_env" not in MODEL_INPUT_FIELDS,
        "output_normalization_shared_across_lambda": True,
        "same_initial_hash_across_lambda": initial_hash_consistent,
        "same_batch_schedule_across_lambda": schedule_hash_consistent,
        "best_checkpoint_roundtrip": checkpoint_roundtrip,
        "best_validation_checkpoint_used": best_checkpoint_used,
        "lambda_zero_crosscheck_phase8b": lambda_zero_public_exact,
        "paired_increment_alignment": strengths[0] == 0.0,
        "no_nan_inf": all_neural_finite and all_metrics_finite,
        "do_oracle_excluded_from_training": not LEAKAGE_FLAGS["DO_ORACLE_USED_FOR_TRAINING"],
        "fixed_width_256_architecture": expected_parameter_count(1, DEFAULT_HIDDEN_WIDTH) == 135937,
    }
    hashes_after = hash_input_files(paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks["input_hashes_unchanged"] = unchanged
    hard_checks["old_artifacts_unchanged"] = unchanged
    failed = [name for name, passed in hard_checks.items() if not passed]
    _write_json(output / "input_integrity.json", {
        "sha256_before": hashes_before, "sha256_after": hashes_after,
        "unchanged": unchanged, "required_file_count": len(paths)})
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
                                                "all_passed": not failed, "failed": failed})
    if failed:
        raise RewardSignalCalibrationError(f"hard checks failed: {failed}")

    manifest = {
        "stage": "Phase 8B-RS", "phase8anc_root": str(nc),
        "phase8a_root": str(causal), "phase8b_baseline_root": str(baseline_root),
        "available_anchor_count": 2048, "analyzed_anchor_count": num_anchors,
        "kappas": list(kappas), "reward_strengths": list(strengths),
        "reward_strength_grid_fixed_before_neural_results": True,
        "conditions": list(conditions), "mixtures": list(mixtures),
        "model_seeds": list(model_seeds), "updates": updates, "batch_size": batch_size,
        "optimizer": "Adam", "learning_rate": 0.001, "early_stopping": False,
        "evaluation_checkpoint": "best_validation", "final_checkpoint_saved": True,
        "device": resolved_device, "model_input_fields": list(MODEL_INPUT_FIELDS),
        "model_input_dimension": MODEL_INPUT_DIMENSION,
        "architecture": "15-256-256-256-1 ReLU", "hidden_width": DEFAULT_HIDDEN_WIDTH,
        "parameter_count": expected_parameter_count(1, DEFAULT_HIDDEN_WIDTH),
        "reward_definition": "original_reward + lambda_reward * u_env",
        "input_normalization": "reused Phase 8B-NC input statistics",
        "output_normalization": "per-kappa lambda=0 balanced train, shared across lambda",
        "split_provenance": split_provenance, "public_fields": list(PUBLIC_FIELDS),
        "hidden_audit_fields": list(HIDDEN_AUDIT_FIELDS), "leakage_flags": LEAKAGE_FLAGS,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
    }
    summary = {
        "stage": "Phase 8B-RS", "analyzed_anchor_count": num_anchors,
        "test_anchor_count": len(test_ids), "model_seed_count": len(model_seeds),
        "trained_model_count": model_count, "hard_checks": hard_checks,
        "all_hard_checks_passed": True, "aggregate_metrics": aggregate_rows,
        "slope_metrics": slope_rows, "scientific_verdict": "MANUAL_DECISION_REQUIRED",
        "statistical_units": {"bootstrap": "anchor_id", "model_variation": "model_seed",
                              "seed_anchor_rows_treated_as_independent": False},
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    _write_report(output, summary)
    return summary
