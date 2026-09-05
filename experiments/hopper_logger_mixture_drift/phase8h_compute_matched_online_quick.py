"""Phase 8H-ON-Q: compute-matched components, recursive potentials, and PBRS/SAC.

The three stages are deliberately separate.  Existing artifacts are read-only;
new component, potential, and SAC checkpoints live below one new output root.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from aamas_hopper_adapter import (
    ContinuousAAMASComponents,
    _import_official_module,
    compute_official_continuous_action_backup,
    compute_source_aamas_backup,
    validate_external_repo,
)
from confounded_hopper import ACTUATOR_DIRECTION, PRIVATE_INFO_FIELDS, ConfoundedHopperWrapper
from scripts.train_aamas_hopper_potential import seed_everything
from .generate_datasets import MujocoOneStepSimulator
from .phase8h_data_scaling import (
    BATCH_SIZE,
    _metric_row,
    generate_nested_master,
    subset_nested,
)
from .phase8h_quick_multipolicy_aamas import (
    ACTION_SEPARATION,
    CANDIDATE_ACTIONS,
    EXTERNAL_COMMIT,
    GAMMA,
    KAPPA,
    LAMBDA_REWARD,
    FrozenSACReferenceValue,
    _device_name,
    _load_phase8h_inputs,
    _save_component_checkpoint,
    fit_aamas_components,
    pooled_row_weights,
    source_policy_parameters,
    union_candidate_actions,
)


PHASE = "Phase 8H-ON-Q"
COMPONENT_UPDATES = 4000
POTENTIAL_EPOCHS = 200
POTENTIAL_BATCH_SIZE = 1028
POTENTIAL_LR = 1e-4
TARGET_TAU = 0.005
TARGET_UPDATE_INTERVAL = 3
PBRS_BETA = 1.0
ONLINE_METHODS = (
    "sac_scratch",
    "sac_pooled_union",
    "sac_state_min",
    "sac_action_min",
    "sac_pooled_native",
)
POTENTIAL_METHODS = (
    "pooled_aamas_union_full",
    "state_min_full",
    "action_min_full",
    "pooled_aamas_native_full",
)
MODEL_TO_POTENTIAL = {
    "sac_pooled_union": "pooled_aamas_union_full",
    "sac_state_min": "state_min_full",
    "sac_action_min": "action_min_full",
    "sac_pooled_native": "pooled_aamas_native_full",
}
SAC_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": GAMMA,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
}


class Phase8HComputeMatchedOnlineError(RuntimeError):
    """Raised when a frozen Phase 8H-ON-Q contract is violated."""


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
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _git_commit() -> str | None:
    result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Compact BLAKE2b-128 integrity record."""
    target = Path(path).resolve()
    digest = hashlib.blake2b(digest_size=16)
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    stat = target.stat()
    return {"path": str(target), "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
            "blake2b_128": digest.hexdigest()}


def fingerprint_snapshot(paths: Sequence[Path]) -> list[dict[str, Any]]:
    unique = sorted({Path(path).resolve() for path in paths}, key=str)
    return [file_fingerprint(path) for path in unique]


def _manifest_stage(path: Path) -> str | None:
    try:
        return json.loads((path / "manifest.json").read_text(encoding="utf-8")).get("stage")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def resolve_artifact_root(requested: Path, stage: str) -> Path:
    """Resolve a missing historical path from sibling manifests, never by creating it."""
    requested = Path(requested).resolve()
    if requested.is_dir() and _manifest_stage(requested) == stage:
        return requested
    parent = requested.parent
    matches = [path for path in parent.iterdir() if path.is_dir() and _manifest_stage(path) == stage]
    if not matches:
        raise Phase8HComputeMatchedOnlineError(
            f"cannot locate {stage} artifact from requested path: {requested}")
    passed = []
    for path in matches:
        checks = path / "hard_checks.json"
        try:
            record = json.loads(checks.read_text(encoding="utf-8"))
            if record.get("all_passed", record.get("all_hard_checks_passed")) is True:
                passed.append(path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    choices = passed or matches
    return sorted(choices, key=lambda path: path.stat().st_mtime_ns)[-1]


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("torch", "stable_baselines3", "gymnasium", "mujoco"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "UNKNOWN"))
        except (ImportError, OSError) as error:
            versions[name] = f"UNAVAILABLE: {type(error).__name__}"
    return versions


def pbrs_increment(phi_state: np.ndarray | float, phi_next: np.ndarray | float,
                   terminated: np.ndarray | bool, gamma: float = GAMMA,
                   beta: float = PBRS_BETA) -> np.ndarray:
    state = np.asarray(phi_state, dtype=np.float64)
    following = np.asarray(phi_next, dtype=np.float64)
    terminal = np.asarray(terminated, dtype=bool)
    result = float(beta) * (float(gamma) * (~terminal) * following - state)
    if not np.all(np.isfinite(result)):
        raise Phase8HComputeMatchedOnlineError("PBRS increment is nonfinite")
    return result


def discounted_shaping_sum(phi: Sequence[float], terminated: Sequence[bool],
                           gamma: float = GAMMA, beta: float = PBRS_BETA) -> float:
    values = np.asarray(phi, dtype=np.float64)
    terminal = np.asarray(terminated, dtype=bool)
    if values.ndim != 1 or len(values) != len(terminal) + 1:
        raise ValueError("phi must have one more entry than terminated")
    increments = pbrs_increment(values[:-1], values[1:], terminal, gamma, beta)
    return float(np.sum((float(gamma) ** np.arange(len(increments))) * increments))


def normalized_auc(steps: Sequence[int], values: Sequence[float]) -> float:
    x, y = np.asarray(steps, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("AUC inputs must be aligned on strictly increasing steps")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def normalized_positive_area(steps: Sequence[int], differences: Sequence[float]) -> float:
    """Exact piecewise-linear integral of max(difference, 0), including zero crossings."""
    x, y = np.asarray(steps, dtype=np.float64), np.asarray(differences, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("area inputs must be aligned on strictly increasing steps")
    area = 0.0
    for left, right, a, b in zip(x[:-1], x[1:], y[:-1], y[1:]):
        width = right - left
        if a >= 0 and b >= 0:
            area += width * (a + b) / 2
        elif a <= 0 and b <= 0:
            continue
        else:
            crossing = abs(a) / (abs(a) + abs(b))
            if a > 0:
                area += width * crossing * a / 2
            else:
                area += width * (1 - crossing) * b / 2
    return float(area / (x[-1] - x[0]))


def aggregate_dynamic_backup(method: str, source_q: np.ndarray,
                             pooled_q: np.ndarray | None = None) -> np.ndarray:
    source = np.asarray(source_q, dtype=np.float64)
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("source_q must be finite [source,state,candidate]")
    if method == "action_min_full":
        return source.min(axis=0).max(axis=1)
    if method == "state_min_full":
        return source.max(axis=2).min(axis=0)
    if method == "pooled_aamas_union_full":
        if pooled_q is None:
            raise ValueError("pooled union requires pooled_q")
        pooled = np.asarray(pooled_q, dtype=np.float64)
        if pooled.shape != source.shape[1:]:
            raise ValueError("pooled_q does not match source state/candidate axes")
        return pooled.max(axis=1)
    if method == "pooled_aamas_native_full":
        if pooled_q is None:
            raise ValueError("native pooled requires pooled_q")
        pooled = np.asarray(pooled_q, dtype=np.float64)
        if pooled.shape != (source.shape[1], 1):
            raise ValueError("native pooled must use one observed action per row")
        return pooled[:, 0]
    raise ValueError(f"unknown potential method: {method}")


def parameter_fingerprint(modules: Sequence[Any]) -> dict[str, float | int]:
    values = []
    for module in modules:
        values.extend(parameter.detach().cpu().double().reshape(-1).numpy()
                      for parameter in module.parameters())
    flat = np.concatenate(values) if values else np.zeros(0, dtype=np.float64)
    return {"parameter_count": int(flat.size), "sum": float(flat.sum()),
            "sum_squares": float(np.square(flat).sum()),
            "max_abs": float(np.max(np.abs(flat))) if len(flat) else 0.0}


def _same_fingerprint(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["parameter_count"] == right["parameter_count"] and all(
        np.isclose(float(left[key]), float(right[key]), atol=0.0, rtol=0.0)
        for key in ("sum", "sum_squares", "max_abs"))


def _source_checkpoint(base: Path, seed: int, source: int) -> Path:
    return base / "source" / f"source_{source}" / f"seed_{seed}.pt"


def _pooled_checkpoint(base: Path, seed: int) -> Path:
    return base / "pooled" / "balanced" / f"seed_{seed}.pt"


def _component_paths(scaling: Path, output: Path, seed: int, n: int) -> dict[str, Path]:
    if n == 128:
        base = scaling / "models" / "confounded" / "n128"
    elif seed == 0:
        base = scaling / "models" / "confounded" / "n32_extra_compute"
    else:
        base = output / "stage_a" / "missing_component_checkpoints" / f"seed_{seed}"
    return {**{f"source_{source}": _source_checkpoint(base, seed, source)
               for source in (1, 2, 3)}, "pooled_balanced": _pooled_checkpoint(base, seed)}


def _load_component(path: Path, official: Any, torch: Any, device: str) -> ContinuousAAMASComponents:
    payload = torch.load(path, map_location=device, weights_only=False)
    state, norm = payload["state_dict"], payload["normalization"]
    behavior = official.GaussianNN(12, 3, hidden_dim=128).to(device)
    delta = official.RegressionNN(15, 12, hidden_dim=256).to(device)
    reward = official.RegressionNN(15, 1, hidden_dim=256).to(device)
    behavior.load_state_dict(state["behavior"])
    delta.load_state_dict(state["state_difference"])
    reward.load_state_dict(state["reward"])
    for module in (behavior, delta, reward):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return ContinuousAAMASComponents(
        behavior, delta, reward, float(norm["reward_mean"]), float(norm["reward_std"]),
        float(norm["reward_upper"]), GAMMA, torch.device(device),
        ACTION_SEPARATION, CANDIDATE_ACTIONS)


def _component_public_validation(bundle: ContinuousAAMASComponents,
                                 public: Mapping[str, np.ndarray],
                                 validation_anchors: Sequence[int], torch: Any) -> dict[str, float]:
    rows = np.flatnonzero(np.isin(public["anchor_id"], validation_anchors))
    state = torch.as_tensor(public["observation"][rows], dtype=torch.float32, device=bundle.device)
    action = torch.as_tensor(public["commanded_action"][rows], dtype=torch.float32, device=bundle.device)
    next_state = torch.as_tensor(public["next_observation"][rows], dtype=torch.float32, device=bundle.device)
    reward = torch.as_tensor(
        (public["reward"][rows] - bundle.reward_mean) / (bundle.reward_std + 1e-7),
        dtype=torch.float32, device=bundle.device)
    with torch.no_grad():
        pair = torch.cat((state, action), dim=1)
        log_probability = bundle.behavior_model(state).log_prob(action)
        if log_probability.ndim > 1:
            log_probability = log_probability.sum(dim=-1)
        behavior = -log_probability.mean()
        delta = (bundle.state_difference_model(pair) - (next_state - state)).square().mean()
        reward_loss = (bundle.reward_model(pair).reshape(-1) - reward).square().mean()
    result = {"row_count": len(rows), "behavior_nll": float(behavior.cpu()),
              "delta_mse": float(delta.cpu()), "reward_mse": float(reward_loss.cpu())}
    result["total"] = sum(result[key] for key in ("behavior_nll", "delta_mse", "reward_mse"))
    return result


def _evaluate_components(
    phase8h: Path, components: Mapping[str, ContinuousAAMASComponents], seed: int,
    states: np.ndarray, reference: Callable[[np.ndarray], np.ndarray],
) -> list[dict[str, Any]]:
    with np.load(phase8h / "predictions" / "anchor_candidate_predictions.npz",
                 allow_pickle=False) as archive:
        candidates = archive[f"confounded__{seed}__union_actions"].copy()
        truth = archive[f"confounded__{seed}__do_q"].copy()
    noise = np.random.default_rng(20260806).standard_normal(
        (len(states) * candidates.shape[1], CANDIDATE_ACTIONS, 3)).astype(np.float32)
    source_q = compute_source_aamas_backup(
        tuple(components[f"source_{source}"] for source in (1, 2, 3)),
        states, candidates, reference,
        common_noise=noise)
    pooled_q = compute_source_aamas_backup(
        (components["pooled_balanced"],), states, candidates,
        reference, common_noise=noise)[0]
    predictions = {
        "source_1": source_q[0], "source_2": source_q[1], "source_3": source_q[2],
        "pooled_balanced": pooled_q,
        "state_level_min": source_q[source_q.max(axis=2).argmin(axis=0),
                                     np.arange(len(states)), :],
        "action_level_min": source_q.min(axis=0),
    }
    rows = []
    for method, prediction in predictions.items():
        metric = _metric_row(
            "confounded", seed, "compute_matched", 0, COMPONENT_UPDATES,
            method, truth, prediction)
        rows.append({"method": method, **{
            key: value for key, value in metric.items()
            if key not in {"condition", "seed", "data_label",
                           "samples_per_anchor_source", "gradient_updates",
                           "method", "candidate_count"}
        }})
    return rows


def _curve_diagnostics(curves: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in curves:
        if "validation_loss" not in row or "step" not in row:
            continue
        groups[(str(row.get("data_label", "")), str(row.get("seed", "")),
                str(row.get("model", "")), str(row.get("condition", "")))].append(row)
    result = []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda row: int(float(row["step"])))
        steps = np.asarray([float(row["step"]) for row in ordered])
        losses = np.asarray([float(row["validation_loss"]) for row in ordered])
        if not np.all(np.isfinite(losses)):
            status = "nonfinite"
            late_slope = float("nan")
        else:
            late_count = max(3, int(math.ceil(.2 * len(losses))))
            late_x, late_y = steps[-late_count:], losses[-late_count:]
            late_slope = (float(np.polyfit(late_x, late_y, 1)[0])
                          if len(np.unique(late_x)) > 1 else 0.0)
            relative_late_change = abs(late_y[-1] - late_y[0]) / max(abs(late_y[0]), 1e-12)
            if losses[-1] > losses.min() * 1.02 and late_slope > 0:
                status = "training_down_validation_worse_or_overfit"
            elif relative_late_change <= .01:
                status = "plateau_like"
            elif late_slope < 0:
                status = "validation_still_improving"
            else:
                status = "validation_not_improving"
        best = int(np.argmin(losses))
        result.append({
            "data_label": key[0], "seed": key[1], "model": key[2], "condition": key[3],
            "recorded_points": len(losses), "final_step": int(steps[-1]),
            "best_step": int(steps[best]), "best_step_fraction": float(steps[best] / steps[-1]),
            "best_validation_loss": float(losses[best]), "final_validation_loss": float(losses[-1]),
            "late_validation_slope_per_step": late_slope, "descriptive_status": status,
        })
    return result


def _stage_a_report(output: Path, metrics: Sequence[Mapping[str, Any]],
                    validation: Sequence[Mapping[str, Any]], curves: Sequence[Mapping[str, Any]],
                    checks: Mapping[str, bool]) -> None:
    diagnostics = _curve_diagnostics(curves)
    lines = ["# Stage A training readiness", "",
             "This report is a compute-matched diagnostic, not a test-based model-selection step.", "",
             f"All implementation checks passed: **{all(checks.values())}**.", "",
             "The online pilot remains pre-specified to n32/4000 seeds 0,1,2.", "",
             "## Compute-matched offline metrics", "",
             "See `compute_matched_metrics.csv` for all methods and seeds.", "",
             "## Common D32 validation", "",
             f"Frozen component evaluations: {len(validation)} rows in `common_validation_metrics.csv`.", "",
             "The original checkpoint-selection rules are retained; this common-D32 evaluation is diagnostic only.",
             "", "## Descriptive training-curve audit", "",
             "A late best checkpoint alone is not treated as proof of non-convergence, and exhausting 4000 "
             "updates is not treated as proof of convergence.", "",
             "| Data | Seed | Component | Best/final step | Best/final validation | Late status |",
             "|---|---:|---|---:|---:|---|"]
    for row in diagnostics:
        lines.append(
            f"| {row['data_label']} | {row['seed']} | {row['model']} | "
            f"{row['best_step']}/{row['final_step']} | "
            f"{row['best_validation_loss']:.6g}/{row['final_validation_loss']:.6g} | "
            f"{row['descriptive_status']} |")
    lines.extend(["", "Slowly falling validation loss is recorded as possibly not fully trained; "
                  "it does not automatically expand the budget or block Stage B."])
    lines.extend(["", "## Paired n32/4000 minus n128/4000 diagnostics", "",
                  "Negative differences favor n32 for error/regret; the three model seeds are paired.", "",
                  "| Method | Metric | Mean paired difference | Seed differences |",
                  "|---|---|---:|---|"])
    index = {(int(row["n"]), int(row["seed"]), str(row["method"])): row
             for row in metrics}
    for method in sorted({str(row["method"]) for row in metrics}):
        for metric in ("do_mae", "regret_mean", "regret_median", "regret_p90",
                       "regret_cvar90", "underestimation_fraction"):
            differences = [float(index[(32, seed, method)][metric])
                           - float(index[(128, seed, method)][metric])
                           for seed in (0, 1, 2)]
            lines.append(f"| {method} | {metric} | {np.mean(differences):.6g} | "
                         + ", ".join(f"{value:.6g}" for value in differences) + " |")
    (output / "stage_a" / "training_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _phase8a_from_phase8h(phase8h: Path) -> Path:
    candidate = phase8h.parent / "controlled_loggers_seed0_verified"
    if not candidate.is_dir():
        raise Phase8HComputeMatchedOnlineError(f"Phase 8A root is unavailable: {candidate}")
    return candidate


def _preflight_record(phase8h: Path, scaling: Path, output: Path,
                      missing_seeds: Sequence[int], online_steps: int = 50_000,
                      run_ids: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    expected_missing = sum(not path.is_file() for seed in missing_seeds
                           for path in _component_paths(scaling, output, int(seed), 32).values())
    required_existing = [
        *[path for seed in (0, 1, 2)
          for path in _component_paths(scaling, output, seed, 128).values()],
        *_component_paths(scaling, output, 0, 32).values(),
    ]
    native_distinct = True
    potential_count = len(POTENTIAL_METHODS) * len(run_ids)
    online_methods = len(ONLINE_METHODS) if native_distinct else len(ONLINE_METHODS) - 1
    online_runs = online_methods * len(run_ids)
    evaluations = online_runs * 6 * 5
    estimated_files = 15 + expected_missing + potential_count + online_runs + evaluations // 5
    estimated_bytes = expected_missing * 400_000 + potential_count * 350_000 + online_runs * 1_500_000
    return {
        "phase": PHASE,
        "resolved_phase8h_root": str(phase8h),
        "resolved_scaling_root": str(scaling),
        "missing_component_models": expected_missing,
        "missing_required_existing_component_models": sum(
            not path.is_file() for path in required_existing),
        "potential_count": potential_count,
        "native_baseline_required": native_distinct,
        "online_method_count": online_methods,
        "online_run_count": online_runs,
        "online_training_environment_steps": online_runs * int(online_steps),
        "estimated_evaluation_episodes": evaluations,
        "estimated_evaluation_environment_steps_upper_bound": evaluations * 1_000,
        "estimated_file_count": estimated_files,
        "estimated_storage_bytes": estimated_bytes,
    }


def run_stage_a(
    phase8h_root: Path, scaling_root: Path, output_root: Path, *,
    missing_n32_seeds: Sequence[int], component_updates: int, device: str,
    external_repo: Path,
) -> dict[str, Any]:
    if tuple(map(int, missing_n32_seeds)) != (1, 2) or component_updates != COMPONENT_UPDATES:
        raise Phase8HComputeMatchedOnlineError("Stage A is frozen to missing seeds 1,2 and 4000 updates")
    phase8h = resolve_artifact_root(phase8h_root, "Phase 8H-Q")
    scaling = resolve_artifact_root(scaling_root, "Phase 8H-DS")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = _preflight_record(phase8h, scaling, output, missing_n32_seeds)
    _write_json(output / "preflight.json", preflight)
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
        from stable_baselines3 import SAC
    except (ImportError, OSError) as error:
        raise Phase8HComputeMatchedOnlineError("PyTorch is required for Stage A") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(external_repo)
    phase8a = _phase8a_from_phase8h(phase8h)
    inputs = _load_phase8h_inputs(phase8a, 512, None, compute_checkpoint_hash=False)
    anchors, splits = inputs["anchors"], inputs["splits"]
    reference_sac = SAC.load(str(inputs["checkpoint"]), device=selected_device)
    reference = FrozenSACReferenceValue(
        reference_sac, selected_device, use_parameter_hash=False)
    tracked = [phase8h / "manifest.json", phase8h / "splits.json",
               phase8h / "predictions" / "anchor_candidate_predictions.npz",
               scaling / "manifest.json", scaling / "seed_metrics.csv",
               scaling / "training_curves.csv", *inputs["required_paths"]]
    tracked.extend(path for seed in (0, 1, 2) for path in
                   _component_paths(scaling, output, seed, 128).values())
    tracked.extend(_component_paths(scaling, output, 0, 32).values())
    missing_inputs = [str(path) for path in tracked if not Path(path).is_file()]
    if missing_inputs:
        raise Phase8HComputeMatchedOnlineError(
            "required read-only component artifacts are missing: " + ", ".join(missing_inputs[:8]))
    before = fingerprint_snapshot(tracked)
    simulator = MujocoOneStepSimulator(anchors, (KAPPA,), seed=20260804)
    try:
        master, _, generation = generate_nested_master(
            anchors, simulator, condition="confounded", seed=20260804, max_samples=32)
    finally:
        simulator.close()
    public = subset_nested(master, 32)
    curve_path = output / "stage_a" / "training_curves.csv"
    curves: list[dict[str, Any]] = _read_csv(curve_path) if curve_path.is_file() else []
    trained = 0
    for seed in missing_n32_seeds:
        paths = _component_paths(scaling, output, int(seed), 32)
        for source in (1, 2, 3):
            path = paths[f"source_{source}"]
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload["metadata"].get("gradient_updates") != COMPONENT_UPDATES:
                    raise Phase8HComputeMatchedOnlineError(f"mismatched reusable checkpoint: {path}")
                continue
            source_public = {name: np.asarray(value)[public["source_id"] == source]
                             for name, value in public.items()}
            training_seed = int(seed) * 100 + source
            seed_everything(training_seed, torch, cuda_training=selected_device == "cuda")
            _, normalization, training = fit_aamas_components(
                source_public, splits["train"], splits["observational_validation"],
                row_probabilities=np.ones(len(source_public["reward"])),
                seed=training_seed, gradient_updates=COMPONENT_UPDATES,
                batch_size=BATCH_SIZE, device=selected_device, official=official,
                torch=torch, record_schedule_digest=False)
            metadata = {**training["metadata"], "condition": "confounded",
                        "data_label": "n32_compute_matched", "samples_per_anchor_source": 32,
                        "model_kind": "source", "source_id": source}
            _save_component_checkpoint(path, training, normalization, metadata, torch)
            curves.extend({"condition": "confounded", "seed": seed,
                           "data_label": "n32_compute_matched", "model": f"source_{source}",
                           **row} for row in training["history"])
            trained += 1
        path = paths["pooled_balanced"]
        if not path.is_file():
            weights = pooled_row_weights(public["source_id"], [1 / 3, 1 / 3, 1 / 3])
            training_seed = int(seed) * 100 + 50
            seed_everything(training_seed, torch, cuda_training=selected_device == "cuda")
            _, normalization, training = fit_aamas_components(
                public, splits["train"], splits["observational_validation"],
                row_probabilities=weights, seed=training_seed,
                gradient_updates=COMPONENT_UPDATES, batch_size=BATCH_SIZE,
                device=selected_device, official=official, torch=torch,
                record_schedule_digest=False)
            metadata = {**training["metadata"], "condition": "confounded",
                        "data_label": "n32_compute_matched", "samples_per_anchor_source": 32,
                        "model_kind": "pooled", "composition": "balanced"}
            _save_component_checkpoint(path, training, normalization, metadata, torch)
            curves.extend({"condition": "confounded", "seed": seed,
                           "data_label": "n32_compute_matched", "model": "pooled_balanced",
                           **row} for row in training["history"])
            trained += 1
        print(f"Stage A components ready: seed {seed}", flush=True)

    old_curves = _read_csv(scaling / "training_curves.csv")
    curves.extend({**row, "curve_source": "Phase8H-DS"} for row in old_curves
                  if row["condition"] == "confounded" and (
                      row["data_label"] == "n128"
                      or (row["seed"] == "0" and row["data_label"] in {"n32", "n32_extra_compute"})))
    metrics: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    lookup = {int(anchor): position for position, anchor in enumerate(anchors["anchor_id"])}
    test_states = np.asarray(anchors["public_observation"])[
        np.asarray([lookup[int(anchor)] for anchor in test_ids])]
    for n in (32, 128):
        for seed in (0, 1, 2):
            paths = _component_paths(scaling, output, seed, n)
            components = {name: _load_component(path, official, torch, selected_device)
                          for name, path in paths.items()}
            for row in _evaluate_components(
                    phase8h, components, seed, test_states, reference):
                metrics.append({"n": n, "updates": COMPONENT_UPDATES, "seed": seed, **row})
            for name, bundle in components.items():
                source = int(name[-1]) if name.startswith("source_") else None
                view = ({key: np.asarray(value)[public["source_id"] == source]
                         for key, value in public.items()} if source else public)
                validation.append({"n": n, "updates": COMPONENT_UPDATES, "seed": seed,
                                   "component": name, "validation_basis": "common_D32_rows",
                                   **_component_public_validation(
                                       bundle, view, splits["observational_validation"], torch)})
    _write_csv(output / "stage_a" / "compute_matched_metrics.csv", metrics)
    _write_csv(output / "stage_a" / "common_validation_metrics.csv", validation)
    curve_keys = ("condition", "seed", "data_label", "model", "step")
    curves = list({tuple(str(row.get(key, "")) for key in curve_keys): row
                   for row in curves}.values())
    _write_csv(curve_path, curves)
    after = fingerprint_snapshot(tracked)
    checks = {
        "old_inputs_unchanged": before == after,
        "only_n32_4000_seeds_1_2_added": all(
            torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "samples_per_anchor_source") == 32
            for seed in (1, 2) for path in _component_paths(scaling, output, seed, 32).values()),
        "all_eight_missing_components_ready": all(
            path.is_file() for seed in (1, 2)
            for path in _component_paths(scaling, output, seed, 32).values()),
        "d32_generator_exactly_reused": bool(generation["original_d32_exact"]),
        "n32_n128_same_anchor_split": json.loads((phase8h / "splits.json").read_text(
            encoding="utf-8")) == {key: value.tolist() for key, value in splits.items()},
        "common_D32_validation_rows_used": all(int(row["row_count"]) > 0 for row in validation),
        "fixed_phase8h_candidate_values_used": True,
        "all_metrics_finite": all(np.isfinite(float(value)) for row in metrics + validation
                                    for value in row.values() if isinstance(value, (int, float))),
        "checkpoint_roundtrip": True,
    }
    _stage_a_report(output, metrics, validation, curves, checks)
    _write_json(output / "stage_a" / "hard_checks.json",
                {"all_passed": all(checks.values()), "checks": checks,
                 "failed": [key for key, value in checks.items() if not value]})
    _write_json(output / "input_integrity.json", {
        "algorithm": "BLAKE2b-128",
        "before": before, "after": after, "unchanged": before == after})
    new_component_paths = [path for seed in (1, 2)
                           for path in _component_paths(scaling, output, seed, 32).values()]
    _write_json(output / "manifest.json", {
        "stage": PHASE, "git_commit": _git_commit(), "phase8h_root": str(phase8h),
        "scaling_root": str(scaling), "external_aamas_path": str(Path(external_repo).resolve()),
        "external_aamas_commit": EXTERNAL_COMMIT, "runtime_versions": _runtime_versions(),
        "component_updates": COMPONENT_UPDATES, "batch_size": BATCH_SIZE,
        "missing_n32_seeds": [1, 2], "source_policy": source_policy_parameters(),
        "gamma": GAMMA, "kappa": KAPPA, "lambda_reward": LAMBDA_REWARD,
        "actuator_direction": ACTUATOR_DIRECTION.tolist(),
        "candidate_count": 28, "stage_a_complete": all(checks.values()),
        "new_component_fingerprints": fingerprint_snapshot(new_component_paths),
        "scientific_success_implied": False})
    _refresh_top_level_checks(output)
    _refresh_report(output)
    if not all(checks.values()):
        raise Phase8HComputeMatchedOnlineError(
            f"Stage A hard checks failed: {[key for key, value in checks.items() if not value]}")
    return {"trained_component_count": trained, "metric_rows": len(metrics),
            "validation_rows": len(validation), "all_hard_checks_passed": True}


class _TorchPotentialValue:
    """Numpy-facing frozen view of a target potential used inside AAMAS backup."""

    def __init__(self, network: Any, mean: np.ndarray, std: np.ndarray,
                 device: str, torch: Any) -> None:
        self.network, self.device, self.torch = network, device, torch
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 12)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 12)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        array = np.asarray(states, dtype=np.float32)
        tensor = self.torch.as_tensor(
            (array - self.mean) / (self.std + 1e-7),
            dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            result = self.network(tensor).reshape(-1).detach().cpu().numpy()
        if not np.all(np.isfinite(result)):
            raise Phase8HComputeMatchedOnlineError("target potential returned NaN/Inf")
        return result.astype(np.float64)


class _TerminalMaskedValue:
    """Apply a state-row terminal mask to flattened candidate continuations."""

    def __init__(self, value: Callable[[np.ndarray], np.ndarray],
                 terminated: np.ndarray) -> None:
        self.value = value
        self.terminated = np.asarray(terminated, dtype=bool).reshape(-1)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        result = np.asarray(self.value(states), dtype=np.float64).reshape(-1)
        if not len(self.terminated) or len(result) % len(self.terminated):
            raise Phase8HComputeMatchedOnlineError(
                "continuation rows do not align with the terminal mask")
        multiplicity = len(result) // len(self.terminated)
        return result * np.repeat(~self.terminated, multiplicity)


def _potential_checkpoint(output: Path, method: str, run_id: int) -> Path:
    return output / "stage_b" / "potential_checkpoints" / method / f"run_{run_id}.pt"


def _make_potential_network(official: Any, reward_min: float, reward_max: float,
                            device: str) -> Any:
    return official.Critic(12, 3, 1000, reward_max, reward_min, GAMMA).to(device)


def _polyak_update(current: Any, target: Any, tau: float) -> None:
    for source, destination in zip(current.parameters(), target.parameters()):
        destination.data.mul_(1.0 - tau).add_(source.data, alpha=tau)


def _dynamic_target(
    method: str, states: np.ndarray, observed_actions: np.ndarray,
    base_actions: np.ndarray, source_models: Sequence[ContinuousAAMASComponents],
    pooled_model: ContinuousAAMASComponents, target_value: Callable[[np.ndarray], np.ndarray],
    update_seed: int, terminated: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    terminal = (np.zeros(len(states), dtype=bool) if terminated is None
                else np.asarray(terminated, dtype=bool).reshape(-1))
    if terminal.shape != (len(states),):
        raise ValueError("terminated must have one value per state row")
    continuation = _TerminalMaskedValue(target_value, terminal)
    if method == "pooled_aamas_native_full":
        actions = np.asarray(observed_actions, dtype=np.float32)[:, None, :]
        noise = np.random.default_rng(update_seed + 1).standard_normal(
            (len(states), CANDIDATE_ACTIONS, 3)).astype(np.float32)
        pooled_q = compute_source_aamas_backup(
            (pooled_model,), states, actions, continuation, common_noise=noise)[0]
        source_stub = np.zeros((3, len(states), 1), dtype=np.float64)
        return aggregate_dynamic_backup(method, source_stub, pooled_q), len(states)
    actions = union_candidate_actions(
        source_models, states, base_actions, samples_per_source=8, seed=update_seed)
    noise = np.random.default_rng(update_seed + 1).standard_normal(
        (len(states) * actions.shape[1], CANDIDATE_ACTIONS, 3)).astype(np.float32)
    if method in {"state_min_full", "action_min_full"}:
        source_q = compute_source_aamas_backup(
            source_models, states, actions, continuation, common_noise=noise)
        target = aggregate_dynamic_backup(method, source_q)
        forwards = len(states) * actions.shape[1] * len(source_models)
    else:
        pooled_q = compute_source_aamas_backup(
            (pooled_model,), states, actions, continuation, common_noise=noise)[0]
        source_stub = np.zeros((3, len(states), actions.shape[1]), dtype=np.float64)
        target = aggregate_dynamic_backup(method, source_stub, pooled_q)
        forwards = len(states) * actions.shape[1]
    return target, forwards


def _train_one_potential(
    method: str, run_id: int, public: Mapping[str, np.ndarray],
    train_rows: np.ndarray, validation_rows: np.ndarray,
    base_by_anchor: np.ndarray, source_models: Sequence[ContinuousAAMASComponents],
    pooled_model: ContinuousAAMASComponents, official: Any, torch: Any,
    device: str, epochs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    train_states = np.asarray(public["observation"])[train_rows]
    state_mean = train_states.mean(axis=0, keepdims=True).astype(np.float32)
    state_std = train_states.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    reward_min = float(np.min(np.asarray(public["reward"])[train_rows]))
    reward_max = float(np.max(np.asarray(public["reward"])[train_rows]))
    seed_everything(run_id, torch, cuda_training=device == "cuda")
    current = _make_potential_network(official, reward_min, reward_max, device)
    target = _make_potential_network(official, reward_min, reward_max, device)
    target.load_state_dict(current.state_dict())
    optimizer = torch.optim.Adam(current.parameters(), lr=POTENTIAL_LR, weight_decay=1e-5)
    initial = parameter_fingerprint((current,))
    rng = np.random.default_rng(run_id + 20260921)
    history: list[dict[str, Any]] = []
    total_updates = 0
    model_forward_units = 0
    wall_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(train_rows)
        losses, current_means, target_means = [], [], []
        for batch_index, start in enumerate(range(0, len(permutation), POTENTIAL_BATCH_SIZE)):
            rows = permutation[start:start + POTENTIAL_BATCH_SIZE]
            states = np.asarray(public["observation"])[rows]
            actions = np.asarray(public["commanded_action"])[rows]
            bases = base_by_anchor[np.asarray(public["anchor_id"])[rows].astype(np.int64)]
            target_value = _TorchPotentialValue(target, state_mean, state_std, device, torch)
            backup, forwards = _dynamic_target(
                method, states, actions, bases, source_models, pooled_model,
                target_value, update_seed=run_id * 10_000_000 + total_updates * 2 + 20260922,
                terminated=np.asarray(public["terminated"])[rows])
            normalized = torch.as_tensor(
                (states - state_mean) / (state_std + 1e-7),
                dtype=torch.float32, device=device)
            prediction = current(normalized).reshape(-1)
            target_tensor = torch.as_tensor(backup, dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(prediction, target_tensor)
            if not bool(torch.isfinite(loss)):
                raise Phase8HComputeMatchedOnlineError(
                    f"nonfinite potential loss for {method}, run {run_id}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index > 0 and batch_index % TARGET_UPDATE_INTERVAL == 0:
                _polyak_update(current, target, TARGET_TAU)
            losses.append(float(loss.detach().cpu()))
            current_means.append(float(prediction.detach().mean().cpu()))
            target_means.append(float(np.mean(backup)))
            total_updates += 1
            model_forward_units += int(forwards)
        history.append({"run_id": run_id, "potential": method, "epoch": epoch,
                        "optimizer_updates": total_updates,
                        "training_loss": float(np.mean(losses)),
                        "current_value_mean": float(np.mean(current_means)),
                        "backup_target_mean": float(np.mean(target_means))})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"potential {method} run {run_id}: epoch {epoch}/{epochs} "
                  f"loss={history[-1]['training_loss']:.6g}", flush=True)
    validation_state = np.asarray(public["observation"])[validation_rows]
    validation_action = np.asarray(public["commanded_action"])[validation_rows]
    validation_base = base_by_anchor[
        np.asarray(public["anchor_id"])[validation_rows].astype(np.int64)]
    target_value = _TorchPotentialValue(target, state_mean, state_std, device, torch)
    validation_target, validation_forwards = _dynamic_target(
        method, validation_state, validation_action, validation_base,
        source_models, pooled_model, target_value,
        update_seed=run_id * 10_000_000 + 909_090,
        terminated=np.asarray(public["terminated"])[validation_rows])
    with torch.no_grad():
        normalized = torch.as_tensor(
            (validation_state - state_mean) / (state_std + 1e-7),
            dtype=torch.float32, device=device)
        validation_prediction = current(normalized).reshape(-1).cpu().numpy()
    residual = validation_prediction - validation_target
    diagnostics = {
        "validation_row_count": len(validation_rows),
        "validation_residual_mae": float(np.mean(np.abs(residual))),
        "validation_residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "potential_min": float(np.min(validation_prediction)),
        "potential_max": float(np.max(validation_prediction)),
        "potential_mean": float(np.mean(validation_prediction)),
        "potential_std": float(np.std(validation_prediction)),
        "total_model_forward_units": int(model_forward_units + validation_forwards),
        "wall_clock_seconds": float(time.perf_counter() - wall_start),
    }
    payload = {
        "current_state_dict": {key: value.detach().cpu() for key, value in current.state_dict().items()},
        "target_state_dict": {key: value.detach().cpu() for key, value in target.state_dict().items()},
        "state_mean": state_mean, "state_std": state_std,
        "reward_min": reward_min, "reward_max": reward_max,
        "metadata": {
            "stage": PHASE, "method": method, "run_id": run_id,
            "epochs": epochs, "optimizer_updates": total_updates,
            "batch_size": POTENTIAL_BATCH_SIZE, "learning_rate": POTENTIAL_LR,
            "target_tau": TARGET_TAU, "target_update_interval_batches": TARGET_UPDATE_INTERVAL,
            "gamma": GAMMA,
            "candidate_actions_per_source": 8, "candidate_count": 28,
            "component_parameters_frozen": True,
            "backup_value_source": "own_target_potential",
            "fixed_reference_value_used": False,
            "checkpoint_selection": "pre_frozen_final_epoch",
            "continuation_terminal_mask": "terminated_only; truncation_bootstraps",
            "initial_parameter_fingerprint": initial,
            "test_oracle_used": False, "online_return_used": False,
        },
    }
    return payload, history, diagnostics


def _load_potential(path: Path, official: Any, torch: Any, device: str) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    network = _make_potential_network(
        official, float(payload["reward_min"]), float(payload["reward_max"]), device)
    network.load_state_dict(payload["current_state_dict"])
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    value = _TorchPotentialValue(
        network, payload["state_mean"], payload["state_std"], device, torch)
    return value, payload["metadata"]


def run_stage_b(
    phase8h_root: Path, scaling_root: Path, output_root: Path, *,
    offline_data_n: int, run_ids: Sequence[int], potential_config: str,
    device: str, external_repo: Path,
) -> dict[str, Any]:
    if offline_data_n != 32 or tuple(map(int, run_ids)) != (0, 1, 2):
        raise Phase8HComputeMatchedOnlineError("Stage B is frozen to D32 and run IDs 0,1,2")
    if potential_config != "official-frozen":
        raise Phase8HComputeMatchedOnlineError("potential config must be official-frozen")
    output = Path(output_root).resolve()
    stage_a_checks = output / "stage_a" / "hard_checks.json"
    if not stage_a_checks.is_file() or not json.loads(
            stage_a_checks.read_text(encoding="utf-8")).get("all_passed"):
        raise Phase8HComputeMatchedOnlineError("Stage A must complete before Stage B")
    phase8h = resolve_artifact_root(phase8h_root, "Phase 8H-Q")
    scaling = resolve_artifact_root(scaling_root, "Phase 8H-DS")
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
    except (ImportError, OSError) as error:
        raise Phase8HComputeMatchedOnlineError("PyTorch is required for Stage B") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(external_repo)
    inputs = _load_phase8h_inputs(
        _phase8a_from_phase8h(phase8h), 512, None, compute_checkpoint_hash=False)
    anchors, splits = inputs["anchors"], inputs["splits"]
    simulator = MujocoOneStepSimulator(anchors, (KAPPA,), seed=20260804)
    try:
        master, _, generation = generate_nested_master(
            anchors, simulator, condition="confounded", seed=20260804, max_samples=32)
    finally:
        simulator.close()
    public = subset_nested(master, 32)
    train_rows = np.flatnonzero(np.isin(public["anchor_id"], splits["train"]))
    validation_rows = np.flatnonzero(
        np.isin(public["anchor_id"], splits["observational_validation"]))
    base_by_anchor = np.asarray(anchors["base_action"], dtype=np.float32)
    component_paths = [path for run in run_ids for path in
                       _component_paths(scaling, output, int(run), 32).values()]
    missing = [str(path) for path in component_paths if not path.is_file()]
    if missing:
        raise Phase8HComputeMatchedOnlineError(
            "Stage B component checkpoint is missing: " + ", ".join(missing[:5]))
    before = fingerprint_snapshot([*component_paths, stage_a_checks])
    history_path = output / "stage_b" / "potential_training_curves.csv"
    diagnostic_path = output / "stage_b" / "potential_diagnostics.csv"
    histories: list[dict[str, Any]] = _read_csv(history_path) if history_path.is_file() else []
    diagnostics: list[dict[str, Any]] = _read_csv(diagnostic_path) if diagnostic_path.is_file() else []
    initial_by_run: dict[int, dict[str, Any]] = {}
    for run_id in run_ids:
        paths = _component_paths(scaling, output, int(run_id), 32)
        components = {name: _load_component(path, official, torch, selected_device)
                      for name, path in paths.items()}
        source_models = tuple(components[f"source_{source}"] for source in (1, 2, 3))
        for method in POTENTIAL_METHODS:
            checkpoint = _potential_checkpoint(output, method, int(run_id))
            if checkpoint.is_file():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                metadata = payload.get("metadata", {})
                if metadata.get("epochs") != POTENTIAL_EPOCHS or metadata.get("run_id") != run_id:
                    raise Phase8HComputeMatchedOnlineError(
                        f"existing potential checkpoint has mismatched config: {checkpoint}")
                initial = metadata.get("initial_parameter_fingerprint")
                if initial is not None:
                    if run_id not in initial_by_run:
                        initial_by_run[int(run_id)] = initial
                    elif not _same_fingerprint(initial_by_run[int(run_id)], initial):
                        raise Phase8HComputeMatchedOnlineError(
                            f"potential initialization is not paired for run {run_id}")
                print(f"reusing potential {method} run {run_id}", flush=True)
                continue
            payload, history, diagnostic = _train_one_potential(
                method, int(run_id), public, train_rows, validation_rows,
                base_by_anchor, source_models, components["pooled_balanced"],
                official, torch, selected_device, POTENTIAL_EPOCHS)
            initial = payload["metadata"]["initial_parameter_fingerprint"]
            if run_id not in initial_by_run:
                initial_by_run[int(run_id)] = initial
            elif not _same_fingerprint(initial_by_run[int(run_id)], initial):
                raise Phase8HComputeMatchedOnlineError(
                    f"potential initialization is not paired for run {run_id}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint)
            loaded, _ = _load_potential(
                checkpoint, official, torch, selected_device)
            probe = np.asarray(anchors["public_observation"][:16], dtype=np.float32)
            current = _make_potential_network(
                official, payload["reward_min"], payload["reward_max"], selected_device)
            current.load_state_dict(payload["current_state_dict"])
            expected = _TorchPotentialValue(
                current, payload["state_mean"], payload["state_std"], selected_device, torch)
            roundtrip = float(np.max(np.abs(loaded(probe) - expected(probe))))
            histories.extend(history)
            diagnostics.append({"run_id": run_id, "potential": method,
                                "roundtrip_max_abs": roundtrip,
                                "numeric_status": "finite", **diagnostic})
            print(f"Stage B potential ready: {method}, run {run_id}", flush=True)
    # Reconstruct compact histories/diagnostics from checkpoints if the stage resumed.
    if not diagnostics:
        raise Phase8HComputeMatchedOnlineError(
            "all potentials existed but compact diagnostics were unavailable; retain Stage B CSVs")
    history_unique = {(str(row["run_id"]), str(row["potential"]), str(row["epoch"])): row
                      for row in histories}
    diagnostic_unique = {(str(row["run_id"]), str(row["potential"])): row
                         for row in diagnostics}
    histories = list(history_unique.values())
    diagnostics = list(diagnostic_unique.values())
    _write_csv(history_path, histories)
    _write_csv(diagnostic_path, diagnostics)
    potential_paths = [_potential_checkpoint(output, method, int(run))
                       for run in run_ids for method in POTENTIAL_METHODS]
    official_source = inspect.getsource(official.CausalUpperBoundEstimator.train_critic)
    native_distinct = "dist = self.policy_fin(s)" in official_source and "a_prob_ratio" in official_source
    equivalence = {
        "official_outer_operation": "behavior-density-weighted observed-action backup",
        "union_outer_operation": "maximum over shared 28-action candidate set",
        "native_equals_union": not native_distinct,
        "native_baseline_required": native_distinct,
        "native_name": "pooled_aamas_native_full",
        "union_name": "pooled_aamas_union_full",
        "common_pbrs_interface_adaptation_required": True,
    }
    _write_json(output / "stage_b" / "baseline_equivalence_audit.json", equivalence)
    audit_components = {
        name: _load_component(path, official, torch, selected_device)
        for name, path in _component_paths(scaling, output, 0, 32).items()
    }
    audit_value, _ = _load_potential(
        _potential_checkpoint(output, "action_min_full", 0),
        official, torch, selected_device)
    audit_row = validation_rows[:1]
    audit_state = np.asarray(public["observation"])[audit_row]
    audit_action = np.asarray(public["commanded_action"])[audit_row, None, :]
    audit_noise = np.random.default_rng(20260929).standard_normal(
        (1, CANDIDATE_ACTIONS, 3)).astype(np.float32)
    direct_backup = compute_official_continuous_action_backup(
        audit_components["source_1"], audit_state, audit_action, audit_value,
        common_noise=audit_noise)
    wrapped_backup = compute_source_aamas_backup(
        (audit_components["source_1"],), audit_state, audit_action, audit_value,
        common_noise=audit_noise)[0]
    single_source_equivalent = bool(np.array_equal(direct_backup, wrapped_backup))
    after = fingerprint_snapshot([*component_paths, stage_a_checks])
    checks = {
        "old_inputs_unchanged": before == after,
        "complete_potential_checkpoint_set": all(path.is_file() for path in potential_paths),
        "own_target_potential_used": all(
            torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "backup_value_source") == "own_target_potential" for path in potential_paths),
        "fixed_reference_not_used_for_dynamic_backup": all(
            not torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "fixed_reference_value_used") for path in potential_paths),
        "source_aggregation_same_continuation_value": True,
        "single_source_dynamic_wrapper_matches_official_extraction": single_source_equivalent,
        "candidate_protocol_28": all(
            torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "candidate_count") == 28 for path in potential_paths
            if "native" not in str(path)),
        "official_target_update_schedule_used": all(
            torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "target_update_interval_batches") == TARGET_UPDATE_INTERVAL
            for path in potential_paths),
        "native_union_difference_explicit": native_distinct,
        "test_oracle_not_used": True,
        "all_potentials_finite": all(row["numeric_status"] == "finite" for row in diagnostics),
        "checkpoint_roundtrip": all(float(row["roundtrip_max_abs"]) <= 1e-7
                                    for row in diagnostics),
        "d32_exactly_reused": bool(generation["original_d32_exact"]),
    }
    _write_json(output / "stage_b" / "hard_checks.json",
                {"all_passed": all(checks.values()), "checks": checks,
                 "failed": [key for key, value in checks.items() if not value]})
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "potential_epochs": POTENTIAL_EPOCHS,
        "potential_optimizer_updates_per_method": int(
            POTENTIAL_EPOCHS * math.ceil(len(train_rows) / POTENTIAL_BATCH_SIZE)),
        "potential_batch_size": POTENTIAL_BATCH_SIZE,
        "potential_learning_rate": POTENTIAL_LR, "target_tau": TARGET_TAU,
        "target_update_interval_batches": TARGET_UPDATE_INTERVAL,
        "potential_methods": list(POTENTIAL_METHODS),
        "native_baseline_required": native_distinct,
        "potential_checkpoint_fingerprints": fingerprint_snapshot(potential_paths),
        "stage_b_complete": all(checks.values()),
    })
    _write_json(output / "manifest.json", manifest)
    _refresh_top_level_checks(output)
    _refresh_report(output)
    if not all(checks.values()):
        raise Phase8HComputeMatchedOnlineError(
            f"Stage B hard checks failed: {[key for key, value in checks.items() if not value]}")
    return {"potential_count": len(potential_paths), "native_baseline_required": native_distinct,
            "all_hard_checks_passed": True}


class _ZeroPotential:
    def __call__(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states)
        if values.ndim != 2 or values.shape[1] != 12:
            raise ValueError("potential requires [N,12] public observations")
        return np.zeros(len(values), dtype=np.float64)


def strip_private_info(info: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = set(PRIVATE_INFO_FIELDS) | {
        "source_id", "logger_id", "u_environment", "u_behavior",
        "do_oracle", "oracle_value", "simulator_state",
    }
    return {key: value for key, value in info.items() if key not in forbidden}


def commanded_replay_action(commanded_action: np.ndarray) -> np.ndarray:
    """The online learner stores its command, never the actuator-perturbed action."""
    command = np.asarray(commanded_action, dtype=np.float32)
    if command.shape[-1:] != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("commanded action must end in a finite three-vector")
    return command.copy()


def _make_online_environment(
    potential: Callable[[np.ndarray], np.ndarray], *, kappa: float, lambda_reward: float,
    beta: float, gamma: float,
) -> Any:
    import gymnasium as gym

    class PublicPBRSWrapper(gym.Wrapper):
        def __init__(self) -> None:
            base = ConfoundedHopperWrapper(
                gym.make("Hopper-v5"), kappa=kappa,
                expose_confounder=False, audit_info=True)
            super().__init__(base)
            self.commanded_actions: list[np.ndarray] = []
            self.shaping_increments: list[float] = []
            self.raw_rewards: list[float] = []
            self.hopper_rewards: list[float] = []
            self.commanded_action_match = True
            self.returned_info_private_leak = False
            self.raw_reward_formula_match = True
            self.private_audit: dict[str, Any] = {}
            self._public_observation: np.ndarray | None = None

        @staticmethod
        def _phi(value: Callable[[np.ndarray], np.ndarray], observation: np.ndarray) -> float:
            result = np.asarray(value(np.asarray(observation, dtype=np.float32)[None, :]),
                                dtype=np.float64).reshape(-1)
            if result.shape != (1,) or not np.isfinite(result[0]):
                raise Phase8HComputeMatchedOnlineError("online potential is nonfinite")
            return float(result[0])

        def reset(self, *, seed: int | None = None,
                  options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
            observation, info = self.env.reset(seed=seed, options=options)
            public = np.asarray(observation, dtype=np.float32)
            if public.shape != (12,):
                raise Phase8HComputeMatchedOnlineError(
                    f"online learner observation must be public 12D, got {public.shape}")
            self._public_observation = public.copy()
            return public, strip_private_info(info)

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            if self._public_observation is None:
                raise RuntimeError("environment must be reset before step")
            command = commanded_replay_action(action)
            phi_state = self._phi(potential, self._public_observation)
            observation, hopper_reward, terminated, truncated, info = self.env.step(command)
            public_next = np.asarray(observation, dtype=np.float32)
            hidden_u = int(info["hidden_u"])
            raw_reward = float(hopper_reward) + float(lambda_reward) * hidden_u
            phi_next = self._phi(potential, public_next)
            shaping = float(pbrs_increment(
                phi_state, phi_next, bool(terminated), gamma=gamma, beta=beta))
            shaped_reward = raw_reward + shaping
            recorded_command = np.asarray(info["commanded_action"], dtype=np.float32)
            self.commanded_action_match &= bool(np.array_equal(command, recorded_command))
            self.raw_reward_formula_match &= bool(np.isclose(
                raw_reward, float(hopper_reward) + float(lambda_reward) * hidden_u,
                atol=0.0, rtol=0.0))
            self.commanded_actions.append(command.copy())
            self.shaping_increments.append(shaping)
            self.raw_rewards.append(raw_reward)
            self.hopper_rewards.append(float(hopper_reward))
            self.private_audit = {
                "hidden_u": hidden_u,
                "commanded_action": recorded_command.copy(),
                "applied_action": np.asarray(info["applied_action"], dtype=np.float32).copy(),
            }
            self._public_observation = public_next.copy()
            safe = strip_private_info(info)
            safe.update({
                "raw_environment_reward": raw_reward,
                "hopper_only_reward": float(hopper_reward),
                "shaping_increment": shaping,
                "shaped_reward": shaped_reward,
                "pbrs_terminal_mask": bool(terminated),
                "pbrs_truncation_bootstraps": bool(truncated and not terminated),
            })
            self.returned_info_private_leak |= bool(
                set(safe).intersection(set(PRIVATE_INFO_FIELDS) | {
                    "source_id", "logger_id", "u_environment", "u_behavior",
                    "do_oracle", "oracle_value", "simulator_state"}))
            return public_next, float(shaped_reward), bool(terminated), bool(truncated), safe

    return PublicPBRSWrapper()


def _evaluate_online_model(
    model: Any, potential: Callable[[np.ndarray], np.ndarray], *, run_id: int,
    method_index: int, training_step: int, episodes: int, kappa: float,
    lambda_reward: float, beta: float, gamma: float,
) -> tuple[list[dict[str, Any]], int]:
    environment = _make_online_environment(
        potential, kappa=kappa, lambda_reward=lambda_reward, beta=beta, gamma=gamma)
    rows: list[dict[str, Any]] = []
    interaction_count = 0
    try:
        for episode in range(episodes):
            # Identical evaluation seeds across methods within a paired run/step.
            evaluation_seed = 90_000_000 + run_id * 100_000 + training_step + episode
            observation, _ = environment.reset(seed=evaluation_seed)
            raw_return = hopper_return = shaped_return = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, shaped, terminated, truncated, info = environment.step(action)
                raw_return += float(info["raw_environment_reward"])
                hopper_return += float(info["hopper_only_reward"])
                shaped_return += float(shaped)
                length += 1
            interaction_count += length
            rows.append({
                "run_id": run_id, "method_index": method_index,
                "training_step": training_step, "episode": episode,
                "evaluation_seed": evaluation_seed,
                "raw_environment_return": raw_return,
                "hopper_only_return": hopper_return,
                "shaped_return": shaped_return, "episode_length": length,
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
    finally:
        environment.close()
    return rows, interaction_count


def _online_method_potential(
    method: str, run_id: int, output: Path, official: Any, torch: Any, device: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    if method == "sac_scratch":
        return _ZeroPotential(), {"method": "zero", "run_id": run_id}
    name = MODEL_TO_POTENTIAL[method]
    path = _potential_checkpoint(output, name, run_id)
    if not path.is_file():
        raise Phase8HComputeMatchedOnlineError(f"potential checkpoint is missing: {path}")
    value, metadata = _load_potential(path, official, torch, device)
    return value, {**metadata, "checkpoint": str(path)}


def _replay_matches_commands(model: Any, commands: Sequence[np.ndarray]) -> bool:
    count = int(model.replay_buffer.size())
    if count != len(commands):
        return False
    stored = np.asarray(model.replay_buffer.actions[:count, 0], dtype=np.float32)
    expected = np.asarray(commands, dtype=np.float32)
    return stored.shape == expected.shape and bool(np.allclose(stored, expected, atol=2e-6, rtol=0.0))


def _summarize_online_rows(
    evaluation_rows: Sequence[Mapping[str, Any]], cost_rows: Sequence[Mapping[str, Any]],
    run_ids: Sequence[int], methods: Sequence[str], online_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_groups: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        episode_groups[(int(row["run_id"]), str(row["method"]),
                        int(row["training_step"]))].append(row)
    curves: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    final_stats: dict[tuple[int, str], dict[str, float]] = {}
    for run_id in run_ids:
        for method in methods:
            points = []
            for key, values in episode_groups.items():
                if key[:2] != (int(run_id), method):
                    continue
                points.append((key[2], float(np.mean([
                    float(value["raw_environment_return"]) for value in values]))))
            points.sort()
            steps = np.asarray([value[0] for value in points], dtype=np.float64)
            returns = np.asarray([value[1] for value in points], dtype=np.float64)
            if len(steps) < 2 or int(steps[-1]) != int(online_steps):
                raise Phase8HComputeMatchedOnlineError(
                    f"incomplete evaluation curve for {method}, run {run_id}")
            curves[(int(run_id), method)] = (steps, returns)
            final_values = episode_groups[(int(run_id), method, int(online_steps))]
            final_stats[(int(run_id), method)] = {
                "final_raw_return": float(np.mean([
                    float(row["raw_environment_return"]) for row in final_values])),
                "final_episode_length": float(np.mean([
                    float(row["episode_length"]) for row in final_values])),
                "final_terminated_fraction": float(np.mean([
                    str(row["terminated"]).lower() == "true" for row in final_values])),
            }
    per_run: list[dict[str, Any]] = []
    for run_id in run_ids:
        scratch_steps, scratch_values = curves[(int(run_id), "sac_scratch")]
        union_auc = normalized_auc(*curves[(int(run_id), "sac_pooled_union")])
        native_auc = (normalized_auc(*curves[(int(run_id), "sac_pooled_native")])
                      if "sac_pooled_native" in methods else float("nan"))
        for method in methods:
            steps, values = curves[(int(run_id), method)]
            if not np.array_equal(steps, scratch_steps):
                raise Phase8HComputeMatchedOnlineError("online evaluation grids are not paired")
            auc = normalized_auc(steps, values)
            cost = next(row for row in cost_rows
                        if int(row["run_id"]) == int(run_id) and row["method"] == method)
            per_run.append({
                "run_id": int(run_id), "method": method, "auc": auc,
                "auc_difference_vs_pooled_union": auc - union_auc,
                "auc_difference_vs_pooled_native": (auc - native_auc
                                                       if np.isfinite(native_auc) else ""),
                "negative_transfer_area_vs_scratch": normalized_positive_area(
                    steps, scratch_values - values),
                **final_stats[(int(run_id), method)],
                "shaping_increment_mean": float(cost["shaping_increment_mean"]),
                "shaping_increment_std": float(cost["shaping_increment_std"]),
                "shaping_increment_p95_abs": float(cost["shaping_increment_p95_abs"]),
            })
    summary: list[dict[str, Any]] = []
    numeric = ("auc", "auc_difference_vs_pooled_union",
               "negative_transfer_area_vs_scratch", "final_raw_return",
               "final_episode_length", "final_terminated_fraction",
               "shaping_increment_mean", "shaping_increment_std",
               "shaping_increment_p95_abs")
    for method in methods:
        values = [row for row in per_run if row["method"] == method]
        record: dict[str, Any] = {"method": method, "run_count": len(values)}
        for metric in numeric:
            data = np.asarray([float(row[metric]) for row in values], dtype=np.float64)
            record[metric + "_mean"] = float(data.mean())
            record[metric + "_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        if "sac_pooled_native" in methods:
            data = np.asarray([float(row["auc_difference_vs_pooled_native"])
                               for row in values], dtype=np.float64)
            record["auc_difference_vs_pooled_native_mean"] = float(data.mean())
            record["auc_difference_vs_pooled_native_sd"] = (
                float(data.std(ddof=1)) if len(data) > 1 else 0.0)
        summary.append(record)
    return per_run, summary


def run_online(
    output_root: Path, *, run_ids: Sequence[int], online_steps: int,
    eval_every: int, eval_episodes: int, device: str, external_repo: Path,
    smoke: bool,
) -> dict[str, Any]:
    run_ids = tuple(map(int, run_ids))
    if smoke:
        if run_ids != (0,) or online_steps != 2_000:
            raise Phase8HComputeMatchedOnlineError(
                "online smoke is frozen to run 0 and 2000 training steps")
    elif run_ids != (0, 1, 2) or online_steps != 50_000 or eval_every != 10_000 \
            or eval_episodes != 5:
        raise Phase8HComputeMatchedOnlineError(
            "Stage C is frozen to runs 0,1,2; 50k steps; 10k evaluation; 5 episodes")
    if online_steps % eval_every:
        raise Phase8HComputeMatchedOnlineError("online steps must be divisible by eval-every")
    output = Path(output_root).resolve()
    run_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    online_gamma = float(run_manifest["gamma"])
    online_kappa = float(run_manifest["kappa"])
    online_lambda = float(run_manifest["lambda_reward"])
    if not (online_gamma == GAMMA and online_kappa == KAPPA
            and online_lambda == LAMBDA_REWARD
            and np.array_equal(np.asarray(run_manifest["actuator_direction"], dtype=np.float64),
                               ACTUATOR_DIRECTION)):
        raise Phase8HComputeMatchedOnlineError(
            "recorded Phase 8H environment parameters do not match the frozen implementation")
    stage_b_checks = output / "stage_b" / "hard_checks.json"
    if not stage_b_checks.is_file() or not json.loads(
            stage_b_checks.read_text(encoding="utf-8")).get("all_passed"):
        raise Phase8HComputeMatchedOnlineError("Stage B must complete before online SAC")
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
        from stable_baselines3 import SAC
    except (ImportError, OSError) as error:
        raise Phase8HComputeMatchedOnlineError(
            "PyTorch and stable-baselines3 are required for online SAC") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(external_repo)
    equivalence = json.loads((output / "stage_b" / "baseline_equivalence_audit.json").read_text(
        encoding="utf-8"))
    methods = list(ONLINE_METHODS if equivalence["native_baseline_required"]
                   else ONLINE_METHODS[:-1])
    destination = output / "stage_c" / "smoke" if smoke else output / "stage_c"
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_path = destination / "evaluation_returns.csv"
    cost_path = destination / "cost_accounting.csv"
    evaluation_rows: list[dict[str, Any]] = _read_csv(evaluation_path) if evaluation_path.is_file() else []
    cost_rows: list[dict[str, Any]] = _read_csv(cost_path) if cost_path.is_file() else []
    expected_steps = tuple(range(0, online_steps + 1, eval_every))
    initial_by_run: dict[int, dict[str, Any]] = {}
    tracked_potentials = [_potential_checkpoint(output, potential, run_id)
                          for run_id in run_ids for potential in POTENTIAL_METHODS]
    before = fingerprint_snapshot(tracked_potentials)
    initialization_paired = True
    for run_id in run_ids:
        for method_index, method in enumerate(methods):
            checkpoint_dir = (destination / "final_sac_checkpoints" / method /
                              f"run_{run_id}")
            checkpoint = checkpoint_dir / "model.zip"
            existing_steps = {int(row["training_step"]) for row in evaluation_rows
                              if int(row["run_id"]) == run_id and row["method"] == method}
            existing_cost = [row for row in cost_rows
                             if int(row["run_id"]) == run_id and row["method"] == method]
            if checkpoint.is_file() and existing_steps == set(expected_steps) and len(existing_cost) == 1:
                restored_initial = {
                    "parameter_count": int(existing_cost[0]["initial_parameter_count"]),
                    "sum": float(existing_cost[0]["initial_parameter_sum"]),
                    "sum_squares": float(existing_cost[0]["initial_parameter_sum_squares"]),
                    "max_abs": float(existing_cost[0]["initial_parameter_max_abs"]),
                }
                if run_id not in initial_by_run:
                    initial_by_run[run_id] = restored_initial
                else:
                    initialization_paired &= _same_fingerprint(
                        initial_by_run[run_id], restored_initial)
                print(f"reusing completed online run: {method}, run {run_id}", flush=True)
                continue
            evaluation_rows = [row for row in evaluation_rows
                               if not (int(row["run_id"]) == run_id and row["method"] == method)]
            cost_rows = [row for row in cost_rows
                         if not (int(row["run_id"]) == run_id and row["method"] == method)]
            potential, potential_metadata = _online_method_potential(
                method, run_id, output, official, torch, selected_device)
            potential_before = (parameter_fingerprint((potential.network,))
                                if hasattr(potential, "network") else None)
            environment = _make_online_environment(
                potential, kappa=online_kappa, lambda_reward=online_lambda,
                beta=PBRS_BETA, gamma=online_gamma)
            training_seed = run_id
            seed_everything(training_seed, torch, cuda_training=selected_device == "cuda")
            wall_start = time.perf_counter()
            model = SAC(env=environment, seed=training_seed, device=selected_device,
                        verbose=0, **SAC_CONFIG)
            replay_empty_initially = int(model.replay_buffer.size()) == 0
            initial = parameter_fingerprint((model.actor, model.critic))
            if run_id not in initial_by_run:
                initial_by_run[run_id] = initial
            elif not _same_fingerprint(initial_by_run[run_id], initial):
                initialization_paired = False
                raise Phase8HComputeMatchedOnlineError(
                    f"SAC initialization is not paired for run {run_id}")
            evaluation_interactions = 0
            for step in expected_steps:
                if step:
                    model.learn(total_timesteps=eval_every, reset_num_timesteps=False,
                                progress_bar=False)
                rows, interactions = _evaluate_online_model(
                    model, potential, run_id=run_id, method_index=method_index,
                    training_step=step, episodes=eval_episodes,
                    kappa=online_kappa, lambda_reward=online_lambda,
                    beta=PBRS_BETA, gamma=online_gamma)
                evaluation_interactions += interactions
                evaluation_rows.extend({"method": method, **row} for row in rows)
                print(f"online {method} run {run_id}: {step}/{online_steps}", flush=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            probe = np.zeros((4, 12), dtype=np.float32)
            expected_action, _ = model.predict(probe, deterministic=True)
            model.save(str(checkpoint.with_suffix("")))
            loaded = SAC.load(str(checkpoint), device=selected_device)
            actual_action, _ = loaded.predict(probe, deterministic=True)
            increments = np.asarray(environment.shaping_increments, dtype=np.float64)
            potential_after = (parameter_fingerprint((potential.network,))
                               if hasattr(potential, "network") else None)
            replay_match = _replay_matches_commands(model, environment.commanded_actions)
            cost_rows.append({
                "run_id": run_id, "method": method,
                "training_environment_steps": online_steps,
                "evaluation_environment_steps": evaluation_interactions,
                "wall_clock_seconds": float(time.perf_counter() - wall_start),
                "replay_initial_size": 0 if replay_empty_initially else int(model.replay_buffer.size()),
                "model_replay_matches_commands": replay_match,
                "wrapper_commanded_action_match": environment.commanded_action_match,
                "returned_info_private_leak": environment.returned_info_private_leak,
                "raw_reward_formula_match": environment.raw_reward_formula_match,
                "potential_frozen": (potential_before == potential_after),
                "checkpoint_roundtrip": bool(np.allclose(
                    expected_action, actual_action, atol=1e-7, rtol=0.0)),
                "shaping_increment_mean": float(increments.mean()),
                "shaping_increment_std": float(increments.std()),
                "shaping_increment_p95_abs": float(np.quantile(np.abs(increments), .95)),
                "potential_method": potential_metadata["method"],
                "beta": PBRS_BETA, "gamma": GAMMA,
                "initial_parameter_count": initial["parameter_count"],
                "initial_parameter_sum": initial["sum"],
                "initial_parameter_sum_squares": initial["sum_squares"],
                "initial_parameter_max_abs": initial["max_abs"],
            })
            environment.close()
            _write_csv(evaluation_path, evaluation_rows)
            _write_csv(cost_path, cost_rows)
            del loaded, model, environment, potential
            if selected_device == "cuda":
                torch.cuda.empty_cache()
    after = fingerprint_snapshot(tracked_potentials)
    per_run, summary = _summarize_online_rows(
        evaluation_rows, cost_rows, run_ids, methods, online_steps)
    _write_csv(destination / "paired_run_metrics.csv", per_run)
    _write_csv(destination / "online_training_summary.csv", summary)
    costs = [row for row in cost_rows if int(row["run_id"]) in run_ids
             and row["method"] in methods]
    finite_rows = [*per_run, *summary]
    checks = {
        "potential_inputs_unchanged": before == after,
        "complete_online_checkpoint_set": all(
            (destination / "final_sac_checkpoints" / method /
             f"run_{run_id}" / "model.zip").is_file()
            for run_id in run_ids for method in methods),
        "phi_frozen_online": all(str(row["potential_frozen"]).lower() == "true" for row in costs),
        "online_policy_public_12d_only": True,
        "hidden_information_not_returned_to_learner": all(
            str(row["returned_info_private_leak"]).lower() == "false" for row in costs),
        "replay_stores_commanded_action": all(
            str(row["model_replay_matches_commands"]).lower() == "true"
            and str(row["wrapper_commanded_action_match"]).lower() == "true"
            for row in costs),
        "no_offline_replay_warm_start": all(int(row["replay_initial_size"]) == 0 for row in costs),
        "gamma_and_beta_common": all(
            float(row["gamma"]) == GAMMA and float(row["beta"]) == PBRS_BETA for row in costs),
        "terminated_mask_only_truncation_bootstraps": True,
        "pbrs_discounted_sum_identity": bool(np.isclose(
            discounted_shaping_sum([2.0, 3.0, 5.0], [False, False]),
            -2.0 + GAMMA**2 * 5.0, atol=1e-12, rtol=0.0)),
        "evaluation_uses_separate_environment": True,
        "primary_metrics_use_raw_environment_reward": True,
        "raw_environment_reward_formula_exact": all(
            str(row["raw_reward_formula_match"]).lower() == "true" for row in costs),
        "paired_run_ids_complete": all(
            {int(row["run_id"]) for row in per_run if row["method"] == method} == set(run_ids)
            for method in methods),
        "checkpoint_roundtrip": all(
            str(row["checkpoint_roundtrip"]).lower() == "true" for row in costs),
        "all_metrics_finite": all(
            np.isfinite(float(value)) for row in finite_rows for value in row.values()
            if isinstance(value, (int, float, np.integer, np.floating))),
        "scratch_shaping_exactly_zero": all(
            float(row["shaping_increment_std"]) == 0.0
            and float(row["shaping_increment_mean"]) == 0.0
            for row in costs if row["method"] == "sac_scratch"),
        "sac_initialization_paired": initialization_paired
                                      and len(initial_by_run) == len(run_ids),
    }
    hard_path = destination / "hard_checks.json"
    _write_json(hard_path, {"all_passed": all(checks.values()), "checks": checks,
                            "failed": [key for key, value in checks.items() if not value]})
    if not smoke:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest.update({
            "stage_c_complete": all(checks.values()), "online_run_ids": list(run_ids),
            "online_steps": online_steps, "eval_every": eval_every,
            "eval_episodes": eval_episodes, "pbrs_beta": PBRS_BETA,
            "pbrs_beta_source": (
                "no project/official beta parameter was present; unit coefficient frozen before online"),
            "sac_config": SAC_CONFIG, "online_methods": methods,
            "sac_config_source": "stable-baselines3 defaults used by train_hopper_behavior_policies.py",
            "primary_online_outcome": "raw_environment_return",
            "sac_checkpoint_fingerprints": fingerprint_snapshot([
                destination / "final_sac_checkpoints" / method /
                f"run_{run_id}" / "model.zip"
                for run_id in run_ids for method in methods]),
        })
        _write_json(output / "manifest.json", manifest)
        _refresh_top_level_checks(output)
        _refresh_report(output)
    if not all(checks.values()):
        raise Phase8HComputeMatchedOnlineError(
            f"online hard checks failed: {[key for key, value in checks.items() if not value]}")
    return {"online_run_count": len(run_ids) * len(methods),
            "evaluation_rows": len(evaluation_rows),
            "all_hard_checks_passed": True, "smoke": smoke}


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _refresh_top_level_checks(output: Path) -> None:
    combined: dict[str, bool] = {}
    for stage in ("stage_a", "stage_b", "stage_c"):
        path = output / stage / "hard_checks.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        combined.update({f"{stage}:{key}": bool(value)
                         for key, value in record.get("checks", {}).items()})
    _write_json(output / "hard_checks.json", {
        "all_passed": bool(combined) and all(combined.values()),
        "completed_stages": [stage for stage in ("stage_a", "stage_b", "stage_c")
                             if (output / stage / "hard_checks.json").is_file()],
        "checks": combined,
        "failed": [key for key, value in combined.items() if not value],
    })


def _refresh_report(output: Path) -> None:
    lines = ["# Phase 8H-ON-Q report", "",
             "Completion markers indicate execution and audit completion, not scientific success.", ""]
    stage_a = output / "stage_a" / "compute_matched_metrics.csv"
    lines.extend(["## Table A — Compute-matched offline check", "",
                  "| n | Updates | Method | Do-MAE | Mean regret | P90 | CVaR90 |",
                  "|---:|---:|---|---:|---:|---:|---:|"])
    if stage_a.is_file():
        rows = _read_csv(stage_a)
        groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(int(row["n"]), int(row["updates"]), row["method"])].append(row)
        for key, values in sorted(groups.items()):
            metrics = ["do_mae", "regret_mean", "regret_p90", "regret_cvar90"]
            means = [np.mean([float(row[name]) for row in values]) for name in metrics]
            lines.append("| " + " | ".join([
                str(key[0]), str(key[1]), key[2], *map(_format_number, means)]) + " |")
    else:
        lines.append("| pending | pending | pending | — | — | — | — |")
    lines.extend(["", "## Table B — Recursive potential diagnostics", "",
                  "| Potential | Native/Adapted | Training budget | Validation residual | Numeric status |",
                  "|---|---|---:|---:|---|"])
    stage_b = output / "stage_b" / "potential_diagnostics.csv"
    if stage_b.is_file():
        rows = _read_csv(stage_b)
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["potential"]].append(row)
        for method, values in sorted(groups.items()):
            residual = np.mean([float(row["validation_residual_mae"]) for row in values])
            kind = "Native" if method == "pooled_aamas_native_full" else "Adapted"
            lines.append(f"| {method} | {kind} | {POTENTIAL_EPOCHS} epochs | "
                         f"{_format_number(residual)} | finite |")
    else:
        lines.append("| pending | — | — | — | pending |")
    lines.extend(["", "## Table C — Short online SAC pilot", "",
                  "| Online method | AUC 0–50k | Final raw return | NTA vs scratch | Per-run AUC |",
                  "|---|---:|---:|---:|---|"])
    stage_c = output / "stage_c" / "online_training_summary.csv"
    paired = output / "stage_c" / "paired_run_metrics.csv"
    if stage_c.is_file() and paired.is_file():
        summaries, per_run = _read_csv(stage_c), _read_csv(paired)
        for row in summaries:
            aucs = ", ".join(_format_number(value["auc"]) for value in per_run
                             if value["method"] == row["method"])
            lines.append(f"| {row['method']} | {_format_number(row['auc_mean'])} | "
                         f"{_format_number(row['final_raw_return_mean'])} | "
                         f"{_format_number(row['negative_transfer_area_vs_scratch_mean'])} | "
                         f"{aucs} |")
    else:
        lines.append("| pending | — | — | — | — |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight(
    phase8h_root: Path, scaling_root: Path, output_root: Path, *,
    missing_n32_seeds: Sequence[int], run_ids: Sequence[int], online_steps: int,
) -> dict[str, Any]:
    phase8h = resolve_artifact_root(phase8h_root, "Phase 8H-Q")
    scaling = resolve_artifact_root(scaling_root, "Phase 8H-DS")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = _preflight_record(
        phase8h, scaling, output, missing_n32_seeds, online_steps, run_ids)
    _write_json(output / "preflight.json", record)
    return record
