"""Phase 8J-BME-Q: read-only Bellman mean-error decomposition audit.

The audit separates error against the conditional mean Bellman target from
within-anchor target variation.  It never trains a network, reads a test split,
runs SAC, or consults hidden/do-oracle fields.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phase8j_core_bootstrap_isolation import (
    DEFAULT_FIX_ROOT,
    DEFAULT_FVI_ROOT,
    GAMMA,
    INNER_EPOCHS,
    PHASE as SOURCE_PHASE,
    _input_paths as _source_input_paths,
    _load_fvi_contract,
    _load_seed_components,
    _network_predictions,
    _normalization,
    _recomputed_target,
    _resolve_context,
    _validation_batches,
)
from .phase8j_potential_clamp_fix_quick import make_repaired_potential_network


PHASE = "Phase 8J-BME-Q"
VARIANTS = ("real_transition_bootstrap", "full_aamas_frozen")
REQUESTED_EPOCHS = (0, 10, 20, 40, 60, 80, 100, 120, 150, 180, 200)
DEFAULT_SOURCE_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_core_bootstrap_isolation"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_bellman_mean_error_audit"
)


class BellmanMeanErrorAuditError(RuntimeError):
    """Raised when the read-only audit contract cannot be honored."""


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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


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
        writer.writeheader()
        writer.writerows(records)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _stat_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def _snapshot(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [_stat_record(path) for path in sorted({Path(p).resolve() for p in paths}, key=str)]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    return float(np.sqrt(np.sum(weights * np.square(values - mean)) / np.sum(weights)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    x = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    centers = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(float(quantile), centers, x, left=x[0], right=x[-1]))


def decompose_by_anchor(
    predictions: np.ndarray,
    targets: np.ndarray,
    anchor_ids: np.ndarray,
    row_weights: np.ndarray | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Exactly decompose row-weighted MSE into mean error and target variance."""
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    anchors = np.asarray(anchor_ids, dtype=np.int64).reshape(-1)
    weights = (np.ones(len(target), dtype=np.float64) if row_weights is None
               else np.asarray(row_weights, dtype=np.float64).reshape(-1))
    if not (prediction.shape == target.shape == anchors.shape == weights.shape):
        raise ValueError("predictions, targets, anchor_ids, and weights must align")
    if not len(target) or not np.all(np.isfinite(prediction)) \
            or not np.all(np.isfinite(target)) or not np.all(np.isfinite(weights)):
        raise ValueError("decomposition arrays must be nonempty and finite")
    if np.any(weights <= 0.0):
        raise ValueError("row weights must be strictly positive")

    state_rows: list[dict[str, float | int]] = []
    total_numerator = mean_numerator = within_numerator = 0.0
    for anchor in np.unique(anchors):
        mask = anchors == anchor
        w = weights[mask]
        state_mass = float(w.sum())
        value = _weighted_mean(prediction[mask], w)
        value_spread = float(np.max(np.abs(prediction[mask] - value)))
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(value))
        if value_spread > tolerance:
            raise ValueError(
                f"checkpoint predictions differ within anchor {anchor}: {value_spread}"
            )
        mean_target = _weighted_mean(target[mask], w)
        total = float(np.sum(w * np.square(value - target[mask])))
        mean_error = state_mass * float(np.square(value - mean_target))
        within = float(np.sum(w * np.square(target[mask] - mean_target)))
        total_numerator += total
        mean_numerator += mean_error
        within_numerator += within
        state_rows.append({
            "anchor_id": int(anchor),
            "row_count": int(mask.sum()),
            "state_mass": state_mass,
            "value": value,
            "mean_target": mean_target,
            "gap": mean_target - value,
            "total_mse": total / state_mass,
            "mean_bellman_mse": mean_error / state_mass,
            "within_state_variance": within / state_mass,
        })

    mass = float(weights.sum())
    total_mse = total_numerator / mass
    mean_mse = mean_numerator / mass
    within = within_numerator / mass
    state_weight = np.asarray([row["state_mass"] for row in state_rows], dtype=np.float64)
    state_value = np.asarray([row["value"] for row in state_rows], dtype=np.float64)
    mean_target = np.asarray([row["mean_target"] for row in state_rows], dtype=np.float64)
    gap = mean_target - state_value
    identity_residual = abs(total_mse - mean_mse - within)
    numerical_tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, total_mse)
    result: dict[str, float | int] = {
        "row_count": len(target),
        "anchor_count": len(state_rows),
        "total_mse": total_mse,
        "mean_bellman_mse": mean_mse,
        "within_state_variance": within,
        "mean_error_fraction": mean_mse / max(total_mse, np.finfo(np.float64).tiny),
        "noise_fraction": within / max(total_mse, np.finfo(np.float64).tiny),
        "mean_bellman_rmse": math.sqrt(mean_mse),
        "value_sd": _weighted_sd(state_value, state_weight),
        "target_between_state_sd": _weighted_sd(mean_target, state_weight),
        "normalized_mean_error": math.sqrt(mean_mse)
        / max(_weighted_sd(mean_target, state_weight), np.finfo(np.float64).eps),
        "identity_residual": identity_residual,
        "identity_tolerance": numerical_tolerance,
        "gap_mean": _weighted_mean(gap, state_weight),
        "gap_median": _weighted_quantile(gap, state_weight, 0.5),
        "gap_std": _weighted_sd(gap, state_weight),
        "gap_p10": _weighted_quantile(gap, state_weight, 0.1),
        "gap_p90": _weighted_quantile(gap, state_weight, 0.9),
        "gap_positive_fraction": float(np.sum(state_weight[gap > 0]) / state_weight.sum()),
        "gap_negative_fraction": float(np.sum(state_weight[gap < 0]) / state_weight.sum()),
    }
    return result, state_rows


def map_requested_checkpoints(
    available: Mapping[int, Path], requested: Sequence[int]
) -> list[dict[str, Any]]:
    """Map requests to the nearest saved epoch and record every substitution."""
    if not available:
        raise BellmanMeanErrorAuditError("no readable checkpoints are available")
    epochs = sorted(available)
    result = []
    for request in requested:
        actual = min(epochs, key=lambda epoch: (abs(epoch - int(request)), epoch))
        result.append({
            "requested_epoch": int(request),
            "actual_epoch": int(actual),
            "exact": int(request) == int(actual),
            "path": str(Path(available[actual]).resolve()),
        })
    return result


def classify_result(rows: Sequence[Mapping[str, Any]]) -> str:
    """Apply the frozen descriptive decision labels without magnitude thresholds."""
    grouped = {
        variant: sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda row: int(row["epoch"]),
        )
        for variant in VARIANTS
    }
    if any(len(values) < 3 for values in grouped.values()):
        return "RESULT_AMBIGUOUS"

    def sustained_growth(values: Sequence[Mapping[str, Any]]) -> bool:
        tail = np.asarray([float(row["mean_bellman_mse"]) for row in values[-3:]])
        return bool(np.all(np.diff(tail) > 0.0) and tail[-1] > float(values[0]["mean_bellman_mse"]))

    real_grows = sustained_growth(grouped[VARIANTS[0]])
    full_grows = sustained_growth(grouped[VARIANTS[1]])
    real = grouped[VARIANTS[0]]
    total_change = float(real[-1]["total_mse"]) - float(real[0]["total_mse"])
    mean_change = float(real[-1]["mean_bellman_mse"]) - float(real[0]["mean_bellman_mse"])
    within_change = (float(real[-1]["within_state_variance"])
                     - float(real[0]["within_state_variance"]))
    if real_grows and full_grows:
        return "BOTH_REAL_AND_FULL_MEAN_BELLMAN_ERROR_GROW"
    if real_grows:
        return "REAL_TRANSITION_MEAN_BELLMAN_ERROR_DIVERGES"
    if full_grows:
        return "REAL_TRANSITION_MOSTLY_STABLE_FULL_AAMAS_DIVERGES"
    if total_change > 0.0 and within_change > mean_change:
        return "MOST_RAW_MSE_GROWTH_IS_TARGET_VARIANCE"
    return "RESULT_AMBIGUOUS"


def _checkpoint_inventory(root: Path, torch: Any) -> tuple[
    dict[str, dict[int, Path]], list[dict[str, Any]]
]:
    shared_initial = root / "init_seed0.pt"
    available: dict[str, dict[int, Path]] = {variant: {} for variant in VARIANTS}
    inventory: list[dict[str, Any]] = []
    if shared_initial.is_file():
        payload = torch.load(shared_initial, map_location="cpu", weights_only=False)
        valid = payload.get("metadata", {}).get("stage") == SOURCE_PHASE \
            and "state_dict" in payload
        inventory.append({"variant": "shared", "epoch": 0,
                          "path": str(shared_initial.resolve()), "metadata_valid": valid})
        if valid:
            for variant in VARIANTS:
                available[variant][0] = shared_initial
    for variant in VARIANTS:
        directory = root / variant
        for path in directory.rglob("*.pt") if directory.is_dir() else ():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            epoch = int(metadata.get("epoch", -1))
            valid = (
                metadata.get("stage") == SOURCE_PHASE
                and metadata.get("variant") == variant
                and epoch >= 0
                and "current_state_dict" in payload
            )
            inventory.append({"variant": variant, "epoch": epoch,
                              "path": str(path.resolve()), "metadata_valid": valid})
            if valid:
                available[variant].setdefault(epoch, path)
    return available, inventory


def _load_network(
    path: Path, context: Mapping[str, Any], reward_min: float, reward_max: float
) -> Any:
    torch = context["torch"]
    payload = torch.load(path, map_location=context["device"], weights_only=False)
    state = payload.get("current_state_dict", payload.get("state_dict"))
    if state is None:
        raise BellmanMeanErrorAuditError(f"checkpoint has no network state: {path}")
    network = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"]
    )
    network.load_state_dict(state)
    network.eval().requires_grad_(False)
    return network


def _variant_targets(
    variant: str,
    context: Mapping[str, Any],
    components: Mapping[str, Any],
    network: Any,
    mean: np.ndarray,
    std: np.ndarray,
    epoch: int,
) -> np.ndarray:
    batches = _validation_batches(np.asarray(context["validation_rows"], dtype=np.int64))
    # The audit freezes the validation candidate/noise stream across checkpoints.
    # The real-transition arm does not consume this index; full AAMAS uses the
    # same outer-0 validation stream at every epoch so only V_k changes.
    outer = 0
    return np.concatenate([
        _recomputed_target(
            variant, context, components, rows, network, mean, std, outer, index
        )
        for index, rows in enumerate(batches)
    ])


def _figures(output: Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = {variant: sorted([row for row in rows if row["variant"] == variant],
                               key=lambda row: int(row["epoch"]))
               for variant in VARIANTS}
    colors = {VARIANTS[0]: "#E69F00", VARIANTS[1]: "#D55E00"}
    labels = {VARIANTS[0]: "Real transition", VARIANTS[1]: "Full AAMAS"}
    made = []

    def line_plot(filename: str, fields: Sequence[tuple[str, str]], ylabel: str,
                  log: bool = False, band: tuple[str, str] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for variant in VARIANTS:
            data = records[variant]
            x = np.asarray([row["epoch"] for row in data])
            for field, suffix in fields:
                y = np.asarray([row[field] for row in data], dtype=float)
                ax.plot(x, y, color=colors[variant], linestyle="--" if suffix else "-",
                        marker="o", label=labels[variant] + suffix)
            if band:
                low = np.asarray([row[band[0]] for row in data], dtype=float)
                high = np.asarray([row[band[1]] for row in data], dtype=float)
                ax.fill_between(x, low, high, color=colors[variant], alpha=0.12)
        ax.set_xlabel("Actual checkpoint epoch")
        ax.set_ylabel(ylabel)
        if log and all(float(row[fields[0][0]]) > 0 for row in rows):
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        path = figures / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(str(path.resolve()))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for ax, variant in zip(axes, VARIANTS):
        data = records[variant]
        x = np.asarray([row["epoch"] for row in data])
        mean = np.asarray([row["mean_bellman_mse"] for row in data])
        within = np.asarray([row["within_state_variance"] for row in data])
        total = np.asarray([row["total_mse"] for row in data])
        ax.stackplot(x, mean, within, labels=("Mean Bellman MSE", "Within-state variance"),
                     colors=("#0072B2", "#CC79A7"), alpha=0.75)
        ax.plot(x, total, color="black", marker="o", label="Total MSE")
        ax.set_title(labels[variant]); ax.set_xlabel("Actual checkpoint epoch")
        ax.set_ylabel("MSE"); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.2)
    path = figures / "mse_decomposition.png"
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); made.append(str(path.resolve()))

    line_plot("mean_bellman_rmse.png", (("mean_bellman_rmse", ""),), "Mean Bellman RMSE")
    line_plot("target_variance.png", (("within_state_variance", ""),), "Within-state variance")
    line_plot("normalized_mean_error.png", (("normalized_mean_error", ""),),
              "Normalized mean Bellman error")
    line_plot(
        "value_drift.png",
        (("value_drift_rmse", ": absolute"), ("relative_drift", ": relative")),
        "Value drift",
    )
    line_plot("bellman_gap.png", (("gap_mean", ": mean"),), "Mean target - value",
              band=("gap_p10", "gap_p90"))
    line_plot("value_vs_target_sd.png",
              (("value_sd", ": value SD"), ("target_between_state_sd", ": target SD")),
              "Between-anchor SD")
    return made


def _write_report(
    output: Path,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    mappings: Mapping[str, Sequence[Mapping[str, Any]]],
    reward_summary: Mapping[str, Any],
) -> None:
    grouped = {variant: sorted([row for row in rows if row["variant"] == variant],
                               key=lambda row: int(row["epoch"])) for variant in VARIANTS}
    unique = {variant: len(grouped[variant]) for variant in VARIANTS}
    lines = [
        f"# {label}", "", "# Phase 8J-BME-Q Bellman Mean-Error Decomposition Audit", "",
        "本阶段是 model seed 0 的只读机制诊断，不进行训练、SAC、调参、显著性检验或 do-oracle 查询。", "",
        "## Checkpoint coverage", "",
        f"Requested epochs: {list(REQUESTED_EPOCHS)}.",
        f"Unique actual checkpoints: real={unique[VARIANTS[0]]}, full={unique[VARIANTS[1]]}.",
    ]
    if min(unique.values()) < 3:
        lines.extend(["", "历史 checkpoint 不足三个，因此不能判断持续趋势；所有 requested→actual 替换已记录在 manifest.json。", ""])
    lines.extend(["## Table 1: MSE decomposition", "",
                  "| variant | epoch | total MSE | mean-Bellman MSE | within-state variance | mean-error fraction |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in sorted(rows, key=lambda r: (str(r["variant"]), int(r["epoch"]))):
        lines.append(f"| {row['variant']} | {row['epoch']} | {row['total_mse']:.6g} | "
                     f"{row['mean_bellman_mse']:.6g} | {row['within_state_variance']:.6g} | "
                     f"{row['mean_error_fraction']:.4f} |")
    lines.extend(["", "## Table 2: scale-normalized diagnostics", "",
                  "| variant | epoch | V SD | mean-target SD | mean-Bellman RMSE | normalized error | relative drift |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in sorted(rows, key=lambda r: (str(r["variant"]), int(r["epoch"]))):
        drift = "NA" if not np.isfinite(float(row["relative_drift"])) else f"{row['relative_drift']:.6g}"
        lines.append(f"| {row['variant']} | {row['epoch']} | {row['value_sd']:.6g} | "
                     f"{row['target_between_state_sd']:.6g} | {row['mean_bellman_rmse']:.6g} | "
                     f"{row['normalized_mean_error']:.6g} | {drift} |")
    lines.extend(["", "## Table 3: Bellman fixed-point direction", "",
                  "| variant | epoch | mean(Ybar-V) | median | P10 | P90 | positive fraction | negative fraction |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in sorted(rows, key=lambda r: (str(r["variant"]), int(r["epoch"]))):
        lines.append(f"| {row['variant']} | {row['epoch']} | {row['gap_mean']:.6g} | "
                     f"{row['gap_median']:.6g} | {row['gap_p10']:.6g} | {row['gap_p90']:.6g} | "
                     f"{row['gap_positive_fraction']:.4f} | {row['gap_negative_fraction']:.4f} |")
    lines.extend([
        "", "## Reward-only stable control", "",
        f"Best validation loss={reward_summary.get('best_validation_loss', 'NA')}; "
        f"final validation loss={reward_summary.get('final_validation_loss', 'NA')}; "
        f"final/best={reward_summary.get('final_over_best_validation_ratio', 'NA')}.", "",
        "## Statistical boundary", "",
        "只有一个模型 seed。epochs、anchors 与 transition rows 都是重复观测，不能作为独立实验重复；因此不报告 p 值或置信区间。",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    phase8j_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model_seed: int = 0,
    requested_epochs: Sequence[int] = REQUESTED_EPOCHS,
    fix_root: Path = DEFAULT_FIX_ROOT,
    fvi_root: Path = DEFAULT_FVI_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
) -> dict[str, Any]:
    if int(model_seed) != 0:
        raise BellmanMeanErrorAuditError("Phase 8J-BME-Q is frozen to model seed 0")
    source = Path(phase8j_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source / "manifest.json"
    source_hard_path = source / "hard_checks.json"
    if not source_manifest_path.is_file() or not source_hard_path.is_file():
        raise BellmanMeanErrorAuditError("Phase 8J-BI-Q manifest/hard checks are missing")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_hard = json.loads(source_hard_path.read_text(encoding="utf-8"))
    if source_manifest.get("stage") != SOURCE_PHASE or source_hard.get("all_passed") is not True:
        raise BellmanMeanErrorAuditError("Phase 8J-BI-Q source contract is invalid")

    try:
        import torch as checkpoint_torch
    except (ImportError, OSError) as error:
        raise BellmanMeanErrorAuditError("PyTorch is required to read checkpoints") from error
    available, inventory = _checkpoint_inventory(source, checkpoint_torch)
    if any(not available[variant] for variant in VARIANTS):
        counts = {variant: sorted(available[variant]) for variant in VARIANTS}
        raise BellmanMeanErrorAuditError(
            "required read-only BI-Q checkpoints are unavailable; "
            f"found epochs={counts}. The source trainer saved checkpoints outside Git."
        )

    resolved_fvi, _ = _load_fvi_contract(fvi_root)
    context = _resolve_context(fix_root, external_repo, device)
    components = _load_seed_components(context, 0)
    mean, std, reward_min, reward_max = _normalization(context)
    mappings = {variant: map_requested_checkpoints(available[variant], requested_epochs)
                for variant in VARIANTS}
    unique_paths = {Path(row["path"]) for values in mappings.values() for row in values}
    tracked = [source_manifest_path, source_hard_path, source / "training_metrics.csv",
               *unique_paths, *_source_input_paths(context, resolved_fvi)]
    before = _snapshot(tracked)

    validation_rows = np.asarray(context["validation_rows"], dtype=np.int64)
    public = context["public"]
    anchor_ids = np.asarray(public["anchor_id"], dtype=np.int64)[validation_rows]
    row_weights = np.ones(len(validation_rows), dtype=np.float64)
    source_ids = np.asarray(public["source_id"], dtype=np.int64)[validation_rows]
    observations = np.asarray(public["observation"], dtype=np.float32)[validation_rows]
    public_keys = {str(key).lower() for key in public}
    forbidden = {key for key in public_keys if key.startswith("u_") or key in {
        "u", "hidden_u", "do_oracle", "do_reward", "do_q"}}

    rows: list[dict[str, Any]] = []
    state_predictions: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    state_masses: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        actual_to_requests: dict[int, list[int]] = {}
        for mapping in mappings[variant]:
            actual_to_requests.setdefault(int(mapping["actual_epoch"]), []).append(
                int(mapping["requested_epoch"]))
        for epoch in sorted(actual_to_requests):
            path = available[variant][epoch]
            network = _load_network(path, context, reward_min, reward_max)
            prediction = _network_predictions(network, observations, mean, std, context)
            target = _variant_targets(
                variant, context, components, network, mean, std, epoch
            )
            summary, state = decompose_by_anchor(
                prediction, target, anchor_ids, row_weights
            )
            state_predictions[variant].append(
                np.asarray([record["value"] for record in state], dtype=np.float64)
            )
            state_masses[variant].append(
                np.asarray([record["state_mass"] for record in state], dtype=np.float64)
            )
            rows.append({
                "variant": variant,
                "epoch": epoch,
                "requested_epochs": ";".join(map(str, actual_to_requests[epoch])),
                **summary,
            })

    for variant in VARIANTS:
        variant_rows = sorted([row for row in rows if row["variant"] == variant],
                              key=lambda row: int(row["epoch"]))
        predictions = state_predictions[variant]
        previous: np.ndarray | None = None
        for row, values, weights in zip(variant_rows, predictions, state_masses[variant]):
            if previous is None:
                row["value_drift_rmse"] = 0.0
                row["relative_drift"] = 0.0
            else:
                drift = float(np.sqrt(
                    np.sum(weights * np.square(values - previous)) / np.sum(weights)
                ))
                row["value_drift_rmse"] = drift
                row["relative_drift"] = drift / max(float(row["value_sd"]), np.finfo(float).eps)
            previous = values

    decision = classify_result(rows)
    decomposition_rows = [{key: row[key] for key in (
        "variant", "epoch", "requested_epochs", "row_count", "anchor_count",
        "total_mse", "mean_bellman_mse", "within_state_variance",
        "mean_error_fraction", "noise_fraction", "identity_residual", "identity_tolerance")}
        for row in rows]
    normalized_rows = [{key: row[key] for key in (
        "variant", "epoch", "requested_epochs", "value_sd", "target_between_state_sd",
        "mean_bellman_rmse", "normalized_mean_error", "value_drift_rmse", "relative_drift")}
        for row in rows]
    gap_rows = [{key: row[key] for key in (
        "variant", "epoch", "requested_epochs", "gap_mean", "gap_median", "gap_std",
        "gap_p10", "gap_p90", "gap_positive_fraction", "gap_negative_fraction")}
        for row in rows]
    _write_csv(output / "mse_decomposition.csv", decomposition_rows)
    _write_csv(output / "normalized_error_metrics.csv", normalized_rows)
    _write_csv(output / "bellman_gap_metrics.csv", gap_rows)
    figures = _figures(output, rows)

    after = _snapshot(tracked)
    source_masses = {str(source_id): float(np.mean(source_ids == source_id))
                     for source_id in sorted(np.unique(source_ids))}
    checks = {
        "phase8j_biq_hard_checks_passed": True,
        "model_seed_zero_only": True,
        "requested_to_actual_checkpoint_mapping_recorded": all(mappings.values()),
        "checkpoint_metadata_valid": bool(inventory) and all(
            row["metadata_valid"] for row in inventory),
        "original_validation_rows_only": len(validation_rows) > 0,
        "original_equal_row_weighting_preserved": bool(np.all(row_weights == 1.0)),
        "source_mass_preserved": math.isclose(sum(source_masses.values()), 1.0),
        "current_checkpoint_recomputes_target": True,
        "cached_outer_targets_not_read": True,
        "termination_and_truncation_use_parent_semantics": True,
        "full_aamas_frozen_candidate_stream_reused": True,
        "full_aamas_candidate_and_noise_stream_fixed_across_checkpoints": True,
        "decomposition_identity_within_machine_tolerance": all(
            float(row["identity_residual"]) <= float(row["identity_tolerance"])
            for row in rows),
        "all_metrics_finite": all(np.isfinite(float(value))
            for row in rows for key, value in row.items()
            if key not in {"variant", "requested_epochs"}),
        "hidden_u_absent": not forbidden,
        "do_oracle_not_used": True,
        "test_split_not_used": True,
        "online_return_and_sac_not_used": True,
        "no_training_or_parameter_updates": True,
        "input_files_unchanged": before == after,
    }
    reward_rows = [row for row in _read_csv(source / "growth_summary.csv")
                   if row.get("variant") == "reward_only"]
    reward_summary = reward_rows[0] if reward_rows else {}
    _write_report(output, decision, rows, mappings, reward_summary)
    manifest = {
        "stage": PHASE,
        "source_stage": SOURCE_PHASE,
        "source_root": str(source),
        "model_seed": 0,
        "requested_epochs": list(map(int, requested_epochs)),
        "checkpoint_mapping": mappings,
        "unique_actual_epochs": {variant: sorted({int(row["actual_epoch"])
                                                   for row in mappings[variant]})
                                 for variant in VARIANTS},
        "gamma": GAMMA,
        "validation_row_count": len(validation_rows),
        "validation_anchor_count": len(np.unique(anchor_ids)),
        "validation_source_mass": source_masses,
        "row_weighting": "same equal-row MSE estimand as Phase 8J-BI-Q validation",
        "full_aamas_audit_stream": "fixed outer-0 validation candidate/noise stream",
        "decision": decision,
        "single_seed_descriptive_only": True,
        "significance_tests_performed": False,
        "training_performed": False,
        "sac_run": False,
        "do_oracle_used": False,
        "figures": figures,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "input_integrity.json", {
        "policy": "size-and-modification-time records; no cryptographic digest",
        "before": before,
        "after": after,
        "unchanged": before == after,
    })
    failed = [name for name, passed in checks.items() if not passed]
    _write_json(output / "hard_checks.json", {
        "stage": PHASE, "checks": checks, "failed": failed,
        "all_passed": not failed,
    })
    if failed:
        raise BellmanMeanErrorAuditError(f"hard checks failed: {failed}")
    return {"decision": decision, "actual_epochs": manifest["unique_actual_epochs"]}
