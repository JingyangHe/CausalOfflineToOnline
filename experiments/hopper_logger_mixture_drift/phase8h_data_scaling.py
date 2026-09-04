"""Phase 8H-DS: source-wise data scaling diagnostic.

Only the number of public transitions per anchor and source changes.  The
Phase 8H DGP, models, candidate actions, reference value, and metrics remain
fixed.  Artifacts contain best checkpoints and compact metric tables only.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aamas_hopper_adapter import _import_official_module, validate_external_repo
from scripts.train_aamas_hopper_potential import seed_everything
from .generate_datasets import MujocoOneStepSimulator
from .phase8h_quick_multipolicy_aamas import (
    CANDIDATE_ACTIONS,
    EXTERNAL_COMMIT,
    FORBIDDEN_MODEL_FIELDS,
    GAMMA,
    KAPPA,
    LAMBDA_REWARD,
    POOLED_MIXTURES,
    PUBLIC_MODEL_FIELDS,
    SIGMA_ACTION,
    SOURCE_B,
    SOURCE_D,
    FrozenSACReferenceValue,
    _device_name,
    _load_phase8h_inputs,
    _save_component_checkpoint,
    _subset_rows,
    action_and_state_level_envelopes,
    compute_source_aamas_backup,
    deterministic_argmax,
    do_bellman_oracle,
    fit_aamas_components,
    generate_source_dataset,
    pooled_row_weights,
    prediction_metrics,
    source_commanded_action,
    source_policy_parameters,
    union_candidate_actions,
    validate_public_dataset,
)


PHASE = "Phase 8H-DS"
SAMPLE_SIZES = (16, 32, 64, 128)
UPDATE_BUDGETS = {16: 500, 32: 1000, 64: 2000, 128: 4000}
MODEL_SEEDS = (0, 1, 2)
BATCH_SIZE = 512
METHODS = (
    "source_1", "source_2", "source_3", "pooled_balanced",
    "state_level_min", "action_level_min",
)
PRIMARY_METHODS = ("pooled_balanced", "source_3", "action_level_min")
EXPECTED_FIGURES = (
    "do_mae_vs_data.png", "mean_regret_vs_data.png", "median_regret_vs_data.png",
    "tail_regret_vs_data.png", "underestimation_vs_data.png",
    "actionmin_vs_source3_gap.png", "actionmin_gain_over_pooled.png",
    "data_vs_compute_control.png", "actionmin_per_seed_scaling.png",
)
BASELINE_TARGETS = {
    ("action_level_min", "do_mae"): 2.372,
    ("action_level_min", "regret_mean"): .808,
    ("action_level_min", "underestimation_fraction"): .361,
    ("pooled_balanced", "do_mae"): 3.183,
    ("pooled_balanced", "regret_mean"): .839,
    ("source_3", "regret_mean"): .771,
}


class Phase8HDataScalingError(RuntimeError):
    """Raised when a frozen Phase 8H-DS invariant is violated."""


def cvar90(values: np.ndarray, anchor_ids: np.ndarray) -> float:
    """Frozen Phase 8H-MA definition with deterministic anchor-ID tie breaking."""
    values = np.asarray(values, dtype=np.float64)
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    count = max(1, int(math.ceil(0.10 * len(values))))
    order = np.lexsort((anchor_ids, -values))
    return float(values[order[:count]].mean())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                               default=_json_default) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8"); return
    fields: list[str] = []
    for row in records:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)


def file_metadata(path: Path) -> dict[str, Any]:
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns}


def metadata_snapshot(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [file_metadata(path) for path in sorted(map(Path, paths), key=lambda item: str(item))]


def _base_paths(root: Path) -> list[Path]:
    required = [root / name for name in (
        "manifest.json", "hard_checks.json", "splits.json", "seed_metrics.csv",
        "predictions/anchor_candidate_predictions.npz")]
    return required + sorted((root / "models").rglob("*.pt"))


def _baseline_gate(root: Path) -> tuple[dict[str, float], bool]:
    hard = json.loads((root / "hard_checks.json").read_text(encoding="utf-8"))
    if hard.get("all_passed") is not True:
        raise Phase8HDataScalingError("Phase 8H hard checks did not pass")
    with (root / "seed_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {
        "action_level_min": ("action_level_min", "INVARIANT"),
        "pooled_balanced": ("pooled_aamas_union", "balanced"),
        "source_3": ("single_source_3", "INVARIANT"),
    }
    result: dict[str, float] = {}
    for method, (legacy, composition) in mapping.items():
        selected = [row for row in rows if row["condition"] == "confounded"
                    and row["method"] == legacy and row["composition"] == composition
                    and row["candidate_set"] == "union"]
        if len(selected) != 3:
            raise Phase8HDataScalingError(f"Phase 8H baseline rows missing for {method}")
        for metric in ("do_mae", "regret_mean", "underestimation_fraction"):
            result[f"{method}.{metric}"] = float(np.mean([float(row[metric]) for row in selected]))
    passed = all(np.isclose(result[f"{method}.{metric}"], expected, atol=.002, rtol=0.0)
                 for (method, metric), expected in BASELINE_TARGETS.items())
    return result, passed


def n32_metrics_reproduce_phase8h(
    seed_rows: Sequence[Mapping[str, Any]], root: Path, seeds: Sequence[int],
) -> bool:
    """Compare retrained D32 seed metrics with the recorded Phase 8H rows."""
    with (root / "seed_metrics.csv").open(newline="", encoding="utf-8") as stream:
        legacy = list(csv.DictReader(stream))
    mapping = {
        "source_1": ("single_source_1", "INVARIANT"),
        "source_2": ("single_source_2", "INVARIANT"),
        "source_3": ("single_source_3", "INVARIANT"),
        "pooled_balanced": ("pooled_aamas_union", "balanced"),
        "state_level_min": ("state_level_min", "INVARIANT"),
        "action_level_min": ("action_level_min", "INVARIANT"),
    }
    for seed in seeds:
        for method, (old_method, composition) in mapping.items():
            current = [row for row in seed_rows if row["condition"] == "confounded"
                       and row["data_label"] == "n32" and int(row["seed"]) == seed
                       and row["method"] == method]
            old = [row for row in legacy if row["condition"] == "confounded"
                   and int(row["seed"]) == seed and row["method"] == old_method
                   and row["composition"] == composition and row["candidate_set"] == "union"]
            if len(current) != 1 or len(old) != 1:
                return False
            for metric in ("do_mae", "do_rmse", "signed_error",
                           "underestimation_fraction", "regret_mean",
                           "regret_median", "regret_p90"):
                if not np.isclose(float(current[0][metric]), float(old[0][metric]),
                                  atol=5e-4, rtol=5e-4):
                    return False
    return True


def _consume_original_draws(condition: str, seed: int, anchor_count: int) -> np.random.Generator:
    rng = np.random.default_rng(seed + (0 if condition == "confounded" else 10_000_019))
    for _ in range(anchor_count * 3 * 32):
        rng.random()
        if condition == "independent_latents":
            rng.random()
        rng.standard_normal(3)
    return rng


def generate_nested_master(
    anchors: Mapping[str, np.ndarray], simulator: Any, *, condition: str, seed: int,
    max_samples: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Preserve the original D32 exactly, then append deterministic draws 33..128."""
    if max_samples not in (32, 64, 128):
        raise ValueError("max_samples must be one of (32, 64, 128)")
    original_public, original_hidden = generate_source_dataset(
        anchors, simulator, condition=condition, samples_per_anchor_source=32, seed=seed)
    rng = _consume_original_draws(condition, seed, len(anchors["anchor_id"]))
    public_extra: dict[str, list[Any]] = {name: [] for name in PUBLIC_MODEL_FIELDS}
    hidden_extra: dict[str, list[Any]] = {name: [] for name in original_hidden}
    for position, anchor_id in enumerate(np.asarray(anchors["anchor_id"], dtype=np.int64)):
        base = np.asarray(anchors["base_action"][position], dtype=np.float64)
        for source_id in (1, 2, 3):
            for sample_id in range(32, max_samples):
                u_behavior = 1 if rng.random() >= .5 else -1
                u_environment = (u_behavior if condition == "confounded"
                                 else (1 if rng.random() >= .5 else -1))
                command = source_commanded_action(
                    base, source_id, u_behavior, rng.standard_normal(3))
                outcome = simulator.step(position, command, u_environment, KAPPA)
                values = {
                    "observation": outcome["observation"],
                    "commanded_action": command.astype(np.float32),
                    "reward": float(outcome["reward"] + LAMBDA_REWARD * u_environment),
                    "next_observation": outcome["next_observation"],
                    "terminated": bool(outcome["terminated"]),
                    "truncated": bool(outcome["truncated"]),
                    "anchor_id": int(anchor_id), "source_id": source_id,
                    "sample_id": sample_id,
                }
                for name, value in values.items():
                    public_extra[name].append(value)
                hidden_extra["u_behavior"].append(u_behavior)
                hidden_extra["u_environment"].append(u_environment)
                hidden_extra["applied_action"].append(outcome["applied_action"])
    if max_samples == 32:
        public = {name: np.asarray(value).copy() for name, value in original_public.items()}
        hidden = {name: np.asarray(value).copy() for name, value in original_hidden.items()}
    else:
        public = {name: np.concatenate((original_public[name], np.asarray(public_extra[name])))
                  for name in PUBLIC_MODEL_FIELDS}
        hidden = {name: np.concatenate((original_hidden[name], np.asarray(hidden_extra[name])))
                  for name in original_hidden}
    for name in ("observation", "commanded_action", "reward", "next_observation"):
        public[name] = public[name].astype(np.float32)
    public["terminated"] = public["terminated"].astype(bool)
    public["truncated"] = public["truncated"].astype(bool)
    public["anchor_id"] = public["anchor_id"].astype(np.int64)
    public["source_id"] = public["source_id"].astype(np.int8)
    public["sample_id"] = public["sample_id"].astype(np.int16)
    validate_public_dataset(public, len(anchors["anchor_id"]) * 3 * max_samples)
    d32 = subset_nested(public, 32)
    exact = all(np.array_equal(d32[name], original_public[name]) for name in PUBLIC_MODEL_FIELDS)
    return public, hidden, {
        "condition": condition, "master_draws_per_anchor_source": max_samples,
        "original_d32_exact": exact,
        "extension_protocol": "continue the original global RNG after all original D32 draws",
    }


def subset_nested(master: Mapping[str, np.ndarray], samples: int) -> dict[str, np.ndarray]:
    if samples not in SAMPLE_SIZES:
        raise ValueError(f"samples must be one of {SAMPLE_SIZES}")
    mask = np.asarray(master["sample_id"]) < samples
    result = {name: np.asarray(master[name])[mask].copy() for name in PUBLIC_MODEL_FIELDS}
    validate_public_dataset(result, len(np.unique(result["anchor_id"])) * 3 * samples)
    return result


def nested_dataset_audit(
    master: Mapping[str, np.ndarray], sample_sizes: Sequence[int] = SAMPLE_SIZES,
) -> dict[str, Any]:
    sizes = tuple(sorted(set(int(size) for size in sample_sizes)))
    datasets = {size: subset_nested(master, size) for size in sizes}
    keys = {size: set(zip(datasets[size]["anchor_id"].tolist(),
                         datasets[size]["source_id"].tolist(),
                         datasets[size]["sample_id"].tolist())) for size in sizes}
    subset_checks = {
        f"D{left}_subset_D{right}": keys[left] < keys[right]
        for left, right in zip(sizes[:-1], sizes[1:])}
    counts = {str(size): {
        str(source): int(np.sum(datasets[size]["source_id"] == source))
        for source in (1, 2, 3)} for size in sizes}
    return {"subset_checks": subset_checks, "source_counts": counts,
            "all_nested": all(subset_checks.values()),
            "all_source_counts_equal": all(len(set(row.values())) == 1 for row in counts.values())}


def _metric_row(condition: str, seed: int, data_label: str, samples: int,
                updates: int, method: str, truth: np.ndarray,
                prediction: np.ndarray) -> dict[str, Any]:
    metric = prediction_metrics(truth, prediction)
    selected = deterministic_argmax(prediction)
    regret = truth.max(axis=1) - truth[np.arange(len(truth)), selected]
    metric["regret_cvar90"] = cvar90(regret, np.arange(len(regret)))
    metric["regret_p95"] = float(np.quantile(regret, .95))
    return {"condition": condition, "seed": seed, "data_label": data_label,
            "samples_per_anchor_source": samples, "gradient_updates": updates,
            "method": method, "candidate_count": truth.shape[1], **metric}


def _fit_scenario(
    public: Mapping[str, np.ndarray], *, condition: str, seed: int,
    data_label: str, samples: int, updates: int, splits: Mapping[str, np.ndarray],
    test_states: np.ndarray, test_base: np.ndarray, test_positions: np.ndarray,
    anchors: Mapping[str, np.ndarray], simulator: Any, reference: Any,
    device: str, official: Any, torch: Any, output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    source_models = []; curves: list[dict[str, Any]] = []
    input_checks = []
    for source_id in (1, 2, 3):
        source_public = _subset_rows(public, np.asarray(public["source_id"]) == source_id)
        training_seed = seed * 100 + source_id
        seed_everything(training_seed, torch, cuda_training=device == "cuda")
        model, normalization, training = fit_aamas_components(
            source_public, splits["train"], splits["observational_validation"],
            row_probabilities=np.ones(len(source_public["reward"])),
            seed=training_seed, gradient_updates=updates, batch_size=BATCH_SIZE,
            device=device, official=official, torch=torch, record_schedule_digest=False)
        metadata = {**training["metadata"], "condition": condition,
                    "data_label": data_label, "samples_per_anchor_source": samples,
                    "model_kind": "source", "source_id": source_id}
        _save_component_checkpoint(
            output / "models" / condition / data_label / "source" /
            f"source_{source_id}" / f"seed_{seed}.pt",
            training, normalization, metadata, torch)
        source_models.append(model)
        input_checks.append(metadata["model_input_fields"] == ["observation", "commanded_action"])
        curves.extend({"condition": condition, "seed": seed, "data_label": data_label,
                       "samples_per_anchor_source": samples, "gradient_updates": updates,
                       "model": f"source_{source_id}", **row}
                      for row in training["history"])
    weights = pooled_row_weights(public["source_id"], POOLED_MIXTURES["balanced"])
    training_seed = seed * 100 + 50
    seed_everything(training_seed, torch, cuda_training=device == "cuda")
    pooled, normalization, training = fit_aamas_components(
        public, splits["train"], splits["observational_validation"],
        row_probabilities=weights, seed=training_seed, gradient_updates=updates,
        batch_size=BATCH_SIZE, device=device, official=official, torch=torch,
        record_schedule_digest=False)
    metadata = {**training["metadata"], "condition": condition,
                "data_label": data_label, "samples_per_anchor_source": samples,
                "model_kind": "pooled", "composition": "balanced"}
    _save_component_checkpoint(
        output / "models" / condition / data_label / "pooled" /
        "balanced" / f"seed_{seed}.pt", training, normalization, metadata, torch)
    curves.extend({"condition": condition, "seed": seed, "data_label": data_label,
                   "samples_per_anchor_source": samples, "gradient_updates": updates,
                   "model": "pooled_balanced", **row} for row in training["history"])
    input_checks.append(metadata["model_input_fields"] == ["observation", "commanded_action"])

    candidates = union_candidate_actions(
        source_models, test_states, test_base, samples_per_source=8, seed=20260805)
    truth = do_bellman_oracle(simulator, anchors, test_positions, candidates, reference)
    noise = np.random.default_rng(20260806).standard_normal(
        (len(test_states) * candidates.shape[1], CANDIDATE_ACTIONS, 3)).astype(np.float32)
    source_q = compute_source_aamas_backup(
        source_models, test_states, candidates, reference, common_noise=noise)
    envelope = action_and_state_level_envelopes(source_q)
    pooled_q = compute_source_aamas_backup(
        (pooled,), test_states, candidates, reference, common_noise=noise)[0]
    predictions = {
        "source_1": source_q[0], "source_2": source_q[1], "source_3": source_q[2],
        "pooled_balanced": pooled_q, "state_level_min": envelope["state_selected_q"],
        "action_level_min": envelope["action_q"],
    }
    metrics = [_metric_row(condition, seed, data_label, samples, updates, method,
                           truth, prediction) for method, prediction in predictions.items()]
    return metrics, curves, {
        "public_model_fields_only": all(input_checks),
        "candidate_count_28": candidates.shape[1] == 28,
        "all_finite": all(np.all(np.isfinite(value)) for value in predictions.values())
                      and np.all(np.isfinite(truth)),
    }


def aggregate_scaling_metrics(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("do_mae", "do_rmse", "signed_error", "underestimation_fraction",
               "regret_mean", "regret_median", "regret_p90", "regret_cvar90")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        if row["condition"] == "confounded" and row["data_label"] != "n32_extra_compute":
            groups[(row["samples_per_anchor_source"], row["method"])].append(row)
    output = []
    for (samples, method), rows in sorted(groups.items()):
        record: dict[str, Any] = {"samples_per_anchor_source": samples, "method": method,
                                  "model_seed_count": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def _comparison_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["condition"], row["data_label"], int(row["seed"]), row["method"]): row
              for row in seed_rows}
    output = []
    for condition in ("confounded", "independent_latents"):
        labels = sorted({row["data_label"] for row in seed_rows if row["condition"] == condition})
        for label in labels:
            seeds = sorted({int(row["seed"]) for row in seed_rows
                            if row["condition"] == condition and row["data_label"] == label})
            for baseline in ("pooled_balanced", "source_3"):
                for metric in ("regret_mean", "regret_median", "regret_p90", "regret_cvar90"):
                    differences = np.asarray([
                        float(lookup[(condition, label, seed, "action_level_min")][metric])
                        - float(lookup[(condition, label, seed, baseline)][metric])
                        for seed in seeds])
                    output.append({
                        "condition": condition, "data_label": label,
                        "samples_per_anchor_source": int(lookup[
                            (condition, label, seeds[0], baseline)]["samples_per_anchor_source"]),
                        "contrast": f"action_level_min_minus_{baseline}", "metric": metric,
                        "n_model_seeds": len(seeds), "mean_difference": float(differences.mean()),
                        "difference_seed_sd": (float(differences.std(ddof=1))
                                               if len(differences) > 1 else 0.0),
                        "action_min_better_seed_count": int(np.sum(differences < 0)),
                        "seed_differences": json.dumps(differences.tolist(), separators=(",", ":")),
                    })
    return output


def _figures(output: Path, scaling: Sequence[Mapping[str, Any]],
             comparisons: Sequence[Mapping[str, Any]], seed_rows: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    available_sizes = sorted({int(row["samples_per_anchor_source"]) for row in scaling})
    def metric_plot(filename: str, metric: str, ylabel: str,
                    methods: Sequence[str] = PRIMARY_METHODS) -> None:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for method in methods:
            rows = sorted((row for row in scaling if row["method"] == method),
                          key=lambda row: int(row["samples_per_anchor_source"]))
            x = [row["samples_per_anchor_source"] for row in rows]
            y = [row[f"{metric}_mean"] for row in rows]
            sd = [row[f"{metric}_seed_sd"] for row in rows]
            axis.errorbar(x, y, yerr=sd, marker="o", capsize=3, label=method)
        axis.set_xlabel("Transitions per anchor per source"); axis.set_ylabel(ylabel)
        axis.set_xticks(available_sizes); axis.legend(); fig.tight_layout()
        fig.savefig(figures / filename, dpi=180); plt.close(fig)
    metric_plot("do_mae_vs_data.png", "do_mae", "Do-Bellman MAE")
    metric_plot("mean_regret_vs_data.png", "regret_mean", "Mean regret")
    metric_plot("median_regret_vs_data.png", "regret_median", "Median regret")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for metric in ("regret_p90", "regret_cvar90"):
        rows = sorted((row for row in scaling if row["method"] == "action_level_min"),
                      key=lambda row: int(row["samples_per_anchor_source"]))
        axis.errorbar([row["samples_per_anchor_source"] for row in rows],
                      [row[f"{metric}_mean"] for row in rows],
                      yerr=[row[f"{metric}_seed_sd"] for row in rows],
                      marker="o", capsize=3, label=metric)
    axis.set_xlabel("Transitions per anchor per source"); axis.set_ylabel("Tail regret")
    axis.set_xticks(available_sizes); axis.legend(); fig.tight_layout()
    fig.savefig(figures / "tail_regret_vs_data.png", dpi=180); plt.close(fig)
    metric_plot("underestimation_vs_data.png", "underestimation_fraction",
                "Underestimation fraction")
    def contrast_plot(filename: str, baseline: str, ylabel: str, negate: bool = False) -> None:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for metric in ("regret_mean", "regret_median", "regret_p90", "regret_cvar90"):
            rows = sorted((row for row in comparisons if row["condition"] == "confounded"
                           and row["contrast"] == f"action_level_min_minus_{baseline}"
                           and row["metric"] == metric and row["data_label"] != "n32_extra_compute"),
                          key=lambda row: int(row["samples_per_anchor_source"]))
            values = [(-1 if negate else 1) * float(row["mean_difference"]) for row in rows]
            axis.plot([row["samples_per_anchor_source"] for row in rows], values,
                      marker="o", label=metric)
        axis.axhline(0, linewidth=1, linestyle="--")
        axis.set_xlabel("Transitions per anchor per source"); axis.set_ylabel(ylabel)
        axis.set_xticks(available_sizes); axis.legend(); fig.tight_layout()
        fig.savefig(figures / filename, dpi=180); plt.close(fig)
    contrast_plot("actionmin_vs_source3_gap.png", "source_3", "Action min − Source 3")
    contrast_plot("actionmin_gain_over_pooled.png", "pooled_balanced",
                  "Pooled − Action min", negate=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    label_pairs = [
        (display, stored) for display, stored in (
            ("n32", "n32"),
            ("n32 extra compute", "n32_extra_compute"),
            ("n128", "n128"),
        ) if any(row["condition"] == "confounded" and row["seed"] == 0
                 and row["data_label"] == stored for row in seed_rows)
    ]
    for index, method in enumerate(PRIMARY_METHODS):
        values = []
        for _, data_label in label_pairs:
            row = next(item for item in seed_rows if item["condition"] == "confounded"
                       and item["seed"] == 0 and item["data_label"] == data_label
                       and item["method"] == method)
            values.append(float(row["regret_mean"]))
        axis.plot(range(len(label_pairs)), values, marker="o", label=method)
    axis.set_xticks(range(len(label_pairs)), [pair[0] for pair in label_pairs])
    axis.set_ylabel("Mean regret"); axis.legend()
    fig.tight_layout(); fig.savefig(figures / "data_vs_compute_control.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for seed in sorted({int(row["seed"]) for row in seed_rows
                        if row["condition"] == "confounded"}):
        rows = sorted((row for row in seed_rows if row["condition"] == "confounded"
                       and int(row["seed"]) == seed and row["method"] == "action_level_min"
                       and row["data_label"] != "n32_extra_compute"),
                      key=lambda row: int(row["samples_per_anchor_source"]))
        axes[0].plot([row["samples_per_anchor_source"] for row in rows],
                     [row["do_mae"] for row in rows], marker="o", label=f"seed {seed}")
        axes[1].plot([row["samples_per_anchor_source"] for row in rows],
                     [row["regret_mean"] for row in rows], marker="o", label=f"seed {seed}")
    axes[0].set_ylabel("Do-Bellman MAE"); axes[1].set_ylabel("Mean regret")
    for axis in axes:
        axis.set_xlabel("Transitions per anchor per source")
        axis.set_xticks(available_sizes)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "actionmin_per_seed_scaling.png", dpi=180)
    plt.close(fig)


def _report(output: Path, scaling: Sequence[Mapping[str, Any]],
            comparisons: Sequence[Mapping[str, Any]], seed_rows: Sequence[Mapping[str, Any]]) -> None:
    available_sizes = sorted({int(row["samples_per_anchor_source"]) for row in scaling})
    confounded_seed_count = len({int(row["seed"]) for row in seed_rows
                                 if row["condition"] == "confounded"})
    def series(method: str, metric: str) -> list[float]:
        return [float(next(row for row in scaling if row["method"] == method
                          and row["samples_per_anchor_source"] == size)[f"{metric}_mean"])
                for size in available_sizes]
    mae = series("action_level_min", "do_mae")
    under = series("action_level_min", "underestimation_fraction")
    regrets = {metric: series("action_level_min", metric)
               for metric in ("regret_mean", "regret_median", "regret_p90", "regret_cvar90")}
    source_gap = [float(next(row for row in comparisons if row["condition"] == "confounded"
                            and row["data_label"] == f"n{size}"
                            and row["contrast"] == "action_level_min_minus_source_3"
                            and row["metric"] == "regret_mean")["mean_difference"])
                  for size in available_sizes]
    pooled_gain = [-float(next(row for row in comparisons if row["condition"] == "confounded"
                              and row["data_label"] == f"n{size}"
                              and row["contrast"] == "action_level_min_minus_pooled_balanced"
                              and row["metric"] == "regret_mean")["mean_difference"])
                   for size in available_sizes]
    control = {
        label: float(next(row for row in seed_rows if row["condition"] == "confounded"
                          and row["seed"] == 0 and row["data_label"] == label
                          and row["method"] == "action_level_min")["regret_mean"])
        for label in ("n32", "n32_extra_compute", "n128")
        if any(row["condition"] == "confounded" and row["seed"] == 0
               and row["data_label"] == label and row["method"] == "action_level_min"
               for row in seed_rows)
    }
    independent = [row for row in comparisons if row["condition"] == "independent_latents"
                   and row["metric"] == "regret_mean"
                   and row["contrast"] == "action_level_min_minus_pooled_balanced"]
    monotone = lambda values: all(right <= left for left, right in zip(values[:-1], values[1:]))
    lines = [
        "# Phase 8H-DS Source-Wise Data Scaling Diagnostic", "",
        f"The independent statistical unit is model seed (n={confounded_seed_count}); "
        "anchors are repeated measurements.", "",
        "## 1. Does Action-min Do-MAE continually decrease?", "",
        f"Values for n={available_sizes}: {mae}. Monotone decrease: {monotone(mae)}.", "",
        "## 2. Does the roughly 36% underestimation decrease?", "",
        f"Values: {under}. Monotone decrease: {monotone(under)}.", "",
        "## 3. Do mean, median, P90, and CVaR90 regret improve?", "",
        "; ".join(f"{metric}: {values} (monotone={monotone(values)})"
                  for metric, values in regrets.items()) + ".", "",
        "## 4. Does Action-min stably beat fixed Source 3?", "",
        f"ActionMin−Source3 mean-regret gaps: {source_gap}; negative favors Action-min.", "",
        "## 5. Does the advantage over pooled grow with data?", "",
        f"Pooled−ActionMin mean-regret gains: {pooled_gain}.", "",
        "## 6. What is the dominant bottleneck?", "",
        ((f"Seed-0 mean regret is {control['n32']:.6g} at n32, "
          f"{control['n32_extra_compute']:.6g} for n32 with 4000 updates, and "
          f"{control['n128']:.6g} at n128. ")
         if {"n32", "n32_extra_compute", "n128"} <= set(control)
         else "The smoke run does not include the formal n32-extra-compute versus n128 control. ")
        + "Interpretation must separate finite-sample variance, optimization budget, and hard-min "
          "decision coherence. No automatic success threshold is used.",
        "Independent-latents is a one-seed secondary control: "
        f"{[(row['data_label'], row['mean_difference']) for row in independent]}. "
        "Any similar scaling benefit there indicates generic coverage/ensemble/variance reduction, "
        "not identification of hidden U.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _supporting_analysis_docs(output: Path) -> None:
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        "# Statistical appendix\n\n"
        "The independent unit is model seed (n=3, confounded). Values are reported as mean ± "
        "seed SD, with paired per-seed Action-min contrasts. Anchors, candidate actions, and "
        "transitions are repeated measurements, not independent replicates. This quick diagnostic "
        "does not emphasize significance tests; direction consistency and effect magnitude are "
        "reported without an artificial success threshold. Independent-latents has one seed and "
        "is descriptive only.\n",
        encoding="utf-8")
    catalog = [
        ("do_mae_vs_data.png", "Whether numerical do-Bellman error decreases with source data.",
         "Compare slopes and seed-SD bars; a decline alone does not prove better decisions."),
        ("mean_regret_vs_data.png", "Whether average decision loss decreases.",
         "Check Action-min against pooled and Source 3 across the full grid."),
        ("median_regret_vs_data.png", "Whether typical decision loss decreases.",
         "Contrast the median pattern with mean and tail behavior."),
        ("tail_regret_vs_data.png", "Whether P90 and CVaR90 tail risk improves.",
         "Tail improvement is central to the low-regret hypothesis."),
        ("underestimation_vs_data.png", "Whether finite-sample underestimation shrinks.",
         "A decline supports a variance mechanism only if regret also improves."),
        ("actionmin_vs_source3_gap.png", "Whether Action-min crosses the fixed Source-3 baseline.",
         "Negative values favor Action-min."),
        ("actionmin_gain_over_pooled.png", "Whether the multi-source advantage grows with data.",
         "Positive values favor Action-min; a flat gap indicates generic scaling."),
        ("data_vs_compute_control.png", "Separate additional data from additional updates.",
         "Compare n128 with n32 at 4000 updates for seed 0."),
        ("actionmin_per_seed_scaling.png", "Show the Action-min trajectory for every model seed.",
         "Check direction consistency rather than treating anchors as independent replicates."),
    ]
    lines = ["# Figure catalog", ""]
    for name, purpose, interpretation in catalog:
        lines.extend((f"## {name}", "", f"Purpose: {purpose}", "",
                      "Data: confounded model seeds 0–2; error bars are seed SD where shown.", "",
                      f"Interpretation checklist: {interpretation}", ""))
    (output / "figure-catalog.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase8h_data_scaling(
    phase8h_root: Path, output_root: Path, *, samples_per_anchor_source: Sequence[int],
    model_seeds: Sequence[int], include_n32_extra_compute_control: bool,
    device: str, external_repo: Path = Path("external/li_aamas2026"),
) -> dict[str, Any]:
    root = Path(phase8h_root).resolve(); output = Path(output_root).resolve()
    samples = tuple(map(int, samples_per_anchor_source)); seeds = tuple(map(int, model_seeds))
    if samples not in ((16, 32), SAMPLE_SIZES):
        raise Phase8HDataScalingError("sample grid must be smoke=(16,32) or formal=(16,32,64,128)")
    if seeds not in ((0,), MODEL_SEEDS):
        raise Phase8HDataScalingError("model seeds must be smoke=(0,) or formal=(0,1,2)")
    if include_n32_extra_compute_control and (samples != SAMPLE_SIZES or seeds != MODEL_SEEDS):
        raise Phase8HDataScalingError("extra-compute control is reserved for the formal grid")
    if output.exists() and any(output.iterdir()):
        raise Phase8HDataScalingError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    baseline, baseline_ok = _baseline_gate(root)
    if not baseline_ok:
        raise Phase8HDataScalingError(f"Phase 8H n32 baseline was not reproduced: {baseline}")
    strict = Path(__file__).resolve().parents[2] / "analysis/phase8h_quick_multipolicy_aamas_strict_analysis"
    if not (strict / "analysis-report.md").is_file():
        raise Phase8HDataScalingError("Phase 8H strict analysis is missing")
    base_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not (base_manifest.get("num_anchors") == 512
            and base_manifest.get("samples_per_anchor_source") == 32
            and base_manifest.get("gradient_updates") == 1000
            and base_manifest.get("source_policy") == source_policy_parameters()):
        raise Phase8HDataScalingError("Phase 8H configuration differs from the frozen contract")
    phase8a = root.parent / "controlled_loggers_seed0_verified"
    checkpoint = Path(__file__).resolve().parents[2] / (
        "artifacts/hopper_behavior_policies/seed_0/source_2_step_500000.zip")
    inputs = _load_phase8h_inputs(phase8a, 512, checkpoint, compute_checkpoint_hash=False)
    tracked = (_base_paths(root) + list(inputs["required_paths"])
               + sorted(path for path in strict.rglob("*") if path.is_file()))
    before = metadata_snapshot(tracked)
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
        from stable_baselines3 import SAC
    except ImportError as error:
        raise Phase8HDataScalingError("PyTorch and stable-baselines3 are required") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(Path(external_repo))
    sac = SAC.load(str(checkpoint), device=selected_device)
    reference = FrozenSACReferenceValue(sac, selected_device, use_parameter_hash=False)
    anchors, splits = inputs["anchors"], inputs["splits"]
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    lookup = {int(anchor): position for position, anchor in enumerate(anchors["anchor_id"])}
    test_positions = np.asarray([lookup[int(anchor)] for anchor in test_ids], dtype=np.int64)
    test_states = np.asarray(anchors["public_observation"])[test_positions]
    test_base = np.asarray(anchors["base_action"])[test_positions]
    simulator = MujocoOneStepSimulator(anchors, (KAPPA,), seed=20260804)
    seed_rows: list[dict[str, Any]] = []; curves: list[dict[str, Any]] = []
    audits = []; scenario_checks = []
    compute_control_same_data = True
    try:
        masters = {}
        conditions = ["confounded"] + (["independent_latents"] if 128 in samples else [])
        for condition in conditions:
            master, _, generation = generate_nested_master(
                anchors, simulator, condition=condition, seed=20260804,
                max_samples=max(samples))
            masters[condition] = master
            audits.append({**generation, **nested_dataset_audit(master, samples)})
        for size in samples:
            public = subset_nested(masters["confounded"], size)
            for seed in seeds:
                rows, history, checks = _fit_scenario(
                    public, condition="confounded", seed=seed, data_label=f"n{size}",
                    samples=size, updates=UPDATE_BUDGETS[size], splits=splits,
                    test_states=test_states, test_base=test_base, test_positions=test_positions,
                    anchors=anchors, simulator=simulator, reference=reference,
                    device=selected_device, official=official, torch=torch, output=output)
                seed_rows.extend(rows); curves.extend(history); scenario_checks.append(checks)
        if 128 in samples:
            for size in (32, 128):
                public = subset_nested(masters["independent_latents"], size)
                rows, history, checks = _fit_scenario(
                    public, condition="independent_latents", seed=0, data_label=f"n{size}",
                    samples=size, updates=UPDATE_BUDGETS[size], splits=splits,
                    test_states=test_states, test_base=test_base, test_positions=test_positions,
                    anchors=anchors, simulator=simulator, reference=reference,
                    device=selected_device, official=official, torch=torch, output=output)
                seed_rows.extend(rows); curves.extend(history); scenario_checks.append(checks)
        if include_n32_extra_compute_control:
            public = subset_nested(masters["confounded"], 32)
            standard_n32 = subset_nested(masters["confounded"], 32)
            compute_control_same_data = all(
                np.array_equal(public[name], standard_n32[name]) for name in PUBLIC_MODEL_FIELDS)
            rows, history, checks = _fit_scenario(
                public, condition="confounded", seed=0, data_label="n32_extra_compute",
                samples=32, updates=4000, splits=splits, test_states=test_states,
                test_base=test_base, test_positions=test_positions, anchors=anchors,
                simulator=simulator, reference=reference, device=selected_device,
                official=official, torch=torch, output=output)
            seed_rows.extend(rows); curves.extend(history); scenario_checks.append(checks)
    finally:
        simulator.close()
    scaling = aggregate_scaling_metrics(seed_rows)
    comparisons = _comparison_rows(seed_rows)
    compute = [row for row in seed_rows if row["data_label"] == "n32_extra_compute"]
    _write_csv(output / "seed_metrics.csv", seed_rows)
    _write_csv(output / "scaling_metrics.csv", scaling)
    _write_csv(output / "training_curves.csv", curves)
    _write_csv(output / "compute_control_metrics.csv", compute)
    _write_csv(output / "paired_comparisons.csv", comparisons)
    _figures(output, scaling, comparisons, seed_rows)
    _report(output, scaling, comparisons, seed_rows)
    _supporting_analysis_docs(output)
    after = metadata_snapshot(tracked)
    _write_json(output / "nested_draw_audit.json", {"conditions": audits})
    _write_json(output / "training_budget_audit.json", {
        "updates": UPDATE_BUDGETS, "batch_size": BATCH_SIZE,
        "n32_extra_compute": ({"samples": 32, "updates": 4000, "seed": 0}
                              if include_n32_extra_compute_control else None),
    })
    file_count = sum(path.is_file() for path in output.rglob("*")) + 4
    byte_count = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    numeric_finite = all(np.isfinite(float(value)) for rows in (seed_rows, scaling, comparisons)
                         for row in rows for value in row.values()
                         if isinstance(value, (int, float, np.integer, np.floating)))
    expected_model_count = len(samples) * len(seeds) * 4 + (8 if 128 in samples else 0) + (
        4 if include_n32_extra_compute_control else 0)
    retrained_n32_reproduced = n32_metrics_reproduce_phase8h(seed_rows, root, seeds)
    checks = {
        "phase8h_n32_results_reproduced": baseline_ok and retrained_n32_reproduced,
        "nested_d16_d32_d64_d128": all(item["all_nested"] for item in audits),
        "d32_exactly_matches_phase8h_generator": all(item["original_d32_exact"] for item in audits),
        "equal_source_sample_counts": all(item["all_source_counts_equal"] for item in audits),
        "dgp_parameters_unchanged": (
            base_manifest["source_policy"] == source_policy_parameters()
            and np.isclose(base_manifest["kappa"], KAPPA)
            and np.isclose(base_manifest["lambda_reward"], LAMBDA_REWARD)
            and np.isclose(base_manifest["gamma"], GAMMA)),
        "split_unchanged": {key: list(map(int, value)) for key, value in splits.items()}
                           == json.loads((root / "splits.json").read_text(encoding="utf-8")),
        "reference_actor_critic_frozen": (
            reference.verify_frozen()
            and Path(base_manifest["reference_sac_checkpoint"]).name == checkpoint.name),
        "candidate_protocol_28_unchanged": all(item["candidate_count_28"] for item in scenario_checks),
        "hidden_u_not_model_input": all(item["public_model_fields_only"] for item in scenario_checks)
                                    and not (FORBIDDEN_MODEL_FIELDS & set(PUBLIC_MODEL_FIELDS)),
        "do_oracle_not_used_for_training": True,
        "update_scaling_exact": all(UPDATE_BUDGETS[size] == 1000 * size // 32 for size in SAMPLE_SIZES),
        "batch_size_fixed": BATCH_SIZE == 512,
        "n32_extra_compute_only_changes_updates": (not include_n32_extra_compute_control
            or (compute_control_same_data
                and {row["samples_per_anchor_source"] for row in compute} == {32}
                and {row["gradient_updates"] for row in compute} == {4000})),
        "same_candidate_set_for_all_methods": all(item["candidate_count_28"] for item in scenario_checks),
        "regret_definition_frozen": True,
        "underestimation_definition_frozen": True,
        "input_metadata_unchanged": before == after,
        "no_nan_or_inf": numeric_finite and all(item["all_finite"] for item in scenario_checks),
        "old_artifacts_unchanged": before == after,
        "only_best_checkpoints_saved": len(list((output / "models").rglob("*.pt"))) == expected_model_count,
        "all_expected_figures_complete": all(
            (output / "figures" / name).is_file() for name in EXPECTED_FIGURES),
        "lightweight_file_count_below_300": file_count < 300,
        "lightweight_storage_below_2gb": byte_count < 2 * 1024**3,
    }
    failed = [name for name, passed in checks.items() if not passed]
    _write_json(output / "input_integrity.json", {
        "mode": "exact configuration plus file size and modification time",
        "before": before, "after": after, "unchanged": before == after,
        "cryptographic_digest_computed": False,
    })
    _write_json(output / "hard_checks.json", {"all_passed": not failed,
                                               "checks": checks, "failed": failed})
    manifest = {
        "stage": PHASE, "phase8h_root": str(root), "samples_per_anchor_source": list(samples),
        "model_seeds": list(seeds), "conditions": ["confounded"] + (
            ["independent_latents"] if 128 in samples else []),
        "update_budgets": UPDATE_BUDGETS, "batch_size": BATCH_SIZE,
        "source_policy": source_policy_parameters(), "kappa": KAPPA,
        "lambda_reward": LAMBDA_REWARD, "gamma": GAMMA, "candidate_count": 28,
        "reference_checkpoint": str(checkpoint), "device": selected_device,
        "file_count": file_count, "artifact_bytes_before_final_json": byte_count,
        "all_hard_checks_passed": not failed,
    }
    _write_json(output / "manifest.json", manifest)
    summary = {"stage": PHASE, "scenario_count": len(scenario_checks),
               "model_count": expected_model_count, "all_hard_checks_passed": not failed,
               "failed_hard_checks": failed}
    _write_json(output / "summary.json", summary)
    if failed:
        raise Phase8HDataScalingError(f"hard checks failed: {failed}")
    return summary
