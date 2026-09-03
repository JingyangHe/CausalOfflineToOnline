"""Phase 8E-Q: lightweight multi-source contrast go/no-go experiment."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .multisource_contrast_calibration import (
    ACTION_MARGINAL,
    FORBIDDEN_PUBLIC_FIELDS,
    Phase8EMultisourceContrastError,
    all_finite,
    audit_population_subspace,
    bic_select_rank,
    budgets_are_nested,
    calibration_features,
    closed_form_calibration,
    decision_metrics,
    do_reward_mean,
    empirical_source_mean_matrix,
    fixed_draw_public_table,
    input_hashes,
    make_source_free_model,
    multisource_behavior_probabilities,
    population_source_means,
    predict_components,
    random_balanced_query_order,
    require_all_passed,
    reward_prediction_metrics,
    shuffle_source_within_anchor_action,
    source_action_marginals,
    svd_initialization,
    validate_public_table,
    validate_source_free_model,
)
from .phase8e_multisource_contrast import (
    _calibration_pool,
    _sample_calibration_outcomes,
    _subset_universe,
    load_anchor_universe,
    select_anchor_splits,
)
from .reward_mechanism_separation import (
    index_derived_public_files,
    kappa_name,
    load_frozen_lambda_grid,
)


QUICK_SOURCE_SETTINGS: dict[str, np.ndarray] = {
    "M2_diverse": np.asarray((0.55, 0.95), dtype=np.float64),
    "M5_diverse": np.asarray((0.55, 0.65, 0.75, 0.85, 0.95), dtype=np.float64),
    "M5_redundant": np.full(5, 0.75, dtype=np.float64),
}
QUICK_LAMBDAS = (0.0, 0.01, 0.05)
QUICK_BUDGETS = (0, 16, 64)
QUICK_METHODS = ("pooled_rank0", "MSCSC_correct_source", "MSCSC_source_shuffle")
QUICK_DATA_SEED = 2026


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase8EMultisourceContrastError(f"required read-only input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_direct_root(phase8a_root: Path) -> Path:
    parent = Path(phase8a_root).resolve().parent
    candidates = (
        parent / "noncomplementary_loggers_seed0_verified" / "phase8c_direct_reward_public_grid",
        parent / "phase8c_direct_reward_public_grid",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise Phase8EMultisourceContrastError(
        "Phase 8E-Q could not locate phase8c_direct_reward_public_grid beside phase8a-root")


def _find_lambda_grid(direct_root: Path) -> Path:
    project = Path(__file__).resolve().parents[2]
    candidates = (
        direct_root / "frozen_lambda_grid.json",
        direct_root.parent / "phase8c_reward_mechanism_separation" / "frozen_lambda_grid.json",
        project / "analysis" / "phase8b_rs_low_dose_threshold_audit" / "frozen_lambda_grid.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise Phase8EMultisourceContrastError("frozen lambda information is unavailable")


def resolve_quick_inputs(phase8a_root: Path, lambda_values: Sequence[float]) -> dict[str, Any]:
    phase8a = Path(phase8a_root).resolve()
    direct = _find_direct_root(phase8a)
    grid_path = _find_lambda_grid(direct)
    required = [
        phase8a / "manifest.json", phase8a / "hard_checks.json",
        phase8a / kappa_name(0.0) / "do_oracle_raw.npz",
        direct / "manifest.json", direct / "hard_checks.json", direct / "splits.json", grid_path,
    ]
    for path in required:
        if not path.is_file():
            raise Phase8EMultisourceContrastError(f"required read-only input is missing: {path}")
    require_all_passed(phase8a / "hard_checks.json")
    require_all_passed(direct / "hard_checks.json")
    frozen, frozen_record = load_frozen_lambda_grid(grid_path)
    if not all(any(np.isclose(value, item) for item in frozen) for value in lambda_values):
        raise Phase8EMultisourceContrastError("requested lambda is absent from the frozen grid")
    index = index_derived_public_files(direct)
    needed = [(0.0, float(value), "confounded") for value in lambda_values]
    needed.append((0.0, 0.05, "independent_latents"))
    missing = [key for key in needed if key not in index]
    if missing:
        raise Phase8EMultisourceContrastError(f"direct reward scenario is unavailable: {missing[0]}")
    required.extend(index[key] for key in needed)
    return {
        "phase8a": phase8a, "direct": direct, "grid_path": grid_path,
        "frozen_grid": frozen, "frozen_record": frozen_record,
        "public_index": index, "required_paths": tuple(dict.fromkeys(required)),
    }


def quick_scenarios(source_settings: Sequence[str], lambda_values: Sequence[float]
                    ) -> list[dict[str, Any]]:
    scenarios = [
        {"setting": setting, "lambda_reward": float(dose), "condition": "confounded"}
        for setting in source_settings for dose in lambda_values
    ]
    if "M5_diverse" in source_settings and any(np.isclose(lambda_values, 0.05)):
        scenarios.append({"setting": "M5_diverse", "lambda_reward": 0.05,
                          "condition": "independent_latents"})
    return scenarios


def preflight_estimate(scenario_count: int, model_seed_count: int,
                       offline_sample_budget: int) -> dict[str, Any]:
    model_count = scenario_count * model_seed_count * len(QUICK_METHODS)
    scenario_bytes = scenario_count * offline_sample_budget * (4 + 2 + 4 + 4)
    model_bytes = model_count * 400_000
    calibration_bytes = scenario_count * 5 * 64 * (4 + 4)
    estimated_files = 16 + scenario_count + model_count + 4
    return {
        "scenario_count": scenario_count,
        "model_count": model_count,
        "estimated_file_count": estimated_files,
        "estimated_disk_bytes": scenario_bytes + model_bytes + calibration_bytes + 8_000_000,
        "estimated_disk_mib": (scenario_bytes + model_bytes + calibration_bytes + 8_000_000) / 2**20,
        "scaling_rule": "files scale with scenario x method x model_seed, not row x budget x replicate",
    }


def _row_mask(public: Mapping[str, np.ndarray], anchor_ids: Sequence[int]) -> np.ndarray:
    return np.isin(np.asarray(public["anchor_id"], dtype=np.int64),
                   np.asarray(anchor_ids, dtype=np.int64))


def _subset_public(public: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in public.items()}


def _network_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for prefix in ("g.", "h."):
        for name, value in sorted(model.state_dict().items()):
            if name.startswith(prefix):
                digest.update(name.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _training_arrays(public: Mapping[str, np.ndarray], train_mask: np.ndarray,
                     device: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch
    selected_device = ("cuda" if device == "auto" and torch.cuda.is_available()
                       else "cpu" if device == "auto" else device)
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise Phase8EMultisourceContrastError("CUDA was requested but is unavailable")
    raw_x = np.concatenate((np.asarray(public["observation"], dtype=np.float64),
                            np.asarray(public["commanded_action"], dtype=np.float64)), axis=1)
    train_x = raw_x[train_mask]
    train_reward = np.asarray(public["reward"], dtype=np.float64)[train_mask]
    x_mean = train_x.mean(axis=0)
    x_std = np.maximum(train_x.std(axis=0), 1e-6)
    y_mean = float(train_reward.mean())
    y_std = max(float(train_reward.std()), 1e-6)
    tensors = {
        "x": torch.as_tensor((raw_x - x_mean) / x_std, dtype=torch.float32,
                             device=selected_device),
        "reward": torch.as_tensor((np.asarray(public["reward"]) - y_mean) / y_std,
                                  dtype=torch.float32, device=selected_device),
        "source": torch.as_tensor(public["source_id"], dtype=torch.long,
                                  device=selected_device),
        "action": torch.as_tensor(public["action_index"], dtype=torch.long,
                                  device=selected_device),
        "train_index": np.flatnonzero(train_mask),
        "validation_index": np.flatnonzero(~train_mask),
        "device": selected_device,
    }
    normalization = {"x_mean": x_mean, "x_std": x_std,
                     "reward_mean": np.asarray(y_mean), "reward_std": np.asarray(y_std)}
    return tensors, normalization


def _validation_loss(model: Any, tensors: Mapping[str, Any], pooled: bool) -> float:
    import torch
    index = torch.as_tensor(tensors["validation_index"], dtype=torch.long,
                            device=tensors["device"])
    with torch.no_grad():
        prediction = (model.g(tensors["x"][index]) if pooled else
                      model.source_mean(tensors["x"][index], tensors["source"][index],
                                        tensors["action"][index]))
        return float((prediction - tensors["reward"][index]).square().mean().cpu())


def fit_quick_model(public: Mapping[str, np.ndarray], initialization: Any,
                    ordered_anchor_ids: Sequence[int], train_anchor_ids: Sequence[int], *,
                    seed: int, updates: int, batch_size: int, device: str,
                    pooled: bool = False) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    """Paired width-128 training with observational-validation best selection."""
    import torch
    if not validate_public_table(public) or updates <= 0:
        raise ValueError("invalid quick training input")
    train_mask = _row_mask(public, train_anchor_ids)
    if not np.any(train_mask) or np.all(train_mask):
        raise ValueError("quick training requires disjoint train and validation rows")
    tensors, normalization = _training_arrays(public, train_mask, device)
    model = make_source_free_model(
        initialization.loadings.shape[0], initialization.loadings, seed=seed, width=128
    ).to(tensors["device"])
    initial_network_hash = _network_hash(model)
    optimizer = torch.optim.Adam(model.g.parameters() if pooled else model.parameters(), lr=1e-3)

    pretrain_updates = 0 if pooled else max(1, updates // 4)
    if not pooled:
        ordered = np.asarray(ordered_anchor_ids, dtype=np.int64)
        train_public = _subset_public(public, train_mask)
        positions = np.searchsorted(ordered, np.asarray(train_public["anchor_id"], dtype=np.int64))
        key = positions * 3 + np.asarray(train_public["action_index"], dtype=np.int64)
        unique_key, first = np.unique(key, return_index=True)
        first = first[np.argsort(unique_key)]
        unique_key = np.sort(unique_key)
        all_rows = np.flatnonzero(train_mask)[first]
        target_x = tensors["x"][torch.as_tensor(all_rows, dtype=torch.long,
                                                 device=tensors["device"])]
        y_mean = float(normalization["reward_mean"])
        y_std = float(normalization["reward_std"])
        center = torch.as_tensor(
            (initialization.center_targets.reshape(-1)[unique_key] - y_mean) / y_std,
            dtype=torch.float32, device=tensors["device"])
        contrast = torch.as_tensor(
            initialization.contrast_targets.reshape(-1)[unique_key] / y_std,
            dtype=torch.float32, device=tensors["device"])
        for _ in range(pretrain_updates):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model.g(target_x) - center).square().mean()
                    + (model.h(target_x) - contrast).square().mean())
            if not torch.isfinite(loss):
                raise Phase8EMultisourceContrastError("non-finite quick SVD pretraining loss")
            loss.backward()
            optimizer.step()

    generator = np.random.default_rng(seed + 901)
    best_loss = _validation_loss(model, tensors, pooled)
    best_step = 0
    best_state = {name: value.detach().cpu().clone()
                  for name, value in model.state_dict().items()}
    train_index = np.asarray(tensors["train_index"], dtype=np.int64)
    schedule_digest = hashlib.sha256()
    for step in range(1, updates + 1):
        chosen = train_index[generator.integers(0, len(train_index),
                                                size=min(batch_size, len(train_index)))]
        schedule_digest.update(np.asarray(chosen, dtype=np.int64).tobytes())
        index = torch.as_tensor(chosen, dtype=torch.long, device=tensors["device"])
        optimizer.zero_grad(set_to_none=True)
        prediction = (model.g(tensors["x"][index]) if pooled else
                      model.source_mean(tensors["x"][index], tensors["source"][index],
                                        tensors["action"][index]))
        loss = (prediction - tensors["reward"][index]).square().mean()
        if not torch.isfinite(loss):
            raise Phase8EMultisourceContrastError("non-finite quick observational loss")
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == updates:
            validation = _validation_loss(model, tensors, pooled)
            if validation < best_loss:
                best_loss, best_step = validation, step
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, normalization, {
        "best_validation_mse_standardized": best_loss,
        "best_step": best_step, "observational_updates": updates,
        "svd_pretrain_updates": pretrain_updates, "batch_size": batch_size,
        "initial_network_hash": initial_network_hash,
        "minibatch_schedule_hash": schedule_digest.hexdigest(),
    }


def _save_best_checkpoint(path: Path, model: Any, normalization: Mapping[str, np.ndarray],
                          metadata: Mapping[str, Any]) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "source_count": model.source_count,
                "loadings": model.raw_loadings.detach().cpu(),
                "width": 128, "normalization": dict(normalization),
                "metadata": dict(metadata)}, path)


def load_quick_checkpoint(path: Path, device: str = "cpu") -> tuple[Any, dict[str, np.ndarray],
                                                                      dict[str, Any]]:
    """Load a best-only width-128 Phase 8E-Q checkpoint."""
    import torch
    record = torch.load(path, map_location=device, weights_only=False)
    model = make_source_free_model(
        int(record["source_count"]), np.asarray(record["loadings"]), seed=0,
        width=int(record["width"]))
    model.load_state_dict(record["state_dict"])
    model.to(device)
    model.eval()
    normalization = {key: np.asarray(value)
                     for key, value in record["normalization"].items()}
    return model, normalization, dict(record["metadata"])


def _scenario_key(setting: str, dose: float, condition: str) -> str:
    token = str(dose).replace(".", "p")
    return f"{setting}__lambda_{token}__{condition}"


def _metric_row(labels: Mapping[str, Any], truth: np.ndarray,
                prediction: np.ndarray, **extra: Any) -> dict[str, Any]:
    return {**labels, **extra, **reward_prediction_metrics(truth, prediction),
            **decision_metrics(truth, prediction)}


def _test_prediction(g: np.ndarray, h: np.ndarray,
                     coefficients: np.ndarray, rank: int) -> np.ndarray:
    actions = np.tile(np.arange(3, dtype=np.int8), len(g))
    features = calibration_features(actions, h.reshape(-1), rank=rank)
    return (g.reshape(-1) + features @ coefficients).reshape(len(g), 3)


def _save_scenario_rows(path: Path, public: Mapping[str, np.ndarray],
                        shared_ids: np.ndarray, configuration: Mapping[str, Any]) -> None:
    positions = np.searchsorted(shared_ids, np.asarray(public["anchor_id"], dtype=np.int64))
    arrays = {
        "row_index": (positions * 3 + np.asarray(public["action_index"], dtype=np.int64)).astype(np.int32),
        "source_id": np.asarray(public["source_id"], dtype=np.int16),
        "reward": np.asarray(public["reward"], dtype=np.float32),
        "weight": np.asarray(public["row_weight"], dtype=np.float32),
        "configuration_json": np.asarray(json.dumps(configuration, sort_keys=True)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = ("setting", "lambda_reward", "condition", "seed", "method", "calibration_budget")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, group))
        for metric in ("do_mae", "top_set_disagreement", "mean_regret", "selected_rank"):
            record[metric] = float(np.mean([float(row[metric]) for row in values]))
        record["calibration_replicates"] = len(values)
        result.append(record)
    return result


def _summary_table(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = ("setting", "lambda_reward", "condition", "method", "calibration_budget")
    for row in seed_rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, group))
        for metric in ("do_mae", "top_set_disagreement", "mean_regret", "selected_rank"):
            data = np.asarray([float(row[metric]) for row in values])
            record[metric + "_mean"] = float(data.mean())
            record[metric + "_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        record["seed_count"] = len(values)
        result.append(record)
    return result


def _select_summary(rows: Sequence[Mapping[str, Any]], setting: str, method: str,
                    budget: int, dose: float = 0.05,
                    condition: str = "confounded") -> Mapping[str, Any]:
    matches = [row for row in rows if row["setting"] == setting and row["method"] == method
               and int(row["calibration_budget"]) == budget
               and np.isclose(float(row["lambda_reward"]), dose)
               and row["condition"] == condition]
    if len(matches) != 1:
        raise Phase8EMultisourceContrastError("quick summary cell is missing or duplicated")
    return matches[0]


def _comparison_rows(summary_rows: Sequence[Mapping[str, Any]],
                     seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("Correct - Shuffle, M=5 diverse",
         ("M5_diverse", "MSCSC_correct_source", 64),
         ("M5_diverse", "MSCSC_source_shuffle", 64)),
        ("M=5 diverse - M=2 diverse",
         ("M5_diverse", "MSCSC_correct_source", 64),
         ("M2_diverse", "MSCSC_correct_source", 64)),
        ("M=5 redundant - M=5 diverse",
         ("M5_redundant", "MSCSC_correct_source", 64),
         ("M5_diverse", "MSCSC_correct_source", 64)),
        ("B=64 - B=0",
         ("M5_diverse", "MSCSC_correct_source", 64),
         ("M5_diverse", "MSCSC_correct_source", 0)),
    )
    result = []
    for name, left_key, right_key in definitions:
        left = _select_summary(summary_rows, *left_key)
        right = _select_summary(summary_rows, *right_key)
        record: dict[str, Any] = {"comparison": name}
        per_seed: dict[str, list[float]] = {
            "do_mae": [], "top_set_disagreement": [], "mean_regret": []}
        for seed in sorted({int(row["seed"]) for row in seed_rows}):
            def seed_cell(key: tuple[str, str, int]) -> Mapping[str, Any]:
                setting, method, budget = key
                matches = [row for row in seed_rows
                           if row["setting"] == setting and row["method"] == method
                           and int(row["calibration_budget"]) == budget
                           and int(row["seed"]) == seed
                           and np.isclose(float(row["lambda_reward"]), 0.05)
                           and row["condition"] == "confounded"]
                if len(matches) != 1:
                    raise Phase8EMultisourceContrastError(
                        "quick per-seed comparison cell is missing or duplicated")
                return matches[0]

            left_seed, right_seed = seed_cell(left_key), seed_cell(right_key)
            for metric in per_seed:
                per_seed[metric].append(float(left_seed[metric]) - float(right_seed[metric]))
        for metric, output_name in (("do_mae", "do_mae_difference"),
                                    ("top_set_disagreement", "rank_difference"),
                                    ("mean_regret", "regret_difference")):
            differences = np.asarray(per_seed[metric], dtype=np.float64)
            record[output_name] = float(differences.mean())
            record[output_name + "_sd"] = float(differences.std(ddof=1))
            record[output_name + "_negative_seed_fraction"] = float(np.mean(differences < 0))
            record[output_name + "_positive_seed_fraction"] = float(np.mean(differences > 0))
            record[output_name + "_per_seed"] = json.dumps(differences.tolist())
        # Cross-check that paired-seed and independently aggregated means agree.
        if not np.isclose(record["do_mae_difference"],
                          float(left["do_mae_mean"]) - float(right["do_mae_mean"])):
            raise Phase8EMultisourceContrastError("paired comparison aggregation is inconsistent")
        result.append(record)
    return result


def _make_figures(output: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def comparison_figure(filename: str, labels: Sequence[tuple[str, str]], metric: str) -> None:
        plt.figure(figsize=(7, 4.5))
        names, values = [], []
        for label, setting_method in labels:
            setting, method = setting_method.split("|")
            row = _select_summary(summary_rows, setting, method, 64)
            names.append(label)
            values.append(float(row[metric + "_mean"]))
        plt.bar(names, values)
        plt.ylabel(metric.replace("_", " "))
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=220)
        plt.close()

    comparison_figure("correct_source_vs_shuffle.png",
                      (("Correct", "M5_diverse|MSCSC_correct_source"),
                       ("Shuffle", "M5_diverse|MSCSC_source_shuffle")), "do_mae")
    comparison_figure("M2_vs_M5_diverse.png",
                      (("M=2", "M2_diverse|MSCSC_correct_source"),
                       ("M=5", "M5_diverse|MSCSC_correct_source")), "do_mae")
    comparison_figure("diverse_vs_redundant.png",
                      (("Diverse", "M5_diverse|MSCSC_correct_source"),
                       ("Redundant", "M5_redundant|MSCSC_correct_source")), "do_mae")
    selected = [row for row in summary_rows if row["setting"] == "M5_diverse"
                and row["method"] == "MSCSC_correct_source"
                and np.isclose(float(row["lambda_reward"]), 0.05)
                and row["condition"] == "confounded"]
    plt.figure(figsize=(7, 4.5))
    budgets = sorted({int(row["calibration_budget"]) for row in selected})
    for metric in ("do_mae", "top_set_disagreement", "mean_regret"):
        values = [float(next(row for row in selected
                             if int(row["calibration_budget"]) == budget)[metric + "_mean"])
                  for budget in budgets]
        plt.plot(budgets, values, marker="o", label=metric)
    plt.xlabel("Calibration budget")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "metrics_vs_calibration_budget.png", dpi=220)
    plt.close()


def _report(output: Path, summary_rows: Sequence[Mapping[str, Any]],
            comparisons: Sequence[Mapping[str, Any]], conclusions: Mapping[str, bool]) -> None:
    primary = [row for row in summary_rows if np.isclose(float(row["lambda_reward"]), 0.05)
               and row["condition"] == "confounded"]
    lines = ["# Phase 8E-Q Quick Go/No-Go Report", "",
             "This is an exploratory quick gate in the controlled three-action, one-step setting.", "",
             "## Table 1 — Primary metrics (lambda=0.05)", "",
             "| Setting | Method | B | Do MAE | Rank error | Regret |", "|---|---|---:|---:|---:|---:|"]
    for row in primary:
        lines.append(f"| {row['setting']} | {row['method']} | {int(row['calibration_budget'])} | "
                     f"{row['do_mae_mean']:.6g} +/- {row['do_mae_sd']:.3g} | "
                     f"{row['top_set_disagreement_mean']:.6g} +/- "
                     f"{row['top_set_disagreement_sd']:.3g} | "
                     f"{row['mean_regret_mean']:.6g} +/- {row['mean_regret_sd']:.3g} |")
    lines.extend(("", "## Table 2 — Go/no-go contrasts", "",
                  "| Comparison | Do MAE difference | Rank difference | Regret difference |",
                  "|---|---:|---:|---:|"))
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['do_mae_difference']:.6g} +/- "
                     f"{row['do_mae_difference_sd']:.3g} | {row['rank_difference']:.6g} +/- "
                     f"{row['rank_difference_sd']:.3g} | {row['regret_difference']:.6g} +/- "
                     f"{row['regret_difference_sd']:.3g} |")
    lines.extend(("", "Paired seed-direction consistency (fraction of three seeds in the "
                  "reported mean direction):", ""))
    for row in comparisons:
        signs = []
        for metric in ("do_mae_difference", "rank_difference", "regret_difference"):
            suffix = "negative_seed_fraction" if float(row[metric]) < 0 else "positive_seed_fraction"
            signs.append(float(row[f"{metric}_{suffix}"]))
        lines.append(f"- {row['comparison']}: " + ", ".join(f"{value:.3g}" for value in signs))
    lines.extend(("", "## Direct answers", ""))
    for question, passed in conclusions.items():
        lines.append(f"- {question}: {'positive trend' if passed else 'not supported'}")
    lines.extend(("", "No confirmatory significance claim is made from three model seeds."))
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase8e_quick(
    phase8a_root: Path,
    output_root: Path,
    *,
    num_anchors: int,
    source_settings: Sequence[str],
    lambda_values: Sequence[float],
    reward_noise_std: float,
    offline_sample_budget: int,
    model_seeds: Sequence[int],
    gradient_updates: int,
    calibration_budgets: Sequence[int],
    calibration_replicates: int,
    device: str,
    data_seed: int = QUICK_DATA_SEED,
) -> dict[str, Any]:
    settings = tuple(source_settings)
    doses = tuple(map(float, lambda_values))
    seeds = tuple(map(int, model_seeds))
    budgets = tuple(sorted(set(map(int, calibration_budgets))))
    if (num_anchors != 512 or len(settings) != 3 or set(settings) != set(QUICK_SOURCE_SETTINGS)
            or len(doses) != len(QUICK_LAMBDAS) or not np.allclose(doses, QUICK_LAMBDAS)
            or not np.isclose(reward_noise_std, 0.02)
            or offline_sample_budget != 49152 or budgets != QUICK_BUDGETS
            or calibration_replicates != 5 or seeds != (0, 1, 2)
            or gradient_updates != 1000):
        raise Phase8EMultisourceContrastError("Phase 8E-Q fixed quick configuration was changed")
    inputs = resolve_quick_inputs(phase8a_root, doses)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise Phase8EMultisourceContrastError(f"output directory is not empty: {output}")
    scenarios = quick_scenarios(settings, doses)
    hashes_before = input_hashes(inputs["required_paths"])
    output.mkdir(parents=True, exist_ok=True)
    for name in ("scenario_data", "models", "figures"):
        (output / name).mkdir(exist_ok=True)
    estimate = preflight_estimate(len(scenarios), len(seeds), offline_sample_budget)
    _write_json(output / "preflight_estimate.json", estimate)
    print(f"preflight: {estimate['scenario_count']} scenarios, "
          f"{estimate['model_count']} models, "
          f"{estimate['estimated_disk_mib']:.1f} MiB estimated", flush=True)

    zero_public = inputs["public_index"][(0.0, 0.0, "confounded")]
    universe = load_anchor_universe(inputs["phase8a"], zero_public, 0.0)
    split_record = _read_json(inputs["direct"] / "splits.json")
    splits = select_anchor_splits(split_record, universe["anchor_id"], num_anchors, max(budgets))
    split_sets = [set(map(int, values)) for values in splits.values()]
    split_disjoint = not any(split_sets[i] & split_sets[j] for i in range(4)
                             for j in range(i + 1, 4))
    training_ids = np.sort(np.concatenate((splits["train"], splits["observational_validation"])))
    training_universe = _subset_universe(universe, training_ids)
    calibration_universe = _subset_universe(universe, splits["do_calibration_pool"])
    test_universe = _subset_universe(universe, splits["test"])
    shared_ids = np.asarray(universe["anchor_id"], dtype=np.int64)
    np.savez_compressed(output / "shared_arrays.npz", anchor_id=shared_ids,
                        observation=np.asarray(universe["observation"], dtype=np.float32),
                        commanded_action=np.asarray(universe["commanded_action"], dtype=np.float32))
    _write_json(output / "splits.json", {name: list(map(int, values))
                                          for name, values in splits.items()})

    method_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    uncalibrated_rank1_rows: list[dict[str, Any]] = []
    calibration_arrays: dict[str, np.ndarray] = {}
    action_marginal_checks: list[bool] = []
    sample_counts: set[int] = set()
    shuffle_checks: list[bool] = []
    paired_initialization_checks: list[bool] = []
    paired_normalization_checks: list[bool] = []
    paired_schedule_checks: list[bool] = []
    paired_data_checks: list[bool] = []
    source_free_checks: list[bool] = []
    hidden_free_checks: list[bool] = []
    nested_checks: list[bool] = []
    calibration_independence_checks: list[bool] = []
    redundant_zero = True
    independent_zero = True
    base_zero = True
    saved_models = 0
    expected_models = len(scenarios) * len(seeds) * len(QUICK_METHODS)
    train_universe = _subset_universe(training_universe, splits["train"])
    calibration_streams: dict[tuple[float, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    calibration_stream_keys: dict[tuple[float, str], str] = {}

    for scenario in scenarios:
        setting = scenario["setting"]
        dose = float(scenario["lambda_reward"])
        condition = str(scenario["condition"])
        p_values = QUICK_SOURCE_SETTINGS[setting]
        source_count = len(p_values)
        behavior = multisource_behavior_probabilities(p_values)
        action_marginal_checks.append(np.allclose(
            source_action_marginals(behavior), ACTION_MARGINAL[None, :], atol=1e-15, rtol=0))
        public, _hidden = fixed_draw_public_table(
            training_universe["anchor_id"], training_universe["observation"],
            training_universe["commanded_action"], training_universe["reward_branches"],
            p_values, kappa=0.0, lambda_reward=dose, sigma_reward=reward_noise_std,
            condition=condition, sample_budget=offline_sample_budget, seed=data_seed + 1009)
        sample_counts.add(len(public["reward"]))
        hidden_free_checks.append(validate_public_table(public)
                                  and not FORBIDDEN_PUBLIC_FIELDS.intersection(public))
        scenario_key = _scenario_key(setting, dose, condition)
        print(f"scenario: {scenario_key}", flush=True)
        _save_scenario_rows(
            output / "scenario_data" / f"{scenario_key}.npz", public, shared_ids,
            {**scenario, "source_count": source_count, "p_values": p_values.tolist(),
             "sample_budget": offline_sample_budget, "data_seed": data_seed})
        train_mask = _row_mask(public, splits["train"])
        train_public = _subset_public(public, train_mask)
        correct_init = svd_initialization(empirical_source_mean_matrix(
            train_public, source_count, splits["train"]))
        shuffled_source = shuffle_source_within_anchor_action(
            public["anchor_id"], public["action_index"], public["source_id"], data_seed + 31)
        shuffled_public = dict(public)
        shuffled_public["source_id"] = shuffled_source
        paired_data_checks.append(all(
            np.array_equal(public[name], shuffled_public[name])
            for name in public if name != "source_id"))
        for anchor in np.unique(public["anchor_id"]):
            for action in range(3):
                mask = ((public["anchor_id"] == anchor) & (public["action_index"] == action))
                shuffle_checks.append(np.array_equal(np.sort(public["source_id"][mask]),
                                                     np.sort(shuffled_source[mask])))
        shuffled_train = _subset_public(shuffled_public, train_mask)
        shuffled_init = svd_initialization(empirical_source_mean_matrix(
            shuffled_train, source_count, splits["train"]))

        population = population_source_means(
            training_universe["reward_branches"], p_values, dose, condition)
        shuffled_empirical = empirical_source_mean_matrix(
            shuffled_train, source_count, splits["train"])
        correct_empirical = empirical_source_mean_matrix(
            train_public, source_count, splits["train"])
        for action, action_name in enumerate(("minus", "base", "plus")):
            population_audit = audit_population_subspace(
                population[:, :, action], do_reward_mean(training_universe["reward_branches"])[:, action])
            correct_audit = audit_population_subspace(
                correct_empirical[:, :, action], do_reward_mean(
                    train_universe["reward_branches"])[:, action])
            shuffle_audit = audit_population_subspace(
                shuffled_empirical[:, :, action], do_reward_mean(
                    train_universe["reward_branches"])[:, action])
            subspace_rows.append({
                "setting": setting, "lambda_reward": dose, "condition": condition,
                "action": action_name,
                "singular_values": json.dumps(population_audit.singular_values.tolist()),
                "rank1_explained_variance": population_audit.rank1_explained_variance,
                "source_reconstruction_error": correct_audit.rank1_reconstruction_error,
                "correct_centered_norm": correct_audit.centered_norm,
                "shuffle_centered_norm": shuffle_audit.centered_norm,
            })
            if setting == "M5_redundant":
                redundant_zero &= population_audit.centered_norm < 1e-12
            if condition == "independent_latents":
                independent_zero &= population_audit.centered_norm < 1e-12
            if action_name == "base":
                base_zero &= population_audit.centered_norm < 1e-12

        pool_template = _calibration_pool(
            calibration_universe, calibration_universe["anchor_id"],
            np.zeros((len(calibration_universe["anchor_id"]), 3)),
            np.zeros((len(calibration_universe["anchor_id"]), 3)))
        stream_identity = (dose, condition)
        if stream_identity not in calibration_streams:
            orders, rewards = [], []
            for replicate in range(calibration_replicates):
                order = random_balanced_query_order(
                    pool_template["anchor_id"], pool_template["action_index"],
                    max(budgets), data_seed + 1000 * replicate)
                reward, u = _sample_calibration_outcomes(
                    order, pool_template, calibration_universe["reward_branches"], dose,
                    reward_noise_std, replicate, data_seed + 333)
                _, reversed_u = _sample_calibration_outcomes(
                    order[::-1], pool_template, calibration_universe["reward_branches"], dose,
                    reward_noise_std, replicate, data_seed + 333)
                nested_checks.append(budgets_are_nested(order, budgets))
                calibration_independence_checks.append(np.array_equal(u, reversed_u))
                orders.append(order.astype(np.int32))
                rewards.append(reward.astype(np.float32))
            stream_key = f"stream_{len(calibration_streams)}"
            calibration_streams[stream_identity] = (orders, rewards)
            calibration_stream_keys[stream_identity] = stream_key
            calibration_arrays[f"{stream_key}__order"] = np.stack(orders)
            calibration_arrays[f"{stream_key}__reward"] = np.stack(rewards)
        orders, rewards = calibration_streams[stream_identity]

        truth = do_reward_mean(test_universe["reward_branches"])
        population_test = population_source_means(
            test_universe["reward_branches"], p_values, dose, condition)
        for seed in seeds:
            correct, correct_norm, correct_history = fit_quick_model(
                public, correct_init, splits["train"], splits["train"], seed=seed,
                updates=gradient_updates, batch_size=512, device=device)
            shuffled, shuffled_norm, shuffled_history = fit_quick_model(
                shuffled_public, shuffled_init, splits["train"], splits["train"], seed=seed,
                updates=gradient_updates, batch_size=512, device=device)
            pooled, pooled_norm, pooled_history = fit_quick_model(
                public, correct_init, splits["train"], splits["train"], seed=seed,
                updates=gradient_updates, batch_size=512, device=device, pooled=True)
            paired_initialization_checks.append(
                correct_history["initial_network_hash"] == shuffled_history["initial_network_hash"])
            paired_normalization_checks.append(all(
                np.array_equal(correct_norm[key], shuffled_norm[key]) for key in correct_norm))
            paired_schedule_checks.append(
                correct_history["minibatch_schedule_hash"]
                == shuffled_history["minibatch_schedule_hash"])
            source_free_checks.extend((validate_source_free_model(correct),
                                       validate_source_free_model(shuffled),
                                       validate_source_free_model(pooled)))
            for method, model, norm, history in (
                ("MSCSC_correct_source", correct, correct_norm, correct_history),
                ("MSCSC_source_shuffle", shuffled, shuffled_norm, shuffled_history),
                ("pooled_rank0", pooled, pooled_norm, pooled_history),
            ):
                _save_best_checkpoint(
                    output / "models" / method / scenario_key / f"seed_{seed}.pt",
                    model, norm, {"method": method, "scenario": scenario,
                                  "seed": seed, **history})
                saved_models += 1
            print(f"completed models: {saved_models}/{expected_models}", flush=True)

            correct_g_test, correct_h_test = predict_components(
                correct, correct_norm, test_universe["observation"],
                test_universe["commanded_action"], device)
            shuffle_g_test, shuffle_h_test = predict_components(
                shuffled, shuffled_norm, test_universe["observation"],
                test_universe["commanded_action"], device)
            pooled_g_test, _ = predict_components(
                pooled, pooled_norm, test_universe["observation"],
                test_universe["commanded_action"], device)
            correct_g_cal, correct_h_cal = predict_components(
                correct, correct_norm, calibration_universe["observation"],
                calibration_universe["commanded_action"], device)
            shuffle_g_cal, shuffle_h_cal = predict_components(
                shuffled, shuffled_norm, calibration_universe["observation"],
                calibration_universe["commanded_action"], device)
            pooled_g_cal, _ = predict_components(
                pooled, pooled_norm, calibration_universe["observation"],
                calibration_universe["commanded_action"], device)

            loading = correct.normalized_loadings().detach().cpu().numpy()
            predicted_sources = correct_g_test[None, :, :] + loading[:, None, :] * correct_h_test[None, :, :]
            source_reconstruction = float(np.mean(np.abs(predicted_sources - population_test)))
            shuffle_loading = shuffled.normalized_loadings().detach().cpu().numpy()
            shuffle_sources = (shuffle_g_test[None, :, :]
                               + shuffle_loading[:, None, :] * shuffle_h_test[None, :, :])
            for method, contrast, reconstruction in (
                ("MSCSC_correct_source", correct_h_test, source_reconstruction),
                ("MSCSC_source_shuffle", shuffle_h_test,
                 float(np.mean(np.abs(shuffle_sources - population_test)))),
            ):
                uncalibrated_rank1_rows.append({
                    "setting": setting, "source_count": source_count,
                    "lambda_reward": dose, "condition": condition, "seed": seed,
                    "method": method,
                    "uncalibrated_rank1_contrast_rms": float(np.sqrt(np.mean(contrast ** 2))),
                    "source_reconstruction_mae": reconstruction,
                    "do_prediction_not_claimed_without_tau": True,
                })
            pools = {
                "MSCSC_correct_source": _calibration_pool(
                    calibration_universe, calibration_universe["anchor_id"],
                    correct_g_cal, correct_h_cal),
                "MSCSC_source_shuffle": _calibration_pool(
                    calibration_universe, calibration_universe["anchor_id"],
                    shuffle_g_cal, shuffle_h_cal),
                "pooled_rank0": _calibration_pool(
                    calibration_universe, calibration_universe["anchor_id"],
                    pooled_g_cal, np.zeros_like(pooled_g_cal)),
            }
            test_components = {
                "MSCSC_correct_source": (correct_g_test, correct_h_test),
                "MSCSC_source_shuffle": (shuffle_g_test, shuffle_h_test),
                "pooled_rank0": (pooled_g_test, np.zeros_like(pooled_g_test)),
            }
            for method in QUICK_METHODS:
                pool = pools[method]
                g_test, h_test = test_components[method]
                for replicate in range(calibration_replicates):
                    order = orders[replicate]
                    full_reward = rewards[replicate].astype(np.float64)
                    for budget in budgets:
                        if budget == 0:
                            prediction, selected_rank = g_test, 0
                        else:
                            prefix = order[:budget]
                            reward = full_reward[:budget]
                            base = np.asarray(pool["g"])[prefix]
                            rank0 = closed_form_calibration(
                                base, reward, np.asarray(pool["features_rank0"])[prefix])
                            if method == "pooled_rank0":
                                fit, selected_rank = rank0, 0
                            else:
                                rank1 = closed_form_calibration(
                                    base, reward, np.asarray(pool["features_rank1"])[prefix])
                                selected_rank = bic_select_rank(
                                    rank0.residual_sum_squares, rank1.residual_sum_squares, budget)
                                fit = rank1 if selected_rank else rank0
                            prediction = _test_prediction(g_test, h_test, fit.coefficients, selected_rank)
                        labels = {"setting": setting, "source_count": source_count,
                                  "lambda_reward": dose, "condition": condition,
                                  "seed": seed, "method": method,
                                  "calibration_budget": budget,
                                  "calibration_replicate": replicate,
                                  "selected_rank": selected_rank}
                        row = _metric_row(labels, truth, prediction)
                        method_rows.append(row)
                        if budget > 0:
                            calibration_rows.append(row)

    calibration_arrays["stream_registry_json"] = np.asarray(json.dumps({
        _scenario_key(scenario["setting"], float(scenario["lambda_reward"]), scenario["condition"]):
            calibration_stream_keys[(float(scenario["lambda_reward"]), scenario["condition"])]
        for scenario in scenarios
    }, sort_keys=True))
    np.savez_compressed(output / "calibration_sequences.npz", **calibration_arrays)
    seed_rows = _aggregate_seed_rows(method_rows)
    summary_rows = _summary_table(seed_rows)
    comparisons = _comparison_rows(summary_rows, seed_rows)
    comparison_map = {row["comparison"]: row for row in comparisons}
    def all_better(name: str) -> bool:
        row = comparison_map[name]
        return all(float(row[key]) < 0 for key in
                   ("do_mae_difference", "rank_difference", "regret_difference"))
    def all_worse(name: str) -> bool:
        row = comparison_map[name]
        return all(float(row[key]) > 0 for key in
                   ("do_mae_difference", "rank_difference", "regret_difference"))
    lambda_zero = [row for row in summary_rows if row["setting"] == "M5_diverse"
                   and np.isclose(float(row["lambda_reward"]), 0.0)
                   and row["condition"] == "confounded" and int(row["calibration_budget"]) > 0
                   and row["method"] == "MSCSC_correct_source"]
    calibration_trend = []
    for metric in ("do_mae_mean", "top_set_disagreement_mean", "mean_regret_mean"):
        values = [float(_select_summary(summary_rows, "M5_diverse",
                                        "MSCSC_correct_source", budget)[metric])
                  for budget in budgets]
        calibration_trend.append(values[1] <= values[0] and values[2] <= values[1]
                                 and values[2] < values[0])
    lambda_zero_safe = []
    for budget in budgets[1:]:
        adaptive = _select_summary(summary_rows, "M5_diverse", "MSCSC_correct_source",
                                   budget, dose=0.0)
        pooled = _select_summary(summary_rows, "M5_diverse", "pooled_rank0",
                                 budget, dose=0.0)
        lambda_zero_safe.append(all(
            float(adaptive[field]) <= float(pooled[field])
            for field in ("do_mae_mean", "top_set_disagreement_mean", "mean_regret_mean")))
    conclusions = {
        "correct_source_beats_shuffle": all_better("Correct - Shuffle, M=5 diverse"),
        "M5_diverse_beats_M2_diverse": all_better("M=5 diverse - M=2 diverse"),
        "M5_redundant_has_no_equal_benefit": all_worse("M=5 redundant - M=5 diverse"),
        "B0_to_B16_to_B64_improves_all_three_metrics": bool(all(calibration_trend)),
        "lambda_zero_safely_returns_to_rank0": bool(lambda_zero)
            and all(float(row["selected_rank_mean"]) < 0.5 for row in lambda_zero)
            and all(lambda_zero_safe),
    }
    conclusions["continuous_action_extension_recommended"] = bool(all(
        conclusions.values()))

    hashes_after = input_hashes(inputs["required_paths"])
    unchanged = hashes_before == hashes_after
    calibration_action = np.tile(np.arange(3), 8)
    calibration_h = np.linspace(-1, 1, len(calibration_action))
    x0 = calibration_features(calibration_action, calibration_h, rank=0)
    x1 = calibration_features(calibration_action, calibration_h, rank=1)
    theta0 = np.asarray((0.1, -0.2, 0.3))
    theta1 = np.asarray((0.1, -0.2, 0.3, 0.4, -0.1, 0.2))
    calibration_formula_check = (
        np.allclose(closed_form_calibration(np.zeros(len(x0)), x0 @ theta0, x0).coefficients, theta0)
        and np.allclose(closed_form_calibration(np.zeros(len(x1)), x1 @ theta1, x1).coefficients, theta1))
    formal_roots = list(Path(inputs["phase8a"]).parent.glob("phase8e_multisource_contrast_calibration*"))
    hard_checks = {
        "all_source_action_marginals_equal": bool(all(action_marginal_checks)),
        "all_source_settings_same_total_sample_count": sample_counts == {offline_sample_budget},
        "correct_and_shuffle_same_network_initialization": bool(all(paired_initialization_checks)),
        "correct_and_shuffle_same_public_rows_except_source": bool(all(paired_data_checks)),
        "correct_and_shuffle_same_normalization_and_schedule": bool(
            all(paired_normalization_checks) and all(paired_schedule_checks)),
        "source_shuffle_within_anchor_action": bool(all(shuffle_checks)),
        "hidden_u_not_in_model": bool(all(hidden_free_checks)),
        "do_oracle_not_in_offline_training": not {
            "do_reward", "reward_branches"}.intersection(fit_quick_model.__code__.co_varnames),
        "source_not_in_g_or_h": bool(all(source_free_checks)),
        "splits_fully_disjoint": bool(split_disjoint),
        "calibration_action_independent_of_u": bool(all(calibration_independence_checks)),
        "B16_is_prefix_of_B64": bool(all(nested_checks)),
        "rank0_rank1_closed_form_correct": bool(calibration_formula_check),
        "bic_tie_returns_rank0": bic_select_rank(1.0, 1.0, 16) == 0,
        "redundant_population_contrast_zero": bool(redundant_zero),
        "independent_latents_population_contrast_zero": bool(independent_zero),
        "base_action_population_contrast_zero": bool(base_zero),
        "input_hashes_unchanged": bool(unchanged),
        "all_arrays_and_metrics_finite": bool(all_finite(
            [subspace_rows, method_rows, uncalibrated_rank1_rows])),
        "formal_artifacts_not_read_or_written": not any(
            str(path).startswith(str(output)) or str(output).startswith(str(path))
            for path in formal_roots),
        "only_best_checkpoint_saved": (
            saved_models == expected_models
            and len(list((output / "models").rglob("*.pt"))) == expected_models),
        "lightweight_file_scaling": estimate["estimated_file_count"] < 500,
    }
    failed = [name for name, passed in hard_checks.items() if not passed]
    _write_csv(output / "subspace_metrics.csv", subspace_rows)
    _write_csv(output / "uncalibrated_rank1_metrics.csv", uncalibrated_rank1_rows)
    _write_csv(output / "method_metrics.csv", method_rows)
    _write_csv(output / "calibration_metrics.csv", calibration_rows)
    _write_csv(output / "seed_metrics.csv", seed_rows)
    _write_csv(output / "summary_metrics.csv", summary_rows)
    _write_csv(output / "comparison_metrics.csv", comparisons)
    _write_json(output / "input_integrity.json", {"sha256_before": hashes_before,
                                                    "sha256_after": hashes_after,
                                                    "unchanged": unchanged})
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
                                               "all_passed": not failed, "failed": failed})
    manifest = {
        "stage": "Phase 8E-Q", "QUICK_GO_NO_GO_ONLY": True,
        "CONTROLLED_THREE_ACTION_SETTING": True,
        "num_anchors": num_anchors, "source_settings": {
            key: value.tolist() for key, value in QUICK_SOURCE_SETTINGS.items()},
        "lambda_values": list(doses), "kappa": 0.0, "condition": "confounded",
        "negative_control": {"setting": "M5_diverse", "lambda_reward": 0.05,
                             "condition": "independent_latents"},
        "reward_noise_std": reward_noise_std, "offline_sample_budget": offline_sample_budget,
        "model_seeds": list(seeds), "gradient_updates": gradient_updates,
        "calibration_budgets": list(budgets),
        "calibration_replicates": calibration_replicates, "data_seed": data_seed,
        "hidden_width": 128, "hidden_layers": 2, "activation": "ReLU",
        "optimizer": "Adam", "learning_rate": 1e-3,
        "checkpoint_selection": "best observational-validation MSE checked every 10 updates",
        "artifact_schema": "phase8e_quick_compact_v1",
        "frozen_lambda_information": inputs["frozen_record"],
        "formal_phase8e_roots_excluded": [str(path) for path in formal_roots],
        "all_hard_checks_passed": not failed,
    }
    _write_json(output / "manifest.json", manifest)
    summary = {
        "stage": "Phase 8E-Q", "scenario_count": len(scenarios),
        "model_count": saved_models, "conclusions": conclusions,
        "go_to_larger_phase8e": bool(all((
            conclusions["correct_source_beats_shuffle"],
            conclusions["M5_diverse_beats_M2_diverse"],
            conclusions["M5_redundant_has_no_equal_benefit"],
            conclusions["B0_to_B16_to_B64_improves_all_three_metrics"],
        ))),
        "continuous_action_extension_recommended":
            conclusions["continuous_action_extension_recommended"],
        "all_hard_checks_passed": not failed, "failed": failed,
    }
    _write_json(output / "summary.json", summary)
    _make_figures(output, summary_rows)
    _report(output, summary_rows, comparisons, conclusions)
    if failed:
        raise Phase8EMultisourceContrastError(f"hard checks failed: {failed}")
    return summary
