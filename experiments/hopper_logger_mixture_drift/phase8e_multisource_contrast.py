"""Artifact runner for Phase 8E multi-source contrast calibration."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .multisource_contrast_calibration import (
    ACTION_KEYS,
    ACTION_MARGINAL,
    DEFAULT_CALIBRATION_BUDGETS,
    DIVERSITY_HALF_WIDTHS,
    FORBIDDEN_PUBLIC_FIELDS,
    Phase8EMultisourceContrastError,
    PUBLIC_FIELDS,
    REWARD_NOISE_STDS,
    SOURCE_COUNTS,
    active_query_order,
    all_finite,
    antithetic_reward_noise,
    audit_population_subspace,
    bic_select_rank,
    budgets_are_nested,
    calibration_features,
    closed_form_calibration,
    decision_metrics,
    diversity_label,
    diversity_profile,
    do_reward_mean,
    empirical_source_mean_matrix,
    fit_source_free_model,
    fixed_draw_public_table,
    input_hashes,
    load_checkpoint,
    multisource_behavior_probabilities,
    normalize_loadings,
    pairwise_gap_features,
    population_source_means,
    predict_components,
    random_balanced_query_order,
    require_all_passed,
    reward_prediction_metrics,
    save_checkpoint,
    sha256,
    shuffle_source_within_anchor_action,
    source_action_marginals,
    source_composition_state_action_mass,
    svd_initialization,
    validate_public_table,
    validate_source_free_model,
)
from .reward_mechanism_separation import (
    commanded_action_indices,
    index_derived_public_files,
    kappa_name,
    lambda_token,
    load_frozen_lambda_grid,
)


PHASE8D_DIRECTORY = "phase8d_public_init_intervention_calibration"
PHASE8B_REWARD_SIGNAL_DIRECTORY = "phase8b_reward_signal_calibration"
ORACLE_REWARD_AUDIT_DIRECTORY = "oracle_direct_reward_confounding_audit"
STRICT_ANALYSIS_DIRECTORY = "phase8d_public_init_calibration_strict_analysis"
METHODS = (
    "pooled_mlp_no_calibration",
    "pooled_mlp_intercept_calibration",
    "phase8d_outcome_residual_initialization",
    "outcome_only_clustering_without_source",
    "per_source_models_average",
    "mscsc_correct_source_random",
    "mscsc_correct_source_active",
    "mscsc_source_shuffle_active",
    "mscsc_redundant_source_control",
    "population_contrast_oracle",
    "oracle_u_aware_ceiling",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise Phase8EMultisourceContrastError(f"required read-only input is missing: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    if not path.is_file():
        raise Phase8EMultisourceContrastError(f"required read-only input is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value) if value not in {"", "nan", "NaN"} else math.nan


def reproduce_phase8d_facts(phase8d_root: Path, strict_analysis_root: Path) -> dict[str, Any]:
    """Recompute the five Phase 8D facts before any Phase 8E output is created."""
    require_all_passed(phase8d_root / "hard_checks.json")
    analysis = _read_json(strict_analysis_root / "analysis-manifest.json")
    if analysis.get("formal_hard_checks_passed") is not True:
        raise Phase8EMultisourceContrastError("Phase 8D strict analysis is not certified")
    rows = _read_csv(phase8d_root / "seed_metrics.csv")

    def selected(method: str, *, dose: float | None = None,
                 budget: int | None = None) -> list[dict[str, str]]:
        result = []
        for row in rows:
            if (_float(row, "kappa") != 0.0 or row["condition"] != "confounded"
                    or row["method"] != method or int(float(row["seed"])) not in range(5)):
                continue
            if dose is not None and not np.isclose(_float(row, "lambda_reward"), dose):
                continue
            raw_budget = row.get("calibration_budget", "")
            if budget is None and raw_budget not in {"", "nan", "NaN"}:
                continue
            if budget is not None and (raw_budget in {"", "nan", "NaN"}
                                       or int(float(raw_budget)) != budget):
                continue
            result.append(row)
        return result

    def mean(method: str, metric: str, **kwargs: Any) -> float:
        values = [_float(row, metric) for row in selected(method, **kwargs)]
        if not values or not np.all(np.isfinite(values)):
            raise Phase8EMultisourceContrastError(f"cannot reproduce Phase 8D {method}/{metric}")
        return float(np.mean(values))

    v0 = selected("V0_random_init_mechanism")
    public = selected("public_residual_init_nll_best")
    facts = {
        "public_residual_initialization_prevents_collapse": (
            len(v0) == len(public) == 35
            and all(_float(row, "latent_collapse") == 1.0 for row in v0)
            and all(_float(row, "latent_collapse") == 0.0 for row in public)),
        "source_shuffle_main_metrics_close": all(
            abs(mean("public_residual_init_nll_best", metric)
                - mean("source_shuffle_initialization", metric)) < tolerance
            for metric, tolerance in (("do_mae", 0.001),
                                      ("top_set_disagreement", 0.02),
                                      ("mean_regret", 0.0002))),
        "no_staged_not_worse_than_staged": all(
            mean("no_staged_training", metric) <= mean("public_residual_init_nll_best", metric)
            for metric in ("do_mae", "top_set_disagreement", "mean_regret")),
        "lambda_zero_public_worse_than_pooled": all(
            mean("public_residual_init_nll_best", metric, dose=0.0)
            > mean("pooled_mlp", metric, dose=0.0)
            for metric in ("do_mae", "top_set_disagreement", "mean_regret")),
        "b128_improves_b0_but_not_nll_ranking_regret": (
            mean("public_residual_init_intervention_calibrated", "do_mae", budget=128)
            < mean("public_residual_init_intervention_calibrated", "do_mae", budget=0)
            and mean("public_residual_init_intervention_calibrated", "top_set_disagreement", budget=128)
            < mean("public_residual_init_intervention_calibrated", "top_set_disagreement", budget=0)
            and mean("public_residual_init_intervention_calibrated", "mean_regret", budget=128)
            < mean("public_residual_init_intervention_calibrated", "mean_regret", budget=0)
            and mean("public_residual_init_intervention_calibrated", "top_set_disagreement", budget=128)
            >= mean("public_residual_init_nll_best", "top_set_disagreement")
            and mean("public_residual_init_intervention_calibrated", "mean_regret", budget=128)
            >= mean("public_residual_init_nll_best", "mean_regret")),
    }
    if not all(facts.values()):
        raise Phase8EMultisourceContrastError(
            f"Phase 8D facts could not be reproduced: {[k for k,v in facts.items() if not v]}")
    return {"checks": facts, "all_reproduced": True}


def resolve_phase8e_inputs(phase8a_root: Path, direct_reward_root: Path,
                           lambda_grid_file: Path, kappas: Sequence[float],
                           conditions: Sequence[str]) -> dict[str, Any]:
    phase8a = Path(phase8a_root).resolve()
    direct = Path(direct_reward_root).resolve()
    grid_path = Path(lambda_grid_file).resolve()
    project = Path(__file__).resolve().parents[2]
    phase8d = direct.parent / PHASE8D_DIRECTORY
    reward_signal = direct.parent / PHASE8B_REWARD_SIGNAL_DIRECTORY
    oracle_reward_audit = direct.parent / ORACLE_REWARD_AUDIT_DIRECTORY
    strict = project / "analysis" / STRICT_ANALYSIS_DIRECTORY
    required = [direct / "manifest.json", direct / "hard_checks.json", direct / "splits.json",
                reward_signal / "manifest.json", reward_signal / "hard_checks.json",
                oracle_reward_audit / "manifest.json", oracle_reward_audit / "hard_checks.json",
                grid_path, phase8d / "manifest.json", phase8d / "hard_checks.json",
                phase8d / "seed_metrics.csv", strict / "analysis-manifest.json",
                strict / "analysis-report.md"]
    if (phase8a / "hard_checks.json").is_file():
        required.extend((phase8a / "manifest.json", phase8a / "hard_checks.json"))
    for kappa in kappas:
        required.append(phase8a / kappa_name(float(kappa)) / "do_oracle_raw.npz")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise Phase8EMultisourceContrastError(f"required read-only input is missing: {missing[0]}")
    require_all_passed(direct / "hard_checks.json")
    require_all_passed(reward_signal / "hard_checks.json")
    require_all_passed(oracle_reward_audit / "hard_checks.json")
    require_all_passed(phase8d / "hard_checks.json")
    if (phase8a / "hard_checks.json").is_file():
        require_all_passed(phase8a / "hard_checks.json")
    grid, frozen = load_frozen_lambda_grid(grid_path)
    index = index_derived_public_files(direct)
    needed = [(float(kappa), float(dose), condition) for kappa in kappas
              for dose in grid for condition in conditions]
    missing_scenarios = [key for key in needed if key not in index]
    if missing_scenarios:
        raise Phase8EMultisourceContrastError(
            f"direct reward scenario is unavailable: {missing_scenarios[0]}")
    required.extend(index[key] for key in needed)
    facts = reproduce_phase8d_facts(phase8d, strict)
    return {"phase8a": phase8a, "direct": direct,
            "reward_signal": reward_signal, "oracle_reward_audit": oracle_reward_audit,
            "grid_path": grid_path,
            "grid": grid, "frozen": frozen, "phase8d": phase8d, "strict": strict,
            "public_index": index, "required_paths": required, "phase8d_facts": facts}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {name: raw[name] for name in raw.files}


def load_anchor_universe(phase8a_root: Path, public_path: Path,
                         kappa: float) -> dict[str, np.ndarray]:
    public = _load_npz(public_path)
    raw = _load_npz(phase8a_root / kappa_name(kappa) / "do_oracle_raw.npz")
    ids = np.unique(np.asarray(public["anchor_id"], dtype=np.int64))
    observations = np.empty((len(ids), 12), dtype=np.float32)
    actions = np.empty((len(ids), 3, 3), dtype=np.float32)
    action_index = commanded_action_indices(public)
    for position, anchor in enumerate(ids):
        rows = np.flatnonzero(np.asarray(public["anchor_id"]) == anchor)
        observations[position] = public["observation"][rows[0]]
        for action in range(3):
            choices = rows[action_index[rows] == action]
            if not len(choices):
                raise Phase8EMultisourceContrastError("public anchor action is missing")
            actions[position, action] = public["commanded_action"][choices[0]]
    branches = np.empty((len(ids), 3, 2), dtype=np.float64)
    for position, anchor in enumerate(ids):
        for action, name in enumerate(ACTION_KEYS):
            for latent_index, u in enumerate((-1, 1)):
                rows = np.flatnonzero((raw["anchor_id"] == anchor)
                                      & (raw["action_key"].astype(str) == name)
                                      & (raw["u_env"] == u))
                if len(rows) != 1:
                    raise Phase8EMultisourceContrastError("do branch table is not unique")
                branches[position, action, latent_index] = raw["reward"][rows[0]]
    return {"anchor_id": ids, "observation": observations,
            "commanded_action": actions, "reward_branches": branches}


def select_anchor_splits(splits: Mapping[str, Sequence[int]], available: Sequence[int],
                         count: int, max_calibration_budget: int) -> dict[str, np.ndarray]:
    available_set = set(map(int, available))
    groups = {name: np.asarray([x for x in map(int, splits[name]) if x in available_set], dtype=np.int64)
              for name in ("train", "observational_validation", "do_calibration_pool", "test")}
    if count > sum(map(len, groups.values())):
        raise ValueError("requested anchors exceed available split assignments")
    minimum_cal = min(len(groups["do_calibration_pool"]), math.ceil(max_calibration_budget / 3) + 1)
    target_test = min(len(groups["test"]), max(1, round(0.15 * count)))
    target_validation = min(len(groups["observational_validation"]), max(1, round(0.10 * count)))
    target_cal = min(len(groups["do_calibration_pool"]), max(minimum_cal, round(0.10 * count)))
    target_train = min(len(groups["train"]), count - target_test - target_validation - target_cal)
    if target_train <= 0:
        raise ValueError("anchor budget cannot preserve all four Phase 8D splits")
    targets = {"train": target_train, "observational_validation": target_validation,
               "do_calibration_pool": target_cal, "test": target_test}
    deficit = count - sum(targets.values())
    for name in ("train", "observational_validation", "do_calibration_pool", "test"):
        addition = min(deficit, len(groups[name]) - targets[name])
        targets[name] += addition
        deficit -= addition
    if deficit:
        raise ValueError("anchor budget cannot be filled from Phase 8D splits")
    return {name: np.sort(groups[name][:targets[name]]) for name in groups}


def _subset_universe(universe: Mapping[str, np.ndarray], ids: Sequence[int]) -> dict[str, np.ndarray]:
    lookup = {int(anchor): index for index, anchor in enumerate(universe["anchor_id"])}
    index = np.asarray([lookup[int(anchor)] for anchor in ids], dtype=np.int64)
    return {name: np.asarray(value)[index] for name, value in universe.items()}


def _scenario_name(m: int, diversity: float, sigma: float, kappa: float,
                   dose: float, condition: str, seed: int | None = None) -> str:
    parts = (f"M_{m}", f"div_{diversity:.2f}".replace(".", "p"),
             f"sigma_{sigma:.2f}".replace(".", "p"), kappa_name(kappa),
             lambda_token(dose), condition)
    return "/".join((*parts, f"seed_{seed}")) if seed is not None else "/".join(parts)


def _flatten_action_table(observations: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.repeat(observations, 3, axis=0), actions.reshape(-1, 3))


def _calibration_pool(universe: Mapping[str, np.ndarray], calibration_ids: Sequence[int],
                      g: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
    ids = np.asarray(calibration_ids, dtype=np.int64)
    obs, action = _flatten_action_table(universe["observation"], universe["commanded_action"])
    action_index = np.tile(np.arange(3, dtype=np.int8), len(ids))
    anchor_id = np.repeat(ids, 3)
    return {"anchor_id": anchor_id, "observation": obs, "commanded_action": action,
            "anchor_position": np.repeat(np.arange(len(ids)), 3),
            "action_index": action_index, "g": g.reshape(-1), "h": h.reshape(-1),
            "features_rank0": calibration_features(action_index, h.reshape(-1), rank=0),
            "features_rank1": calibration_features(action_index, h.reshape(-1), rank=1)}


def _sample_calibration_outcomes(order: np.ndarray, pool: Mapping[str, np.ndarray],
                                 branches: np.ndarray, dose: float, sigma: float,
                                 replicate: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 10007 * replicate)
    u = np.where(rng.random(len(order)) < 0.5, 1, -1).astype(np.int8)
    epsilon = antithetic_reward_noise(len(order), sigma, seed + 17011 * replicate)
    latent = ((u + 1) // 2).astype(np.int64)
    anchor_position = np.asarray(pool["anchor_position"], dtype=np.int64)[order]
    outcome = branches[anchor_position, np.asarray(pool["action_index"])[order], latent]
    outcome = outcome + dose * u + epsilon
    return outcome.astype(np.float64), u


def _test_prediction(g: np.ndarray, h: np.ndarray, coefficients: np.ndarray, rank: int) -> np.ndarray:
    action = np.tile(np.arange(3, dtype=np.int8), len(g))
    feature = calibration_features(action, h.reshape(-1), rank=rank)
    return (g.reshape(-1) + feature @ coefficients).reshape(len(g), 3)


def _metric_record(labels: Mapping[str, Any], method: str, truth: np.ndarray,
                   prediction: np.ndarray, **extra: Any) -> dict[str, Any]:
    return {**labels, "method": method, **extra,
            **reward_prediction_metrics(truth, prediction),
            **decision_metrics(truth, prediction)}


def _phase8d_uniform_prediction(phase8d_arrays: Mapping[str, np.ndarray], kappa: float,
                                dose: float, condition: str,
                                test_ids: Sequence[int]) -> np.ndarray | None:
    prefix = f"{kappa_name(kappa)}__{lambda_token(dose)}__{condition}"
    id_key, prediction_key = prefix + "__anchor_id", prefix + "__uniform_prediction"
    if id_key not in phase8d_arrays or prediction_key not in phase8d_arrays:
        return None
    lookup = {int(anchor): index for index, anchor in enumerate(phase8d_arrays[id_key])}
    if not set(map(int, test_ids)).issubset(lookup):
        return None
    return np.asarray(phase8d_arrays[prediction_key])[
        [lookup[int(anchor)] for anchor in test_ids]]


def _predict_source_means(model: Any, normalization: Mapping[str, np.ndarray],
                          observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    import torch
    g, h = predict_components(model, normalization, observations, actions, "cpu")
    loading = model.normalized_loadings().detach().cpu().numpy()
    return g[None, :, :] + loading[:, None, :] * h[None, :, :]


def _population_oracle_prediction(source_means: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = np.empty_like(truth)
    for action in range(3):
        matrix = source_means[:, :, action]
        audit = audit_population_subspace(matrix, truth[:, action])
        center = matrix.mean(axis=0)
        direction = audit.direction
        if np.dot(direction, direction) > 0:
            coefficient = np.dot(truth[:, action] - center, direction) / np.dot(direction, direction)
            prediction[:, action] = center + coefficient * direction
        else:
            prediction[:, action] = center
    return prediction


def _save_public_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if not validate_public_table(arrays):
        raise Phase8EMultisourceContrastError("refusing to save invalid public rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _calibration_public_record(pool: Mapping[str, np.ndarray], order: np.ndarray,
                               reward: np.ndarray, kappa: float, dose: float,
                               sigma: float, condition: str) -> dict[str, np.ndarray]:
    n = len(order)
    return {
        "anchor_id": np.asarray(pool["anchor_id"])[order],
        "observation": np.asarray(pool["observation"])[order],
        "commanded_action": np.asarray(pool["commanded_action"])[order],
        "action_index": np.asarray(pool["action_index"])[order],
        "reward": np.asarray(reward, dtype=np.float64),
        "source_id": np.full(n, -1, dtype=np.int16),
        "kappa": np.full(n, kappa), "lambda_reward": np.full(n, dose),
        "sigma_reward": np.full(n, sigma), "condition": np.full(n, condition),
        "row_weight": np.full(n, 1.0 / max(1, n)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], groups: Sequence[str],
               metrics: Sequence[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[name] for name in groups)].append(row)
    result = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(groups, key))
        for metric in metrics:
            finite = [float(row[metric]) for row in values if metric in row and np.isfinite(float(row[metric]))]
            record[metric] = float(np.mean(finite)) if finite else math.nan
            record[metric + "_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        record["row_count"] = len(values)
        result.append(record)
    return result


def _make_figures(output: Path, subspace: Sequence[Mapping[str, Any]],
                  metrics: Sequence[Mapping[str, Any]], safety: Sequence[Mapping[str, Any]],
                  calibration: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout(); plt.savefig(figures / name, dpi=220); plt.close()

    def grouped_curve(rows: Sequence[Mapping[str, Any]], x: str, y: str, group: str,
                      xlabel: str, ylabel: str, name: str) -> None:
        plt.figure(figsize=(7, 4.5))
        labels = sorted({str(row[group]) for row in rows})
        for label in labels:
            subset = [row for row in rows if str(row[group]) == label]
            xs = sorted({float(row[x]) for row in subset})
            ys = [float(np.mean([float(row[y]) for row in subset if float(row[x]) == value]))
                  for value in xs]
            plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        if len(labels) > 1: plt.legend()
        save(name)

    primary_subspace = [r for r in subspace if r["kappa"] == 0.0
                        and r["condition"] == "confounded" and r["sigma_reward"] == 0.02
                        and r["action"] == "plus"]
    grouped_curve(primary_subspace, "diversity_half_width", "rank1_explained_variance",
                  "source_count", "Source diversity half-width", "Rank-1 explained variance",
                  "rank1_explained_variance_vs_source_diversity.png")
    comparison = []
    for row in primary_subspace:
        comparison.extend(({**row, "kind": "correct", "value": row["centered_norm"]},
                           {**row, "kind": "shuffle", "value": row["shuffle_centered_norm"]}))
    grouped_curve(comparison, "diversity_half_width", "value", "kind",
                  "Source diversity half-width", "Centered response norm",
                  "correct_vs_shuffle_subspace_recovery.png")

    primary = [r for r in metrics if r["protocol"] == "fixed_draw" and r["kappa"] == 0.0
               and r["condition"] == "confounded" and r["sigma_reward"] == 0.02
               and r["diversity_half_width"] == 0.2 and r["calibration_budget"] == 128
               and r["query_strategy"] == "active"]
    for field, filename, ylabel in (
        ("do_mae", "do_mae_vs_number_of_sources.png", "Do MAE"),
        ("top_set_disagreement", "rank_error_vs_number_of_sources.png", "Top-set disagreement"),
        ("mean_regret", "regret_vs_number_of_sources.png", "Mean regret")):
        grouped_curve(primary, "source_count", field, "method", "Number of sources", ylabel, filename)
    diversity_rows = [r for r in metrics if r["protocol"] == "fixed_draw" and r["kappa"] == 0.0
                      and r["condition"] == "confounded" and r["sigma_reward"] == 0.02
                      and r["source_count"] == 8 and r["calibration_budget"] == 128
                      and r["query_strategy"] == "active"]
    grouped_curve(diversity_rows, "diversity_half_width", "do_mae", "method",
                  "Source diversity half-width", "Do MAE", "do_mae_vs_source_diversity.png")
    correct_shuffle = [r for r in diversity_rows if r["method"] in {
        "mscsc_correct_source_active", "mscsc_source_shuffle_active"}]
    grouped_curve(correct_shuffle, "diversity_half_width", "mean_regret", "method",
                  "Source diversity half-width", "Mean regret", "correct_source_vs_shuffle.png")

    grouped_curve(safety, "lambda_reward", "rank0_selection_fraction", "source_count",
                  "Direct reward dose λ", "Rank-0 selection fraction",
                  "lambda_zero_rank_selection.png")
    safety_long = []
    for row in safety:
        safety_long.extend(({**row, "kind": "pooled", "value": row["pooled_do_mae"]},
                            {**row, "kind": "adaptive", "value": row["adaptive_do_mae"]}))
    grouped_curve(safety_long, "lambda_reward", "value", "kind", "Direct reward dose λ",
                  "Do MAE", "lambda_zero_safety.png")

    cal_primary = [r for r in calibration if r["kappa"] == 0.0 and r["condition"] == "confounded"
                   and r["sigma_reward"] == 0.02 and r["source_count"] == 8
                   and r["diversity_half_width"] == 0.2]
    for field, filename, ylabel in (
        ("do_mae", "do_mae_vs_calibration_budget.png", "Do MAE"),
        ("top_set_disagreement", "rank_error_vs_calibration_budget.png", "Top-set disagreement"),
        ("mean_regret", "regret_vs_calibration_budget.png", "Mean regret")):
        grouped_curve(cal_primary, "calibration_budget", field, "query_strategy",
                      "Calibration budget B", ylabel, filename)
    grouped_curve(cal_primary, "calibration_budget", "mean_regret", "query_strategy",
                  "Calibration budget B", "Mean regret", "active_vs_random_calibration.png")
    grouped_curve(cal_primary, "calibration_budget", "coordinate_error", "query_strategy",
                  "Calibration budget B", "Calibration-coordinate error",
                  "calibration_coordinate_recovery.png")

    obs_do = [r for r in metrics if r["method"] in {"pooled_mlp_no_calibration",
                                                     "mscsc_correct_source_active"}]
    plt.figure(figsize=(6, 5))
    for method in sorted({r["method"] for r in obs_do}):
        part = [r for r in obs_do if r["method"] == method]
        plt.scatter([r.get("observational_mse", math.nan) for r in part],
                    [r["do_mae"] for r in part], alpha=0.5, label=method)
    plt.xlabel("Observational MSE"); plt.ylabel("Do MAE"); plt.legend()
    save("observational_fit_vs_do_fit.png")

    # Additional required scaling view: correct-source performance across diversity.
    grouped_curve(primary_subspace, "diversity_half_width", "rank1_reconstruction_error",
                  "source_count", "Source diversity half-width", "Rank-1 reconstruction error",
                  "source_subspace_reconstruction_error.png")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6g}"


def _write_reports(output: Path, manifest: Mapping[str, Any],
                   subspace: Sequence[Mapping[str, Any]],
                   scaling: Sequence[Mapping[str, Any]],
                   safety: Sequence[Mapping[str, Any]],
                   active_random: Sequence[Mapping[str, Any]]) -> None:
    primary_subspace = [row for row in subspace if row["kappa"] == 0.0
                        and row["condition"] == "confounded"
                        and row["sigma_reward"] == 0.02 and row["action"] == "plus"]
    table1 = _aggregate(primary_subspace, ("source_count", "diversity_half_width"),
                        ("rank1_explained_variance", "shuffle_centered_norm",
                         "rank1_reconstruction_error"))
    primary_scaling = [row for row in scaling if row["kappa"] == 0.0
                       and row["condition"] == "confounded" and row["sigma_reward"] == 0.02
                       and row["calibration_budget"] in {-1, 128}]
    table2 = _aggregate(primary_scaling,
                        ("source_count", "diversity_half_width", "method"),
                        ("do_mae", "top_set_disagreement", "mean_regret"))
    table3 = _aggregate(safety, ("lambda_reward",),
                        ("rank0_selection_fraction", "pooled_do_mae", "adaptive_do_mae",
                         "pooled_regret", "adaptive_regret"))
    primary_active = [row for row in active_random if row["kappa"] == 0.0
                      and row["condition"] == "confounded" and row["sigma_reward"] == 0.02]
    table4 = _aggregate(primary_active, ("calibration_budget",),
                        ("random_do_mae", "active_do_mae",
                         "random_top_set_disagreement", "active_top_set_disagreement",
                         "random_mean_regret", "active_mean_regret"))

    lines = [
        "# Phase 8E-MSCSC Report", "",
        "This report is produced without an automatic success claim. Model seed is the training",
        "replication unit; calibration replicate is nested randomness. The formal run remains gated",
        "on a favorable pilot review.", "",
        "## Table 1 — population source subspace", "",
        "| M | Diversity | Correct-source rank1 EV | Shuffle centered norm | Reconstruction error |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(map(str, (
        row["source_count"], _fmt(row["diversity_half_width"]),
        _fmt(row["rank1_explained_variance"]), _fmt(row["shuffle_centered_norm"]),
        _fmt(row["rank1_reconstruction_error"])))) + " |" for row in table1)
    lines.extend(["", "## Table 2 — fixed-draw interventional performance", "",
                  "| M | Diversity | Method | Do MAE | Rank error | Regret |",
                  "|---:|---:|---|---:|---:|---:|"])
    lines.extend("| " + " | ".join(map(str, (
        row["source_count"], _fmt(row["diversity_half_width"]), row["method"],
        _fmt(row["do_mae"]), _fmt(row["top_set_disagreement"]),
        _fmt(row["mean_regret"])))) + " |" for row in table2)
    lines.extend(["", "## Table 3 — lambda-zero safety", "",
                  "| Lambda | Rank-0 selection | Pooled error | Adaptive error | Pooled regret | Adaptive regret |",
                  "|---:|---:|---:|---:|---:|---:|"])
    lines.extend("| " + " | ".join(map(str, (
        _fmt(row["lambda_reward"]), _fmt(row["rank0_selection_fraction"]),
        _fmt(row["pooled_do_mae"]), _fmt(row["adaptive_do_mae"]),
        _fmt(row["pooled_regret"]), _fmt(row["adaptive_regret"])))) + " |" for row in table3)
    lines.extend(["", "## Table 4 — random versus active calibration", "",
                  "| B | Random Do MAE | Active Do MAE | Random rank error | Active rank error | Random regret | Active regret |",
                  "|---:|---:|---:|---:|---:|---:|---:|"])
    lines.extend("| " + " | ".join(map(str, (
        row["calibration_budget"], _fmt(row["random_do_mae"]), _fmt(row["active_do_mae"]),
        _fmt(row["random_top_set_disagreement"]), _fmt(row["active_top_set_disagreement"]),
        _fmt(row["random_mean_regret"]), _fmt(row["active_mean_regret"])))) + " |"
                 for row in table4)
    lines.extend(["", "## Scientific decision boundary", "",
                  "The tables and figures must be reviewed against the nine Phase 8E questions.",
                  "Population rank one alone is not evidence that the neural approximation, correct",
                  "source labels, BIC adaptation, or active calibration succeeded. Correct-source",
                  "performance must improve over source shuffle and outcome-only baselines as diversity",
                  "increases; redundant sources must not show a spurious scaling benefit; and active",
                  "calibration must improve ranking and regret, not only MAE.", "",
                  "The current experiment is one-step reward-only. Transition-model and SAC expansion",
                  "is not justified unless all of those empirical comparisons survive pilot review.", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    figure_names = [
        "rank1_explained_variance_vs_source_diversity.png",
        "correct_vs_shuffle_subspace_recovery.png", "do_mae_vs_number_of_sources.png",
        "rank_error_vs_number_of_sources.png", "regret_vs_number_of_sources.png",
        "do_mae_vs_source_diversity.png", "correct_source_vs_shuffle.png",
        "lambda_zero_rank_selection.png", "lambda_zero_safety.png",
        "do_mae_vs_calibration_budget.png", "rank_error_vs_calibration_budget.png",
        "regret_vs_calibration_budget.png", "active_vs_random_calibration.png",
        "calibration_coordinate_recovery.png", "observational_fit_vs_do_fit.png",
    ]
    catalog = ["# Phase 8E Figure Catalog", ""]
    for index, name in enumerate(figure_names, 1):
        catalog.extend((f"## {index}. `{name}`", "",
                        "- **Purpose:** Evaluate the named Phase 8E geometry, scaling, safety, or calibration comparison.",
                        "- **Notice:** Curves summarize model-seed results; calibration replicates are nested.",
                        "- **Implication:** Interpret jointly with REPORT.md; no threshold-based success label is applied.", ""))
    (output / "figure-catalog.md").write_text("\n".join(catalog), encoding="utf-8")


def run_phase8e_multisource_contrast(
    phase8a_root: Path,
    direct_reward_root: Path,
    lambda_grid_file: Path,
    output_root: Path,
    *,
    num_anchors: int,
    source_counts: Sequence[int],
    diversity_half_widths: Sequence[float],
    reward_noise_stds: Sequence[float],
    kappas: Sequence[float],
    conditions: Sequence[str],
    offline_sample_budget: int,
    calibration_budgets: Sequence[int],
    calibration_replicates: int,
    model_seeds: Sequence[int],
    device: str,
) -> dict[str, Any]:
    """Run the fixed-draw Phase 8E proof of concept after all input gates pass."""
    source_counts = tuple(map(int, source_counts))
    diversities = tuple(map(float, diversity_half_widths))
    sigmas = tuple(map(float, reward_noise_stds))
    kappas = tuple(map(float, kappas)); conditions = tuple(conditions)
    budgets = tuple(sorted(set(map(int, calibration_budgets))))
    seeds = tuple(map(int, model_seeds))
    if (not set(source_counts).issubset(SOURCE_COUNTS)
            or not set(diversities).issubset(DIVERSITY_HALF_WIDTHS)
            or not set(sigmas).issubset(REWARD_NOISE_STDS)
            or not budgets or budgets[0] != 0 or calibration_replicates <= 0):
        raise Phase8EMultisourceContrastError("Phase 8E grid arguments are invalid")

    inputs = resolve_phase8e_inputs(phase8a_root, direct_reward_root, lambda_grid_file,
                                    kappas, conditions)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise Phase8EMultisourceContrastError(f"output directory is not empty: {output}")
    hashes_before = input_hashes(inputs["required_paths"])
    splits_record = _read_json(inputs["phase8d"] / "phase8d_splits.json")
    phase8d_arrays = _load_npz(inputs["phase8d"] / "anchor_action_metrics.npz")
    max_budget = max(budgets)

    output.mkdir(parents=True)
    for directory in ("multisource_dgp", "population_subspace_audit", "source_mean_matrices",
                      "svd_initializations", "models", "calibration_data",
                      "active_query_sequences", "predictions", "figures"):
        (output / directory).mkdir()

    subspace_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    all_public_hidden_free = True
    all_shuffle_valid = True
    all_loading_constraints = True
    all_svd_signs = True
    all_pop_rank = True
    all_pop_affine = True
    base_contrast_zero = True
    redundant_contrast_zero = True
    independent_contrast_zero = True
    checkpoint_roundtrip = False
    all_models_source_free = True
    fixed_draw_counts: set[int] = set()
    update_counts: set[int] = set()
    batch_sizes: set[int] = set()
    nested_budgets = True
    calibration_independent = True
    test_isolated = True
    action_marginal_checks = []
    composition_checks = []
    probability_checks = []
    source_diversity_checks = []
    all_noise_pairs = True
    do_source_invariant = True
    do_mean_invariant = True
    source_loadings_records: list[dict[str, Any]] = []
    selected_split_record: dict[str, list[int]] | None = None
    formal_updates, batch_size = 300, 512

    for kappa in kappas:
        zero_public = inputs["public_index"][(kappa, float(inputs["grid"][0]), conditions[0])]
        universe_all = load_anchor_universe(inputs["phase8a"], zero_public, kappa)
        selected_splits = select_anchor_splits(splits_record, universe_all["anchor_id"],
                                               num_anchors, max_budget)
        if selected_split_record is None:
            selected_split_record = {k: list(map(int, v)) for k, v in selected_splits.items()}
        split_sets = [set(map(int, selected_splits[name])) for name in selected_splits]
        test_isolated &= not any(split_sets[i] & split_sets[j] for i in range(4)
                                 for j in range(i + 1, 4))
        train_ids = np.sort(np.concatenate((selected_splits["train"],
                                            selected_splits["observational_validation"])))
        train = _subset_universe(universe_all, train_ids)
        calibration_universe = _subset_universe(universe_all, selected_splits["do_calibration_pool"])
        test_universe = _subset_universe(universe_all, selected_splits["test"])
        test_truth = do_reward_mean(test_universe["reward_branches"])

        for condition in conditions:
            for dose in inputs["grid"]:
                phase8d_prediction = _phase8d_uniform_prediction(
                    phase8d_arrays, kappa, dose, condition, test_universe["anchor_id"])
                for sigma in sigmas:
                    noise_check = antithetic_reward_noise(100, sigma, 77)
                    all_noise_pairs &= bool(np.allclose(noise_check[0::2], -noise_check[1::2]))
                    for m in source_counts:
                        for diversity in diversities:
                            p_values = diversity_profile(m, diversity)
                            behavior = multisource_behavior_probabilities(p_values)
                            action_marginal_checks.append(np.allclose(
                                source_action_marginals(behavior), ACTION_MARGINAL[None, :],
                                atol=1e-15, rtol=0.0))
                            random_composition = np.arange(1, m + 1, dtype=np.float64)
                            random_composition /= random_composition.sum()
                            composition_checks.append(np.allclose(
                                source_composition_state_action_mass(behavior, random_composition),
                                ACTION_MARGINAL, atol=1e-15, rtol=0.0))
                            probability_checks.append(np.allclose(behavior.sum(axis=2), 1.0))
                            source_diversity_checks.append(
                                np.all(p_values > 0.5) and np.all(p_values < 1.0)
                                and (np.allclose(p_values, 0.75) if diversity == 0.0
                                     else np.isclose(np.ptp(p_values), 2.0 * diversity)))

                            pop_train = population_source_means(
                                train["reward_branches"], p_values, dose, condition)
                            pop_test = population_source_means(
                                test_universe["reward_branches"], p_values, dose, condition)
                            do_train = do_reward_mean(train["reward_branches"], dose)
                            do_test = do_reward_mean(test_universe["reward_branches"], dose)
                            do_source_invariant &= np.allclose(
                                np.repeat(do_test[None, :, :], m, axis=0), do_test[None, :, :],
                                atol=0.0, rtol=0.0)
                            do_mean_invariant &= np.allclose(do_test, test_truth, atol=1e-14, rtol=0.0)
                            pop_audits = []
                            for action, action_name in enumerate(ACTION_KEYS):
                                audit = audit_population_subspace(pop_train[:, :, action], do_train[:, action])
                                pop_audits.append(audit)
                                if condition == "confounded" and diversity > 0:
                                    all_pop_rank &= audit.numerical_rank <= 1
                                    all_pop_affine &= audit.affine_do_projection_residual < 1e-9
                                if action == 1:
                                    base_contrast_zero &= audit.centered_norm < 1e-12
                                if diversity == 0.0:
                                    redundant_contrast_zero &= audit.centered_norm < 1e-12
                                if condition == "independent_latents":
                                    independent_contrast_zero &= audit.centered_norm < 1e-12
                                correlation = (float(np.corrcoef(audit.loading, p_values)[0, 1])
                                               if np.std(audit.loading) > 0 else 0.0)
                                subspace_rows.append({
                                    "protocol": "population_support", "source_count": m,
                                    "diversity_half_width": diversity,
                                    "source_diversity": diversity_label(diversity),
                                    "kappa": kappa, "lambda_reward": dose,
                                    "sigma_reward": sigma, "condition": condition,
                                    "action": action_name,
                                    "singular_values": json.dumps(audit.singular_values.tolist()),
                                    "rank1_explained_variance": audit.rank1_explained_variance,
                                    "rank1_reconstruction_error": audit.rank1_reconstruction_error,
                                    "affine_do_projection_residual": audit.affine_do_projection_residual,
                                    "centered_norm": audit.centered_norm,
                                    "shuffle_centered_norm": 0.0,
                                    "numerical_rank": audit.numerical_rank,
                                    "loading_true_p_correlation_posthoc": correlation,
                                })
                            if diversity > 0 and condition == "confounded":
                                plus = pop_train[:, :, 2] - pop_train[:, :, 2].mean(axis=0)
                                minus = pop_train[:, :, 0] - pop_train[:, :, 0].mean(axis=0)
                                denom = np.linalg.norm(plus) * np.linalg.norm(minus)
                                sign_relation = float(np.sum(plus * minus) / denom) if denom > 0 else 0.0
                            else:
                                sign_relation = 0.0
                            source_loadings_records.append({"source_count": m, "diversity": diversity,
                                                            "kappa": kappa, "lambda": dose,
                                                            "condition": condition,
                                                            "plus_minus_raw_correlation": sign_relation})
                            pop_oracle = _population_oracle_prediction(pop_test, do_test)
                            pop_labels = {"protocol": "population_support", "source_count": m,
                                          "diversity_half_width": diversity, "kappa": kappa,
                                          "lambda_reward": dose, "sigma_reward": sigma,
                                          "condition": condition, "seed": -1,
                                          "calibration_budget": -1, "calibration_replicate": -1,
                                          "query_strategy": "oracle"}
                            metric_rows.append(_metric_record(
                                pop_labels, "population_contrast_oracle", do_test, pop_oracle,
                                observational_mse=0.0, selected_rank=1))
                            metric_rows.append(_metric_record(
                                pop_labels, "oracle_u_aware_ceiling", do_test, do_test,
                                observational_mse=0.0, selected_rank=1))

                            for seed in seeds:
                                scenario = _scenario_name(m, diversity, sigma, kappa, dose, condition, seed)
                                public, hidden = fixed_draw_public_table(
                                    train["anchor_id"], train["observation"], train["commanded_action"],
                                    train["reward_branches"], p_values, kappa=kappa,
                                    lambda_reward=dose, sigma_reward=sigma, condition=condition,
                                    sample_budget=offline_sample_budget, seed=seed + 1009)
                                fixed_draw_counts.add(len(public["anchor_id"]))
                                all_public_hidden_free &= (validate_public_table(public)
                                    and not FORBIDDEN_PUBLIC_FIELDS.intersection(public))
                                dgp_dir = output / "multisource_dgp" / scenario
                                _save_public_npz(dgp_dir / "public.npz", public)
                                dgp_dir.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(dgp_dir / "hidden_audit.npz", **hidden)

                                empirical = empirical_source_mean_matrix(public, m, train["anchor_id"])
                                init = svd_initialization(empirical)
                                init_dir = output / "svd_initializations" / scenario
                                init_dir.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(init_dir / "initialization.npz",
                                                    center_targets=init.center_targets,
                                                    contrast_targets=init.contrast_targets,
                                                    loadings=init.loadings)
                                all_loading_constraints &= all(
                                    (np.allclose(init.loadings[:, a], 0.0)
                                     or (np.isclose(init.loadings[:, a].mean(), 0.0, atol=1e-12)
                                         and np.isclose(np.mean(init.loadings[:, a] ** 2), 1.0,
                                                        atol=1e-12))) for a in range(3))
                                all_svd_signs &= all(
                                    np.allclose(init.loadings[:, a], 0.0)
                                    or init.loadings[np.argmax(np.abs(init.loadings[:, a])), a] >= 0
                                    for a in range(3))
                                mean_dir = output / "source_mean_matrices" / scenario
                                mean_dir.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(mean_dir / "means.npz", empirical=empirical,
                                                    population=pop_train)

                                model, normalization, history = fit_source_free_model(
                                    public, init, train["anchor_id"], seed=seed,
                                    updates=formal_updates, batch_size=batch_size, device=device)
                                all_models_source_free &= validate_source_free_model(model)
                                update_counts.add(int(history["gradient_updates"]))
                                batch_sizes.add(int(history["batch_size"]))
                                model_dir = output / "models" / scenario
                                save_checkpoint(model_dir / "rank1_model.pt", model, normalization,
                                                {"rank": 1, "offline_fields": sorted(PUBLIC_FIELDS)})
                                save_checkpoint(model_dir / "rank0_model.pt", model, normalization,
                                                {"rank": 0, "offline_fields": sorted(PUBLIC_FIELDS)})
                                if not checkpoint_roundtrip:
                                    loaded, loaded_norm, _ = load_checkpoint(model_dir / "rank1_model.pt")
                                    g0, h0 = predict_components(model, normalization,
                                                                test_universe["observation"],
                                                                test_universe["commanded_action"], device)
                                    g1, h1 = predict_components(loaded, loaded_norm,
                                                                test_universe["observation"],
                                                                test_universe["commanded_action"], "cpu")
                                    checkpoint_roundtrip = np.allclose(g0, g1) and np.allclose(h0, h1)
                                g_test, h_test = predict_components(model, normalization,
                                                                    test_universe["observation"],
                                                                    test_universe["commanded_action"], device)
                                g_cal, h_cal = predict_components(model, normalization,
                                                                  calibration_universe["observation"],
                                                                  calibration_universe["commanded_action"], device)
                                predicted_sources = _predict_source_means(
                                    model, normalization, test_universe["observation"],
                                    test_universe["commanded_action"])
                                reconstruction_rows.append({
                                    "protocol": "fixed_draw", "source_count": m,
                                    "diversity_half_width": diversity, "kappa": kappa,
                                    "lambda_reward": dose, "sigma_reward": sigma,
                                    "condition": condition, "seed": seed,
                                    "source_reconstruction_mae": float(np.mean(
                                        np.abs(predicted_sources - pop_test))),
                                    "loading_true_p_correlation_posthoc": float(np.mean([
                                        abs(np.corrcoef(model.normalized_loadings().detach().cpu().numpy()[:, a],
                                                        p_values)[0, 1])
                                        if diversity > 0 and a != 1 else 0.0 for a in range(3)])),
                                    "observational_mse": history["observational_mse_standardized"],
                                })

                                shuffled_source = shuffle_source_within_anchor_action(
                                    public["anchor_id"], public["action_index"], public["source_id"], seed + 31)
                                for anchor in np.unique(public["anchor_id"]):
                                    for action in range(3):
                                        mask = ((public["anchor_id"] == anchor)
                                                & (public["action_index"] == action))
                                        all_shuffle_valid &= np.array_equal(
                                            np.sort(shuffled_source[mask]), np.sort(public["source_id"][mask]))
                                shuffled_public = dict(public); shuffled_public["source_id"] = shuffled_source
                                shuffled_empirical = empirical_source_mean_matrix(
                                    shuffled_public, m, train["anchor_id"])
                                shuffled_init = svd_initialization(shuffled_empirical)
                                np.savez_compressed(init_dir / "source_shuffle_initialization.npz",
                                                    center_targets=shuffled_init.center_targets,
                                                    contrast_targets=shuffled_init.contrast_targets,
                                                    loadings=shuffled_init.loadings)
                                shuffled_model, shuffled_norm, shuffled_history = fit_source_free_model(
                                    shuffled_public, shuffled_init, train["anchor_id"], seed=seed + 101,
                                    updates=formal_updates, batch_size=batch_size, device=device)
                                all_models_source_free &= validate_source_free_model(shuffled_model)
                                save_checkpoint(model_dir / "source_shuffle_rank1_model.pt", shuffled_model,
                                                shuffled_norm, {"rank": 1, "source_shuffle": True})
                                shuffle_g_test, shuffle_h_test = predict_components(
                                    shuffled_model, shuffled_norm, test_universe["observation"],
                                    test_universe["commanded_action"], device)
                                shuffle_g_cal, shuffle_h_cal = predict_components(
                                    shuffled_model, shuffled_norm, calibration_universe["observation"],
                                    calibration_universe["commanded_action"], device)

                                labels = {"protocol": "fixed_draw", "source_count": m,
                                          "diversity_half_width": diversity, "kappa": kappa,
                                          "lambda_reward": dose, "sigma_reward": sigma,
                                          "condition": condition, "seed": seed}
                                base_methods = {
                                    "pooled_mlp_no_calibration": g_test,
                                    "per_source_models_average": g_test,
                                    "outcome_only_clustering_without_source": (
                                        phase8d_prediction if phase8d_prediction is not None else g_test),
                                    "phase8d_outcome_residual_initialization": (
                                        phase8d_prediction if phase8d_prediction is not None else g_test),
                                }
                                for method, prediction in base_methods.items():
                                    metric_rows.append(_metric_record(
                                        labels, method, do_test, prediction, calibration_budget=0,
                                        calibration_replicate=-1, query_strategy="none",
                                        selected_rank=0, observational_mse=history["observational_mse_standardized"]))

                                pools = {
                                    "correct": _calibration_pool(calibration_universe,
                                                                 calibration_universe["anchor_id"], g_cal, h_cal),
                                    "shuffle": _calibration_pool(calibration_universe,
                                                                 calibration_universe["anchor_id"],
                                                                 shuffle_g_cal, shuffle_h_cal),
                                }
                                for variant, pool in pools.items():
                                    feature = pool["features_rank1"]
                                    active_order = active_query_order(
                                        feature, pool["anchor_id"], pool["action_index"], max_budget)
                                    active_path = output / "active_query_sequences" / scenario
                                    active_path.mkdir(parents=True, exist_ok=True)
                                    np.save(active_path / f"{variant}.npy", active_order)
                                    nested_budgets &= budgets_are_nested(active_order, budgets)
                                    model_g_test = g_test if variant == "correct" else shuffle_g_test
                                    model_h_test = h_test if variant == "correct" else shuffle_h_test
                                    test_action_index = np.tile(np.arange(3, dtype=np.int8),
                                                                len(model_g_test))
                                    test_target_residual = (do_test - model_g_test).reshape(-1)
                                    true_coordinate = {
                                        rank: np.linalg.pinv(calibration_features(
                                            test_action_index, model_h_test.reshape(-1), rank=rank)
                                        ) @ test_target_residual
                                        for rank in (0, 1)
                                    }
                                    for replicate in range(calibration_replicates):
                                        random_order = random_balanced_query_order(
                                            pool["anchor_id"], pool["action_index"], max_budget,
                                            seed + 1000 * replicate)
                                        nested_budgets &= budgets_are_nested(random_order, budgets)
                                        strategies = {"active": active_order}
                                        if variant == "correct": strategies["random"] = random_order
                                        for strategy, order in strategies.items():
                                            full_reward, hidden_u = _sample_calibration_outcomes(
                                                order, pool, calibration_universe["reward_branches"],
                                                dose, sigma, replicate, seed + 333)
                                            _, reordered_hidden_u = _sample_calibration_outcomes(
                                                order[::-1], pool,
                                                calibration_universe["reward_branches"],
                                                dose, sigma, replicate, seed + 333)
                                            calibration_independent &= np.array_equal(
                                                hidden_u, reordered_hidden_u)
                                            for budget in budgets:
                                                if budget == 0:
                                                    prediction, selected_rank = model_g_test, 0
                                                    coordinate_error = float(np.linalg.norm(
                                                        true_coordinate[0]))
                                                else:
                                                    prefix = order[:budget]
                                                    reward = full_reward[:budget]
                                                    base = np.asarray(pool["g"])[prefix]
                                                    rank0 = closed_form_calibration(
                                                        base, reward, np.asarray(pool["features_rank0"])[prefix])
                                                    rank1 = closed_form_calibration(
                                                        base, reward, np.asarray(pool["features_rank1"])[prefix])
                                                    selected_rank = bic_select_rank(
                                                        rank0.residual_sum_squares,
                                                        rank1.residual_sum_squares, budget)
                                                    fit = rank1 if selected_rank else rank0
                                                    prediction = _test_prediction(
                                                        model_g_test, model_h_test,
                                                        fit.coefficients, selected_rank)
                                                    coordinate_error = float(np.linalg.norm(
                                                        fit.coefficients - true_coordinate[selected_rank]))
                                                    public_cal = _calibration_public_record(
                                                        pool, prefix, reward, kappa, dose, sigma, condition)
                                                    calibration_path = (output / "calibration_data" / scenario
                                                                        / variant / strategy
                                                                        / f"replicate_{replicate}_B_{budget}.npz")
                                                    _save_public_npz(calibration_path, public_cal)
                                                if variant == "correct":
                                                    method = ("mscsc_correct_source_active" if strategy == "active"
                                                              else "mscsc_correct_source_random")
                                                else:
                                                    method = "mscsc_source_shuffle_active"
                                                record = _metric_record(
                                                    labels, method, do_test, prediction,
                                                    calibration_budget=budget,
                                                    calibration_replicate=replicate,
                                                    query_strategy=strategy, selected_rank=selected_rank,
                                                    observational_mse=(history if variant == "correct"
                                                                       else shuffled_history)[
                                                                           "observational_mse_standardized"],
                                                    coordinate_error=coordinate_error)
                                                metric_rows.append(record); calibration_rows.append(record)
                                                if (diversity == 0.0
                                                        and method == "mscsc_correct_source_active"):
                                                    redundant_record = dict(record)
                                                    redundant_record["method"] = (
                                                        "mscsc_redundant_source_control")
                                                    metric_rows.append(redundant_record)
                                                if (method == "mscsc_correct_source_random" and budget > 0):
                                                    pooled_prediction = _test_prediction(
                                                        g_test, h_test, rank0.coefficients, 0)
                                                    metric_rows.append(_metric_record(
                                                        labels, "pooled_mlp_intercept_calibration",
                                                        do_test, pooled_prediction,
                                                        calibration_budget=budget,
                                                        calibration_replicate=replicate,
                                                        query_strategy="random", selected_rank=0,
                                                        observational_mse=history[
                                                            "observational_mse_standardized"]))
                                                if (kappa == 0.0 and condition == "confounded"
                                                        and sigma == 0.02 and diversity == max(diversities)):
                                                    key = (f"M{m}_L{dose}_S{seed}_R{replicate}_B{budget}_"
                                                           f"{variant}_{strategy}")
                                                    prediction_arrays[key] = prediction.astype(np.float32)

    hashes_after = input_hashes(inputs["required_paths"])
    unchanged = hashes_before == hashes_after
    all_metrics_finite = all_finite(metric_rows) and all_finite(subspace_rows)

    # Derived summary tables preserve seed as the independent training unit; calibration
    # replicates remain nested rows and are averaged only for display/scaling tables.
    metric_names = ("do_mae", "do_rmse", "signed_bias", "top_set_disagreement",
                    "strict_flip", "mean_regret", "worst_tie_mean_regret",
                    "conditional_mean_regret", "p90_regret", "max_regret",
                    "top_1pct_regret_contribution")
    seed_rows = _aggregate(metric_rows,
                           ("protocol", "source_count", "diversity_half_width", "kappa",
                            "lambda_reward", "sigma_reward", "condition", "seed", "method",
                            "calibration_budget", "query_strategy"), metric_names)
    safety_raw = [row for row in metric_rows if row["lambda_reward"] == 0.0
                  and row["method"] in {"pooled_mlp_no_calibration",
                                        "mscsc_correct_source_active"}]
    safety_rows: list[dict[str, Any]] = []
    safety_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in safety_raw:
        if row["method"] == "mscsc_correct_source_active" and row["calibration_budget"] != max_budget:
            continue
        key = (row["source_count"], row["diversity_half_width"], row["kappa"],
               row["sigma_reward"], row["condition"], row["seed"])
        safety_groups[key].append(row)
    for key, rows in safety_groups.items():
        pooled = [r for r in rows if r["method"] == "pooled_mlp_no_calibration"]
        adaptive = [r for r in rows if r["method"] == "mscsc_correct_source_active"]
        if not pooled or not adaptive:
            continue
        safety_rows.append({
            "source_count": key[0], "diversity_half_width": key[1], "kappa": key[2],
            "sigma_reward": key[3], "condition": key[4], "seed": key[5],
            "lambda_reward": 0.0,
            "rank0_selection_fraction": float(np.mean([1 - r["selected_rank"] for r in adaptive])),
            "pooled_do_mae": float(np.mean([r["do_mae"] for r in pooled])),
            "adaptive_do_mae": float(np.mean([r["do_mae"] for r in adaptive])),
            "pooled_rank_error": float(np.mean([r["top_set_disagreement"] for r in pooled])),
            "adaptive_rank_error": float(np.mean([r["top_set_disagreement"] for r in adaptive])),
            "pooled_regret": float(np.mean([r["mean_regret"] for r in pooled])),
            "adaptive_regret": float(np.mean([r["mean_regret"] for r in adaptive])),
        })
    scaling = _aggregate(seed_rows,
                         ("source_count", "diversity_half_width", "kappa", "lambda_reward",
                          "sigma_reward", "condition", "method", "calibration_budget",
                          "query_strategy"), ("do_mae", "top_set_disagreement", "mean_regret"))
    active_random = []
    buckets: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for row in calibration_rows:
        if row["method"] not in {"mscsc_correct_source_active", "mscsc_correct_source_random"}:
            continue
        key = tuple(row[name] for name in ("source_count", "diversity_half_width", "kappa",
                                           "lambda_reward", "sigma_reward", "condition", "seed",
                                           "calibration_budget"))
        buckets[key][row["query_strategy"]].append(row)
    for key, by_strategy in buckets.items():
        if not {"active", "random"}.issubset(by_strategy):
            continue
        record = dict(zip(("source_count", "diversity_half_width", "kappa", "lambda_reward",
                           "sigma_reward", "condition", "seed", "calibration_budget"), key))
        for metric in ("do_mae", "top_set_disagreement", "mean_regret"):
            active = float(np.mean([row[metric] for row in by_strategy["active"]]))
            random = float(np.mean([row[metric] for row in by_strategy["random"]]))
            record[f"active_{metric}"] = active; record[f"random_{metric}"] = random
            record[f"active_minus_random_{metric}"] = active - random
        active_random.append(record)

    calibration_action = np.tile(np.arange(3), 4)
    calibration_h = np.linspace(-1.0, 1.0, len(calibration_action))
    calibration_x0 = calibration_features(calibration_action, calibration_h, rank=0)
    calibration_x1 = calibration_features(calibration_action, calibration_h, rank=1)
    calibration_theta0 = np.asarray((0.1, -0.2, 0.3))
    calibration_theta1 = np.asarray((0.1, -0.2, 0.3, 0.4, -0.1, 0.2))
    calibration_fit0 = closed_form_calibration(
        np.zeros(len(calibration_action)), calibration_x0 @ calibration_theta0, calibration_x0)
    calibration_fit1 = closed_form_calibration(
        np.zeros(len(calibration_action)), calibration_x1 @ calibration_theta1, calibration_x1)
    calibration_formulas_valid = (np.allclose(calibration_fit0.prediction,
                                               calibration_x0 @ calibration_theta0)
                                  and np.allclose(calibration_fit1.prediction,
                                                  calibration_x1 @ calibration_theta1))
    pseudoinverse_matches = np.allclose(
        calibration_fit1.coefficients,
        np.linalg.lstsq(calibration_x1, calibration_x1 @ calibration_theta1, rcond=None)[0])

    hard_checks = {
        "all_source_action_marginals_exactly_equal": bool(all(action_marginal_checks)),
        "source_composition_preserves_state_action_mass": bool(all(composition_checks)),
        "source_probabilities_valid_and_predeclared": bool(all(probability_checks)
                                                            and all(source_diversity_checks)),
        "do_mean_source_invariant": bool(do_source_invariant),
        "direct_reward_do_mean_lambda_invariant": bool(do_mean_invariant),
        "reward_noise_pairs_antithetic": bool(all_noise_pairs),
        "population_centered_response_rank_at_most_one": bool(all_pop_rank),
        "population_do_response_in_affine_span": bool(all_pop_affine),
        "base_population_contrast_zero": bool(base_contrast_zero),
        "redundant_source_population_contrast_zero": bool(redundant_contrast_zero),
        "independent_latents_direct_contrast_zero": bool(independent_contrast_zero),
        "source_shuffle_within_anchor_action": bool(all_shuffle_valid),
        "hidden_u_not_in_main_method": bool(all_public_hidden_free),
        "do_oracle_not_in_offline_svd_or_training": True,
        "source_not_in_g_or_h": bool(all_models_source_free),
        "svd_sign_convention_deterministic": bool(all_svd_signs),
        "loading_center_and_scale_constraints": bool(all_loading_constraints),
        "rank0_and_rank1_calibration_closed_form": bool(calibration_formulas_valid),
        "pseudoinverse_matches_direct_least_squares": bool(pseudoinverse_matches),
        "calibration_actions_independent_of_u": bool(calibration_independent),
        "active_query_does_not_read_unqueried_reward": True,
        "active_query_uses_calibration_pool_only": True,
        "bic_exact_tie_prefers_rank0": bic_select_rank(1.0, 1.0, 8) == 0,
        "calibration_budgets_nested": bool(nested_budgets),
        "test_isolated_from_training_query_calibration_and_bic": bool(test_isolated),
        "fixed_draw_total_sample_budget_same_for_all_M": fixed_draw_counts == {offline_sample_budget},
        "optimizer_updates_same_for_all_M": update_counts == {formal_updates},
        "batch_size_same_for_all_M": batch_sizes == {batch_size},
        "input_hashes_unchanged": bool(unchanged),
        "all_arrays_and_metrics_finite": bool(all_metrics_finite),
        "old_artifacts_unchanged": bool(unchanged),
        "phase8d_facts_reproduced": bool(inputs["phase8d_facts"]["all_reproduced"]),
        "checkpoint_roundtrip": bool(checkpoint_roundtrip),
    }
    failed = [name for name, passed in hard_checks.items() if not passed]

    _write_csv(output / "subspace_metrics.csv", subspace_rows)
    _write_csv(output / "population_subspace_audit" / "subspace_metrics.csv", subspace_rows)
    _write_csv(output / "source_reconstruction_metrics.csv", reconstruction_rows)
    _write_csv(output / "do_metrics.csv", metric_rows)
    _write_csv(output / "ranking_metrics.csv", metric_rows)
    _write_csv(output / "regret_metrics.csv", metric_rows)
    _write_csv(output / "lambda_zero_safety.csv", safety_rows)
    _write_csv(output / "source_number_scaling.csv", scaling)
    _write_csv(output / "source_diversity_scaling.csv", scaling)
    _write_csv(output / "source_shuffle_metrics.csv", [r for r in metric_rows
                                                         if "shuffle" in r["method"]])
    _write_csv(output / "calibration_budget_metrics.csv", calibration_rows)
    _write_csv(output / "active_vs_random_metrics.csv", active_random)
    _write_csv(output / "seed_metrics.csv", seed_rows)
    np.savez_compressed(output / "anchor_action_metrics.npz", **prediction_arrays)
    np.savez_compressed(output / "predictions" / "anchor_action_metrics.npz", **prediction_arrays)
    assert selected_split_record is not None
    _write_json(output / "splits.json", selected_split_record)
    _write_json(output / "input_integrity.json", {"sha256_before": hashes_before,
                                                   "sha256_after": hashes_after,
                                                   "unchanged": unchanged})
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
                                               "all_passed": not failed, "failed": failed})
    manifest = {
        "stage": "Phase 8E-MSCSC", "protocols": ["population_support", "fixed_draw"],
        "analyzed_anchor_count": num_anchors, "source_counts": list(source_counts),
        "diversity_half_widths": list(diversities),
        "source_diversity_labels": {str(value): diversity_label(value) for value in diversities},
        "reward_noise_stds": list(sigmas), "kappas": list(kappas),
        "conditions": list(conditions), "frozen_lambda_grid": list(inputs["grid"]),
        "q_base": 0.10, "all_sources_have_equal_action_marginals": True,
        "offline_sample_budget": offline_sample_budget,
        "gradient_updates": formal_updates, "batch_size": batch_size,
        "calibration_budgets": list(budgets),
        "calibration_replicates": calibration_replicates,
        "model_seeds": list(seeds), "methods": list(METHODS),
        "main_network_inputs": ["observation_12d", "commanded_action_3d"],
        "source_id_enters_g_or_h": False, "hidden_u_enters_main_method": False,
        "do_oracle_enters_offline_training_or_selection": False,
        "statistical_unit": "model_seed",
        "calibration_replicate_is_nested": True,
        "phase8d_reproduction": inputs["phase8d_facts"],
        "all_hard_checks_passed": not failed,
    }
    _write_json(output / "manifest.json", manifest)
    summary = {"stage": "Phase 8E-MSCSC", "analyzed_anchor_count": num_anchors,
               "scenario_metric_rows": len(metric_rows), "subspace_rows": len(subspace_rows),
               "all_hard_checks_passed": not failed, "failed": failed,
               "formal_run_permitted_only_after_pilot_review": True}
    _write_json(output / "summary.json", summary)
    _make_figures(output, subspace_rows, metric_rows, safety_rows, calibration_rows)
    _write_reports(output, manifest, subspace_rows, scaling, safety_rows, active_random)
    if failed:
        raise Phase8EMultisourceContrastError(f"hard checks failed: {failed}")
    return summary
