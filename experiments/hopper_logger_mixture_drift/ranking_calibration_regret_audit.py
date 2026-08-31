"""Read-only ranking, calibration, and regret audit for Phase 8B-RS.

The audit consumes saved test-anchor predictions.  It never loads a training
checkpoint, retrains a model, or writes inside either input artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ACTIONS = ("minus", "base", "plus")
KAPPAS = (0.0, 0.3)
LAMBDAS = (0.0, 0.05, 0.10, 0.20)
CONDITIONS = ("confounded", "independent_latents")
MIXTURES = ("logger1_heavy", "logger12_balanced", "logger2_heavy")
MODEL_SEEDS = (0, 1, 2)
TOP_ATOL = 1e-7
TOP_RTOL = 1e-7


class RankingCalibrationRegretAuditError(RuntimeError):
    """Raised when the read-only audit cannot be completed faithfully."""


def kappa_name(value: float) -> str:
    return f"kappa_{value:.2f}".replace(".", "p")


def lambda_name(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "p")


def prediction_path(root: Path, kappa: float, strength: float, condition: str,
                    mixture: str, model_seed: int) -> Path:
    return (root / "predictions" / kappa_name(kappa) / lambda_name(strength)
            / condition / mixture / f"seed_{model_seed}.npz")


def top_masks(values: np.ndarray, atol: float = TOP_ATOL,
              rtol: float = TOP_RTOL) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(ACTIONS):
        raise RankingCalibrationRegretAuditError("reward arrays must have shape (anchors, 3)")
    return np.isclose(array, np.max(array, axis=1, keepdims=True), atol=atol, rtol=rtol)


def set_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.all(np.asarray(left, dtype=bool) == np.asarray(right, dtype=bool), axis=1)


def strict_disjoint(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return ~np.any(np.asarray(left, dtype=bool) & np.asarray(right, dtype=bool), axis=1)


def classify_failure_types(do_top: np.ndarray, obs_top: np.ndarray,
                           nn_top: np.ndarray) -> np.ndarray:
    obs_do = set_equal(obs_top, do_top)
    nn_do = set_equal(nn_top, do_top)
    nn_obs = set_equal(nn_top, obs_top)
    result = np.full(len(obs_do), "E", dtype="<U1")
    result[obs_do & nn_do] = "A"
    result[obs_do & ~nn_do] = "B"
    result[~obs_do & nn_obs] = "C"
    result[~obs_do & nn_do] = "D"
    return result


def top_set_regret(do_reward: np.ndarray, candidate_top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reward = np.asarray(do_reward, dtype=np.float64)
    mask = np.asarray(candidate_top, dtype=bool)
    if reward.shape != mask.shape or not np.all(mask.any(axis=1)):
        raise RankingCalibrationRegretAuditError("candidate top sets are empty or misaligned")
    maximum = reward.max(axis=1)
    selected_best = np.max(np.where(mask, reward, -np.inf), axis=1)
    selected_worst = np.min(np.where(mask, reward, np.inf), axis=1)
    return maximum - selected_best, maximum - selected_worst


def do_action_pair(do_reward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reward = np.asarray(do_reward, dtype=np.float64)
    order = np.argsort(-reward, axis=1, kind="stable")
    rows = np.arange(len(reward))
    best, second = order[:, 0], order[:, 1]
    margin = reward[rows, best] - reward[rows, second]
    return best, second, margin


def margins_on_pair(values: np.ndarray, best: np.ndarray, second: np.ndarray) -> np.ndarray:
    reward = np.asarray(values, dtype=np.float64)
    rows = np.arange(len(reward))
    return reward[rows, best] - reward[rows, second]


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return safe_pearson(_average_ranks(left), _average_ranks(right))


def distribution(values: np.ndarray, *, positive_tolerance: float = TOP_ATOL) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p99": None,
                "max": None, "min": None, "conditional_positive_mean": None,
                "top_1pct_total_contribution": None}
    positive = x > positive_tolerance
    count = max(1, int(math.ceil(0.01 * len(x))))
    total = float(np.sum(x))
    contribution = float(np.sum(np.sort(x)[-count:]) / total) if abs(total) > 1e-15 else 0.0
    return {
        "n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)), "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)), "min": float(np.min(x)),
        "conditional_positive_mean": float(np.mean(x[positive])) if np.any(positive) else None,
        "top_1pct_total_contribution": contribution,
    }


def error_distribution(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(x)), "signed_mean": float(np.mean(x)),
        "mae": float(np.mean(np.abs(x))), "rmse": float(np.sqrt(np.mean(x ** 2))),
        "median": float(np.median(x)), "p90_abs": float(np.quantile(np.abs(x), 0.90)),
        "max_abs": float(np.max(np.abs(x))), "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in sorted(set(paths))}


def _require_passed(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("all_passed", value.get("all_hard_checks_passed", False)):
        raise RankingCalibrationRegretAuditError(f"input hard checks did not pass: {path}")


def _oracle_lookup(oracle_path: Path) -> dict[tuple[int, float, float, str, str, str], tuple[float, float]]:
    raw = _load_npz(oracle_path)
    required = {"anchor_id", "kappa", "lambda_reward", "condition", "mixture", "action",
                "augmented_observational_reward", "do_reward"}
    if not required.issubset(raw):
        raise RankingCalibrationRegretAuditError(
            f"oracle anchor table lacks {sorted(required.difference(raw))}")
    lookup: dict[tuple[int, float, float, str, str, str], tuple[float, float]] = {}
    for i in range(len(raw["anchor_id"])):
        key = (int(raw["anchor_id"][i]), float(raw["kappa"][i]),
               float(raw["lambda_reward"][i]), str(raw["condition"][i]),
               str(raw["mixture"][i]), str(raw["action"][i]))
        if key in lookup:
            raise RankingCalibrationRegretAuditError(f"duplicate oracle row: {key}")
        lookup[key] = (float(raw["augmented_observational_reward"][i]),
                       float(raw["do_reward"][i]))
    return lookup


def _expected_prediction_paths(root: Path) -> list[Path]:
    return [prediction_path(root, k, lam, condition, mixture, seed)
            for k in KAPPAS for lam in LAMBDAS for condition in CONDITIONS
            for mixture in MIXTURES for seed in MODEL_SEEDS]


def preflight(neural_root: Path, oracle_root: Path, output_root: Path) -> tuple[list[int], list[Path]]:
    neural, oracle, output = neural_root.resolve(), oracle_root.resolve(), output_root.resolve()
    for required in (neural / "manifest.json", neural / "splits.json",
                     neural / "hard_checks.json", oracle / "manifest.json",
                     oracle / "hard_checks.json", oracle / "anchor_action_metrics.npz"):
        if not required.is_file():
            raise RankingCalibrationRegretAuditError(f"required input is missing: {required}")
    _require_passed(neural / "hard_checks.json")
    _require_passed(oracle / "hard_checks.json")
    if output == neural or output == oracle or neural in output.parents or oracle in output.parents:
        raise RankingCalibrationRegretAuditError("output must not be inside an input artifact")
    if output.exists() and any(output.iterdir()):
        raise RankingCalibrationRegretAuditError(f"output directory is not empty: {output}")
    split = json.loads((neural / "splits.json").read_text(encoding="utf-8"))
    test_ids = [int(value) for value in split.get("test", [])]
    if len(test_ids) != 78 or len(test_ids) != len(set(test_ids)):
        raise RankingCalibrationRegretAuditError("expected exactly 78 unique test anchor IDs")
    predictions = _expected_prediction_paths(neural)
    missing = [path for path in predictions if not path.is_file()]
    if missing:
        examples = "\n".join(str(path) for path in missing[:5])
        raise RankingCalibrationRegretAuditError(
            f"raw test-anchor predictions are unavailable ({len(missing)}/{len(predictions)} missing). "
            f"Examples:\n{examples}")
    return test_ids, predictions


def _group(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def _pair_gap(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _scenario(kappa: float, strength: float, condition: str, mixture: str,
              seed: int, anchor_ids: np.ndarray, do: np.ndarray, obs: np.ndarray,
              nn: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    do_top, obs_top, nn_top = top_masks(do), top_masks(obs), top_masks(nn)
    types = classify_failure_types(do_top, obs_top, nn_top)
    obs_best, obs_worst = top_set_regret(do, obs_top)
    nn_best, nn_worst = top_set_regret(do, nn_top)
    best, second, margin_do = do_action_pair(do)
    margin_obs, margin_nn = margins_on_pair(obs, best, second), margins_on_pair(nn, best, second)
    b_obs, e_nn, b_nn = obs - do, nn - obs, nn - do
    c_obs, c_nn = b_obs.mean(axis=1), b_nn.mean(axis=1)
    d_obs, d_nn = b_obs - c_obs[:, None], b_nn - c_nn[:, None]
    nn_gap, do_gap = _pair_gap(nn), _pair_gap(do)
    obs_do_equal, nn_obs_equal, nn_do_equal = (
        set_equal(obs_top, do_top), set_equal(nn_top, obs_top), set_equal(nn_top, do_top))
    scenario_rows, action_rows = [], []
    for i, anchor in enumerate(anchor_ids):
        base = {"anchor_id": int(anchor), "kappa": kappa, "lambda_reward": strength,
                "condition": condition, "mixture": mixture, "model_seed": seed}
        random_regret = float(np.max(do[i]) - np.mean(do[i]))
        scenario_rows.append({**base,
            "obs_do_top_set_disagreement": float(not obs_do_equal[i]),
            "nn_obs_top_set_disagreement": float(not nn_obs_equal[i]),
            "nn_do_top_set_disagreement": float(not nn_do_equal[i]),
            "obs_do_strict_disjoint_flip": float(strict_disjoint(obs_top[i:i+1], do_top[i:i+1])[0]),
            "nn_obs_strict_disjoint_flip": float(strict_disjoint(nn_top[i:i+1], obs_top[i:i+1])[0]),
            "nn_do_strict_disjoint_flip": float(strict_disjoint(nn_top[i:i+1], do_top[i:i+1])[0]),
            "failure_type": str(types[i]), "obs_regret_best": float(obs_best[i]),
            "obs_regret_worst": float(obs_worst[i]), "nn_regret_best": float(nn_best[i]),
            "nn_regret_worst": float(nn_worst[i]),
            "additional_regret": float(nn_best[i] - obs_best[i]),
            "random_action_regret": random_regret,
            "random_action_oracle_hit_probability": float(np.mean(do_top[i])),
            "margin_do": float(margin_do[i]), "margin_obs_on_do_pair": float(margin_obs[i]),
            "margin_nn_on_do_pair": float(margin_nn[i]),
            "gap_error_obs": float(margin_obs[i] - margin_do[i]),
            "gap_error_nn": float(margin_nn[i] - margin_do[i]),
            "nn_top_second_gap": float(nn_gap[i]), "do_top_second_gap": float(do_gap[i]),
            "mean_abs_b_nn": float(np.mean(np.abs(b_nn[i]))),
            "mean_abs_b_obs": float(np.mean(np.abs(b_obs[i]))),
            "c_obs": float(c_obs[i]), "c_nn": float(c_nn[i]),
            "max_abs_d_obs": float(np.max(np.abs(d_obs[i]))),
            "max_abs_d_nn": float(np.max(np.abs(d_nn[i]))),
            "centered_l2_obs": float(np.linalg.norm(d_obs[i])),
            "centered_l2_nn": float(np.linalg.norm(d_nn[i])),
            "nn_do_top_set_equal": bool(nn_do_equal[i])})
        for j, action in enumerate(ACTIONS):
            action_rows.append({**base, "action": action,
                "r_do": float(do[i, j]), "r_obs": float(obs[i, j]), "r_nn": float(nn[i, j]),
                "b_obs": float(b_obs[i, j]), "e_nn": float(e_nn[i, j]),
                "b_nn": float(b_nn[i, j]), "d_obs": float(d_obs[i, j]),
                "d_nn": float(d_nn[i, j]), "do_top": bool(do_top[i, j]),
                "obs_top": bool(obs_top[i, j]), "nn_top": bool(nn_top[i, j])})
    return scenario_rows, action_rows


def _summaries(scenario_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dims = ("kappa", "lambda_reward", "condition", "mixture", "model_seed")
    scenario_groups, action_groups = _group(scenario_rows, dims), _group(action_rows, (*dims, "action"))
    ranking, failures, calibration, gaps, regrets, seeds, components = [], [], [], [], [], [], []
    comparisons = (("observational_vs_do", "obs_do"), ("neural_vs_observational", "nn_obs"),
                   ("neural_vs_do", "nn_do"))
    for key, rows in scenario_groups.items():
        base = dict(zip(dims, key))
        aligned_actions = [row for row in action_rows if all(row[d] == base[d] for d in dims)]
        for label, prefix in comparisons:
            left = "obs" if prefix.startswith("obs") else "nn"
            right = "do" if prefix.endswith("do") else "obs"
            ranking.append({**base, "comparison": label, "n_anchors": len(rows),
                "top_set_disagreement": float(np.mean([row[f"{prefix}_top_set_disagreement"] for row in rows])),
                "strict_disjoint_flip": float(np.mean([row[f"{prefix}_strict_disjoint_flip"] for row in rows])),
                **{f"{side}_{action}_top_fraction": float(np.mean([
                    row[f"{side}_top"] for row in aligned_actions if row["action"] == action]))
                   for side in (left, right) for action in ACTIONS}})
        for failure in "ABCDE":
            count = sum(row["failure_type"] == failure for row in rows)
            failures.append({**base, "failure_type": failure, "anchor_count": count,
                             "fraction": count / len(rows), "n_anchors": len(rows)})
        true_margin = np.asarray([row["margin_do"] for row in rows])
        non_tie = true_margin > TOP_ATOL
        for source, margin_key, error_key in (
                ("observational", "margin_obs_on_do_pair", "gap_error_obs"),
                ("neural", "margin_nn_on_do_pair", "gap_error_nn")):
            margin = np.asarray([row[margin_key] for row in rows])
            error = np.asarray([row[error_key] for row in rows])
            gap_row = {**base, "source": source, "n_anchors": len(rows),
                "n_non_tied_do_pairs": int(np.sum(non_tie)),
                "signed_gap_error": float(np.mean(error)), "absolute_gap_error": float(np.mean(np.abs(error))),
                "median_absolute_gap_error": float(np.median(np.abs(error))),
                "p90_absolute_gap_error": float(np.quantile(np.abs(error), 0.90)),
                "mean_margin_do": float(np.mean(true_margin)), "mean_margin_on_do_pair": float(np.mean(margin)),
                "aggregate_margin_ratio": (float(np.mean(margin) / np.mean(true_margin))
                    if abs(np.mean(true_margin)) > 1e-15 else None),
                "pearson_true_margin": safe_pearson(true_margin, margin),
                "spearman_true_margin": safe_spearman(true_margin, margin),
                "sign_preservation_rate": (float(np.mean(margin[non_tie] > TOP_ATOL))
                    if np.any(non_tie) else None)}
            if source == "neural":
                matched = np.asarray([row["nn_do_top_set_equal"] for row in rows], dtype=bool)
                nn_gap = np.asarray([row["nn_top_second_gap"] for row in rows])
                do_gap = np.asarray([row["do_top_second_gap"] for row in rows])
                delta = nn_gap[matched] - do_gap[matched]
                gap_row.update({"n_matching_top_sets": int(np.sum(matched)),
                    "overconfident_fraction": float(np.mean(delta > TOP_ATOL)) if len(delta) else None,
                    "underconfident_fraction": float(np.mean(delta < -TOP_ATOL)) if len(delta) else None,
                    "mean_matched_gap_difference": float(np.mean(delta)) if len(delta) else None,
                    "matched_aggregate_gap_ratio": (float(np.mean(nn_gap[matched]) / np.mean(do_gap[matched]))
                        if len(delta) and abs(np.mean(do_gap[matched])) > 1e-15 else None)})
            gaps.append(gap_row)
        for source, best_key, worst_key in (
                ("observational", "obs_regret_best", "obs_regret_worst"),
                ("neural", "nn_regret_best", "nn_regret_worst")):
            for set_rule, value_key in (("best", best_key), ("worst", worst_key)):
                values = np.asarray([row[value_key] for row in rows])
                regrets.append({**base, "source": source, "top_set_rule": set_rule,
                                **distribution(values)})
        additional = np.asarray([row["additional_regret"] for row in rows])
        regrets.append({**base, "source": "neural_minus_observational", "top_set_rule": "best",
            **distribution(additional), "positive_fraction": float(np.mean(additional > TOP_ATOL)),
            "zero_fraction": float(np.mean(np.abs(additional) <= TOP_ATOL)),
            "negative_fraction": float(np.mean(additional < -TOP_ATOL))})
        seeds.append({**base, "n_anchors": len(rows),
            "obs_do_rank_error": float(np.mean([row["obs_do_top_set_disagreement"] for row in rows])),
            "nn_obs_rank_error": float(np.mean([row["nn_obs_top_set_disagreement"] for row in rows])),
            "nn_do_rank_error": float(np.mean([row["nn_do_top_set_disagreement"] for row in rows])),
            "obs_regret": float(np.mean([row["obs_regret_best"] for row in rows])),
            "nn_regret": float(np.mean([row["nn_regret_best"] for row in rows])),
            "neural_calibration_mae": float(np.mean([row["mean_abs_b_nn"] for row in rows])),
            "neural_gap_absolute_error": float(np.mean(np.abs([row["gap_error_nn"] for row in rows]))),
            "random_action_regret": float(np.mean([row["random_action_regret"] for row in rows]))})
        for source, c_key, d_key, l2_key in (
                ("population_observational", "c_obs", "max_abs_d_obs", "centered_l2_obs"),
                ("neural_do", "c_nn", "max_abs_d_nn", "centered_l2_nn")):
            components.append({**base, "record_type": "bias_component", "error_type": source,
                "action": "all", "n_anchors": len(rows),
                "mean_abs_common_shift": float(np.mean(np.abs([row[c_key] for row in rows]))),
                "mean_max_abs_action_residual": float(np.mean([row[d_key] for row in rows])),
                "mean_centered_bias_l2": float(np.mean([row[l2_key] for row in rows]))})
    for key, rows in action_groups.items():
        base = dict(zip((*dims, "action"), key))
        for error_type, field in (("population_causal_bias", "b_obs"),
                                  ("neural_approximation_error", "e_nn"),
                                  ("total_neural_do_error", "b_nn")):
            calibration.append({**base, "record_type": "action_error", "error_type": error_type,
                                **error_distribution(np.asarray([row[field] for row in rows]))})
    calibration.extend(components)
    return {"ranking": ranking, "failures": failures, "calibration": calibration,
            "gaps": gaps, "regrets": regrets, "seeds": seeds}


def _main_table(scenario_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for strength in LAMBDAS:
        rows = [row for row in scenario_rows
                if row["condition"] == "confounded" and row["lambda_reward"] == strength]
        result.append({"lambda": strength,
            "obs_do_rank_error": float(np.mean([r["obs_do_top_set_disagreement"] for r in rows])),
            "neural_obs_rank_error": float(np.mean([r["nn_obs_top_set_disagreement"] for r in rows])),
            "neural_do_rank_error": float(np.mean([r["nn_do_top_set_disagreement"] for r in rows])),
            "obs_regret": float(np.mean([r["obs_regret_best"] for r in rows])),
            "neural_regret": float(np.mean([r["nn_regret_best"] for r in rows])),
            "random_action_regret": float(np.mean([r["random_action_regret"] for r in rows])),
            "neural_calibration_mae": float(np.mean([r["mean_abs_b_nn"] for r in rows])),
            "neural_gap_error": float(np.mean(np.abs([r["gap_error_nn"] for r in rows]))),
            "obs_gap_error": float(np.mean(np.abs([r["gap_error_obs"] for r in rows]))),
            "neural_gap_signed_error": float(np.mean([r["gap_error_nn"] for r in rows])),
            "obs_gap_signed_error": float(np.mean([r["gap_error_obs"] for r in rows])),
            "neural_oracle_hit_rate": 1.0 - float(np.mean([r["nn_do_strict_disjoint_flip"] for r in rows])),
            "observational_oracle_hit_rate": 1.0 - float(np.mean([r["obs_do_strict_disjoint_flip"] for r in rows])),
            "random_oracle_hit_rate": float(np.mean([r["random_action_oracle_hit_probability"] for r in rows]))})
    return result


def _mean_by_lambda(rows: list[dict[str, Any]], field: str, condition: str) -> list[float]:
    return [float(np.mean([row[field] for row in rows
                           if row["condition"] == condition and row["lambda_reward"] == value]))
            for value in LAMBDAS]


def _make_figures(output: Path, scenarios: list[dict[str, Any]],
                  actions: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / name, dpi=300)
        plt.close()

    plt.figure(figsize=(7, 4.5))
    for field, label in (("obs_do_top_set_disagreement", "observational vs do"),
                         ("nn_obs_top_set_disagreement", "neural vs observational"),
                         ("nn_do_top_set_disagreement", "neural vs do")):
        plt.plot(LAMBDAS, _mean_by_lambda(scenarios, field, "confounded"), marker="o", label=label)
    plt.xlabel("lambda"); plt.ylabel("top-set disagreement"); plt.ylim(bottom=0); plt.legend()
    save("ranking_error_by_lambda.png")

    plt.figure(figsize=(7, 4.5))
    for failure in "BCD":
        values = [np.mean([row["failure_type"] == failure for row in scenarios
                           if row["condition"] == "confounded" and row["lambda_reward"] == lam])
                  for lam in LAMBDAS]
        plt.plot(LAMBDAS, values, marker="o", label=f"Type {failure}")
    plt.xlabel("lambda"); plt.ylabel("failure-source fraction"); plt.ylim(bottom=0); plt.legend()
    save("observational_vs_neural_error_source.png")

    confounded_actions = [row for row in actions if row["condition"] == "confounded"]
    plt.figure(figsize=(6, 5))
    plt.scatter([r["r_do"] for r in confounded_actions], [r["r_nn"] for r in confounded_actions],
                s=5, alpha=0.2)
    low = min(min(r["r_do"] for r in confounded_actions), min(r["r_nn"] for r in confounded_actions))
    high = max(max(r["r_do"] for r in confounded_actions), max(r["r_nn"] for r in confounded_actions))
    plt.plot([low, high], [low, high]); plt.xlabel("R_do"); plt.ylabel("R_nn")
    save("neural_vs_do_reward_scatter.png")

    plt.figure(figsize=(7, 4.5))
    for field, label in (("gap_error_obs", "observational"), ("gap_error_nn", "neural")):
        values = [np.mean(np.abs([row[field] for row in scenarios
                                 if row["condition"] == "confounded" and row["lambda_reward"] == lam]))
                  for lam in LAMBDAS]
        plt.plot(LAMBDAS, values, marker="o", label=label)
    plt.xlabel("lambda"); plt.ylabel("mean absolute gap error"); plt.ylim(bottom=0); plt.legend()
    save("gap_distortion_by_lambda.png")

    confounded = [row for row in scenarios if row["condition"] == "confounded"]
    plt.figure(figsize=(6, 5)); plt.scatter([r["margin_do"] for r in confounded],
        [r["margin_nn_on_do_pair"] for r in confounded], s=6, alpha=0.25)
    low = min(min(r["margin_do"] for r in confounded), min(r["margin_nn_on_do_pair"] for r in confounded))
    high = max(max(r["margin_do"] for r in confounded), max(r["margin_nn_on_do_pair"] for r in confounded))
    plt.plot([low, high], [low, high]); plt.xlabel("do top-second margin"); plt.ylabel("neural margin on do pair")
    save("true_margin_vs_neural_margin.png")

    plt.figure(figsize=(6, 5)); plt.scatter([r["mean_abs_b_nn"] for r in confounded],
        [r["nn_regret_best"] for r in confounded], s=6, alpha=0.25)
    plt.xlabel("neural calibration MAE across actions"); plt.ylabel("neural decision regret")
    save("calibration_error_vs_regret.png")

    plt.figure(figsize=(6, 5)); plt.scatter([r["max_abs_d_nn"] for r in confounded],
        [r["nn_regret_best"] for r in confounded], s=6, alpha=0.25)
    plt.xlabel("max centered action-dependent neural bias"); plt.ylabel("neural decision regret")
    save("centered_bias_vs_regret.png")

    plt.figure(figsize=(6, 5)); plt.scatter([r["obs_regret_best"] for r in confounded],
        [r["nn_regret_best"] for r in confounded], s=6, alpha=0.25)
    high = max(max(r["obs_regret_best"] for r in confounded), max(r["nn_regret_best"] for r in confounded))
    plt.plot([0, high], [0, high]); plt.xlabel("observational regret"); plt.ylabel("neural regret")
    save("observational_regret_vs_neural_regret.png")

    plt.figure(figsize=(7, 4.5))
    for failure in "ABCDE":
        values = [np.mean([row["failure_type"] == failure for row in confounded
                           if row["lambda_reward"] == lam]) for lam in LAMBDAS]
        plt.plot(LAMBDAS, values, marker="o", label=f"Type {failure}")
    plt.xlabel("lambda"); plt.ylabel("failure-type fraction"); plt.ylim(bottom=0); plt.legend(ncol=2)
    save("failure_type_fraction_vs_lambda.png")

    plt.figure(figsize=(7, 4.5))
    for condition in CONDITIONS:
        plt.plot(LAMBDAS, _mean_by_lambda(scenarios, "nn_do_top_set_disagreement", condition),
                 marker="o", label=condition)
    plt.xlabel("lambda"); plt.ylabel("neural-vs-do top-set disagreement"); plt.ylim(bottom=0); plt.legend()
    save("confounded_vs_independent_ranking.png")


def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _report(output: Path, main: list[dict[str, Any]], scenarios: list[dict[str, Any]],
            calibration: list[dict[str, Any]], hard_checks: Mapping[str, Any]) -> None:
    confounded = [row for row in scenarios if row["condition"] == "confounded"]
    mean_common = float(np.mean(np.abs([row["c_nn"] for row in confounded])))
    mean_centered = float(np.mean([row["max_abs_d_nn"] for row in confounded]))
    first, last = main[0], main[-1]
    observational_usually_correct = all(row["obs_do_rank_error"] < 0.5 for row in main)
    neural_additional_rank_error = last["neural_do_rank_error"] - last["obs_do_rank_error"]
    component_label = ("common calibration shift" if mean_common > mean_centered
                       else "action-dependent distortion")
    gap_direction = ("夸大" if last["neural_gap_signed_error"] > TOP_ATOL else
                     "缩小" if last["neural_gap_signed_error"] < -TOP_ATOL else "无系统方向")
    additional_regret = last["neural_regret"] - last["obs_regret"]
    regret_source = ("neural approximation" if additional_regret > last["obs_regret"] else
                     "observational confounding" if last["obs_regret"] > max(additional_regret, 0.0)
                     else "neither source dominates")
    closer = "A：有用但校准较差的 one-step prior" if (
        np.mean([row["neural_regret"] < row["random_action_regret"] for row in main]) > 0.5
        and np.mean([row["neural_oracle_hit_rate"] > row["random_oracle_hit_rate"] for row in main]) > 0.5
    ) else "B：存在直接误导 one-step 决策的风险"
    lines = [
        "# Phase 8B-RS Ranking–Calibration–Regret Audit", "",
        "## Scope and evidence", "",
        "This is a read-only **one-step reward** audit. `R_do`, `R_obs`, and `R_nn` are not Q-values. "
        "All results use the 78 held-out test anchors and three saved neural seeds. Anchor is the "
        "decision unit; seed variation is reported descriptively and seed×anchor rows are not treated "
        "as independent experimental replications.", "",
        "## Compact primary table", "",
        "| lambda | obs→do rank error | neural→obs rank error | neural→do rank error | obs regret | neural regret | neural calibration MAE | neural gap error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main:
        lines.append("| " + " | ".join([_fmt(row["lambda"], 2), _fmt(row["obs_do_rank_error"]),
            _fmt(row["neural_obs_rank_error"]), _fmt(row["neural_do_rank_error"]),
            _fmt(row["obs_regret"]), _fmt(row["neural_regret"]),
            _fmt(row["neural_calibration_mae"]), _fmt(row["neural_gap_error"])]) + " |")
    lines += ["", "Rank error is exact top-set disagreement under the existing project tolerance "
              f"(`atol={TOP_ATOL:g}`, `rtol={TOP_RTOL:g}`). Regret uses the best member of a tied top set.", "",
        "## Direct answers", "",
        f"**Q1. Is observational ranking usually correct?** **{'Yes' if observational_usually_correct else 'No'}** "
        f"under the literal majority criterion. At λ={last['lambda']:.2f}, observational-vs-do top-set "
        f"disagreement is {_fmt(last['obs_do_rank_error'])}; the complete dose response is in the table.", "",
        f"**Q2. How much additional ranking error comes from the neural model?** At λ={last['lambda']:.2f}, "
        f"neural-vs-do disagreement is {_fmt(last['neural_do_rank_error'])}, while neural-vs-observational "
        f"disagreement is {_fmt(last['neural_obs_rank_error'])}; the net neural-minus-observational do-error is "
        f"{_fmt(neural_additional_rank_error)}. Type B isolates neural-created errors and "
        "Type C isolates inherited observational errors in `failure_type_metrics.csv`.", "",
        f"**Q3. Common calibration or action-dependent distortion?** Mean |common shift| is {_fmt(mean_common)}; "
        f"mean max-action centered distortion is {_fmt(mean_centered)}. Their full distributions, rather than "
        f"a thresholded label, are retained in `anchor_action_metrics.npz`. On mean magnitude, **{component_label}** "
        "is larger.", "",
        f"**Q4. Is the top-second gap systematically distorted?** At λ={last['lambda']:.2f}, the mean absolute "
        f"neural gap error is {_fmt(last['neural_gap_error'])}, and the signed mean indicates **{gap_direction}** "
        f"({_fmt(last['neural_gap_signed_error'])}). Matched-top over/underconfidence "
        "fractions are in `gap_metrics.csv`.", "",
        f"**Q5. Where does decision regret come from?** At λ={last['lambda']:.2f}, mean observational regret is "
        f"{_fmt(last['obs_regret'])}, mean neural regret is {_fmt(last['neural_regret'])}, and their difference is "
        f"{_fmt(additional_regret)}. By mean positive contribution, **{regret_source}** is larger. The signed "
        "additional-regret distribution separates neural harm from accidental correction.", "",
        f"**Q6. Does increasing λ move from calibration error to decision error?** From λ={first['lambda']:.2f} "
        f"to λ={last['lambda']:.2f}, observational rank error changes {_fmt(first['obs_do_rank_error'])}→"
        f"{_fmt(last['obs_do_rank_error'])}, and neural rank error changes {_fmt(first['neural_do_rank_error'])}→"
        f"{_fmt(last['neural_do_rank_error'])}. This is one-step evidence only.", "",
        f"**Q7. Which prior is it closer to?** Relative to a uniform-random action baseline, the saved model is "
        f"closer to **{closer}**. This does not establish SAC transfer, positive or negative.", "",
        "## Ranking-correct versus ranking-wrong anchors", "",
        "No arbitrary 'large error' threshold was introduced. Ranking correctness is discrete; calibration, "
        "gap error, centered bias, and regret remain continuous and are shown by distributions and scatter plots. "
        "Thus categories I/II and III/IV are represented as continua rather than forced binary counts.", "",
        "## Negative controls", "",
        "`independent_latents` is analyzed with the identical pipeline. Any neural ranking, gap, or regret error "
        "there is ordinary approximation error, not direct reward confounding. Base-action action-level errors and "
        "λ=0 are retained in `calibration_metrics.csv`.", "",
        "## Hard checks", "",
        f"All checks passed: **{hard_checks['all_passed']}**. Maximum identity residual "
        f"`b_nn-(b_obs+e_nn)` = {hard_checks['checks']['max_bias_identity_residual']:.3e}. "
        "Input hashes were unchanged.", "",
        "## Boundary", "",
        "The audit describes three discrete candidate actions at fixed states and one-step rewards. It cannot by "
        "itself prove online SAC negative transfer or long-horizon policy value.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "analysis-report.md").write_text(
        "# Analysis report\n\nSee `REPORT.md` for the evidence-first audit and direct answers.\n",
        encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        "# Statistical appendix\n\nThe primary unit is held-out anchor (n=78). Three model seeds quantify "
        "optimization variation descriptively. No p-values are used: the comparisons are paired deterministic "
        "decompositions on a fixed audit set. Correlations are reported as Pearson and Spearman when both "
        "variables have nonzero variance; otherwise they are N/A.\n", encoding="utf-8")


def _figure_catalog(output: Path) -> None:
    purposes = {
        "ranking_error_by_lambda.png": "Tracks the three ranking disagreements as direct U→R strength changes.",
        "observational_vs_neural_error_source.png": "Separates neural-created, inherited, and corrected ranking errors.",
        "neural_vs_do_reward_scatter.png": "Shows absolute reward calibration against the do oracle.",
        "gap_distortion_by_lambda.png": "Tests whether action-relative margins deteriorate with λ.",
        "true_margin_vs_neural_margin.png": "Checks preservation and scaling of do action gaps.",
        "calibration_error_vs_regret.png": "Shows why magnitude error need not imply decision regret.",
        "centered_bias_vs_regret.png": "Links action-dependent distortion, rather than common shift, to regret.",
        "observational_regret_vs_neural_regret.png": "Displays neural additional harm or accidental correction.",
        "failure_type_fraction_vs_lambda.png": "Shows the full A/B/C/D/E decomposition over λ.",
        "confounded_vs_independent_ranking.png": "Uses independent latents as the neural-error negative control.",
    }
    lines = ["# Figure catalog", ""]
    for name, purpose in purposes.items():
        lines += [f"## `{name}`", "", f"- **Purpose:** {purpose}",
                  "- **Data:** 78 test anchors; three neural seeds; saved Phase 8B-RS predictions.",
                  "- **Interpretation:** Read together with the exact numeric CSV tables; points sharing an anchor "
                  "across conditions are repeated measurements, not independent runs.",
                  "- **Implication:** Determines whether the evidence concerns calibration only or actual action choice.", ""]
    (output / "figure-catalog.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(neural_root: Path, oracle_root: Path, output_root: Path) -> dict[str, Any]:
    neural, oracle, output = Path(neural_root), Path(oracle_root), Path(output_root)
    test_ids, prediction_paths = preflight(neural, oracle, output)
    input_paths = [neural / "manifest.json", neural / "splits.json", neural / "hard_checks.json",
                   oracle / "manifest.json", oracle / "hard_checks.json",
                   oracle / "anchor_action_metrics.npz", *prediction_paths]
    before = _hashes(input_paths)
    lookup = _oracle_lookup(oracle / "anchor_action_metrics.npz")
    scenarios: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    population_match = True
    only_test_anchors = True
    all_finite = True
    for kappa in KAPPAS:
        for strength in LAMBDAS:
            for condition in CONDITIONS:
                for mixture in MIXTURES:
                    obs = np.asarray([[lookup[(anchor, kappa, strength, condition, mixture, action)][0]
                                       for action in ACTIONS] for anchor in test_ids], dtype=np.float64)
                    do = np.asarray([[lookup[(anchor, kappa, strength, condition, mixture, action)][1]
                                      for action in ACTIONS] for anchor in test_ids], dtype=np.float64)
                    for seed in MODEL_SEEDS:
                        raw = _load_npz(prediction_path(neural, kappa, strength, condition, mixture, seed))
                        if not {"anchor_id", "prediction", "population_target"}.issubset(raw):
                            raise RankingCalibrationRegretAuditError("prediction file schema is incomplete")
                        ids = np.asarray(raw["anchor_id"], dtype=np.int64)
                        pred = np.asarray(raw["prediction"], dtype=np.float64)
                        target = np.asarray(raw["population_target"], dtype=np.float64)
                        if ids.tolist() != test_ids or pred.shape != (78, 3) or target.shape != (78, 3):
                            raise RankingCalibrationRegretAuditError("prediction rows are not canonical test anchors")
                        only_test_anchors &= set(ids.tolist()) == set(test_ids)
                        population_match &= np.allclose(target, obs, atol=1e-12, rtol=1e-12)
                        all_finite &= np.isfinite(pred).all() and np.isfinite(target).all()
                        srows, arows = _scenario(kappa, strength, condition, mixture, seed,
                                                 ids, do, obs, pred)
                        scenarios.extend(srows); actions.extend(arows)
    summaries = _summaries(scenarios, actions)
    main = _main_table(scenarios)
    residual = max(abs(row["b_nn"] - (row["b_obs"] + row["e_nn"])) for row in actions)
    checks: dict[str, Any] = {
        "input_hard_checks_passed": True, "all_144_prediction_files_present": len(prediction_paths) == 144,
        "only_78_test_anchors_used": bool(only_test_anchors),
        "prediction_population_targets_match_oracle": bool(population_match),
        "all_arrays_finite": bool(all_finite), "all_top_sets_nonempty": True,
        "bias_identity_within_floating_precision": residual <= 1e-12,
        "max_bias_identity_residual": residual, "existing_top_tolerance_used": True,
    }
    failed = [key for key, value in checks.items()
              if key != "max_bias_identity_residual" and not bool(value)]
    if failed:
        raise RankingCalibrationRegretAuditError(f"hard checks failed before writing output: {failed}")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "ranking_metrics.csv", summaries["ranking"])
    _write_csv(output / "calibration_metrics.csv", summaries["calibration"])
    _write_csv(output / "gap_metrics.csv", summaries["gaps"])
    _write_csv(output / "regret_metrics.csv", summaries["regrets"])
    _write_csv(output / "failure_type_metrics.csv", summaries["failures"])
    _write_csv(output / "seed_metrics.csv", summaries["seeds"])
    _write_csv(output / "primary_table.csv", main)
    np.savez_compressed(output / "anchor_action_metrics.npz",
        **{key: np.asarray([row[key] for row in actions]) for key in actions[0]},
        **{f"scenario__{key}": np.asarray([row[key] for row in scenarios]) for key in scenarios[0]})
    _make_figures(output, scenarios, actions)
    after = _hashes(input_paths)
    checks["input_hashes_unchanged"] = before == after
    checks["old_artifacts_unchanged"] = before == after
    failed = [key for key, value in checks.items()
              if key != "max_bias_identity_residual" and not bool(value)]
    hard = {"checks": checks, "all_passed": not failed, "failed": failed}
    _write_json(output / "hard_checks.json", hard)
    _write_json(output / "input_integrity.json", {"sha256_before": before, "sha256_after": after,
                                                    "unchanged": before == after})
    summary = {"stage": "Phase 8B-RS Ranking-Calibration-Regret Audit",
               "test_anchor_count": 78, "model_seeds": list(MODEL_SEEDS),
               "tie_tolerance": {"atol": TOP_ATOL, "rtol": TOP_RTOL},
               "primary_table": main, "all_hard_checks_passed": not failed,
               "one_step_only": True, "online_transfer_claimed": False}
    _write_json(output / "summary.json", summary)
    _report(output, main, scenarios, summaries["calibration"], hard)
    _figure_catalog(output)
    return summary
