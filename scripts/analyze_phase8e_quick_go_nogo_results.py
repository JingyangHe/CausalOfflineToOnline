"""Strict read-only analysis for completed Phase 8E-Q artifacts.

The script never edits the experiment directory.  It treats model seed as the
independent unit and averages the five nested calibration replicates before
performing paired comparisons.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


METRICS = ("do_mae", "top_set_disagreement", "mean_regret")
METRIC_LABELS = {
    "do_mae": "Do-oracle MAE",
    "top_set_disagreement": "Top-action disagreement",
    "mean_regret": "Mean one-step regret",
}
PRIMARY_COMPARISONS = (
    (
        "Correct - Shuffle, M=5 diverse",
        ("M5_diverse", "MSCSC_correct_source", 64),
        ("M5_diverse", "MSCSC_source_shuffle", 64),
    ),
    (
        "M=5 diverse - M=2 diverse",
        ("M5_diverse", "MSCSC_correct_source", 64),
        ("M2_diverse", "MSCSC_correct_source", 64),
    ),
    (
        "M=5 redundant - M=5 diverse",
        ("M5_redundant", "MSCSC_correct_source", 64),
        ("M5_diverse", "MSCSC_correct_source", 64),
    ),
    (
        "B=64 - B=0",
        ("M5_diverse", "MSCSC_correct_source", 64),
        ("M5_diverse", "MSCSC_correct_source", 0),
    ),
)
COLORS = ("#0072B2", "#D55E00", "#009E73")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def seed_cell(
    rows: Sequence[Mapping[str, str]],
    setting: str,
    method: str,
    budget: int,
    *,
    dose: float = 0.05,
    condition: str = "confounded",
) -> list[Mapping[str, str]]:
    selected = [
        row
        for row in rows
        if row["setting"] == setting
        and row["method"] == method
        and int(row["calibration_budget"]) == budget
        and math.isclose(float(row["lambda_reward"]), dose, abs_tol=1e-12)
        and row["condition"] == condition
    ]
    selected.sort(key=lambda row: int(row["seed"]))
    if len(selected) != 3 or [int(row["seed"]) for row in selected] != [0, 1, 2]:
        raise RuntimeError(f"missing or duplicated seed cell: {setting}, {method}, {budget}, {dose}, {condition}")
    return selected


def exact_sign_flip_p(differences: np.ndarray) -> float:
    """Two-sided exact paired randomization p-value using |mean difference|."""
    values = np.asarray(differences, dtype=np.float64)
    observed = abs(float(values.mean()))
    distribution = [
        abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(distribution) >= observed - 1e-15))


def paired_statistics(differences: Iterable[float]) -> dict[str, object]:
    values = np.asarray(tuple(differences), dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    critical = float(student_t.ppf(0.975, n - 1))
    half_width = critical * sd / math.sqrt(n)
    return {
        "n_model_seeds": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "paired_dz": mean / sd if sd > 0 else None,
        "exact_sign_flip_p": exact_sign_flip_p(values),
        "negative_seeds": int(np.sum(values < 0)),
        "positive_seeds": int(np.sum(values > 0)),
        "zero_seeds": int(np.sum(values == 0)),
        "per_seed_differences": json.dumps(values.tolist()),
    }


def holm_adjust(rows: list[dict[str, object]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["exact_sign_flip_p"]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(rows[index]["exact_sign_flip_p"]))
        running = max(running, adjusted)
        rows[index]["holm_p_across_12_primary_tests"] = running


def comparison_rows(seed_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name, left_key, right_key in PRIMARY_COMPARISONS:
        left = seed_cell(seed_rows, *left_key)
        right = seed_cell(seed_rows, *right_key)
        for metric in METRICS:
            left_values = np.asarray([float(row[metric]) for row in left])
            right_values = np.asarray([float(row[metric]) for row in right])
            stats = paired_statistics(left_values - right_values)
            result.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "left_mean": float(left_values.mean()),
                    "right_mean": float(right_values.mean()),
                    "relative_difference_percent": 100.0 * float((left_values - right_values).mean()) / float(right_values.mean()),
                    "favorable_direction": "negative",
                    **stats,
                }
            )
    holm_adjust(result)
    return result


def auxiliary_method_rows(seed_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    pooled = seed_cell(seed_rows, "M5_diverse", "pooled_rank0", 64)
    for method in ("MSCSC_correct_source", "MSCSC_source_shuffle"):
        adaptive = seed_cell(seed_rows, "M5_diverse", method, 64)
        for metric in METRICS:
            av = np.asarray([float(row[metric]) for row in adaptive])
            pv = np.asarray([float(row[metric]) for row in pooled])
            result.append(
                {
                    "comparison": f"{method} - pooled_rank0, M=5 diverse, B=64",
                    "metric": metric,
                    "adaptive_mean": float(av.mean()),
                    "pooled_mean": float(pv.mean()),
                    **paired_statistics(av - pv),
                }
            )
    return result


def budget_rows(seed_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    levels: list[dict[str, object]] = []
    cells = {budget: seed_cell(seed_rows, "M5_diverse", "MSCSC_correct_source", budget) for budget in (0, 16, 64)}
    for budget, rows in cells.items():
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows])
            levels.append(
                {
                    "budget": budget,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd_across_model_seeds": float(values.std(ddof=1)),
                    "per_seed": json.dumps(values.tolist()),
                }
            )
    contrasts: list[dict[str, object]] = []
    for left_budget, right_budget in ((16, 0), (64, 16), (64, 0)):
        for metric in METRICS:
            left = np.asarray([float(row[metric]) for row in cells[left_budget]])
            right = np.asarray([float(row[metric]) for row in cells[right_budget]])
            contrasts.append(
                {
                    "comparison": f"B={left_budget} - B={right_budget}",
                    "metric": metric,
                    "relative_difference_percent": 100.0 * float((left - right).mean()) / float(right.mean()),
                    **paired_statistics(left - right),
                }
            )
    return levels, contrasts


def control_rows(seed_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lambda_zero: list[dict[str, object]] = []
    for method in ("pooled_rank0", "MSCSC_correct_source", "MSCSC_source_shuffle"):
        for budget in (0, 16, 64):
            rows = seed_cell(seed_rows, "M5_diverse", method, budget, dose=0.0)
            record: dict[str, object] = {"method": method, "budget": budget}
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in rows])
                record[metric + "_mean"] = float(values.mean())
                record[metric + "_sd"] = float(values.std(ddof=1))
            rank = np.asarray([float(row["selected_rank"]) for row in rows])
            record["rank1_selection_rate"] = float(rank.mean())
            record["rank1_selections_out_of_15"] = int(round(float(rank.sum()) * 5))
            lambda_zero.append(record)

    negative: list[dict[str, object]] = []
    for method in ("pooled_rank0", "MSCSC_correct_source", "MSCSC_source_shuffle"):
        rows = seed_cell(
            seed_rows,
            "M5_diverse",
            method,
            64,
            dose=0.05,
            condition="independent_latents",
        )
        record = {"method": method, "budget": 64}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows])
            record[metric + "_mean"] = float(values.mean())
            record[metric + "_sd"] = float(values.std(ddof=1))
        rank = np.asarray([float(row["selected_rank"]) for row in rows])
        record["rank1_selection_rate"] = float(rank.mean())
        record["rank1_selections_out_of_15"] = int(round(float(rank.sum()) * 5))
        negative.append(record)
    return lambda_zero, negative


def control_contrast_rows(seed_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    definitions = (
        ("Lambda=0 correct - pooled, B=16", 0.0, "confounded", "MSCSC_correct_source", "pooled_rank0", 16),
        ("Lambda=0 correct - pooled, B=64", 0.0, "confounded", "MSCSC_correct_source", "pooled_rank0", 64),
        ("Independent correct - shuffle, B=64", 0.05, "independent_latents", "MSCSC_correct_source", "MSCSC_source_shuffle", 64),
        ("Independent correct - pooled, B=64", 0.05, "independent_latents", "MSCSC_correct_source", "pooled_rank0", 64),
    )
    result: list[dict[str, object]] = []
    for name, dose, condition, left_method, right_method, budget in definitions:
        left = seed_cell(seed_rows, "M5_diverse", left_method, budget, dose=dose, condition=condition)
        right = seed_cell(seed_rows, "M5_diverse", right_method, budget, dose=dose, condition=condition)
        for metric in METRICS:
            lv = np.asarray([float(row[metric]) for row in left])
            rv = np.asarray([float(row[metric]) for row in right])
            result.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "left_mean": float(lv.mean()),
                    "right_mean": float(rv.mean()),
                    "relative_difference_percent": 100.0 * float((lv - rv).mean()) / float(rv.mean()),
                    **paired_statistics(lv - rv),
                }
            )
    return result


def rank_selection_rows(seed_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    scenarios = (
        ("positive_control", 0.05, "confounded"),
        ("lambda_zero", 0.0, "confounded"),
        ("independent_latents", 0.05, "independent_latents"),
    )
    result: list[dict[str, object]] = []
    for scenario, dose, condition in scenarios:
        for method in ("MSCSC_correct_source", "MSCSC_source_shuffle"):
            for budget in (16, 64):
                rows = seed_cell(seed_rows, "M5_diverse", method, budget, dose=dose, condition=condition)
                values = np.asarray([float(row["selected_rank"]) for row in rows])
                result.append(
                    {
                        "scenario": scenario,
                        "lambda_reward": dose,
                        "condition": condition,
                        "method": method,
                        "budget": budget,
                        "rank1_selection_rate": float(values.mean()),
                        "rank1_selections_out_of_15": int(round(float(values.sum()) * 5)),
                        "per_seed_rates": json.dumps(values.tolist()),
                    }
                )
    return result


def mechanism_rows(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subspace = read_csv(root / "subspace_metrics.csv")
    result: list[dict[str, object]] = []
    for row in subspace:
        singular = np.asarray(json.loads(row["singular_values"]), dtype=np.float64)
        centered = float(row["correct_centered_norm"])
        residual = float(row["source_reconstruction_error"])
        result.append(
            {
                "setting": row["setting"],
                "lambda_reward": float(row["lambda_reward"]),
                "condition": row["condition"],
                "action": row["action"],
                "population_leading_singular_value": float(singular[0]),
                "population_numerical_nonzero_singular_values": int(np.sum(singular > 1e-10)),
                "population_rank1_explained_variance": float(row["rank1_explained_variance"]),
                "empirical_correct_centered_norm": centered,
                "empirical_shuffle_centered_norm": float(row["shuffle_centered_norm"]),
                "empirical_rank1_residual": residual,
                "empirical_rank1_residual_fraction": residual / centered if centered > 0 else 0.0,
            }
        )

    uncalibrated = read_csv(root / "uncalibrated_rank1_metrics.csv")
    rank1_summary: list[dict[str, object]] = []
    for method in ("MSCSC_correct_source", "MSCSC_source_shuffle"):
        rows = [
            row
            for row in uncalibrated
            if row["setting"] == "M5_diverse"
            and math.isclose(float(row["lambda_reward"]), 0.05, abs_tol=1e-12)
            and row["condition"] == "confounded"
            and row["method"] == method
        ]
        for metric in ("uncalibrated_rank1_contrast_rms", "source_reconstruction_mae"):
            values = np.asarray([float(row[metric]) for row in rows])
            rank1_summary.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd_across_model_seeds": float(values.std(ddof=1)),
                    "per_seed": json.dumps(values.tolist()),
                }
            )
    return result, rank1_summary


def cell_count_rows(root: Path, splits: Mapping[str, Sequence[int]]) -> list[dict[str, object]]:
    train_ids = set(map(int, splits["train"]))
    with np.load(root / "shared_arrays.npz", allow_pickle=False) as shared:
        shared_anchor_ids = np.asarray(shared["anchor_id"], dtype=np.int64)
    ordered_train_ids = np.asarray(sorted(train_ids), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for setting in ("M2_diverse", "M5_diverse"):
        path = root / "scenario_data" / f"{setting}__lambda_0p05__confounded.npz"
        with np.load(path, allow_pickle=False) as values:
            row_index = np.asarray(values["row_index"], dtype=np.int64)
            source = np.asarray(values["source_id"], dtype=np.int64)
        anchor = shared_anchor_ids[row_index // 3]
        action = row_index % 3
        mask = np.fromiter((int(value) in train_ids for value in anchor), dtype=bool, count=len(anchor))
        source_count = 2 if setting == "M2_diverse" else 5
        train_position = np.searchsorted(ordered_train_ids, anchor[mask])
        flat = (source[mask] * len(ordered_train_ids) + train_position) * 3 + action[mask]
        counts = np.bincount(
            flat, minlength=source_count * len(ordered_train_ids) * 3
        ).reshape(source_count, len(ordered_train_ids), 3)
        used = counts.reshape(-1)
        rows.append(
            {
                "setting": setting,
                "source_count": source_count,
                "training_rows": int(mask.sum()),
                "cells": len(used),
                "mean_rows_per_source_anchor_action_cell": float(used.mean()),
                "median_rows_per_cell": float(np.median(used)),
                "min_rows_per_cell": int(used.min()),
                "p10_rows_per_cell": float(np.quantile(used, 0.1)),
                "max_rows_per_cell": int(used.max()),
            }
        )
    return rows


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    primary: Sequence[Mapping[str, object]],
    budget_levels: Sequence[Mapping[str, object]],
    seed_rows: Sequence[Mapping[str, str]],
    mechanism: Sequence[Mapping[str, object]],
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    short_names = ["Correct-Shuffle", "M5-M2", "Redundant-Diverse", "B64-B0"]
    for metric in METRICS:
        selected = [row for row in primary if row["metric"] == metric]
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for index, row in enumerate(selected):
            values = np.asarray(json.loads(str(row["per_seed_differences"])))
            ax.scatter(np.full(len(values), index), values, color=COLORS[0], s=40, zorder=3)
            ax.errorbar(
                index,
                float(row["mean_difference"]),
                yerr=[[float(row["mean_difference"]) - float(row["ci95_low"])],
                      [float(row["ci95_high"]) - float(row["mean_difference"])]],
                fmt="D",
                color=COLORS[1],
                capsize=5,
                markersize=6,
                zorder=4,
            )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(range(len(short_names)), short_names, rotation=18, ha="right")
        ax.set_ylabel(f"Paired difference: {METRIC_LABELS[metric]}\n(negative favors left method/setting)")
        ax.set_title("Phase 8E-Q primary paired contrasts (n=3 model seeds)")
        save_figure(fig, figures, f"figure-0{METRICS.index(metric)+1}-{metric}-paired-contrasts")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for color, metric in zip(COLORS, METRICS):
        selected = sorted((row for row in budget_levels if row["metric"] == metric), key=lambda row: int(row["budget"]))
        base = float(selected[0]["mean"])
        ax.plot(
            [int(row["budget"]) for row in selected],
            [100.0 * float(row["mean"]) / base for row in selected],
            marker="o",
            linewidth=2,
            label=METRIC_LABELS[metric],
            color=color,
        )
    ax.axhline(100, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Interventional calibration budget")
    ax.set_ylabel("Metric relative to B=0 (%)")
    ax.set_title("Calibration improves decisions, not Do-oracle MAE")
    ax.legend(frameon=False)
    save_figure(fig, figures, "figure-04-calibration-budget-relative")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    series = (
        ("Positive: correct source", "confounded", 0.05, "MSCSC_correct_source", COLORS[0], "o", "--", 6),
        ("Positive: shuffled source", "confounded", 0.05, "MSCSC_source_shuffle", COLORS[1], "x", "-", 7),
        ("Lambda=0: correct source", "confounded", 0.0, "MSCSC_correct_source", COLORS[2], "o", "-", 4),
        ("Independent U: correct source", "independent_latents", 0.05, "MSCSC_correct_source", "#CC79A7", "s", "-", 3),
    )
    for label, condition, dose, method, color, marker, linestyle, zorder in series:
        rates = []
        for budget in (16, 64):
            rows = seed_cell(seed_rows, "M5_diverse", method, budget, dose=dose, condition=condition)
            rates.append(100.0 * np.mean([float(row["selected_rank"]) for row in rows]))
        ax.plot(
            (16, 64), rates, marker=marker, linestyle=linestyle, linewidth=2,
            markersize=8, markeredgewidth=2, label=label, color=color, zorder=zorder,
        )
    ax.set_xticks((16, 64))
    ax.set_ylim(-2, 45)
    ax.set_xlabel("Interventional calibration budget")
    ax.set_ylabel("BIC rank-1 selection rate (%)")
    ax.set_title("The true positive-control contrast is rarely selected")
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, figures, "figure-05-rank1-selection-rate")

    selected = [
        row for row in mechanism
        if math.isclose(float(row["lambda_reward"]), 0.05, abs_tol=1e-12)
        and row["condition"] == "confounded"
        and row["action"] in {"minus", "plus"}
    ]
    labels = [f"{row['setting']}\n{row['action']}" for row in selected]
    x = np.arange(len(selected))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width, [float(row["population_leading_singular_value"]) for row in selected], width,
           label="Population rank-1 signal", color=COLORS[0])
    ax.bar(x, [float(row["empirical_rank1_residual"]) for row in selected], width,
           label="Empirical off-rank residual", color=COLORS[1])
    ax.bar(x + width, [float(row["empirical_shuffle_centered_norm"]) for row in selected], width,
           label="Shuffled-source noise norm", color=COLORS[2])
    ax.set_xticks(x, labels)
    ax.set_ylabel("L2 norm across training anchors")
    ax.set_title("Designed signal exists, but M=5 empirical source structure is noisy")
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, figures, "figure-06-population-signal-vs-empirical-noise")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("artifacts/hopper_logger_mixture_drift/phase8e_quick_go_nogo"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/phase8e_quick_go_nogo_strict_analysis"),
    )
    args = parser.parse_args()
    root = args.input_root.resolve()
    output = args.output_root.resolve()
    hard = json.loads((root / "hard_checks.json").read_text(encoding="utf-8"))
    if hard.get("all_passed") is not True or not all(hard.get("checks", {}).values()):
        raise RuntimeError("Phase 8E-Q hard checks did not all pass")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    seed_rows = read_csv(root / "seed_metrics.csv")

    primary = comparison_rows(seed_rows)
    auxiliary = auxiliary_method_rows(seed_rows)
    budget_levels, budget_contrasts = budget_rows(seed_rows)
    lambda_zero, negative = control_rows(seed_rows)
    control_contrasts = control_contrast_rows(seed_rows)
    rank_selection = rank_selection_rows(seed_rows)
    mechanism, rank1 = mechanism_rows(root)
    cell_counts = cell_count_rows(root, splits)

    write_csv(output / "primary-paired-contrasts.csv", primary)
    write_csv(output / "adaptive-vs-pooled-contrasts.csv", auxiliary)
    write_csv(output / "calibration-budget-levels.csv", budget_levels)
    write_csv(output / "calibration-budget-contrasts.csv", budget_contrasts)
    write_csv(output / "lambda-zero-safety.csv", lambda_zero)
    write_csv(output / "independent-latents-negative-control.csv", negative)
    write_csv(output / "control-paired-contrasts.csv", control_contrasts)
    write_csv(output / "rank1-selection-rates.csv", rank_selection)
    write_csv(output / "mechanism-subspace-summary.csv", mechanism)
    write_csv(output / "uncalibrated-rank1-summary.csv", rank1)
    write_csv(output / "training-cell-counts.csv", cell_counts)
    make_figures(primary, budget_levels, seed_rows, mechanism, output / "figures")

    input_files = list(root.rglob("*"))
    input_bytes = sum(path.stat().st_size for path in input_files if path.is_file())
    audit = {
        "stage": "Phase 8E-Q strict read-only analysis",
        "input_root": str(root),
        "input_files": sum(path.is_file() for path in input_files),
        "input_bytes": input_bytes,
        "hard_checks_passed": True,
        "hard_check_count": len(hard["checks"]),
        "scenario_count": manifest["scenario_count"] if "scenario_count" in manifest else 10,
        "model_count": manifest.get("model_count", 90),
        "num_anchors": manifest["num_anchors"],
        "split_counts": {name: len(values) for name, values in splits.items()},
        "independent_unit": "model seed after averaging five nested calibration replicates",
        "independent_seed_count": len(manifest["model_seeds"]),
        "test_anchor_count": len(splits["test"]),
        "confirmatory_inference": False,
        "minimum_attainable_two_sided_exact_p_for_three_nonzero_paired_differences": 0.25,
    }
    write_json(output / "analysis-manifest.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
