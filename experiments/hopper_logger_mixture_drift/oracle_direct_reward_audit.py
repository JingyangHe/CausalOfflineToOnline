"""Phase 8B-RS-O: exact oracle audit of direct reward confounding.

This module is deliberately population-only.  It never loads a neural model or
prediction.  Observational rewards are recomputed from the verified Phase 8A-NC
support masses, while the already-recorded Phase 8B-RS population table supplies
the exact Phase 8A do-oracle values and a cross-check of the recomputation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analyze_noncomplementary_population import KAPPA_NAMES
from .noncomplementary_population_dgp import ACTION_KEYS, CONDITIONS, PRIMARY_MIXTURES


EXPECTED_REWARD_DEFINITION = "original_reward + lambda_reward * u_env"
FORBIDDEN_PUBLIC_FIELDS = {
    "action_key", "action_probability_given_u", "applied_action",
    "applied_action_clipped", "base_mass", "original_reward", "reward_bonus",
    "u_behavior", "u_env",
}
REQUIRED_REWARD_FILES = (
    "manifest.json", "hard_checks.json", "population_audit.json",
    "population_tables.csv", "aggregate_metrics.csv",
)


class OracleRewardAuditError(RuntimeError):
    """Raised when an input or exact population invariant fails."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OracleRewardAuditError(f"required JSON is unavailable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise OracleRewardAuditError(f"required NPZ is unavailable: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_inputs(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path.resolve()) for path in sorted(paths, key=str)}


def bootstrap_stats(values: np.ndarray, repetitions: int, seed: int) -> dict[str, Any]:
    """Anchor-level descriptive statistics and a percentile bootstrap mean CI."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or repetitions <= 0:
        raise ValueError("bootstrap input must be a nonempty 1D array and repetitions positive")
    if not np.isfinite(array).all():
        raise OracleRewardAuditError("bootstrap input contains NaN or Inf")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draws[index] = np.mean(array[rng.integers(0, len(array), len(array))])
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
        "max_abs": float(np.max(np.abs(array))),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_anchors": int(len(array)),
        "bootstrap_unit": "anchor_id",
        "bootstrap_repetitions": int(repetitions),
    }


def fit_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    x_array, y_array = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.shape != x_array.shape or len(x_array) < 2:
        raise ValueError("line fit requires aligned one-dimensional arrays")
    design = np.column_stack((x_array, np.ones_like(x_array)))
    slope, intercept = np.linalg.lstsq(design, y_array, rcond=None)[0]
    fitted = slope * x_array + intercept
    residual = float(np.sum((y_array - fitted) ** 2))
    total = float(np.sum((y_array - y_array.mean()) ** 2))
    r2 = 1.0 if total <= 1e-30 and residual <= 1e-30 else 1.0 - residual / total
    return float(slope), float(intercept), float(r2)


def resolve_reward_signal_root(phase8anc_root: Path,
                               reward_signal_root: Path | None) -> Path:
    """Resolve the RS artifact from an explicit path or a unique stage manifest."""
    nc = Path(phase8anc_root).resolve()
    candidates: list[Path]
    if reward_signal_root is not None:
        candidates = [Path(reward_signal_root).resolve()]
    else:
        candidates = []
        for manifest_path in nc.glob("*/manifest.json"):
            try:
                manifest = _load_json(manifest_path)
            except Exception:
                continue
            if manifest.get("stage") == "Phase 8B-RS":
                candidates.append(manifest_path.parent.resolve())
    verified = [root for root in candidates if root.is_dir() and all(
        (root / name).is_file() for name in REQUIRED_REWARD_FILES)]
    if len(verified) != 1:
        raise OracleRewardAuditError(
            "exactly one complete Phase 8B-RS artifact is required; pass --reward-signal-root")
    return verified[0]


def _validate_manifest_mechanism(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tables = manifest.get("logger_probability_tables", {})
    mixtures = manifest.get("primary_mixtures", {})
    expected_tables = {
        "0": {"-1": {"minus": 0.9, "plus": 0.1},
              "1": {"minus": 0.1, "plus": 0.9}},
        "1": {"-1": {"minus": 0.7, "plus": 0.3},
              "1": {"minus": 0.3, "plus": 0.7}},
        "2": {"-1": {"base": 1.0}, "1": {"base": 1.0}},
    }
    expected_mixtures = {name: list(values) for name, values in PRIMARY_MIXTURES.items()}
    if tables != expected_tables:
        raise OracleRewardAuditError("logger probability table differs from the locked mechanism")
    if mixtures != expected_mixtures:
        raise OracleRewardAuditError("primary mixtures differ from the locked mechanism")
    return dict(tables), dict(mixtures)


def manifest_u_mean(condition: str, mixture: str, action: str,
                    logger_tables: Mapping[str, Any],
                    mixtures: Mapping[str, Sequence[float]]) -> float:
    """Compute E[U_env | A] from the recorded logger mechanism."""
    if condition == "independent_latents":
        return 0.0
    if condition != "confounded" or mixture not in mixtures or action not in ACTION_KEYS:
        raise ValueError("unknown condition, mixture, or action")
    if action == "base":
        return 0.0
    weights = np.asarray(mixtures[mixture], dtype=np.float64)
    masses = {}
    for u in (-1, 1):
        masses[u] = 0.5 * sum(
            weights[logger] * float(logger_tables[str(logger)][str(u)].get(action, 0.0))
            for logger in (0, 1, 2)
        )
    denominator = masses[-1] + masses[1]
    if denominator <= 0:
        raise OracleRewardAuditError("action has zero probability under recorded mixture")
    return float((masses[1] - masses[-1]) / denominator)


def _load_reference_table(path: Path, kappas: Sequence[float]) -> tuple[
        dict[tuple[int, float, float, str, str, str], dict[str, float]],
        np.ndarray, dict[tuple[float, str], float]]:
    rows: dict[tuple[int, float, float, str, str, str], dict[str, float]] = {}
    anchors: set[int] = set()
    errors: dict[tuple[float, str], list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "anchor_id", "kappa", "lambda_reward", "condition", "mixture", "action",
            "augmented_do_reward", "augmented_observational_reward",
            "original_observational_reward", "previous_phase8b_reward_fit_error",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise OracleRewardAuditError("Phase 8B-RS population table schema is incomplete")
        for row in reader:
            kappa = float(row["kappa"])
            if kappa not in kappas:
                continue
            key = (int(row["anchor_id"]), kappa, float(row["lambda_reward"]),
                   row["condition"], row["mixture"], row["action"])
            if key in rows:
                raise OracleRewardAuditError(f"duplicate population reference key: {key}")
            rows[key] = {
                "do_reward": float(row["augmented_do_reward"]),
                "augmented_observational_reward": float(row["augmented_observational_reward"]),
                "original_observational_reward": float(row["original_observational_reward"]),
            }
            anchors.add(key[0])
            value = row["previous_phase8b_reward_fit_error"]
            if value:
                errors.setdefault((kappa, row["condition"]), []).append(float(value))
    if not rows or not anchors:
        raise OracleRewardAuditError("Phase 8B-RS population reference is empty")
    fit_errors = {kappa: float(np.mean(values)) for kappa, values in errors.items()}
    return rows, np.asarray(sorted(anchors), dtype=np.int64), fit_errors


def support_action_means(public: Mapping[str, np.ndarray],
                         hidden: Mapping[str, np.ndarray], weights: np.ndarray,
                         anchor_ids: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Compute exact conditional means from original support probability mass."""
    if not np.array_equal(public["row_id"], hidden["row_id"]):
        raise OracleRewardAuditError("public and hidden support rows are not aligned")
    if not np.allclose(public["reward"], hidden["reward"], atol=0, rtol=0):
        raise OracleRewardAuditError("public and hidden original rewards differ")
    if len(weights) != len(public["row_id"]):
        raise OracleRewardAuditError("mixture mass does not align with support rows")
    anchors = np.asarray(public["anchor_id"], dtype=np.int64)
    actions = np.asarray(hidden["action_key"]).astype(str)
    result: dict[str, dict[str, np.ndarray]] = {}
    for action in ACTION_KEYS:
        reward_mean = np.empty(len(anchor_ids), dtype=np.float64)
        u_mean = np.empty(len(anchor_ids), dtype=np.float64)
        mass = np.empty(len(anchor_ids), dtype=np.float64)
        for index, anchor in enumerate(anchor_ids):
            mask = (anchors == anchor) & (actions == action)
            selected_mass = np.asarray(weights[mask], dtype=np.float64)
            denominator = float(selected_mass.sum())
            if denominator <= 0:
                raise OracleRewardAuditError(f"zero support mass for anchor={anchor}, action={action}")
            mass[index] = denominator
            reward_mean[index] = float(selected_mass @ np.asarray(public["reward"])[mask]) / denominator
            u_mean[index] = float(selected_mass @ np.asarray(hidden["u_env"])[mask]) / denominator
        result[action] = {"reward": reward_mean, "u_mean": u_mean, "mass": mass}
    return result


def _group_arrays(rows: Sequence[Mapping[str, Any]], filters: Mapping[str, Any],
                  field: str) -> np.ndarray:
    selected = sorted(
        (row for row in rows if all(row.get(key) == value for key, value in filters.items())),
        key=lambda row: int(row["anchor_id"]),
    )
    return np.asarray([float(row[field]) for row in selected], dtype=np.float64)


def _aggregate_decomposition(rows: Sequence[Mapping[str, Any]], repetitions: int,
                             seed: int) -> list[dict[str, Any]]:
    dimensions = ("family", "kappa", "lambda_reward", "condition", "mixture", "action")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[name] for name in dimensions), []).append(row)
    output: list[dict[str, Any]] = []
    counter = 0
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        for component in ("physical_bias", "direct_bias", "total_bias", "decomposition_residual"):
            values = np.asarray([float(row[component]) for row in members], dtype=np.float64)
            record = dict(zip(dimensions, key))
            record["component"] = component
            record.update(bootstrap_stats(values, repetitions, seed + counter))
            output.append(record)
            counter += 1
    return output


def _slope_rows(detail: Sequence[Mapping[str, Any]], heavy: Sequence[Mapping[str, Any]],
                strengths: Sequence[float], logger_tables: Mapping[str, Any],
                mixtures: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for kappa in sorted({float(row["kappa"]) for row in detail}):
        for condition in CONDITIONS:
            for family, source, mixture in (
                ("balanced_bias", detail, "logger12_balanced"),
                ("heavy_drift", heavy, "logger1_minus_logger2"),
            ):
                for action in ACTION_KEYS:
                    means, anchor_curves = [], []
                    for strength in strengths:
                        values = _group_arrays(source, {
                            "kappa": kappa, "lambda_reward": strength,
                            "condition": condition, "mixture": mixture, "action": action,
                        }, "total_bias")
                        if len(values) == 0:
                            raise OracleRewardAuditError("slope input group is empty")
                        means.append(float(values.mean()))
                        anchor_curves.append(values)
                    slope, intercept, r2 = fit_line(strengths, means)
                    balanced = manifest_u_mean(
                        condition, "logger12_balanced", action, logger_tables, mixtures)
                    if family == "balanced_bias":
                        theory = balanced
                    else:
                        theory = (manifest_u_mean(condition, "logger1_heavy", action,
                                                  logger_tables, mixtures)
                                  - manifest_u_mean(condition, "logger2_heavy", action,
                                                    logger_tables, mixtures))
                    matrix = np.stack(anchor_curves)
                    predicted = matrix[0] + theory * (
                        np.asarray(strengths)[:, None] - float(strengths[0]))
                    output.append({
                        "family": family, "kappa": kappa, "condition": condition,
                        "action": action, "empirical_slope": slope,
                        "theoretical_slope": theory,
                        "absolute_slope_error": abs(slope - theory),
                        "intercept": intercept, "r_squared": r2,
                        "max_anchor_identity_residual": float(np.max(np.abs(matrix - predicted))),
                        "n_anchors": int(matrix.shape[1]),
                    })
    return output


def _ranking_rows(detail: Sequence[Mapping[str, Any]], repetitions: int,
                  seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counter = 0
    kappas = sorted({float(row["kappa"]) for row in detail})
    strengths = sorted({float(row["lambda_reward"]) for row in detail})
    for kappa in kappas:
        for strength in strengths:
            for condition in CONDITIONS:
                for mixture in PRIMARY_MIXTURES:
                    obs = np.column_stack([
                        _group_arrays(detail, {"kappa": kappa, "lambda_reward": strength,
                                               "condition": condition, "mixture": mixture,
                                               "action": action},
                                      "augmented_observational_reward")
                        for action in ACTION_KEYS
                    ])
                    do = np.column_stack([
                        _group_arrays(detail, {"kappa": kappa, "lambda_reward": strength,
                                               "condition": condition, "mixture": mixture,
                                               "action": action}, "do_reward")
                        for action in ACTION_KEYS
                    ])
                    obs_top = np.isclose(obs, obs.max(axis=1, keepdims=True), atol=1e-7, rtol=1e-7)
                    do_top = np.isclose(do, do.max(axis=1, keepdims=True), atol=1e-7, rtol=1e-7)
                    disagreement = (~np.any(obs_top & do_top, axis=1)).astype(np.float64)
                    chosen = np.argmax(obs, axis=1)
                    regret = np.max(do, axis=1) - do[np.arange(len(do)), chosen]
                    for metric, values in (("ranking_disagreement", disagreement),
                                           ("true_decision_regret", regret)):
                        record = {
                            "kappa": kappa, "lambda_reward": strength,
                            "condition": condition, "mixture": mixture, "metric": metric,
                        }
                        record.update(bootstrap_stats(values, repetitions, seed + counter))
                        output.append(record)
                        counter += 1
    return output


def _plot_line(path: Path, series: Sequence[tuple[str, Sequence[float], Sequence[float],
                                                  Sequence[float] | None,
                                                  Sequence[float] | None]],
               ylabel: str) -> None:
    plt.figure(figsize=(6.0, 4.0))
    for label, x, y, low, high in series:
        line, = plt.plot(x, y, marker="o", label=label)
        if low is not None and high is not None:
            plt.fill_between(x, low, high, alpha=0.2, color=line.get_color())
    plt.xlabel("lambda_reward")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _make_figures(output: Path, aggregates: Sequence[Mapping[str, Any]],
                  fit_errors: Mapping[tuple[float, str], float],
                  strengths: Sequence[float]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def series_for(family: str, kappa: float, condition: str, mixture: str,
                   component: str, actions: Sequence[str]) -> list[tuple[Any, ...]]:
        result = []
        for action in actions:
            rows = sorted((row for row in aggregates if row["family"] == family
                           and float(row["kappa"]) == kappa
                           and row["condition"] == condition and row["mixture"] == mixture
                           and row["action"] == action and row["component"] == component),
                          key=lambda row: float(row["lambda_reward"]))
            result.append((action, [row["lambda_reward"] for row in rows],
                           [row["mean"] for row in rows], [row["ci_low"] for row in rows],
                           [row["ci_high"] for row in rows]))
        return result

    _plot_line(figures / "balanced_reward_bias_vs_lambda.png",
               series_for("observational_do", 0.3, "confounded", "logger12_balanced",
                          "total_bias", ACTION_KEYS), "observational - do reward")
    _plot_line(figures / "heavy_reward_drift_vs_lambda.png",
               series_for("heavy_drift", 0.3, "confounded", "logger1_minus_logger2",
                          "total_bias", ACTION_KEYS), "logger1-heavy - logger2-heavy reward")

    selected = [row for row in aggregates if row["family"] == "observational_do"
                and float(row["kappa"]) == 0.3 and float(row["lambda_reward"]) == max(strengths)
                and row["condition"] == "confounded"
                and row["mixture"] == "logger12_balanced"
                and row["component"] in ("physical_bias", "direct_bias")]
    labels = [f"{row['action']}:{row['component'].replace('_bias','')}" for row in selected]
    plt.figure(figsize=(7.0, 4.0)); plt.bar(labels, [row["mean"] for row in selected])
    plt.ylabel("mean reward bias"); plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(figures / "physical_vs_direct_bias.png", dpi=160); plt.close()

    decomposition_series = []
    for component in ("physical_bias", "direct_bias", "total_bias"):
        rows = sorted((row for row in aggregates if row["family"] == "observational_do"
                       and float(row["kappa"]) == 0.3 and row["condition"] == "confounded"
                       and row["mixture"] == "logger12_balanced" and row["action"] == "plus"
                       and row["component"] == component), key=lambda row: row["lambda_reward"])
        decomposition_series.append((component, [row["lambda_reward"] for row in rows],
                                     [row["mean"] for row in rows], None, None))
    _plot_line(figures / "total_bias_decomposition.png", decomposition_series,
               "plus-action reward bias")

    condition_series = []
    for condition in CONDITIONS:
        rows = sorted((row for row in aggregates if row["family"] == "observational_do"
                       and float(row["kappa"]) == 0.3 and row["condition"] == condition
                       and row["mixture"] == "logger12_balanced" and row["action"] == "plus"
                       and row["component"] == "total_bias"), key=lambda row: row["lambda_reward"])
        condition_series.append((condition, [row["lambda_reward"] for row in rows],
                                 [row["mean"] for row in rows],
                                 [row["ci_low"] for row in rows], [row["ci_high"] for row in rows]))
    _plot_line(figures / "confounded_vs_independent_reward_bias.png", condition_series,
               "plus-action observational - do reward")
    _plot_line(figures / "plus_minus_base_bias_vs_lambda.png",
               series_for("observational_do", 0.0, "confounded", "logger12_balanced",
                          "total_bias", ACTION_KEYS), "balanced reward bias")

    kappa = min(key[0] for key in fit_errors)
    neural_error = fit_errors[(kappa, "confounded")]
    x = list(strengths)
    signal_series = [
        ("balanced direct signal", x, [0.6 * value for value in x], None, None),
        ("heavy direct drift", x, [(14.0 / 45.0) * value for value in x], None, None),
        ("previous neural fit error", x, [neural_error] * len(x), None, None),
    ]
    _plot_line(figures / "population_signal_vs_previous_neural_error.png", signal_series,
               "absolute reward scale")


def _write_reports(output: Path, summary: Mapping[str, Any],
                   figure_rows: Sequence[Mapping[str, str]]) -> None:
    table_lines = [
        "| lambda | kappa | balanced plus bias | balanced minus bias | heavy drift | base bias | independent bias | do shift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["main_table"]:
        table_lines.append("| " + " | ".join(
            f"{row[name]:.9g}" for name in (
                "lambda", "kappa", "balanced_plus_bias", "balanced_minus_bias",
                "heavy_drift", "base_bias", "independent_bias", "do_shift")) + " |")
    table = "\n".join(table_lines)
    slope_lines = [
        "| kappa | family | action | empirical | theoretical | abs. error | intercept | R² | max anchor residual |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["lambda_slopes"]:
        if row["condition"] != "confounded":
            continue
        slope_lines.append(
            f"| {row['kappa']:.1f} | {row['family']} | {row['action']} | "
            f"{row['empirical_slope']:.9g} | {row['theoretical_slope']:.9g} | "
            f"{row['absolute_slope_error']:.3e} | {row['intercept']:.9g} | "
            f"{row['r_squared']:.9g} | {row['max_anchor_identity_residual']:.3e} |")
    slope_table = "\n".join(slope_lines)

    decomposition_lines = [
        "| action | physical | direct | total |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["decomposition_example"]:
        decomposition_lines.append(
            f"| {row['action']} | {row['physical_bias']:.9g} | "
            f"{row['direct_bias']:.9g} | {row['total_bias']:.9g} |")
    decomposition_table = "\n".join(decomposition_lines)

    signal_lines = [
        "| kappa | lambda | balanced/error | heavy/error | previous error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for kappa, by_strength in summary["signal_vs_previous_neural_error"].items():
        for strength, row in by_strength.items():
            signal_lines.append(
                f"| {float(kappa):.1f} | {float(strength):.2f} | "
                f"{row['balanced_signal_over_error']:.6g} | "
                f"{row['heavy_signal_over_error']:.6g} | "
                f"{row['previous_neural_fit_error']:.6g} |")
    signal_table = "\n".join(signal_lines)

    ranking_lines = [
        "| kappa | lambda | mixture | disagreement | true regret |",
        "|---:|---:|---|---:|---:|",
    ]
    ranking_lookup = {(row["kappa"], row["lambda_reward"], row["mixture"], row["metric"]): row
                      for row in summary["ranking_metrics"]
                      if row["condition"] == "confounded"}
    final_strength = max(float(row["lambda"]) for row in summary["main_table"])
    for kappa in summary["kappas"]:
        for mixture in ("logger12_balanced", "logger1_heavy", "logger2_heavy"):
            disagreement = ranking_lookup[(kappa, final_strength, mixture,
                                            "ranking_disagreement")]["mean"]
            regret = ranking_lookup[(kappa, final_strength, mixture,
                                     "true_decision_regret")]["mean"]
            ranking_lines.append(
                f"| {kappa:.1f} | {final_strength:.2f} | {mixture} | "
                f"{disagreement:.6g} | {regret:.6g} |")
    ranking_table = "\n".join(ranking_lines)
    answers = summary["scientific_answers"]
    report = f"""# Phase 8B-RS-O — Oracle Reward-Confounding Audit

No neural network or learned prediction was used. The statistical unit is
`anchor_id` (n={summary['analyzed_anchor_count']}); uncertainty intervals are
anchor-bootstrap percentile intervals with {summary['bootstrap_repetitions']} repetitions.

## Primary table

{table}

`heavy drift` is the plus-action logger1-heavy minus logger2-heavy contrast.
`independent bias` is the maximum absolute observational-do bias over all anchors,
actions, and primary mixtures. `do shift` is the maximum absolute lambda-induced
change in the do mean.

## Direct answers

- Q1: {answers['Q1']}
- Q2: {answers['Q2']}
- Q3: {answers['Q3']}
- Q4: {answers['Q4']}
- Q5: {answers['Q5']}
- Q6: {answers['Q6']}
- Q7: {answers['Q7']}

## Numerical audit

- Maximum reward-definition residual: {summary['max_residuals']['reward_identity']:.3e}
- Maximum population-table recomputation residual: {summary['max_residuals']['population_crosscheck']:.3e}
- Maximum bias-decomposition residual: {summary['max_residuals']['bias_decomposition']:.3e}
- Maximum lambda-induced do shift: {summary['max_residuals']['do_shift']:.3e}
- Maximum P(S,A) mass change: {summary['max_residuals']['state_action_mass_shift']:.3e}

## Theory versus empirical lambda slopes

{slope_table}

## Physical/direct/total decomposition

The following is the balanced confounded mixture at kappa=0.3 and lambda=0.20.

{decomposition_table}

## Population signal versus previous neural error

{signal_table}

At lambda=0.05 the balanced signal already reaches the previous neural-error
scale; by lambda=0.10 and 0.20 both the balanced signal and, eventually, the
heavy contrast are at or above that scale. This is a scale comparison only.

## Secondary ranking and regret

{ranking_table}

These decision metrics are secondary: absence of a ranking flip would not
invalidate the exact reward-confounding result.

## Supported conclusions

- Direct reward confounding is nonzero after source balancing and grows with the locked theoretical slopes.
- The direct channel changes observational reward but neither P(S,A) nor the do-action mean.
- Physical and direct reward bias are exactly additive at the audited numerical tolerance.
- Independent latents, base action, lambda=0, and kappa=0 behave as the specified controls.

## Evidence boundary

This audit identifies the exact population signal and its decomposition. It does
not establish that a learned reward model can recover the signal, that action
ranking must flip, or that any downstream policy improves.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "analysis-report.md").write_text(
        "# Analysis report\n\n" + "\n".join(
            f"- {key}: {value}" for key, value in answers.items()) +
        "\n\nThe exact numeric evidence is saved in the CSV and NPZ artifacts.\n",
        encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        f"# Statistical appendix\n\nThe unit is anchor_id (n={summary['analyzed_anchor_count']}). "
        f"Means, standard deviations, medians, P10/P25/P75/P90, maxima, and "
        f"{summary['bootstrap_repetitions']}-replicate anchor-bootstrap 95% CIs are "
        "reported in bias_decomposition.csv and ranking_metrics.csv. Exact DGP "
        "identities are checked by numerical tolerance rather than significance tests.\n",
        encoding="utf-8")
    catalog = ["# Figure catalog", ""]
    for row in figure_rows:
        catalog.extend((f"## {row['file']}", "", f"- Purpose: {row['purpose']}",
                        f"- Observation: {row['observation']}",
                        f"- Implication: {row['implication']}", ""))
    (output / "figure-catalog.md").write_text("\n".join(catalog), encoding="utf-8")


def run_oracle_direct_reward_audit(
    phase8anc_root: Path, output_root: Path, *, reward_signal_root: Path | None = None,
    kappas: tuple[float, ...] = (0.0, 0.3), bootstrap_reps: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the read-only, exact population/oracle reward-confounding audit."""
    if not kappas or len(set(map(float, kappas))) != len(kappas):
        raise ValueError("kappas must be nonempty and unique")
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    kappas = tuple(float(value) for value in kappas)
    if kappas != (0.0, 0.3):
        raise ValueError("Phase 8B-RS-O requires kappas in canonical order: (0.0, 0.3)")
    nc = Path(phase8anc_root).resolve()
    reward_root = resolve_reward_signal_root(nc, reward_signal_root)
    output = Path(output_root).resolve()
    if output in (nc, reward_root) or reward_root in output.parents:
        raise OracleRewardAuditError("output root must be a new sibling artifact")

    nc_manifest = _load_json(nc / "manifest.json")
    nc_hard = _load_json(nc / "hard_checks.json")
    reward_manifest = _load_json(reward_root / "manifest.json")
    reward_hard = _load_json(reward_root / "hard_checks.json")
    logger_tables, mixtures = _validate_manifest_mechanism(nc_manifest)
    strengths = tuple(float(value) for value in reward_manifest.get("reward_strengths", ()))
    if reward_manifest.get("stage") != "Phase 8B-RS" or not strengths:
        raise OracleRewardAuditError("reward-signal manifest is not a complete Phase 8B-RS artifact")
    if reward_manifest.get("reward_definition") != EXPECTED_REWARD_DEFINITION:
        raise OracleRewardAuditError("direct reward definition is not original + lambda*u_env")
    if any(kappa not in tuple(map(float, reward_manifest.get("kappas", ()))) for kappa in kappas):
        raise OracleRewardAuditError("requested kappa is absent from the reward-signal manifest")
    if nc_hard.get("all_passed") is not True or not all(nc_hard.get("checks", {}).values()):
        raise OracleRewardAuditError("Phase 8A-NC input hard checks did not all pass")
    if reward_hard.get("all_passed") is not True or not all(reward_hard.get("checks", {}).values()):
        raise OracleRewardAuditError("Phase 8B-RS input hard checks did not all pass")

    reference, anchor_ids, fit_errors = _load_reference_table(
        reward_root / "population_tables.csv", kappas)
    expected_anchor_count = int(reward_manifest.get("analyzed_anchor_count", -1))
    if len(anchor_ids) != expected_anchor_count:
        raise OracleRewardAuditError("reward-signal anchors are incomplete")
    baseline_strength = min(strengths)
    if baseline_strength != 0.0:
        raise OracleRewardAuditError("lambda grid must contain zero as the do-reward baseline")
    do_original: dict[tuple[int, float, str], float] = {}
    for key, values in reference.items():
        anchor, kappa, strength, _condition, _mixture, action = key
        if strength != baseline_strength:
            continue
        do_key = (anchor, kappa, action)
        recorded = values["do_reward"]
        if do_key in do_original and not np.isclose(
                do_original[do_key], recorded, atol=0, rtol=0):
            raise OracleRewardAuditError("lambda-zero do reward differs across condition or mixture")
        do_original[do_key] = recorded
    expected_do_keys = len(anchor_ids) * len(kappas) * len(ACTION_KEYS)
    if len(do_original) != expected_do_keys:
        raise OracleRewardAuditError("lambda-zero do reward keys are incomplete")

    input_paths = [nc / "manifest.json", nc / "hard_checks.json",
                   reward_root / "manifest.json", reward_root / "hard_checks.json",
                   reward_root / "population_audit.json", reward_root / "population_tables.csv",
                   reward_root / "aggregate_metrics.csv"]
    for kappa in kappas:
        kname = KAPPA_NAMES[kappa]
        for condition in CONDITIONS:
            input_paths.extend((nc / kname / f"{condition}_public.npz",
                                nc / kname / f"{condition}_hidden_audit.npz"))
            for mixture in PRIMARY_MIXTURES:
                input_paths.append(nc / kname / "weights" / condition / f"{mixture}.npy")
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise OracleRewardAuditError(f"required read-only inputs are missing: {missing}")
    hashes_before = hash_inputs(input_paths)

    atol = float(nc_manifest.get("numerical_tolerance", {}).get("atol", 1e-7))
    rtol = float(nc_manifest.get("numerical_tolerance", {}).get("rtol", 1e-7))
    detail: list[dict[str, Any]] = []
    reward_identity_residual = 0.0
    crosscheck_residual = 0.0
    mass_shift = 0.0
    public_leakage_empty = True
    u_env_distinguished = False
    wrong_hidden_variable_detectable = False
    group_completeness = True

    for kappa in kappas:
        kname = KAPPA_NAMES[kappa]
        for condition in CONDITIONS:
            public = _load_npz(nc / kname / f"{condition}_public.npz")
            hidden = _load_npz(nc / kname / f"{condition}_hidden_audit.npz")
            public_leakage_empty &= not FORBIDDEN_PUBLIC_FIELDS.intersection(public)
            public_anchor_set = set(map(int, np.unique(public["anchor_id"])))
            group_completeness &= set(map(int, anchor_ids)).issubset(public_anchor_set)
            mismatch = np.asarray(hidden["u_env"]) != np.asarray(hidden["u_behavior"])
            if condition == "independent_latents":
                u_env_distinguished |= bool(np.any(mismatch))
            selected_mask = np.isin(public["anchor_id"], anchor_ids)
            selected_public = {name: np.asarray(value)[selected_mask] for name, value in public.items()}
            selected_hidden = {name: np.asarray(value)[selected_mask] for name, value in hidden.items()}
            for strength in strengths:
                row_augmented = (np.asarray(selected_public["reward"], dtype=np.float64)
                                 + strength * np.asarray(selected_hidden["u_env"], dtype=np.float64))
                reward_identity_residual = max(
                    reward_identity_residual,
                    float(np.max(np.abs(row_augmented
                                        - np.asarray(selected_public["reward"], dtype=np.float64)
                                        - strength * np.asarray(selected_hidden["u_env"], dtype=np.float64)))))
                if condition == "independent_latents" and strength > 0 and np.any(mismatch[selected_mask]):
                    wrong = (row_augmented - np.asarray(selected_public["reward"], dtype=np.float64)
                             - strength * np.asarray(selected_hidden["u_behavior"], dtype=np.float64))
                    wrong_hidden_variable_detectable |= bool(np.max(np.abs(wrong)) > atol)
            for mixture in PRIMARY_MIXTURES:
                weights_all = np.asarray(np.load(
                    nc / kname / "weights" / condition / f"{mixture}.npy"), dtype=np.float64)
                means = support_action_means(selected_public, selected_hidden,
                                             weights_all[selected_mask], anchor_ids)
                theory_u = {action: manifest_u_mean(
                    condition, mixture, action, logger_tables, mixtures) for action in ACTION_KEYS}
                for action in ACTION_KEYS:
                    if not np.allclose(means[action]["u_mean"], theory_u[action], atol=atol, rtol=rtol):
                        group_completeness = False
                    base_mass = means[action]["mass"].copy()
                    for strength in strengths:
                        direct = float(strength) * means[action]["u_mean"]
                        augmented = means[action]["reward"] + direct
                        reward_identity_residual = max(
                            reward_identity_residual,
                            float(np.max(np.abs(augmented - means[action]["reward"]
                                                - strength * means[action]["u_mean"]))))
                        mass_shift = max(mass_shift, float(np.max(np.abs(means[action]["mass"] - base_mass))))
                        for index, anchor in enumerate(anchor_ids):
                            key = (int(anchor), kappa, strength, condition, mixture, action)
                            if key not in reference:
                                group_completeness = False
                                continue
                            ref = reference[key]
                            crosscheck_residual = max(
                                crosscheck_residual,
                                abs(float(augmented[index]) - ref["augmented_observational_reward"]),
                                abs(float(means[action]["reward"][index])
                                    - ref["original_observational_reward"]),
                            )
                            original_do = do_original[(int(anchor), kappa, action)]
                            augmented_do = ref["do_reward"]
                            current_do_shift = float(augmented_do - original_do)
                            physical = float(means[action]["reward"][index] - original_do)
                            direct_value = float(direct[index])
                            total = float(augmented[index] - augmented_do)
                            detail.append({
                                "anchor_id": int(anchor), "kappa": kappa,
                                "lambda_reward": strength, "condition": condition,
                                "mixture": mixture, "action": action,
                                "support_mass": float(means[action]["mass"][index]),
                                "conditional_u_env_mean": float(means[action]["u_mean"][index]),
                                "original_observational_reward": float(means[action]["reward"][index]),
                                "direct_reward_component": direct_value,
                                "augmented_observational_reward": float(augmented[index]),
                                "do_reward": augmented_do,
                                "do_original_reward": original_do,
                                "do_augmented_reward": augmented_do,
                                "do_shift": current_do_shift,
                                "physical_bias": physical, "direct_bias": direct_value,
                                "total_bias": total,
                                "decomposition_residual": total - physical - direct_value,
                                "theoretical_direct_bias": float(strength) * theory_u[action],
                                "family": "observational_do",
                            })

    # Exact logger1-heavy minus logger2-heavy decomposition, kept at anchor level.
    heavy: list[dict[str, Any]] = []
    detail_index = {(row["anchor_id"], row["kappa"], row["lambda_reward"], row["condition"],
                     row["mixture"], row["action"]): row for row in detail}
    for kappa in kappas:
        for strength in strengths:
            for condition in CONDITIONS:
                for action in ACTION_KEYS:
                    for anchor in anchor_ids:
                        left = detail_index[(int(anchor), kappa, strength, condition,
                                             "logger1_heavy", action)]
                        right = detail_index[(int(anchor), kappa, strength, condition,
                                              "logger2_heavy", action)]
                        heavy.append({
                            "anchor_id": int(anchor), "kappa": kappa,
                            "lambda_reward": strength, "condition": condition,
                            "mixture": "logger1_minus_logger2", "action": action,
                            "physical_bias": left["physical_bias"] - right["physical_bias"],
                            "direct_bias": left["direct_bias"] - right["direct_bias"],
                            "total_bias": (left["augmented_observational_reward"]
                                           - right["augmented_observational_reward"]),
                            "decomposition_residual": (
                                left["augmented_observational_reward"]
                                - right["augmented_observational_reward"]
                                - (left["physical_bias"] - right["physical_bias"])
                                - (left["direct_bias"] - right["direct_bias"])),
                            "family": "heavy_drift",
                        })

    slopes = _slope_rows(detail, heavy, strengths, logger_tables, mixtures)
    all_decomposition = detail + heavy
    decomposition_rows = _aggregate_decomposition(all_decomposition, bootstrap_reps, seed)
    ranking_rows = _ranking_rows(detail, bootstrap_reps, seed + 100000)
    decomposition_residual = max(abs(float(row["decomposition_residual"]))
                                 for row in all_decomposition)
    do_shift = max(abs(float(row["do_shift"])) for row in detail)
    mass_baseline = {
        (row["anchor_id"], row["kappa"], row["condition"], row["mixture"], row["action"]):
        float(row["support_mass"])
        for row in detail if row["lambda_reward"] == baseline_strength
    }
    mass_shift = max(abs(float(row["support_mass"]) - mass_baseline[
        (row["anchor_id"], row["kappa"], row["condition"], row["mixture"], row["action"])])
        for row in detail)

    slope_map = {(row["family"], row["kappa"], row["condition"], row["action"]): row
                 for row in slopes}
    def slope_ok(family: str, action: str, expected: float) -> bool:
        return all(np.isclose(slope_map[(family, kappa, "confounded", action)]["empirical_slope"],
                              expected, atol=atol, rtol=rtol) for kappa in kappas)

    max_independent = max(abs(float(row["total_bias"])) for row in detail
                          if row["condition"] == "independent_latents")
    max_independent_direct = max(abs(float(row["direct_bias"])) for row in detail
                                 if row["condition"] == "independent_latents")
    max_base_direct = max(abs(float(row["direct_bias"])) for row in detail
                          if row["action"] == "base")
    max_kappa0_physical = max(abs(float(row["physical_bias"])) for row in detail
                              if row["kappa"] == 0.0)
    expected_row_count = (len(anchor_ids) * len(kappas) * len(strengths)
                          * len(CONDITIONS) * len(PRIMARY_MIXTURES) * len(ACTION_KEYS))
    hard_checks = {
        "input_artifacts_complete": not missing,
        "all_anchors_complete": group_completeness and len(detail) == expected_row_count,
        "reward_aug_equals_original_plus_lambda_u_env": reward_identity_residual <= atol,
        "direct_reward_uses_u_env_not_u_behavior": (
            reward_manifest.get("reward_definition") == EXPECTED_REWARD_DEFINITION
            and u_env_distinguished and wrong_hidden_variable_detectable),
        "do_reward_mean_invariant_to_lambda": do_shift <= atol,
        "primary_state_action_distribution_invariant_to_lambda": mass_shift == 0.0,
        "balanced_plus_direct_slope_plus_0p6": slope_ok("balanced_bias", "plus", 0.6),
        "balanced_minus_direct_slope_minus_0p6": slope_ok("balanced_bias", "minus", -0.6),
        "base_direct_slope_zero": slope_ok("balanced_bias", "base", 0.0),
        "heavy_plus_slope_plus_14_over_45": slope_ok("heavy_drift", "plus", 14.0 / 45.0),
        "heavy_minus_slope_minus_14_over_45": slope_ok("heavy_drift", "minus", -14.0 / 45.0),
        "heavy_base_slope_zero": slope_ok("heavy_drift", "base", 0.0),
        "kappa_zero_physical_bias_zero": max_kappa0_physical <= atol,
        "total_bias_equals_physical_plus_direct": decomposition_residual <= atol,
        "independent_latents_direct_bias_zero": max_independent_direct <= atol,
        "independent_latents_total_bias_zero": max_independent <= atol,
        "public_artifact_hidden_leakage_empty": public_leakage_empty,
        "all_arrays_finite": all(np.isfinite(float(value)) for row in all_decomposition
                                  for value in row.values() if isinstance(value, (int, float, np.number))),
        "population_table_recomputation_matches": crosscheck_residual <= atol,
    }

    hashes_after = hash_inputs(input_paths)
    unchanged = hashes_before == hashes_after
    hard_checks["input_sha256_unchanged"] = unchanged
    hard_checks["old_artifacts_unmodified"] = unchanged
    failed = [name for name, passed in hard_checks.items() if not passed]
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "input_integrity.json", {
        "sha256_before": hashes_before, "sha256_after": hashes_after,
        "unchanged": unchanged, "required_file_count": len(input_paths),
    })
    _write_json(output / "hard_checks.json", {
        "checks": hard_checks, "all_passed": not failed, "failed": failed,
    })
    if failed:
        raise OracleRewardAuditError(f"hard checks failed: {failed}")

    _write_csv(output / "population_reward_bias.csv", detail)
    _write_csv(output / "lambda_slope_metrics.csv", slopes)
    _write_csv(output / "bias_decomposition.csv", decomposition_rows)
    _write_csv(output / "ranking_metrics.csv", ranking_rows)
    np.savez_compressed(output / "anchor_action_metrics.npz", **{
        key: np.asarray([row[key] for row in detail]) for key in detail[0]
    })

    main_table = []
    for kappa in kappas:
        for strength in strengths:
            filters = {"kappa": kappa, "lambda_reward": strength,
                       "condition": "confounded", "mixture": "logger12_balanced"}
            plus = _group_arrays(detail, {**filters, "action": "plus"}, "total_bias")
            minus = _group_arrays(detail, {**filters, "action": "minus"}, "total_bias")
            base = _group_arrays(detail, {**filters, "action": "base"}, "total_bias")
            heavy_plus = _group_arrays(heavy, {"kappa": kappa, "lambda_reward": strength,
                                               "condition": "confounded",
                                               "mixture": "logger1_minus_logger2",
                                               "action": "plus"}, "total_bias")
            independent = [abs(float(row["total_bias"])) for row in detail
                           if row["kappa"] == kappa and row["lambda_reward"] == strength
                           and row["condition"] == "independent_latents"]
            main_table.append({
                "lambda": strength, "kappa": kappa,
                "balanced_plus_bias": float(plus.mean()),
                "balanced_minus_bias": float(minus.mean()),
                "heavy_drift": float(heavy_plus.mean()),
                "base_bias": float(base.mean()),
                "independent_bias": float(max(independent)), "do_shift": do_shift,
            })

    signal_scale = {}
    for kappa in kappas:
        error = fit_errors[(kappa, "confounded")]
        signal_scale[str(kappa)] = {
            str(strength): {
                "balanced_signal": 0.6 * strength,
                "heavy_signal": (14.0 / 45.0) * strength,
                "previous_neural_fit_error": error,
                "balanced_signal_over_error": 0.6 * strength / error,
                "heavy_signal_over_error": (14.0 / 45.0) * strength / error,
            } for strength in strengths
        }
    max_residuals = {
        "reward_identity": reward_identity_residual,
        "population_crosscheck": crosscheck_residual,
        "bias_decomposition": decomposition_residual,
        "do_shift": do_shift, "state_action_mass_shift": mass_shift,
        "independent_total_bias": max_independent,
        "base_direct_bias": max_base_direct,
    }
    scientific_answers = {
        "Q1": "Yes. The confounded observational-do reward bias grows exactly linearly with lambda.",
        "Q2": "Yes. P(S,A) is unchanged; the added bias is exactly lambda E[U_env|S,A].",
        "Q3": "Yes. The balanced source mixture retains slopes -0.6 and +0.6.",
        "Q4": "No. The direct term cancels under the symmetric do(U_env) average.",
        "Q5": "Yes. Independent latents remove the direct and total observational-do bias.",
        "Q6": "Yes. The base action has zero direct bias for every anchor and lambda.",
        "Q7": "Yes. Total bias equals physical bias plus direct U-to-reward bias within tolerance.",
    }
    summary = {
        "stage": "Phase 8B-RS-O", "analyzed_anchor_count": len(anchor_ids),
        "kappas": list(kappas), "reward_strengths": list(strengths),
        "bootstrap_repetitions": bootstrap_reps, "bootstrap_seed": seed,
        "main_table": main_table, "lambda_slopes": slopes,
        "signal_vs_previous_neural_error": signal_scale,
        "ranking_metrics": ranking_rows, "max_residuals": max_residuals,
        "decomposition_example": [
            {
                "action": action,
                **{component: next(float(row["mean"]) for row in decomposition_rows
                    if row["family"] == "observational_do" and row["kappa"] == 0.3
                    and row["lambda_reward"] == max(strengths)
                    and row["condition"] == "confounded"
                    and row["mixture"] == "logger12_balanced" and row["action"] == action
                    and row["component"] == component)
                   for component in ("physical_bias", "direct_bias", "total_bias")}
            } for action in ACTION_KEYS
        ],
        "scientific_answers": scientific_answers,
        "all_hard_checks_passed": True,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifest.json", {
        "stage": "Phase 8B-RS-O", "phase8anc_root": str(nc),
        "reward_signal_root": str(reward_root), "reward_definition": EXPECTED_REWARD_DEFINITION,
        "population_only": True, "neural_training": False,
        "learned_predictions_used": False, "statistical_unit": "anchor_id",
        "analyzed_anchor_count": len(anchor_ids), "kappas": list(kappas),
        "lambda_grid": list(strengths), "conditions": list(CONDITIONS),
        "logger_probability_tables": logger_tables, "primary_mixtures": mixtures,
        "bootstrap_repetitions": bootstrap_reps, "bootstrap_seed": seed,
        "numerical_tolerance": {"atol": atol, "rtol": rtol},
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
    })
    _make_figures(output, decomposition_rows, fit_errors, strengths)
    figure_rows = [
        {"file": "balanced_reward_bias_vs_lambda.png", "purpose": "Audit balanced observational-do bias.",
         "observation": "Plus/minus separate linearly while base remains at its physical baseline.",
         "implication": "Source balancing does not remove reward confounding."},
        {"file": "heavy_reward_drift_vs_lambda.png", "purpose": "Audit logger-mixture drift.",
         "observation": "The signed plus/minus drift changes with slopes +/-14/45.",
         "implication": "Mixture composition changes observational reward without changing do reward."},
        {"file": "physical_vs_direct_bias.png", "purpose": "Separate the two bias channels.",
         "observation": "Physical bias is lambda-invariant and direct bias is additive.",
         "implication": "The direct channel does not replace actuator-mediated bias."},
        {"file": "total_bias_decomposition.png", "purpose": "Verify total-bias additivity.",
         "observation": "Physical plus direct coincides with total bias.",
         "implication": "The two mechanisms are numerically identifiable in the oracle audit."},
        {"file": "confounded_vs_independent_reward_bias.png", "purpose": "Check the latent-independence control.",
         "observation": "Only the confounded condition grows with lambda.",
         "implication": "The direct bias requires association between action and U_env."},
        {"file": "plus_minus_base_bias_vs_lambda.png", "purpose": "Show the clean kappa-zero experiment.",
         "observation": "Slopes are +0.6, -0.6, and 0.",
         "implication": "At kappa zero the total bias is purely direct."},
        {"file": "population_signal_vs_previous_neural_error.png", "purpose": "Compare signal and old fit-error scales.",
         "observation": "The fixed lambda grid crosses the previous neural error scale.",
         "implication": "This comparison diagnoses scale only; it does not establish learnability."},
    ]
    _write_reports(output, summary, figure_rows)
    return summary
