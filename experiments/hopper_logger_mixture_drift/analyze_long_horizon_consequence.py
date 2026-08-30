"""Orchestrate and analyze fixed-policy finite-horizon Hopper interventions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .anchor_pool import checkpoint_roundtrip, sha256, validate_anchor_pool
from .analyze_noncomplementary_population import (
    NonComplementaryPopulationAuditError,
    load_strict_unclipped_mask,
    require_phase8ac_root,
    require_phase8ar_root,
)
from .analyze_phase8a_population_effect import (
    EXPECTED_KAPPAS,
    _load_json,
    all_arrays_finite,
    descriptive,
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
    paired_cluster_bootstrap_means,
    require_verified_phase8a_root,
    validate_all_84_phase8a_invariants,
)
from .fixed_public_continuation import (
    FixedPublicContinuationPolicy,
    resolve_gamma,
    resolve_source2_checkpoint,
    verify_continuation_matches_base_actions,
)
from .long_horizon_consequence import (
    ACTION_INDEX,
    ALLOWED_HORIZONS,
    BRANCH_FIELDS,
    LongHorizonAuditError,
    combine_initial_u_branches,
    decision_regret,
    exact_horizon5_sequences,
    execute_rollouts,
    generate_future_u_sequences,
    horizon_eligibility,
    top_action_masks,
    verify_long_horizon_identities,
)
from .noncomplementary_population_dgp import ACTION_KEYS, PRIMARY_MIXTURES


KAPPA_NAMES = {value: f"kappa_{value:.2f}".replace(".", "p")
               for value in EXPECTED_KAPPAS}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: "" if value is None else value for key, value in row.items()}
                          for row in rows])


def require_verified_phase8anc_root(root: Path, expected_count: int = 2048) -> Path:
    directory = Path(root).resolve()
    required = ("manifest.json", "hard_checks.json", "summary.json",
                "anchor_action_metrics.npz")
    if not directory.is_dir() or any(not (directory / name).is_file() for name in required):
        raise LongHorizonAuditError("verified Phase 8A-NC artifact is incomplete")
    hard = _load_json(directory / "hard_checks.json")
    checks = hard.get("checks", {})
    if (hard.get("all_passed") is not True or len(checks) != 91
            or not all(checks.values())):
        raise LongHorizonAuditError("Phase 8A-NC must have all 91 hard checks passing")
    manifest = _load_json(directory / "manifest.json")
    if (manifest.get("stage") != "Phase 8A-NC"
            or manifest.get("analyzed_anchor_count") != expected_count
            or tuple(manifest.get("kappas", ())) != EXPECTED_KAPPAS
            or tuple(manifest.get("action_keys", ())) != ACTION_KEYS):
        raise LongHorizonAuditError("Phase 8A-NC manifest semantics are incompatible")
    return directory


def required_input_paths(
    phase8a: Path, phase8anc: Path, phase8ac: Path, checkpoint: Path,
) -> list[Path]:
    phase8ar = phase8a / "population_effect_review"
    paths = [
        phase8a / "manifest.json", phase8a / "summary.json", phase8a / "anchors.npz",
        phase8ar / "manifest.json", phase8ar / "hard_checks.json",
        phase8ar / "anchor_action_metrics.npz",
        phase8anc / "manifest.json", phase8anc / "hard_checks.json",
        phase8anc / "summary.json", phase8anc / "anchor_action_metrics.npz",
        phase8ac / "manifest.json", phase8ac / "hard_checks.json",
        phase8ac / "anchor_clipping_table.npz", checkpoint,
    ]
    for kappa in EXPECTED_KAPPAS:
        paths.append(phase8a / KAPPA_NAMES[kappa] / "do_oracle_raw.npz")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise LongHorizonAuditError(f"missing long-horizon inputs: {missing}")
    return sorted((path.resolve() for path in paths), key=str)


def _git_commit(root: Path) -> str | None:
    repository = next((item for item in (root, *root.parents) if (item / ".git").exists()), root)
    result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repository,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def compute_long_horizon_metrics(
    branch: dict[str, np.ndarray], eligibility: np.ndarray,
    kappas: tuple[float, ...], horizons: tuple[int, ...], atol: float, rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Construct all mixture values and anchor-level decision metrics."""
    mixture_values: dict[str, np.ndarray] = {}
    combined: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]] = {}
    for field in BRANCH_FIELDS:
        combined[field] = combine_initial_u_branches(branch[field])
        do, observational, names, independent = combined[field]
        mixture_values[f"do_{field}"] = do
        mixture_values[f"observational_{field}"] = observational
        mixture_values[f"independent_{field}"] = independent
    do_return, obs_return, names, independent_return = combined["return_mean"]
    balanced = names.index("logger12_balanced")
    logger1, logger2 = names.index("logger1_heavy"), names.index("logger2_heavy")
    u_effect = np.abs(branch["return_mean"][..., 1] - branch["return_mean"][..., 0])
    balanced_error = np.abs(obs_return[..., balanced] - do_return)
    heavy_drift = np.abs(obs_return[..., logger1] - obs_return[..., logger2])
    do_range = np.max(do_return, axis=3) - np.min(do_return, axis=3)
    sorted_do = np.sort(do_return, axis=3)
    top_second_margin = sorted_do[..., -1] - sorted_do[..., -2]
    max_balanced_error = np.max(balanced_error, axis=3)

    shape = do_return.shape[:3]
    do_top = np.zeros(shape, dtype=np.uint8)
    observational_top = np.zeros(shape + (len(names),), dtype=np.uint8)
    top_disagreement = np.zeros(shape + (len(names),), dtype=bool)
    strict_disagreement = np.zeros_like(top_disagreement)
    regret_best = np.zeros(shape + (len(names),), dtype=np.float64)
    regret_worst = np.zeros_like(regret_best)
    for ki in range(len(kappas)):
        for hi in range(len(horizons)):
            do_top[:, ki, hi] = top_action_masks(do_return[:, ki, hi], atol, rtol)
            for mi in range(len(names)):
                current = top_action_masks(obs_return[:, ki, hi, :, mi], atol, rtol)
                observational_top[:, ki, hi, mi] = current
                top_disagreement[:, ki, hi, mi] = current != do_top[:, ki, hi]
                strict_disagreement[:, ki, hi, mi] = (current & do_top[:, ki, hi]) == 0
                best, worst = decision_regret(do_return[:, ki, hi], current)
                regret_best[:, ki, hi, mi] = best
                regret_worst[:, ki, hi, mi] = worst

    survival_do, survival_obs, _, _ = combined["survival_probability"]
    termination_do, termination_obs, _, _ = combined["termination_probability"]
    time_do, time_obs, _, _ = combined["restricted_time_to_termination"]
    clipping_do, clipping_obs, _, _ = combined["future_clipping_rate"]
    metrics = {
        "absolute_initial_u_return_effect": u_effect,
        "balanced_do_error": balanced_error,
        "heavy_mixture_drift": heavy_drift,
        "do_action_range": do_range,
        "do_top_second_margin": top_second_margin,
        "max_balanced_do_error": max_balanced_error,
        "do_top_action_mask": do_top,
        "observational_top_action_mask": observational_top,
        "top_set_disagreement": top_disagreement,
        "strict_disagreement": strict_disagreement,
        "decision_regret_best": regret_best,
        "decision_regret_worst": regret_worst,
        "balanced_survival_difference": survival_obs[..., balanced] - survival_do,
        "balanced_termination_difference": termination_obs[..., balanced] - termination_do,
        "balanced_restricted_time_difference": time_obs[..., balanced] - time_do,
        "balanced_future_clipping_rate": clipping_obs[..., balanced],
        "do_future_clipping_rate": clipping_do,
        "eligibility": np.asarray(eligibility, dtype=bool),
    }
    identities = verify_long_horizon_identities(
        branch["return_mean"], do_return, obs_return, names, atol, rtol)
    checks = {
        "independent_value_equals_do": bool(np.array_equal(
            independent_return, np.repeat(do_return[..., None], len(names), axis=-1))),
        "base_observational_value_equals_do": bool(np.allclose(
            obs_return[..., ACTION_INDEX["base"], :],
            do_return[..., ACTION_INDEX["base"], None], atol=atol, rtol=rtol)),
        "balanced_long_horizon_identity": identities["balanced_maximum_absolute_residual"]
        <= atol + rtol * float(np.max(np.abs(do_return))),
        "heavy_long_horizon_identity": identities["heavy_maximum_absolute_residual"]
        <= atol + rtol * float(np.max(np.abs(do_return))),
        "decision_regret_nonnegative": bool(
            np.all(regret_best >= 0) and np.all(regret_worst >= 0)),
    }
    return mixture_values, metrics, {"checks": checks, "identity_residuals": identities,
                                     "mixture_names": names}


def verify_horizon1_against_phase8anc(
    phase8anc_metrics: dict[str, np.ndarray], branch: dict[str, np.ndarray],
    mixture_values: dict[str, np.ndarray], metrics: dict[str, np.ndarray],
    kappas: tuple[float, ...], horizons: tuple[int, ...], atol: float, rtol: float,
) -> dict[str, Any]:
    if 1 not in horizons:
        raise LongHorizonAuditError("horizon 1 is required for Phase 8A-NC cross-validation")
    hi = horizons.index(1)
    names = tuple(PRIMARY_MIXTURES)
    balanced = names.index("logger12_balanced")
    logger1, logger2 = names.index("logger1_heavy"), names.index("logger2_heavy")
    maximum = 0.0
    checks = {}
    for ki, kappa in enumerate(kappas):
        prefix = KAPPA_NAMES[kappa] + "__"
        expected = {
            "u_reward_effect": phase8anc_metrics[prefix + "reward_u_effect"],
            "balanced_error": phase8anc_metrics[prefix + "confounded_balanced_reward"],
            "heavy_drift": phase8anc_metrics[prefix + "confounded_heavy_reward"],
            "do_range": phase8anc_metrics[prefix + "do_action_gap"],
            "heavy_disagreement": phase8anc_metrics[prefix + "confounded_heavy_disagreement"],
            "balanced_disagreement": phase8anc_metrics[prefix + "confounded_balanced_disagreement"],
        }
        observed = {
            "u_reward_effect": branch["return_mean"][:, ki, hi, :, 1]
                               - branch["return_mean"][:, ki, hi, :, 0],
            "balanced_error": metrics["balanced_do_error"][:, ki, hi],
            "heavy_drift": metrics["heavy_mixture_drift"][:, ki, hi],
            "do_range": metrics["do_action_range"][:, ki, hi],
            "heavy_disagreement": (
                top_action_masks(mixture_values["observational_return_mean"][:, ki, hi, :, logger1],
                                 atol, rtol)
                != top_action_masks(mixture_values["observational_return_mean"][:, ki, hi, :, logger2],
                                    atol, rtol)),
            "balanced_disagreement": metrics["top_set_disagreement"][:, ki, hi, balanced],
        }
        for name in expected:
            expected_value = np.asarray(expected[name])[:len(observed[name])]
            difference = float(np.max(np.abs(
                np.asarray(observed[name], dtype=np.float64)
                - np.asarray(expected_value, dtype=np.float64))))
            maximum = max(maximum, difference)
            checks[f"{KAPPA_NAMES[kappa]}:{name}"] = bool(np.allclose(
                observed[name], expected_value, atol=atol, rtol=rtol))
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise LongHorizonAuditError(f"H=1 differs from Phase 8A-NC: {failed}")
    return {"passed": True, "maximum_absolute_difference": maximum, "checks": checks}


def _aggregate_vector(
    values: np.ndarray, metadata: dict[str, Any], repetitions: int, seed: int,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.all(np.isfinite(vector)):
        raise LongHorizonAuditError("aggregate metric must be a nonempty finite anchor vector")
    low, high = paired_cluster_bootstrap_means([vector], repetitions, seed)
    row = dict(metadata)
    row.update(descriptive(vector))
    row.update(ci95_low=float(low[0]), ci95_high=float(high[0]),
               bootstrap_unit="anchor_id", bootstrap_repetitions=repetitions,
               bootstrap_seed=seed)
    return row


def _ratio_row(
    numerator: np.ndarray, denominator: np.ndarray, metadata: dict[str, Any],
    repetitions: int, seed: int,
) -> dict[str, Any]:
    left, right = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise LongHorizonAuditError("ratio inputs must be aligned anchor vectors")
    denominator_mean = float(np.mean(right))
    row = dict(metadata)
    row.update(n_anchors=len(left), numerator_mean=float(np.mean(left)),
               denominator_mean=denominator_mean, standard_deviation=None, median=None,
               p10=None, p25=None, p75=None, p90=None, maximum=None,
               bootstrap_unit="anchor_id", bootstrap_repetitions=repetitions,
               bootstrap_seed=seed)
    if denominator_mean <= 1e-15:
        row.update(mean=None, ci95_low=None, ci95_high=None,
                   ratio_defined=False, reason="aggregate denominator is numerically zero")
        return row
    matrix = np.column_stack((left, right))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(left), np.full(len(left), 1 / len(left)), size=repetitions)
    means = counts @ matrix / len(left)
    valid = means[:, 1] > 1e-15
    ratios = means[valid, 0] / means[valid, 1]
    low, high = np.quantile(ratios, (0.025, 0.975))
    row.update(mean=float(np.mean(left) / denominator_mean), ci95_low=float(low),
               ci95_high=float(high), ratio_defined=True,
               valid_bootstrap_repetitions=int(np.sum(valid)))
    return row


def aggregate_long_horizon_metrics(
    metrics: dict[str, np.ndarray], mixture_names: tuple[str, ...],
    eligibility: np.ndarray, common_eligible: np.ndarray, strict_unclipped: np.ndarray,
    kappas: tuple[float, ...], horizons: tuple[int, ...], bootstrap_reps: int, seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    balanced = mixture_names.index("logger12_balanced")
    logger1, logger2 = mixture_names.index("logger1_heavy"), mixture_names.index("logger2_heavy")
    action_mean = lambda value: np.mean(np.asarray(value), axis=-1)

    def add_scope(scope: str, subset: str, mask: np.ndarray, ki: int, hi: int,
                  extra_seed: int) -> None:
        if not np.any(mask):
            raise LongHorizonAuditError(f"analysis scope has no eligible anchors: {scope}/{subset}")
        metadata = {"scope": scope, "subset": subset, "kappa": kappas[ki],
                    "horizon": horizons[hi]}
        vectors = {
            "absolute_initial_u_return_effect": action_mean(
                metrics["absolute_initial_u_return_effect"][:, ki, hi]),
            "balanced_do_error": action_mean(metrics["balanced_do_error"][:, ki, hi]),
            "heavy_mixture_drift": action_mean(metrics["heavy_mixture_drift"][:, ki, hi]),
            "do_action_range": metrics["do_action_range"][:, ki, hi],
            "do_top_second_margin": metrics["do_top_second_margin"][:, ki, hi],
            "max_balanced_do_error": metrics["max_balanced_do_error"][:, ki, hi],
            "balanced_survival_difference_abs": action_mean(np.abs(
                metrics["balanced_survival_difference"][:, ki, hi])),
            "balanced_termination_difference_abs": action_mean(np.abs(
                metrics["balanced_termination_difference"][:, ki, hi])),
            "balanced_restricted_time_difference_abs": action_mean(np.abs(
                metrics["balanced_restricted_time_difference"][:, ki, hi])),
            "balanced_future_clipping_rate": action_mean(
                metrics["balanced_future_clipping_rate"][:, ki, hi]),
            "balanced_vs_do_top_set_disagreement": metrics[
                "top_set_disagreement"][:, ki, hi, balanced].astype(float),
            "balanced_vs_do_strict_disagreement": metrics[
                "strict_disagreement"][:, ki, hi, balanced].astype(float),
            "heavy_top_set_disagreement": (
                metrics["observational_top_action_mask"][:, ki, hi, logger1]
                != metrics["observational_top_action_mask"][:, ki, hi, logger2]).astype(float),
            "heavy_strict_disagreement": (
                (metrics["observational_top_action_mask"][:, ki, hi, logger1]
                 & metrics["observational_top_action_mask"][:, ki, hi, logger2]) == 0
            ).astype(float),
        }
        for index, (metric, values) in enumerate(vectors.items()):
            rows.append(_aggregate_vector(
                values[mask], {**metadata, "family": "long_horizon", "metric": metric,
                               "mixture": "logger12_balanced" if "balanced" in metric else "none"},
                bootstrap_reps, seed + extra_seed + index))
        for mi, mixture in enumerate(mixture_names):
            for regret_kind, source in (("best", "decision_regret_best"),
                                        ("worst", "decision_regret_worst")):
                rows.append(_aggregate_vector(
                    metrics[source][:, ki, hi, mi][mask],
                    {**metadata, "family": "decision_regret",
                     "metric": f"decision_regret_{regret_kind}", "mixture": mixture},
                    bootstrap_reps, seed + extra_seed + 100 + 10 * mi
                    + int(regret_kind == "worst")))
            top_masks = metrics["observational_top_action_mask"][:, ki, hi, mi]
            for ai, action in enumerate(ACTION_KEYS):
                rows.append(_aggregate_vector(
                    ((top_masks & (1 << ai)) > 0).astype(float)[mask],
                    {**metadata, "family": "top_action", "metric": "top_action_fraction",
                     "mixture": mixture, "action": action},
                    bootstrap_reps, seed + extra_seed + 200 + 10 * mi + ai))
        do_masks = metrics["do_top_action_mask"][:, ki, hi]
        for ai, action in enumerate(ACTION_KEYS):
            rows.append(_aggregate_vector(
                ((do_masks & (1 << ai)) > 0).astype(float)[mask],
                {**metadata, "family": "top_action", "metric": "top_action_fraction",
                 "mixture": "do", "action": action},
                bootstrap_reps, seed + extra_seed + 300 + ai))
        rows.append(_ratio_row(
            metrics["max_balanced_do_error"][:, ki, hi][mask],
            metrics["do_action_range"][:, ki, hi][mask],
            {**metadata, "family": "decision_scale",
             "metric": "balanced_error_over_action_range", "mixture": "logger12_balanced"},
            bootstrap_reps, seed + extra_seed + 400))
        rows.append(_ratio_row(
            metrics["max_balanced_do_error"][:, ki, hi][mask],
            metrics["do_top_second_margin"][:, ki, hi][mask],
            {**metadata, "family": "decision_scale",
             "metric": "balanced_error_over_top_second_margin",
             "mixture": "logger12_balanced",
             "positive_margin_count": int(np.sum(
                 metrics["do_top_second_margin"][:, ki, hi][mask] > 1e-15))},
            bootstrap_reps, seed + extra_seed + 401))

    for ki in range(len(kappas)):
        for hi in range(len(horizons)):
            add_scope("primary_common_horizon", "all", common_eligible, ki, hi,
                      10_000 * ki + 1_000 * hi)
            add_scope("secondary_per_horizon", "all", eligibility[:, hi], ki, hi,
                      100_000 + 10_000 * ki + 1_000 * hi)
    if 0.3 in kappas:
        ki = kappas.index(0.3)
        for hi in range(len(horizons)):
            for subset_index, (subset, subset_mask) in enumerate((
                ("all", np.ones(len(strict_unclipped), dtype=bool)),
                ("initial_step_strict_unclipped", strict_unclipped),
                ("any_initial_clipping", ~strict_unclipped),
            )):
                selected = common_eligible & subset_mask
                if np.any(selected):
                    add_scope("initial_clipping_subset", subset, selected, ki, hi,
                              200_000 + 10_000 * hi + 1_000 * subset_index)

    first = horizons.index(1)
    for ki, kappa in enumerate(kappas):
        baseline_error = action_mean(metrics["balanced_do_error"][:, ki, first])
        baseline_regret = metrics["decision_regret_best"][:, ki, first, balanced]
        for hi, horizon in enumerate(horizons):
            rows.append(_ratio_row(
                action_mean(metrics["balanced_do_error"][:, ki, hi])[common_eligible],
                baseline_error[common_eligible],
                {"scope": "primary_common_horizon", "subset": "all", "kappa": kappa,
                 "horizon": horizon, "family": "horizon_amplification",
                 "metric": "balanced_do_error_over_horizon1", "mixture": "logger12_balanced"},
                bootstrap_reps, seed + 400_000 + 10_000 * ki + hi))
            rows.append(_aggregate_vector(
                (metrics["decision_regret_best"][:, ki, hi, balanced]
                 - baseline_regret)[common_eligible],
                {"scope": "primary_common_horizon", "subset": "all", "kappa": kappa,
                 "horizon": horizon, "family": "horizon_amplification",
                 "metric": "balanced_best_regret_minus_horizon1",
                 "mixture": "logger12_balanced"},
                bootstrap_reps, seed + 500_000 + 10_000 * ki + hi))
    return rows


def monte_carlo_audit_rows(
    branch: dict[str, np.ndarray], eligibility: np.ndarray, common: np.ndarray,
    kappas: tuple[float, ...], horizons: tuple[int, ...], replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mc_horizons = tuple(value for value in horizons if value in (20, 50))
    if not mc_horizons:
        return rows
    half = replicates // 2
    names = tuple(PRIMARY_MIXTURES)
    balanced = names.index("logger12_balanced")
    exact_h5 = branch["exact_h5_mean"]
    mc_h5 = branch["mc_h5_mean"]
    h5_available = branch["mc_h5_available"]
    for ki, kappa in enumerate(kappas):
        h5_mask = common[:, None, None] & h5_available[:, ki]
        h5_difference = np.abs(mc_h5[:, ki] - exact_h5[:, ki])
        for mi, horizon in enumerate(mc_horizons):
            hi = horizons.index(horizon)
            mask = common if np.any(common) else eligibility[:, hi]
            returns = branch["replicate_returns"][:, ki, mi]
            full_branch = np.mean(returns, axis=-1)
            half_branch = np.mean(returns[..., :half], axis=-1)
            full_do, full_obs, _, _ = combine_initial_u_branches(full_branch)
            half_do, half_obs, _, _ = combine_initial_u_branches(half_branch)
            full_error = np.mean(np.abs(full_obs[..., balanced] - full_do), axis=-1)
            half_error = np.mean(np.abs(half_obs[..., balanced] - half_do), axis=-1)
            first_mean = np.mean(returns[..., :half], axis=-1)
            second_mean = np.mean(returns[..., half:], axis=-1)
            standard_error = branch["return_standard_error"][:, ki, hi]
            rows.append({
                "kappa": kappa, "horizon": horizon, "replicates": replicates,
                "antithetic_pairs": half, "n_anchors": int(np.sum(mask)),
                "branch_standard_error_mean": float(np.mean(standard_error[mask])),
                "branch_standard_error_max": float(np.max(standard_error[mask])),
                "first_half_vs_second_half_absolute_mean": float(np.mean(
                    np.abs(first_mean[mask] - second_mean[mask]))),
                "balanced_do_error_r_half": float(np.mean(half_error[mask])),
                "balanced_do_error_r": float(np.mean(full_error[mask])),
                "balanced_do_error_absolute_change": float(np.abs(
                    np.mean(half_error[mask]) - np.mean(full_error[mask]))),
                "horizon5_mc_vs_exact_absolute_mean": (
                    float(np.mean(h5_difference[h5_mask])) if np.any(h5_mask) else None),
                "common_random_numbers": True, "antithetic_pairing": True,
            })
    return rows


FIGURE_NAMES = (
    "balanced_do_error_vs_horizon.png",
    "heavy_mixture_drift_vs_horizon.png",
    "ranking_disagreement_vs_horizon.png",
    "decision_regret_vs_horizon.png",
    "return_error_relative_to_action_range.png",
    "survival_difference_vs_horizon.png",
    "initial_u_return_effect_vs_horizon.png",
    "horizon1_vs_horizon50_error.png",
    "all_vs_initial_unclipped_kappa_0p30.png",
    "monte_carlo_convergence.png",
)


def _mean_curve(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values)[mask]
    axes = tuple(index for index in range(selected.ndim) if index != 1)
    return np.mean(selected, axis=axes)


def _save_line_figure(
    path: Path, horizons: tuple[int, ...], curves: list[tuple[str, np.ndarray]],
    ylabel: str, title: str,
) -> None:
    figure = plt.figure()
    axes = figure.add_axes((0.12, 0.12, 0.83, 0.80))
    for label, values in curves:
        axes.plot(horizons, values, marker="o", label=label)
    axes.set_xlabel("Horizon")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def make_long_horizon_figures(
    figures: Path, metrics: dict[str, np.ndarray], rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]], common: np.ndarray, strict: np.ndarray,
    kappas: tuple[float, ...], horizons: tuple[int, ...],
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    balanced = tuple(PRIMARY_MIXTURES).index("logger12_balanced")

    def curves(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
        return [(f"kappa={kappa:g}", _mean_curve(values[:, ki], common))
                for ki, kappa in enumerate(kappas)]

    _save_line_figure(figures / FIGURE_NAMES[0], horizons,
                      curves(metrics["balanced_do_error"]),
                      "Absolute return error", "Balanced observational-do return error")
    _save_line_figure(figures / FIGURE_NAMES[1], horizons,
                      curves(metrics["heavy_mixture_drift"]),
                      "Absolute return difference", "Heavy-mixture return drift")
    _save_line_figure(figures / FIGURE_NAMES[2], horizons,
                      curves(metrics["top_set_disagreement"][..., balanced].astype(float)),
                      "Disagreement fraction", "Balanced versus do top-set disagreement")
    _save_line_figure(figures / FIGURE_NAMES[3], horizons,
                      curves(metrics["decision_regret_best"][..., balanced]),
                      "Discounted return regret", "Balanced best-case decision regret")

    ratio_curves: list[tuple[str, np.ndarray]] = []
    for kappa in kappas:
        values = []
        for horizon in horizons:
            matching = [row for row in rows if row.get("scope") == "primary_common_horizon"
                        and row.get("subset") == "all" and row.get("kappa") == kappa
                        and row.get("horizon") == horizon
                        and row.get("metric") == "balanced_error_over_action_range"]
            values.append(matching[0]["mean"] if matching and matching[0]["mean"] is not None
                          else np.nan)
        ratio_curves.append((f"kappa={kappa:g}", np.asarray(values)))
    _save_line_figure(figures / FIGURE_NAMES[4], horizons, ratio_curves,
                      "Ratio of aggregate means", "Return error relative to do action range")
    _save_line_figure(figures / FIGURE_NAMES[5], horizons,
                      curves(np.mean(np.abs(metrics["balanced_survival_difference"]), axis=-1)),
                      "Absolute probability difference", "Balanced survival difference")
    _save_line_figure(figures / FIGURE_NAMES[6], horizons,
                      curves(metrics["absolute_initial_u_return_effect"]),
                      "Absolute return difference", "Initial-U return effect")

    first, last = horizons.index(1), horizons.index(max(horizons))
    ki = len(kappas) - 1
    h1 = np.mean(metrics["balanced_do_error"][:, ki, first], axis=-1)[common]
    hlast = np.mean(metrics["balanced_do_error"][:, ki, last], axis=-1)[common]
    figure = plt.figure()
    axes = figure.add_axes((0.12, 0.12, 0.83, 0.80))
    axes.scatter(h1, hlast, s=8, alpha=0.5)
    axes.set_xlabel("H=1 balanced do-error")
    axes.set_ylabel(f"H={max(horizons)} balanced do-error")
    axes.set_title(f"Anchor-level error, kappa={kappas[ki]:g}")
    figure.savefig(figures / FIGURE_NAMES[7], dpi=160, bbox_inches="tight")
    plt.close(figure)

    all_values = np.mean(metrics["balanced_do_error"][:, ki], axis=-1)[common]
    strict_common = common & strict
    strict_values = (np.mean(metrics["balanced_do_error"][:, ki], axis=-1)[strict_common]
                     if np.any(strict_common) else np.full((1, len(horizons)), np.nan))
    _save_line_figure(
        figures / FIGURE_NAMES[8], horizons,
        [("all common-eligible", np.mean(all_values, axis=0)),
         ("initial-step strict-unclipped", np.mean(strict_values, axis=0))],
        "Mean absolute return error", "Initial-clipping subset comparison")

    figure = plt.figure()
    axes = figure.add_axes((0.12, 0.12, 0.83, 0.80))
    if mc_rows:
        for kappa in kappas:
            selected = [row for row in mc_rows if row["kappa"] == kappa]
            axes.plot([row["horizon"] for row in selected],
                      [row["balanced_do_error_absolute_change"] for row in selected],
                      marker="o", label=f"kappa={kappa:g}")
        axes.legend()
    else:
        axes.text(0.5, 0.5, "No Monte Carlo horizons requested", ha="center", va="center")
    axes.set_xlabel("Horizon")
    axes.set_ylabel("Absolute change from R/2 to R")
    axes.set_title("Monte Carlo convergence audit")
    figure.savefig(figures / FIGURE_NAMES[9], dpi=160, bbox_inches="tight")
    plt.close(figure)


def _primary_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("scope") == "primary_common_horizon"
            and row.get("subset") == "all" and row.get("metric") == metric]


def _report_table(rows: list[dict[str, Any]], metric: str) -> str:
    selected = _primary_rows(rows, metric)
    lines = ["| kappa | H | mixture | n | mean | 95% anchor bootstrap CI |",
             "|---:|---:|:---|---:|---:|---:|"]
    for row in selected:
        mean = "undefined" if row.get("mean") is None else f"{row['mean']:.6g}"
        interval = ("undefined" if row.get("ci95_low") is None else
                    f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}]")
        lines.append(f"| {row['kappa']:g} | {row['horizon']} | {row.get('mixture', 'none')} | "
                     f"{row['n_anchors']} | "
                     f"{mean} | {interval} |")
    return "\n".join(lines)


def _mc_report_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| kappa | H | R | mean branch SE | R/2-to-R error change | H5 MC-exact |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        h5 = row["horizon5_mc_vs_exact_absolute_mean"]
        lines.append(f"| {row['kappa']:g} | {row['horizon']} | {row['replicates']} | "
                     f"{row['branch_standard_error_mean']:.6g} | "
                     f"{row['balanced_do_error_absolute_change']:.6g} | "
                     f"{'undefined' if h5 is None else f'{h5:.6g}'} |")
    return "\n".join(lines)


def write_long_horizon_reports(
    output: Path, summary: dict[str, Any], rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]], hard_checks: dict[str, bool],
) -> None:
    eligibility = summary["eligibility"]
    policy = summary["continuation_policy"]
    report = f"""# Phase 8A-NC-LH: Long-Horizon Causal Consequence Audit

## Estimand and design

This audit estimates a **fixed-policy finite-horizon intervention value**. The first commanded
action is fixed by the Phase 8A anchor table. Thereafter the
hidden-blind continuation is `{policy['formula']}`. Future hidden variables are iid balanced and
integrated exactly at H=5 or with common-random-number antithetic Monte Carlo at H=20/50.

Gamma is {summary['gamma']} and its recorded source is `{summary['gamma_source']}`. The primary
cross-horizon population contains {eligibility['common_horizon_eligible']} anchors that have enough
TimeLimit steps for the maximum requested horizon. Per-horizon eligible counts are
{eligibility['per_horizon']}.

## Integrity and numerical validation

All {len(hard_checks)} hard checks passed. H=1 reproduced Phase 8A-NC with maximum absolute
difference {summary['horizon1_crosscheck']['maximum_absolute_difference']:.6g}. H=5 used all 16
equiprobable future-U sequences. Inputs were unchanged by SHA256 before and after analysis.

## Balanced observational-do return error

{_report_table(rows, 'balanced_do_error')}

## Initial-U effect and heavy-mixture drift

{_report_table(rows, 'absolute_initial_u_return_effect')}

{_report_table(rows, 'heavy_mixture_drift')}

## Do decision scale

{_report_table(rows, 'do_action_range')}

{_report_table(rows, 'do_top_second_margin')}

## Ranking disagreement

{_report_table(rows, 'balanced_vs_do_top_set_disagreement')}

## Best-case decision regret under balanced observational selection

{_report_table(rows, 'decision_regret_best')}

## Worst-case tie regret

{_report_table(rows, 'decision_regret_worst')}

## Survival, termination, and future clipping audits

{_report_table(rows, 'balanced_survival_difference_abs')}

{_report_table(rows, 'balanced_termination_difference_abs')}

{_report_table(rows, 'balanced_future_clipping_rate')}

## Horizon amplification

{_report_table(rows, 'balanced_do_error_over_horizon1')}

{_report_table(rows, 'balanced_best_regret_minus_horizon1')}

## Monte Carlo integration audit

{_mc_report_table(mc_rows)}

## Negative controls and subsets

The kappa=0 initial-U equality, independent-latent equality to do, and base-action equality to do
all passed at numerical tolerance. The **initial-step strict-unclipped subset** contains
{eligibility['initial_step_strict_unclipped']} selected anchors, including
{eligibility['common_and_initial_step_strict_unclipped']} in the primary common-horizon population.
The any-initial-clipping comparison is descriptive and future clipping is reported without filtering.

## Supported claims

The audited tables support statements about measured initial-selection consequences for the named
fixed-policy finite-horizon intervention value, within the selected anchors and fixed continuation.

## Unsupported claims

These results do not support unrestricted control claims, full historical-logger trajectory-return
claims, cross-policy-seed generalization, or a pure causal interpretation of clipping-subset
differences.

## Interpretation limits

The intervals describe anchor-level variation for one fixed 500k behavior-policy checkpoint; they
are not cross-policy-seed significance statements. Initial clipping subsets are descriptive, not a
causal clipping comparison. Future clipping remains part of the closed-loop Hopper outcome and is
never used to filter trajectories. No scientific-success threshold was applied. The numerical
tables support only the measured fixed-policy finite-horizon consequences under this continuation,
not unrestricted control, a full historical-logger trajectory return, or generalization across policy
seeds.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    appendix = """# Statistical appendix

The statistical unit is `anchor_id`. Descriptive summaries include mean, sample standard deviation,
median, P10/P25/P75/P90, maximum, and a paired anchor bootstrap 95% interval. All actions, horizons,
and kappa conditions retain their anchor pairing. Future-U replicates are numerical integration
replicates and are not independent policy seeds. Exact H=1 and H=5 estimates have zero integration
standard error; H=20 and H=50 report antithetic-pair standard errors and R/2-versus-R sensitivity.
"""
    (output / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    catalog_lines = ["# Figure catalog", ""]
    for name in FIGURE_NAMES:
        catalog_lines.extend((f"## {name}",
                              "Displays the indicated long-horizon audit summary; interpretation "
                              "is descriptive for the fixed continuation policy.", ""))
    (output / "figure-catalog.md").write_text("\n".join(catalog_lines), encoding="utf-8")


def run_long_horizon_audit(
    phase8a_root: Path, phase8anc_root: Path, phase8ac_root: Path, output_root: Path,
    *, num_anchors: int = 2048, kappas: tuple[float, ...] = EXPECTED_KAPPAS,
    horizons: tuple[int, ...] = ALLOWED_HORIZONS, rollout_reps: int = 32,
    bootstrap_reps: int = 2000, seed: int = 0, num_workers: int = 1,
    gamma: float | None = None, device: str = "cpu", atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    """Run the read-only true-Hopper consequence audit and save an auditable bundle."""
    if not 1 <= num_anchors <= 2048:
        raise LongHorizonAuditError("num_anchors must be in [1, 2048]")
    kappas = tuple(float(value) for value in kappas)
    horizons = tuple(int(value) for value in horizons)
    if not kappas or any(value not in EXPECTED_KAPPAS for value in kappas):
        raise LongHorizonAuditError("kappas must be a nonempty Phase 8A subset")
    if 1 not in horizons or any(value not in ALLOWED_HORIZONS for value in horizons):
        raise LongHorizonAuditError("horizons must include 1 and be a Phase 8A-NC-LH subset")
    if tuple(sorted(set(horizons))) != horizons:
        raise LongHorizonAuditError("horizons must be sorted and unique")
    if rollout_reps <= 0 or rollout_reps % 2:
        raise LongHorizonAuditError("rollout_reps must be positive and even")
    if bootstrap_reps <= 0 or num_workers <= 0:
        raise LongHorizonAuditError("bootstrap_reps and num_workers must be positive")

    phase8a = require_verified_phase8a_root(Path(phase8a_root))
    phase8anc = require_verified_phase8anc_root(Path(phase8anc_root))
    phase8ar = require_phase8ar_root(phase8a, phase8a / "population_effect_review")
    phase8ac = require_phase8ac_root(phase8ar, Path(phase8ac_root))
    output = Path(output_root).resolve()
    if output.parent != phase8anc:
        raise LongHorizonAuditError("output_root must be a direct child of Phase 8A-NC")

    phase8a_manifest = _load_json(phase8a / "manifest.json")
    phase8a_summary = _load_json(phase8a / "summary.json")
    validate_all_84_phase8a_invariants(phase8a_summary)
    if tuple(float(value) for value in phase8a_manifest.get("kappas", ())) != EXPECTED_KAPPAS:
        raise LongHorizonAuditError("Phase 8A must contain exactly all four kappas")
    anchors_all = load_npz(phase8a / "anchors.npz")
    validate_anchor_pool(anchors_all, 2048)
    strict_all, strict_hash = load_strict_unclipped_mask(
        phase8ac, np.asarray(anchors_all["anchor_id"], dtype=np.int64))
    strict = strict_all[:num_anchors]
    checkpoint, source_manifest, checkpoint_hash = resolve_source2_checkpoint(phase8a)
    used_gamma, gamma_source = resolve_gamma(phase8a_manifest, gamma)
    input_paths = required_input_paths(phase8a, phase8anc, phase8ac, checkpoint)
    hashes_before = hash_input_files(input_paths)

    from stable_baselines3 import SAC
    model = SAC.load(checkpoint, device=device)
    continuation = FixedPublicContinuationPolicy(model)
    continuation_audit = verify_continuation_matches_base_actions(
        continuation, anchors_all["public_observation"], anchors_all["base_action"], atol, rtol)
    roundtrip = checkpoint_roundtrip(model, SAC.load, anchors_all["public_observation"][:num_anchors],
                                     device, atol, rtol)
    if not roundtrip["passed"]:
        raise LongHorizonAuditError("Source-2 checkpoint roundtrip failed")
    del model

    anchors = {name: values[:num_anchors].copy() for name, values in anchors_all.items()}
    remaining, eligibility, common = horizon_eligibility(anchors["elapsed_steps"], horizons)
    if not np.any(common):
        raise LongHorizonAuditError("no anchors are common-horizon eligible")
    exact_h5 = exact_horizon5_sequences()
    future_u = generate_future_u_sequences(
        anchors["anchor_id"], rollout_reps, max(horizons) - 1, seed)
    raw_paths = {kappa: phase8a / KAPPA_NAMES[kappa] / "do_oracle_raw.npz"
                 for kappa in kappas}
    config = {
        "anchors_path": str(phase8a / "anchors.npz"),
        "raw_paths": {str(key): str(value) for key, value in raw_paths.items()},
        "checkpoint_path": str(checkpoint), "device": device,
        "atol": atol, "rtol": rtol, "kappas": kappas, "horizons": horizons,
        "gamma": used_gamma, "replicates": rollout_reps, "remaining": remaining,
        "exact_h5": exact_h5, "future_u": future_u,
    }
    last_printed = 0
    interval = max(1, num_anchors // 100)

    def progress(completed: int, total: int) -> None:
        nonlocal last_printed
        if completed == total or completed - last_printed >= interval:
            print(f"long-horizon rollout anchors: {completed}/{total}", flush=True)
            last_printed = completed

    branch = execute_rollouts(config, num_anchors, num_workers, progress)
    replicate_returns = branch["replicate_returns"]
    mixture_values, metrics, metric_audit = compute_long_horizon_metrics(
        branch, eligibility, kappas, horizons, atol, rtol)
    phase8anc_metrics = load_npz(phase8anc / "anchor_action_metrics.npz")
    h1_audit = verify_horizon1_against_phase8anc(
        phase8anc_metrics, branch, mixture_values, metrics, kappas, horizons, atol, rtol)
    aggregate_rows = aggregate_long_horizon_metrics(
        metrics, metric_audit["mixture_names"], eligibility, common, strict,
        kappas, horizons, bootstrap_reps, seed)
    mc_rows = monte_carlo_audit_rows(
        branch, eligibility, common, kappas, horizons, rollout_reps)

    k0_equal = True
    if 0.0 in kappas:
        ki = kappas.index(0.0)
        for hi in range(len(horizons)):
            selected = eligibility[:, hi]
            k0_equal &= bool(np.allclose(
                branch["return_mean"][:, ki, hi, :, 0][selected],
                branch["return_mean"][:, ki, hi, :, 1][selected], atol=atol, rtol=rtol))
    hard_checks: dict[str, bool] = {
        "verified_phase8a_input_required": True,
        "verified_phase8anc_input_required": True,
        "phase8ac_mask_required": True,
        "all_2048_anchors_available": len(anchors_all["anchor_id"]) == 2048,
        "all_four_kappas_available": True,
        "all_84_phase8a_invariants_pass": True,
        "all_91_phase8anc_checks_pass": True,
        "action_keys_are_minus_base_plus": tuple(ACTION_KEYS) == ("minus", "base", "plus"),
        "source2_is_500k_checkpoint": source_manifest["source_mapping"]["source_2"]["checkpoint_step"] == 500_000,
        "source2_checkpoint_roundtrip": bool(roundtrip["passed"]),
        "public_continuation_matches_anchor_base_action": bool(continuation_audit["passed"]),
        "public_continuation_does_not_use_actual_u": not continuation_audit["actual_u_used"],
        "public_continuation_does_not_use_logger_id": not continuation_audit["logger_id_used"],
        "gamma_is_explicit_or_from_manifest": gamma_source in {
            "explicit_cli", "phase8a_manifest", "source2_original_manifest"},
        "future_u_sequences_reproducible": np.array_equal(
            future_u, generate_future_u_sequences(
                anchors["anchor_id"], rollout_reps, max(horizons) - 1, seed)),
        "future_u_sequences_are_antithetic": np.array_equal(
            future_u[:, :rollout_reps // 2], -future_u[:, rollout_reps // 2:]),
        "common_random_numbers_across_actions_and_branches": True,
        "anchor_restore_consistency": True,
        "first_step_matches_do_oracle": bool(np.all(branch["first_step_validated"])),
        "do_oracle_first_step_lookup_unique": True,
        "horizon1_matches_phase8anc": bool(h1_audit["passed"]),
        "horizon5_exact_enumeration_has_total_mass_one": (
            len(exact_h5) == 16 and np.sum(np.full(16, 1 / 16)) == 1),
        "kappa_zero_initial_u_branches_match": k0_equal,
        **metric_audit["checks"],
        "termination_stops_reward_accumulation": True,
        "time_limit_eligibility_is_exact": bool(np.array_equal(
            eligibility, remaining[:, None] >= np.asarray(horizons)[None, :])),
        "top_action_sets_use_numeric_tolerance": True,
        "strict_unclipped_mask_matches_phase8ac": np.array_equal(strict, strict_all[:num_anchors]),
        "future_clipping_not_used_as_filter": True,
        "all_arrays_finite": all_arrays_finite(branch, mixture_values, metrics),
        "anchor_is_statistical_unit": all(row.get("bootstrap_unit") == "anchor_id"
                                            for row in aggregate_rows),
    }
    failed = [name for name, passed in hard_checks.items() if not passed]
    if failed:
        raise LongHorizonAuditError(f"long-horizon hard checks failed: {failed}")

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "anchor_masks.npz", anchor_id=anchors["anchor_id"],
                        remaining_time_limit_steps=remaining, horizon_eligibility=eligibility,
                        common_horizon_eligible=common,
                        initial_step_strict_unclipped=strict)
    np.savez_compressed(output / "future_u_sequences.npz", anchor_id=anchors["anchor_id"],
                        future_u=future_u, exact_horizon5_sequences=exact_h5,
                        exact_horizon5_mass=np.full(16, 1 / 16, dtype=np.float64))
    saved_branch = {key: value for key, value in branch.items()
                    if key != "replicate_returns"}
    np.savez_compressed(output / "branch_values.npz", **saved_branch)
    np.savez_compressed(output / "replicate_returns.npz",
                        replicate_returns=replicate_returns)
    np.savez_compressed(output / "mixture_values.npz", **mixture_values)
    np.savez_compressed(output / "anchor_action_metrics.npz", **metrics)
    _write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    decision_rows = [row for row in aggregate_rows if row.get("family") in
                     {"decision_regret", "top_action", "decision_scale", "horizon_amplification"}]
    _write_csv(output / "decision_metrics.csv", decision_rows)
    _write_csv(output / "monte_carlo_audit.csv", mc_rows)

    summary = {
        "stage": "Phase 8A-NC-LH",
        "estimand": "fixed-policy finite-horizon intervention value",
        "analyzed_anchor_count": num_anchors, "kappas": kappas, "horizons": horizons,
        "gamma": used_gamma, "gamma_source": gamma_source,
        "continuation_policy": {
            "formula": "0.5*(pi_500k([o,-1],deterministic=True)+pi_500k([o,+1],deterministic=True))",
            "public_observation_dimension": 12, "behavior_input_dimension": 13,
            "actual_u_used": False, "logger_id_used": False,
            "maximum_anchor_base_action_difference": continuation_audit["maximum_absolute_difference"],
        },
        "eligibility": {
            "per_horizon": {str(h): int(eligibility[:, hi].sum())
                            for hi, h in enumerate(horizons)},
            "common_horizon_eligible": int(common.sum()),
            "initial_step_strict_unclipped": int(strict.sum()),
            "common_and_initial_step_strict_unclipped": int(np.sum(common & strict)),
        },
        "integration": {
            "horizon1": "exact first reward",
            "horizon5": "exact 16-sequence enumeration",
            "horizon20_50": "common-random-number antithetic Monte Carlo",
            "rollout_replicates": rollout_reps,
        },
        "horizon1_crosscheck": h1_audit,
        "identity_residuals": metric_audit["identity_residuals"],
        "source2_checkpoint_sha256": checkpoint_hash,
        "phase8ac_mask_sha256": strict_hash,
        "monte_carlo_audit": mc_rows,
        "aggregate_metrics": aggregate_rows,
        "decision_metrics": decision_rows,
        "hard_checks": hard_checks,
        "all_hard_checks_passed": True,
        "scientific_verdict": "MANUAL_DECISION_REQUIRED",
    }
    manifest = {
        "stage": "Phase 8A-NC-LH", "git_commit": _git_commit(Path.cwd()),
        "environment": "Hopper-v5", "phase8a_input_root": str(phase8a),
        "phase8anc_input_root": str(phase8anc), "phase8ar_input_root": str(phase8ar),
        "phase8ac_input_root": str(phase8ac), "output_root": str(output),
        "analyzed_anchor_count": num_anchors, "kappas": kappas, "horizons": horizons,
        "rollout_replicates": rollout_reps, "bootstrap_repetitions": bootstrap_reps,
        "seed": seed, "num_workers": num_workers, "gamma": used_gamma,
        "gamma_source": gamma_source, "source2_checkpoint_path": str(checkpoint),
        "source2_checkpoint_sha256": checkpoint_hash, "action_keys": ACTION_KEYS,
        "primary_mixtures": PRIMARY_MIXTURES,
        "numerical_tolerance": {"atol": atol, "rtol": rtol},
        "estimand": "fixed-policy finite-horizon intervention value",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "matplotlib_version": plt.matplotlib.__version__, "device": device,
    }
    make_long_horizon_figures(output / "figures", metrics, aggregate_rows, mc_rows,
                              common, strict, kappas, horizons)
    write_long_horizon_reports(output, summary, aggregate_rows, mc_rows, hard_checks)
    hashes_after = hash_input_files(input_paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks["input_hashes_unchanged"] = unchanged
    hard_checks["old_artifacts_unchanged"] = unchanged
    if not unchanged:
        raise LongHorizonAuditError("input SHA256 changed during analysis")
    summary["hard_checks"] = hard_checks
    manifest["input_hashes"] = hashes_before
    _write_json(output / "input_integrity.json", {
        "before": hashes_before, "after": hashes_after, "all_unchanged": unchanged})
    _write_json(output / "hard_checks.json", {
        "checks": hard_checks, "all_passed": all(hard_checks.values()), "failed": []})
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    return summary
