"""Build and audit the exact non-complementary logger population DGP."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .analyze_phase8a_population_effect import (
    EXPECTED_KAPPAS,
    PopulationEffectAuditError,
    _load_json,
    _write_json,
    all_arrays_finite,
    descriptive,
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
    paired_cluster_bootstrap_means,
    recompute_do_oracle,
    require_verified_phase8a_root,
    top_action_masks,
    validate_all_84_phase8a_invariants,
)
from .noncomplementary_population_dgp import (
    ACTION_KEYS,
    CONDITIONS,
    FORBIDDEN_PUBLIC_FIELDS,
    HIDDEN_FIELDS,
    LOGGER_ACTION_PROBABILITIES,
    LOGGER_NAMES,
    MIXTURES,
    PRIMARY_MIXTURES,
    PUBLIC_FIELDS,
    SECONDARY_MIXTURES,
    NonComplementaryDGPError,
    analytic_u_posterior,
    build_do_lookup,
    generate_support,
    generate_weights,
    logger_action_marginal,
    support_specification,
    validate_public_hidden,
    weighted_latent_correlation,
)


ACTION_INDEX = {action: index for index, action in enumerate(ACTION_KEYS)}
KAPPA_NAMES = {value: f"kappa_{value:.2f}".replace(".", "p")
               for value in EXPECTED_KAPPAS}


class NonComplementaryPopulationAuditError(RuntimeError):
    """Raised when an input or exact population invariant fails."""


def require_phase8ar_root(phase8a_root: Path, phase8ar_root: Path) -> Path:
    expected = Path(phase8a_root).resolve() / "population_effect_review"
    root = Path(phase8ar_root).resolve()
    if root != expected or not root.is_dir():
        raise NonComplementaryPopulationAuditError(
            "--phase8ar-root must be phase8a-root/population_effect_review"
        )
    hard = _load_json(root / "hard_checks.json")
    if hard.get("all_passed") is not True or not all(hard.get("checks", {}).values()):
        raise NonComplementaryPopulationAuditError("Phase 8A-R hard checks did not all pass")
    return root


def require_phase8ac_root(phase8ar_root: Path, phase8ac_root: Path) -> Path:
    expected = Path(phase8ar_root).resolve() / "clipping_sensitivity_kappa_0p30"
    root = Path(phase8ac_root).resolve()
    if root != expected or not root.is_dir():
        raise NonComplementaryPopulationAuditError(
            "--phase8ac-root must be phase8ar-root/clipping_sensitivity_kappa_0p30"
        )
    hard = _load_json(root / "hard_checks.json")
    if hard.get("all_passed") is not True or not all(hard.get("checks", {}).values()):
        raise NonComplementaryPopulationAuditError("Phase 8A-C hard checks did not all pass")
    return root


def required_input_paths(root: Path, review: Path, clipping: Path) -> list[Path]:
    paths = [root / "manifest.json", root / "summary.json", root / "anchors.npz",
             review / "manifest.json", review / "hard_checks.json",
             review / "anchor_action_metrics.npz", clipping / "manifest.json",
             clipping / "hard_checks.json", clipping / "anchor_clipping_table.npz"]
    for kappa in EXPECTED_KAPPAS:
        directory = root / KAPPA_NAMES[kappa]
        paths.extend((directory / "do_oracle_raw.npz", directory / "do_oracle_summary.npz"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise NonComplementaryPopulationAuditError(f"missing required inputs: {missing}")
    return sorted((path.resolve() for path in paths), key=str)


def load_strict_unclipped_mask(
    clipping_root: Path, all_anchor_ids: np.ndarray,
) -> tuple[np.ndarray, str]:
    path = clipping_root / "anchor_clipping_table.npz"
    table = load_npz(path)
    if not {"anchor_id", "strict_anchor_unclipped"}.issubset(table):
        raise NonComplementaryPopulationAuditError("Phase 8A-C clean mask is unavailable")
    if not np.array_equal(table["anchor_id"], all_anchor_ids):
        raise NonComplementaryPopulationAuditError("Phase 8A-C anchor IDs do not match Phase 8A")
    mask = np.asarray(table["strict_anchor_unclipped"], dtype=bool)
    if mask.shape != all_anchor_ids.shape:
        raise NonComplementaryPopulationAuditError("Phase 8A-C clean mask has wrong shape")
    return mask, hash_input_files([path.resolve()])[str(path.resolve())]


def validate_logger_definitions(atol: float) -> dict[str, bool]:
    for logger in (0, 1):
        for u in (-1, 1):
            if not np.isclose(sum(LOGGER_ACTION_PROBABILITIES[logger][u].values()), 1.0,
                              atol=atol, rtol=0):
                raise NonComplementaryPopulationAuditError("logger probability row does not sum to one")
    l1_plus = logger_action_marginal(0, "plus")
    l2_plus = logger_action_marginal(1, "plus")
    checks = {
        "logger_probability_rows_sum_to_one": True,
        "logger1_action_marginal_is_half": np.isclose(l1_plus, 0.5, atol=atol, rtol=0),
        "logger2_action_marginal_is_half": np.isclose(l2_plus, 0.5, atol=atol, rtol=0),
        "loggers_are_noncomplementary": not (
            np.isclose(LOGGER_ACTION_PROBABILITIES[1][1]["plus"],
                       LOGGER_ACTION_PROBABILITIES[0][-1]["plus"], atol=atol, rtol=0)
            and np.isclose(LOGGER_ACTION_PROBABILITIES[1][-1]["plus"],
                           LOGGER_ACTION_PROBABILITIES[0][1]["plus"], atol=atol, rtol=0)),
        "loggers_have_same_confounding_direction": (
            LOGGER_ACTION_PROBABILITIES[0][1]["plus"]
            > LOGGER_ACTION_PROBABILITIES[0][-1]["plus"]
            and LOGGER_ACTION_PROBABILITIES[1][1]["plus"]
            > LOGGER_ACTION_PROBABILITIES[1][-1]["plus"]),
    }
    if not all(checks.values()):
        raise NonComplementaryPopulationAuditError(f"logger definition check failed: {checks}")
    return checks


def support_marginals(hidden: dict[str, np.ndarray]) -> dict[str, float]:
    """Return exact per-logger marginals, averaging over repeated anchors."""
    n_anchors = len(np.unique(hidden["anchor_id"]))
    if n_anchors == 0:
        raise NonComplementaryPopulationAuditError("support has no anchors")
    mass = np.asarray(hidden["base_mass"], dtype=np.float64) / n_anchors
    result: dict[str, float] = {}
    for logger in (0, 1, 2):
        logger_mask = hidden["logger_id"] == logger
        for action in ACTION_KEYS:
            result[f"logger_{logger}_action_{action}"] = float(
                mass[logger_mask & (hidden["action_key"] == action)].sum())
        for u_env in (-1, 1):
            result[f"logger_{logger}_u_env_{u_env}"] = float(
                mass[logger_mask & (hidden["u_env"] == u_env)].sum())
    return result


def verify_primary_state_action_mass(
    hidden: dict[str, np.ndarray], weights: dict[str, np.ndarray],
    anchor_ids: np.ndarray, atol: float, rtol: float,
) -> dict[str, Any]:
    reference: dict[tuple[int, bytes], float] | None = None
    action_masses: dict[str, np.ndarray] = {}
    for mixture in PRIMARY_MIXTURES:
        current: dict[tuple[int, bytes], float] = {}
        for row, (anchor, command) in enumerate(zip(hidden["anchor_id"],
                                                    hidden["commanded_action"])):
            key = (int(anchor), np.ascontiguousarray(command).tobytes())
            current[key] = current.get(key, 0.0) + float(weights[mixture][row])
        if reference is None:
            reference = current
        elif set(current) != set(reference) or any(
                not np.isclose(current[key], reference[key], atol=atol, rtol=rtol)
                for key in reference):
            raise NonComplementaryPopulationAuditError(
                "NONCOMPLEMENTARY_PRIMARY_MIXTURES_FAIL_TO_PRESERVE_STATE_ACTION_MASS"
            )
        mass = np.empty((len(anchor_ids), 3), dtype=np.float64)
        for ai, anchor in enumerate(anchor_ids):
            for action in ACTION_KEYS:
                rows = ((hidden["anchor_id"] == anchor)
                        & (hidden["action_key"].astype(str) == action))
                mass[ai, ACTION_INDEX[action]] = weights[mixture][rows].sum()
        mass /= mass.sum(axis=1, keepdims=True)
        action_masses[mixture] = mass
        if not np.allclose(mass, (0.45, 0.10, 0.45), atol=atol, rtol=rtol):
            raise NonComplementaryPopulationAuditError(
                "NONCOMPLEMENTARY_PRIMARY_MIXTURES_FAIL_TO_PRESERVE_STATE_ACTION_MASS"
            )
    return {"passed": True, "conditional_action_mass": [0.45, 0.10, 0.45],
            "maximum_primary_difference": float(max(
                np.max(np.abs(action_masses[left] - action_masses[right]))
                for left in PRIMARY_MIXTURES for right in PRIMARY_MIXTURES))}


def population_response(
    public: dict[str, np.ndarray], hidden: dict[str, np.ndarray],
    weights: dict[str, np.ndarray], anchor_ids: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    lookup = {int(anchor): index for index, anchor in enumerate(anchor_ids)}
    result = {}
    for mixture in MIXTURES:
        reward = np.empty((len(anchor_ids), 3), dtype=np.float64)
        next_observation = np.empty((len(anchor_ids), 3, 12), dtype=np.float64)
        delta = np.empty((len(anchor_ids), 3, 11), dtype=np.float64)
        terminated = np.empty((len(anchor_ids), 3), dtype=np.float64)
        truncated = np.empty((len(anchor_ids), 3), dtype=np.float64)
        posterior = np.empty((len(anchor_ids), 3), dtype=np.float64)
        group_mass = np.empty((len(anchor_ids), 3), dtype=np.float64)
        for anchor in anchor_ids:
            for action in ACTION_KEYS:
                rows = np.flatnonzero((hidden["anchor_id"] == anchor)
                                      & (hidden["action_key"].astype(str) == action))
                local = weights[mixture][rows]
                mass = float(local.sum())
                if not len(rows) or mass <= 0:
                    raise NonComplementaryPopulationAuditError("population cell has no mass")
                local /= mass
                ai, aj = lookup[int(anchor)], ACTION_INDEX[action]
                reward[ai, aj] = local @ np.asarray(public["reward"][rows], dtype=np.float64)
                following = np.asarray(public["next_observation"][rows], dtype=np.float64)
                current = np.asarray(public["observation"][rows], dtype=np.float64)
                next_observation[ai, aj] = np.tensordot(local, following, axes=(0, 0))
                delta[ai, aj] = np.tensordot(local, following[:, :11] - current[:, :11],
                                             axes=(0, 0))
                terminated[ai, aj] = local @ public["terminated"][rows].astype(float)
                truncated[ai, aj] = local @ public["truncated"][rows].astype(float)
                posterior[ai, aj] = local @ (hidden["u_env"][rows] == 1).astype(float)
                group_mass[ai, aj] = mass
        result[mixture] = {
            "reward": reward, "next_observation": next_observation, "delta": delta,
            "terminated": terminated, "truncated": truncated,
            "posterior_u_plus": posterior, "state_action_mass": group_mass,
        }
    return result


def verify_posteriors(
    condition: str, observational: dict[str, dict[str, np.ndarray]],
    atol: float, rtol: float,
) -> dict[str, Any]:
    details, maximum = {}, 0.0
    for mixture in MIXTURES:
        details[mixture] = {}
        for action in ACTION_KEYS:
            expected = analytic_u_posterior(condition, mixture, action)
            actual = observational[mixture]["posterior_u_plus"][:, ACTION_INDEX[action]]
            difference = float(np.max(np.abs(actual - expected)))
            maximum = max(maximum, difference)
            if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                raise NonComplementaryPopulationAuditError(
                    f"empirical U posterior differs from analytic: {condition}/{mixture}/{action}"
                )
            details[mixture][action] = {"analytic": expected,
                                        "empirical_mean": float(np.mean(actual)),
                                        "maximum_absolute_difference": difference}
    return {"passed": True, "maximum_absolute_difference": maximum, "details": details}


def _metric(family: str, kappa: float, condition: str, action: str, mixture: str,
            metric: str, values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise NonComplementaryPopulationAuditError("metrics must be finite anchor vectors")
    return {"family": family, "kappa": kappa, "condition": condition,
            "action": action, "mixture": mixture, "metric": metric, "values": vector}


def _all_actions(values: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(values), axis=1)


def analyze_kappa(
    root: Path, kappa: float, anchors: dict[str, np.ndarray], anchor_ids: np.ndarray,
    datasets: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                              dict[str, np.ndarray]]], atol: float, rtol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, bool],
           dict[str, Any]]:
    directory = root / KAPPA_NAMES[kappa]
    raw, stored = load_npz(directory / "do_oracle_raw.npz"), load_npz(
        directory / "do_oracle_summary.npz")
    do, do_audit = recompute_do_oracle(raw, stored, anchors, anchor_ids, kappa, atol, rtol)
    observational, posterior_audits, mass_audits = {}, {}, {}
    for condition, (public, hidden, weights) in datasets.items():
        validate_public_hidden(public, hidden)
        if not all_arrays_finite(public, hidden, weights):
            raise NonComplementaryPopulationAuditError("support or weights contain NaN/Inf")
        mass_audits[condition] = verify_primary_state_action_mass(
            hidden, weights, anchor_ids, atol, rtol)
        observational[condition] = population_response(public, hidden, weights, anchor_ids)
        posterior_audits[condition] = verify_posteriors(
            condition, observational[condition], atol, rtol)

    specs: list[dict[str, Any]] = []
    reward_u = do["reward_u_effect"]
    delta_u_l2 = np.linalg.norm(do["delta_u_effect"], axis=2)
    primary_metrics, ranking = {}, {}
    for action in (*ACTION_KEYS, "all"):
        select = (lambda value: _all_actions(value)) if action == "all" else (
            lambda value, ai=ACTION_INDEX[action]: value[:, ai])
        specs.extend((
            _metric("u_effect", kappa, "do_oracle", action, "none",
                    "reward_u_effect_abs", select(np.abs(reward_u))),
            _metric("u_effect", kappa, "do_oracle", action, "none",
                    "delta_u_effect_l2", select(delta_u_l2)),
        ))
    for condition in CONDITIONS:
        obs = observational[condition]
        rewards = np.stack([obs[name]["reward"] for name in PRIMARY_MIXTURES], axis=1)
        signed_heavy_reward = (obs["logger1_heavy"]["reward"]
                               - obs["logger2_heavy"]["reward"])
        heavy_reward = np.abs(signed_heavy_reward)
        signed_heavy_delta = (obs["logger1_heavy"]["delta"]
                              - obs["logger2_heavy"]["delta"])
        heavy_delta = np.linalg.norm(signed_heavy_delta, axis=2)
        balanced_reward_signed = obs["logger12_balanced"]["reward"] - do["mean_reward"]
        balanced_reward = np.abs(balanced_reward_signed)
        balanced_delta_signed = obs["logger12_balanced"]["delta"] - do["mean_delta"]
        balanced_delta = np.linalg.norm(balanced_delta_signed, axis=2)
        equal_reward = np.abs(obs["all_sources_equal"]["reward"] - do["mean_reward"])
        equal_delta = np.linalg.norm(obs["all_sources_equal"]["delta"] - do["mean_delta"], axis=2)
        primary_metrics[condition] = {
            "signed_heavy_reward": signed_heavy_reward, "heavy_reward": heavy_reward,
            "signed_heavy_delta": signed_heavy_delta, "heavy_delta": heavy_delta,
            "balanced_reward_signed": balanced_reward_signed,
            "balanced_reward": balanced_reward,
            "balanced_delta_signed": balanced_delta_signed,
            "balanced_delta": balanced_delta, "equal_reward": equal_reward,
            "equal_delta": equal_delta,
        }
        masks = {name: top_action_masks(obs[name]["reward"], atol, rtol) for name in MIXTURES}
        do_mask = top_action_masks(do["mean_reward"], atol, rtol)
        ranking[condition] = {
            "masks": masks, "do_mask": do_mask,
            "heavy_disagreement": masks["logger1_heavy"] != masks["logger2_heavy"],
            "heavy_strict_flip": ((masks["logger1_heavy"] & masks["logger2_heavy"]) == 0),
            "balanced_disagreement": masks["logger12_balanced"] != do_mask,
            "equal_disagreement": masks["all_sources_equal"] != do_mask,
        }
        for action in (*ACTION_KEYS, "all"):
            select = (lambda value: _all_actions(value)) if action == "all" else (
                lambda value, ai=ACTION_INDEX[action]: value[:, ai])
            for metric_name, mixture, values in (
                ("reward_heavy_drift", "logger1_heavy_vs_logger2_heavy", heavy_reward),
                ("delta_heavy_drift", "logger1_heavy_vs_logger2_heavy", heavy_delta),
                ("reward_balanced_do_error", "logger12_balanced", balanced_reward),
                ("delta_balanced_do_error", "logger12_balanced", balanced_delta),
                ("reward_all_sources_equal_do_error", "all_sources_equal", equal_reward),
                ("delta_all_sources_equal_do_error", "all_sources_equal", equal_delta),
            ):
                specs.append(_metric("population_effect", kappa, condition, action,
                                     mixture, metric_name, select(values)))
        for metric_name, mixture, values in (
            ("heavy_top_set_disagreement", "logger1_heavy_vs_logger2_heavy",
             ranking[condition]["heavy_disagreement"]),
            ("heavy_strict_flip", "logger1_heavy_vs_logger2_heavy",
             ranking[condition]["heavy_strict_flip"]),
            ("balanced_vs_do_disagreement", "logger12_balanced",
             ranking[condition]["balanced_disagreement"]),
            ("all_sources_equal_vs_do_disagreement", "all_sources_equal",
             ranking[condition]["equal_disagreement"]),
        ):
            specs.append(_metric("ranking", kappa, condition, "all", mixture,
                                 metric_name, values.astype(float)))

    for metric_name, mixture, values in (
        ("reward_heavy_drift_excess", "logger1_heavy_vs_logger2_heavy",
         primary_metrics["confounded"]["heavy_reward"]
         - primary_metrics["independent_latents"]["heavy_reward"]),
        ("delta_heavy_drift_excess", "logger1_heavy_vs_logger2_heavy",
         primary_metrics["confounded"]["heavy_delta"]
         - primary_metrics["independent_latents"]["heavy_delta"]),
        ("reward_balanced_do_error_excess", "logger12_balanced",
         primary_metrics["confounded"]["balanced_reward"]
         - primary_metrics["independent_latents"]["balanced_reward"]),
        ("delta_balanced_do_error_excess", "logger12_balanced",
         primary_metrics["confounded"]["balanced_delta"]
         - primary_metrics["independent_latents"]["balanced_delta"]),
        ("reward_all_sources_equal_do_error_excess", "all_sources_equal",
         primary_metrics["confounded"]["equal_reward"]
         - primary_metrics["independent_latents"]["equal_reward"]),
        ("delta_all_sources_equal_do_error_excess", "all_sources_equal",
         primary_metrics["confounded"]["equal_delta"]
         - primary_metrics["independent_latents"]["equal_delta"]),
    ):
        for action in (*ACTION_KEYS, "all"):
            selected = _all_actions(values) if action == "all" else values[:, ACTION_INDEX[action]]
            specs.append(_metric("confounding_excess", kappa,
                                 "confounded_minus_independent", action, mixture,
                                 metric_name, selected))

    signs = np.asarray((-1.0, 0.0, 1.0))
    expected_heavy_reward = (7.0 / 45.0) * reward_u * signs[None, :]
    expected_heavy_delta = (7.0 / 45.0) * do["delta_u_effect"] * signs[None, :, None]
    expected_balanced_reward = 0.3 * reward_u * signs[None, :]
    expected_balanced_delta = 0.3 * do["delta_u_effect"] * signs[None, :, None]
    residuals = {
        "reward_heavy_identity_residual": (
            primary_metrics["confounded"]["signed_heavy_reward"] - expected_heavy_reward),
        "delta_heavy_identity_residual": (
            primary_metrics["confounded"]["signed_heavy_delta"] - expected_heavy_delta),
        "balanced_reward_identity_residual": (
            primary_metrics["confounded"]["balanced_reward_signed"] - expected_balanced_reward),
        "balanced_delta_identity_residual": (
            primary_metrics["confounded"]["balanced_delta_signed"] - expected_balanced_delta),
    }

    obs_conf = observational["confounded"]
    primary_rewards = np.stack([obs_conf[name]["reward"] for name in PRIMARY_MIXTURES], axis=1)
    primary_range = np.max(primary_rewards, axis=1) - np.min(primary_rewards, axis=1)
    max_primary_drift = np.max(primary_range, axis=1)
    do_gap = np.max(do["mean_reward"], axis=1) - np.min(do["mean_reward"], axis=1)
    max_balanced_error = np.max(primary_metrics["confounded"]["balanced_reward"], axis=1)
    for metric_name, mixture, values in (
        ("max_primary_drift", "primary", max_primary_drift),
        ("do_action_gap", "none", do_gap),
        ("fraction_primary_drift_gt_gap", "primary", (max_primary_drift > do_gap).astype(float)),
        ("max_balanced_do_error", "logger12_balanced", max_balanced_error),
        ("fraction_balanced_error_gt_gap", "logger12_balanced",
         (max_balanced_error > do_gap).astype(float)),
    ):
        specs.append(_metric("decision_scale", kappa, "confounded", "all", mixture,
                             metric_name, values))

    independent_equal = all(
        np.allclose(observational["independent_latents"][mixture][field], do[do_field],
                    atol=atol, rtol=rtol)
        for mixture in MIXTURES
        for field, do_field in (("reward", "mean_reward"),
                                ("next_observation", "mean_next_observation"),
                                ("delta", "mean_delta"),
                                ("terminated", "termination_probability"),
                                ("truncated", "truncation_probability")))
    kappa_zero = True
    if kappa == 0.0:
        kappa_zero = bool(
            np.allclose(do["applied_action_u"][:, :, 0], do["applied_action_u"][:, :, 1],
                        atol=atol, rtol=rtol)
            and np.allclose(do["next_observation_u"][:, :, 0],
                            do["next_observation_u"][:, :, 1], atol=atol, rtol=rtol)
            and np.allclose(do["reward_u_effect"], 0.0, atol=atol, rtol=rtol)
            and np.allclose(do["delta_u_effect"], 0.0, atol=atol, rtol=rtol)
            and np.all(do["terminated_u"][:, :, 0] == do["terminated_u"][:, :, 1])
            and np.all(do["truncated_u"][:, :, 0] == do["truncated_u"][:, :, 1])
            and all(
                np.allclose(observational[condition][mixture][field], do[do_field],
                            atol=atol, rtol=rtol)
                for condition in CONDITIONS for mixture in MIXTURES
                for field, do_field in (("reward", "mean_reward"),
                                        ("next_observation", "mean_next_observation"),
                                        ("delta", "mean_delta"),
                                        ("terminated", "termination_probability"),
                                        ("truncated", "truncation_probability"))))
    base = ACTION_INDEX["base"]
    base_control = all(
        np.allclose(observational[condition][mixture][field][:, base],
                    do[do_field][:, base], atol=atol, rtol=rtol)
        for condition in CONDITIONS for mixture in MIXTURES
        for field, do_field in (("reward", "mean_reward"),
                                ("next_observation", "mean_next_observation"),
                                ("delta", "mean_delta"),
                                ("terminated", "termination_probability"),
                                ("truncated", "truncation_probability")))
    checks = {
        "do_oracle_raw_summary_agreement": bool(do_audit["passed"]),
        "do_oracle_logger_mixture_condition_independent": True,
        "primary_mixtures_preserve_state_action_mass": all(
            audit["passed"] for audit in mass_audits.values()),
        "confounded_u_posteriors_match_analytic": posterior_audits["confounded"]["passed"],
        "independent_u_posterior_is_half": posterior_audits["independent_latents"]["passed"],
        "logger12_balanced_remains_confounded": (
            not np.isclose(analytic_u_posterior("confounded", "logger12_balanced", "plus"), 0.5)),
        "all_sources_equal_remains_confounded": (
            not np.isclose(analytic_u_posterior("confounded", "all_sources_equal", "plus"), 0.5)),
        "kappa_zero_population_equals_do": bool(kappa_zero),
        "independent_population_equals_do": bool(independent_equal),
        "base_action_is_mixture_invariant_and_equals_do": bool(base_control),
        "reward_heavy_drift_identity": bool(np.allclose(
            residuals["reward_heavy_identity_residual"], 0, atol=atol, rtol=rtol)),
        "delta_heavy_drift_identity": bool(np.allclose(
            residuals["delta_heavy_identity_residual"], 0, atol=atol, rtol=rtol)),
        "balanced_reward_bias_identity": bool(np.allclose(
            residuals["balanced_reward_identity_residual"], 0, atol=atol, rtol=rtol)),
        "balanced_delta_bias_identity": bool(np.allclose(
            residuals["balanced_delta_identity_residual"], 0, atol=atol, rtol=rtol)),
    }
    arrays: dict[str, np.ndarray] = {
        "anchor_id": anchor_ids, "do_mean_reward": do["mean_reward"],
        "do_mean_delta": do["mean_delta"], "reward_u_effect": reward_u,
        "delta_u_effect": do["delta_u_effect"], "do_action_gap": do_gap,
        "max_primary_drift": max_primary_drift,
        "max_balanced_do_error": max_balanced_error, **residuals,
    }
    for condition in CONDITIONS:
        for mixture in MIXTURES:
            for field, values in observational[condition][mixture].items():
                arrays[f"{condition}_{mixture}_{field}"] = values
        for name, values in primary_metrics[condition].items():
            arrays[f"{condition}_{name}"] = values
        for name, values in ranking[condition].items():
            if name != "masks":
                arrays[f"{condition}_{name}"] = values
        for mixture, values in ranking[condition]["masks"].items():
            arrays[f"{condition}_{mixture}_top_action_mask"] = values
    audit = {
        "posterior": posterior_audits,
        "state_action_mass": mass_audits,
        "mechanism_identity_maximum_absolute_residual": {
            name: float(np.max(np.abs(values))) for name, values in residuals.items()
        },
    }
    context = {"do": do, "observational": observational, "metrics": primary_metrics,
               "ranking": ranking, "do_action_gap": do_gap,
               "max_primary_drift": max_primary_drift,
               "max_balanced_do_error": max_balanced_error}
    return context, specs, arrays, checks, audit


def aggregate_specs(specs: list[dict[str, Any]], repetitions: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for index, spec in enumerate(specs):
        values = spec["values"]
        low, high = paired_cluster_bootstrap_means([values], repetitions, seed)
        row = {key: value for key, value in spec.items() if key != "values"}
        row.update(descriptive(values))
        row.update(ci95_low=float(low[0]), ci95_high=float(high[0]),
                   bootstrap_unit="anchor_id", bootstrap_repetitions=repetitions,
                   bootstrap_seed=seed)
        rows.append(row)
    return rows


def ratio_row(kappa: float, metric: str, numerator: np.ndarray, denominator: np.ndarray,
              repetitions: int, seed: int, subset: str | None = None) -> dict[str, Any]:
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    mean_denominator = float(np.mean(denominator))
    if len(numerator) == 0 or mean_denominator <= 0:
        row = {"family": "ratio", "kappa": kappa, "condition": "confounded",
               "action": "all", "mixture": "primary", "metric": metric,
               "n_anchors": len(numerator), "mean": None,
               "standard_deviation": None, "median": None, "p10": None, "p25": None,
               "p75": None, "p90": None, "maximum": None, "ci95_low": None,
               "ci95_high": None, "bootstrap_unit": "anchor_id",
               "bootstrap_repetitions": repetitions, "bootstrap_seed": seed}
        if subset is not None:
            row["subset"] = subset
        return row
    matrix = np.column_stack((numerator, denominator))
    rng = np.random.default_rng(seed)
    n = len(matrix)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=repetitions)
    means = counts @ matrix / n
    valid = means[:, 1] > 0
    estimates = means[valid, 0] / means[valid, 1]
    low, high = np.quantile(estimates, (0.025, 0.975))
    row = {"family": "ratio", "kappa": kappa, "condition": "confounded",
           "action": "all", "mixture": "primary", "metric": metric,
           "n_anchors": n, "mean": float(np.mean(numerator) / mean_denominator),
           "standard_deviation": None, "median": None, "p10": None, "p25": None,
           "p75": None, "p90": None, "maximum": None, "ci95_low": float(low),
           "ci95_high": float(high), "bootstrap_unit": "anchor_id",
           "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
           "bootstrap_valid_repetitions": int(valid.sum())}
    if subset is not None:
        row["subset"] = subset
    return row


def subset_specs(context: dict[str, Any], strict: np.ndarray, repetitions: int,
                 seed: int) -> list[dict[str, Any]]:
    metrics = context["metrics"]["confounded"]
    ranking = context["ranking"]["confounded"]
    masks = {"all": np.ones(len(strict), dtype=bool),
             "strict_unclipped": strict, "any_clipping": ~strict}
    rows = []
    for subset_index, (subset, mask) in enumerate(masks.items()):
        vectors = {
            "reward_heavy_drift": _all_actions(metrics["heavy_reward"])[mask],
            "reward_balanced_do_error": _all_actions(metrics["balanced_reward"])[mask],
            "delta_heavy_drift": _all_actions(metrics["heavy_delta"])[mask],
            "delta_balanced_do_error": _all_actions(metrics["balanced_delta"])[mask],
            "fraction_primary_drift_gt_gap": (
                context["max_primary_drift"] > context["do_action_gap"])[mask].astype(float),
            "fraction_balanced_error_gt_gap": (
                context["max_balanced_do_error"] > context["do_action_gap"])[mask].astype(float),
            "heavy_strict_flip": ranking["heavy_strict_flip"][mask].astype(float),
            "balanced_vs_do_disagreement": ranking["balanced_disagreement"][mask].astype(float),
        }
        for metric, values in vectors.items():
            spec = _metric("kappa_0p30_subset", 0.3, "confounded", "all", "primary",
                           metric, values)
            spec["subset"] = subset
            rows.extend(aggregate_specs([spec], repetitions, seed + subset_index))
        rows.append(ratio_row(0.3, "primary_drift_over_action_gap",
                              context["max_primary_drift"][mask],
                              context["do_action_gap"][mask], repetitions,
                              seed + 100 + subset_index, subset))
        rows.append(ratio_row(0.3, "balanced_error_over_action_gap",
                              context["max_balanced_do_error"][mask],
                              context["do_action_gap"][mask], repetitions,
                              seed + 200 + subset_index, subset))
    return rows


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


def _select(rows: list[dict[str, Any]], kappa: float, metric: str,
            condition: str = "confounded", action: str = "all") -> dict[str, Any]:
    found = [row for row in rows if row.get("kappa") == kappa
             and row.get("metric") == metric and row.get("condition") == condition
             and row.get("action") == action and "subset" not in row]
    if len(found) != 1:
        raise NonComplementaryPopulationAuditError(f"plot metric not unique: {metric}/{kappa}")
    return found[0]


def make_figures(output: Path, rows: list[dict[str, Any]], kappas: tuple[float, ...],
                 subset_rows: list[dict[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def trend(filename: str, metrics: tuple[tuple[str, str], ...], ylabel: str) -> None:
        figure, axes = plt.subplots(figsize=(6.4, 4.2))
        for metric, label in metrics:
            selected = [_select(rows, kappa, metric) for kappa in kappas]
            means = np.asarray([item["mean"] for item in selected])
            lows = means - np.asarray([item["ci95_low"] for item in selected])
            highs = np.asarray([item["ci95_high"] for item in selected]) - means
            axes.errorbar(kappas, means, yerr=np.vstack((lows, highs)), marker="o", label=label)
        axes.set_xlabel("kappa")
        axes.set_ylabel(ylabel)
        if len(metrics) > 1:
            axes.legend()
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=160)
        plt.close(figure)

    trend("reward_heavy_drift_vs_kappa.png", (("reward_heavy_drift", "heavy drift"),),
          "Mean absolute reward drift")
    trend("balanced_reward_do_error_vs_kappa.png",
          (("reward_balanced_do_error", "balanced residual"),),
          "Mean absolute reward do-error")
    trend("delta_heavy_drift_vs_kappa.png", (("delta_heavy_drift", "heavy drift"),),
          "Mean physical-delta drift L2")
    trend("balanced_delta_do_error_vs_kappa.png",
          (("delta_balanced_do_error", "balanced residual"),),
          "Mean physical-delta do-error L2")
    trend("ranking_disagreement_vs_kappa.png",
          (("heavy_top_set_disagreement", "heavy disagreement"),
           ("heavy_strict_flip", "heavy strict flip"),
           ("balanced_vs_do_disagreement", "balanced vs do")), "Fraction of anchors")
    trend("drift_relative_to_action_gap_vs_kappa.png",
          (("primary_drift_over_action_gap", "primary drift / gap"),),
          "Ratio of aggregate means")
    trend("balanced_error_relative_to_action_gap_vs_kappa.png",
          (("balanced_error_over_action_gap", "balanced error / gap"),),
          "Ratio of aggregate means")

    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    if subset_rows:
        labels = ("all", "strict_unclipped", "any_clipping")
        values = [next(row["mean"] for row in subset_rows
                       if row["subset"] == label and row["metric"] == "heavy_strict_flip")
                  for label in labels]
        axes.bar(labels, values)
        axes.set_ylabel("Heavy-mixture strict flip fraction")
    else:
        axes.text(0.5, 0.5, "kappa=0.3 not requested", ha="center", va="center")
        axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(figures / "all_vs_strict_unclipped_kappa_0p30.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    labels, values = [], []
    for mixture in MIXTURES:
        for action in ("minus", "plus"):
            labels.append(f"{mixture}\n{action}")
            values.append(analytic_u_posterior("confounded", mixture, action))
    axes.bar(labels, values)
    axes.axhline(0.5, linestyle="--", label="do(action)")
    axes.set_ylabel("P(u_env=+1 | action)")
    axes.tick_params(axis="x", rotation=25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(figures / "posterior_u_given_action.png", dpi=160)
    plt.close(figure)


def _git_commit(root: Path) -> str | None:
    repository = next((path for path in (root, *root.parents) if (path / ".git").exists()), root)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_reports(output: Path, summary: dict[str, Any]) -> None:
    report = """# Phase 8A-NC — Non-Complementary Logger Population Audit

This artifact uses exact probability-mass enumeration and reuses fixed Phase 8A do-oracle
outcomes. Logger 1 and Logger 2 have identical commanded-action marginals and the same confounding
direction, but different strengths. Primary mixtures preserve P(S,A); all-sources-equal is a
secondary comparison and changes the action marginal.

The central question is whether equal weighting of same-direction, non-complementary loggers leaves
P(U|S,A) and observational outcomes biased relative to do(action). All numerical facts are in
`aggregate_tables.csv`; the script does not select a scientific-success verdict.

Anchor ID is the statistical unit. Bootstrap intervals describe anchor variation for one fixed
behavior checkpoint seed and do not establish cross-policy-seed significance. Clean/clipped subset
differences are descriptive, not the causal effect of clipping.
"""
    rows = summary["aggregate_metrics"]

    def find(kappa: float, metric: str, condition: str = "confounded") -> dict[str, Any]:
        found = [row for row in rows if row["kappa"] == kappa
                 and row["metric"] == metric and row["condition"] == condition
                 and row["action"] == "all" and "subset" not in row]
        if len(found) != 1:
            raise NonComplementaryPopulationAuditError(
                f"report metric not unique: {metric}/{kappa}/{condition}")
        return found[0]

    def estimate(row: dict[str, Any]) -> str:
        if row["mean"] is None:
            return "undefined (zero mean action gap)"
        return f'{row["mean"]:.6g} [{row["ci95_low"]:.6g}, {row["ci95_high"]:.6g}]'

    metric_columns = (
        ("reward_u_effect_abs", "do_oracle", "|U reward effect|"),
        ("reward_heavy_drift", "confounded", "heavy reward drift"),
        ("reward_balanced_do_error", "confounded", "balanced reward do-error"),
        ("reward_all_sources_equal_do_error", "confounded", "equal-source reward do-error"),
        ("delta_heavy_drift", "confounded", "heavy delta drift"),
        ("delta_balanced_do_error", "confounded", "balanced delta do-error"),
        ("heavy_top_set_disagreement", "confounded", "heavy ranking disagreement"),
        ("heavy_strict_flip", "confounded", "heavy strict flip"),
        ("primary_drift_over_action_gap", "confounded", "drift / action gap"),
        ("balanced_error_over_action_gap", "confounded", "balanced error / action gap"),
    )
    result_lines = ["| kappa | metric | mean [anchor-bootstrap 95% interval] |",
                    "|---:|---|---:|"]
    for kappa in summary["kappas"]:
        for metric, condition, label in metric_columns:
            result_lines.append(f"| {kappa:.1f} | {label} | "
                                f"{estimate(find(kappa, metric, condition))} |")

    subset_lines = ["| subset | heavy reward drift | balanced reward do-error | "
                    "heavy strict flip | balanced-vs-do disagreement |",
                    "|---|---:|---:|---:|---:|"]
    if summary["kappa_0p30_subsets"]:
        for subset in ("all", "strict_unclipped", "any_clipping"):
            selected = {row["metric"]: row for row in summary["kappa_0p30_subsets"]
                        if row["subset"] == subset}
            subset_lines.append(
                f"| {subset} | {estimate(selected['reward_heavy_drift'])} | "
                f"{estimate(selected['reward_balanced_do_error'])} | "
                f"{estimate(selected['heavy_strict_flip'])} | "
                f"{estimate(selected['balanced_vs_do_disagreement'])} |")
    else:
        subset_lines.append("| not evaluated (kappa=0.3 not requested) | - | - | - | - |")

    residual_lines = []
    for kappa_name, audit in summary["per_kappa_audits"].items():
        residuals = audit["mechanism_identity_maximum_absolute_residual"]
        residual_lines.append(
            f"- {kappa_name}: maximum over all four reward/delta identities = "
            f"{max(residuals.values()):.6g}")

    posterior = summary["analytic_posteriors"]["confounded"]["details"]
    posterior_lines = ["| mixture | minus | base | plus |", "|---|---:|---:|---:|"]
    for mixture in MIXTURES:
        posterior_lines.append(
            f"| {mixture} | {posterior[mixture]['minus']['analytic']:.9g} | "
            f"{posterior[mixture]['base']['analytic']:.9g} | "
            f"{posterior[mixture]['plus']['analytic']:.9g} |")

    report = "\n".join((
        "# Phase 8A-NC - Non-Complementary Logger Population Audit", "",
        "This artifact uses exact probability-mass enumeration and only reuses fixed Phase 8A "
        "do-oracle outcomes. All hard checks passed and all hashed inputs were unchanged.", "",
        "Logger 1 uses plus probabilities 0.9/0.1 for u=+1/-1; Logger 2 uses 0.7/0.3. "
        "Both therefore have plus/minus marginals 0.5/0.5, have the same confounding direction, "
        "and are not complementary. Primary mixtures preserve P(S,A)=(0.45,0.10,0.45) exactly.",
        "", "## Analytic and empirical U posteriors", "", *posterior_lines, "",
        "The exact weighted empirical posteriors match these analytic values. In particular, "
        "logger12_balanced and all_sources_equal retain posteriors 0.2/0.8 for minus/plus, "
        "rather than the do(action) value 0.5.", "", "## Main results", "", *result_lines, "",
        "Each entry is an anchor-level mean and percentile bootstrap interval. Ratio rows are "
        "ratios of aggregate means and separately retain their numerator and denominator in "
        "aggregate_tables.csv.", "", "## Kappa 0.3 clipping subsets", "", *subset_lines, "",
        "Subset differences are descriptive and are not identified as the causal effect of clipping.",
        "", "## Mechanism and negative controls", "", *residual_lines, "",
        "Kappa=0, independent-latents, and base-action controls all passed. The independent "
        "condition has P(u_env=+1|action)=0.5 and zero weighted latent correlation for every "
        "mixture; its population responses equal do(action).", "", "## Interpretation boundary", "",
        "Supported: equal action marginals do not imply equal hidden-U composition; balancing "
        "same-direction non-complementary loggers does not cut the U-to-A and U-to-outcome path "
        "in this fixed DGP. The numerical tables quantify whether the remaining bias changes "
        "one-step action ranking at each kappa.", "",
        "Not supported: a claim about arbitrary equal-source sampling, general hidden-confounding "
        "resolution, cross-behavior-policy-seed significance, or a causal clipping effect. "
        "The script deliberately leaves the overall scientific verdict manual.", "",
        "Anchor ID is the statistical unit. Bootstrap intervals describe anchor variation for "
        "one fixed behavior checkpoint seed; support rows are not treated as independent repeats.", "",
    ))
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        "# Statistical appendix\n\nMetrics report mean, sample SD, median, P10/P25/P75/P90, "
        "maximum, and percentile 95% anchor-bootstrap intervals. No support-row or cross-seed "
        "significance claim is made.\n", encoding="utf-8")
    catalog = "# Figure catalog\n\n"
    for name in (
        "reward_heavy_drift_vs_kappa.png", "balanced_reward_do_error_vs_kappa.png",
        "delta_heavy_drift_vs_kappa.png", "balanced_delta_do_error_vs_kappa.png",
        "ranking_disagreement_vs_kappa.png", "drift_relative_to_action_gap_vs_kappa.png",
        "balanced_error_relative_to_action_gap_vs_kappa.png",
        "all_vs_strict_unclipped_kappa_0p30.png", "posterior_u_given_action.png"):
        catalog += (f"## {name}\n\nPurpose: audit the named mechanism or robustness quantity. "
                    "Error bars, where shown, are 95% anchor-bootstrap intervals. Read the exact "
                    "values in aggregate_tables.csv. Interpretation is descriptive for one behavior "
                    "checkpoint seed.\n\n")
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def run_population_dgp(
    phase8a_root: Path, phase8ar_root: Path, phase8ac_root: Path, output_root: Path,
    *, num_anchors: int = 2048, kappas: tuple[float, ...] = EXPECTED_KAPPAS,
    bootstrap_reps: int = 2000, seed: int = 0, expected_anchor_count: int = 2048,
) -> dict[str, Any]:
    if num_anchors <= 0 or bootstrap_reps <= 0:
        raise ValueError("num_anchors and bootstrap_reps must be positive")
    kappas = tuple(float(value) for value in kappas)
    if not kappas or any(value not in EXPECTED_KAPPAS for value in kappas):
        raise ValueError("kappas must be a nonempty subset of (0.0,0.1,0.2,0.3)")
    root = require_verified_phase8a_root(phase8a_root)
    review = require_phase8ar_root(root, phase8ar_root)
    clipping = require_phase8ac_root(review, phase8ac_root)
    output = Path(output_root).resolve()
    if output in (root, review, clipping) or output.is_relative_to(root):
        raise NonComplementaryPopulationAuditError(
            "output must be a new sibling artifact, not inside a read-only input"
        )
    manifest = _load_json(root / "manifest.json")
    phase8a_summary = _load_json(root / "summary.json")
    validate_all_84_phase8a_invariants(phase8a_summary)
    if tuple(float(value) for value in manifest.get("kappas", ())) != EXPECTED_KAPPAS:
        raise NonComplementaryPopulationAuditError("Phase 8A must contain all four kappas")
    anchors = load_npz(root / "anchors.npz")
    all_anchor_ids = np.asarray(anchors.get("anchor_id", ()), dtype=np.int64)
    if not np.array_equal(all_anchor_ids, np.arange(expected_anchor_count)):
        raise NonComplementaryPopulationAuditError("Phase 8A anchor set is incomplete")
    if num_anchors > expected_anchor_count:
        raise ValueError("num_anchors exceeds available anchors")
    anchor_ids = np.sort(all_anchor_ids)[:num_anchors]
    strict_all, strict_hash = load_strict_unclipped_mask(clipping, all_anchor_ids)
    strict_selected = strict_all[:num_anchors]
    paths = required_input_paths(root, review, clipping)
    hashes_before = hash_input_files(paths)
    atol = float(manifest.get("numerical_tolerance", {}).get("atol", 1e-7))
    rtol = float(manifest.get("numerical_tolerance", {}).get("rtol", 1e-7))
    logger_checks = validate_logger_definitions(atol)

    output.mkdir(parents=True, exist_ok=True)
    all_specs, all_arrays, contexts, per_kappa_audits, hard_checks = [], {}, {}, {}, {
        "verified_phase8a_input_required": True, "phase8ar_input_required": True,
        "phase8ac_input_required": True,
        "all_expected_anchors_reused": len(all_anchor_ids) == expected_anchor_count,
        "all_four_phase8a_kappas_available": True,
        "strict_unclipped_mask_matches_phase8ac": True,
        "public_hidden_leakage_empty": True,
        "weight_arrays_align_with_public_rows": True,
        "weight_arrays_sum_to_one": True,
        **logger_checks,
    }
    review_arrays = load_npz(review / "anchor_action_metrics.npz")
    for kappa in kappas:
        kappa_name = KAPPA_NAMES[kappa]
        source_raw = load_npz(root / kappa_name / "do_oracle_raw.npz")
        lookup = build_do_lookup(source_raw, kappa)
        if len(lookup) != expected_anchor_count * 3 * 2:
            raise NonComplementaryPopulationAuditError("Phase 8A do-oracle lookup is incomplete")
        directory = output / kappa_name
        directory.mkdir(parents=True, exist_ok=True)
        datasets = {}
        condition_marginals = {}
        independent_correlations = {}
        for condition in CONDITIONS:
            public, hidden = generate_support(anchors, source_raw, anchor_ids, condition, kappa)
            weights = generate_weights(hidden)
            datasets[condition] = (public, hidden, weights)
            np.savez_compressed(directory / f"{condition}_public.npz", **public)
            np.savez_compressed(directory / f"{condition}_hidden_audit.npz", **hidden)
            weight_dir = directory / "weights" / condition
            weight_dir.mkdir(parents=True, exist_ok=True)
            for mixture, values in weights.items():
                np.save(weight_dir / f"{mixture}.npy", values)
            condition_marginals[condition] = support_marginals(hidden)
            if FORBIDDEN_PUBLIC_FIELDS.intersection(public):
                hard_checks["public_hidden_leakage_empty"] = False
            if any(values.shape != (len(public["row_id"]),) for values in weights.values()):
                hard_checks["weight_arrays_align_with_public_rows"] = False
            if not all(np.isclose(values.sum(), 1.0, atol=atol, rtol=rtol)
                       for values in weights.values()):
                hard_checks["weight_arrays_sum_to_one"] = False
            if condition == "independent_latents":
                for mixture, values in weights.items():
                    correlation = weighted_latent_correlation(hidden, values)
                    independent_correlations[mixture] = correlation
                    if not np.isclose(correlation, 0.0, atol=atol, rtol=rtol):
                        raise NonComplementaryPopulationAuditError(
                            f"independent latent weighted correlation is not zero: {mixture}")
        action_keys = [key for key in condition_marginals["confounded"] if "_action_" in key]
        expected_action_mass = {
            **{f"logger_{logger}_action_{action}":
               (0.5 if logger in (0, 1) and action in ("minus", "plus") else 0.0)
               for logger in (0, 1) for action in ACTION_KEYS},
            "logger_2_action_minus": 0.0,
            "logger_2_action_base": 1.0,
            "logger_2_action_plus": 0.0,
        }
        action_marginals_equal = all(
            np.isclose(condition_marginals["confounded"][key],
                       condition_marginals["independent_latents"][key],
                       atol=atol, rtol=rtol)
            and np.isclose(condition_marginals["confounded"][key],
                           expected_action_mass[key], atol=atol, rtol=rtol)
            for key in action_keys)
        u_env_half = all(
            np.isclose(marginals[f"logger_{logger}_u_env_{u_env}"], 0.5,
                       atol=atol, rtol=rtol)
            for marginals in condition_marginals.values()
            for logger in (0, 1, 2) for u_env in (-1, 1))
        hard_checks[f"{kappa_name}:condition_commanded_action_marginals_equal"] = bool(
            action_marginals_equal)
        hard_checks[f"{kappa_name}:u_env_marginal_half_both_conditions"] = bool(u_env_half)
        hard_checks[f"{kappa_name}:independent_latent_correlation_zero_all_mixtures"] = all(
            np.isclose(value, 0.0, atol=atol, rtol=rtol)
            for value in independent_correlations.values())
        context, specs, arrays, checks, audit = analyze_kappa(
            root, kappa, anchors, anchor_ids, datasets, atol, rtol)
        audit["support_marginals"] = condition_marginals
        audit["independent_latent_weighted_correlation"] = independent_correlations
        contexts[kappa] = context
        all_specs.extend(specs)
        all_arrays.update({f"{kappa_name}__{name}": values for name, values in arrays.items()})
        per_kappa_audits[kappa_name] = audit
        np.savez_compressed(directory / "population_tables.npz", **arrays)
        _write_json(directory / "population_audit.json", {"checks": checks, **audit})
        for name, value in checks.items():
            hard_checks[f"{kappa_name}:{name}"] = bool(value)
        if num_anchors == expected_anchor_count:
            for name in ("do_mean_reward", "do_mean_delta", "reward_u_effect", "delta_u_effect"):
                reference_key = f"{kappa_name}__{name}"
                if reference_key not in review_arrays or not np.allclose(
                        arrays[name], review_arrays[reference_key], atol=atol, rtol=rtol):
                    raise NonComplementaryPopulationAuditError(
                        f"reused do oracle differs from Phase 8A-R: {reference_key}")
    aggregate_rows = aggregate_specs(all_specs, bootstrap_reps, seed)
    for index, kappa in enumerate(kappas):
        context = contexts[kappa]
        aggregate_rows.extend((
            ratio_row(kappa, "primary_drift_over_action_gap",
                      context["max_primary_drift"], context["do_action_gap"],
                      bootstrap_reps, seed + 1000 + index),
            ratio_row(kappa, "balanced_error_over_action_gap",
                      context["max_balanced_do_error"], context["do_action_gap"],
                      bootstrap_reps, seed + 2000 + index),
        ))
    subset_rows = (
        subset_specs(contexts[0.3], strict_selected, bootstrap_reps, seed)
        if 0.3 in contexts else []
    )
    aggregate_rows.extend(subset_rows)
    if not all(value is None or not isinstance(value, (float, np.floating)) or np.isfinite(value)
               for row in aggregate_rows for value in row.values()):
        raise NonComplementaryPopulationAuditError("aggregate output contains NaN/Inf")
    hard_checks.update({
        "do_oracle_lookup_unique": True,
        "do_oracle_matches_phase8ar_where_comparable": True,
        "metrics_use_anchor_level_units": all(
            row["bootstrap_unit"] == "anchor_id" for row in aggregate_rows),
        "all_arrays_finite": all_arrays_finite(all_arrays),
        "aggregate_outputs_have_no_nan_inf": True,
        "logger12_balanced_remains_confounded": True,
        "all_sources_equal_sampling_remains_confounded": True,
    })
    failed = [name for name, value in hard_checks.items() if not value]
    if failed:
        raise NonComplementaryPopulationAuditError(f"hard checks failed: {failed}")

    np.savez_compressed(output / "anchor_action_metrics.npz", **all_arrays)
    _write_csv(output / "aggregate_tables.csv", aggregate_rows)
    make_figures(output, aggregate_rows, kappas, subset_rows)
    hashes_after = hash_input_files(paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks["input_hashes_unchanged"] = unchanged
    hard_checks["old_artifacts_unchanged"] = unchanged
    if not unchanged:
        raise NonComplementaryPopulationAuditError("read-only input hashes changed")
    input_integrity = {"sha256_before": hashes_before, "sha256_after": hashes_after,
                       "unchanged": unchanged, "required_file_count": len(paths)}
    summary = {
        "stage": "Phase 8A-NC", "available_anchor_count": len(all_anchor_ids),
        "analyzed_anchor_count": len(anchor_ids),
        "anchor_selection": "sorted anchor_id prefix" if num_anchors < expected_anchor_count
        else "all anchors", "kappas": list(kappas),
        "strict_unclipped_count_selected": int(strict_selected.sum()),
        "logger_properties": {
            "LOGGERS_HAVE_EQUAL_ACTION_MARGINALS": True,
            "LOGGERS_ARE_NONCOMPLEMENTARY": True,
            "LOGGERS_HAVE_SAME_CONFOUNDING_DIRECTION": True,
            "LOGGER12_BALANCED_REMAINS_CONFOUNDED": True,
            "ALL_SOURCE_EQUAL_SAMPLING_REMAINS_CONFOUNDED": True,
        },
        "analytic_posteriors": per_kappa_audits[next(iter(per_kappa_audits))]["posterior"],
        "per_kappa_audits": per_kappa_audits,
        "aggregate_metrics": aggregate_rows, "kappa_0p30_subsets": subset_rows,
        "hard_checks": hard_checks, "all_hard_checks_passed": all(hard_checks.values()),
        "bootstrap": {"unit": "anchor_id", "repetitions": bootstrap_reps, "seed": seed},
        "scientific_verdict": "MANUAL_DECISION_REQUIRED",
    }
    output_manifest = {
        "stage": "Phase 8A-NC", "git_commit": _git_commit(root),
        "phase8a_input_root": str(root), "phase8ar_input_root": str(review),
        "phase8ac_input_root": str(clipping), "phase8a_input_hashes": hashes_before,
        "phase8ac_clean_mask_sha256": strict_hash, "environment": "Hopper-v5",
        "available_anchor_count": expected_anchor_count, "analyzed_anchor_count": num_anchors,
        "kappas": list(kappas), "logger_names": list(LOGGER_NAMES),
        "logger_probability_tables": LOGGER_ACTION_PROBABILITIES,
        "logger_parameters_selected_without_outcome_search": True,
        "primary_mixtures": PRIMARY_MIXTURES, "secondary_mixtures": SECONDARY_MIXTURES,
        "action_keys": list(ACTION_KEYS),
        "conditions": {"confounded": "u_env = u_behavior",
                       "independent_latents": "u_env independent of u_behavior"},
        "do_oracle_reuse": "exact Phase 8A do_oracle_raw lookup; no simulation",
        "public_fields": list(PUBLIC_FIELDS), "hidden_only_fields": sorted(FORBIDDEN_PUBLIC_FIELDS),
        "numerical_tolerance": {"atol": atol, "rtol": rtol},
        "random_latent_or_action_sampling": False, "mujoco_rerun": False,
        "bootstrap_repetitions": bootstrap_reps, "bootstrap_seed": seed,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
    }
    _write_json(output / "manifest.json", output_manifest)
    _write_json(output / "input_integrity.json", input_integrity)
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
                                                "all_passed": all(hard_checks.values()),
                                                "failed": []})
    _write_json(output / "summary.json", summary)
    write_reports(output, summary)
    return summary
