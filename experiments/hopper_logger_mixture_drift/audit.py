"""Population audit and hard mechanism invariants for Phase 8A."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .controlled_loggers import ACTION_KEYS, MIXTURES, controlled_action
from .generate_datasets import FORBIDDEN_PUBLIC_FIELDS, all_arrays_finite


ATOL = 1e-7
RTOL = 1e-7
POPULATION_TABLE_FIELDS = (
    "anchor_id", "action_key", "mixture", "observational_mean_reward",
    "observational_mean_next_observation", "observational_mean_delta_observation",
    "observational_termination_probability", "do_mean_reward", "do_mean_next_observation",
    "do_mean_delta_observation", "do_termination_probability", "reward_do_error",
    "next_observation_do_error_l2", "delta_observation_do_error_l2",
    "termination_probability_do_error",
)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray | float:
    selected = np.asarray(weights, dtype=np.float64)
    mass = float(selected.sum())
    if mass <= 0.0:
        raise RuntimeError("conditional observational cell has zero mixture mass")
    result = np.tensordot(selected / mass, np.asarray(values, dtype=np.float64), axes=(0, 0))
    return float(result) if np.ndim(result) == 0 else result


def population_observational_table(
    anchors: dict[str, np.ndarray], public: dict[str, np.ndarray], hidden: dict[str, np.ndarray],
    mixture_weights: dict[str, np.ndarray], do_summary: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute the infinite-data weighted regression target cell by cell."""
    oracle_lookup = {(int(anchor), str(key)): index for index, (anchor, key) in enumerate(
        zip(do_summary["anchor_id"], do_summary["action_key"]))}
    rows = []
    for anchor_index, anchor_id in enumerate(anchors["anchor_id"]):
        observation = np.asarray(anchors["public_observation"][anchor_index], dtype=np.float64)
        for action_key in ACTION_KEYS:
            cell = (public["anchor_id"] == anchor_id) & (hidden["action_key"] == action_key)
            oracle = oracle_lookup[(int(anchor_id), action_key)]
            for mixture_name, weights in mixture_weights.items():
                selected_weights = np.asarray(weights[cell], dtype=np.float64)
                mean_reward = _weighted_mean(public["reward"][cell], selected_weights)
                mean_next = _weighted_mean(public["next_observation"][cell], selected_weights)
                mean_delta = mean_next - observation
                mean_terminated = _weighted_mean(public["terminated"][cell], selected_weights)
                do_reward = float(do_summary["do_mean_reward"][oracle])
                do_next = np.asarray(do_summary["do_mean_next_observation"][oracle], dtype=np.float64)
                do_delta = np.asarray(do_summary["do_mean_delta_observation"][oracle], dtype=np.float64)
                do_terminated = float(do_summary["do_termination_probability"][oracle])
                rows.append({
                    "anchor_id": anchor_id, "action_key": action_key, "mixture": mixture_name,
                    "observational_mean_reward": mean_reward,
                    "observational_mean_next_observation": mean_next,
                    "observational_mean_delta_observation": mean_delta,
                    "observational_termination_probability": mean_terminated,
                    "do_mean_reward": do_reward, "do_mean_next_observation": do_next,
                    "do_mean_delta_observation": do_delta,
                    "do_termination_probability": do_terminated,
                    "reward_do_error": mean_reward - do_reward,
                    "next_observation_do_error_l2": float(np.linalg.norm(mean_next - do_next)),
                    "delta_observation_do_error_l2": float(np.linalg.norm(mean_delta - do_delta)),
                    "termination_probability_do_error": mean_terminated - do_terminated,
                })
    return {field: np.asarray([row[field] for row in rows]) for field in POPULATION_TABLE_FIELDS}


def _maximum_pairwise_l2(values: np.ndarray) -> float:
    return max((float(np.linalg.norm(values[left] - values[right]))
                for left, right in combinations(range(len(values)), 2)), default=0.0)


def summarize_population_table(table: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Aggregate only after reducing transition enumeration to anchor-level cells."""
    anchor_action_rows = []
    for anchor_id in np.unique(table["anchor_id"]):
        for action_key in ACTION_KEYS:
            mask = (table["anchor_id"] == anchor_id) & (table["action_key"] == action_key)
            rewards = np.asarray(table["observational_mean_reward"][mask], dtype=np.float64)
            deltas = np.asarray(table["observational_mean_delta_observation"][mask], dtype=np.float64)
            anchor_action_rows.append((
                int(anchor_id), action_key, float(np.max(rewards) - np.min(rewards)),
                _maximum_pairwise_l2(deltas),
                float(np.mean(np.abs(table["reward_do_error"][mask]))),
                float(np.mean(table["delta_observation_do_error_l2"][mask])),
                float(np.mean(np.abs(table["termination_probability_do_error"][mask]))),
            ))
    row_array = np.asarray(anchor_action_rows, dtype=object)
    reward_drift = np.asarray(row_array[:, 2], dtype=np.float64)
    next_drift = np.asarray(row_array[:, 3], dtype=np.float64)
    reward_do_error = np.asarray(row_array[:, 4], dtype=np.float64)
    next_do_error = np.asarray(row_array[:, 5], dtype=np.float64)
    termination_do_error = np.asarray(row_array[:, 6], dtype=np.float64)

    mixture_rankings: dict[tuple[int, str], str] = {}
    do_rankings: dict[int, str] = {}
    for anchor_id in np.unique(table["anchor_id"]):
        do_rewards = []
        for action_key in ACTION_KEYS:
            mask = (table["anchor_id"] == anchor_id) & (table["action_key"] == action_key)
            do_rewards.append(float(np.asarray(table["do_mean_reward"][mask])[0]))
        do_rankings[int(anchor_id)] = ACTION_KEYS[int(np.argmax(do_rewards))]
        for mixture in MIXTURES:
            rewards = []
            for action_key in ACTION_KEYS:
                mask = ((table["anchor_id"] == anchor_id) & (table["action_key"] == action_key)
                        & (table["mixture"] == mixture))
                rewards.append(float(np.asarray(table["observational_mean_reward"][mask])[0]))
            mixture_rankings[(int(anchor_id), mixture)] = ACTION_KEYS[int(np.argmax(rewards))]
    anchors = sorted(do_rankings)
    mixture_flip = np.asarray([
        len({mixture_rankings[(anchor, mixture)] for mixture in MIXTURES}) > 1 for anchor in anchors
    ], dtype=bool)
    oracle_flip = np.asarray([
        any(mixture_rankings[(anchor, mixture)] != do_rankings[anchor] for mixture in MIXTURES)
        for anchor in anchors
    ], dtype=bool)
    summary = {
        "anchor_count": len(anchors), "anchor_action_cell_count": len(anchor_action_rows),
        "reward_mixture_drift": {"mean": float(np.mean(reward_drift)),
                                 "median": float(np.median(reward_drift)),
                                 "maximum": float(np.max(reward_drift))},
        "next_state_delta_mixture_drift_l2": {"mean": float(np.mean(next_drift)),
                                              "median": float(np.median(next_drift)),
                                              "maximum": float(np.max(next_drift))},
        "reward_do_error_absolute": {"mean": float(np.mean(reward_do_error)),
                                     "maximum": float(np.max(reward_do_error))},
        "next_state_delta_do_error_l2": {"mean": float(np.mean(next_do_error)),
                                         "maximum": float(np.max(next_do_error))},
        "termination_probability_do_error_absolute": {
            "mean": float(np.mean(termination_do_error)), "maximum": float(np.max(termination_do_error))},
        "mixture_action_ranking_flip_rate": float(np.mean(mixture_flip)),
        "any_mixture_vs_do_action_ranking_flip_rate": float(np.mean(oracle_flip)),
    }
    arrays = {
        "anchor_action_anchor_id": np.asarray(row_array[:, 0], dtype=np.int64),
        "anchor_action_action_key": np.asarray(row_array[:, 1], dtype=str),
        "reward_mixture_drift": reward_drift, "next_state_delta_mixture_drift_l2": next_drift,
        "reward_do_error_absolute": reward_do_error,
        "next_state_delta_do_error_l2": next_do_error,
        "termination_probability_do_error_absolute": termination_do_error,
        "anchor_mixture_ranking_flip": mixture_flip,
        "anchor_any_mixture_vs_do_ranking_flip": oracle_flip,
    }
    return summary, arrays


def anchor_distribution_audit(
    anchor_ids: np.ndarray, weights_by_mixture: dict[str, np.ndarray],
) -> dict[str, Any]:
    unique = np.unique(anchor_ids)
    target = 1.0 / len(unique)
    deviations = {}
    for name, weights in weights_by_mixture.items():
        masses = np.asarray([weights[anchor_ids == anchor].sum() for anchor in unique])
        deviations[name] = float(np.max(np.abs(masses - target)))
    return {"target_mass_per_anchor": target, "maximum_absolute_deviation": deviations,
            "passed": bool(max(deviations.values()) <= ATOL)}


def latent_weighted_correlation(hidden: dict[str, np.ndarray]) -> float:
    weights = np.asarray(hidden["pair_mass"], dtype=np.float64).copy()
    weights /= weights.sum()
    left = np.asarray(hidden["u_behavior"], dtype=np.float64)
    right = np.asarray(hidden["u_env"], dtype=np.float64)
    covariance = float(weights @ (left * right) - (weights @ left) * (weights @ right))
    variance_left = float(weights @ np.square(left) - (weights @ left) ** 2)
    variance_right = float(weights @ np.square(right) - (weights @ right) ** 2)
    return covariance / np.sqrt(variance_left * variance_right)


def _condition_latent_checks(hidden: dict[str, np.ndarray], condition: str) -> dict[str, bool]:
    if condition == "confounded":
        return {"pairing": bool(np.all(hidden["u_behavior"] == hidden["u_env"])),
                "pair_mass": bool(np.allclose(hidden["pair_mass"], 0.5, atol=ATOL, rtol=RTOL))}
    expected_pairs = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    exact = True
    for anchor in np.unique(hidden["anchor_id"]):
        for logger in (0, 1, 2):
            mask = (hidden["anchor_id"] == anchor) & (hidden["logger_id"] == logger)
            pairs = set(zip(hidden["u_behavior"][mask].tolist(), hidden["u_env"][mask].tolist()))
            exact &= pairs == expected_pairs and np.allclose(hidden["pair_mass"][mask], 0.25,
                                                              atol=ATOL, rtol=RTOL)
    return {"pairing": bool(exact), "pair_mass": bool(exact)}


def _latent_marginal(hidden: dict[str, np.ndarray], field: str, value: int) -> float:
    weights = np.asarray(hidden["pair_mass"], dtype=np.float64)
    return float(weights[np.asarray(hidden[field]) == value].sum() / weights.sum())


def _commanded_marginals_match(
    confounded: dict[str, np.ndarray], independent: dict[str, np.ndarray],
) -> bool:
    for anchor in np.unique(confounded["anchor_id"]):
        for logger in (0, 1, 2):
            for u_behavior in (-1, 1):
                masks = [((bundle["anchor_id"] == anchor) & (bundle["logger_id"] == logger)
                          & (bundle["u_behavior"] == u_behavior))
                         for bundle in (confounded, independent)]
                actions = [np.asarray(bundle["commanded_action"][mask])
                           for bundle, mask in zip((confounded, independent), masks)]
                masses = [float(np.sum(bundle["pair_mass"][mask]))
                          for bundle, mask in zip((confounded, independent), masks)]
                if not np.isclose(masses[0], masses[1], atol=ATOL, rtol=RTOL):
                    return False
                if not np.allclose(actions[0][0], actions[1][0], atol=ATOL, rtol=RTOL):
                    return False
    return True


def _outcome_invariant_across_conditions(
    confounded: dict[str, np.ndarray], independent: dict[str, np.ndarray],
) -> bool:
    fields = ("applied_action", "reward", "next_qpos", "next_qvel", "terminated", "truncated")
    for anchor in np.unique(confounded["anchor_id"]):
        for action_key in ACTION_KEYS:
            for u_env in (-1, 1):
                values = []
                for bundle in (confounded, independent):
                    mask = ((bundle["anchor_id"] == anchor) & (bundle["action_key"] == action_key)
                            & (bundle["u_env"] == u_env))
                    values.extend([{field: bundle[field][index] for field in fields}
                                   for index in np.flatnonzero(mask)])
                reference = values[0]
                for value in values[1:]:
                    for field in fields:
                        if not np.allclose(value[field], reference[field], atol=ATOL, rtol=RTOL):
                            return False
    return True


def _kappa_zero_has_no_u_effect(do_raw: dict[str, np.ndarray]) -> bool:
    for anchor in np.unique(do_raw["anchor_id"]):
        for action_key in ACTION_KEYS:
            mask = (do_raw["anchor_id"] == anchor) & (do_raw["action_key"] == action_key)
            indices = np.flatnonzero(mask)
            for field in ("applied_action", "reward", "next_observation", "terminated", "truncated"):
                if not np.allclose(do_raw[field][indices[0]], do_raw[field][indices[1]],
                                   atol=ATOL, rtol=RTOL):
                    return False
    return True


def _population_equals_do(table: dict[str, np.ndarray]) -> bool:
    return bool(
        np.allclose(table["reward_do_error"], 0.0, atol=ATOL, rtol=RTOL)
        and np.allclose(table["next_observation_do_error_l2"], 0.0, atol=ATOL, rtol=RTOL)
        and np.allclose(table["termination_probability_do_error"], 0.0, atol=ATOL, rtol=RTOL)
    )


def _time_to_go_consistent(
    anchors: dict[str, np.ndarray], datasets: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> bool:
    expected = np.clip((1000 - anchors["elapsed_steps"]) / 1000.0, 0.0, 1.0)
    expected_next = np.clip((1000 - anchors["elapsed_steps"] - 1) / 1000.0, 0.0, 1.0)
    lookup = {int(anchor): index for index, anchor in enumerate(anchors["anchor_id"])}
    for public, _ in datasets.values():
        indices = np.asarray([lookup[int(anchor)] for anchor in public["anchor_id"]])
        if not np.allclose(public["observation"][:, -1], expected[indices], atol=ATOL, rtol=RTOL):
            return False
        if not np.allclose(public["next_observation"][:, -1], expected_next[indices],
                           atol=ATOL, rtol=RTOL):
            return False
    return True


def hard_invariants(
    anchors: dict[str, np.ndarray], datasets: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    weights: dict[str, dict[str, np.ndarray]], tables: dict[str, dict[str, np.ndarray]],
    do_raw: dict[str, np.ndarray], kappa_env: float, behavior_offset: float,
    checkpoint_roundtrip_passed: bool, deterministic_restore_passed: bool,
) -> dict[str, bool]:
    public_conf, hidden_conf = datasets["confounded"]
    public_ind, hidden_ind = datasets["independent_latents"]
    anchor_set = set(anchors["anchor_id"].tolist())
    same_anchor_set = all(
        set(public["anchor_id"][public["logger_id"] == logger].tolist()) == anchor_set
        for public, _ in datasets.values() for logger in (0, 1, 2)
    )
    logger_overlap = []
    for base in anchors["base_action"]:
        l1_plus, _ = controlled_action(base, 0, 1, behavior_offset)
        l2_minus, _ = controlled_action(base, 1, -1, behavior_offset)
        l1_minus, _ = controlled_action(base, 0, -1, behavior_offset)
        l2_plus, _ = controlled_action(base, 1, 1, behavior_offset)
        l3_minus, _ = controlled_action(base, 2, -1, behavior_offset)
        l3_plus, _ = controlled_action(base, 2, 1, behavior_offset)
        logger_overlap.append((np.allclose(l1_plus, l2_minus, atol=ATOL, rtol=RTOL),
                               np.allclose(l1_minus, l2_plus, atol=ATOL, rtol=RTOL),
                               np.allclose(l3_minus, l3_plus, atol=ATOL, rtol=RTOL)))
    conf_checks = _condition_latent_checks(hidden_conf, "confounded")
    ind_checks = _condition_latent_checks(hidden_ind, "independent_latents")
    anchor_weight_pass = all(anchor_distribution_audit(public["anchor_id"], weights[condition])["passed"]
                             for condition, (public, _) in datasets.items())
    independent_correlation = latent_weighted_correlation(hidden_ind)
    invariants = {
        "same_anchor_ids_all_kappa_condition_logger": same_anchor_set,
        "public_observation_dimension_12": all(public["observation"].shape[1] == 12
                                                for public, _ in datasets.values()),
        "commanded_action_dimension_3": all(public["action"].shape[1] == 3
                                             for public, _ in datasets.values()),
        "logger1_plus_equals_logger2_minus": all(item[0] for item in logger_overlap),
        "logger1_minus_equals_logger2_plus": all(item[1] for item in logger_overlap),
        "logger3_action_independent_of_u_behavior": all(item[2] for item in logger_overlap),
        "confounded_u_behavior_equals_u_env": conf_checks["pairing"],
        "independent_four_pairs_mass_exact": ind_checks["pairing"] and ind_checks["pair_mass"],
        "commanded_action_marginals_match_conditions": _commanded_marginals_match(hidden_conf, hidden_ind),
        "u_env_marginal_half_both_conditions": all(
            np.isclose(_latent_marginal(bundle, "u_env", value), 0.5, atol=ATOL, rtol=RTOL)
            for bundle in (hidden_conf, hidden_ind) for value in (-1, 1)),
        "independent_latent_weighted_correlation_zero": bool(abs(independent_correlation) <= ATOL),
        "condition_label_does_not_change_environment_outcome": _outcome_invariant_across_conditions(
            hidden_conf, hidden_ind),
        "kappa_zero_removes_u_env_effect": True if kappa_env != 0.0 else _kappa_zero_has_no_u_effect(do_raw),
        "independent_population_equals_do_oracle": _population_equals_do(tables["independent_latents"]),
        "confounded_kappa_zero_equals_do_oracle": (
            True if kappa_env != 0.0 else _population_equals_do(tables["confounded"])),
        "public_hidden_leakage_empty": all(not FORBIDDEN_PUBLIC_FIELDS.intersection(public)
                                           for public, _ in datasets.values()),
        "source2_checkpoint_roundtrip": bool(checkpoint_roundtrip_passed),
        "all_arrays_finite": all_arrays_finite(
            anchors, public_conf, hidden_conf, public_ind, hidden_ind, do_raw,
            tables["confounded"], tables["independent_latents"],
            weights["confounded"], weights["independent_latents"]),
        "anchor_restore_deterministic": bool(deterministic_restore_passed),
        "mixture_weights_keep_anchor_distribution_fixed": anchor_weight_pass,
        "time_to_go_consistent_with_existing_wrapper": _time_to_go_consistent(anchors, datasets),
    }
    return {name: bool(value) for name, value in invariants.items()}


def logger_sensitivity(anchors: dict[str, np.ndarray], behavior_offset: float) -> dict[str, float]:
    result = {}
    for logger_id, name in enumerate(("diagnostic_logger_1", "diagnostic_logger_2", "diagnostic_logger_3")):
        differences = []
        for base in anchors["base_action"]:
            minus, _ = controlled_action(base, logger_id, -1, behavior_offset)
            plus, _ = controlled_action(base, logger_id, 1, behavior_offset)
            differences.append(np.linalg.norm(plus - minus))
        result[name] = float(np.mean(differences))
    return result


def outcome_strength(do_raw: dict[str, np.ndarray]) -> dict[str, float]:
    reward, next_state, termination = [], [], []
    for anchor in np.unique(do_raw["anchor_id"]):
        for action_key in ACTION_KEYS:
            mask = (do_raw["anchor_id"] == anchor) & (do_raw["action_key"] == action_key)
            indices = np.flatnonzero(mask)
            reward.append(abs(float(do_raw["reward"][indices[1]] - do_raw["reward"][indices[0]])))
            next_state.append(float(np.linalg.norm(
                do_raw["next_observation"][indices[1]] - do_raw["next_observation"][indices[0]])))
            termination.append(bool(do_raw["terminated"][indices[1]])
                               != bool(do_raw["terminated"][indices[0]]))
    return {"mean_absolute_reward_u_difference": float(np.mean(reward)),
            "mean_next_observation_u_difference_l2": float(np.mean(next_state)),
            "termination_disagreement_rate": float(np.mean(termination))}


def make_figures(output: Path, kappa_summaries: dict[str, dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(kappa_summaries.items(), key=lambda item: float(item[1]["kappa_env"]))
    kappas = np.asarray([item[1]["kappa_env"] for item in ordered])
    plt.figure(figsize=(7, 4.5))
    mixture_names = tuple(MIXTURES)
    mixture_positions = np.arange(len(mixture_names))
    for _, item in ordered:
        for condition in ("confounded", "independent_latents"):
            means = [item["mixture_mean_reward"][condition][mixture] for mixture in mixture_names]
            plt.plot(mixture_positions, means, label=f"kappa={item['kappa_env']:.2f}:{condition}")
    plt.xticks(mixture_positions, mixture_names, rotation=15)
    plt.xlabel("Logger mixture"); plt.ylabel("Anchor/action mean observational reward")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(output / "reward_prediction_vs_mixture.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 4.5))
    for condition in ("confounded", "independent_latents"):
        values = [item[1]["population"][condition]["next_state_delta_mixture_drift_l2"]["mean"]
                  for item in ordered]
        plt.plot(kappas, values, label=condition)
    plt.xlabel("kappa_env"); plt.ylabel("Mean next-state delta mixture drift L2")
    plt.legend(); plt.tight_layout(); plt.savefig(output / "next_state_drift_vs_kappa.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 4.5))
    for condition in ("confounded", "independent_latents"):
        values = [item[1]["population"][condition]["reward_do_error_absolute"]["mean"]
                  for item in ordered]
        plt.plot(kappas, values, label=condition)
    plt.xlabel("kappa_env"); plt.ylabel("Mean absolute reward do-error")
    plt.legend(); plt.tight_layout(); plt.savefig(output / "do_error_vs_kappa.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 4.5))
    for condition in ("confounded", "independent_latents"):
        values = [item[1]["population"][condition]["mixture_action_ranking_flip_rate"]
                  for item in ordered]
        plt.plot(kappas, values, label=condition)
    plt.xlabel("kappa_env"); plt.ylabel("Anchor action-ranking flip rate")
    plt.legend(); plt.tight_layout()
    plt.savefig(output / "action_ranking_flip_vs_kappa.png", dpi=300); plt.close()
