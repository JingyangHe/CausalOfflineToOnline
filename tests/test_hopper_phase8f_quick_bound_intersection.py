from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.hopper_logger_mixture_drift.phase8f_quick_bound_intersection import (
    FIXED_SOURCE_SETTINGS,
    augmented_canonical_rewards,
    best_single_interval,
    certified_actions,
    direct_pooled_joint_mass,
    interval_intersection,
    natural_reward_bounds,
    observational_population,
    original_opposite_probabilities,
    pool_observational_mass,
    source_probability_tables,
)


def synthetic_rewards(anchor_count: int = 7) -> np.ndarray:
    center = 1.0 + np.arange(anchor_count)[:, None] * 0.01 + np.array((0.0, 0.1, 0.2))
    effect = np.broadcast_to(np.array((0.05, -0.02, 0.08)), center.shape)
    return np.stack((center - effect, center + effect), axis=2)


def test_source_probability_tables_are_the_fixed_four_settings():
    tables = source_probability_tables()
    assert tuple(tables) == FIXED_SOURCE_SETTINGS
    assert tables["M2_same_direction_diverse"].shape == (2, 2, 3)
    assert tables["M5_same_direction_diverse"].shape == (5, 2, 3)
    assert tables["M5_redundant"].shape == (5, 2, 3)
    opposite = original_opposite_probabilities()
    assert np.array_equal(opposite[0, 0], (1.0, 0.0, 0.0))
    assert np.array_equal(opposite[0, 1], (0.0, 0.0, 1.0))
    assert np.array_equal(opposite[1], opposite[0, ::-1])


def test_observational_population_has_no_do_mean_input_and_uses_joint_mass():
    assert "do_reward" not in inspect.signature(observational_population).parameters
    rewards = augmented_canonical_rewards(synthetic_rewards(), 0.05)
    behavior = source_probability_tables()["M2_same_direction_diverse"]
    observed = observational_population(behavior, rewards, "confounded")
    expected_pi = 0.5 * behavior.sum(axis=1)
    expected_mass = sum(
        0.5 * behavior[:, u, :][:, None, :] * rewards[None, :, :, u]
        for u in range(2)
    )
    assert np.allclose(observed["pi"], expected_pi)
    assert np.allclose(observed["reward_mass"], expected_mass)


def test_natural_bounds_cover_do_and_intersection_tightens_components():
    rewards = synthetic_rewards()
    behavior = source_probability_tables()["M5_same_direction_diverse"]
    observed = observational_population(behavior, rewards, "confounded")
    lower, upper = natural_reward_bounds(observed, rewards.min(), rewards.max())
    cap_lower, cap_upper = interval_intersection(lower, upper)
    do = rewards.mean(axis=2)
    assert np.all((do >= lower - 1e-10) & (do <= upper + 1e-10))
    assert np.all((do >= cap_lower - 1e-10) & (do <= cap_upper + 1e-10))
    assert np.all(cap_upper - cap_lower <= upper - lower + 1e-10)


def test_redundant_and_duplicated_sources_do_not_change_intersection():
    rewards = synthetic_rewards()
    behavior = source_probability_tables()["M5_redundant"]
    observed = observational_population(behavior, rewards, "confounded")
    lower, upper = natural_reward_bounds(observed, rewards.min(), rewards.max())
    cap = interval_intersection(lower, upper)
    duplicate = interval_intersection(
        np.concatenate((lower, lower[:1])), np.concatenate((upper, upper[:1]))
    )
    assert np.allclose(cap[0], lower[0])
    assert np.allclose(cap[1], upper[0])
    assert np.allclose(duplicate[0], cap[0])
    assert np.allclose(duplicate[1], cap[1])


def test_pooled_bound_uses_joint_mass_not_mean_of_conditionals():
    rewards = synthetic_rewards()
    behavior = source_probability_tables()["M2_same_direction_diverse"]
    observed = observational_population(behavior, rewards, "confounded")
    weights = np.asarray((0.8, 0.2))
    pooled = pool_observational_mass(observed, weights)
    direct_pi, direct_mass = direct_pooled_joint_mass(
        behavior, rewards, "confounded", weights
    )
    assert np.allclose(pooled["pi"][0], direct_pi)
    assert np.allclose(pooled["reward_mass"][0], direct_mass)
    assert np.allclose(pooled["mu"][0] * direct_pi[None, :], direct_mass)


def test_independent_latents_is_exact_negative_control():
    rewards = synthetic_rewards()
    behavior = source_probability_tables()["M5_same_direction_diverse"]
    observed = observational_population(behavior, rewards, "independent_latents")
    do = rewards.mean(axis=2)
    supported = np.broadcast_to(observed["supported"][:, None, :], observed["mu"].shape)
    expected = np.broadcast_to(do[None, :, :], observed["mu"].shape)
    assert np.allclose(observed["mu"][supported], expected[supported])


def test_certification_cannot_be_false_when_valid_bounds_separate_actions():
    lower = np.asarray(((0.0, 0.2, 0.8), (0.0, 0.4, 0.3)))
    upper = np.asarray(((0.1, 0.3, 0.9), (0.2, 0.5, 0.6)))
    certified = certified_actions(lower, upper)
    assert np.array_equal(certified, (2, -1))
    truth = np.asarray(((0.05, 0.25, 0.85), (0.1, 0.45, 0.5)))
    assert np.argmax(truth[0]) == certified[0]


def test_best_single_interval_selects_one_component_per_cell():
    lower = np.asarray(([[0.0, 0.1]], [[0.2, 0.0]]))
    upper = np.asarray(([[0.7, 0.5]], [[0.5, 0.8]]))
    best_lower, best_upper, index = best_single_interval(lower, upper)
    assert np.array_equal(index, [[1, 0]])
    assert np.allclose(best_lower, [[0.2, 0.1]])
    assert np.allclose(best_upper, [[0.5, 0.5]])


def test_cli_exposes_exact_phase8f_inputs():
    result = subprocess.run(
        [sys.executable, "scripts/run_hopper_phase8f_quick_bound_intersection.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--phase8a-root" in result.stdout
    assert "--phase8anc-root" in result.stdout
    assert "--phase8eq-root" in result.stdout
    assert "--source-settings" in result.stdout
