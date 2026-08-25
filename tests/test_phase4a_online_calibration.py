import inspect

import numpy as np
import pytest

import fixed_state_online_calibration as calibration
from confounded_smooth_regulator import return_bounds
from finite_sample_joint_bound import evaluate_finite_sample_query
from fixed_state_online_calibration import (
    anytime_hoeffding_radius,
    intersect_offline_online_interval,
    run_fixed_state_online_calibration,
    update_online_interval,
)
from generate_offline_dataset import (
    generate_fixed_state_dataset,
    sample_fixed_state_online_intervention,
)
from oracle_ground_truth import oracle_q, solve_oracle


def test_unsampled_online_interval_is_global_range():
    assert update_online_interval(0, np.nan, 0.0, 4.0, 0.025, 5) == (0.0, 4.0)


def test_anytime_radius_decreases_with_count():
    radii = [anytime_hoeffding_radius(n, 3.0, 0.025, 5) for n in (1, 10, 100)]
    assert radii[0] > radii[1] > radii[2]


def test_offline_online_intersection_and_conflict_fallback():
    assert intersect_offline_online_interval(0.2, 0.8, 0.5, 0.9, 0.0, 1.0) == (
        0.5, 0.8, False
    )
    assert intersect_offline_online_interval(0.0, 0.2, 0.7, 0.9, 0.0, 1.0) == (
        0.7, 0.9, True
    )


def test_initial_certification_stops_without_sampling():
    calls = []
    result = run_fixed_state_online_calibration(
        [-1.0, 0.0, 1.0], [0.8, 0.0, 0.1], [1.0, 0.7, 0.6],
        lambda action: calls.append(action), 0.0, 1.0, 0.025, 20, 3,
    )
    assert result["certified"] and result["certified_action_index"] == 0
    assert result["online_interactions"] == 0 and calls == []


def test_sampling_stays_in_best_challenger_pair():
    result = run_fixed_state_online_calibration(
        [-1.0, 0.0, 1.0], [0.4, 0.3, 0.0], [0.9, 1.0, 0.2],
        lambda action: 0.5, 0.0, 1.0, 0.025, 1, 9, record_history=True,
    )
    assert result["sampled_action_indices"].tolist() == [1]
    assert result["sampled_action_indices"][0] in (0, 1)
    assert result["sampled_action_indices"][0] != 2


def test_conflict_is_recorded_and_offline_constraint_is_discarded(monkeypatch):
    monkeypatch.setattr(calibration, "_radius_unchecked", lambda *args: 0.0)
    result = run_fixed_state_online_calibration(
        [0.0, 1.0], [0.0, 0.5], [0.6, 1.0], lambda action: 0.8,
        0.0, 1.0, 0.9, 1, 2, record_history=True,
    )
    assert result["conflict_count"] == 1
    assert result["conflict_action_indices"].tolist() == [0]
    conflict_time = result["conflict_interactions"][0]
    assert conflict_time == 1
    assert not result["history"][conflict_time]["offline_active"][0]


def test_fixed_seed_is_exactly_reproducible():
    def run():
        streams = [np.random.default_rng(seed) for seed in (11, 12, 13)]
        lookup = {-1.0: 0, 0.0: 1, 1.0: 2}
        return run_fixed_state_online_calibration(
            [-1.0, 0.0, 1.0], [0.0] * 3, [1.0] * 3,
            lambda action: streams[lookup[action]].uniform(), 0.0, 1.0, 0.025, 30, 77,
        )
    first, second = run(), run()
    for key in ("counts", "sampled_action_indices", "final_lower", "final_upper"):
        np.testing.assert_array_equal(first[key], second[key])


def test_online_sampler_matches_oracle_mean_and_reveals_only_scalar():
    reference = solve_oracle(horizon=2, n_state=101, n_action=101)
    rng = np.random.default_rng(123)
    samples = np.array([
        sample_fixed_state_online_intervention(1, 0.2, -0.3, reference, rng)
        for _ in range(20000)
    ])
    assert samples.mean() == pytest.approx(oracle_q(reference, 1, 0.2, -0.3), abs=0.02)
    assert samples.ndim == 1


def test_public_calibrator_interface_has_no_hidden_or_oracle_inputs():
    parameters = set(inspect.signature(run_fixed_state_online_calibration).parameters)
    assert parameters.isdisjoint({"true_q", "c", "w", "oracle_optimal_action"})
    result = run_fixed_state_online_calibration(
        [0.0], [0.0], [1.0], lambda action: 0.5, 0.0, 1.0, 0.025, 1, 1
    )
    assert not any(key.lower() in {"c", "w", "true_q"} for key in result)


def test_oracle_optimum_survives_when_all_intervals_cover_truth():
    actions = np.array([-1.0, 0.0, 1.0])
    truth = np.array([0.4, 0.8, 0.6])
    result = run_fixed_state_online_calibration(
        actions, truth - 0.1, truth + 0.1, lambda action: truth[np.where(actions == action)[0][0]],
        0.0, 1.0, 0.025, 20, 5, record_history=True,
    )
    optimum = int(np.argmax(truth))
    assert all(
        optimum in snapshot["admissible_action_indices"] for snapshot in result["history"]
    )


def test_phase3a_joint_initialization_is_not_wider_than_separate():
    reference = solve_oracle(horizon=3, n_state=101, n_action=101)
    train, _ = generate_fixed_state_dataset([2], [0.0], 100, 801, horizon=3)
    for action in (-1.0, 0.0, 1.0):
        result = evaluate_finite_sample_query(train, reference, 2, 0.0, action, delta=0.025)
        assert result["finite_joint_lower"] >= result["finite_separate_lower"] - 1e-8
        assert result["finite_joint_upper"] <= result["finite_separate_upper"] + 1e-8


def test_budget_exhaustion_does_not_fake_certification():
    lower, upper = return_bounds(1, 3, 0.95)
    result = run_fixed_state_online_calibration(
        [-1.0, 0.0], [lower, lower], [upper, upper], lambda action: 0.5,
        lower, upper, 0.025, 0, 4,
    )
    assert not result["certified"]
    assert result["certified_action"] is None
