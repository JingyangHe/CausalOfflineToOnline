import inspect

import numpy as np

from fixed_state_online_calibration import run_fixed_budget_online_evaluation
from scripts.validate_phase4b_fixed_budget_efficiency import calculate_regret_metrics


ACTIONS = np.array((-1.0, 0.0, 1.0))


def run(initial_lower=None, initial_upper=None, budgets=(0, 2, 5), seed=4, record=True):
    lower = [0.0, 0.0, 0.0] if initial_lower is None else initial_lower
    upper = [1.0, 1.0, 1.0] if initial_upper is None else initial_upper
    return run_fixed_budget_online_evaluation(
        ACTIONS, lower, upper, lambda action: 0.5, 0.0, 1.0, 0.025,
        budgets, seed, record_actions=record,
    )


def test_zero_budget_has_no_online_sample():
    result = run(budgets=(0,))
    assert result["total_interactions"] == 0
    assert len(result["sampled_action_indices"]) == 0
    np.testing.assert_array_equal(result["counts_at_checkpoints"], [[0, 0, 0]])


def test_exact_budget_and_all_checkpoints_are_recorded():
    result = run()
    assert result["total_interactions"] == 5
    assert len(result["sampled_action_indices"]) == 5
    np.testing.assert_array_equal(result["checkpoint_budgets"], [0, 2, 5])
    np.testing.assert_array_equal(result["counts_at_checkpoints"].sum(axis=1), [0, 2, 5])


def test_certification_commits_all_remaining_actions():
    result = run([0.8, 0.0, 0.1], [1.0, 0.7, 0.6])
    assert result["certification_time"] == 0
    assert result["committed_action_index"] == 0
    np.testing.assert_array_equal(result["sampled_action_indices"], np.zeros(5, dtype=int))


def test_uncertified_first_sample_uses_best_challenger_rule():
    result = run([0.4, 0.3, 0.0], [0.9, 1.0, 0.2], budgets=(0, 1))
    assert result["sampled_action_indices"].tolist() == [1]
    assert result["sampled_action_indices"][0] in (0, 1)


def test_fixed_seed_is_reproducible():
    first, second = run(seed=88), run(seed=88)
    for key in ("sampled_action_indices", "counts_at_checkpoints", "lower_at_checkpoints", "upper_at_checkpoints"):
        np.testing.assert_array_equal(first[key], second[key])


def test_per_action_random_stream_prefixes_are_shared_across_methods():
    def execute(lower):
        streams = [np.random.default_rng(child) for child in np.random.SeedSequence(19).spawn(3)]
        observed = [[] for _ in ACTIONS]
        def sample(action):
            index = int(np.flatnonzero(ACTIONS == action)[0]); value = streams[index].uniform()
            observed[index].append(value); return value
        run_fixed_budget_online_evaluation(
            ACTIONS, lower, [1.0] * 3, sample, 0.0, 1.0, 0.025, (0, 30), 7
        )
        return observed
    left, right = execute([0.0, 0.0, 0.0]), execute([0.2, 0.0, 0.0])
    for first, second in zip(left, right):
        common = min(len(first), len(second))
        np.testing.assert_array_equal(first[:common], second[:common])


def test_pseudo_regret_metrics_are_correct_and_monotone():
    cumulative, simple = calculate_regret_metrics(
        [0.2, 0.8, 0.5], [0, 1, 2, 1], [0, 1, 2], [0, 2, 4]
    )
    np.testing.assert_allclose(cumulative, [0.0, 0.6, 0.9])
    np.testing.assert_allclose(simple, [0.6, 0.0, 0.3])
    assert np.all(np.diff(cumulative) >= 0.0)


def test_recommendation_uses_lower_bound_best_before_certification():
    result = run([0.1, 0.4, 0.2], [1.0, 1.0, 1.0], budgets=(0,))
    assert result["recommended_action_indices"].tolist() == [1]


def test_fixed_budget_interface_has_no_oracle_or_hidden_inputs():
    parameters = set(inspect.signature(run_fixed_budget_online_evaluation).parameters)
    assert parameters.isdisjoint(
        {"oracle_q", "oracle_best_action", "oracle_gap", "c", "w", "true_reward_expectation"}
    )
