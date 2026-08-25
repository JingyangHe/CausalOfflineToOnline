import inspect

import numpy as np
import pytest

from fixed_state_online_calibration import anytime_hoeffding_radius
from scripts.diagnose_phase4a_decision_bottleneck import (
    decision_deficit,
    equal_count_hoeffding_benchmark,
    first_divergence,
    oracle_geometry,
    run_diagnostics,
)


def test_oracle_ranking_and_top_two_gap():
    result = oracle_geometry([-1.0, 0.0, 1.0], [0.2, 0.8, 0.5])
    np.testing.assert_array_equal(result["ranking"], [1, 2, 0])
    assert result["best"] == 1 and result["second"] == 2
    assert result["gap"] == pytest.approx(0.3)


def test_decision_deficit_formula():
    assert decision_deficit([0.2, 0.6, 0.1], [0.5, 0.8, 0.9], 1) == pytest.approx(0.3)


def test_critical_gain_formula():
    separate = decision_deficit([0.2, 0.4], [0.8, 0.9], 1)
    joint = decision_deficit([0.3, 0.5], [0.7, 0.8], 1)
    assert separate - joint == pytest.approx(0.2)


def test_path_identity_and_first_divergence():
    assert first_divergence([0, 1, 1], [0, 1, 1]) == -1
    assert first_divergence([0, 1, 1], [0, 2, 1]) == 2
    assert first_divergence([0, 1], [0, 1, 2]) == 3


def test_hoeffding_benchmark_is_minimal():
    gap, value_range, delta, actions = 0.08, 1.0, 0.025, 5
    n_equal = equal_count_hoeffding_benchmark(gap, value_range, delta, actions)
    assert 2 * anytime_hoeffding_radius(n_equal, value_range, delta, actions) <= gap
    assert 2 * anytime_hoeffding_radius(n_equal - 1, value_range, delta, actions) > gap


def test_diagnostic_entry_point_has_no_hidden_inputs():
    parameters = set(inspect.signature(run_diagnostics).parameters)
    assert parameters == {"reference"}
    assert parameters.isdisjoint({"c", "w", "true_q", "oracle_optimal_action"})
