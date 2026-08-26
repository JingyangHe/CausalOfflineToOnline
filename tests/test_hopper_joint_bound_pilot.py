"""Focused tests for the Phase 7C local empirical Joint-bound pilot."""

import numpy as np
import pytest

from exact_population_joint_bound import (
    enumerate_feasible_tuples,
    prepare_empirical_coupling_problem,
    solve_empirical_joint_interval,
)
from hopper_joint_bound_pilot import (
    LocalEmpiricalConditioner,
    compute_bellman_outcomes,
    dual_baselines,
    evenly_spaced_indices,
    exact_violation_correction,
    fit_train_normalization,
    normalize_outcomes,
    normalize_states,
    solve_local_problem,
    source3_contribution,
    target_actions,
    validate_public_data,
    zero_initialize_output_layer,
)


def _public_data(per_source=5):
    count = 3 * per_source
    source = np.repeat((1, 2, 3), per_source)
    observations = np.arange(count * 12, dtype=np.float64).reshape(count, 12) / 20
    return {
        "observations": observations,
        "actions": np.linspace(-0.8, 0.8, count * 3).reshape(count, 3),
        "rewards": np.linspace(-1, 2, count),
        "next_observations": observations + 0.1,
        "terminated": np.zeros(count, dtype=bool),
        "truncated": np.zeros(count, dtype=bool),
        "collector_truncated": np.zeros(count, dtype=bool),
        "source_id": source,
    }


def _atoms(offset=0.0):
    return {
        "actions": np.asarray(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        "outcomes": np.asarray((-0.5 + offset, 0.5 + offset)),
        "probabilities": np.asarray((0.5, 0.5)),
    }


def test_hidden_fields_are_rejected():
    data = _public_data()
    data["qpos"] = np.zeros((len(data["rewards"]), 6))
    with pytest.raises(ValueError, match="hidden fields"):
        validate_public_data(data)


def test_collector_truncation_does_not_close_bootstrap():
    data = _public_data(1)
    data["collector_truncated"][0] = True
    outcomes = compute_bellman_outcomes(data, 0.5, 0.0, 1.0, lambda states: np.full(len(states), 4.0))
    assert outcomes[0] == pytest.approx(data["rewards"][0] / 1.0000001 + 2.0)


def test_terminated_and_truncated_close_bootstrap():
    data = _public_data(1)
    data["terminated"][0], data["truncated"][1] = True, True
    outcomes = compute_bellman_outcomes(data, 0.9, 0.0, 1.0, lambda states: np.full(len(states), 9.0))
    assert outcomes[0] == pytest.approx(data["rewards"][0] / 1.0000001)
    assert outcomes[1] == pytest.approx(data["rewards"][1] / 1.0000001)
    assert outcomes[2] > data["rewards"][2]


def test_normalization_and_query_indices_are_train_only_and_deterministic():
    train_states = np.arange(60, dtype=float).reshape(5, 12)
    train_z = np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0))
    normalization = fit_train_normalization(train_states, train_z)
    audit_z = normalize_outcomes(np.asarray((100.0,)), normalization)
    assert normalization["z_mean"] == 0.0
    assert audit_z[0] > 50
    np.testing.assert_array_equal(evenly_spaced_indices(10, 4), (0, 3, 6, 9))


def test_train_query_excludes_its_own_row_and_audit_neighbors_stay_in_train():
    data = _public_data()
    normalization = fit_train_normalization(data["observations"], data["rewards"])
    states = normalize_states(data["observations"], normalization)
    conditioner = LocalEmpiricalConditioner(data, states, normalize_outcomes(data["rewards"], normalization))
    local = conditioner.query(states[0], 3, excluded_train_row=0)
    assert 0 not in local["source_atoms"][0]["train_rows"]
    audit_state = np.full(12, 1000.0)
    audit_local = conditioner.query(audit_state, 3)
    assert all(np.all(atoms["train_rows"] < len(data["rewards"])) for atoms in audit_local["source_atoms"])


def test_local_empirical_marginals_are_uniform_and_unit_mass():
    data = _public_data()
    normalization = fit_train_normalization(data["observations"], data["rewards"])
    states = normalize_states(data["observations"], normalization)
    conditioner = LocalEmpiricalConditioner(data, states, normalize_outcomes(data["rewards"], normalization))
    local = conditioner.query(states[2], 4, excluded_train_row=2)
    for atoms in local["source_atoms"]:
        np.testing.assert_allclose(atoms["probabilities"], 0.25)
        assert atoms["probabilities"].sum() == 1.0


def test_tuple_compatibility_uses_all_pairwise_outcome_action_conditions():
    atoms = [_atoms(), _atoms()]
    tuples = enumerate_feasible_tuples(atoms, rho_coefficient_value=0.0)
    assert {tuple(row) for row in tuples} == {(0, 0), (1, 1)}


def test_toy_two_source_lp_has_known_answer():
    problem = prepare_empirical_coupling_problem([_atoms(), _atoms()], 1.0)
    solved = solve_empirical_joint_interval(problem, np.zeros(3))
    assert solved["joint_lower"] == pytest.approx(-1.0)
    assert solved["joint_upper"] == pytest.approx(1.0)
    assert solved["max_upper_marginal_error"] < 1e-10


def test_toy_three_source_lp_has_known_answer():
    problem = prepare_empirical_coupling_problem([_atoms(), _atoms(), _atoms()], 1.0)
    solved = solve_empirical_joint_interval(problem, np.zeros(3))
    assert solved["joint_lower"] == pytest.approx(-1.0)
    assert solved["joint_upper"] == pytest.approx(1.0)


def test_exact_joint_dominates_separate_without_clipping():
    solved = solve_local_problem([_atoms(), _atoms(), _atoms()], np.zeros(3), 2.0)
    assert solved["feasible"]
    assert solved["joint_upper"] <= solved["separate_upper"] + 1e-8
    assert solved["joint_lower"] >= solved["separate_lower"] - 1e-8
    assert solved["joint_lower"] <= solved["joint_upper"] + 1e-8


def test_infeasible_coupling_is_reported_without_fallback():
    left = {"actions": np.zeros((1, 3)), "outcomes": np.asarray((0.0,)),
            "probabilities": np.asarray((1.0,))}
    right = {"actions": np.zeros((1, 3)), "outcomes": np.asarray((2.0,)),
             "probabilities": np.asarray((1.0,))}
    solved = solve_local_problem([left, right], np.zeros(3), 1.0)
    assert solved["feasible"] is False
    assert solved["prepared"] is None
    assert "joint" in solved["failure"]


def test_source3_contribution_formula_and_violations():
    joint12 = {"feasible": True, "joint_upper": 3.0, "joint_lower": 0.0}
    joint123 = {"feasible": True, "joint_upper": 2.0, "joint_lower": 1.0}
    result = source3_contribution(joint12, joint123)
    assert result["source3_upper_gain"] == 1.0
    assert result["source3_lower_gain"] == 1.0
    assert result["source3_width_gain"] == 2.0
    assert result["source3_upper_violation"] == result["source3_lower_violation"] == 0.0


def test_target_actions_are_deterministic_unique_and_in_range():
    identical = [{**_atoms(), "actions": np.zeros((2, 3))} for _ in range(3)]
    first, labels = target_actions(identical, query_index=7, seed=11)
    second, second_labels = target_actions(identical, query_index=7, seed=11)
    np.testing.assert_array_equal(first, second)
    assert labels == second_labels and len(labels) == 6
    assert all(not np.allclose(first[i], first[j], atol=1e-12, rtol=0.0)
               for i in range(6) for j in range(i))
    assert np.max(np.abs(first)) <= 1.0


def test_separate_dual_baselines_have_exact_separate_objectives():
    atoms = [_atoms(), _atoms(0.1), _atoms(-0.1)]
    upper, lower, separate = dual_baselines(atoms, np.zeros(3), 2.0)
    assert sum(np.mean(values) for values in upper) == pytest.approx(separate["separate_upper"])
    assert sum(np.mean(values) for values in lower) == pytest.approx(separate["separate_lower"])


def test_exact_violation_correction_and_separate_fallback():
    corrected = exact_violation_correction(5.0, 1.0, 2.0, 2.0, 6.0, 0.0)
    assert corrected == {"corrected_upper": 7.0, "corrected_lower": -1.0,
                         "neural_upper_final": 6.0, "neural_lower_final": 0.0}


def test_dual_network_zero_head_initializes_to_separate_residual_zero():
    class FakeLayer:
        weight = np.ones((2, 128))
        bias = np.ones(2)

    layer = FakeLayer()
    zero_initialize_output_layer(layer, lambda values: values.fill(0.0))
    np.testing.assert_array_equal(layer.weight, 0.0)
    np.testing.assert_array_equal(layer.bias, 0.0)
