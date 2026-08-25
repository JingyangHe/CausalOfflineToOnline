import numpy as np
import pytest

from exact_population_joint_bound import (
    build_source_atoms_from_continuation,
    evaluate_population_query,
    prepare_joint_problem,
    solve_joint_bound,
)
from exact_population_sequential_bound import (
    close_state_interval,
    evaluate_joint_action_grid,
    evaluate_lower_extension,
    evaluate_upper_extension,
    solve_population_sequential_joint,
)
from oracle_ground_truth import solve_oracle


@pytest.fixture(scope="module")
def reference():
    return solve_oracle(horizon=3, gamma=0.95, n_state=101, n_action=101)


@pytest.fixture(scope="module")
def sequential(reference):
    return solve_population_sequential_joint(
        reference, horizon=3, gamma=0.95, n_state=7, n_action=11
    )


def test_safe_state_extensions_cover_known_lipschitz_function():
    state_grid = np.linspace(-1.0, 1.0, 5)
    truth_grid = 0.3 * state_grid
    upper_values = truth_grid + 0.08
    lower_values = truth_grid - 0.06
    query = np.linspace(-1.0, 1.0, 401)
    truth = 0.3 * query
    upper = evaluate_upper_extension(query, state_grid, upper_values, 1.0, 2.0)
    lower = evaluate_lower_extension(query, state_grid, lower_values, 1.0, -2.0)
    assert np.all(lower <= truth + 1e-14)
    assert np.all(truth <= upper + 1e-14)
    spacing = query[1] - query[0]
    assert np.max(np.abs(np.diff(lower))) <= spacing + 1e-12
    assert np.max(np.abs(np.diff(upper))) <= spacing + 1e-12


def test_state_closure_tightens_in_safe_direction():
    state_grid = np.linspace(-1.0, 1.0, 5)
    truth = 0.3 * state_grid
    lower_raw = truth - np.array([0.10, 0.20, 0.08, 0.18, 0.09])
    upper_raw = truth + np.array([0.11, 0.19, 0.07, 0.17, 0.10])
    lower, upper = close_state_interval(
        state_grid, lower_raw, upper_raw, 0.3, -2.0, 2.0
    )
    assert np.all(lower >= lower_raw - 1e-14)
    assert np.all(upper <= upper_raw + 1e-14)
    assert np.all(lower <= truth + 1e-14)
    assert np.all(truth <= upper + 1e-14)


def test_sequential_array_structure_and_terminal_rows(sequential):
    assert sequential["joint_lower"].shape == (5, 7)
    assert sequential["joint_upper"].shape == (5, 7)
    assert sequential["source_lower"].shape == (3, 5, 7)
    assert sequential["source_upper"].shape == (3, 5, 7)
    for key in ("joint_lower", "joint_upper", "joint_lower_raw", "joint_upper_raw"):
        np.testing.assert_array_equal(sequential[key][4], 0.0)
    assert np.all(np.isnan(sequential["joint_lower_argmax"][4]))
    assert np.all(np.isnan(sequential["joint_upper_argmax"][4]))


def _terminal_action_grids(reference, state, action_grid):
    zero_next = lambda next_state: np.zeros_like(next_state, dtype=np.float64)
    atoms = [
        build_source_atoms_from_continuation(3, state, source, 0.95, 1.0, zero_next)
        for source in (1, 2, 3)
    ]
    problem = prepare_joint_problem(atoms, reference["rho_coefficients"][3])
    lower, _ = evaluate_joint_action_grid(problem, action_grid, 0.0, 1.0, "lower")
    upper, _ = evaluate_joint_action_grid(problem, action_grid, 0.0, 1.0, "upper")
    return lower, upper


def test_terminal_action_grid_matches_phase2a_queries(reference, sequential):
    state = 0.0
    action_grid = sequential["action_grid"]
    lower, upper = _terminal_action_grids(reference, state, action_grid)
    phase2a = [evaluate_population_query(reference, 3, state, action) for action in action_grid]
    np.testing.assert_allclose(lower, [item["joint_lower"] for item in phase2a], atol=1e-13)
    np.testing.assert_allclose(upper, [item["joint_upper"] for item in phase2a], atol=1e-13)


def test_action_grid_upper_correction_and_no_lower_correction(reference, sequential):
    state_index, state = 3, 0.0
    lower, upper = _terminal_action_grids(reference, state, sequential["action_grid"])
    rho = sequential["rho_coefficients"][3]
    action_spacing = sequential["action_grid"][1] - sequential["action_grid"][0]
    expected_upper = min(1.0, np.max(upper) + rho * action_spacing / 2.0)
    assert sequential["joint_upper_raw"][3, state_index] == pytest.approx(expected_upper)
    assert sequential["joint_lower_raw"][3, state_index] == pytest.approx(np.max(lower))
    zero_next = lambda next_state: np.zeros_like(next_state, dtype=np.float64)
    atoms = [
        build_source_atoms_from_continuation(3, state, source, 0.95, 1.0, zero_next)
        for source in (1, 2, 3)
    ]
    problem = prepare_joint_problem(atoms, rho)
    with pytest.raises(ValueError):
        solve_joint_bound(problem, 0.0, 0.0, 1.0, "both")


def test_small_recursion_has_oracle_error_aware_validity(sequential):
    for h in range(1, 4):
        error = sequential["reference_error"][h]
        assert np.all(
            sequential["joint_lower"][h] <= sequential["reference_value"][h] + error + 1e-10
        )
        assert np.all(
            sequential["joint_upper"][h] >= sequential["reference_value"][h] - error - 1e-10
        )


def test_small_recursion_joint_bounds_dominate_independent_sources(sequential):
    separate_lower = np.max(sequential["source_lower"], axis=0)
    separate_upper = np.min(sequential["source_upper"], axis=0)
    assert np.all(sequential["joint_lower"][1:4] >= separate_lower[1:4] - 1e-8)
    assert np.all(sequential["joint_upper"][1:4] <= separate_upper[1:4] + 1e-8)


def test_all_closed_bounds_pass_state_lipschitz_audit(sequential):
    spacing = sequential["state_grid"][1] - sequential["state_grid"][0]
    arrays = [sequential["joint_lower"], sequential["joint_upper"]]
    arrays.extend(sequential["source_lower"])
    arrays.extend(sequential["source_upper"])
    for values in arrays:
        for h in range(1, 4):
            assert np.max(np.abs(np.diff(values[h]))) <= (
                sequential["value_lipschitz"][h] * spacing + 1e-10
            )


def test_true_coupling_is_feasible_for_both_continuations_at_all_grid_states(sequential):
    grid = sequential["state_grid"]
    for h in range(3, 0, -1):
        upper_next = lambda x: evaluate_upper_extension(
            x, grid, sequential["joint_upper"][h + 1], sequential["value_lipschitz"][h + 1],
            sequential["B_plus"][h + 1]
        )
        lower_next = lambda x: evaluate_lower_extension(
            x, grid, sequential["joint_lower"][h + 1], sequential["value_lipschitz"][h + 1],
            sequential["B_minus"][h + 1]
        )
        for state in grid:
            for continuation in (upper_next, lower_next):
                atoms = [
                    build_source_atoms_from_continuation(
                        h, state, source, 0.95, 1.0, continuation
                    )
                    for source in (1, 2, 3)
                ]
                problem = prepare_joint_problem(atoms, sequential["rho_coefficients"][h])
                assert problem["true_coupling_weights"].sum() == pytest.approx(1.0)
                assert problem["max_true_coupling_marginal_error"] <= 1e-12


def test_small_recursion_is_deterministic(reference, sequential):
    repeated = solve_population_sequential_joint(
        reference, horizon=3, gamma=0.95, n_state=7, n_action=11
    )
    for key, value in sequential.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, repeated[key], atol=1e-15, equal_nan=True)
        else:
            assert value == repeated[key]


def test_result_does_not_store_full_q_tensor(sequential):
    forbidden_shape = (3, 7, 11)
    assert not any(
        isinstance(value, np.ndarray) and value.shape == forbidden_shape
        for value in sequential.values()
    )
