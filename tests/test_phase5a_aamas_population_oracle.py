"""Focused tests for the Phase 5A AAMAS26 population/oracle analogue."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from aamas_population_oracle import (
    ACTION_ATOL,
    UnsupportedRoadSupportError,
    aamas_population_state_target,
    build_pooled_observable_population,
    observable_action_summary,
    population_behavior_weights,
    road_candidate_mask,
    solve_aamas_population_mass_analogue,
)
from scripts.validate_phase5a_aamas_population_oracle import _classification


def test_builder_returns_normalized_observable_joint_law():
    population = build_pooled_observable_population(0.17)
    assert population.primitive_support_size == 12
    assert 1 <= population.merged_action_support_size <= 12
    assert population.actions.shape == population.rewards.shape
    assert population.actions.shape == population.next_states.shape
    assert population.actions.shape == population.probabilities.shape
    assert np.all(population.probabilities > 0.0)
    assert np.sum(population.probabilities) == pytest.approx(1.0, abs=1e-12)
    assert np.sum(population.action_masses) == pytest.approx(1.0, abs=1e-12)
    assert population.action_atoms.size == population.merged_action_support_size
    for conditional in population.conditional_probabilities:
        assert np.sum(conditional) == pytest.approx(1.0, abs=1e-12)


def test_hidden_fields_are_stripped_and_operator_interface_is_public_only():
    population = build_pooled_observable_population(0.0)
    forbidden = {"c", "w", "source", "source_id", "oracle_q", "oracle_v", "hidden"}
    assert forbidden.isdisjoint(vars(population))
    parameters = set(inspect.signature(aamas_population_state_target).parameters)
    assert forbidden.isdisjoint(parameters)
    with pytest.raises(TypeError):
        aamas_population_state_target(oracle_v=np.zeros(3))


def test_observational_delta_is_the_conditional_public_mean():
    population = build_pooled_observable_population(-0.23)
    summary = observable_action_summary(
        -0.23,
        population.actions,
        population.next_states,
        population.probabilities,
    )
    for index, action in enumerate(summary["action_atoms"]):
        mask = np.isclose(
            population.actions, action, atol=ACTION_ATOL, rtol=0.0
        )
        expected = np.sum(
            population.probabilities[mask]
            * (population.next_states[mask] + 0.23)
        ) / np.sum(population.probabilities[mask])
        assert summary["delta_means"][index] == pytest.approx(expected)


def test_road_candidate_rule_uses_absolute_point_one_separation():
    atoms = np.array([-0.2, -0.1, -0.099, 0.0, 0.099, 0.1, 0.2])
    mask = road_candidate_mask(0.0, atoms)
    assert np.array_equal(mask, [True, True, False, False, False, True, True])


def test_behavior_weight_matches_mass_geometric_mean_formula():
    atoms = np.array([-0.5, 0.0, 0.5])
    masses = np.array([0.2, 0.3, 0.5])
    weights, counts = population_behavior_weights(atoms, masses)
    normalized = np.array([0.3, 0.5]) / 0.8
    q_road = np.exp(np.sum(normalized * np.log([0.3, 0.5])))
    assert weights[0] == pytest.approx(0.2 / (0.2 + q_road))
    assert np.all((weights > 0.0) & (weights < 1.0))
    assert np.array_equal(counts, [2, 2, 2])


def test_empty_road_support_raises_instead_of_falling_back():
    with pytest.raises(UnsupportedRoadSupportError, match="no ROAD candidate"):
        population_behavior_weights(np.array([0.0]), np.array([1.0]))


def test_state_target_is_joint_mass_expectation_without_action_maximization():
    actions = np.array([-0.5, 0.0, 0.5])
    rewards = np.array([0.2, 0.4, 0.8])
    probabilities = np.array([0.2, 0.3, 0.5])
    result = aamas_population_state_target(
        state=0.0,
        stage=1,
        actions=actions,
        rewards=rewards,
        next_states=np.array([-0.1, 0.0, 0.1]),
        probabilities=probabilities,
        state_grid=np.array([-1.0, 0.0, 1.0]),
        continuation=np.zeros(3),
    )
    weights, _ = population_behavior_weights(actions, probabilities)
    expected = np.sum(probabilities * (weights * rewards + (1.0 - weights)))
    assert result["potential"] == pytest.approx(expected)
    assert result["potential"] != pytest.approx(
        np.max(weights * rewards + (1.0 - weights))
    )


def test_backward_recursion_is_finite_terminal_zero_and_reproducible():
    grid = np.linspace(-1.0, 1.0, 13)
    first = solve_aamas_population_mass_analogue(grid, horizon=3)
    second = solve_aamas_population_mass_analogue(grid, horizon=3)
    assert first["phi"].shape == (5, 13)
    assert np.array_equal(first["phi"][4], np.zeros(13))
    assert np.all(np.isfinite(first["phi"]))
    assert np.array_equal(first["phi"], second["phi"])


def test_prescribed_full_state_grid_has_no_empty_road_support():
    for state in np.linspace(-1.0, 1.0, 1001):
        population = build_pooled_observable_population(state)
        summary = observable_action_summary(
            state,
            population.actions,
            population.next_states,
            population.probabilities,
        )
        _, counts = population_behavior_weights(
            summary["action_atoms"], summary["action_masses"]
        )
        assert np.all(counts > 0)


def test_module_has_no_neural_or_joint_solver_dependency():
    source_path = Path(__file__).resolve().parents[1] / "aamas_population_oracle.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "torch" not in imported_roots
    assert "exact_population_joint_bound" not in imported_roots
    assert "exact_population_sequential_bound" not in imported_roots


def test_epsilon_classification_has_three_disjoint_exhaustive_categories():
    upper = np.array([1.11, 0.89, 1.05, 0.90])
    reference = np.ones(4)
    epsilon = np.full(4, 0.1)
    classes = _classification(upper, reference, epsilon)
    assert np.array_equal(classes["certified_upper"], [True, False, False, False])
    assert np.array_equal(
        classes["certified_undercoverage"], [False, True, False, False]
    )
    total = sum(values.astype(int) for values in classes.values())
    assert np.array_equal(total, np.ones(4, dtype=int))
