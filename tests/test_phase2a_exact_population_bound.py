import numpy as np
import pytest

from exact_population_joint_bound import (
    LATENT_TYPES,
    build_source_atoms,
    evaluate_population_query,
    evaluate_true_coupling_envelope,
    interpolated_value_lipschitz,
    merge_observed_atoms,
    prepare_joint_problem,
    solve_joint_interval,
)
from oracle_ground_truth import solve_oracle


@pytest.fixture(scope="module")
def reference():
    return solve_oracle(horizon=3, gamma=0.95, n_state=51, n_action=51)


def _safe_rho(reference, h):
    next_lipschitz = max(
        reference["value_lipschitz"][h + 1],
        interpolated_value_lipschitz(reference, h + 1),
    )
    return 1.06 + reference["gamma"] * 0.40 * next_lipschitz


def _problem(reference, h, state):
    atoms = [build_source_atoms(reference, h, state, source) for source in (1, 2, 3)]
    return atoms, prepare_joint_problem(atoms, _safe_rho(reference, h))


def test_each_source_population_has_complete_probability_and_latent_map(reference):
    for source_id in (1, 2, 3):
        atoms = build_source_atoms(reference, 2, 0.0, source_id)
        assert 1 <= len(atoms["actions"]) <= 4
        assert atoms["probabilities"].sum() == pytest.approx(1.0, abs=1e-15)
        assert set(atoms["latent_to_atom"]) == set(LATENT_TYPES)
        assert set(atoms["latent_to_atom"].values()) <= set(range(len(atoms["actions"])))


def test_identical_observed_atoms_are_merged_without_latent_separation():
    atoms = merge_observed_atoms(
        actions=np.array([0.0, 0.0, 0.5, -0.5]),
        outcomes=np.array([0.7, 0.7, 0.8, 0.6]),
        probabilities=np.full(4, 0.25),
    )
    assert len(atoms["actions"]) == 3
    assert atoms["probabilities"][0] == pytest.approx(0.5)
    assert atoms["latent_to_atom"][(-1, -1)] == atoms["latent_to_atom"][(-1, 1)]


@pytest.mark.parametrize("h,state", [(1, -0.4), (2, 0.0), (3, 0.6)])
def test_true_shared_u_coupling_is_feasible_and_reproduces_marginals(reference, h, state):
    _, problem = _problem(reference, h, state)
    feasible = {tuple(atom_tuple) for atom_tuple in problem["feasible_tuples"]}
    for latent in LATENT_TYPES:
        true_tuple = tuple(
            atoms["latent_to_atom"][latent] for atoms in problem["source_atoms"]
        )
        assert true_tuple in feasible
    assert problem["true_coupling_weights"].sum() == pytest.approx(1.0, abs=1e-15)
    assert problem["max_true_coupling_marginal_error"] <= 1e-12


def test_artificial_single_tuple_lp_equals_its_envelope():
    source_atoms = []
    for source_id in (1, 2, 3):
        source_atoms.append(
            {
                "actions": np.array([0.0]),
                "outcomes": np.array([0.5]),
                "probabilities": np.array([1.0]),
                "latent_to_atom": {latent: 0 for latent in LATENT_TYPES},
                "h": 1,
                "state": 0.0,
                "source_id": source_id,
            }
        )
    problem = prepare_joint_problem(source_atoms, rho_coefficient_value=2.0)
    interval = solve_joint_interval(problem, target_action=0.2, B_minus=0.0, B_plus=1.0)
    envelope = evaluate_true_coupling_envelope(problem, 0.2, 0.0, 1.0)
    assert interval["joint_lower"] == pytest.approx(0.1, abs=1e-14)
    assert interval["joint_upper"] == pytest.approx(0.9, abs=1e-14)
    assert envelope == pytest.approx(
        {"true_coupling_lower": 0.1, "true_coupling_upper": 0.9}, abs=1e-14
    )


def test_lp_solutions_reproduce_every_omitted_and_retained_marginal(reference):
    atoms, problem = _problem(reference, 2, 0.23)
    interval = solve_joint_interval(problem, -0.37, 0.0, reference["return_upper"][2])
    tuples = problem["feasible_tuples"]
    for coupling_name in ("lower_coupling", "upper_coupling"):
        coupling = interval[coupling_name]
        for source_index, source in enumerate(atoms):
            for atom_index, probability in enumerate(source["probabilities"]):
                actual = coupling[tuples[:, source_index] == atom_index].sum()
                assert actual == pytest.approx(probability, abs=1e-8)


@pytest.mark.parametrize("state", [-0.7, 0.0, 0.8])
@pytest.mark.parametrize("action", [-1.0, -0.2, 0.6, 1.0])
def test_terminal_queries_cover_reference_q(reference, state, action):
    result = evaluate_population_query(reference, 3, state, action)
    assert result["joint_lower"] <= result["reference_q"] + 1e-8
    assert result["reference_q"] <= result["joint_upper"] + 1e-8


@pytest.mark.parametrize("h,state,action", [(1, -0.5, 0.4), (2, 0.6, -0.7)])
def test_intermediate_queries_cover_reference_q(reference, h, state, action):
    result = evaluate_population_query(reference, h, state, action)
    assert result["joint_lower"] <= result["reference_q"] <= result["joint_upper"]
    assert result["reference_safe_rho_coefficient"] >= result["analytic_rho_coefficient"] - 1e-12


def test_joint_interval_dominates_separate_rho_intersection_on_grid(reference):
    for h in (1, 3):
        for state in (-0.5, 0.0, 0.5):
            for action in (-1.0, -0.5, 0.0, 0.5, 1.0):
                result = evaluate_population_query(reference, h, state, action)
                assert result["joint_lower"] >= result["separate_lower"] - 1e-8
                assert result["joint_upper"] <= result["separate_upper"] + 1e-8


def test_true_coupling_envelope_chain(reference):
    result = evaluate_population_query(reference, 2, -0.31, 0.77)
    assert result["joint_lower"] <= result["true_coupling_lower"] + 1e-8
    assert result["true_coupling_lower"] <= result["reference_q"] + 1e-8
    assert result["reference_q"] <= result["true_coupling_upper"] + 1e-8
    assert result["true_coupling_upper"] <= result["joint_upper"] + 1e-8


def test_reference_safe_rho_never_uses_smaller_continuation_slope(reference):
    for h in range(1, 4):
        result = evaluate_population_query(reference, h, 0.1, -0.2)
        assert result["reference_safe_rho_coefficient"] >= (
            result["analytic_rho_coefficient"] - 1e-12
        )
        assert result["continuation_lipschitz_numeric"] >= 0.0
    assert interpolated_value_lipschitz(reference, 4) == 0.0


def test_query_and_lp_are_deterministic(reference):
    first = evaluate_population_query(reference, 2, 0.19, -0.63)
    second = evaluate_population_query(reference, 2, 0.19, -0.63)
    scalar_keys = (
        "reference_q",
        "joint_lower",
        "joint_upper",
        "separate_lower",
        "separate_upper",
        "true_coupling_lower",
        "true_coupling_upper",
        "upper_gain",
        "lower_gain",
        "width_gain",
    )
    for key in scalar_keys:
        assert first[key] == pytest.approx(second[key], abs=1e-15)
    np.testing.assert_allclose(first["upper_coupling"], second["upper_coupling"], atol=1e-15)
    np.testing.assert_allclose(first["lower_coupling"], second["lower_coupling"], atol=1e-15)
