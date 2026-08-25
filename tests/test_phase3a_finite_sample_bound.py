import numpy as np
import pytest

from exact_population_joint_bound import build_source_atoms, reference_safe_rho
from finite_sample_joint_bound import (
    build_empirical_source_atoms,
    build_probability_intervals,
    clopper_pearson_interval,
    evaluate_finite_sample_query,
    prepare_finite_sample_joint_problem,
    solve_finite_sample_joint_interval,
    support_failure_bound,
)
from generate_offline_dataset import (
    FIXED_STATE_AUDIT_FIELDS,
    FIXED_STATE_TRAIN_FIELDS,
    generate_fixed_state_dataset,
)
from oracle_ground_truth import solve_oracle


@pytest.fixture(scope="module")
def reference():
    return solve_oracle(horizon=3, gamma=0.95, n_state=101, n_action=101)


@pytest.fixture(scope="module")
def fixed_data():
    return generate_fixed_state_dataset(
        [1, 3], [-0.5, 0.5], 5, 301, horizon=3, gamma=0.95
    )


@pytest.fixture(scope="module")
def certified_data():
    return generate_fixed_state_dataset([2], [0.0], 100, 808, horizon=3, gamma=0.95)


def test_fixed_state_generation_size_and_alignment(fixed_data):
    train, audit = fixed_data
    assert set(train) == set(FIXED_STATE_TRAIN_FIELDS)
    assert set(audit) == set(FIXED_STATE_AUDIT_FIELDS)
    assert all(len(values) == 60 for values in train.values())
    assert all(len(values) == 60 for values in audit.values())
    for key in ("row_id", "query_id", "time_step", "state", "source_id", "sample_index"):
        np.testing.assert_array_equal(train[key], audit[key])


def test_public_fixed_state_data_has_no_hidden_fields(fixed_data):
    train, audit = fixed_data
    keys = {key.lower() for key in train}
    assert keys.isdisjoint({"c", "w", "u"})
    assert not any(
        token in key for key in keys for token in ("confounder", "randomizer", "exogenous")
    )
    assert {"confounder_c", "randomizer_w"} <= set(audit)


def test_each_query_id_has_the_exact_requested_state(fixed_data):
    train, _ = fixed_data
    expected = {0: -0.5, 1: 0.5, 2: -0.5, 3: 0.5}
    for query_id, state in expected.items():
        observed = np.unique(train["state"][train["query_id"] == query_id])
        np.testing.assert_array_equal(observed, [state])


def test_empirical_atoms_merge_repeated_observations(reference):
    train_slice = {
        "time_step": np.full(4, 3),
        "state": np.full(4, 0.2),
        "source_id": np.full(4, 1),
        "action": np.array([0.1, 0.1, -0.2, -0.2]),
        "reward": np.array([0.8, 0.8, 0.7, 0.7]),
        "next_state": np.array([0.3, 0.3, -0.1, -0.1]),
    }
    atoms = build_empirical_source_atoms(train_slice, reference, 3)
    np.testing.assert_array_equal(atoms["counts"], [2, 2])
    assert atoms["empirical_probabilities"].sum() == 1.0


def test_support_failure_bound_formula_and_monotonicity():
    values = [support_failure_bound(n) for n in (1, 5, 10, 50)]
    assert values == pytest.approx([min(1.0, 12 * 0.75**n) for n in (1, 5, 10, 50)])
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_clopper_pearson_boundary_cases():
    lower_zero, upper_zero = clopper_pearson_interval(0, 20, 0.01)
    lower_full, upper_full = clopper_pearson_interval(20, 20, 0.01)
    lower_mid, upper_mid = clopper_pearson_interval(7, 20, 0.01)
    assert lower_zero == 0.0
    assert upper_full == 1.0
    for lower, upper in ((lower_zero, upper_zero), (lower_full, upper_full), (lower_mid, upper_mid)):
        assert 0.0 <= lower <= upper <= 1.0


def test_insufficient_sample_size_returns_global_range(reference, fixed_data):
    train, _ = fixed_data
    result = evaluate_finite_sample_query(train, reference, 1, -0.5, 0.2)
    assert not result["certified"]
    assert result["delta_support"] >= result["delta"]
    assert result["finite_joint_lower"] == 0.0
    assert result["finite_joint_upper"] == pytest.approx(reference["return_upper"][1])
    assert result["finite_separate_lower"] == 0.0
    assert result["finite_separate_upper"] == pytest.approx(reference["return_upper"][1])


def test_artificial_single_atom_robust_joint_lp():
    atoms = [
        {
            "actions": np.array([0.0]),
            "outcomes": np.array([0.5]),
            "counts": np.array([10]),
            "empirical_probabilities": np.array([1.0]),
            "sample_size": 10,
        }
        for _ in range(3)
    ]
    intervals = {
        "probability_lower": [np.array([1.0]) for _ in range(3)],
        "probability_upper": [np.array([1.0]) for _ in range(3)],
    }
    problem = prepare_finite_sample_joint_problem(atoms, intervals, 2.0)
    solved = solve_finite_sample_joint_interval(problem, 0.2, 0.0, 1.0)
    assert solved["joint_lower"] == pytest.approx(0.1)
    assert solved["joint_upper"] == pytest.approx(0.9)
    assert solved["fallback_reason"] is None


def _empirical_query_parts(train, reference, h=2, state=0.0):
    atoms = []
    for source in (1, 2, 3):
        selected = (
            (train["time_step"] == h)
            & (train["state"] == state)
            & (train["source_id"] == source)
        )
        atoms.append(
            build_empirical_source_atoms(
                {key: values[selected] for key, values in train.items()}, reference, h
            )
        )
    return atoms, build_probability_intervals(atoms)


def test_robust_joint_couplings_respect_all_probability_intervals(reference, certified_data):
    train, _ = certified_data
    atoms, intervals = _empirical_query_parts(train, reference)
    rho = reference_safe_rho(reference, 2)["reference_safe_rho_coefficient"]
    problem = prepare_finite_sample_joint_problem(atoms, intervals, rho)
    solved = solve_finite_sample_joint_interval(
        problem, -0.3, 0.0, reference["return_upper"][2]
    )
    assert solved["fallback_reason"] is None
    tuples = problem["feasible_tuples"]
    for coupling in (solved["lower_coupling"], solved["upper_coupling"]):
        for source_index, source_atoms in enumerate(atoms):
            for atom_index in range(len(source_atoms["actions"])):
                marginal = coupling[tuples[:, source_index] == atom_index].sum()
                assert marginal >= intervals["probability_lower"][source_index][atom_index] - 1e-8
                assert marginal <= intervals["probability_upper"][source_index][atom_index] + 1e-8


@pytest.mark.parametrize("action", [-1.0, -0.4, 0.0, 0.6, 1.0])
def test_finite_joint_dominates_finite_separate(reference, certified_data, action):
    train, _ = certified_data
    result = evaluate_finite_sample_query(train, reference, 2, 0.0, action)
    assert result["finite_joint_lower"] >= result["finite_separate_lower"] - 1e-8
    assert result["finite_joint_upper"] <= result["finite_separate_upper"] + 1e-8


def test_confidence_event_implies_reference_validity(reference, certified_data):
    train, audit = certified_data
    atoms, intervals = _empirical_query_parts(train, reference)
    support_complete = True
    probabilities_inside = True
    for source_index, source_id in enumerate((1, 2, 3)):
        true_atoms = build_source_atoms(reference, 2, 0.0, source_id)
        for action, outcome, probability in zip(
            true_atoms["actions"], true_atoms["outcomes"], true_atoms["probabilities"]
        ):
            matches = np.flatnonzero(
                np.isclose(atoms[source_index]["actions"], action, atol=1e-12, rtol=0)
                & np.isclose(atoms[source_index]["outcomes"], outcome, atol=1e-12, rtol=0)
            )
            support_complete &= len(matches) == 1
            if len(matches) == 1:
                index = matches[0]
                probabilities_inside &= (
                    intervals["probability_lower"][source_index][index] <= probability
                    <= intervals["probability_upper"][source_index][index]
                )
    assert len(audit["confounder_c"]) == 300
    assert support_complete and probabilities_inside
    result = evaluate_finite_sample_query(train, reference, 2, 0.0, 0.45)
    assert result["finite_joint_lower"] <= result["reference_q"] + 1e-8
    assert result["reference_q"] <= result["finite_joint_upper"] + 1e-8


def test_fixed_state_generation_and_bounds_are_reproducible(reference):
    first = generate_fixed_state_dataset([2], [0.0], 50, 900, horizon=3)
    second = generate_fixed_state_dataset([2], [0.0], 50, 900, horizon=3)
    different = generate_fixed_state_dataset([2], [0.0], 50, 901, horizon=3)
    for first_part, second_part in zip(first, second):
        for key in first_part:
            np.testing.assert_array_equal(first_part[key], second_part[key])
    assert not np.array_equal(first[1]["confounder_c"], different[1]["confounder_c"])
    first_bound = evaluate_finite_sample_query(first[0], reference, 2, 0.0, -0.25)
    second_bound = evaluate_finite_sample_query(second[0], reference, 2, 0.0, -0.25)
    for key in ("finite_joint_lower", "finite_joint_upper", "finite_separate_lower", "finite_separate_upper"):
        assert first_bound[key] == pytest.approx(second_bound[key], abs=1e-15)
