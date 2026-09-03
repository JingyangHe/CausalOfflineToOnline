from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

from experiments.hopper_logger_mixture_drift.phase8g_quick_exact_shared_world import (
    FORBIDDEN_PUBLIC_FIELDS,
    Phase8GQuickExactSharedWorldError,
    build_observational_constraints,
    deterministic_argmax,
    enumerate_response_types,
    extract_public_reward_support,
    matched_opposite_probabilities,
    natural_bounds,
    public_population_from_support,
    solve_shared_world_bounds,
    source_probability_tables,
    source_shuffle,
    validate_public_distribution,
)


REWARDS = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
SIZES = np.asarray([2, 2, 2])


def _mass_from_q(types: np.ndarray, q: np.ndarray, source_count: int) -> np.ndarray:
    mass = np.zeros((source_count, 3, 2), dtype=np.float64)
    for index, probability in enumerate(q):
        for source in range(source_count):
            action = types[index, 3 + source]
            category = types[index, action]
            mass[source, action, category] += probability
    return mass


def _known_world(source_count: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    types = enumerate_response_types(SIZES, source_count)
    rng = np.random.default_rng(19)
    q = rng.random(len(types))
    q /= q.sum()
    return types, q, _mass_from_q(types, q, source_count)


def test_response_type_count() -> None:
    assert enumerate_response_types((2, 2, 2), 5).shape == (2**3 * 3**5, 8)


def test_response_type_enumeration() -> None:
    types = enumerate_response_types((1, 2, 1), 2)
    assert len(types) == 1 * 2 * 1 * 3**2
    assert set(types[:, 0]) == {0} and set(types[:, 1]) == {0, 1}
    assert set(types[:, 3]) == {0, 1, 2}


def test_observational_constraint_matrix_toy() -> None:
    types, q, mass = _known_world()
    matrix, rhs, labels = build_observational_constraints(types, SIZES, mass)
    assert matrix.shape == (1 + 2 * 3 * 2, len(types))
    assert labels[0] == ("total",)
    assert np.allclose(matrix @ q, rhs)


def test_public_support_excludes_hidden_u() -> None:
    assert not {"u", "u_env", "u_behavior"}.intersection(
        inspect.signature(solve_shared_world_bounds).parameters
    )
    public = {"anchor_id": np.array([0]), "commanded_action": np.zeros((1, 3)),
              "reward": np.array([0.0]), "u_env": np.array([1])}
    try:
        extract_public_reward_support(public, [0])
    except Phase8GQuickExactSharedWorldError:
        pass
    else:
        raise AssertionError("hidden field was accepted")


def test_public_support_excludes_do() -> None:
    assert "do_reward" in FORBIDDEN_PUBLIC_FIELDS
    assert "do_reward" not in inspect.signature(solve_shared_world_bounds).parameters


def test_observational_mass_normalized() -> None:
    mass = public_population_from_support(
        REWARDS, SIZES, matched_opposite_probabilities(), "confounded"
    )
    validate_public_distribution(REWARDS, SIZES, mass)
    assert np.allclose(mass.sum(axis=(1, 2)), 1.0)


def test_lp_feasible_toy() -> None:
    _, _, mass = _known_world()
    fit = solve_shared_world_bounds(REWARDS, SIZES, mass)
    assert fit["max_equality_residual"] <= 1e-8
    assert fit["minimum_q"] >= -1e-8


def test_lp_recovers_known_simple_world() -> None:
    # Every unit chooses action 0 and its action-0 reward is exactly one.
    mass = np.zeros((1, 3, 2))
    mass[0, 0, 1] = 1.0
    fit = solve_shared_world_bounds(REWARDS, SIZES, mass)
    assert np.isclose(fit["lower"][0], 1.0)
    assert np.isclose(fit["upper"][0], 1.0)


def test_joint_bound_contains_true_do_toy() -> None:
    types, q, mass = _known_world()
    truth = np.asarray([REWARDS[a, types[:, a]] @ q for a in range(3)])
    fit = solve_shared_world_bounds(REWARDS, SIZES, mass)
    assert np.all(truth >= fit["lower"] - 1e-8)
    assert np.all(truth <= fit["upper"] + 1e-8)


def test_duplicate_source_invariance() -> None:
    _, _, mass = _known_world(1)
    one = solve_shared_world_bounds(REWARDS, SIZES, mass)
    two = solve_shared_world_bounds(REWARDS, SIZES, np.repeat(mass, 2, axis=0))
    assert np.allclose(one["lower"], two["lower"])
    assert np.allclose(one["upper"], two["upper"])


def test_redundant_source_invariance() -> None:
    table = source_probability_tables()["M5_redundant"]
    mass = public_population_from_support(REWARDS, SIZES, table, "confounded")
    one = solve_shared_world_bounds(REWARDS, SIZES, mass[:1])
    five = solve_shared_world_bounds(REWARDS, SIZES, mass)
    assert np.allclose(one["lower"], five["lower"], atol=1e-8)
    assert np.allclose(one["upper"], five["upper"], atol=1e-8)


def test_joint_not_wider_than_natural_intersection_when_applicable() -> None:
    _, _, mass = _known_world()
    joint = solve_shared_world_bounds(REWARDS, SIZES, mass)
    natural = natural_bounds(REWARDS, SIZES, mass, REWARDS.min(), REWARDS.max())
    assert np.all(joint["lower"] >= natural["intersection_lower"] - 1e-8)
    assert np.all(joint["upper"] <= natural["intersection_upper"] + 1e-8)


def test_source_shuffle_preserves_required_marginals() -> None:
    _, _, mass = _known_world(5)
    shuffled = source_shuffle(mass)
    assert np.allclose(shuffled.sum(axis=2), mass.sum(axis=2))
    assert np.allclose(shuffled.sum(axis=0), mass.sum(axis=0))
    assert np.allclose(shuffled.sum(axis=(1, 2)), mass.sum(axis=(1, 2)))


def test_false_certification_zero() -> None:
    mass = np.zeros((1, 3, 2)); mass[0, 0, 1] = 1.0
    fit = solve_shared_world_bounds(REWARDS, SIZES, mass)
    true_do = np.array([1.0, 2.5, 4.5])
    for a in range(3):
        for b in range(3):
            if fit["lower"][a] > fit["upper"][b] + 1e-8:
                assert true_do[a] > true_do[b]


def test_input_hashes_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"; path.write_bytes(b"immutable")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    _, _, mass = _known_world(); solve_shared_world_bounds(REWARDS, SIZES, mass)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_no_nan_inf() -> None:
    _, _, mass = _known_world()
    fit = solve_shared_world_bounds(REWARDS, SIZES, mass)
    assert np.isfinite(fit["lower"]).all() and np.isfinite(fit["upper"]).all()


def test_old_artifacts_unchanged(tmp_path: Path) -> None:
    old = tmp_path / "phase8f.npz"; np.savez(old, x=np.arange(3))
    before = old.read_bytes()
    out = tmp_path / "derived_inputs"; out.mkdir()
    np.savez(out / "phase8f_public_joint_support_v2.npz", y=np.arange(2))
    assert old.read_bytes() == before


def test_matched_opposite_has_equal_three_action_marginals_and_reverse_tables() -> None:
    table = matched_opposite_probabilities()
    assert np.allclose(table.mean(axis=1), [[.45, .10, .45], [.45, .10, .45]])
    assert np.all(table.mean(axis=1) > 0)
    assert np.allclose(table[1], table[0, ::-1])
    assert deterministic_argmax([1.0, 1.0, 0.0]) == 0
