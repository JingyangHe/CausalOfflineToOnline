import numpy as np
import pytest

from confounded_smooth_regulator import rho_fn
from generate_offline_dataset import (
    AUDIT_FIELDS,
    TRAIN_FIELDS,
    generate_offline_dataset,
    save_offline_dataset,
)
from oracle_ground_truth import (
    oracle_delta_bounds,
    oracle_q,
    save_oracle_reference,
    solve_oracle,
)


@pytest.fixture(scope="module")
def reference():
    return solve_oracle(horizon=3, gamma=0.95, n_state=51, n_action=51)


@pytest.fixture(scope="module")
def dataset():
    return generate_offline_dataset(episodes_per_source=2, base_seed=37, horizon=3)


def test_oracle_array_structure(reference):
    assert reference["values"].shape == (5, 51)
    assert reference["greedy_actions"].shape == (5, 51)
    np.testing.assert_array_equal(reference["values"][4], 0.0)
    assert np.all(np.isnan(reference["greedy_actions"][4]))
    assert reference["state_grid"][25] == 0.0
    assert reference["action_grid"][25] == 0.0


def test_oracle_rejects_invalid_grids_and_time(reference):
    for kwargs in ({"n_state": 2}, {"n_state": 50}, {"n_action": 2}, {"n_action": 50}):
        with pytest.raises(ValueError):
            solve_oracle(horizon=2, **kwargs)
    for h in (0, 4):
        with pytest.raises(ValueError):
            oracle_q(reference, h, 0.0, 0.0)


def test_last_stage_q_matches_independent_reward_formula(reference):
    state, action = 0.2, -0.4
    expected = 0.0
    for confounder in (-1, 1):
        next_state = np.tanh(
            0.55 * state
            + (0.30 + 0.10 * np.tanh(state)) * action
            + 0.15 * confounder
        )
        reward = (
            1.0
            - 0.45 * next_state**2
            - 0.15 * action**2
            - 0.10 * (action + confounder) ** 2
        )
        expected += 0.5 * reward
    assert oracle_q(reference, 3, state, action) == pytest.approx(expected, abs=1e-14)


def test_oracle_bellman_values_are_consistent(reference):
    action_grid = reference["action_grid"]
    for h in range(1, 4):
        for index in (0, 7, 25, 43, 50):
            state = reference["state_grid"][index]
            q_values = oracle_q(reference, h, state, action_grid)
            assert reference["values"][h, index] == pytest.approx(
                np.max(q_values), abs=1e-13
            )


def test_greedy_actions_belong_to_grid_and_attain_maximum(reference):
    action_grid = reference["action_grid"]
    for h in range(1, 4):
        for index in (0, 25, 50):
            state = reference["state_grid"][index]
            q_values = oracle_q(reference, h, state, action_grid)
            maximum = np.max(q_values)
            greedy_action = reference["greedy_actions"][h, index]
            assert np.any(action_grid == greedy_action)
            assert oracle_q(reference, h, state, greedy_action) == pytest.approx(
                maximum, abs=1e-13
            )


def _independent_response(reference, h, state, action, confounder):
    next_state = np.tanh(
        0.55 * state
        + (0.30 + 0.10 * np.tanh(state)) * action
        + 0.15 * confounder
    )
    reward = (
        1.0
        - 0.45 * next_state**2
        - 0.15 * action**2
        - 0.10 * (action + confounder) ** 2
    )
    continuation = np.interp(
        next_state, reference["state_grid"], reference["values"][h + 1]
    )
    return reward + reference["gamma"] * continuation


def test_delta_bounds_match_independent_response_differences(reference):
    h, state, action, reference_action = 2, 0.17, -0.44, 0.36
    expected = {
        confounder: _independent_response(reference, h, state, action, confounder)
        - _independent_response(reference, h, state, reference_action, confounder)
        for confounder in (-1, 1)
    }
    bounds = oracle_delta_bounds(reference, h, state, action, reference_action)
    assert bounds["delta_c_minus"] == pytest.approx(expected[-1], abs=1e-14)
    assert bounds["delta_c_plus"] == pytest.approx(expected[1], abs=1e-14)
    assert bounds["delta_minus"] == pytest.approx(min(expected.values()), abs=1e-14)
    assert bounds["delta_plus"] == pytest.approx(max(expected.values()), abs=1e-14)
    assert bounds["delta_minus"] <= bounds["delta_plus"]


def test_analytic_rho_contains_sampled_true_differences(reference):
    for h in range(1, 4):
        for state in (-0.8, 0.0, 0.7):
            for action, reference_action in ((-0.9, 0.2), (-0.1, 0.8), (0.4, 0.4)):
                bounds = oracle_delta_bounds(reference, h, state, action, reference_action)
                radius = rho_fn(h, action, reference_action, 3, 0.95)
                assert abs(bounds["delta_c_minus"]) <= radius + 1e-12
                assert abs(bounds["delta_c_plus"]) <= radius + 1e-12


def test_numerical_error_recursion(reference):
    error = reference["numerical_error_bound"]
    value_lipschitz = reference["value_lipschitz"]
    rho = reference["rho_coefficients"]
    assert error[4] == 0.0
    for h in range(3, 0, -1):
        expected = 0.95 * (error[h + 1] + value_lipschitz[h + 1] * 0.04 / 2.0)
        expected += rho[h] * 0.04 / 2.0
        assert error[h] == pytest.approx(expected, abs=1e-15)


def test_nested_grid_refinement_is_within_certificates(reference):
    fine = solve_oracle(horizon=3, gamma=0.95, n_state=101, n_action=101)
    np.testing.assert_allclose(fine["state_grid"][::2], reference["state_grid"], atol=0.0)
    for h in range(1, 4):
        difference = np.max(np.abs(reference["values"][h] - fine["values"][h, ::2]))
        certificate = (
            reference["numerical_error_bound"][h]
            + fine["numerical_error_bound"][h]
            + 1e-12
        )
        assert difference <= certificate


def test_dataset_size_schema_and_alignment(dataset):
    train_data, audit_data = dataset
    assert set(train_data) == set(TRAIN_FIELDS)
    assert set(audit_data) == set(AUDIT_FIELDS)
    assert all(len(values) == 18 for values in train_data.values())
    assert all(len(values) == 18 for values in audit_data.values())
    for source_id in (1, 2, 3):
        assert np.count_nonzero(train_data["source_id"] == source_id) == 6
    for key in ("row_id", "source_id", "episode_id", "time_step"):
        np.testing.assert_array_equal(train_data[key], audit_data[key])
    lower_keys = {key.lower() for key in train_data}
    assert lower_keys.isdisjoint({"c", "w", "u"})
    assert not any(
        token in key for key in lower_keys for token in ("confounder", "randomizer", "exogenous")
    )
    assert {"confounder_c", "randomizer_w"} <= set(audit_data)


def test_every_dataset_row_matches_frozen_structural_formulas(dataset):
    train_data, audit_data = dataset
    state = train_data["state"].astype(np.float64)
    action = train_data["action"]
    confounder = audit_data["confounder_c"]
    expected_next = np.tanh(
        0.55 * state
        + (0.30 + 0.10 * np.tanh(state)) * action
        + 0.15 * confounder
    )
    expected_reward = (
        1.0
        - 0.45 * expected_next**2
        - 0.15 * action**2
        - 0.10 * (action + confounder) ** 2
    )
    np.testing.assert_allclose(train_data["next_state"], expected_next, rtol=1e-7, atol=5e-8)
    np.testing.assert_allclose(train_data["reward"], expected_reward, rtol=1e-7, atol=5e-8)


def test_every_historical_action_matches_its_independent_formula(dataset):
    train_data, audit_data = dataset
    parameters = {
        1: (0.25, 0.55, 0.55),
        2: (0.70, 0.30, 0.30),
        3: (1.00, -0.20, 0.20),
    }
    expected = np.empty(len(train_data["row_id"]))
    for source_id, (state_gain, confounder_gain, w_scale) in parameters.items():
        selected = train_data["source_id"] == source_id
        expected[selected] = np.tanh(
            -state_gain * train_data["state"][selected]
            - confounder_gain * audit_data["confounder_c"][selected]
            + w_scale * audit_data["randomizer_w"][selected]
        )
    np.testing.assert_allclose(train_data["action"], expected, rtol=1e-7, atol=5e-8)


def test_generation_is_reproducible_and_seed_sensitive(dataset):
    same = generate_offline_dataset(2, 37, horizon=3)
    different = generate_offline_dataset(2, 38, horizon=3)
    for first_part, same_part in zip(dataset, same):
        for key in first_part:
            np.testing.assert_array_equal(first_part[key], same_part[key])
    train_data, audit_data = dataset
    other_train, other_audit = different
    assert not np.array_equal(train_data["state"], other_train["state"])
    assert not np.array_equal(audit_data["confounder_c"], other_audit["confounder_c"])


def test_episode_structure(dataset):
    train_data, _ = dataset
    for source_id in (1, 2, 3):
        for episode_id in (0, 1):
            selected = (train_data["source_id"] == source_id) & (
                train_data["episode_id"] == episode_id
            )
            np.testing.assert_array_equal(train_data["time_step"][selected], (1, 2, 3))
            np.testing.assert_array_equal(train_data["terminated"][selected], (False, False, True))


def test_npz_saving_and_overwrite_protection(reference, dataset, tmp_path):
    oracle_path = save_oracle_reference(reference, tmp_path / "oracle_reference.npz")
    with np.load(oracle_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(reference)
    train_path, audit_path = save_offline_dataset(*dataset, tmp_path)
    with np.load(train_path, allow_pickle=False) as saved_train:
        assert set(saved_train.files) == set(TRAIN_FIELDS)
    with np.load(audit_path, allow_pickle=False) as saved_audit:
        assert set(saved_audit.files) == set(AUDIT_FIELDS)
    with pytest.raises(FileExistsError):
        save_offline_dataset(*dataset, tmp_path)
