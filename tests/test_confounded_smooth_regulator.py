import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from confounded_smooth_regulator import (
    ENV_ID,
    ConfoundedSmoothRegulatorEnv,
    behavior_action_fn,
    bellman_response_fn,
    response_difference_fn,
    return_bounds,
    reward_fn,
    rho_coefficient,
    rho_fn,
    sample_exogenous,
    transition_fn,
    value_lipschitz_constants,
)


def test_gymnasium_interface_and_registration():
    env = ConfoundedSmoothRegulatorEnv(horizon=3)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)
    assert observation.shape == (2,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info == {}
    observation, reward, terminated, truncated, info = env.step(np.array([0.2], np.float32))
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert (terminated, truncated, info) == (False, False, {})
    registered = gym.make(ENV_ID, horizon=1)
    registered.reset(seed=1)
    _, _, terminated, truncated, _ = registered.step(0.0)
    assert terminated and not truncated


def test_transition_and_reward_match_independent_reference_formula():
    state, action, confounder = 0.2, -0.4, 1
    expected_next = float(
        np.tanh(
            0.55 * state
            + (0.30 + 0.10 * np.tanh(state)) * action
            + 0.15 * confounder
        )
    )
    expected_reward = float(
        1.0
        - 0.45 * expected_next**2
        - 0.15 * action**2
        - 0.10 * (action + confounder) ** 2
    )
    assert transition_fn(state, action, confounder) == pytest.approx(expected_next, abs=1e-15)
    assert reward_fn(state, action, confounder) == pytest.approx(expected_reward, abs=1e-15)
    assert reward_fn(state, action, confounder, expected_next) == pytest.approx(
        expected_reward, abs=1e-15
    )


def test_exogenous_format_is_two_fair_binary_components():
    rng = np.random.default_rng(11)
    for _ in range(100):
        exogenous = sample_exogenous(rng)
        assert len(exogenous) == 2
        assert exogenous[0] in (-1, 1)
        assert exogenous[1] in (-1, 1)


def test_w_does_not_affect_transition_or_reward():
    state, action, confounder = -0.31, 0.72, -1
    exogenous_minus = (confounder, -1)
    exogenous_plus = (confounder, 1)
    response_minus = (
        transition_fn(state, action, exogenous_minus[0]),
        reward_fn(state, action, exogenous_minus[0]),
    )
    response_plus = (
        transition_fn(state, action, exogenous_plus[0]),
        reward_fn(state, action, exogenous_plus[0]),
    )
    assert response_minus == response_plus


@pytest.mark.parametrize("source_id", [1, 2, 3])
def test_w_affects_each_historical_behavior(source_id):
    action_minus = behavior_action_fn(source_id, 0.13, (1, -1), 1.0)
    action_plus = behavior_action_fn(source_id, 0.13, (1, 1), 1.0)
    assert action_minus != pytest.approx(action_plus)


@pytest.mark.parametrize("source_id", [1, 2, 3])
def test_zero_kappa_removes_confounder_from_behavior(source_id):
    exogenous_minus = (-1, 1)
    exogenous_plus = (1, 1)
    assert behavior_action_fn(source_id, 0.4, exogenous_minus, 0.0) == behavior_action_fn(
        source_id, 0.4, exogenous_plus, 0.0
    )


def test_source_one_action_zero_does_not_reveal_confounder():
    assert behavior_action_fn(1, 0.0, (-1, -1), 1.0) == pytest.approx(0.0, abs=1e-15)
    assert behavior_action_fn(1, 0.0, (1, 1), 1.0) == pytest.approx(0.0, abs=1e-15)


def test_source_one_retains_strong_confounding_at_fixed_w():
    action_minus = behavior_action_fn(1, 0.0, (-1, 1), 1.0)
    action_plus = behavior_action_fn(1, 0.0, (1, 1), 1.0)
    assert action_minus == pytest.approx(np.tanh(1.10), abs=1e-15)
    assert action_plus == pytest.approx(np.tanh(0.0), abs=1e-15)
    assert action_minus - action_plus > 0.7


@pytest.mark.parametrize(
    "source_id,state,confounder,w,state_gain,confounder_gain,w_scale",
    [
        (2, 0.3, -1, 1, 0.70, 0.30, 0.30),
        (3, -0.4, 1, -1, 1.00, -0.20, 0.20),
    ],
)
def test_source_two_and_three_match_independent_formulas(
    source_id, state, confounder, w, state_gain, confounder_gain, w_scale
):
    expected = float(
        np.tanh(-state_gain * state - confounder_gain * confounder + w_scale * w)
    )
    assert behavior_action_fn(source_id, state, (confounder, w), 1.0) == pytest.approx(
        expected, abs=1e-15
    )


def test_reward_and_state_ranges_on_dense_grid():
    for state in np.linspace(-1.0, 1.0, 51):
        for action in np.linspace(-1.0, 1.0, 51):
            for confounder in (-1, 1):
                next_state = transition_fn(state, action, confounder)
                reward = reward_fn(state, action, confounder, next_state)
                assert -1.0 < next_state < 1.0
                assert -1e-12 <= reward <= 1.0 + 1e-12


def test_invalid_actions_sources_states_and_post_terminal_steps_raise():
    env = ConfoundedSmoothRegulatorEnv(horizon=1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0.0)
    with pytest.raises(ValueError):
        env.reset(options={"state": 1.01})
    for action in (-1.01, 1.01, np.nan, np.array([0.0, 0.1])):
        env.reset(seed=2)
        with pytest.raises(ValueError):
            env.step(action)
    with pytest.raises(ValueError):
        behavior_action_fn(0, 0.0, (1, 1), 1.0)
    env.reset(seed=2)
    with pytest.raises(ValueError):
        env.privileged_behavior_action(4)
    env.step(0.0)
    with pytest.raises(RuntimeError, match="terminated"):
        env.step(0.0)
    with pytest.raises(RuntimeError, match="unavailable"):
        env.get_exogenous_for_audit()


def _trajectory(seed):
    env = ConfoundedSmoothRegulatorEnv(horizon=5)
    observation, info = env.reset(seed=seed)
    trajectory = [(observation.copy(), info.copy(), env.get_exogenous_for_audit())]
    for action in (-0.7, 0.1, 0.8, -0.2, 0.0):
        exogenous = env.get_exogenous_for_audit()
        result = env.step(action)
        trajectory.append((result[0].copy(), *result[1:], exogenous))
    return trajectory


def test_same_seed_reproduces_states_rewards_and_exogenous_sequence():
    first, second = _trajectory(12345), _trajectory(12345)
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            if isinstance(left_item, np.ndarray):
                np.testing.assert_array_equal(left_item, right_item)
            else:
                assert left_item == right_item


def test_observations_and_infos_do_not_expose_hidden_variables():
    env = ConfoundedSmoothRegulatorEnv(horizon=2)
    observation, info = env.reset(seed=9, options={"state": 0.25})
    assert observation.tolist() == pytest.approx([0.25, 1.0])
    assert info == {}
    observation, _, _, _, info = env.step(0.1)
    assert observation.shape == (2,)
    assert info == {}


def test_return_bounds_match_backward_recursion_and_gamma_one_case():
    horizon, gamma = 20, 0.95
    recursive_upper = 0.0
    expected = {}
    for h in range(horizon, 0, -1):
        recursive_upper = 1.0 + gamma * recursive_upper
        expected[h] = recursive_upper
    for h in range(1, horizon + 1):
        lower, upper = return_bounds(h, horizon, gamma)
        assert lower == 0.0
        assert upper == pytest.approx(expected[h], rel=1e-14)
    assert return_bounds(3, 7, 1.0) == (0.0, 5.0)


def test_lipschitz_recursion_rho_and_response_helpers():
    horizon, gamma = 20, 0.95
    constants = value_lipschitz_constants(horizon, gamma)
    assert constants.shape == (horizon + 2,)
    assert constants[horizon + 1] == 0.0
    for h in range(1, horizon + 1):
        assert constants[h] == pytest.approx(0.585 + gamma * 0.65 * constants[h + 1])
        coefficient = rho_coefficient(h, horizon, gamma)
        assert coefficient == pytest.approx(1.06 + gamma * 0.40 * constants[h + 1])
        assert rho_fn(h, -0.2, 0.7, horizon, gamma) == pytest.approx(coefficient * 0.9)
    value_next = lambda next_state: 0.3 - 0.2 * next_state
    response = bellman_response_fn(0.1, 0.4, -1, gamma, value_next)
    expected_next = transition_fn(0.1, 0.4, -1)
    expected = reward_fn(0.1, 0.4, -1, expected_next) + gamma * value_next(expected_next)
    assert response == pytest.approx(expected)
    assert response_difference_fn(0.1, 0.4, -0.6, -1, gamma, value_next) == pytest.approx(
        response - bellman_response_fn(0.1, -0.6, -1, gamma, value_next)
    )


def test_sampled_smoothness_bounds():
    rng = np.random.default_rng(2027)
    for _ in range(5000):
        state, other_state, action, other_action = rng.uniform(-1.0, 1.0, size=4)
        confounder = int(rng.choice((-1, 1)))
        f = transition_fn(state, action, confounder)
        assert abs(f - transition_fn(state, other_action, confounder)) <= (
            0.40 * abs(action - other_action) + 1e-12
        )
        assert abs(f - transition_fn(other_state, action, confounder)) <= (
            0.65 * abs(state - other_state) + 1e-12
        )
        r = reward_fn(state, action, confounder)
        assert abs(r - reward_fn(state, other_action, confounder)) <= (
            1.06 * abs(action - other_action) + 1e-12
        )
        assert abs(r - reward_fn(other_state, action, confounder)) <= (
            0.585 * abs(state - other_state) + 1e-12
        )
