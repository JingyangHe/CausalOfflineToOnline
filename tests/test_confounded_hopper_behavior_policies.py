"""Unit tests for Phase 6A that do not require MuJoCo or SAC training."""

from types import SimpleNamespace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from confounded_hopper import ACTUATOR_DIRECTION, ConfoundedHopperWrapper
from train_hopper_behavior_policies import ExactCheckpointCallback


class TinyContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 4):
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.spec = SimpleNamespace(max_episode_steps=max_episode_steps)
        self.step_count = 0
        self.last_action = None
        self.was_closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.array((0.25, -0.5), dtype=np.float32), {"base": "reset"}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        self.step_count += 1
        observation = np.array((self.step_count, -self.step_count), dtype=np.float32)
        return observation, 2.5, False, self.step_count >= 4, {"base": "step"}

    def close(self):
        self.was_closed = True


def make_wrapper(expose=True, audit=False, kappa=0.2):
    return ConfoundedHopperWrapper(
        TinyContinuousEnv(),
        kappa=kappa,
        expose_confounder=expose,
        audit_info=audit,
    )


def test_reset_hidden_u_is_binary():
    environment = make_wrapper()
    for seed in range(10):
        observation, _ = environment.reset(seed=seed)
        assert observation[-1] in (-1.0, 1.0)


def test_fixed_seed_reproduces_hidden_sequence_and_applied_actions():
    first, second = make_wrapper(audit=True), make_wrapper(audit=True)
    first_observation, _ = first.reset(seed=37)
    second_observation, _ = second.reset(seed=37)
    np.testing.assert_array_equal(first_observation, second_observation)
    command = np.array((0.2, -0.1, 0.4), dtype=np.float32)
    for _ in range(4):
        first_observation, _, _, _, first_info = first.step(command)
        second_observation, _, _, _, second_info = second.step(command)
        np.testing.assert_array_equal(first_observation, second_observation)
        np.testing.assert_array_equal(
            first_info["applied_action"], second_info["applied_action"]
        )


def test_hidden_u_is_resampled_after_each_step():
    environment = make_wrapper()
    hidden_values = iter((-1, 1, -1))
    environment._sample_hidden_u = lambda: next(hidden_values)
    first, _ = environment.reset()
    second, _, _, _, _ = environment.step(np.zeros(3))
    third, _, _, _, _ = environment.step(np.zeros(3))
    assert (first[-1], second[-1], third[-1]) == (-1.0, 1.0, -1.0)


def test_applied_action_matches_hidden_actuator_formula():
    environment = make_wrapper(audit=True, kappa=0.3)
    _, reset_info = environment.reset(seed=8)
    command = np.array((0.2, -0.4, 0.7), dtype=np.float32)
    _, _, _, _, info = environment.step(command)
    expected = np.clip(
        command + 0.3 * reset_info["hidden_u"] * ACTUATOR_DIRECTION, -1.0, 1.0
    )
    np.testing.assert_allclose(info["applied_action"], expected, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(info["applied_action"], environment.env.last_action)


def test_applied_action_is_always_clipped_to_action_space():
    environment = make_wrapper(audit=True, kappa=2.0)
    environment.reset(seed=1)
    for command in (np.ones(3), -np.ones(3)):
        _, _, _, _, info = environment.step(command)
        assert np.all(info["applied_action"] >= -1.0)
        assert np.all(info["applied_action"] <= 1.0)


def test_exposed_behavior_observation_has_hidden_dimension():
    environment = make_wrapper(expose=True)
    observation, info = environment.reset(seed=3)
    assert observation.shape == (4,)
    assert environment.observation_space.shape == (4,)
    assert info["public_observation"].shape == (3,)


def test_public_observation_mode_does_not_include_hidden_u():
    environment = make_wrapper(expose=False)
    observation, info = environment.reset(seed=3)
    assert observation.shape == (3,)
    np.testing.assert_array_equal(observation, info["public_observation"])
    assert observation[-1] == 1.0


def test_default_info_never_leaks_private_transition_fields():
    environment = make_wrapper(audit=False)
    _, reset_info = environment.reset(seed=9)
    _, _, _, _, step_info = environment.step(np.zeros(3))
    for info in (reset_info, step_info):
        assert "hidden_u" not in info
        assert "commanded_action" not in info
        assert "applied_action" not in info
        assert {"public_observation", "elapsed_steps", "time_to_go"} <= info.keys()


def test_audit_info_reports_transition_u_command_and_applied_action():
    environment = make_wrapper(audit=True)
    _, reset_info = environment.reset(seed=12)
    command = np.array((-0.3, 0.2, 0.1), dtype=np.float32)
    _, _, _, _, info = environment.step(command)
    assert info["hidden_u"] == reset_info["hidden_u"]
    np.testing.assert_array_equal(info["commanded_action"], command)
    np.testing.assert_array_equal(info["applied_action"], environment.env.last_action)


def test_public_time_to_go_decreases_monotonically():
    environment = make_wrapper()
    observation, _ = environment.reset(seed=0)
    time_values = [observation[-2]]
    for _ in range(3):
        observation, _, _, _, _ = environment.step(np.zeros(3))
        time_values.append(observation[-2])
    np.testing.assert_allclose(time_values, (1.0, 0.75, 0.5, 0.25))
    assert np.all(np.diff(time_values) < 0.0)


def test_get_public_observation_removes_only_hidden_coordinate():
    environment = make_wrapper()
    observation, info = environment.reset(seed=21)
    public = environment.get_public_observation(observation)
    np.testing.assert_array_equal(public, info["public_observation"])
    batch = np.stack((observation, observation))
    assert environment.get_public_observation(batch).shape == (2, 3)


def test_checkpoint_callback_saves_only_once_at_exact_requested_steps(tmp_path):
    class FakeModel:
        def __init__(self):
            self.num_timesteps = 0
            self.saved_paths = []

        def save(self, path):
            self.saved_paths.append(path)
            Path(path).touch()

    callback = ExactCheckpointCallback((500, 1_000, 2_000), tmp_path)
    model = FakeModel()
    for step in (1, 499, 500, 500, 501, 999, 1_000, 1_999, 2_000):
        model.num_timesteps = step
        assert callback({"self": model}, {})
    assert callback.saved_steps == {500, 1_000, 2_000}
    assert [Path(path).name for path in model.saved_paths] == [
        "source_1_step_500.zip",
        "source_2_step_1000.zip",
        "source_3_step_2000.zip",
    ]
