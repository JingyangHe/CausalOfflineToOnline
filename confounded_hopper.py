"""Hidden-actuator-confounding wrapper for Gymnasium Hopper environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTUATOR_DIRECTION = np.asarray((1.0, -1.0, 1.0), dtype=np.float64) / np.sqrt(3.0)
PRIVATE_INFO_FIELDS = ("hidden_u", "commanded_action", "applied_action")


class ConfoundedHopperWrapper(gym.Wrapper):
    """Add time-to-go and a fresh hidden actuator bias to a Hopper-like env."""

    def __init__(
        self,
        env: gym.Env,
        kappa: float = 0.2,
        expose_confounder: bool = True,
        audit_info: bool = False,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("the wrapped observation space must be Box")
        if len(env.observation_space.shape) != 1:
            raise ValueError("the wrapped observation must be one-dimensional")
        if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (3,):
            raise ValueError("the wrapped action space must be a three-dimensional Box")
        if not np.isfinite(kappa) or kappa < 0.0:
            raise ValueError("kappa must be finite and nonnegative")
        if env.spec is None or env.spec.max_episode_steps is None:
            raise ValueError("the wrapped environment spec must define max_episode_steps")

        self.kappa = float(kappa)
        self.expose_confounder = bool(expose_confounder)
        self.audit_info = bool(audit_info)
        self.max_episode_steps = int(env.spec.max_episode_steps)
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.elapsed_steps = 0
        self._hidden_u = 1
        self._hidden_rng = np.random.default_rng()

        raw_low = np.asarray(env.observation_space.low, dtype=np.float32)
        raw_high = np.asarray(env.observation_space.high, dtype=np.float32)
        public_low = np.concatenate((raw_low, np.array((0.0,), dtype=np.float32)))
        public_high = np.concatenate((raw_high, np.array((1.0,), dtype=np.float32)))
        self.public_observation_dimension = public_low.size
        if self.expose_confounder:
            public_low = np.concatenate((public_low, np.array((-1.0,), dtype=np.float32)))
            public_high = np.concatenate((public_high, np.array((1.0,), dtype=np.float32)))
        self.observation_space = spaces.Box(
            low=public_low, high=public_high, dtype=np.float32
        )

    @property
    def time_to_go(self) -> float:
        """Return the public remaining-time fraction."""
        remaining = self.max_episode_steps - self.elapsed_steps
        return float(np.clip(remaining / self.max_episode_steps, 0.0, 1.0))

    def _sample_hidden_u(self) -> int:
        return 2 * int(self._hidden_rng.integers(0, 2)) - 1

    def _public_observation(self, raw_observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_observation, dtype=np.float32)
        expected_shape = self.env.observation_space.shape
        if raw.shape != expected_shape:
            raise ValueError(
                f"wrapped observation shape {raw.shape} does not match {expected_shape}"
            )
        return np.concatenate(
            (raw, np.asarray((self.time_to_go,), dtype=np.float32))
        ).astype(np.float32, copy=False)

    def _policy_observation(self, public_observation: np.ndarray) -> np.ndarray:
        if not self.expose_confounder:
            return public_observation
        return np.concatenate(
            (public_observation, np.asarray((self._hidden_u,), dtype=np.float32))
        ).astype(np.float32, copy=False)

    def get_public_observation(self, observation: np.ndarray) -> np.ndarray:
        """Remove the final hidden coordinate from a behavior observation."""
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim == 0:
            raise ValueError("observation must have at least one dimension")
        if values.shape[-1] == self.public_observation_dimension + 1:
            return values[..., : self.public_observation_dimension].copy()
        if values.shape[-1] == self.public_observation_dimension:
            return values.copy()
        raise ValueError("observation has an unexpected final dimension")

    def _public_info(
        self, info: dict[str, Any], public_observation: np.ndarray
    ) -> dict[str, Any]:
        result = dict(info)
        for field in PRIVATE_INFO_FIELDS:
            result.pop(field, None)
        result.update(
            public_observation=public_observation.copy(),
            elapsed_steps=self.elapsed_steps,
            time_to_go=self.time_to_go,
        )
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._hidden_rng = np.random.default_rng(seed)
        raw_observation, info = self.env.reset(seed=seed, options=options)
        self.elapsed_steps = 0
        self._hidden_u = self._sample_hidden_u()
        public_observation = self._public_observation(raw_observation)
        result_info = self._public_info(info, public_observation)
        if self.audit_info:
            result_info["hidden_u"] = self._hidden_u
        return self._policy_observation(public_observation), result_info

    def step(
        self, commanded_action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        command = np.asarray(commanded_action, dtype=np.float64)
        if command.shape != self.action_space.shape or not np.all(np.isfinite(command)):
            raise ValueError("commanded_action must be a finite action-space vector")
        transition_u = self._hidden_u
        applied = np.clip(
            command + self.kappa * transition_u * ACTUATOR_DIRECTION,
            self.action_space.low,
            self.action_space.high,
        ).astype(self.action_space.dtype, copy=False)
        raw_observation, reward, terminated, truncated, info = self.env.step(applied)
        self.elapsed_steps += 1
        self._hidden_u = self._sample_hidden_u()
        public_observation = self._public_observation(raw_observation)
        result_info = self._public_info(info, public_observation)
        if self.audit_info:
            result_info.update(
                hidden_u=transition_u,
                commanded_action=command.astype(self.action_space.dtype),
                applied_action=applied.copy(),
            )
        return (
            self._policy_observation(public_observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            result_info,
        )

    def capture_audit_state(self) -> dict[str, Any]:
        """Capture simulator state for paired audit steps only."""
        if not self.audit_info:
            raise RuntimeError("simulator snapshots require audit_info=True")
        base = self.env.unwrapped
        if not hasattr(base, "model") or not hasattr(base, "data"):
            raise RuntimeError("the wrapped environment has no MuJoCo simulator state")
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - server dependency
            raise RuntimeError("mujoco is required for simulator snapshots") from exc

        state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        simulator_state = np.empty(
            mujoco.mj_stateSize(base.model, state_spec), dtype=np.float64
        )
        mujoco.mj_getState(base.model, base.data, simulator_state, state_spec)
        wrapper_elapsed_steps: list[int | None] = []
        current: Any = self.env
        while isinstance(current, gym.Wrapper):
            value = getattr(current, "_elapsed_steps", None)
            wrapper_elapsed_steps.append(None if value is None else int(value))
            current = current.env
        return {
            "simulator_state": simulator_state,
            "state_spec": int(state_spec),
            "elapsed_steps": int(self.elapsed_steps),
            "wrapper_elapsed_steps": tuple(wrapper_elapsed_steps),
        }

    def audit_step_from_state(
        self,
        snapshot: dict[str, Any],
        commanded_action: np.ndarray,
        hidden_u: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Restore one audit snapshot and step with a forced transition confounder."""
        if not self.audit_info:
            raise RuntimeError("paired simulator steps require audit_info=True")
        if hidden_u not in (-1, 1):
            raise ValueError("hidden_u must be -1 or +1")
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - server dependency
            raise RuntimeError("mujoco is required for simulator snapshots") from exc

        base = self.env.unwrapped
        state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        if int(snapshot["state_spec"]) != int(state_spec):
            raise ValueError("snapshot uses an unsupported MuJoCo state specification")
        simulator_state = np.asarray(snapshot["simulator_state"], dtype=np.float64)
        expected = mujoco.mj_stateSize(base.model, state_spec)
        if simulator_state.shape != (expected,) or not np.all(np.isfinite(simulator_state)):
            raise ValueError("snapshot contains an invalid simulator state")
        mujoco.mj_setState(base.model, base.data, simulator_state, state_spec)
        mujoco.mj_forward(base.model, base.data)

        elapsed_values = tuple(snapshot["wrapper_elapsed_steps"])
        current: Any = self.env
        index = 0
        while isinstance(current, gym.Wrapper):
            if index >= len(elapsed_values):
                raise ValueError("snapshot does not match the environment wrapper stack")
            if elapsed_values[index] is not None:
                current._elapsed_steps = int(elapsed_values[index])
            current = current.env
            index += 1
        if index != len(elapsed_values):
            raise ValueError("snapshot does not match the environment wrapper stack")
        self.elapsed_steps = int(snapshot["elapsed_steps"])
        self._hidden_u = int(hidden_u)
        return self.step(commanded_action)

    def close(self) -> None:
        self.env.close()
