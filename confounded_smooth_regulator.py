"""Continuous regulator environment with an unobserved per-step confounder."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry


ENV_ID = "ConfoundedSmoothRegulator-v0"
_BEHAVIOR_PARAMETERS = {
    1: (0.25, 0.55, 0.55),
    2: (0.70, 0.30, 0.30),
    3: (1.00, -0.20, 0.20),
}


def sample_exogenous(rng: np.random.Generator) -> tuple[int, int]:
    """Sample independent fair binary components ``(C, W)``."""
    confounder = 2 * int(rng.integers(0, 2)) - 1
    behavior_randomization = 2 * int(rng.integers(0, 2)) - 1
    return confounder, behavior_randomization


def transition_fn(
    state: float | np.ndarray,
    action: float | np.ndarray,
    confounder: int | np.ndarray,
) -> float | np.ndarray:
    """Return the deterministic next state for the structural transition."""
    state_array = np.asarray(state)
    action_array = np.asarray(action)
    result = np.tanh(
        0.55 * state_array
        + (0.30 + 0.10 * np.tanh(state_array)) * action_array
        + 0.15 * confounder
    )
    return float(result) if result.ndim == 0 else result


def reward_fn(
    state: float | np.ndarray,
    action: float | np.ndarray,
    confounder: int | np.ndarray,
    next_state: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Return the one-step reward, deriving the next state when omitted."""
    action_array = np.asarray(action)
    if next_state is None:
        next_state = transition_fn(state, action_array, confounder)
    result = (
        1.0
        - 0.45 * np.asarray(next_state) ** 2
        - 0.15 * action_array**2
        - 0.10 * (action_array + confounder) ** 2
    )
    return float(result) if result.ndim == 0 else result


def behavior_action_fn(
    source_id: int,
    state: float,
    exogenous: tuple[int, int] | np.ndarray,
    kappa: float,
) -> float:
    """Return source ``source_id``'s privileged historical-policy action."""
    try:
        state_gain, confounder_gain, w_scale = _BEHAVIOR_PARAMETERS[source_id]
    except (KeyError, TypeError) as exc:
        raise ValueError("source_id must be one of 1, 2, or 3") from exc
    if len(exogenous) != 2:
        raise ValueError("exogenous must contain (C, W)")
    confounder = float(exogenous[0])
    behavior_randomization = float(exogenous[1])
    return float(
        np.tanh(
            -state_gain * float(state)
            - float(kappa) * confounder_gain * confounder
            + w_scale * behavior_randomization
        )
    )


def bellman_response_fn(
    state: float,
    action: float,
    confounder: int,
    gamma: float,
    value_next: Callable[[float], float],
) -> float:
    """Return ``r(s,a,c) + gamma * value_next(f(s,a,c))``."""
    next_state = transition_fn(state, action, confounder)
    return float(
        reward_fn(state, action, confounder, next_state)
        + float(gamma) * value_next(next_state)
    )


def response_difference_fn(
    state: float,
    action: float,
    reference_action: float,
    confounder: int,
    gamma: float,
    value_next: Callable[[float], float],
) -> float:
    """Return the Bellman-response difference between two actions."""
    return bellman_response_fn(state, action, confounder, gamma, value_next) - bellman_response_fn(
        state, reference_action, confounder, gamma, value_next
    )


def _validate_theory_inputs(horizon: int, gamma: float) -> None:
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, (bool, np.bool_)):
        raise ValueError("horizon must be a positive integer")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")


def return_bounds(h: int, horizon: int, gamma: float) -> tuple[float, float]:
    """Return the lower and upper discounted-return bounds at stage ``h``."""
    _validate_theory_inputs(horizon, gamma)
    if not isinstance(h, (int, np.integer)) or isinstance(h, (bool, np.bool_)) or not 1 <= h <= horizon:
        raise ValueError("h must be an integer in [1, horizon]")
    remaining_steps = horizon - h + 1
    upper = float(remaining_steps) if gamma == 1.0 else float(
        (1.0 - gamma**remaining_steps) / (1.0 - gamma)
    )
    return 0.0, upper


def value_lipschitz_constants(horizon: int, gamma: float) -> np.ndarray:
    """Return ``L_h`` in an array indexed by stages ``1, ..., H+1``."""
    _validate_theory_inputs(horizon, gamma)
    constants = np.zeros(horizon + 2, dtype=np.float64)
    for h in range(horizon, 0, -1):
        constants[h] = 0.585 + gamma * 0.65 * constants[h + 1]
    return constants


def rho_coefficient(h: int, horizon: int, gamma: float) -> float:
    """Return the response-difference coefficient at stage ``h``."""
    if not isinstance(h, (int, np.integer)) or isinstance(h, (bool, np.bool_)) or not 1 <= h <= horizon:
        raise ValueError("h must be an integer in [1, horizon]")
    constants = value_lipschitz_constants(horizon, gamma)
    return float(1.06 + gamma * 0.40 * constants[h + 1])


def rho_fn(
    h: int,
    action: float,
    reference_action: float,
    horizon: int,
    gamma: float,
) -> float:
    """Return the stage-wise response-difference safety radius."""
    return rho_coefficient(h, horizon, gamma) * abs(float(action) - float(reference_action))


class ConfoundedSmoothRegulatorEnv(gym.Env[np.ndarray, np.ndarray]):
    """Finite-horizon continuous regulator with hidden i.i.d. confounding.

    ``get_exogenous_for_audit`` and ``privileged_behavior_action`` are FOR AUDIT /
    OFFLINE DATA GENERATION ONLY. They MUST NOT BE USED BY LEARNING ALGORITHMS.
    """

    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 20, gamma: float = 0.95, kappa: float = 1.0):
        super().__init__()
        _validate_theory_inputs(horizon, gamma)
        if not np.isfinite(kappa) or kappa < 0.0:
            raise ValueError("kappa must be finite and nonnegative")
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._state: float | None = None
        self._h: int | None = None
        self._exogenous: tuple[int, int] | None = None
        self._terminated = False

    def _observation(self) -> np.ndarray:
        assert self._state is not None and self._h is not None
        remaining_fraction = max(0.0, (self.horizon - self._h + 1) / self.horizon)
        return np.array([self._state, remaining_fraction], dtype=np.float32)

    @staticmethod
    def _parse_action(action: float | np.ndarray) -> float:
        array = np.asarray(action)
        if array.shape not in ((), (1,)):
            raise ValueError("action must be a scalar or an array with shape (1,)")
        try:
            value = float(array.reshape(-1)[0] if array.shape else array)
        except (TypeError, ValueError) as exc:
            raise ValueError("action must be a finite real number") from exc
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("action must be finite and in [-1, 1]")
        return value

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = {} if options is None else options
        if "state" in options:
            try:
                state = float(options["state"])
            except (TypeError, ValueError) as exc:
                raise ValueError("reset state must be a finite scalar in [-1, 1]") from exc
            if not np.isfinite(state) or not -1.0 <= state <= 1.0:
                raise ValueError("reset state must be a finite scalar in [-1, 1]")
        else:
            state = float(self.np_random.uniform(-0.8, 0.8))
        self._state = state
        self._h = 1
        self._exogenous = sample_exogenous(self.np_random)
        self._terminated = False
        return self._observation(), {}

    def step(self, action: float | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset() before step()")
        if self._state is None or self._h is None or self._exogenous is None:
            raise RuntimeError("reset() must be called before step()")
        action_value = self._parse_action(action)
        confounder = self._exogenous[0]
        next_state = transition_fn(self._state, action_value, confounder)
        reward = reward_fn(self._state, action_value, confounder, next_state)
        terminated = self._h == self.horizon
        self._state = next_state
        self._h += 1
        self._terminated = terminated
        if terminated:
            self._exogenous = None
        else:
            self._exogenous = sample_exogenous(self.np_random)
        return self._observation(), float(reward), terminated, False, {}

    def get_exogenous_for_audit(self) -> tuple[int, int]:
        """Expose current U FOR AUDIT / OFFLINE DATA GENERATION ONLY.

        MUST NOT BE USED BY LEARNING ALGORITHMS.
        """
        if self._exogenous is None:
            raise RuntimeError("current exogenous variable is unavailable; reset the environment")
        return self._exogenous

    def privileged_behavior_action(self, source_id: int) -> float:
        """Return an action FOR AUDIT / OFFLINE DATA GENERATION ONLY.

        MUST NOT BE USED BY LEARNING ALGORITHMS.
        """
        if self._state is None or self._exogenous is None:
            raise RuntimeError("current privileged data is unavailable; reset the environment")
        return behavior_action_fn(source_id, self._state, self._exogenous, self.kappa)


if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="confounded_smooth_regulator:ConfoundedSmoothRegulatorEnv")
