"""High-resolution dynamic-programming reference for the frozen environment."""

from pathlib import Path

import numpy as np

from confounded_smooth_regulator import (
    response_difference_fn,
    return_bounds,
    reward_fn,
    rho_coefficient,
    transition_fn,
    value_lipschitz_constants,
)


def _validate_grid_size(name: str, size: int) -> None:
    if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)):
        raise ValueError(f"{name} must be an odd integer at least 3")
    if size < 3 or size % 2 == 0:
        raise ValueError(f"{name} must be an odd integer at least 3")


def _validate_h(reference: dict, h: int) -> int:
    horizon = int(reference["horizon"])
    if not isinstance(h, (int, np.integer)) or isinstance(h, (bool, np.bool_)):
        raise ValueError("h must be an integer in [1, horizon]")
    if not 1 <= h <= horizon:
        raise ValueError("h must be an integer in [1, horizon]")
    return int(h)


def _q_from_continuation(
    state_grid: np.ndarray,
    continuation_values: np.ndarray,
    gamma: float,
    state: float | np.ndarray,
    action: float | np.ndarray,
) -> float | np.ndarray:
    q_value = 0.0
    for confounder in (-1, 1):
        next_state = transition_fn(state, action, confounder)
        continuation = np.interp(next_state, state_grid, continuation_values)
        q_value = q_value + 0.5 * (
            reward_fn(state, action, confounder, next_state) + gamma * continuation
        )
    return float(q_value) if np.ndim(q_value) == 0 else q_value


def solve_oracle(
    horizon: int = 20,
    gamma: float = 0.95,
    n_state: int = 1001,
    n_action: int = 1001,
) -> dict[str, np.ndarray | int | float]:
    """Solve the grid Bellman recursion and return a numerical reference."""
    _validate_grid_size("n_state", n_state)
    _validate_grid_size("n_action", n_action)
    value_lipschitz = value_lipschitz_constants(horizon, gamma)
    state_grid = np.linspace(-1.0, 1.0, n_state)
    action_grid = np.linspace(-1.0, 1.0, n_action)
    values = np.zeros((horizon + 2, n_state), dtype=np.float64)
    greedy_actions = np.full((horizon + 2, n_state), np.nan, dtype=np.float64)
    return_lower = np.zeros(horizon + 2, dtype=np.float64)
    return_upper = np.zeros(horizon + 2, dtype=np.float64)
    rho_coefficients = np.full(horizon + 2, np.nan, dtype=np.float64)
    numerical_error_bound = np.zeros(horizon + 2, dtype=np.float64)

    states = state_grid[:, None]
    actions = action_grid[None, :]
    for h in range(horizon, 0, -1):
        q_values = _q_from_continuation(
            state_grid, values[h + 1], gamma, states, actions
        )
        maximizing_indices = np.argmax(q_values, axis=1)
        values[h] = q_values[np.arange(n_state), maximizing_indices]
        greedy_actions[h] = action_grid[maximizing_indices]
        return_lower[h], return_upper[h] = return_bounds(h, horizon, gamma)
        rho_coefficients[h] = rho_coefficient(h, horizon, gamma)

    state_spacing = 2.0 / (n_state - 1)
    action_spacing = 2.0 / (n_action - 1)
    for h in range(horizon, 0, -1):
        numerical_error_bound[h] = gamma * (
            numerical_error_bound[h + 1]
            + value_lipschitz[h + 1] * state_spacing / 2.0
        ) + rho_coefficients[h] * action_spacing / 2.0

    return {
        "state_grid": state_grid,
        "action_grid": action_grid,
        "values": values,
        "greedy_actions": greedy_actions,
        "return_lower": return_lower,
        "return_upper": return_upper,
        "value_lipschitz": value_lipschitz,
        "rho_coefficients": rho_coefficients,
        "numerical_error_bound": numerical_error_bound,
        "horizon": horizon,
        "gamma": gamma,
    }


def oracle_q(
    reference: dict,
    h: int,
    state: float | np.ndarray,
    action: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate the reference Q function on demand without storing a Q tensor."""
    h = _validate_h(reference, h)
    return _q_from_continuation(
        reference["state_grid"],
        reference["values"][h + 1],
        float(reference["gamma"]),
        state,
        action,
    )


def oracle_delta_bounds(
    reference: dict,
    h: int,
    state: float,
    action: float,
    reference_action: float,
) -> dict[str, float]:
    """Return the true response differences over the two confounder values."""
    h = _validate_h(reference, h)
    state_grid = reference["state_grid"]
    continuation_values = reference["values"][h + 1]
    value_next = lambda next_state: float(
        np.interp(next_state, state_grid, continuation_values)
    )
    differences = {
        confounder: response_difference_fn(
            state,
            action,
            reference_action,
            confounder,
            float(reference["gamma"]),
            value_next,
        )
        for confounder in (-1, 1)
    }
    return {
        "delta_minus": min(differences.values()),
        "delta_plus": max(differences.values()),
        "delta_c_minus": differences[-1],
        "delta_c_plus": differences[1],
    }


def save_oracle_reference(reference: dict, path: str | Path) -> Path:
    """Save the numerical reference as a non-pickled compressed NPZ file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        "state_grid",
        "action_grid",
        "values",
        "greedy_actions",
        "return_lower",
        "return_upper",
        "value_lipschitz",
        "rho_coefficients",
        "numerical_error_bound",
        "horizon",
        "gamma",
    )
    np.savez_compressed(output_path, **{key: reference[key] for key in keys})
    return output_path
