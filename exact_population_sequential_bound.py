"""Exact population sequential joint recursion with safe grid extensions."""

import numpy as np

from confounded_smooth_regulator import (
    return_bounds,
    rho_coefficient,
    value_lipschitz_constants,
)
from exact_population_joint_bound import (
    build_source_atoms_from_continuation,
    prepare_joint_problem,
    solve_joint_bound,
)


def _validate_grid_size(name: str, size: int) -> None:
    if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)):
        raise ValueError(f"{name} must be an odd integer at least 3")
    if size < 3 or size % 2 == 0:
        raise ValueError(f"{name} must be an odd integer at least 3")


def evaluate_upper_extension(
    query_state: float | np.ndarray,
    state_grid: np.ndarray,
    upper_values: np.ndarray,
    lipschitz_constant: float,
    B_plus: float,
) -> float | np.ndarray:
    """Evaluate the smallest clipped Lipschitz upper cone envelope."""
    query = np.asarray(query_state, dtype=np.float64)
    values = upper_values + lipschitz_constant * np.abs(query[..., None] - state_grid)
    result = np.minimum(B_plus, np.min(values, axis=-1))
    return float(result) if result.ndim == 0 else result


def evaluate_lower_extension(
    query_state: float | np.ndarray,
    state_grid: np.ndarray,
    lower_values: np.ndarray,
    lipschitz_constant: float,
    B_minus: float,
) -> float | np.ndarray:
    """Evaluate the largest clipped Lipschitz lower cone envelope."""
    query = np.asarray(query_state, dtype=np.float64)
    values = lower_values - lipschitz_constant * np.abs(query[..., None] - state_grid)
    result = np.maximum(B_minus, np.max(values, axis=-1))
    return float(result) if result.ndim == 0 else result


def close_state_interval(
    state_grid: np.ndarray,
    lower_raw: np.ndarray,
    upper_raw: np.ndarray,
    lipschitz_constant: float,
    B_minus: float,
    B_plus: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tighten grid intervals using the proven state Lipschitz modulus."""
    lower_closed = evaluate_lower_extension(
        state_grid, state_grid, lower_raw, lipschitz_constant, B_minus
    )
    upper_closed = evaluate_upper_extension(
        state_grid, state_grid, upper_raw, lipschitz_constant, B_plus
    )
    return lower_closed, upper_closed


def evaluate_joint_action_grid(
    prepared_problem: dict,
    action_grid: np.ndarray,
    B_minus: float,
    B_plus: float,
    bound_type: str,
) -> tuple[np.ndarray, float]:
    """Solve one LP direction for every action in a temporary one-dimensional grid."""
    values = np.empty(len(action_grid), dtype=np.float64)
    max_marginal_error = 0.0
    for index, action in enumerate(action_grid):
        solved = solve_joint_bound(
            prepared_problem, float(action), B_minus, B_plus, bound_type
        )
        values[index] = solved["value"]
        max_marginal_error = max(max_marginal_error, solved["max_marginal_error"])
    return values, max_marginal_error


def _source_action_grid(
    atoms: dict,
    action_grid: np.ndarray,
    B_minus: float,
    B_plus: float,
    rho: float,
    bound_type: str,
) -> np.ndarray:
    radii = rho * np.abs(action_grid[:, None] - atoms["actions"][None, :])
    if bound_type == "upper":
        atom_bounds = np.minimum(B_plus, atoms["outcomes"][None, :] + radii)
    else:
        atom_bounds = np.maximum(B_minus, atoms["outcomes"][None, :] - radii)
    return atom_bounds @ atoms["probabilities"]


def solve_population_sequential_joint(
    reference: dict,
    horizon: int = 20,
    gamma: float = 0.95,
    kappa: float = 1.0,
    n_state: int = 21,
    n_action: int = 41,
) -> dict:
    """Run independent joint and source-wise population bound recursions."""
    _validate_grid_size("n_state", n_state)
    _validate_grid_size("n_action", n_action)
    if int(reference["horizon"]) != horizon or not np.isclose(
        float(reference["gamma"]), gamma, atol=0.0, rtol=0.0
    ):
        raise ValueError("reference horizon and gamma must match the requested recursion")
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and nonnegative")

    state_grid = np.linspace(-1.0, 1.0, n_state, dtype=np.float64)
    action_grid = np.linspace(-1.0, 1.0, n_action, dtype=np.float64)
    action_spacing = 2.0 / (n_action - 1)
    shape = (horizon + 2, n_state)
    joint_lower = np.zeros(shape)
    joint_upper = np.zeros(shape)
    joint_lower_raw = np.zeros(shape)
    joint_upper_raw = np.zeros(shape)
    joint_lower_argmax = np.full(shape, np.nan)
    joint_upper_argmax = np.full(shape, np.nan)
    source_lower = np.zeros((3, *shape))
    source_upper = np.zeros((3, *shape))
    upper_tuple_count = np.zeros(shape, dtype=np.int16)
    lower_tuple_count = np.zeros(shape, dtype=np.int16)
    value_lipschitz = value_lipschitz_constants(horizon, gamma)
    rho_coefficients = np.full(horizon + 2, np.nan)
    B_minus = np.zeros(horizon + 2)
    B_plus = np.zeros(horizon + 2)
    reference_value = np.zeros(shape)
    reference_error = np.asarray(reference["numerical_error_bound"], dtype=np.float64).copy()
    max_solver_error = 0.0
    max_true_error = 0.0
    joint_lp_calls = 0

    for h in range(1, horizon + 1):
        B_minus[h], B_plus[h] = return_bounds(h, horizon, gamma)
        rho_coefficients[h] = rho_coefficient(h, horizon, gamma)
        reference_value[h] = np.interp(
            state_grid, reference["state_grid"], reference["values"][h]
        )

    for h in range(horizon, 0, -1):
        upper_next = lambda x: evaluate_upper_extension(
            x, state_grid, joint_upper[h + 1], value_lipschitz[h + 1], B_plus[h + 1]
        )
        lower_next = lambda x: evaluate_lower_extension(
            x, state_grid, joint_lower[h + 1], value_lipschitz[h + 1], B_minus[h + 1]
        )
        source_lower_raw = np.empty((3, n_state))
        source_upper_raw = np.empty((3, n_state))
        rho = rho_coefficients[h]

        for state_index, state in enumerate(state_grid):
            upper_atoms = [
                build_source_atoms_from_continuation(
                    h, state, source, gamma, kappa, upper_next
                )
                for source in (1, 2, 3)
            ]
            upper_problem = prepare_joint_problem(upper_atoms, rho)
            upper_q, error = evaluate_joint_action_grid(
                upper_problem, action_grid, B_minus[h], B_plus[h], "upper"
            )
            joint_lp_calls += n_action
            max_solver_error = max(max_solver_error, error)
            max_true_error = max(
                max_true_error, upper_problem["max_true_coupling_marginal_error"]
            )
            upper_index = int(np.argmax(upper_q))
            joint_upper_raw[h, state_index] = min(
                B_plus[h], upper_q[upper_index] + rho * action_spacing / 2.0
            )
            joint_upper_argmax[h, state_index] = action_grid[upper_index]
            upper_tuple_count[h, state_index] = len(upper_problem["feasible_tuples"])

            lower_atoms = [
                build_source_atoms_from_continuation(
                    h, state, source, gamma, kappa, lower_next
                )
                for source in (1, 2, 3)
            ]
            lower_problem = prepare_joint_problem(lower_atoms, rho)
            lower_q, error = evaluate_joint_action_grid(
                lower_problem, action_grid, B_minus[h], B_plus[h], "lower"
            )
            joint_lp_calls += n_action
            max_solver_error = max(max_solver_error, error)
            max_true_error = max(
                max_true_error, lower_problem["max_true_coupling_marginal_error"]
            )
            lower_index = int(np.argmax(lower_q))
            joint_lower_raw[h, state_index] = lower_q[lower_index]
            joint_lower_argmax[h, state_index] = action_grid[lower_index]
            lower_tuple_count[h, state_index] = len(lower_problem["feasible_tuples"])

            for source_index, source_id in enumerate((1, 2, 3)):
                source_upper_next = lambda x, e=source_index: evaluate_upper_extension(
                    x, state_grid, source_upper[e, h + 1], value_lipschitz[h + 1], B_plus[h + 1]
                )
                source_lower_next = lambda x, e=source_index: evaluate_lower_extension(
                    x, state_grid, source_lower[e, h + 1], value_lipschitz[h + 1], B_minus[h + 1]
                )
                atoms_upper = build_source_atoms_from_continuation(
                    h, state, source_id, gamma, kappa, source_upper_next
                )
                atoms_lower = build_source_atoms_from_continuation(
                    h, state, source_id, gamma, kappa, source_lower_next
                )
                q_upper = _source_action_grid(
                    atoms_upper, action_grid, B_minus[h], B_plus[h], rho, "upper"
                )
                q_lower = _source_action_grid(
                    atoms_lower, action_grid, B_minus[h], B_plus[h], rho, "lower"
                )
                source_upper_raw[source_index, state_index] = min(
                    B_plus[h], np.max(q_upper) + rho * action_spacing / 2.0
                )
                source_lower_raw[source_index, state_index] = np.max(q_lower)

        joint_lower[h], joint_upper[h] = close_state_interval(
            state_grid,
            joint_lower_raw[h],
            joint_upper_raw[h],
            value_lipschitz[h],
            B_minus[h],
            B_plus[h],
        )
        if np.any(joint_lower[h] < joint_lower_raw[h] - 1e-10) or np.any(
            joint_upper[h] > joint_upper_raw[h] + 1e-10
        ):
            raise RuntimeError(f"joint state closure moved in an unsafe direction at h={h}")
        for source_index in range(3):
            source_lower[source_index, h], source_upper[source_index, h] = close_state_interval(
                state_grid,
                source_lower_raw[source_index],
                source_upper_raw[source_index],
                value_lipschitz[h],
                B_minus[h],
                B_plus[h],
            )
            if np.any(source_lower[source_index, h] < source_lower_raw[source_index] - 1e-10) or np.any(
                source_upper[source_index, h] > source_upper_raw[source_index] + 1e-10
            ):
                raise RuntimeError(
                    f"source {source_index + 1} state closure is unsafe at h={h}"
                )

    return {
        "state_grid": state_grid,
        "action_grid": action_grid,
        "joint_lower": joint_lower,
        "joint_upper": joint_upper,
        "joint_lower_raw": joint_lower_raw,
        "joint_upper_raw": joint_upper_raw,
        "joint_lower_argmax": joint_lower_argmax,
        "joint_upper_argmax": joint_upper_argmax,
        "source_lower": source_lower,
        "source_upper": source_upper,
        "reference_value": reference_value,
        "reference_error": reference_error,
        "B_minus": B_minus,
        "B_plus": B_plus,
        "value_lipschitz": value_lipschitz,
        "rho_coefficients": rho_coefficients,
        "upper_feasible_tuple_count": upper_tuple_count,
        "lower_feasible_tuple_count": lower_tuple_count,
        "max_solver_marginal_error": max_solver_error,
        "max_true_coupling_marginal_error": max_true_error,
        "joint_lp_calls": joint_lp_calls,
        "horizon": horizon,
        "gamma": gamma,
        "kappa": kappa,
    }
