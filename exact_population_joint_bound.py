"""Exact population one-step joint bounds for a fixed oracle continuation."""

from itertools import product

import numpy as np
from scipy.optimize import linprog

from confounded_smooth_regulator import behavior_action_fn, bellman_response_fn, return_bounds, rho_coefficient
from oracle_ground_truth import oracle_q


LATENT_TYPES = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def merge_observed_atoms(
    actions: np.ndarray, outcomes: np.ndarray, probabilities: np.ndarray,
    latent_types: tuple[tuple[int, int], ...] = LATENT_TYPES,
) -> dict:
    """Merge observationally identical atoms without retaining latent labels."""
    merged_actions: list[float] = []
    merged_outcomes: list[float] = []
    merged_probabilities: list[float] = []
    latent_to_atom = {}
    for latent, action, outcome, probability in zip(latent_types, actions, outcomes, probabilities):
        atom_index = None
        for index, (known_action, known_outcome) in enumerate(zip(merged_actions, merged_outcomes)):
            same_action = np.isclose(action, known_action, atol=1e-12, rtol=0.0)
            same_outcome = np.isclose(outcome, known_outcome, atol=1e-12, rtol=0.0)
            if same_action and same_outcome:
                atom_index = index
                break
        if atom_index is None:
            atom_index = len(merged_actions)
            merged_actions.append(float(action))
            merged_outcomes.append(float(outcome))
            merged_probabilities.append(0.0)
        merged_probabilities[atom_index] += float(probability)
        latent_to_atom[tuple(latent)] = atom_index
    return {
        "actions": np.asarray(merged_actions, dtype=np.float64),
        "outcomes": np.asarray(merged_outcomes, dtype=np.float64),
        "probabilities": np.asarray(merged_probabilities, dtype=np.float64),
        "latent_to_atom": latent_to_atom,
    }


def build_source_atoms(
    reference: dict, h: int, state: float, source_id: int, kappa: float = 1.0
) -> dict:
    """Enumerate one source's exact population law of observed ``(A, Z)``."""
    return_bounds(h, int(reference["horizon"]), float(reference["gamma"]))
    state_grid = reference["state_grid"]
    continuation_values = reference["values"][h + 1]
    value_next = lambda next_state: float(np.interp(next_state, state_grid, continuation_values))
    return build_source_atoms_from_continuation(
        h, state, source_id, float(reference["gamma"]), kappa, value_next
    )


def build_source_atoms_from_continuation(
    h: int,
    state: float,
    source_id: int,
    gamma: float,
    kappa: float,
    value_next,
) -> dict:
    """Enumerate one source's atoms under an arbitrary continuation callable."""
    actions = np.array(
        [behavior_action_fn(source_id, state, latent, kappa) for latent in LATENT_TYPES]
    )
    outcomes = np.array([
        bellman_response_fn(state, action, latent[0], gamma, value_next)
        for action, latent in zip(actions, LATENT_TYPES)
    ])
    atoms = merge_observed_atoms(actions, outcomes, np.full(4, 0.25))
    atoms.update({"h": int(h), "state": float(state), "source_id": int(source_id)})
    return atoms


def interpolated_value_lipschitz(reference: dict, h: int) -> float:
    """Return the maximum slope of the stored piecewise-linear value reference."""
    horizon = int(reference["horizon"])
    if not isinstance(h, (int, np.integer)) or isinstance(h, (bool, np.bool_)):
        raise ValueError("h must be an integer in [1, horizon + 1]")
    if not 1 <= h <= horizon + 1:
        raise ValueError("h must be an integer in [1, horizon + 1]")
    if h == horizon + 1:
        return 0.0
    slopes = np.diff(reference["values"][h]) / np.diff(reference["state_grid"])
    return float(np.max(np.abs(slopes)))


def reference_safe_rho(reference: dict, h: int) -> dict[str, float]:
    """Return analytic and interpolation-safe response-modulus diagnostics."""
    horizon, gamma = int(reference["horizon"]), float(reference["gamma"])
    return_bounds(h, horizon, gamma)
    analytic = rho_coefficient(h, horizon, gamma)
    stored = float(reference["rho_coefficients"][h])
    if not np.isclose(analytic, stored, atol=1e-12, rtol=0.0):
        raise RuntimeError("stored analytic rho coefficient is inconsistent")
    numerical = interpolated_value_lipschitz(reference, h + 1)
    next_lipschitz = max(float(reference["value_lipschitz"][h + 1]), numerical)
    return {"analytic_rho_coefficient": analytic,
            "reference_safe_rho_coefficient": 1.06 + gamma * 0.40 * next_lipschitz,
            "continuation_lipschitz_numeric": numerical}


def _all_marginal_error(problem: dict, weights: np.ndarray) -> float:
    errors = []
    tuples = problem["feasible_tuples"]
    for source_index, atoms in enumerate(problem["source_atoms"]):
        for atom_index, probability in enumerate(atoms["probabilities"]):
            marginal = weights[tuples[:, source_index] == atom_index].sum()
            errors.append(abs(marginal - probability))
    return float(max(errors, default=0.0))


def enumerate_feasible_tuples(
    source_atoms: list[dict],
    rho_coefficient_value: float,
    feasibility_tolerance: float = 1e-12,
) -> np.ndarray:
    """Enumerate tuples satisfying every pairwise response-modulus constraint."""
    candidates = product(*(range(len(atoms["actions"])) for atoms in source_atoms))
    feasible = []
    for atom_tuple in candidates:
        compatible = True
        for left, right in ((0, 1), (0, 2), (1, 2)):
            left_atom, right_atom = atom_tuple[left], atom_tuple[right]
            outcome_gap = abs(source_atoms[left]["outcomes"][left_atom]
                              - source_atoms[right]["outcomes"][right_atom])
            action_gap = abs(source_atoms[left]["actions"][left_atom]
                             - source_atoms[right]["actions"][right_atom])
            if outcome_gap > rho_coefficient_value * action_gap + feasibility_tolerance:
                compatible = False
                break
        if compatible:
            feasible.append(atom_tuple)
    return np.asarray(feasible, dtype=np.int64).reshape(-1, len(source_atoms))


def prepare_joint_problem(
    source_atoms: list[dict], rho_coefficient_value: float, feasibility_tolerance: float = 1e-12
) -> dict:
    """Enumerate compatible tuples and construct reduced marginal equalities."""
    if len(source_atoms) != 3:
        raise ValueError("source_atoms must contain exactly three sources")
    feasible_tuples = enumerate_feasible_tuples(
        source_atoms, rho_coefficient_value, feasibility_tolerance
    )
    if not len(feasible_tuples):
        raise RuntimeError("joint problem has no feasible tuples")

    rows, probabilities = [], []
    for source_index, atoms in enumerate(source_atoms):
        retained_count = len(atoms["actions"]) if source_index == 0 else len(atoms["actions"]) - 1
        for atom_index in range(retained_count):
            rows.append((feasible_tuples[:, source_index] == atom_index).astype(float))
            probabilities.append(atoms["probabilities"][atom_index])
    problem = {
        "source_atoms": source_atoms,
        "feasible_tuples": feasible_tuples,
        "A_eq": np.asarray(rows, dtype=np.float64),
        "b_eq": np.asarray(probabilities, dtype=np.float64),
        "rho_coefficient": float(rho_coefficient_value),
        "h": source_atoms[0].get("h", "unknown"),
        "state": source_atoms[0].get("state", "unknown"),
    }

    true_weights = np.zeros(len(feasible_tuples), dtype=np.float64)
    tuple_to_index = {tuple(atom_tuple): index for index, atom_tuple in enumerate(feasible_tuples)}
    for latent in LATENT_TYPES:
        true_tuple = tuple(atoms["latent_to_atom"][latent] for atoms in source_atoms)
        if true_tuple not in tuple_to_index:
            raise RuntimeError(f"true shared-U tuple {true_tuple} is infeasible")
        true_weights[tuple_to_index[true_tuple]] += 0.25
    problem["true_coupling_weights"] = true_weights
    problem["max_true_coupling_marginal_error"] = _all_marginal_error(problem, true_weights)
    if not np.isclose(true_weights.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("true shared-U coupling does not have unit mass")
    if problem["max_true_coupling_marginal_error"] > 1e-12:
        raise RuntimeError("true shared-U coupling does not reproduce source marginals")
    return problem


def compute_separate_intervals(
    source_atoms: list[dict], target_action: float, B_minus: float, B_plus: float,
    rho_coefficient_value: float,
) -> dict:
    """Compute the separate-rho intervals and their source-wise intersection."""
    source_lower, source_upper = [], []
    for atoms in source_atoms:
        radii = rho_coefficient_value * np.abs(target_action - atoms["actions"])
        source_lower.append(np.sum(
            atoms["probabilities"] * np.maximum(B_minus, atoms["outcomes"] - radii)))
        source_upper.append(np.sum(
            atoms["probabilities"] * np.minimum(B_plus, atoms["outcomes"] + radii)))
    lower = np.asarray(source_lower)
    upper = np.asarray(source_upper)
    return {
        "source_lower": lower,
        "source_upper": upper,
        "separate_lower": float(np.max(lower)),
        "separate_upper": float(np.min(upper)),
        "name": "separate_rho_intersection",
    }


def _tuple_envelopes(
    problem: dict, target_action: float, B_minus: float, B_plus: float
) -> tuple[np.ndarray, np.ndarray]:
    lowers, uppers = [], []
    for source_index, atoms in enumerate(problem["source_atoms"]):
        indices = problem["feasible_tuples"][:, source_index]
        radii = problem["rho_coefficient"] * np.abs(target_action - atoms["actions"][indices])
        lowers.append(atoms["outcomes"][indices] - radii)
        uppers.append(atoms["outcomes"][indices] + radii)
    tuple_lower = np.maximum(B_minus, np.max(lowers, axis=0))
    tuple_upper = np.minimum(B_plus, np.min(uppers, axis=0))
    return tuple_lower, tuple_upper


def _solve_lp(problem: dict, objective: np.ndarray, label: str):
    result = linprog(objective, A_eq=problem["A_eq"], b_eq=problem["b_eq"],
                     bounds=(0, None), method="highs")
    if not result.success:
        atom_counts = [len(atoms["actions"]) for atoms in problem["source_atoms"]]
        raise RuntimeError(
            f"{label} LP failed: h={problem['h']}, state={problem['state']}, "
            f"source atoms={atom_counts}, feasible tuples={len(problem['feasible_tuples'])}, "
            f"solver message={result.message}"
        )
    return result


def solve_joint_interval(
    prepared_problem: dict, target_action: float, B_minus: float, B_plus: float
) -> dict:
    """Solve exact population lower and upper coupling LPs."""
    lower = solve_joint_bound(prepared_problem, target_action, B_minus, B_plus, "lower")
    upper = solve_joint_bound(prepared_problem, target_action, B_minus, B_plus, "upper")
    return {
        "joint_lower": lower["value"], "joint_upper": upper["value"],
        "upper_coupling": upper["coupling"], "lower_coupling": lower["coupling"],
        "upper_solver_status": upper["solver_status"],
        "lower_solver_status": lower["solver_status"],
        "max_upper_marginal_error": upper["max_marginal_error"],
        "max_lower_marginal_error": lower["max_marginal_error"],
    }


def solve_joint_bound(
    prepared_problem: dict,
    target_action: float,
    B_minus: float,
    B_plus: float,
    bound_type: str,
) -> dict:
    """Solve only the requested upper or lower population coupling LP."""
    if bound_type not in {"upper", "lower"}:
        raise ValueError("bound_type must be 'upper' or 'lower'")
    tuple_lower, tuple_upper = _tuple_envelopes(
        prepared_problem, target_action, B_minus, B_plus
    )
    objective = -tuple_upper if bound_type == "upper" else tuple_lower
    result = _solve_lp(prepared_problem, objective, bound_type)
    marginal_error = _all_marginal_error(prepared_problem, result.x)
    if marginal_error >= 1e-8:
        raise RuntimeError(
            f"LP marginal audit failed: h={prepared_problem['h']}, "
            f"state={prepared_problem['state']}, {bound_type}={marginal_error}"
        )
    value = -result.fun if bound_type == "upper" else result.fun
    return {
        "value": float(value),
        "coupling": result.x,
        "solver_status": result.status,
        "max_marginal_error": marginal_error,
    }


def evaluate_true_coupling_envelope(
    prepared_problem: dict, target_action: float, B_minus: float, B_plus: float
) -> dict[str, float]:
    """Evaluate tuple envelopes under the audit-only shared-U coupling."""
    tuple_lower, tuple_upper = _tuple_envelopes(prepared_problem, target_action, B_minus, B_plus)
    weights = prepared_problem["true_coupling_weights"]
    return {"true_coupling_lower": float(weights @ tuple_lower),
            "true_coupling_upper": float(weights @ tuple_upper)}


def evaluate_population_query(
    reference: dict, h: int, state: float, target_action: float, kappa: float = 1.0
) -> dict:
    """Evaluate all one-step population intervals for one fixed query."""
    horizon, gamma = int(reference["horizon"]), float(reference["gamma"])
    B_minus, B_plus = return_bounds(h, horizon, gamma)
    source_atoms = [
        build_source_atoms(reference, h, state, source_id, kappa) for source_id in (1, 2, 3)
    ]
    rho_diagnostics = reference_safe_rho(reference, h)
    safe_rho = rho_diagnostics["reference_safe_rho_coefficient"]
    problem = prepare_joint_problem(source_atoms, safe_rho)
    joint = solve_joint_interval(problem, target_action, B_minus, B_plus)
    separate = compute_separate_intervals(source_atoms, target_action, B_minus, B_plus, safe_rho)
    true_envelope = evaluate_true_coupling_envelope(problem, target_action, B_minus, B_plus)
    result = {
        "h": int(h),
        "state": float(state),
        "target_action": float(target_action),
        "reference_q": float(oracle_q(reference, h, state, target_action)),
        **joint,
        **separate,
        **true_envelope,
        **rho_diagnostics,
        "number_of_atoms_per_source": np.array([len(a["actions"]) for a in source_atoms]),
        "number_of_feasible_tuples": len(problem["feasible_tuples"]),
        "max_solver_marginal_error": max(joint["max_upper_marginal_error"],
                                         joint["max_lower_marginal_error"]),
    }
    result["upper_gain"] = result["separate_upper"] - result["joint_upper"]
    result["lower_gain"] = result["joint_lower"] - result["separate_lower"]
    result["width_gain"] = (result["separate_upper"] - result["separate_lower"]
                            - result["joint_upper"] + result["joint_lower"])
    return result
