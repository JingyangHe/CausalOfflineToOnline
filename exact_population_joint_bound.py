"""Exact population one-step joint bounds for a fixed oracle continuation."""

from itertools import combinations, product

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


def _action_distance(left: np.ndarray | float, right: np.ndarray | float) -> float:
    """Euclidean action distance shared by scalar and continuous-control LPs."""
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64)
                                - np.asarray(right, dtype=np.float64)))


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
        for left, right in combinations(range(len(source_atoms)), 2):
            left_atom, right_atom = atom_tuple[left], atom_tuple[right]
            outcome_gap = abs(source_atoms[left]["outcomes"][left_atom]
                              - source_atoms[right]["outcomes"][right_atom])
            action_gap = _action_distance(source_atoms[left]["actions"][left_atom],
                                          source_atoms[right]["actions"][right_atom])
            if outcome_gap > rho_coefficient_value * action_gap + feasibility_tolerance:
                compatible = False
                break
        if compatible:
            feasible.append(atom_tuple)
    return np.asarray(feasible, dtype=np.int64).reshape(-1, len(source_atoms))


def prepare_empirical_coupling_problem(
    source_atoms: list[dict], rho_coefficient_value: float,
    feasibility_tolerance: float = 1e-12,
) -> dict:
    """Build the shared marginal LP for generic weighted continuous-action atoms.

    This is the latent-free core used by the Hopper pilot.  The legacy oracle
    wrapper below adds its true-coupling audit without changing this LP.
    """
    if len(source_atoms) < 2:
        raise ValueError("source_atoms must contain at least two sources")
    normalized_atoms = []
    for atoms in source_atoms:
        actions = np.asarray(atoms["actions"], dtype=np.float64)
        outcomes = np.asarray(atoms["outcomes"], dtype=np.float64).reshape(-1)
        probabilities = np.asarray(atoms["probabilities"], dtype=np.float64).reshape(-1)
        if actions.ndim not in (1, 2) or len(actions) != len(outcomes) or len(actions) != len(probabilities):
            raise ValueError("atom actions, outcomes, and probabilities have incompatible shapes")
        if len(actions) == 0 or not np.all(np.isfinite(actions)) or not np.all(np.isfinite(outcomes)):
            raise ValueError("atoms must be nonempty and finite")
        if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
            raise ValueError("atom probabilities must be nonnegative and sum to one")
        normalized_atoms.append({**atoms, "actions": actions, "outcomes": outcomes,
                                 "probabilities": probabilities})

    feasible_tuples = enumerate_feasible_tuples(
        normalized_atoms, rho_coefficient_value, feasibility_tolerance
    )
    if not len(feasible_tuples):
        raise RuntimeError("joint problem has no feasible tuples")
    rows, probabilities = [], []
    for source_index, atoms in enumerate(normalized_atoms):
        retained_count = len(atoms["actions"]) if source_index == 0 else len(atoms["actions"]) - 1
        for atom_index in range(retained_count):
            rows.append((feasible_tuples[:, source_index] == atom_index).astype(float))
            probabilities.append(atoms["probabilities"][atom_index])
    return {
        "source_atoms": normalized_atoms,
        "feasible_tuples": feasible_tuples,
        "A_eq": np.asarray(rows, dtype=np.float64),
        "b_eq": np.asarray(probabilities, dtype=np.float64),
        "rho_coefficient": float(rho_coefficient_value),
        "h": normalized_atoms[0].get("h", "unknown"),
        "state": normalized_atoms[0].get("state", "unknown"),
    }


def prepare_joint_problem(
    source_atoms: list[dict], rho_coefficient_value: float, feasibility_tolerance: float = 1e-12
) -> dict:
    """Enumerate compatible tuples and construct reduced marginal equalities."""
    if len(source_atoms) != 3:
        raise ValueError("source_atoms must contain exactly three sources")
    problem = prepare_empirical_coupling_problem(
        source_atoms, rho_coefficient_value, feasibility_tolerance
    )
    feasible_tuples = problem["feasible_tuples"]

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
    result = compute_empirical_separate_interval(
        source_atoms, np.asarray([target_action]), rho_coefficient_value, B_minus, B_plus
    )
    result["name"] = "separate_rho_intersection"
    return result


def _tuple_envelopes(
    problem: dict, target_action: float, B_minus: float, B_plus: float
) -> tuple[np.ndarray, np.ndarray]:
    return empirical_tuple_envelopes(
        problem, np.asarray([target_action]), B_minus=B_minus, B_plus=B_plus
    )


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


def compute_empirical_separate_interval(
    source_atoms: list[dict], target_action: np.ndarray, rho_coefficient_value: float,
    B_minus: float | None = None, B_plus: float | None = None,
) -> dict:
    """Unclipped Separate intersection for generic empirical atoms."""
    source_lower, source_upper = [], []
    for atoms in source_atoms:
        actions = np.asarray(atoms["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[:, None]
        target = np.asarray(target_action, dtype=np.float64).reshape(1, -1)
        radii = float(rho_coefficient_value) * np.linalg.norm(actions - target, axis=1)
        weights = np.asarray(atoms["probabilities"], dtype=np.float64)
        outcomes = np.asarray(atoms["outcomes"], dtype=np.float64)
        lower_values, upper_values = outcomes - radii, outcomes + radii
        if B_minus is not None:
            lower_values = np.maximum(float(B_minus), lower_values)
        if B_plus is not None:
            upper_values = np.minimum(float(B_plus), upper_values)
        source_lower.append(float(weights @ lower_values))
        source_upper.append(float(weights @ upper_values))
    lower, upper = np.asarray(source_lower), np.asarray(source_upper)
    return {"source_lower": lower, "source_upper": upper,
            "separate_lower": float(np.max(lower)),
            "separate_upper": float(np.min(upper))}


def empirical_tuple_envelopes(
    prepared_problem: dict, target_action: np.ndarray,
    B_minus: float | None = None, B_plus: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unclipped lower/upper envelopes for generic feasible tuples."""
    lowers, uppers = [], []
    target = np.asarray(target_action, dtype=np.float64).reshape(1, -1)
    for source_index, atoms in enumerate(prepared_problem["source_atoms"]):
        indices = prepared_problem["feasible_tuples"][:, source_index]
        actions = np.asarray(atoms["actions"], dtype=np.float64)[indices]
        if actions.ndim == 1:
            actions = actions[:, None]
        radii = prepared_problem["rho_coefficient"] * np.linalg.norm(actions - target, axis=1)
        outcomes = np.asarray(atoms["outcomes"], dtype=np.float64)[indices]
        lowers.append(outcomes - radii)
        uppers.append(outcomes + radii)
    lower, upper = np.max(lowers, axis=0), np.min(uppers, axis=0)
    if B_minus is not None:
        lower = np.maximum(float(B_minus), lower)
    if B_plus is not None:
        upper = np.minimum(float(B_plus), upper)
    return lower, upper


def solve_empirical_joint_interval(prepared_problem: dict, target_action: np.ndarray) -> dict:
    """Solve the exact unclipped empirical Joint interval with existing LP audits."""
    tuple_lower, tuple_upper = empirical_tuple_envelopes(prepared_problem, target_action)
    lower = _solve_lp(prepared_problem, tuple_lower, "lower")
    upper = _solve_lp(prepared_problem, -tuple_upper, "upper")
    lower_error = _all_marginal_error(prepared_problem, lower.x)
    upper_error = _all_marginal_error(prepared_problem, upper.x)
    if max(lower_error, upper_error) >= 1e-8:
        raise RuntimeError(f"LP marginal audit failed: {max(lower_error, upper_error)}")
    return {
        "joint_lower": float(lower.fun), "joint_upper": float(-upper.fun),
        "lower_coupling": lower.x, "upper_coupling": upper.x,
        "lower_solver_status": int(lower.status), "upper_solver_status": int(upper.status),
        "max_lower_marginal_error": lower_error, "max_upper_marginal_error": upper_error,
    }


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
