"""Finite-sample fixed-state one-step joint bounds with exact support atoms."""

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta

from confounded_smooth_regulator import return_bounds
from exact_population_joint_bound import (
    _tuple_envelopes,
    enumerate_feasible_tuples,
    evaluate_population_query,
    merge_observed_atoms,
    reference_safe_rho,
)
from oracle_ground_truth import oracle_q


def build_empirical_source_atoms(train_slice: dict, reference: dict, h: int) -> dict:
    """Merge a single source/state/stage sample into observed ``(A, Z)`` atoms."""
    lower_keys = {key.lower() for key in train_slice}
    forbidden = lower_keys & {"c", "w", "u", "confounder_c", "randomizer_w", "exogenous"}
    if forbidden or any("confounder" in key or "randomizer" in key for key in lower_keys):
        raise ValueError("train_slice contains hidden audit fields")
    required = {"time_step", "state", "source_id", "action", "reward", "next_state"}
    if not required <= set(train_slice):
        raise ValueError("train_slice is missing required public fields")
    sample_size = len(train_slice["action"])
    if sample_size <= 0 or any(len(train_slice[key]) != sample_size for key in required):
        raise ValueError("train_slice arrays must have one common positive length")
    if not np.all(np.asarray(train_slice["time_step"]) == h):
        raise ValueError("train_slice must contain one requested time step")
    if len(np.unique(train_slice["state"])) != 1 or len(np.unique(train_slice["source_id"])) != 1:
        raise ValueError("train_slice must contain one state and one source")
    continuation = np.interp(
        train_slice["next_state"], reference["state_grid"], reference["values"][h + 1]
    )
    outcomes = np.asarray(train_slice["reward"], dtype=np.float64) + float(
        reference["gamma"]
    ) * continuation
    labels = tuple((index, 0) for index in range(sample_size))
    merged = merge_observed_atoms(
        np.asarray(train_slice["action"], dtype=np.float64),
        outcomes,
        np.ones(sample_size),
        labels,
    )
    counts = merged["probabilities"].astype(np.int64)
    return {
        "actions": merged["actions"],
        "outcomes": merged["outcomes"],
        "counts": counts,
        "empirical_probabilities": counts / sample_size,
        "sample_size": sample_size,
        "h": int(h),
        "state": float(train_slice["state"][0]),
        "source_id": int(train_slice["source_id"][0]),
    }


def support_failure_bound(sample_size: int) -> float:
    """Return the union bound for missing any of twelve latent source-types."""
    if not isinstance(sample_size, (int, np.integer)) or isinstance(
        sample_size, (bool, np.bool_)
    ) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    return float(min(1.0, 12.0 * 0.75 ** int(sample_size)))


def clopper_pearson_interval(count: int, sample_size: int, alpha: float) -> tuple[float, float]:
    """Return the two-sided exact binomial confidence interval."""
    if not 0 <= count <= sample_size or sample_size <= 0:
        raise ValueError("count must be in [0, sample_size] with positive sample_size")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    lower = 0.0 if count == 0 else float(beta.ppf(alpha / 2.0, count, sample_size - count + 1))
    upper = 1.0 if count == sample_size else float(
        beta.ppf(1.0 - alpha / 2.0, count + 1, sample_size - count)
    )
    if not 0.0 <= lower <= upper <= 1.0:
        raise RuntimeError("invalid Clopper-Pearson interval")
    return lower, upper


def build_probability_intervals(
    empirical_source_atoms: list[dict], delta: float = 0.05
) -> dict:
    """Allocate support and simultaneous atom-probability failure budgets."""
    if len(empirical_source_atoms) != 3:
        raise ValueError("exactly three empirical source distributions are required")
    sample_sizes = [atoms["sample_size"] for atoms in empirical_source_atoms]
    if len(set(sample_sizes)) != 1:
        raise ValueError("all three sources must have the same sample size")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    delta_support = support_failure_bound(sample_sizes[0])
    certified = delta_support < delta
    delta_probability = delta - delta_support if certified else 0.0
    atom_count = sum(len(atoms["actions"]) for atoms in empirical_source_atoms)
    alpha_per_atom = delta_probability / atom_count if certified else 0.0
    probability_lower, probability_upper = [], []
    for atoms in empirical_source_atoms:
        if certified:
            intervals = [
                clopper_pearson_interval(int(count), atoms["sample_size"], alpha_per_atom)
                for count in atoms["counts"]
            ]
            probability_lower.append(np.array([interval[0] for interval in intervals]))
            probability_upper.append(np.array([interval[1] for interval in intervals]))
        else:
            probability_lower.append(np.zeros(len(atoms["actions"])))
            probability_upper.append(np.ones(len(atoms["actions"])))
    return {"probability_lower": probability_lower, "probability_upper": probability_upper,
            "delta": float(delta), "delta_support": delta_support,
            "delta_probability": delta_probability, "alpha_per_atom": alpha_per_atom,
            "certified": certified}


def prepare_finite_sample_joint_problem(
    empirical_source_atoms: list[dict],
    probability_intervals: dict,
    rho_coefficient_value: float,
    feasibility_tolerance: float = 1e-12,
) -> dict:
    """Construct tuple compatibility and interval-valued marginal constraints."""
    feasible = enumerate_feasible_tuples(
        empirical_source_atoms, rho_coefficient_value, feasibility_tolerance
    )
    rows, limits = [], []
    for source_index, atoms in enumerate(empirical_source_atoms):
        for atom_index in range(len(atoms["actions"])):
            row = (feasible[:, source_index] == atom_index).astype(float)
            rows.extend((row, -row))
            limits.extend(
                (
                    probability_intervals["probability_upper"][source_index][atom_index],
                    -probability_intervals["probability_lower"][source_index][atom_index],
                )
            )
    return {
        "source_atoms": empirical_source_atoms,
        "feasible_tuples": feasible,
        "A_eq": np.ones((1, len(feasible))),
        "b_eq": np.ones(1),
        "A_ub": np.asarray(rows, dtype=np.float64),
        "b_ub": np.asarray(limits, dtype=np.float64),
        "rho_coefficient": float(rho_coefficient_value),
        "probability_intervals": probability_intervals,
    }


def _fallback_joint(B_minus: float, B_plus: float) -> dict:
    return {
        "joint_lower": float(B_minus), "joint_upper": float(B_plus),
        "upper_coupling": np.array([]), "lower_coupling": np.array([]),
        "max_marginal_interval_violation": 0.0,
        "solver_status": "infeasible", "fallback_reason": "infeasible_confidence_problem",
    }


def _marginal_interval_violation(problem: dict, coupling: np.ndarray) -> float:
    violation = 0.0
    for source_index, atoms in enumerate(problem["source_atoms"]):
        for atom_index in range(len(atoms["actions"])):
            selected = problem["feasible_tuples"][:, source_index] == atom_index
            marginal = coupling[selected].sum()
            lower = problem["probability_intervals"]["probability_lower"][source_index][atom_index]
            upper = problem["probability_intervals"]["probability_upper"][source_index][atom_index]
            violation = max(violation, lower - marginal, marginal - upper)
    return float(max(0.0, violation))


def solve_finite_sample_joint_interval(
    prepared_problem: dict, target_action: float, B_minus: float, B_plus: float
) -> dict:
    """Optimize both joint directions over interval-valued source marginals."""
    if not len(prepared_problem["feasible_tuples"]):
        return _fallback_joint(B_minus, B_plus)
    tuple_lower, tuple_upper = _tuple_envelopes(
        prepared_problem, target_action, B_minus, B_plus
    )
    options = {"A_ub": prepared_problem["A_ub"], "b_ub": prepared_problem["b_ub"],
               "A_eq": prepared_problem["A_eq"], "b_eq": prepared_problem["b_eq"],
               "bounds": (0, None), "method": "highs"}
    lower = linprog(tuple_lower, **options)
    upper = linprog(-tuple_upper, **options)
    if lower.status == 2 or upper.status == 2:
        return _fallback_joint(B_minus, B_plus)
    if not lower.success or not upper.success:
        raise RuntimeError(f"finite-sample joint LP failed: {lower.message}; {upper.message}")
    violation = max(
        _marginal_interval_violation(prepared_problem, lower.x),
        _marginal_interval_violation(prepared_problem, upper.x),
    )
    return {
        "joint_lower": float(lower.fun), "joint_upper": float(-upper.fun),
        "upper_coupling": upper.x, "lower_coupling": lower.x,
        "max_marginal_interval_violation": violation,
        "solver_status": (lower.status, upper.status), "fallback_reason": None,
    }


def solve_finite_sample_separate_interval(
    empirical_source_atoms: list[dict], probability_intervals: dict,
    target_action: float, B_minus: float, B_plus: float, rho_coefficient_value: float,
) -> dict:
    """Solve independent robust source intervals and intersect them."""
    source_lower, source_upper = [], []
    fallback = False
    for source_index, atoms in enumerate(empirical_source_atoms):
        radii = rho_coefficient_value * np.abs(target_action - atoms["actions"])
        lower_values = np.maximum(B_minus, atoms["outcomes"] - radii)
        upper_values = np.minimum(B_plus, atoms["outcomes"] + radii)
        bounds = list(zip(probability_intervals["probability_lower"][source_index],
                          probability_intervals["probability_upper"][source_index]))
        options = {"A_eq": np.ones((1, len(bounds))), "b_eq": np.ones(1),
                   "bounds": bounds, "method": "highs"}
        lower, upper = linprog(lower_values, **options), linprog(-upper_values, **options)
        if lower.status == 2 or upper.status == 2:
            source_lower.append(B_minus)
            source_upper.append(B_plus)
            fallback = True
        elif lower.success and upper.success:
            source_lower.append(lower.fun)
            source_upper.append(-upper.fun)
        else:
            raise RuntimeError(f"finite-sample source LP failed: {lower.message}; {upper.message}")
    lower_array, upper_array = np.asarray(source_lower), np.asarray(source_upper)
    return {
        "source_lower": lower_array, "source_upper": upper_array,
        "separate_lower": float(np.max(lower_array)),
        "separate_upper": float(np.min(upper_array)),
        "fallback_reason": "infeasible_source_probability_problem" if fallback else None,
    }


def _query_slice(train_data: dict, h: int, state: float, source_id: int) -> dict:
    selected = (
        (np.asarray(train_data["time_step"]) == h)
        & (np.asarray(train_data["state"]) == state)
        & (np.asarray(train_data["source_id"]) == source_id)
    )
    return {key: np.asarray(values)[selected] for key, values in train_data.items()}


def evaluate_finite_sample_query(
    train_data: dict, reference: dict, h: int, state: float,
    target_action: float, delta: float = 0.05,
) -> dict:
    """Evaluate one finite-sample query using public fixed-state data only."""
    atoms = [
        build_empirical_source_atoms(_query_slice(train_data, h, state, source), reference, h)
        for source in (1, 2, 3)
    ]
    sample_sizes = [source_atoms["sample_size"] for source_atoms in atoms]
    if len(set(sample_sizes)) != 1:
        raise ValueError("all three sources must have equal sample sizes")
    intervals = build_probability_intervals(atoms, delta)
    horizon, gamma = int(reference["horizon"]), float(reference["gamma"])
    B_minus, B_plus = return_bounds(h, horizon, gamma)
    rho = reference_safe_rho(reference, h)["reference_safe_rho_coefficient"]
    if intervals["certified"]:
        problem = prepare_finite_sample_joint_problem(atoms, intervals, rho)
        joint = solve_finite_sample_joint_interval(problem, target_action, B_minus, B_plus)
        separate = solve_finite_sample_separate_interval(
            atoms, intervals, target_action, B_minus, B_plus, rho
        )
        fallback_reason = joint["fallback_reason"] or separate["fallback_reason"]
        max_violation = joint["max_marginal_interval_violation"]
    else:
        joint = {"joint_lower": B_minus, "joint_upper": B_plus}
        separate = {"separate_lower": B_minus, "separate_upper": B_plus}
        fallback_reason = "insufficient_sample_support_guarantee"
        max_violation = 0.0
    population = evaluate_population_query(reference, h, state, target_action)
    result = {
        "h": int(h), "state": float(state), "target_action": float(target_action),
        "sample_size": sample_sizes[0], "delta": float(delta),
        "certified": intervals["certified"], "delta_support": intervals["delta_support"],
        "delta_probability": intervals["delta_probability"],
        "alpha_per_atom": intervals["alpha_per_atom"],
        "empirical_atom_counts": np.array([len(item["actions"]) for item in atoms]),
        "finite_joint_lower": float(joint["joint_lower"]),
        "finite_joint_upper": float(joint["joint_upper"]),
        "finite_separate_lower": float(separate["separate_lower"]),
        "finite_separate_upper": float(separate["separate_upper"]),
        "population_joint_lower": population["joint_lower"],
        "population_joint_upper": population["joint_upper"],
        "reference_q": float(oracle_q(reference, h, state, target_action)),
        "fallback_reason": fallback_reason, "max_solver_violation": max_violation,
    }
    result["joint_width"] = result["finite_joint_upper"] - result["finite_joint_lower"]
    result["separate_width"] = result["finite_separate_upper"] - result["finite_separate_lower"]
    result["population_joint_width"] = (
        result["population_joint_upper"] - result["population_joint_lower"]
    )
    result["width_inflation_over_population"] = (
        result["joint_width"] - result["population_joint_width"]
    )
    result["finite_upper_gain"] = result["finite_separate_upper"] - result["finite_joint_upper"]
    result["finite_lower_gain"] = result["finite_joint_lower"] - result["finite_separate_lower"]
    result["finite_width_gain"] = result["separate_width"] - result["joint_width"]
    return result
