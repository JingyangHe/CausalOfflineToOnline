"""Validate exact population one-step joint bounds on the frozen query grid."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from confounded_smooth_regulator import return_bounds  # noqa: E402
from exact_population_joint_bound import (  # noqa: E402
    build_source_atoms,
    compute_separate_intervals,
    evaluate_true_coupling_envelope,
    prepare_joint_problem,
    reference_safe_rho,
    solve_joint_interval,
)
from oracle_ground_truth import oracle_q  # noqa: E402


SCALAR_FIELDS = (
    "h",
    "state",
    "action",
    "reference_q",
    "joint_lower",
    "joint_upper",
    "separate_lower",
    "separate_upper",
    "true_coupling_lower",
    "true_coupling_upper",
    "analytic_rho_coefficient",
    "reference_safe_rho_coefficient",
    "feasible_tuple_count",
    "upper_gain",
    "lower_gain",
    "width_gain",
)


def load_reference(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required Phase 1B oracle artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def validate_query(record: dict, B_minus: float, B_plus: float) -> tuple[bool, bool, bool]:
    tolerance = 1e-8
    q = record["reference_q"]
    bounds = np.concatenate(
        (
            record["source_lower"],
            record["source_upper"],
            [
                record["joint_lower"],
                record["joint_upper"],
                record["separate_lower"],
                record["separate_upper"],
                record["true_coupling_lower"],
                record["true_coupling_upper"],
            ],
        )
    )
    validity = (
        record["joint_lower"] <= q + tolerance
        and q <= record["joint_upper"] + tolerance
        and record["separate_lower"] <= q + tolerance
        and q <= record["separate_upper"] + tolerance
        and record["joint_lower"] <= record["joint_upper"] + tolerance
        and np.all(bounds >= B_minus - tolerance)
        and np.all(bounds <= B_plus + tolerance)
    )
    dominance = (
        record["joint_lower"] >= record["separate_lower"] - tolerance
        and record["joint_upper"] <= record["separate_upper"] + tolerance
    )
    true_chain = (
        record["joint_lower"] <= record["true_coupling_lower"] + tolerance
        and record["true_coupling_lower"] <= q + tolerance
        and q <= record["true_coupling_upper"] + tolerance
        and record["true_coupling_upper"] <= record["joint_upper"] + tolerance
    )
    return validity, dominance, true_chain


def run_queries(reference: dict) -> tuple[list[dict], dict[str, int]]:
    horizon, gamma = int(reference["horizon"]), float(reference["gamma"])
    records = []
    counts = {
        "total_queries": 0,
        "lp_failures": 0,
        "validity_violations": 0,
        "dominance_violations": 0,
        "true_coupling_violations": 0,
    }
    for h in (1, 5, 10, 15, 20):
        B_minus, B_plus = return_bounds(h, horizon, gamma)
        rho_diagnostics = reference_safe_rho(reference, h)
        safe_rho = rho_diagnostics["reference_safe_rho_coefficient"]
        for state in np.linspace(-0.8, 0.8, 9):
            source_atoms = [
                build_source_atoms(reference, h, state, source) for source in (1, 2, 3)
            ]
            problem = prepare_joint_problem(source_atoms, safe_rho)
            if not len(problem["feasible_tuples"]):
                raise RuntimeError(f"no feasible tuple for h={h}, state={state}")
            for action in np.linspace(-1.0, 1.0, 17):
                counts["total_queries"] += 1
                try:
                    joint = solve_joint_interval(problem, action, B_minus, B_plus)
                except RuntimeError as exc:
                    counts["lp_failures"] += 1
                    print(f"LP_FAILURE h={h} state={state} action={action}: {exc}")
                    continue
                separate = compute_separate_intervals(
                    source_atoms, action, B_minus, B_plus, safe_rho
                )
                true_envelope = evaluate_true_coupling_envelope(
                    problem, action, B_minus, B_plus
                )
                record = {
                    "h": h,
                    "state": float(state),
                    "action": float(action),
                    "reference_q": float(oracle_q(reference, h, state, action)),
                    **joint,
                    **separate,
                    **true_envelope,
                    **rho_diagnostics,
                    "feasible_tuple_count": len(problem["feasible_tuples"]),
                }
                record["upper_gain"] = record["separate_upper"] - record["joint_upper"]
                record["lower_gain"] = record["joint_lower"] - record["separate_lower"]
                record["width_gain"] = (
                    record["separate_upper"]
                    - record["separate_lower"]
                    - record["joint_upper"]
                    + record["joint_lower"]
                )
                validity, dominance, true_chain = validate_query(record, B_minus, B_plus)
                counts["validity_violations"] += int(not validity)
                counts["dominance_violations"] += int(not dominance)
                counts["true_coupling_violations"] += int(not true_chain)
                records.append(record)
    return records, counts


def save_audit(records: list[dict], path: Path) -> None:
    artifact = {key: np.asarray([record[key] for record in records]) for key in SCALAR_FIELDS}
    artifact["source_lower"] = np.stack([record["source_lower"] for record in records])
    artifact["source_upper"] = np.stack([record["source_upper"] for record in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **artifact)


def print_statistics(records: list[dict], counts: dict[str, int]) -> tuple[float, bool]:
    values = lambda key: np.asarray([record[key] for record in records])
    joint_width = values("joint_upper") - values("joint_lower")
    separate_width = values("separate_upper") - values("separate_lower")
    upper_gain, lower_gain, width_gain = map(values, ("upper_gain", "lower_gain", "width_gain"))
    tuple_counts = values("feasible_tuple_count")
    marginal_error = max(
        max(record["max_upper_marginal_error"], record["max_lower_marginal_error"])
        for record in records
    )
    rho_inflation = values("reference_safe_rho_coefficient") - values(
        "analytic_rho_coefficient"
    )
    strict = (upper_gain > 1e-10) | (lower_gain > 1e-10) | (width_gain > 1e-10)
    for key, value in counts.items():
        print(f"{key}={value}")
    print(f"max_solver_marginal_error={marginal_error:.3e}")
    print(f"mean_joint_width={joint_width.mean():.10f}")
    print(f"mean_separate_width={separate_width.mean():.10f}")
    for name, gain in (("upper", upper_gain), ("lower", lower_gain), ("width", width_gain)):
        print(f"mean_{name}_gain={gain.mean():.10f}")
        print(f"max_{name}_gain={gain.max():.10f}")
        print(f"fraction_{name}_strict={np.mean(gain > 1e-10):.10f}")
    print(f"mean_feasible_tuple_count={tuple_counts.mean():.6f}")
    print(f"min_feasible_tuple_count={int(tuple_counts.min())}")
    print(f"max_feasible_tuple_count={int(tuple_counts.max())}")
    print(f"max_reference_rho_inflation={rho_inflation.max():.3e}")
    return marginal_error, bool(np.any(strict))


def main() -> int:
    try:
        reference = load_reference(ROOT / "artifacts" / "phase1b" / "oracle_reference.npz")
        records, counts = run_queries(reference)
        require_counts = (
            counts["total_queries"] == 765
            and counts["lp_failures"] == 0
            and counts["validity_violations"] == 0
            and counts["dominance_violations"] == 0
            and counts["true_coupling_violations"] == 0
        )
        marginal_error, strict_gain = print_statistics(records, counts)
        if not require_counts or marginal_error > 1e-8:
            raise RuntimeError("one or more Phase 2A acceptance checks failed")
        save_audit(records, ROOT / "artifacts" / "phase2a" / "population_joint_audit.npz")
    except Exception as exc:
        print(f"FAIL {exc}")
        print("STRICT_GAIN_OBSERVED = False")
        print("PHASE2A_MISMATCH")
        return 1
    print(f"STRICT_GAIN_OBSERVED = {strict_gain}")
    print("PHASE2A_FAITHFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
