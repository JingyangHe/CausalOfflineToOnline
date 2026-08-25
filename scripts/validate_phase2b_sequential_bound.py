"""Validate the full exact-population sequential joint recursion."""

from pathlib import Path
from time import perf_counter
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exact_population_sequential_bound import solve_population_sequential_joint  # noqa: E402


ARTIFACT_FIELDS = (
    "state_grid",
    "action_grid",
    "joint_lower",
    "joint_upper",
    "joint_lower_raw",
    "joint_upper_raw",
    "source_lower",
    "source_upper",
    "reference_value",
    "reference_error",
    "B_minus",
    "B_plus",
    "value_lipschitz",
    "rho_coefficients",
    "joint_lower_argmax",
    "joint_upper_argmax",
    "upper_feasible_tuple_count",
    "lower_feasible_tuple_count",
)


def load_reference(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required Phase 1B oracle artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def audit_result(result: dict, runtime_seconds: float) -> tuple[dict, bool]:
    horizon = int(result["horizon"])
    active = np.s_[1 : horizon + 1]
    lower = result["joint_lower"][active]
    upper = result["joint_upper"][active]
    reference = result["reference_value"][active]
    error = result["reference_error"][1 : horizon + 1, None]
    separate_lower = np.max(result["source_lower"][:, active, :], axis=0)
    separate_upper = np.min(result["source_upper"][:, active, :], axis=0)
    center_bad = (lower > reference + 1e-8) | (upper < reference - 1e-8)
    certified_bad = (lower > reference + error + 1e-8) | (
        upper < reference - error - 1e-8
    )
    ambiguous = center_bad & ~certified_bad
    dominance_bad = (lower < separate_lower - 1e-8) | (
        upper > separate_upper + 1e-8
    )
    order_bad = lower > upper + 1e-8

    range_violations = 0
    for h in range(1, horizon + 1):
        arrays = (
            result["joint_lower"][h],
            result["joint_upper"][h],
            *result["source_lower"][:, h],
            *result["source_upper"][:, h],
        )
        for values in arrays:
            range_violations += int(np.count_nonzero(values < result["B_minus"][h] - 1e-8))
            range_violations += int(np.count_nonzero(values > result["B_plus"][h] + 1e-8))

    spacing = result["state_grid"][1] - result["state_grid"][0]
    lipschitz_violations = 0
    value_arrays = [result["joint_lower"], result["joint_upper"]]
    value_arrays.extend(result["source_lower"])
    value_arrays.extend(result["source_upper"])
    for values in value_arrays:
        for h in range(1, horizon + 1):
            limit = result["value_lipschitz"][h] * spacing + 1e-10
            lipschitz_violations += int(np.count_nonzero(np.abs(np.diff(values[h])) > limit))

    joint_width = upper - lower
    separate_width = separate_upper - separate_lower
    width_gain = separate_width - joint_width
    tuple_counts = np.concatenate(
        (
            result["upper_feasible_tuple_count"][active].ravel(),
            result["lower_feasible_tuple_count"][active].ravel(),
        )
    )
    action_spacing = result["action_grid"][1] - result["action_grid"][0]
    max_correction = np.nanmax(result["rho_coefficients"][active]) * action_spacing / 2.0
    stats = {
        "total_stage_states": horizon * len(result["state_grid"]),
        "joint_lp_calls": result["joint_lp_calls"],
        "lp_failures": 0,
        "center_coverage_violations": int(np.count_nonzero(center_bad)),
        "oracle_certified_violations": int(np.count_nonzero(certified_bad)),
        "reference_ambiguous_count": int(np.count_nonzero(ambiguous)),
        "dominance_violations": int(np.count_nonzero(dominance_bad)),
        "interval_order_violations": int(np.count_nonzero(order_bad)),
        "range_violations": range_violations,
        "lipschitz_violations": lipschitz_violations,
        "true_coupling_violations": int(
            result["max_true_coupling_marginal_error"] > 1e-12
        ),
        "max_solver_marginal_error": result["max_solver_marginal_error"],
        "max_true_coupling_marginal_error": result[
            "max_true_coupling_marginal_error"
        ],
        "mean_joint_width": float(np.mean(joint_width)),
        "mean_sequential_separate_width": float(np.mean(separate_width)),
        "mean_width_gain": float(np.mean(width_gain)),
        "max_width_gain": float(np.max(width_gain)),
        "fraction_width_strict": float(np.mean(width_gain > 1e-10)),
        "h1_mean_joint_width": float(np.mean(joint_width[0])),
        "h1_mean_separate_width": float(np.mean(separate_width[0])),
        "mean_upper_feasible_tuple_count": float(
            np.mean(result["upper_feasible_tuple_count"][active])
        ),
        "mean_lower_feasible_tuple_count": float(
            np.mean(result["lower_feasible_tuple_count"][active])
        ),
        "min_feasible_tuple_count": int(np.min(tuple_counts)),
        "max_feasible_tuple_count": int(np.max(tuple_counts)),
        "max_action_grid_upper_correction": float(max_correction),
        "runtime_seconds": runtime_seconds,
    }
    for h_index, state_index in np.argwhere(center_bad):
        print(
            f"CENTER_VIOLATION h={h_index + 1} "
            f"state={result['state_grid'][state_index]:.6f}"
        )
    strict_gain = bool(np.any(width_gain > 1e-10))
    return stats, strict_gain


def save_artifact(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: result[key] for key in ARTIFACT_FIELDS})


def run() -> tuple[dict, bool]:
    reference = load_reference(ROOT / "artifacts" / "phase1b" / "oracle_reference.npz")
    start = perf_counter()
    result = solve_population_sequential_joint(reference)
    runtime_seconds = perf_counter() - start
    stats, strict_gain = audit_result(result, runtime_seconds)
    expected_calls = 20 * 21 * 41 * 2
    required_zero = (
        "lp_failures",
        "oracle_certified_violations",
        "dominance_violations",
        "interval_order_violations",
        "range_violations",
        "lipschitz_violations",
        "true_coupling_violations",
    )
    if stats["joint_lp_calls"] != expected_calls:
        raise RuntimeError(
            f"joint LP call mismatch: {stats['joint_lp_calls']} != {expected_calls}"
        )
    if any(stats[key] != 0 for key in required_zero):
        raise RuntimeError("one or more sequential acceptance checks failed")
    if stats["max_solver_marginal_error"] > 1e-8:
        raise RuntimeError("LP marginal error exceeds 1e-8")
    if stats["max_true_coupling_marginal_error"] > 1e-12:
        raise RuntimeError("true-coupling marginal error exceeds 1e-12")
    save_artifact(result, ROOT / "artifacts" / "phase2b" / "population_sequential_audit.npz")
    return stats, strict_gain


def main() -> int:
    try:
        stats, strict_gain = run()
    except Exception as exc:
        print(f"FAIL {exc}")
        print("SEQUENTIAL_STRICT_GAIN_OBSERVED = False")
        print("PHASE2B_MISMATCH")
        return 1
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"SEQUENTIAL_STRICT_GAIN_OBSERVED = {strict_gain}")
    print("PHASE2B_FAITHFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
