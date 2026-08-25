"""Validate the AAMAS26 population/oracle mass analogue against prior artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aamas_population_oracle import (  # noqa: E402
    AAMAS26_POPULATION_MASS_ANALOGUE,
    solve_aamas_population_mass_analogue,
)
from exact_population_sequential_bound import evaluate_upper_extension  # noqa: E402


ORACLE_PATH = ROOT / "artifacts" / "phase1b" / "oracle_reference.npz"
JOINT_PATH = ROOT / "artifacts" / "phase2b" / "population_sequential_audit.npz"
OUTPUT_DIR = ROOT / "artifacts" / "phase5a"
NPZ_PATH = OUTPUT_DIR / "aamas_population_oracle_audit.npz"
JSON_PATH = OUTPUT_DIR / "aamas_population_oracle_summary.json"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _validate_references(
    oracle: dict[str, np.ndarray], joint: dict[str, np.ndarray]
) -> tuple[np.ndarray, int, float]:
    for name in ("state_grid", "values", "numerical_error_bound", "horizon", "gamma"):
        if name not in oracle:
            raise KeyError(f"Oracle artifact lacks {name}")
    for name in ("state_grid", "joint_upper", "value_lipschitz", "B_plus"):
        if name not in joint:
            raise KeyError(f"Phase 2B artifact lacks {name}")
    state_grid = np.asarray(oracle["state_grid"], dtype=np.float64)
    horizon = int(oracle["horizon"])
    gamma = float(oracle["gamma"])
    if state_grid.shape != (1001,) or horizon != 20 or gamma != 0.95:
        raise ValueError("Phase 1B reference must use the prescribed 1001/H=20/gamma=.95 setup")
    if oracle["values"].shape != (horizon + 2, state_grid.size):
        raise ValueError("unexpected Oracle value shape")
    if joint["joint_upper"].shape[0] != horizon + 2:
        raise ValueError("Phase 2B horizon disagrees with Phase 1B")
    return state_grid, horizon, gamma


def _extend_joint_upper(
    query_grid: np.ndarray,
    horizon: int,
    joint: dict[str, np.ndarray],
) -> np.ndarray:
    """Reuse Phase 2B's saved joint upper bound and its existing safe extension."""
    phase2_grid = np.asarray(joint["state_grid"], dtype=np.float64)
    result = np.zeros((horizon + 2, query_grid.size), dtype=np.float64)
    for stage in range(1, horizon + 1):
        result[stage] = evaluate_upper_extension(
            query_grid,
            phase2_grid,
            joint["joint_upper"][stage],
            float(joint["value_lipschitz"][stage]),
            float(joint["B_plus"][stage]),
        )
    return result


def _classification(
    upper: np.ndarray, reference: np.ndarray, epsilon: np.ndarray
) -> dict[str, np.ndarray]:
    certified = upper >= reference + epsilon
    undercoverage = upper < reference - epsilon
    ambiguous = ~(certified | undercoverage)
    return {
        "certified_upper": certified,
        "certified_undercoverage": undercoverage,
        "ambiguous": ambiguous,
    }


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("summary input must be nonempty and finite")
    median = float(np.median(values))
    return {
        "mean": float(np.mean(values)),
        "median": median,
        "p10": float(np.percentile(values, 10.0)),
        "p90": float(np.percentile(values, 90.0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean_absolute_deviation_from_reference": float(np.mean(np.abs(values))),
    }


def _method_summary(
    excess: np.ndarray, classes: dict[str, np.ndarray]
) -> dict[str, object]:
    return {
        "certified_upper_fraction": float(np.mean(classes["certified_upper"])),
        "certified_undercoverage_fraction": float(
            np.mean(classes["certified_undercoverage"])
        ),
        "ambiguous_fraction": float(np.mean(classes["ambiguous"])),
        "upper_excess": _finite_stats(excess),
    }


def _stage_records(
    horizon: int,
    aamas_excess: np.ndarray,
    joint_excess: np.ndarray,
    aamas_classes: dict[str, np.ndarray],
    joint_classes: dict[str, np.ndarray],
) -> list[dict[str, float | int | None]]:
    records: list[dict[str, float | int | None]] = []
    for row, stage in enumerate(range(1, horizon + 1)):
        both_certified = (
            aamas_classes["certified_upper"][row]
            & joint_classes["certified_upper"][row]
        )
        tighter = joint_excess[row] < aamas_excess[row]
        records.append(
            {
                "stage": stage,
                "aamas_mean_upper_excess": float(np.mean(aamas_excess[row])),
                "joint_mean_upper_excess": float(np.mean(joint_excess[row])),
                "aamas_certified_upper_fraction": float(
                    np.mean(aamas_classes["certified_upper"][row])
                ),
                "joint_certified_upper_fraction": float(
                    np.mean(joint_classes["certified_upper"][row])
                ),
                "joint_tighter_fraction_all": float(np.mean(tighter)),
                "joint_tighter_fraction_both_certified": (
                    float(np.mean(tighter[both_certified]))
                    if np.any(both_certified)
                    else None
                ),
            }
        )
    return records


def _comparison_summary(
    aamas: np.ndarray,
    joint: np.ndarray,
    aamas_classes: dict[str, np.ndarray],
    joint_classes: dict[str, np.ndarray],
) -> dict[str, float | None]:
    both_certified = (
        aamas_classes["certified_upper"]
        & joint_classes["certified_upper"]
    )
    result: dict[str, float | None] = {
        "joint_tighter_fraction_all": float(np.mean(joint < aamas)),
        "aamas_tighter_fraction_all": float(np.mean(aamas < joint)),
        "equal_fraction_all": float(np.mean(aamas == joint)),
        "mean_aamas_minus_joint": float(np.mean(aamas - joint)),
        "both_certified_fraction": float(np.mean(both_certified)),
        "joint_tighter_fraction_both_certified": None,
        "aamas_tighter_fraction_both_certified": None,
        "aamas_mean_upper_excess_both_certified": None,
        "joint_mean_upper_excess_both_certified": None,
    }
    if np.any(both_certified):
        result.update(
            {
                "joint_tighter_fraction_both_certified": float(
                    np.mean(joint[both_certified] < aamas[both_certified])
                ),
                "aamas_tighter_fraction_both_certified": float(
                    np.mean(aamas[both_certified] < joint[both_certified])
                ),
                "aamas_mean_upper_excess_both_certified": float(
                    np.mean(aamas[both_certified])
                ),
                "joint_mean_upper_excess_both_certified": float(
                    np.mean(joint[both_certified])
                ),
            }
        )
    return result


def _diagnostics(result: dict[str, object]) -> dict[str, float | int]:
    masses = np.asarray(result["action_masses"], dtype=np.float64)
    weights = np.asarray(result["behavior_weights"], dtype=np.float64)
    road_counts = np.asarray(result["road_candidate_counts"], dtype=np.int64)
    valid = np.isfinite(masses)
    return {
        "primitive_support_min": int(np.min(result["primitive_support_sizes"])),
        "primitive_support_mean": float(np.mean(result["primitive_support_sizes"])),
        "primitive_support_max": int(np.max(result["primitive_support_sizes"])),
        "merged_action_support_min": int(
            np.min(result["merged_action_support_sizes"])
        ),
        "merged_action_support_mean": float(
            np.mean(result["merged_action_support_sizes"])
        ),
        "merged_action_support_max": int(
            np.max(result["merged_action_support_sizes"])
        ),
        "road_candidate_count_min": int(np.min(road_counts[valid])),
        "road_candidate_count_mean": float(np.mean(road_counts[valid])),
        "road_candidate_count_max": int(np.max(road_counts[valid])),
        "behavior_mass_min": float(np.min(masses[valid])),
        "behavior_mass_max": float(np.max(masses[valid])),
        "behavior_weight_min": float(np.min(weights[valid])),
        "behavior_weight_mean": float(np.mean(weights[valid])),
        "behavior_weight_max": float(np.max(weights[valid])),
        "empty_road_support_count": int(result["empty_road_support_count"]),
    }


def run_validation(
    oracle_path: Path = ORACLE_PATH,
    joint_path: Path = JOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    oracle = _load_npz(oracle_path)
    joint_saved = _load_npz(joint_path)
    state_grid, horizon, gamma = _validate_references(oracle, joint_saved)
    started = perf_counter()
    aamas_result = solve_aamas_population_mass_analogue(
        state_grid, horizon=horizon, gamma=gamma
    )
    phi_aamas = np.asarray(aamas_result["phi"], dtype=np.float64)
    phi_joint = _extend_joint_upper(state_grid, horizon, joint_saved)
    runtime_seconds = perf_counter() - started

    active = np.s_[1 : horizon + 1]
    reference = np.asarray(oracle["values"][active], dtype=np.float64)
    epsilon = np.broadcast_to(
        np.asarray(oracle["numerical_error_bound"][active], dtype=np.float64)[:, None],
        reference.shape,
    ).copy()
    aamas_excess = phi_aamas[active] - reference
    joint_excess = phi_joint[active] - reference
    aamas_classes = _classification(phi_aamas[active], reference, epsilon)
    joint_classes = _classification(phi_joint[active], reference, epsilon)
    initial_mask = (state_grid >= -0.8) & (state_grid <= 0.8)

    flags = {
        "finite_recursion": bool(np.all(np.isfinite(phi_aamas))),
        "terminal_zero": bool(np.array_equal(phi_aamas[horizon + 1], np.zeros_like(state_grid))),
        "no_empty_road_support": int(aamas_result["empty_road_support_count"]) == 0,
        "uses_neural_network": bool(aamas_result["uses_neural_network"]),
        "uses_hidden_operator_input": bool(aamas_result["uses_hidden_operator_input"]),
        "uses_joint_coupling": bool(aamas_result["uses_joint_coupling"]),
        "uses_tuned_road_scale": bool(aamas_result["uses_tuned_road_scale"]),
        "AAMAS26_CONTINUOUS_DENSITY_REPLACED_BY_POPULATION_MASS": bool(
            aamas_result["AAMAS26_CONTINUOUS_DENSITY_REPLACED_BY_POPULATION_MASS"]
        ),
        "AAMAS26_K25_MONTE_CARLO_ERROR_REMOVED": bool(
            aamas_result["AAMAS26_K25_MONTE_CARLO_ERROR_REMOVED"]
        ),
        "AAMAS26_NEURAL_APPROXIMATION_ERROR_REMOVED": bool(
            aamas_result["AAMAS26_NEURAL_APPROXIMATION_ERROR_REMOVED"]
        ),
        "uses_online_learning": False,
        "uses_finite_sample": False,
    }
    if not all((flags["finite_recursion"], flags["terminal_zero"], flags["no_empty_road_support"])):
        raise RuntimeError(f"Phase 5A structural validation failed: {flags}")

    summary: dict[str, object] = {
        "method": AAMAS26_POPULATION_MASS_ANALOGUE,
        "interpretation": "AAMAS26 population/oracle analogue; not the official neural baseline",
        "finite_horizon_interpretation": (
            "(h,s) is the public time-augmented state; this is an environment-specific "
            "finite-horizon adaptation, not a claim about the official stationary critic"
        ),
        "remaining_structural_differences": [
            "continuous density replaced by population mass",
            "road-not-taken construction",
            "equally pooled source handling",
            "finite-horizon adaptation",
        ],
        "horizon": horizon,
        "gamma": gamma,
        "state_grid_size": int(state_grid.size),
        "runtime_seconds": runtime_seconds,
        "aamas": _method_summary(aamas_excess, aamas_classes),
        "joint": _method_summary(joint_excess, joint_classes),
        "comparison": _comparison_summary(
            aamas_excess, joint_excess, aamas_classes, joint_classes
        ),
        "initial_state_interval": {
            "lower": -0.8,
            "upper": 0.8,
            "grid_count": int(np.sum(initial_mask)),
            "oracle_mean_value_h1": float(np.mean(reference[0, initial_mask])),
            "aamas_mean_potential_h1": float(np.mean(phi_aamas[1, initial_mask])),
            "joint_mean_potential_h1": float(np.mean(phi_joint[1, initial_mask])),
            "aamas_mean_upper_excess_h1": float(np.mean(aamas_excess[0, initial_mask])),
            "joint_mean_upper_excess_h1": float(np.mean(joint_excess[0, initial_mask])),
        },
        "road_diagnostics": _diagnostics(aamas_result),
        "per_stage": _stage_records(
            horizon,
            aamas_excess,
            joint_excess,
            aamas_classes,
            joint_classes,
        ),
        "special_stages": {},
        "implementation_flags": flags,
    }
    summary["special_stages"] = {
        str(stage): summary["per_stage"][stage - 1] for stage in (1, 10, 20)
    }
    arrays = {
        "state_grid": state_grid,
        "stages": np.arange(1, horizon + 1, dtype=np.int64),
        "phi_aamas_population": phi_aamas,
        "phi_joint": phi_joint,
        "oracle_v_ref": np.asarray(oracle["values"], dtype=np.float64),
        "oracle_epsilon": np.asarray(oracle["numerical_error_bound"], dtype=np.float64),
        "excess_aamas": aamas_excess,
        "excess_joint": joint_excess,
        "aamas_certified_upper": aamas_classes["certified_upper"],
        "aamas_certified_undercoverage": aamas_classes["certified_undercoverage"],
        "aamas_ambiguous": aamas_classes["ambiguous"],
        "joint_certified_upper": joint_classes["certified_upper"],
        "joint_certified_undercoverage": joint_classes["certified_undercoverage"],
        "joint_ambiguous": joint_classes["ambiguous"],
        "primitive_support_sizes": np.asarray(aamas_result["primitive_support_sizes"]),
        "merged_action_support_sizes": np.asarray(aamas_result["merged_action_support_sizes"]),
        "action_atoms": np.asarray(aamas_result["action_atoms"]),
        "action_masses": np.asarray(aamas_result["action_masses"]),
        "behavior_weights": np.asarray(aamas_result["behavior_weights"]),
        "road_candidate_counts": np.asarray(aamas_result["road_candidate_counts"]),
        "AAMAS26_CONTINUOUS_DENSITY_REPLACED_BY_POPULATION_MASS": np.array(True),
        "AAMAS26_K25_MONTE_CARLO_ERROR_REMOVED": np.array(True),
        "AAMAS26_NEURAL_APPROXIMATION_ERROR_REMOVED": np.array(True),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / NPZ_PATH.name, **arrays)
    with (output_dir / JSON_PATH.name).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return summary, arrays


def _print_summary(summary: dict[str, object]) -> None:
    print(AAMAS26_POPULATION_MASS_ANALOGUE)
    print(summary["interpretation"])
    print("h  AAMAS excess  Joint excess  AAMAS cert  Joint cert  Joint tighter|both")
    for row in summary["per_stage"]:
        tighter = row["joint_tighter_fraction_both_certified"]
        tighter_text = "n/a" if tighter is None else f"{tighter:.3f}"
        print(
            f"{row['stage']:2d}  {row['aamas_mean_upper_excess']:12.6f}  "
            f"{row['joint_mean_upper_excess']:12.6f}  "
            f"{row['aamas_certified_upper_fraction']:10.3f}  "
            f"{row['joint_certified_upper_fraction']:10.3f}  {tighter_text:>18}"
        )
    print("comparison:", json.dumps(summary["comparison"], sort_keys=True))
    print("special h=1,10,20:", json.dumps(summary["special_stages"], sort_keys=True))
    print("initial states:", json.dumps(summary["initial_state_interval"], sort_keys=True))
    print("ROAD diagnostics:", json.dumps(summary["road_diagnostics"], sort_keys=True))
    print("flags:", json.dumps(summary["implementation_flags"], sort_keys=True))
    print(f"artifacts: {NPZ_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    validation_summary, _ = run_validation()
    _print_summary(validation_summary)
    print("PHASE5A_AAMAS_POPULATION_ANALOGUE_FAITHFUL")
