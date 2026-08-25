"""Minimal post-run diagnosis of the Phase 4A fixed-state decision bottleneck."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixed_state_online_calibration import (  # noqa: E402
    anytime_hoeffding_radius,
    run_fixed_state_online_calibration,
)
from generate_offline_dataset import generate_fixed_state_dataset  # noqa: E402
from oracle_ground_truth import oracle_q  # noqa: E402
from scripts.validate_phase4a_online_calibration import (  # noqa: E402
    ACTION_VALUES,
    DATA_H_VALUES,
    DATA_STATE_VALUES,
    DELTA_OFF,
    DELTA_ON,
    EPSILON,
    MAX_ONLINE_INTERACTIONS,
    METHODS,
    SAMPLE_SIZES,
    load_reference,
    make_action_sampler,
    phase3a_initial_intervals,
    source_atoms_from_prefix,
)
from finite_sample_joint_bound import build_probability_intervals  # noqa: E402


H, STATE, REPLICATES, TOL = 20, 0.6, 20, 1e-12


def oracle_geometry(actions, q_values, tol=TOL):
    actions, q_values = np.asarray(actions), np.asarray(q_values)
    ranking = np.argsort(-q_values, kind="stable")
    best, second = int(ranking[0]), int(ranking[1])
    ties = np.flatnonzero(np.abs(q_values - q_values[best]) <= tol)
    spacing = np.diff(actions)
    return {
        "ranking": ranking, "best": best, "second": second,
        "gap": float(q_values[best] - q_values[second]),
        "grid_spacing": float(spacing[0]) if np.all(np.abs(spacing - spacing[0]) <= tol) else np.nan,
        "ties": ties, "unique": len(ties) == 1,
    }


def decision_deficit(lower, upper, optimal_index):
    competitors = np.delete(np.asarray(upper), optimal_index)
    return float(np.max(competitors) - np.asarray(lower)[optimal_index])


def first_divergence(left, right):
    left, right = np.asarray(left), np.asarray(right)
    common = min(len(left), len(right))
    differences = np.flatnonzero(left[:common] != right[:common])
    if len(differences):
        return int(differences[0] + 1)
    return -1 if len(left) == len(right) else common + 1


def equal_count_hoeffding_benchmark(gap, value_range, delta_online, n_actions, cap=10**9):
    if gap <= 0.0:
        return None
    condition = lambda n: 2.0 * anytime_hoeffding_radius(
        n, value_range, delta_online, n_actions
    ) <= gap
    if not condition(cap):
        return None
    low, high = 1, cap
    while low < high:
        middle = (low + high) // 2
        if condition(middle):
            high = middle
        else:
            low = middle + 1
    return low


def admissible_indices(lower, upper):
    return np.flatnonzero(np.asarray(upper) >= np.max(lower) - TOL)


def best_challenger(lower, upper):
    best = int(np.argmax(lower))
    candidates = np.flatnonzero(np.arange(len(lower)) != best)
    challenger = int(candidates[np.argmax(np.asarray(upper)[candidates])])
    return best, challenger


def initialize_storage():
    shape = (len(SAMPLE_SIZES), REPLICATES, len(METHODS))
    return {
        "initial_lower": np.empty(shape + (len(ACTION_VALUES),)),
        "initial_upper": np.empty(shape + (len(ACTION_VALUES),)),
        "decision_deficits": np.empty(shape),
        "admissible_counts": np.empty(shape, dtype=np.int8),
        "zero_certified": np.empty(shape, dtype=bool),
        "final_counts": np.empty(shape + (len(ACTION_VALUES),), dtype=np.int64),
        "interactions": np.empty(shape, dtype=np.int64),
        "top_two_fractions": np.empty(shape),
        "sampled_action_counts": np.empty(shape, dtype=np.int8),
        "final_radius_sums": np.empty(shape),
        "final_margins": np.empty(shape),
        "final_best": np.empty(shape, dtype=np.int8),
        "final_challenger": np.empty(shape, dtype=np.int8),
        "dominant_pairs": np.empty(shape + (2,), dtype=np.int8),
        "path_identical": np.empty((len(SAMPLE_SIZES), REPLICATES), dtype=bool),
        "first_divergence": np.empty((len(SAMPLE_SIZES), REPLICATES), dtype=np.int64),
        "counts_identical": np.empty((len(SAMPLE_SIZES), REPLICATES), dtype=bool),
        "bounds_identical": np.empty((len(SAMPLE_SIZES), REPLICATES), dtype=bool),
        "certified_action_identical": np.empty((len(SAMPLE_SIZES), REPLICATES), dtype=bool),
        "interactions_identical": np.empty((len(SAMPLE_SIZES), REPLICATES), dtype=bool),
    }


def run_diagnostics(reference):
    q_values = np.asarray(oracle_q(reference, H, STATE, ACTION_VALUES))
    geometry = oracle_geometry(ACTION_VALUES, q_values)
    best, second = geometry["best"], geometry["second"]
    value_range = float(reference["return_upper"][H] - reference["return_lower"][H])
    n_equal = equal_count_hoeffding_benchmark(
        geometry["gap"], value_range, DELTA_ON, len(ACTION_VALUES)
    )
    data = initialize_storage()
    for replicate in range(REPLICATES):
        train, _ = generate_fixed_state_dataset(
            DATA_H_VALUES, DATA_STATE_VALUES, max(SAMPLE_SIZES), 4000 + replicate
        )
        for n_index, sample_size in enumerate(SAMPLE_SIZES):
            atoms = source_atoms_from_prefix(train, reference, H, STATE, sample_size)
            intervals = build_probability_intervals(atoms, DELTA_OFF)
            initial = phase3a_initial_intervals(atoms, intervals, reference, H)
            base = [9100, replicate, DATA_H_VALUES.index(H), DATA_STATE_VALUES.index(STATE), sample_size]
            tie_seed = int(np.random.SeedSequence(base + [0]).generate_state(1)[0])
            results = {}
            for method_index, method in enumerate(METHODS):
                lower, upper = initial[method]
                widths = upper - lower
                admissible = admissible_indices(lower, upper)
                data["initial_lower"][n_index, replicate, method_index] = lower
                data["initial_upper"][n_index, replicate, method_index] = upper
                data["decision_deficits"][n_index, replicate, method_index] = decision_deficit(lower, upper, best)
                data["admissible_counts"][n_index, replicate, method_index] = len(admissible)
                initial_best, initial_challenger = best_challenger(lower, upper)
                data["zero_certified"][n_index, replicate, method_index] = (
                    lower[initial_best] >= upper[initial_challenger]
                )
                sample, _ = make_action_sampler(reference, H, STATE, base + [1])
                result = run_fixed_state_online_calibration(
                    ACTION_VALUES, lower, upper, sample,
                    float(reference["return_lower"][H]), float(reference["return_upper"][H]),
                    DELTA_ON, MAX_ONLINE_INTERACTIONS, tie_seed, epsilon=EPSILON,
                )
                results[method] = result
                counts = result["counts"]
                final_best, final_challenger = best_challenger(
                    result["final_lower"], result["final_upper"]
                )
                dominant = np.argsort(-counts, kind="stable")[:2]
                total = result["online_interactions"]
                radii = [
                    anytime_hoeffding_radius(int(counts[index]), value_range, DELTA_ON, len(ACTION_VALUES))
                    for index in (best, second)
                ]
                key = (n_index, replicate, method_index)
                data["final_counts"][key] = counts
                data["interactions"][key] = total
                data["top_two_fractions"][key] = (counts[best] + counts[second]) / total
                data["sampled_action_counts"][key] = np.count_nonzero(counts)
                data["final_radius_sums"][key] = sum(radii)
                data["final_margins"][key] = (
                    result["final_lower"][final_best] - result["final_upper"][final_challenger]
                )
                data["final_best"][key], data["final_challenger"][key] = final_best, final_challenger
                data["dominant_pairs"][key] = dominant
            joint, separate = results["joint"], results["separate"]
            divergence = first_divergence(
                joint["sampled_action_indices"], separate["sampled_action_indices"]
            )
            pair_key = (n_index, replicate)
            data["first_divergence"][pair_key] = divergence
            data["path_identical"][pair_key] = divergence == -1
            data["counts_identical"][pair_key] = np.array_equal(joint["counts"], separate["counts"])
            data["bounds_identical"][pair_key] = (
                np.all(np.abs(joint["final_lower"] - separate["final_lower"]) <= TOL)
                and np.all(np.abs(joint["final_upper"] - separate["final_upper"]) <= TOL)
            )
            data["certified_action_identical"][pair_key] = joint["certified_action"] == separate["certified_action"]
            data["interactions_identical"][pair_key] = joint["online_interactions"] == separate["online_interactions"]
    data["initial_widths"] = data["initial_upper"] - data["initial_lower"]
    data["joint_separate_gains"] = data["initial_widths"][:, :, 1] - data["initial_widths"][:, :, 2]
    data["critical_gains"] = data["decision_deficits"][:, :, 1] - data["decision_deficits"][:, :, 2]
    return geometry, q_values, n_equal, data


def modal_pair(pairs):
    pairs = np.sort(np.asarray(pairs), axis=1)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    return unique[np.argmax(counts)]


def print_tables(geometry, q_values, n_equal, data):
    best, second = geometry["best"], geometry["second"]
    print("TABLE1_ORACLE_GEOMETRY")
    print("best second best_q second_q gap grid_spacing n_equal two_n_equal")
    print(ACTION_VALUES[best], ACTION_VALUES[second], q_values[best], q_values[second], geometry["gap"], geometry["grid_spacing"], n_equal, 2 * n_equal)
    print(f"unique_optimum={geometry['unique']} tied_actions={ACTION_VALUES[geometry['ties']]}")
    for rank, index in enumerate(geometry["ranking"][:10], 1):
        print(f"rank={rank} action={ACTION_VALUES[index]:.6f} q={q_values[index]:.10f} optimality_gap={q_values[best]-q_values[index]:.10f}")
    for index in (best - 1, best + 1):
        if 0 <= index < len(ACTION_VALUES):
            print(f"neighbor_action={ACTION_VALUES[index]:.6f} q_gap={q_values[best]-q_values[index]:.10f}")
    print("action_level_q_certificate_available = False")
    print("TABLE2_INITIAL_INTERVALS")
    print("n method mean_width median_width best_interval second_interval admissible_count decision_deficit zero_cert_rate")
    for ni, n in enumerate(SAMPLE_SIZES):
        for mi, method in enumerate(METHODS):
            widths = data["initial_widths"][ni, :, mi]
            lower, upper = data["initial_lower"][ni, :, mi], data["initial_upper"][ni, :, mi]
            print(n, method, widths.mean(), np.median(widths), (lower[:, best].mean(), upper[:, best].mean()), (lower[:, second].mean(), upper[:, second].mean()), data["admissible_counts"][ni, :, mi].mean(), data["decision_deficits"][ni, :, mi].mean(), data["zero_certified"][ni, :, mi].mean())
    print("TABLE3_JOINT_SEPARATE_GAIN")
    print("n strict_gain_actions gain_in_joint_admissible gain_best gain_second critical_gain identical_path_rate")
    for ni, n in enumerate(SAMPLE_SIZES):
        gains = data["joint_separate_gains"][ni]
        strict_counts, inside_counts = [], []
        for replicate in range(REPLICATES):
            strict = np.flatnonzero(gains[replicate] > TOL)
            admissible = admissible_indices(data["initial_lower"][ni, replicate, 2], data["initial_upper"][ni, replicate, 2])
            strict_counts.append(len(strict)); inside_counts.append(np.intersect1d(strict, admissible).size)
        print(n, np.mean(strict_counts), np.mean(inside_counts), gains[:, best].mean(), gains[:, second].mean(), data["critical_gains"][ni].mean(), data["path_identical"][ni].mean())
        strict_gap_values, inside_gains, outside_gains = [], [], []
        best_hits, second_hits, separate_challenger_hits, joint_challenger_hits = 0, 0, 0, 0
        for replicate in range(REPLICATES):
            strict_mask = gains[replicate] > TOL
            oracle_gaps = q_values[best] - q_values
            strict_gap_values.extend(oracle_gaps[strict_mask])
            admissible = admissible_indices(data["initial_lower"][ni, replicate, 2], data["initial_upper"][ni, replicate, 2])
            outside = np.setdiff1d(np.arange(len(ACTION_VALUES)), admissible)
            inside_gains.extend(gains[replicate, admissible])
            outside_gains.extend(gains[replicate, outside])
            best_hits += int(strict_mask[best]); second_hits += int(strict_mask[second])
            _, sc = best_challenger(data["initial_lower"][ni, replicate, 1], data["initial_upper"][ni, replicate, 1])
            _, jc = best_challenger(data["initial_lower"][ni, replicate, 2], data["initial_upper"][ni, replicate, 2])
            separate_challenger_hits += int(strict_mask[sc]); joint_challenger_hits += int(strict_mask[jc])
        strict_gap_values = np.asarray(strict_gap_values)
        mean_gain = gains.mean(axis=0)
        top_gain = np.argsort(-mean_gain, kind="stable")[:10]
        print(f"n={n} max_gain={gains.max():.10f} mean_gain_inside_admissible={np.mean(inside_gains):.10f} mean_gain_outside_admissible={np.mean(outside_gains) if outside_gains else np.nan:.10f} strict_gap_min_median_max={(strict_gap_values.min(), np.median(strict_gap_values), strict_gap_values.max())} best_hits={best_hits} second_hits={second_hits} separate_challenger_hits={separate_challenger_hits} joint_challenger_hits={joint_challenger_hits}")
        print(f"n={n} top_gain_actions={[(float(ACTION_VALUES[i]), float(mean_gain[i]), float(q_values[best]-q_values[i])) for i in top_gain]}")
    print("TABLE4_ONLINE_ALLOCATION")
    print("n method mean_interactions top_two_fraction sampled_actions dominant_pair final_best_count final_challenger_count radius_sum final_margin")
    for ni, n in enumerate(SAMPLE_SIZES):
        for mi, method in enumerate(METHODS):
            pair = modal_pair(data["dominant_pairs"][ni, :, mi])
            final_counts = data["final_counts"][ni, :, mi]
            rows = np.arange(REPLICATES)
            best_counts = final_counts[rows, data["final_best"][ni, :, mi]]
            challenger_counts = final_counts[rows, data["final_challenger"][ni, :, mi]]
            print(n, method, data["interactions"][ni, :, mi].mean(), data["top_two_fractions"][ni, :, mi].mean(), data["sampled_action_counts"][ni, :, mi].mean(), tuple(ACTION_VALUES[pair]), best_counts.mean(), challenger_counts.mean(), data["final_radius_sums"][ni, :, mi].mean(), data["final_margins"][ni, :, mi].mean())
    divergences = data["first_divergence"]
    finite = divergences[divergences > 0]
    for ni, n in enumerate(SAMPLE_SIZES):
        current = divergences[ni]
        print(f"n={n} identical_path_rate={np.mean(current == -1):.8f} never_diverged={np.sum(current == -1)} mean_first_divergence={np.mean(current[current > 0]) if np.any(current > 0) else np.nan}")
        print(f"n={n} identical_counts_rate={data['counts_identical'][ni].mean():.8f} identical_bounds_rate={data['bounds_identical'][ni].mean():.8f} identical_certified_action_rate={data['certified_action_identical'][ni].mean():.8f} identical_interactions_rate={data['interactions_identical'][ni].mean():.8f}")
    actual_mean = data["interactions"].mean()
    print(f"gap_based_benchmark_not_strict_prediction=True actual_mean_interactions={data['interactions'].mean():.6f} benchmark_total={2*n_equal}")
    print(f"same_order_of_magnitude={0.1 <= actual_mean/(2*n_equal) <= 10.0}")
    print(f"JOINT_AND_SEPARATE_DECISION_PATHS_IDENTICAL = {bool(np.all(data['path_identical']))}")


def save_artifact(geometry, q_values, n_equal, data):
    path = ROOT / "artifacts" / "phase4a" / "phase4a_decision_bottleneck_diagnostics.npz"
    payload = {
        "action_grid": ACTION_VALUES, "oracle_q_values": q_values,
        "oracle_ranking": geometry["ranking"], "oracle_top_two_gap": geometry["gap"],
        "hoeffding_equal_count_benchmark": n_equal, **data,
    }
    np.savez_compressed(path, **payload)
    return path


def main():
    reference = load_reference(ROOT / "artifacts" / "phase1b" / "oracle_reference.npz")
    geometry, q_values, n_equal, data = run_diagnostics(reference)
    print_tables(geometry, q_values, n_equal, data)
    path = save_artifact(geometry, q_values, n_equal, data)
    print(f"artifact={path}")
    print("PHASE4A_DIAGNOSTICS_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
