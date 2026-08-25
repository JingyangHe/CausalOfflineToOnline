"""Validate fixed-budget decision efficiency at five Phase 4B states."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finite_sample_joint_bound import build_probability_intervals  # noqa: E402
from fixed_state_online_calibration import run_fixed_budget_online_evaluation  # noqa: E402
from generate_offline_dataset import generate_fixed_state_dataset  # noqa: E402
from oracle_ground_truth import oracle_q  # noqa: E402
from scripts.validate_phase4a_online_calibration import (  # noqa: E402
    ACTION_VALUES, DELTA_OFF, DELTA_ON, METHODS, load_reference,
    make_action_sampler, phase3a_initial_intervals, source_atoms_from_prefix,
)


H = 20
STATES = np.array((-0.6, -0.3, 0.0, 0.3, 0.6))
SAMPLE_SIZES = np.array((25, 50, 100, 250))
BUDGETS = np.array((0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000))
PRIMARY_BUDGET = 1000
REPLICATES = 50
DELTA_TOTAL = 0.05
TOL = 1e-12


def calculate_regret_metrics(q_values, sampled_indices, recommended_indices, budgets):
    """Compute cumulative pseudo-regret and checkpoint simple regret after a run."""
    q_values = np.asarray(q_values, dtype=np.float64)
    sampled_indices = np.asarray(sampled_indices, dtype=np.int64)
    recommended_indices = np.asarray(recommended_indices, dtype=np.int64)
    budgets = np.asarray(budgets, dtype=np.int64)
    q_star = float(np.max(q_values))
    instantaneous = q_star - q_values[sampled_indices]
    prefix = np.concatenate(([0.0], np.cumsum(instantaneous)))
    cumulative = prefix[budgets]
    simple = q_star - q_values[recommended_indices]
    return cumulative, simple


def paired_summary(differences):
    values = np.asarray(differences, dtype=np.float64).ravel()
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": mean, "standard_error": se,
        "ci95_lower": mean - 1.96 * se, "ci95_upper": mean + 1.96 * se,
        "joint_lower_fraction": float(np.mean(values < -TOL)),
        "equal_fraction": float(np.mean(np.abs(values) <= TOL)),
    }


def initialize_storage():
    base = (len(STATES), len(SAMPLE_SIZES), REPLICATES, len(METHODS))
    checkpoints = base + (len(BUDGETS),)
    return {
        "cumulative_regret": np.empty(checkpoints),
        "simple_regret": np.empty(checkpoints),
        "recommended_action_indices": np.empty(checkpoints, dtype=np.int8),
        "recommended_action_ranks": np.empty(checkpoints, dtype=np.int8),
        "admissible_counts": np.empty(checkpoints, dtype=np.int8),
        "sampled_action_counts": np.empty(checkpoints, dtype=np.int8),
        "certified_flags": np.empty(checkpoints, dtype=bool),
        "certification_times": np.full(base, -1, dtype=np.int64),
        "conflict_counts": np.empty(base, dtype=np.int16),
        "wrong_certifications": np.empty(base, dtype=bool),
        "optimal_action_deleted": np.empty(base, dtype=bool),
        "interval_violations": np.empty(base, dtype=np.int16),
    }


def run_experiments(reference):
    data = initialize_storage()
    oracle_values = np.asarray([oracle_q(reference, H, state, ACTION_VALUES) for state in STATES])
    rankings = np.argsort(-oracle_values, axis=1, kind="stable")
    b_lower, b_upper = float(reference["return_lower"][H]), float(reference["return_upper"][H])
    for replicate in range(REPLICATES):
        train, _ = generate_fixed_state_dataset([H], STATES, max(SAMPLE_SIZES), 12000 + replicate)
        for state_index, state in enumerate(STATES):
            oracle_best = int(rankings[state_index, 0])
            for n_index, sample_size in enumerate(SAMPLE_SIZES):
                atoms = source_atoms_from_prefix(train, reference, H, state, sample_size)
                intervals = build_probability_intervals(atoms, DELTA_OFF)
                if not intervals["certified"]:
                    raise RuntimeError("uncertified Phase 3A initialization")
                initial = phase3a_initial_intervals(atoms, intervals, reference, H)
                base_seed = [14000, replicate, state_index, int(sample_size)]
                tie_seed = int(np.random.SeedSequence(base_seed + [0]).generate_state(1)[0])
                for method_index, method in enumerate(METHODS):
                    sample, _ = make_action_sampler(reference, H, state, base_seed + [1])
                    result = run_fixed_budget_online_evaluation(
                        ACTION_VALUES, initial[method][0], initial[method][1], sample,
                        b_lower, b_upper, DELTA_ON, BUDGETS, tie_seed, record_actions=True,
                    )
                    cumulative, simple = calculate_regret_metrics(
                        oracle_values[state_index], result["sampled_action_indices"],
                        result["recommended_action_indices"], BUDGETS,
                    )
                    key = (state_index, n_index, replicate, method_index)
                    data["cumulative_regret"][key] = cumulative
                    data["simple_regret"][key] = simple
                    data["recommended_action_indices"][key] = result["recommended_action_indices"]
                    inverse_ranking = np.empty(len(ACTION_VALUES), dtype=np.int8)
                    inverse_ranking[rankings[state_index]] = np.arange(1, len(ACTION_VALUES) + 1)
                    data["recommended_action_ranks"][key] = inverse_ranking[
                        result["recommended_action_indices"]
                    ]
                    data["admissible_counts"][key] = result["admissible_counts_at_checkpoints"]
                    data["sampled_action_counts"][key] = np.count_nonzero(
                        result["counts_at_checkpoints"], axis=1
                    )
                    data["certified_flags"][key] = result["certified_at_checkpoint"]
                    data["certification_times"][key] = (
                        -1 if result["certification_time"] is None else result["certification_time"]
                    )
                    data["conflict_counts"][key] = result["conflict_count"]
                    data["wrong_certifications"][key] = (
                        result["committed_action_index"] is not None
                        and result["committed_action_index"] != oracle_best
                    )
                    deleted = result["upper_at_checkpoints"][:, oracle_best] < np.max(
                        result["lower_at_checkpoints"], axis=1
                    ) - TOL
                    data["optimal_action_deleted"][key] = np.any(deleted)
                    range_bad = (
                        np.any(result["lower_at_checkpoints"] < b_lower - TOL)
                        or np.any(result["upper_at_checkpoints"] > b_upper + TOL)
                        or np.any(result["lower_at_checkpoints"] > result["upper_at_checkpoints"] + TOL)
                    )
                    data["interval_violations"][key] = result["interval_violation_count"] + int(range_bad)
                    if result["total_interactions"] != BUDGETS[-1]:
                        raise RuntimeError("fixed online budget was not executed exactly")
    paired = np.empty((len(STATES), len(SAMPLE_SIZES), REPLICATES, len(BUDGETS), 2, 2))
    for comparison, baseline in enumerate((0, 1)):
        paired[..., comparison, 0] = data["cumulative_regret"][..., 2, :] - data["cumulative_regret"][..., baseline, :]
        paired[..., comparison, 1] = data["simple_regret"][..., 2, :] - data["simple_regret"][..., baseline, :]
    data["paired_differences"] = paired
    return oracle_values, rankings, data


def stratum_summary(data, state_index, n_index, budget_index, method_index):
    key = (state_index, n_index, slice(None), method_index, budget_index)
    cumulative = data["cumulative_regret"][key]
    simple = data["simple_regret"][key]
    recommended = data["recommended_action_indices"][key]
    certified = data["certified_flags"][key]
    certification_times = data["certification_times"][state_index, n_index, :, method_index]
    observed_times = certification_times[
        (certification_times >= 0) & (certification_times <= BUDGETS[budget_index])
    ]
    return {
        "cumulative_mean": float(np.mean(cumulative)), "cumulative_median": float(np.median(cumulative)),
        "cumulative_p90": float(np.quantile(cumulative, 0.9)),
        "simple_mean": float(np.mean(simple)), "simple_median": float(np.median(simple)),
        "simple_p90": float(np.quantile(simple, 0.9)),
        "optimal_recommendation_rate": float(np.mean(simple <= TOL)),
        "mean_recommended_rank": float(np.mean(data["recommended_action_ranks"][key])),
        "mean_admissible_count": float(np.mean(data["admissible_counts"][key])),
        "mean_sampled_action_count": float(np.mean(data["sampled_action_counts"][key])),
        "certification_rate": float(np.mean(certified)),
        "mean_certification_time": None if not len(observed_times) else float(np.mean(observed_times)),
        "mean_conflict_count": float(np.mean(data["conflict_counts"][state_index, n_index, :, method_index])),
        "modal_recommended_action_index": int(np.bincount(recommended).argmax()),
    }


def cross_state_summary(data, budget_index, method_index):
    cumulative = data["cumulative_regret"][..., method_index, budget_index].ravel()
    simple = data["simple_regret"][..., method_index, budget_index].ravel()
    certification_times = data["certification_times"][..., method_index].ravel()
    observed_times = certification_times[
        (certification_times >= 0) & (certification_times <= BUDGETS[budget_index])
    ]
    return {
        "cumulative_mean": float(np.mean(cumulative)), "cumulative_median": float(np.median(cumulative)),
        "cumulative_p90": float(np.quantile(cumulative, 0.9)),
        "simple_mean": float(np.mean(simple)), "simple_median": float(np.median(simple)),
        "simple_p90": float(np.quantile(simple, 0.9)),
        "optimal_recommendation_rate": float(np.mean(simple <= TOL)),
        "mean_recommended_rank": float(
            np.mean(data["recommended_action_ranks"][..., method_index, budget_index])
        ),
        "mean_admissible_count": float(np.mean(data["admissible_counts"][..., method_index, budget_index])),
        "mean_sampled_action_count": float(np.mean(data["sampled_action_counts"][..., method_index, budget_index])),
        "certification_rate": float(np.mean(data["certified_flags"][..., method_index, budget_index])),
        "mean_certification_time": None if not len(observed_times) else float(np.mean(observed_times)),
        "mean_conflict_count": float(np.mean(data["conflict_counts"][..., method_index])),
    }


def build_summary(oracle_values, rankings, data):
    strata, curves, paired = {}, {}, {}
    for si, state in enumerate(STATES):
        for ni, sample_size in enumerate(SAMPLE_SIZES):
            for bi, budget in enumerate(BUDGETS):
                for mi, method in enumerate(METHODS):
                    key = f"state={state}:n={sample_size}:budget={budget}:method={method}"
                    strata[key] = stratum_summary(data, si, ni, bi, mi)
    for bi, budget in enumerate(BUDGETS):
        for mi, method in enumerate(METHODS):
            curves[f"budget={budget}:method={method}"] = cross_state_summary(data, bi, mi)
        for comparison, label in enumerate(("joint-scratch", "joint-separate")):
            for metric, metric_name in enumerate(("cumulative", "simple")):
                paired[f"budget={budget}:{label}:{metric_name}"] = paired_summary(
                    data["paired_differences"][..., bi, comparison, metric]
                )
    diagnostics = []
    primary_index = int(np.flatnonzero(BUDGETS == PRIMARY_BUDGET)[0])
    for si, state in enumerate(STATES):
        ranking = rankings[si]
        item = {
            "state": float(state), "best_action": float(ACTION_VALUES[ranking[0]]),
            "second_action": float(ACTION_VALUES[ranking[1]]),
            "top_two_gap": float(oracle_values[si, ranking[0]] - oracle_values[si, ranking[1]]),
            "unique_optimum": bool(np.sum(np.abs(oracle_values[si] - oracle_values[si, ranking[0]]) <= TOL) == 1),
        }
        for mi, method in enumerate(METHODS):
            for bi, label in ((0, "budget0"), (primary_index, "budget1000")):
                values = data["recommended_action_indices"][si, :, :, mi, bi].ravel()
                item[f"{method}_{label}_modal_action"] = float(ACTION_VALUES[np.bincount(values).argmax()])
            item[f"{method}_budget1000_mean_admissible"] = float(
                np.mean(data["admissible_counts"][si, :, :, mi, primary_index])
            )
        diagnostics.append(item)
    return {"strata": strata, "cross_state_curves": curves, "paired": paired, "state_diagnostics": diagnostics}


def print_summary(summary, data):
    print(f"delta_total={DELTA_TOTAL} delta_off={DELTA_OFF} delta_on={DELTA_ON} primary_budget={PRIMARY_BUDGET}")
    print("budget method cumulative_mean cumulative_median cumulative_p90 simple_mean simple_median simple_p90 optimal_rate mean_rank admissible sampled certified mean_cert_time conflicts")
    for budget in BUDGETS:
        for method in METHODS:
            item = summary["cross_state_curves"][f"budget={budget}:method={method}"]
            print(budget, method, *item.values())
    print("PAIRED_DIFFERENCES mean standard_error ci95_lower ci95_upper joint_lower_fraction equal_fraction")
    for budget in BUDGETS:
        for label in ("joint-scratch", "joint-separate"):
            for metric in ("cumulative", "simple"):
                item = summary["paired"][f"budget={budget}:{label}:{metric}"]
                print(budget, label, metric, *item.values())
    print("STATE_DIAGNOSTICS")
    for item in summary["state_diagnostics"]:
        print(item)
    primary = {}
    for label in ("joint-scratch", "joint-separate"):
        for metric in ("cumulative", "simple"):
            primary[label, metric] = summary["paired"][f"budget={PRIMARY_BUDGET}:{label}:{metric}"]["mean"] < 0.0
    print(f"wrong_certifications={np.sum(data['wrong_certifications'])}")
    print(f"optimal_action_deletions={np.sum(data['optimal_action_deleted'])}")
    print(f"conflicts={np.sum(data['conflict_counts'])}")
    print(f"interval_violations={np.sum(data['interval_violations'])}")
    print("action_level_q_certificate_available = False")
    print(f"JOINT_LOWER_CUMULATIVE_REGRET_VS_SCRATCH_AT_1000 = {primary['joint-scratch', 'cumulative']}")
    print(f"JOINT_LOWER_CUMULATIVE_REGRET_VS_SEPARATE_AT_1000 = {primary['joint-separate', 'cumulative']}")
    print(f"JOINT_LOWER_SIMPLE_REGRET_VS_SCRATCH_AT_1000 = {primary['joint-scratch', 'simple']}")
    print(f"JOINT_LOWER_SIMPLE_REGRET_VS_SEPARATE_AT_1000 = {primary['joint-separate', 'simple']}")
    return primary


def save_outputs(reference, oracle_values, rankings, data, summary):
    output = ROOT / "artifacts" / "phase4b"
    output.mkdir(parents=True, exist_ok=True)
    gaps = np.array([oracle_values[i, rankings[i, 0]] - oracle_values[i, rankings[i, 1]] for i in range(len(STATES))])
    np.savez_compressed(
        output / "fixed_budget_online_efficiency_audit.npz",
        states=STATES, action_grid=ACTION_VALUES, offline_sample_sizes=SAMPLE_SIZES,
        budgets=BUDGETS, oracle_q=oracle_values, oracle_rankings=rankings,
        top_two_gaps=gaps, **data,
    )
    with (output / "fixed_budget_online_efficiency_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return output


def main():
    try:
        reference = load_reference(ROOT / "artifacts" / "phase1b" / "oracle_reference.npz")
        oracle_values, rankings, data = run_experiments(reference)
        if np.any(data["cumulative_regret"] < -TOL) or np.any(data["simple_regret"] < -TOL):
            raise RuntimeError("negative regret detected")
        if np.any(np.diff(data["cumulative_regret"], axis=-1) < -TOL):
            raise RuntimeError("cumulative regret is not monotone")
        if np.sum(data["interval_violations"]):
            raise RuntimeError("interval range or order violation")
        summary = build_summary(oracle_values, rankings, data)
        print_summary(summary, data)
        output = save_outputs(reference, oracle_values, rankings, data, summary)
    except Exception as exc:
        print(f"FAIL {exc}"); print("PHASE4B_MISMATCH"); return 1
    print(f"artifact={output / 'fixed_budget_online_efficiency_audit.npz'}")
    print(f"summary={output / 'fixed_budget_online_efficiency_summary.json'}")
    print("PHASE4B_FAITHFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
