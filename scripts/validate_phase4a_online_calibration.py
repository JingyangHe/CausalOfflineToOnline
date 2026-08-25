"""Audit fixed-state online calibration initialized by Phase 3A intervals."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from confounded_smooth_regulator import return_bounds  # noqa: E402
from exact_population_joint_bound import build_source_atoms, reference_safe_rho  # noqa: E402
from finite_sample_joint_bound import (  # noqa: E402
    build_empirical_source_atoms,
    build_probability_intervals,
    prepare_finite_sample_joint_problem,
    solve_finite_sample_joint_interval,
    solve_finite_sample_separate_interval,
)
from fixed_state_online_calibration import run_fixed_state_online_calibration  # noqa: E402
from generate_offline_dataset import (  # noqa: E402
    generate_fixed_state_dataset,
    sample_fixed_state_online_intervention,
)
from oracle_ground_truth import oracle_q  # noqa: E402


DATA_H_VALUES = (1, 10, 20)
DATA_STATE_VALUES = (-0.6, 0.0, 0.6)
H_VALUES = (20,)
STATE_VALUES = (0.6,)
ACTION_VALUES = np.array((-1.0, -0.5, 0.0, 0.5, 1.0))
SAMPLE_SIZES = (25, 50, 100, 250)
METHODS = ("scratch", "separate", "joint")
DELTA_TOTAL, DELTA_OFF, DELTA_ON = 0.05, 0.025, 0.025
REPLICATES = 20
MAX_ONLINE_INTERACTIONS = 150000
EPSILON = 0.0

ARTIFACT_FIELDS = """
replicate h state sample_size method_code online_interactions certified
certified_action_index oracle_optimal_action_index final_admissible_count
history_covered oracle_optimum_always_admissible wrong_certification
oracle_optimum_eliminated conflict_count numerical_violations offline_confidence_event
""".split()


def load_reference(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required Oracle artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def source_atoms_from_prefix(train, reference, h, state, sample_size):
    atoms = []
    for source in (1, 2, 3):
        selected = (
            (train["time_step"] == h)
            & (train["state"] == state)
            & (train["source_id"] == source)
            & (train["sample_index"] < sample_size)
        )
        public_slice = {key: values[selected] for key, values in train.items()}
        atoms.append(build_empirical_source_atoms(public_slice, reference, h))
    return atoms


def offline_confidence_event(atoms, intervals, reference, h, state):
    for source_index, source in enumerate((1, 2, 3)):
        truth = build_source_atoms(reference, h, state, source)
        for action, outcome, probability in zip(
            truth["actions"], truth["outcomes"], truth["probabilities"]
        ):
            matches = np.flatnonzero(
                np.isclose(atoms[source_index]["actions"], action, atol=1e-12, rtol=0)
                & np.isclose(
                    atoms[source_index]["outcomes"], outcome, atol=1e-12, rtol=0
                )
            )
            if len(matches) != 1:
                return False
            index = int(matches[0])
            if not (
                intervals["probability_lower"][source_index][index]
                <= probability
                <= intervals["probability_upper"][source_index][index]
            ):
                return False
    return True


def phase3a_initial_intervals(atoms, intervals, reference, h):
    b_lower, b_upper = return_bounds(h, int(reference["horizon"]), float(reference["gamma"]))
    rho = reference_safe_rho(reference, h)["reference_safe_rho_coefficient"]
    problem = prepare_finite_sample_joint_problem(atoms, intervals, rho)
    joint_lower, joint_upper, separate_lower, separate_upper = [], [], [], []
    for action in ACTION_VALUES:
        joint = solve_finite_sample_joint_interval(problem, action, b_lower, b_upper)
        separate = solve_finite_sample_separate_interval(
            atoms, intervals, action, b_lower, b_upper, rho
        )
        if joint["fallback_reason"] or separate["fallback_reason"]:
            raise RuntimeError("unexpected Phase 3A confidence-problem fallback")
        joint_lower.append(joint["joint_lower"])
        joint_upper.append(joint["joint_upper"])
        separate_lower.append(separate["separate_lower"])
        separate_upper.append(separate["separate_upper"])
    return {
        "scratch": (
            np.full(len(ACTION_VALUES), b_lower), np.full(len(ACTION_VALUES), b_upper)
        ),
        "separate": (np.asarray(separate_lower), np.asarray(separate_upper)),
        "joint": (np.asarray(joint_lower), np.asarray(joint_upper)),
    }


def make_action_sampler(reference, h, state, seed_components):
    streams = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed_components).spawn(len(ACTION_VALUES))
    ]
    action_to_index = {float(action): index for index, action in enumerate(ACTION_VALUES)}
    observations = [[] for _ in ACTION_VALUES]

    def sample(action):
        index = action_to_index[action]
        outcome = sample_fixed_state_online_intervention(
            h, state, action, reference, streams[index]
        )
        observations[index].append(outcome)
        return outcome

    return sample, observations


def audit_run(result, true_q, initial_lower, initial_upper, observations, b_lower, b_upper):
    tolerance = 1e-10
    optimum = int(np.argmax(true_q))
    offline_covered = bool(
        np.all(initial_lower <= true_q + tolerance)
        and np.all(true_q <= initial_upper + tolerance)
    )
    online_covered = True
    value_range = b_upper - b_lower
    for action_index, action_observations in enumerate(observations):
        if not action_observations:
            continue
        values = np.asarray(action_observations)
        counts = np.arange(1, len(values) + 1, dtype=np.float64)
        means = np.cumsum(values) / counts
        radii = value_range * np.sqrt(
            np.log(2.0 * len(ACTION_VALUES) * counts * (counts + 1.0) / DELTA_ON)
            / (2.0 * counts)
        )
        online_covered &= bool(
            np.all(means - radii <= true_q[action_index] + tolerance)
            and np.all(true_q[action_index] <= means + radii + tolerance)
        )
    covered = offline_covered and online_covered
    retained = covered
    violations = int(result["interval_violation_count"])
    violations += int(
        np.any(result["final_lower"] < b_lower - tolerance)
        or np.any(result["final_upper"] > b_upper + tolerance)
        or np.any(result["final_lower"] > result["final_upper"] + tolerance)
    )
    wrong = bool(result["certified"] and result["certified_action_index"] != optimum)
    return bool(covered), bool(retained), wrong, int(violations), optimum


def make_record(replicate, h, state, sample_size, method, result, audit, offline_event):
    covered, retained, wrong, violations, optimum = audit
    certified_index = -1 if result["certified_action_index"] is None else result["certified_action_index"]
    return {
        "replicate": replicate,
        "h": h,
        "state": state,
        "sample_size": sample_size,
        "method_code": METHODS.index(method),
        "online_interactions": result["online_interactions"],
        "certified": result["certified"],
        "certified_action_index": certified_index,
        "oracle_optimal_action_index": optimum,
        "final_admissible_count": len(result["final_admissible_action_indices"]),
        "history_covered": covered,
        "oracle_optimum_always_admissible": retained,
        "wrong_certification": wrong,
        "oracle_optimum_eliminated": not retained,
        "conflict_count": result["conflict_count"],
        "numerical_violations": violations,
        "offline_confidence_event": offline_event,
    }


def run_experiments(reference):
    records = []
    for replicate in range(REPLICATES):
        train, _ = generate_fixed_state_dataset(
            DATA_H_VALUES, DATA_STATE_VALUES, max(SAMPLE_SIZES), 4000 + replicate
        )
        for h in H_VALUES:
            h_index = DATA_H_VALUES.index(h)
            b_lower, b_upper = return_bounds(
                h, int(reference["horizon"]), float(reference["gamma"])
            )
            for state in STATE_VALUES:
                state_index = DATA_STATE_VALUES.index(state)
                true_q = np.asarray(oracle_q(reference, h, state, ACTION_VALUES))
                for sample_size in SAMPLE_SIZES:
                    atoms = source_atoms_from_prefix(train, reference, h, state, sample_size)
                    intervals = build_probability_intervals(atoms, DELTA_OFF)
                    if not intervals["certified"]:
                        raise RuntimeError("Phase 3A interval is uncertified at a formal sample size")
                    initial = phase3a_initial_intervals(atoms, intervals, reference, h)
                    offline_event = offline_confidence_event(
                        atoms, intervals, reference, h, state
                    )
                    base_components = [9100, replicate, h_index, state_index, sample_size]
                    tie_seed = int(
                        np.random.SeedSequence(base_components + [0]).generate_state(1)[0]
                    )
                    for method in METHODS:
                        sample, observations = make_action_sampler(
                            reference, h, state, base_components + [1]
                        )
                        result = run_fixed_state_online_calibration(
                            ACTION_VALUES,
                            initial[method][0],
                            initial[method][1],
                            sample,
                            b_lower,
                            b_upper,
                            DELTA_ON,
                            MAX_ONLINE_INTERACTIONS,
                            tie_seed,
                            record_history=False,
                            epsilon=EPSILON,
                        )
                        audit = audit_run(
                            result, true_q, initial[method][0], initial[method][1],
                            observations, b_lower, b_upper,
                        )
                        records.append(
                            make_record(
                                replicate, h, state, sample_size, method,
                                result, audit, offline_event,
                            )
                        )
    return records


def subset_values(records, sample_size, method, key):
    code = METHODS.index(method)
    return np.asarray(
        [
            record[key]
            for record in records
            if record["sample_size"] == sample_size and record["method_code"] == code
        ]
    )


def print_statistics(records):
    print(f"delta_total={DELTA_TOTAL} delta_off={DELTA_OFF} delta_on={DELTA_ON}")
    print(f"max_online_interactions={MAX_ONLINE_INTERACTIONS} epsilon={EPSILON}")
    for sample_size in SAMPLE_SIZES:
        means = {}
        for method in METHODS:
            interactions = subset_values(records, sample_size, method, "online_interactions")
            certified = subset_values(records, sample_size, method, "certified")
            admissible = subset_values(records, sample_size, method, "final_admissible_count")
            coverage = subset_values(records, sample_size, method, "history_covered")
            zero = certified & (interactions == 0)
            means[method] = float(np.mean(interactions))
            print(
                f"n={sample_size} method={method} mean={np.mean(interactions):.6f} "
                f"median={np.median(interactions):.6f} p90={np.quantile(interactions, 0.9):.6f} "
                f"certification_rate={np.mean(certified):.8f} "
                f"zero_interaction_rate={np.mean(zero):.8f} "
                f"mean_final_admissible={np.mean(admissible):.6f} "
                f"anytime_coverage_rate={np.mean(coverage):.8f}"
            )
        scratch_saving = 1.0 - means["joint"] / means["scratch"]
        separate_saving = 1.0 - means["joint"] / means["separate"]
        print(
            f"n={sample_size} joint_saving_vs_scratch={scratch_saving:.8f} "
            f"joint_saving_vs_separate={separate_saving:.8f}"
        )
    for key in (
        "wrong_certification", "oracle_optimum_eliminated", "conflict_count",
        "numerical_violations",
    ):
        print(f"total_{key}={sum(record[key] for record in records)}")
    print(
        "overall_anytime_coverage_rate="
        f"{np.mean([record['history_covered'] for record in records]):.8f}"
    )
    print(
        "offline_confidence_event_rate="
        f"{np.mean([record['offline_confidence_event'] for record in records]):.8f}"
    )


def save_artifact(records):
    output = ROOT / "artifacts" / "phase4a"
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        key: np.asarray([record[key] for record in records]) for key in ARTIFACT_FIELDS
    }
    path = output / "fixed_state_online_calibration_audit.npz"
    np.savez_compressed(path, **artifact)
    return path


def main():
    try:
        reference = load_reference(ROOT / "artifacts" / "phase1b" / "oracle_reference.npz")
        records = run_experiments(reference)
        print_statistics(records)
        required_zero = (
            "wrong_certification", "oracle_optimum_eliminated", "numerical_violations"
        )
        if any(sum(record[key] for record in records) for key in required_zero):
            raise RuntimeError("a correctness audit failed")
        if not all(record["history_covered"] for record in records):
            raise RuntimeError("an anytime confidence sequence failed to cover Oracle Q")
        joint_beats_scratch = any(
            np.mean(subset_values(records, n, "joint", "online_interactions"))
            < np.mean(subset_values(records, n, "scratch", "online_interactions"))
            for n in SAMPLE_SIZES
        )
        joint_beats_separate = any(
            np.mean(subset_values(records, n, "joint", "online_interactions"))
            < np.mean(subset_values(records, n, "separate", "online_interactions"))
            for n in SAMPLE_SIZES
        )
        if not joint_beats_scratch:
            raise RuntimeError("joint initialization never improved mean scratch interactions")
        path = save_artifact(records)
    except Exception as exc:
        print(f"FAIL {exc}")
        print("JOINT_MEAN_LT_SCRATCH_OBSERVED = False")
        print("JOINT_MEAN_LT_SEPARATE_OBSERVED = False")
        print("PHASE4A_MISMATCH")
        return 1
    print(f"artifact={path}")
    print(f"JOINT_MEAN_LT_SCRATCH_OBSERVED = {joint_beats_scratch}")
    print(f"JOINT_MEAN_LT_SEPARATE_OBSERVED = {joint_beats_separate}")
    print("PHASE4A_FAITHFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
