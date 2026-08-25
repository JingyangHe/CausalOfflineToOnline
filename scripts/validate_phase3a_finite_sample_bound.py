"""Repeated fixed-state audit of finite-sample population joint bounds."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from confounded_smooth_regulator import return_bounds  # noqa: E402
from exact_population_joint_bound import (  # noqa: E402
    build_source_atoms,
    evaluate_population_query,
    reference_safe_rho,
)
from finite_sample_joint_bound import (  # noqa: E402
    build_empirical_source_atoms,
    build_probability_intervals,
    prepare_finite_sample_joint_problem,
    solve_finite_sample_joint_interval,
    solve_finite_sample_separate_interval,
    support_failure_bound,
)
from generate_offline_dataset import generate_fixed_state_dataset  # noqa: E402


H_VALUES = (1, 10, 20)
STATE_VALUES = (-0.6, 0.0, 0.6)
ACTION_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)
SAMPLE_SIZES = (25, 50, 100, 250)
DELTA = 0.05
REPLICATES = 20

ARTIFACT_FIELDS = """
replicate h state action sample_size delta certified confidence_event support_complete
probabilities_inside_intervals reference_q finite_joint_lower finite_joint_upper
finite_separate_lower finite_separate_upper population_joint_lower population_joint_upper
joint_width separate_width population_joint_width width_inflation_over_population
finite_upper_gain finite_lower_gain finite_width_gain fallback_code
""".split()


def load_reference(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required Phase 1B oracle artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def source_atoms_from_prefix(train: dict, reference: dict, h: int, state: float, n: int):
    atoms = []
    for source in (1, 2, 3):
        selected = (
            (train["time_step"] == h) & (train["state"] == state)
            & (train["source_id"] == source) & (train["sample_index"] < n)
        )
        atoms.append(build_empirical_source_atoms(
            {key: values[selected] for key, values in train.items()}, reference, h
        ))
    return atoms


def audit_confidence_event(
    atoms: list[dict], intervals: dict, reference: dict, h: int, state: float
):
    support_complete, probabilities_inside = True, True
    for source_index, source in enumerate((1, 2, 3)):
        truth = build_source_atoms(reference, h, state, source)
        for action, outcome, probability in zip(
            truth["actions"], truth["outcomes"], truth["probabilities"]
        ):
            matches = np.flatnonzero(
                np.isclose(atoms[source_index]["actions"], action, atol=1e-12, rtol=0)
                & np.isclose(atoms[source_index]["outcomes"], outcome, atol=1e-12, rtol=0)
            )
            support_complete &= len(matches) == 1
            if len(matches) == 1:
                index = matches[0]
                probabilities_inside &= (
                    intervals["probability_lower"][source_index][index] <= probability
                    <= intervals["probability_upper"][source_index][index])
            else:
                probabilities_inside = False
    return bool(support_complete), bool(probabilities_inside)


def precompute_population(reference: dict) -> dict:
    return {(h, state, action): evaluate_population_query(reference, h, state, action)
            for h in H_VALUES for state in STATE_VALUES for action in ACTION_VALUES}


def make_record(
    replicate, h, state, action, n, intervals, support_complete, probabilities_inside,
    joint, separate, population, fallback_code,
):
    joint_width = joint["joint_upper"] - joint["joint_lower"]
    separate_width = separate["separate_upper"] - separate["separate_lower"]
    population_width = population["joint_upper"] - population["joint_lower"]
    return {
        "replicate": replicate, "h": h, "state": state, "action": action,
        "sample_size": n, "delta": DELTA, "certified": intervals["certified"],
        "confidence_event": support_complete and probabilities_inside, "support_complete": support_complete,
        "probabilities_inside_intervals": probabilities_inside,
        "reference_q": population["reference_q"],
        "finite_joint_lower": joint["joint_lower"], "finite_joint_upper": joint["joint_upper"],
        "finite_separate_lower": separate["separate_lower"], "finite_separate_upper": separate["separate_upper"],
        "population_joint_lower": population["joint_lower"], "population_joint_upper": population["joint_upper"],
        "joint_width": joint_width, "separate_width": separate_width,
        "population_joint_width": population_width, "width_inflation_over_population": joint_width - population_width,
        "finite_upper_gain": separate["separate_upper"] - joint["joint_upper"], "finite_lower_gain": joint["joint_lower"] - separate["separate_lower"],
        "finite_width_gain": separate_width - joint_width, "fallback_code": fallback_code,
    }


def run_experiments(reference: dict):
    population = precompute_population(reference)
    records, set_events = [], []
    counts = {
        "total_confidence_sets": 0, "total_action_queries": 0,
        "certified_set_count": 0, "insufficient_sample_fallback_count": 0,
        "infeasible_joint_fallback_count": 0, "infeasible_separate_fallback_count": 0,
        "lp_failures": 0, "dominance_violations": 0, "interval_order_violations": 0,
        "range_violations": 0, "confidence_event_count": 0,
        "confidence_event_failures": 0,
        "confidence_event_validity_violations": 0, "all_query_validity_violations": 0,
    }
    max_solver_violation = 0.0
    for replicate in range(REPLICATES):
        train, audit = generate_fixed_state_dataset(H_VALUES, STATE_VALUES, max(SAMPLE_SIZES), 4000 + replicate)
        if len(train["row_id"]) != len(audit["row_id"]):
            raise RuntimeError("fixed-state train/audit row mismatch")
        for h in H_VALUES:
            B_minus, B_plus = return_bounds(h, 20, 0.95)
            rho = reference_safe_rho(reference, h)["reference_safe_rho_coefficient"]
            for state in STATE_VALUES:
                for n in SAMPLE_SIZES:
                    counts["total_confidence_sets"] += 1
                    atoms = source_atoms_from_prefix(train, reference, h, state, n)
                    intervals = build_probability_intervals(atoms, DELTA)
                    support_complete, probabilities_inside = audit_confidence_event(
                        atoms, intervals, reference, h, state)
                    confidence_event = support_complete and probabilities_inside
                    set_events.append((n, support_complete, probabilities_inside, confidence_event))
                    counts["certified_set_count"] += int(intervals["certified"])
                    counts["confidence_event_count"] += int(confidence_event)
                    counts["confidence_event_failures"] += int(not confidence_event)
                    if intervals["certified"]:
                        problem = prepare_finite_sample_joint_problem(atoms, intervals, rho)
                    else:
                        counts["insufficient_sample_fallback_count"] += 1
                        problem = None
                    for action in ACTION_VALUES:
                        counts["total_action_queries"] += 1
                        fallback_code = 0
                        try:
                            if problem is None:
                                joint = {"joint_lower": B_minus, "joint_upper": B_plus}
                                separate = {"separate_lower": B_minus, "separate_upper": B_plus}
                                fallback_code = 1
                            else:
                                joint = solve_finite_sample_joint_interval(
                                    problem, action, B_minus, B_plus
                                )
                                separate = solve_finite_sample_separate_interval(
                                    atoms, intervals, action, B_minus, B_plus, rho
                                )
                                if joint["fallback_reason"]:
                                    counts["infeasible_joint_fallback_count"] += 1
                                    fallback_code += 2
                                if separate["fallback_reason"]:
                                    counts["infeasible_separate_fallback_count"] += 1
                                    fallback_code += 4
                                max_solver_violation = max(
                                    max_solver_violation,
                                    joint["max_marginal_interval_violation"],
                                )
                        except RuntimeError as exc:
                            print(f"LP_FAILURE rep={replicate} h={h} state={state} n={n} a={action}: {exc}")
                            counts["lp_failures"] += 1
                            joint = {"joint_lower": B_minus, "joint_upper": B_plus}
                            separate = {"separate_lower": B_minus, "separate_upper": B_plus}
                            fallback_code = 8
                        record = make_record(
                            replicate, h, state, action, n, intervals, support_complete,
                            probabilities_inside, joint, separate, population[(h, state, action)], fallback_code)
                        tolerance = 1e-8
                        solved = fallback_code == 0
                        counts["interval_order_violations"] += int(
                            record["finite_joint_lower"] > record["finite_joint_upper"] + tolerance
                        )
                        if solved:
                            counts["dominance_violations"] += int(
                                record["finite_joint_lower"] < record["finite_separate_lower"] - tolerance
                                or record["finite_joint_upper"] > record["finite_separate_upper"] + tolerance
                            )
                        bounds = (record["finite_joint_lower"], record["finite_joint_upper"],
                                  record["finite_separate_lower"], record["finite_separate_upper"])
                        counts["range_violations"] += int(
                            min(bounds) < B_minus - tolerance or max(bounds) > B_plus + tolerance
                        )
                        covered = (record["finite_joint_lower"] <= record["reference_q"] + tolerance
                                   and record["reference_q"] <= record["finite_joint_upper"] + tolerance)
                        counts["all_query_validity_violations"] += int(not covered)
                        counts["confidence_event_validity_violations"] += int(confidence_event and not covered)
                        records.append(record)
    return records, set_events, counts, max_solver_violation


def print_statistics(records, set_events, counts, max_solver_violation):
    for key, value in counts.items():
        print(f"{key}={value}")
    print(f"empirical_coverage_rate={1 - counts['all_query_validity_violations'] / len(records):.8f}")
    print(f"max_solver_violation={max_solver_violation:.3e}")
    for n in SAMPLE_SIZES:
        subset = [record for record in records if record["sample_size"] == n]
        events = [event for event in set_events if event[0] == n]
        values = lambda key: np.asarray([record[key] for record in subset])
        coverage = np.mean((values("finite_joint_lower") <= values("reference_q") + 1e-8)
                           & (values("reference_q") <= values("finite_joint_upper") + 1e-8))
        print(f"sample_size={n} support_failure_bound={support_failure_bound(n):.8g}")
        print(f"sample_size={n} support_complete_rate={np.mean([e[1] for e in events]):.8f}")
        print(f"sample_size={n} probability_interval_event_rate={np.mean([e[2] for e in events]):.8f}")
        print(f"sample_size={n} confidence_event_rate={np.mean([e[3] for e in events]):.8f}")
        print(f"sample_size={n} empirical_coverage_rate={coverage:.8f}")
        for key in ("joint_width", "separate_width", "population_joint_width",
                    "width_inflation_over_population", "finite_upper_gain", "finite_lower_gain",
                    "finite_width_gain"):
            print(f"sample_size={n} mean_{key}={values(key).mean():.10f}")
        print(f"sample_size={n} fraction_finite_width_strict={np.mean(values('finite_width_gain') > 1e-10):.8f}")


def save_outputs(records, reference):
    output = ROOT / "artifacts" / "phase3a"
    output.mkdir(parents=True, exist_ok=True)
    smoke_train, smoke_audit = generate_fixed_state_dataset(H_VALUES, STATE_VALUES, 100, 3027)
    np.savez_compressed(output / "fixed_state_smoke_train.npz", **smoke_train)
    np.savez_compressed(output / "fixed_state_smoke_audit.npz", **smoke_audit)
    artifact = {key: np.asarray([record[key] for record in records]) for key in ARTIFACT_FIELDS}
    np.savez_compressed(output / "finite_sample_joint_audit.npz", **artifact)
    if len(smoke_train["row_id"]) != 2700 or int(reference["horizon"]) != 20:
        raise RuntimeError("saved Phase 3A artifact audit failed")


def main() -> int:
    try:
        reference = load_reference(ROOT / "artifacts" / "phase1b" / "oracle_reference.npz")
        records, events, counts, max_violation = run_experiments(reference)
        print_statistics(records, events, counts, max_violation)
        required_zero = ("lp_failures", "dominance_violations", "interval_order_violations",
                         "range_violations", "confidence_event_validity_violations")
        if any(counts[key] for key in required_zero) or max_violation > 1e-8:
            raise RuntimeError("one or more Phase 3A acceptance checks failed")
        save_outputs(records, reference)
        strict = any(record["finite_width_gain"] > 1e-10 for record in records)
    except Exception as exc:
        print(f"FAIL {exc}")
        print("FINITE_SAMPLE_STRICT_GAIN_OBSERVED = False")
        print("PHASE3A_MISMATCH")
        return 1
    print(f"FINITE_SAMPLE_STRICT_GAIN_OBSERVED = {strict}")
    print("PHASE3A_FAITHFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
