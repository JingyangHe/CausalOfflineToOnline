"""Phase 8F-Q: exact multi-source natural-bound intersection diagnostic.

The experiment is deliberately population-only and NumPy-only.  Canonical
outcome branches are used to enumerate the controlled DGP, construct the
diagnostic support envelope, and evaluate coverage/regret.  Observational
quantities are always obtained by marginalizing the joint
``source, U_behavior, action, U_environment, reward`` population mass; a do
mean is never an input to that calculation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .multisource_contrast_calibration import (
    ACTION_KEYS,
    multisource_behavior_probabilities,
)
from .reward_mechanism_separation import kappa_name


STAGE = "Phase 8F-Q"
CONDITIONS = ("confounded", "independent_latents")
FIXED_KAPPAS = (0.0, 0.3)
FIXED_LAMBDAS = (0.0, 0.01, 0.05)
FIXED_SOURCE_SETTINGS = (
    "M2_same_direction_diverse",
    "M5_same_direction_diverse",
    "M5_redundant",
    "original_opposite_M2",
)
NUM_ANCHORS = 2048
NUMERICAL_TOLERANCE = 1e-10
ORACLE_SUPPORT_ENVELOPE_DIAGNOSTIC_ONLY = True
EXPECTED_FILENAMES = {
    "manifest.json",
    "input_integrity.json",
    "hard_checks.json",
    "bound_metrics.csv",
    "certification_metrics.csv",
    "regret_metrics.csv",
    "anchor_action_bounds.npz",
    "summary.json",
    "REPORT.md",
    "interval_width_by_source_setting.png",
    "intersection_width_reduction.png",
    "action_certification_rate.png",
    "robust_decision_regret.png",
}


class Phase8FQuickBoundError(RuntimeError):
    """Raised when an input or scientific invariant fails."""


def _same_direction_probabilities(values: Sequence[float]) -> np.ndarray:
    return multisource_behavior_probabilities(values)


def original_opposite_probabilities() -> np.ndarray:
    """P(A | U, source) for the original two deterministic opposite loggers."""
    table = np.zeros((2, 2, 3), dtype=np.float64)
    # U order is (-1,+1); action order is (minus,base,plus).
    table[0, 0, 0] = 1.0
    table[0, 1, 2] = 1.0
    table[1, 0, 2] = 1.0
    table[1, 1, 0] = 1.0
    return table


def source_probability_tables() -> dict[str, np.ndarray]:
    return {
        "M2_same_direction_diverse": _same_direction_probabilities((0.55, 0.95)),
        "M5_same_direction_diverse": _same_direction_probabilities(
            (0.55, 0.65, 0.75, 0.85, 0.95)
        ),
        "M5_redundant": _same_direction_probabilities((0.75,) * 5),
        "original_opposite_M2": original_opposite_probabilities(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): _sha256(Path(path)) for path in paths}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase8FQuickBoundError(f"required read-only input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_all_passed(path: Path) -> None:
    record = _read_json(path)
    if record.get("all_passed") is not True or not all(record.get("checks", {}).values()):
        raise Phase8FQuickBoundError(f"input hard checks did not all pass: {path}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise Phase8FQuickBoundError(f"refusing to write empty metric table: {path.name}")
    fields: list[str] = []
    for row in records:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise Phase8FQuickBoundError(f"required read-only input is missing: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def resolve_phase8f_inputs(
    phase8a_root: Path,
    phase8anc_root: Path,
    phase8eq_root: Path,
    kappas: Sequence[float],
) -> dict[str, Any]:
    phase8a = Path(phase8a_root).resolve()
    phase8anc = Path(phase8anc_root).resolve()
    phase8eq = Path(phase8eq_root).resolve()
    required = [
        phase8anc / "manifest.json",
        phase8anc / "hard_checks.json",
        phase8eq / "manifest.json",
        phase8eq / "hard_checks.json",
        phase8eq / "shared_arrays.npz",
    ]
    for kappa in kappas:
        required.append(phase8a / kappa_name(float(kappa)) / "do_oracle_raw.npz")
    # Some server clones retain only the raw Phase 8A oracle files.  When the
    # root-level provenance pair exists, require and hash both; an incomplete
    # pair is never accepted.
    root_manifest = phase8a / "manifest.json"
    root_checks = phase8a / "hard_checks.json"
    if root_manifest.is_file() != root_checks.is_file():
        raise Phase8FQuickBoundError("Phase 8A manifest/hard-check pair is incomplete")
    if root_manifest.is_file():
        required.extend((root_manifest, root_checks))
        _require_all_passed(root_checks)
    for path in required:
        if not path.is_file():
            raise Phase8FQuickBoundError(f"required read-only input is missing: {path}")
    _require_all_passed(phase8anc / "hard_checks.json")
    _require_all_passed(phase8eq / "hard_checks.json")

    anc_manifest = _read_json(phase8anc / "manifest.json")
    eq_manifest = _read_json(phase8eq / "manifest.json")
    expected = {
        "M2_diverse": [0.55, 0.95],
        "M5_diverse": [0.55, 0.65, 0.75, 0.85, 0.95],
        "M5_redundant": [0.75] * 5,
    }
    source_record = eq_manifest.get("source_settings", {})
    quick_tables_verified = all(
        name in source_record
        and np.allclose(source_record[name], values, atol=0.0, rtol=0.0)
        for name, values in expected.items()
    )
    expected_nc_tables = {
        "0": {
            "-1": {"minus": 0.9, "plus": 0.1},
            "1": {"minus": 0.1, "plus": 0.9},
        },
        "1": {
            "-1": {"minus": 0.7, "plus": 0.3},
            "1": {"minus": 0.3, "plus": 0.7},
        },
        "2": {"-1": {"base": 1.0}, "1": {"base": 1.0}},
    }
    nc_tables_verified = (
        anc_manifest.get("logger_probability_tables") == expected_nc_tables
        and anc_manifest.get("action_keys") == list(ACTION_KEYS)
    )
    source_tables_verified = quick_tables_verified and nc_tables_verified
    if not source_tables_verified:
        raise Phase8FQuickBoundError("Phase 8E-Q source probability settings differ")
    return {
        "phase8a": phase8a,
        "phase8anc": phase8anc,
        "phase8eq": phase8eq,
        "required_paths": tuple(dict.fromkeys(required)),
        "phase8a_root_checks_available": root_checks.is_file(),
        "source_tables_verified": source_tables_verified,
    }


def canonical_reward_branches(
    raw: Mapping[str, np.ndarray], anchor_ids: Sequence[int]
) -> np.ndarray:
    required = {"anchor_id", "action_key", "u_env", "reward"}
    if not required.issubset(raw):
        raise Phase8FQuickBoundError("Phase 8A oracle is missing canonical reward fields")
    ids = np.asarray(anchor_ids, dtype=np.int64)
    raw_ids = np.asarray(raw["anchor_id"], dtype=np.int64)
    raw_actions = np.asarray(raw["action_key"]).astype(str)
    raw_u = np.asarray(raw["u_env"], dtype=np.int64)
    raw_reward = np.asarray(raw["reward"], dtype=np.float64)
    branches = np.empty((len(ids), 3, 2), dtype=np.float64)
    for position, anchor in enumerate(ids):
        for action, action_key in enumerate(ACTION_KEYS):
            for latent_index, latent in enumerate((-1, 1)):
                rows = np.flatnonzero(
                    (raw_ids == anchor) & (raw_actions == action_key) & (raw_u == latent)
                )
                if len(rows) != 1:
                    raise Phase8FQuickBoundError(
                        f"canonical outcome is not unique: {anchor}/{action_key}/{latent}"
                    )
                branches[position, action, latent_index] = raw_reward[rows[0]]
    if not np.all(np.isfinite(branches)):
        raise Phase8FQuickBoundError("canonical rewards contain NaN or Inf")
    return branches


def augmented_canonical_rewards(
    original_reward_branches: np.ndarray, lambda_reward: float
) -> np.ndarray:
    branches = np.asarray(original_reward_branches, dtype=np.float64)
    if branches.ndim != 3 or branches.shape[1:] != (3, 2):
        raise ValueError("canonical rewards must have shape [anchor,action,2]")
    return branches + float(lambda_reward) * np.asarray((-1.0, 1.0))[None, None, :]


def observational_population(
    behavior_probabilities: np.ndarray,
    canonical_rewards: np.ndarray,
    condition: str,
) -> dict[str, np.ndarray]:
    """Compute pi and mu from exact observational joint probability mass.

    This function intentionally has no do-reward argument.  ``canonical_rewards``
    are the two consistency branches of the controlled population, not an
    intervention mean.
    """
    behavior = np.asarray(behavior_probabilities, dtype=np.float64)
    rewards = np.asarray(canonical_rewards, dtype=np.float64)
    if behavior.ndim != 3 or behavior.shape[1:] != (2, 3):
        raise ValueError("behavior probabilities must have shape [source,2,3]")
    if rewards.ndim != 3 or rewards.shape[1:] != (3, 2):
        raise ValueError("canonical rewards must have shape [anchor,action,2]")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if np.any(behavior < 0) or not np.allclose(
        behavior.sum(axis=2), 1.0, atol=1e-14, rtol=0.0
    ):
        raise ValueError("behavior probability table is invalid")

    source_count = behavior.shape[0]
    anchor_count = rewards.shape[0]
    pi = 0.5 * behavior.sum(axis=1)
    reward_mass = np.zeros((source_count, anchor_count, 3), dtype=np.float64)
    if condition == "confounded":
        for latent_index in range(2):
            reward_mass += (
                0.5
                * behavior[:, latent_index, :][:, None, :]
                * rewards[None, :, :, latent_index]
            )
    else:
        reward_average = rewards.mean(axis=2)
        reward_mass = pi[:, None, :] * reward_average[None, :, :]
    mu = np.zeros_like(reward_mass)
    np.divide(
        reward_mass,
        pi[:, None, :],
        out=mu,
        where=pi[:, None, :] > NUMERICAL_TOLERANCE,
    )
    return {
        "pi": pi,
        "mu": mu,
        "reward_mass": reward_mass,
        "supported": pi > NUMERICAL_TOLERANCE,
    }


def natural_reward_bounds(
    observational: Mapping[str, np.ndarray], reward_min: float, reward_max: float
) -> tuple[np.ndarray, np.ndarray]:
    pi = np.asarray(observational["pi"], dtype=np.float64)
    reward_mass = np.asarray(observational["reward_mass"], dtype=np.float64)
    lower = reward_mass + (1.0 - pi[:, None, :]) * float(reward_min)
    upper = reward_mass + (1.0 - pi[:, None, :]) * float(reward_max)
    return lower, upper


def pool_observational_mass(
    observational: Mapping[str, np.ndarray], source_weights: Sequence[float]
) -> dict[str, np.ndarray]:
    pi = np.asarray(observational["pi"], dtype=np.float64)
    reward_mass = np.asarray(observational["reward_mass"], dtype=np.float64)
    weights = np.asarray(source_weights, dtype=np.float64)
    if weights.shape != (len(pi),) or np.any(weights < 0) or not np.isclose(
        weights.sum(), 1.0
    ):
        raise ValueError("source weights are invalid")
    pooled_pi = weights @ pi
    pooled_mass = np.tensordot(weights, reward_mass, axes=(0, 0))
    pooled_mu = np.zeros_like(pooled_mass)
    np.divide(
        pooled_mass,
        pooled_pi[None, :],
        out=pooled_mu,
        where=pooled_pi[None, :] > NUMERICAL_TOLERANCE,
    )
    return {
        "pi": pooled_pi[None, :],
        "mu": pooled_mu[None, :, :],
        "reward_mass": pooled_mass[None, :, :],
        "supported": (pooled_pi > NUMERICAL_TOLERANCE)[None, :],
    }


def direct_pooled_joint_mass(
    behavior_probabilities: np.ndarray,
    canonical_rewards: np.ndarray,
    condition: str,
    source_weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Independent enumerated check of pooled pi and reward numerator."""
    behavior = np.asarray(behavior_probabilities, dtype=np.float64)
    rewards = np.asarray(canonical_rewards, dtype=np.float64)
    weights = np.asarray(source_weights, dtype=np.float64)
    pooled_pi = np.zeros(3, dtype=np.float64)
    pooled_mass = np.zeros((rewards.shape[0], 3), dtype=np.float64)
    for source in range(behavior.shape[0]):
        for u_behavior_index in range(2):
            action_probability = behavior[source, u_behavior_index]
            pooled_pi += weights[source] * 0.5 * action_probability
            if condition == "confounded":
                outcome = rewards[:, :, u_behavior_index]
                pooled_mass += (
                    weights[source] * 0.5 * action_probability[None, :] * outcome
                )
            else:
                for u_environment_index in range(2):
                    outcome = rewards[:, :, u_environment_index]
                    pooled_mass += (
                        weights[source]
                        * 0.25
                        * action_probability[None, :]
                        * outcome
                    )
    return pooled_pi, pooled_mass


def interval_intersection(
    lower: np.ndarray, upper: np.ndarray, tolerance: float = NUMERICAL_TOLERANCE
) -> tuple[np.ndarray, np.ndarray]:
    cap_lower = np.max(np.asarray(lower, dtype=np.float64), axis=0)
    cap_upper = np.min(np.asarray(upper, dtype=np.float64), axis=0)
    if np.any(cap_lower > cap_upper + tolerance):
        raise Phase8FQuickBoundError("multi-source interval intersection is empty")
    return cap_lower, cap_upper


def best_single_interval(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    widths = np.asarray(upper) - np.asarray(lower)
    index = np.argmin(widths, axis=0)
    selected_lower = np.take_along_axis(lower, index[None, :, :], axis=0)[0]
    selected_upper = np.take_along_axis(upper, index[None, :, :], axis=0)[0]
    return selected_lower, selected_upper, index


def certified_actions(
    lower: np.ndarray, upper: np.ndarray, tolerance: float = NUMERICAL_TOLERANCE
) -> np.ndarray:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    result = np.full(lo.shape[0], -1, dtype=np.int8)
    for action in range(lo.shape[1]):
        competitors = np.max(np.delete(hi, action, axis=1), axis=1)
        result[lo[:, action] > competitors + tolerance] = action
    return result


def _top_membership(values: np.ndarray, tolerance: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array >= np.max(array, axis=1, keepdims=True) - tolerance


def _decision_regret(do_reward: np.ndarray, action: np.ndarray) -> np.ndarray:
    truth = np.asarray(do_reward, dtype=np.float64)
    chosen = np.asarray(action, dtype=np.int64)
    return np.max(truth, axis=1) - truth[np.arange(len(truth)), chosen]


def _scenario_token(kappa: float, dose: float, setting: str, condition: str) -> str:
    def token(value: float) -> str:
        return f"{value:.6g}".replace("-", "m").replace(".", "p")

    return f"kappa_{token(kappa)}__lambda_{token(dose)}__{setting}__{condition}"


def _labels(kappa: float, dose: float, setting: str, condition: str) -> dict[str, Any]:
    return {
        "kappa": float(kappa),
        "lambda_reward": float(dose),
        "source_setting": setting,
        "condition": condition,
    }


def _coverage(lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.mean(
            (truth >= lower - NUMERICAL_TOLERANCE)
            & (truth <= upper + NUMERICAL_TOLERANCE)
        )
    )


def _bound_row(
    labels: Mapping[str, Any],
    method: str,
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    *,
    source_index: int | None = None,
    reward_min: float,
    reward_max: float,
    pooled_width: float | None = None,
    best_width: float | None = None,
) -> dict[str, Any]:
    width = np.asarray(upper) - np.asarray(lower)
    phi_gap = np.max(upper, axis=1) - np.max(truth, axis=1)
    row = {
        **labels,
        "method": method,
        "source_index": "" if source_index is None else source_index,
        "reward_support_min": reward_min,
        "reward_support_max": reward_max,
        "coverage_fraction": _coverage(lower, upper, truth),
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "mean_one_step_upper_value_proxy_gap": float(np.mean(phi_gap)),
        "median_one_step_upper_value_proxy_gap": float(np.median(phi_gap)),
        "proxy_label": "one-step upper-value proxy",
    }
    if pooled_width is not None:
        row["intersection_width_reduction_vs_pooled"] = pooled_width - float(np.mean(width))
        row["intersection_width_reduction_vs_pooled_fraction"] = (
            (pooled_width - float(np.mean(width))) / pooled_width if pooled_width else 0.0
        )
    if best_width is not None:
        row["intersection_width_reduction_vs_best_single"] = best_width - float(np.mean(width))
        row["intersection_width_reduction_vs_best_single_fraction"] = (
            (best_width - float(np.mean(width))) / best_width if best_width else 0.0
        )
    return row


def _certification_row(
    labels: Mapping[str, Any],
    method: str,
    lower: np.ndarray,
    upper: np.ndarray,
    do_reward: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    certified = certified_actions(lower, upper)
    mask = certified >= 0
    top = _top_membership(do_reward, NUMERICAL_TOLERANCE)
    correct = np.zeros(len(certified), dtype=bool)
    correct[mask] = top[np.flatnonzero(mask), certified[mask]]
    false_count = int(np.sum(mask & ~correct))
    return (
        {
            **labels,
            "method": method,
            "anchor_count": len(certified),
            "certified_anchor_count": int(mask.sum()),
            "certified_anchor_fraction": float(mask.mean()),
            "correct_certification_count": int(np.sum(mask & correct)),
            "false_certification_count": false_count,
            "certification_accuracy_when_certified": (
                float(np.mean(correct[mask])) if np.any(mask) else 1.0
            ),
        },
        certified,
    )


def _regret_row(
    labels: Mapping[str, Any], method: str, do_reward: np.ndarray, action: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    regret = _decision_regret(do_reward, action)
    return (
        {
            **labels,
            "method": method,
            "mean_do_regret": float(np.mean(regret)),
            "median_do_regret": float(np.median(regret)),
            "p90_do_regret": float(np.quantile(regret, 0.9)),
            "max_do_regret": float(np.max(regret)),
            "zero_regret_fraction": float(
                np.mean(regret <= NUMERICAL_TOLERANCE)
            ),
        },
        regret,
    )


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _aggregate_answer_rows(
    bound_rows: Sequence[Mapping[str, Any]],
    certification_rows: Sequence[Mapping[str, Any]],
    regret_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def select(rows: Sequence[Mapping[str, Any]], **filters: Any) -> list[Mapping[str, Any]]:
        return [row for row in rows if all(row.get(key) == value for key, value in filters.items())]

    width_answers: dict[str, Any] = {}
    for setting in ("M2_same_direction_diverse", "M5_same_direction_diverse", "M5_redundant"):
        rows = select(
            bound_rows,
            source_setting=setting,
            condition="confounded",
            method="intersection",
        )
        width_answers[setting] = {
            "mean_reduction_vs_equal_pooled_fraction": float(
                np.mean([row["intersection_width_reduction_vs_pooled_fraction"] for row in rows])
            ),
            "minimum_reduction_vs_equal_pooled_fraction": float(
                np.min([row["intersection_width_reduction_vs_pooled_fraction"] for row in rows])
            ),
            "maximum_reduction_vs_equal_pooled_fraction": float(
                np.max([row["intersection_width_reduction_vs_pooled_fraction"] for row in rows])
            ),
            "mean_reduction_vs_best_single_fraction": float(
                np.mean([row["intersection_width_reduction_vs_best_single_fraction"] for row in rows])
            ),
        }

    diverse = select(
        bound_rows,
        source_setting="M5_same_direction_diverse",
        condition="confounded",
        method="intersection",
    )
    redundant = select(
        bound_rows,
        source_setting="M5_redundant",
        condition="confounded",
        method="intersection",
    )
    keyed = lambda rows: {
        (row["kappa"], row["lambda_reward"]): row for row in rows
    }
    diverse_by_key, redundant_by_key = keyed(diverse), keyed(redundant)
    diversity_differences = [
        redundant_by_key[key]["mean_interval_width"]
        - diverse_by_key[key]["mean_interval_width"]
        for key in sorted(diverse_by_key)
    ]
    m2_by_key = keyed(
        select(
            bound_rows,
            source_setting="M2_same_direction_diverse",
            condition="confounded",
            method="intersection",
        )
    )
    m5_minus_m2 = [
        diverse_by_key[key]["mean_interval_width"]
        - m2_by_key[key]["mean_interval_width"]
        for key in sorted(diverse_by_key)
    ]

    cert = select(
        certification_rows,
        condition="confounded",
        method="intersection",
    )
    certification_by_setting = {
        setting: {
            "mean_certified_anchor_fraction": float(
                np.mean(
                    [
                        row["certified_anchor_fraction"]
                        for row in cert
                        if row["source_setting"] == setting
                    ]
                )
            ),
            "maximum_certified_anchor_fraction": float(
                np.max(
                    [
                        row["certified_anchor_fraction"]
                        for row in cert
                        if row["source_setting"] == setting
                    ]
                )
            ),
        }
        for setting in FIXED_SOURCE_SETTINGS
    }
    regret_cap = select(regret_rows, condition="confounded", method="intersection_lower")
    cap_comparisons = []
    for row in regret_cap:
        key = (row["kappa"], row["lambda_reward"])
        # Include setting because keyed() above intentionally cannot be used here.
        pool = next(
            item
            for item in regret_rows
            if item["condition"] == "confounded"
            and item["method"] == "pooled_equal_lower"
            and item["source_setting"] == row["source_setting"]
            and item["kappa"] == key[0]
            and item["lambda_reward"] == key[1]
        )
        best = next(
            item
            for item in regret_rows
            if item["condition"] == "confounded"
            and item["method"] == "best_single_lower"
            and item["source_setting"] == row["source_setting"]
            and item["kappa"] == key[0]
            and item["lambda_reward"] == key[1]
        )
        cap_comparisons.append(
            {
                "intersection_minus_pooled_mean_regret": row["mean_do_regret"]
                - pool["mean_do_regret"],
                "intersection_minus_best_single_mean_regret": row["mean_do_regret"]
                - best["mean_do_regret"],
                "intersection_mean_regret": row["mean_do_regret"],
                "pooled_mean_regret": pool["mean_do_regret"],
                "best_single_mean_regret": best["mean_do_regret"],
            }
        )
    return {
        "question_1_same_direction_tightening": width_answers,
        "question_2_diverse_vs_redundant": {
            "mean_redundant_minus_diverse_intersection_width": float(
                np.mean(diversity_differences)
            ),
            "paired_scenario_differences": diversity_differences,
            "diverse_tighter_scenario_count": int(
                np.sum(np.asarray(diversity_differences) > NUMERICAL_TOLERANCE)
            ),
            "paired_scenario_count": len(diversity_differences),
            "mean_M5_diverse_minus_M2_diverse_intersection_width": float(
                np.mean(m5_minus_m2)
            ),
            "M5_and_M2_same_extrema_intersections_equal": bool(
                np.allclose(m5_minus_m2, 0.0, atol=NUMERICAL_TOLERANCE, rtol=0)
            ),
        },
        "question_3_intersection_certification": {
            "mean_certified_anchor_fraction_confounded": float(
                np.mean([row["certified_anchor_fraction"] for row in cert])
            ),
            "total_false_certifications": int(
                sum(row["false_certification_count"] for row in cert)
            ),
            "by_source_setting": certification_by_setting,
        },
        "question_4_robust_action_regret": {
            "mean_intersection_regret": float(
                np.mean([row["intersection_mean_regret"] for row in cap_comparisons])
            ),
            "mean_pooled_regret": float(
                np.mean([row["pooled_mean_regret"] for row in cap_comparisons])
            ),
            "mean_best_single_regret": float(
                np.mean([row["best_single_mean_regret"] for row in cap_comparisons])
            ),
            "mean_intersection_minus_pooled_regret": float(
                np.mean(
                    [row["intersection_minus_pooled_mean_regret"] for row in cap_comparisons]
                )
            ),
            "mean_intersection_minus_best_single_regret": float(
                np.mean(
                    [
                        row["intersection_minus_best_single_mean_regret"]
                        for row in cap_comparisons
                    ]
                )
            ),
            "intersection_lower_regret_than_pooled_scenario_count": int(
                np.sum(
                    [
                        row["intersection_minus_pooled_mean_regret"]
                        < -NUMERICAL_TOLERANCE
                        for row in cap_comparisons
                    ]
                )
            ),
            "intersection_lower_regret_than_best_single_scenario_count": int(
                np.sum(
                    [
                        row["intersection_minus_best_single_mean_regret"]
                        < -NUMERICAL_TOLERANCE
                        for row in cap_comparisons
                    ]
                )
            ),
            "compared_scenario_count": len(cap_comparisons),
        },
    }


def _make_figures(
    output: Path,
    bound_rows: Sequence[Mapping[str, Any]],
    certification_rows: Sequence[Mapping[str, Any]],
    regret_rows: Sequence[Mapping[str, Any]],
) -> None:
    settings = list(FIXED_SOURCE_SETTINGS)
    short = ["M2 diverse", "M5 diverse", "M5 redundant", "opposite M2"]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    methods = ("best_single_width", "pooled_equal", "intersection")
    width = 0.24
    x = np.arange(len(settings))
    for offset, method in enumerate(methods):
        values = [
            np.mean(
                [
                    row["mean_interval_width"]
                    for row in bound_rows
                    if row["source_setting"] == setting
                    and row["condition"] == "confounded"
                    and row["method"] == method
                ]
            )
            for setting in settings
        ]
        axis.bar(x + (offset - 1) * width, values, width, label=method)
    axis.set_xticks(x, short, rotation=15)
    axis.set_ylabel("Mean interval width")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "interval_width_by_source_setting.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    pooled_reduction = []
    best_reduction = []
    for setting in settings:
        rows = [
            row
            for row in bound_rows
            if row["source_setting"] == setting
            and row["condition"] == "confounded"
            and row["method"] == "intersection"
        ]
        pooled_reduction.append(
            np.mean([row["intersection_width_reduction_vs_pooled_fraction"] for row in rows])
        )
        best_reduction.append(
            np.mean(
                [row["intersection_width_reduction_vs_best_single_fraction"] for row in rows]
            )
        )
    axis.bar(x - width / 2, pooled_reduction, width, label="vs pooled")
    axis.bar(x + width / 2, best_reduction, width, label="vs best single")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, short, rotation=15)
    axis.set_ylabel("Fractional width reduction")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "intersection_width_reduction.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    certification = [
        np.mean(
            [
                row["certified_anchor_fraction"]
                for row in certification_rows
                if row["source_setting"] == setting
                and row["condition"] == "confounded"
                and row["method"] == "intersection"
            ]
        )
        for setting in settings
    ]
    axis.bar(x, certification)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x, short, rotation=15)
    axis.set_ylabel("Certified-anchor fraction")
    fig.tight_layout()
    fig.savefig(output / "action_certification_rate.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    regret_methods = (
        "intersection_lower",
        "pooled_equal_lower",
        "best_single_lower",
        "observational_pooled",
    )
    for offset, method in enumerate(regret_methods):
        values = [
            np.mean(
                [
                    row["mean_do_regret"]
                    for row in regret_rows
                    if row["source_setting"] == setting
                    and row["condition"] == "confounded"
                    and row["method"] == method
                ]
            )
            for setting in settings
        ]
        local_width = 0.18
        axis.bar(x + (offset - 1.5) * local_width, values, local_width, label=method)
    axis.set_xticks(x, short, rotation=15)
    axis.set_ylabel("Mean true do-regret")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "robust_decision_regret.png", dpi=180)
    plt.close(fig)


def _report(summary: Mapping[str, Any]) -> str:
    answers = summary["answers"]
    lines = [
        "# Phase 8F-Q — 多来源因果奖励界交集快速审计",
        "",
        "本实验是受控 population 下的精确 NumPy 诊断。全局奖励支持由 canonical oracle "
        "outcomes 构造，是有利且乐观的诊断设置，不能假定可部署算法已知该范围。",
        "",
        "## 1. 同方向来源的交集收紧",
        "",
        "| Source setting | 相对 equal pooled | 相对 best single |",
        "|---|---:|---:|",
    ]
    for setting, row in answers["question_1_same_direction_tightening"].items():
        lines.append(
            f"| {setting} | {100 * row['mean_reduction_vs_equal_pooled_fraction']:.3f}% "
            f"| {100 * row['mean_reduction_vs_best_single_fraction']:.3f}% |"
        )
    question2 = answers["question_2_diverse_vs_redundant"]
    question3 = answers["question_3_intersection_certification"]
    question4 = answers["question_4_robust_action_regret"]
    lines.extend(
        [
            "",
            "## 2. 多样来源与冗余来源",
            "",
            "配对场景中 redundant-minus-diverse 的平均交集宽度为 "
            f"{question2['mean_redundant_minus_diverse_intersection_width']:.8g}；"
            f"diverse 在 {question2['diverse_tighter_scenario_count']}/"
            f"{question2['paired_scenario_count']} 个场景更窄。M5 diverse 与具有相同两个"
            f"极端来源的 M2 diverse 交集是否一致："
            f"{question2['M5_and_M2_same_extrema_intersections_equal']}。",
            "",
            "## 3. 正确动作认证",
            "",
            "confounded 场景的平均 certified-anchor fraction 为 "
            f"{question3['mean_certified_anchor_fraction_confounded']:.6f}；"
            f"false certification 共 {question3['total_false_certifications']} 个。",
            "",
            "## 4. Lower-bound robust action 的真实 do-regret",
            "",
            "平均 intersection-minus-pooled do-regret 为 "
            f"{question4['mean_intersection_minus_pooled_regret']:.8g}，"
            "intersection-minus-best-single do-regret 为 "
            f"{question4['mean_intersection_minus_best_single_regret']:.8g}。交集分别在 "
            f"{question4['intersection_lower_regret_than_pooled_scenario_count']} 和 "
            f"{question4['intersection_lower_regret_than_best_single_scenario_count']} / "
            f"{question4['compared_scenario_count']} 个配对场景降低 regret。",
            "",
            "`bound_metrics.csv` 中类似势函数的量仅称为 **one-step upper-value proxy**，"
            "不是 AAMAS potential 或 Bellman potential。",
            "",
            "本报告不自动作出论文层面的 Go/No-Go 宣告。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase8f_quick_bound_intersection(
    phase8a_root: Path,
    phase8anc_root: Path,
    phase8eq_root: Path,
    output_root: Path,
    *,
    kappas: Sequence[float],
    lambda_values: Sequence[float],
    source_settings: Sequence[str],
) -> dict[str, Any]:
    kappas_tuple = tuple(map(float, kappas))
    lambdas_tuple = tuple(map(float, lambda_values))
    settings_tuple = tuple(source_settings)
    if (
        kappas_tuple != FIXED_KAPPAS
        or lambdas_tuple != FIXED_LAMBDAS
        or settings_tuple != FIXED_SOURCE_SETTINGS
    ):
        raise Phase8FQuickBoundError("fixed Phase 8F-Q configuration was changed")

    inputs = resolve_phase8f_inputs(
        phase8a_root, phase8anc_root, phase8eq_root, kappas_tuple
    )
    before_hashes = _input_hashes(inputs["required_paths"])
    shared = _load_npz(inputs["phase8eq"] / "shared_arrays.npz")
    anchor_ids = np.asarray(shared.get("anchor_id"), dtype=np.int64)
    actions = np.asarray(shared.get("commanded_action"), dtype=np.float64)
    if (
        anchor_ids.shape != (NUM_ANCHORS,)
        or len(np.unique(anchor_ids)) != NUM_ANCHORS
        or actions.shape != (NUM_ANCHORS, 3, 3)
        or not np.all(np.isfinite(actions))
    ):
        raise Phase8FQuickBoundError("the complete 2048-anchor three-action universe is absent")

    probability_tables = source_probability_tables()
    bound_rows: list[dict[str, Any]] = []
    certification_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {"anchor_id": anchor_ids}
    checks_accumulator: dict[str, list[bool]] = {
        "support": [],
        "single_coverage": [],
        "nonempty": [],
        "intersection_coverage": [],
        "intersection_not_wider": [],
        "redundant_unchanged": [],
        "duplicate_unchanged": [],
        "pooled_joint": [],
        "false_certification": [],
        "independent_control": [],
        "finite": [],
    }

    for kappa in kappas_tuple:
        raw_path = inputs["phase8a"] / kappa_name(kappa) / "do_oracle_raw.npz"
        original = canonical_reward_branches(_load_npz(raw_path), anchor_ids)
        for dose in lambdas_tuple:
            canonical = augmented_canonical_rewards(original, dose)
            reward_min = float(np.min(canonical))
            reward_max = float(np.max(canonical))
            do_reward = canonical.mean(axis=2)
            checks_accumulator["support"].append(
                bool(
                    np.all(canonical >= reward_min - NUMERICAL_TOLERANCE)
                    and np.all(canonical <= reward_max + NUMERICAL_TOLERANCE)
                )
            )
            for setting in settings_tuple:
                behavior = probability_tables[setting]
                source_count = len(behavior)
                source_weights = np.full(source_count, 1.0 / source_count)
                for condition in CONDITIONS:
                    labels = _labels(kappa, dose, setting, condition)
                    observational = observational_population(behavior, canonical, condition)
                    lower, upper = natural_reward_bounds(
                        observational, reward_min, reward_max
                    )
                    cap_lower, cap_upper = interval_intersection(lower, upper)
                    best_lower, best_upper, best_index = best_single_interval(lower, upper)
                    pooled_equal = pool_observational_mass(observational, source_weights)
                    pooled_proportion = pool_observational_mass(observational, source_weights)
                    pooled_equal_lower, pooled_equal_upper = natural_reward_bounds(
                        pooled_equal, reward_min, reward_max
                    )
                    pooled_prop_lower, pooled_prop_upper = natural_reward_bounds(
                        pooled_proportion, reward_min, reward_max
                    )
                    pooled_equal_lower, pooled_equal_upper = (
                        pooled_equal_lower[0], pooled_equal_upper[0]
                    )
                    pooled_prop_lower, pooled_prop_upper = (
                        pooled_prop_lower[0], pooled_prop_upper[0]
                    )
                    direct_pi, direct_mass = direct_pooled_joint_mass(
                        behavior, canonical, condition, source_weights
                    )

                    single_coverage = [
                        _coverage(lower[index], upper[index], do_reward)
                        for index in range(source_count)
                    ]
                    checks_accumulator["single_coverage"].append(
                        all(np.isclose(value, 1.0) for value in single_coverage)
                    )
                    checks_accumulator["nonempty"].append(
                        bool(np.all(cap_lower <= cap_upper + NUMERICAL_TOLERANCE))
                    )
                    checks_accumulator["intersection_coverage"].append(
                        np.isclose(_coverage(cap_lower, cap_upper, do_reward), 1.0)
                    )
                    checks_accumulator["intersection_not_wider"].append(
                        bool(
                            np.all(
                                cap_upper - cap_lower
                                <= upper - lower + NUMERICAL_TOLERANCE
                            )
                        )
                    )
                    if setting == "M5_redundant":
                        checks_accumulator["redundant_unchanged"].append(
                            bool(
                                np.allclose(cap_lower, lower[0], atol=NUMERICAL_TOLERANCE, rtol=0)
                                and np.allclose(
                                    cap_upper, upper[0], atol=NUMERICAL_TOLERANCE, rtol=0
                                )
                            )
                        )
                    duplicate_lower = np.concatenate((lower, lower[:1]), axis=0)
                    duplicate_upper = np.concatenate((upper, upper[:1]), axis=0)
                    duplicate_cap = interval_intersection(duplicate_lower, duplicate_upper)
                    checks_accumulator["duplicate_unchanged"].append(
                        bool(
                            np.allclose(
                                duplicate_cap[0], cap_lower, atol=NUMERICAL_TOLERANCE, rtol=0
                            )
                            and np.allclose(
                                duplicate_cap[1], cap_upper, atol=NUMERICAL_TOLERANCE, rtol=0
                            )
                        )
                    )
                    checks_accumulator["pooled_joint"].append(
                        bool(
                            np.allclose(
                                pooled_equal["pi"][0], direct_pi, atol=NUMERICAL_TOLERANCE, rtol=0
                            )
                            and np.allclose(
                                pooled_equal["reward_mass"][0],
                                direct_mass,
                                atol=NUMERICAL_TOLERANCE,
                                rtol=0,
                            )
                        )
                    )
                    if condition == "independent_latents":
                        checks_accumulator["independent_control"].append(
                            bool(
                                np.allclose(
                                    observational["mu"][:, :, :][
                                        np.broadcast_to(
                                            observational["supported"][:, None, :],
                                            observational["mu"].shape,
                                        )
                                    ],
                                    np.broadcast_to(
                                        do_reward[None, :, :], observational["mu"].shape
                                    )[
                                        np.broadcast_to(
                                            observational["supported"][:, None, :],
                                            observational["mu"].shape,
                                        )
                                    ],
                                    atol=NUMERICAL_TOLERANCE,
                                    rtol=0,
                                )
                            )
                        )

                    pooled_width = float(
                        np.mean(pooled_equal_upper - pooled_equal_lower)
                    )
                    best_width = float(np.mean(best_upper - best_lower))
                    for source_index in range(source_count):
                        bound_rows.append(
                            _bound_row(
                                labels,
                                "single_source",
                                lower[source_index],
                                upper[source_index],
                                do_reward,
                                source_index=source_index,
                                reward_min=reward_min,
                                reward_max=reward_max,
                            )
                        )
                    bound_rows.extend(
                        (
                            _bound_row(
                                labels,
                                "best_single_width",
                                best_lower,
                                best_upper,
                                do_reward,
                                reward_min=reward_min,
                                reward_max=reward_max,
                            ),
                            _bound_row(
                                labels,
                                "pooled_equal",
                                pooled_equal_lower,
                                pooled_equal_upper,
                                do_reward,
                                reward_min=reward_min,
                                reward_max=reward_max,
                            ),
                            _bound_row(
                                labels,
                                "pooled_source_proportion",
                                pooled_prop_lower,
                                pooled_prop_upper,
                                do_reward,
                                reward_min=reward_min,
                                reward_max=reward_max,
                            ),
                            _bound_row(
                                labels,
                                "intersection",
                                cap_lower,
                                cap_upper,
                                do_reward,
                                reward_min=reward_min,
                                reward_max=reward_max,
                                pooled_width=pooled_width,
                                best_width=best_width,
                            ),
                        )
                    )

                    for method, lo, hi in (
                        ("intersection", cap_lower, cap_upper),
                        ("pooled_equal", pooled_equal_lower, pooled_equal_upper),
                        ("best_single_width", best_lower, best_upper),
                    ):
                        row, _ = _certification_row(labels, method, lo, hi, do_reward)
                        certification_rows.append(row)
                        checks_accumulator["false_certification"].append(
                            row["false_certification_count"] == 0
                        )

                    globally_best_source = int(
                        np.argmin(np.mean(upper - lower, axis=(1, 2)))
                    )
                    pooled_mu = pooled_equal["mu"][0]
                    supported = pooled_equal["supported"][0]
                    observational_score = np.where(
                        supported[None, :], pooled_mu, -np.inf
                    )
                    decision_definitions = {
                        "intersection_lower": np.argmax(cap_lower, axis=1),
                        "pooled_equal_lower": np.argmax(pooled_equal_lower, axis=1),
                        "pooled_source_proportion_lower": np.argmax(
                            pooled_prop_lower, axis=1
                        ),
                        "best_single_lower": np.argmax(
                            lower[globally_best_source], axis=1
                        ),
                        "observational_pooled": np.argmax(observational_score, axis=1),
                        "do_oracle": np.argmax(do_reward, axis=1),
                    }
                    for method, selected_action in decision_definitions.items():
                        row, _ = _regret_row(
                            labels, method, do_reward, selected_action
                        )
                        row["best_single_source_index"] = (
                            globally_best_source if method == "best_single_lower" else ""
                        )
                        regret_rows.append(row)

                    token = _scenario_token(kappa, dose, setting, condition)
                    saved_arrays[f"{token}__do_reward"] = do_reward
                    saved_arrays[f"{token}__source_lower"] = lower
                    saved_arrays[f"{token}__source_upper"] = upper
                    saved_arrays[f"{token}__intersection_lower"] = cap_lower
                    saved_arrays[f"{token}__intersection_upper"] = cap_upper
                    saved_arrays[f"{token}__pooled_equal_lower"] = pooled_equal_lower
                    saved_arrays[f"{token}__pooled_equal_upper"] = pooled_equal_upper
                    saved_arrays[f"{token}__best_single_lower"] = best_lower
                    saved_arrays[f"{token}__best_single_upper"] = best_upper
                    saved_arrays[f"{token}__best_single_source_index"] = best_index.astype(
                        np.int8
                    )
                    checks_accumulator["finite"].append(
                        _finite_tree(
                            {
                                "observational": observational,
                                "lower": lower,
                                "upper": upper,
                                "cap_lower": cap_lower,
                                "cap_upper": cap_upper,
                                "do_reward": do_reward,
                            }
                        )
                    )

    after_hashes = _input_hashes(inputs["required_paths"])
    hashes_unchanged = before_hashes == after_hashes
    checks = {
        "observational_quantities_do_not_take_do_oracle_mean": True,
        "input_source_probability_tables_verified": bool(
            inputs["source_tables_verified"]
        ),
        "oracle_support_contains_all_canonical_outcomes": bool(
            all(checks_accumulator["support"])
        ),
        "every_single_source_interval_covers_do_reward": bool(
            all(checks_accumulator["single_coverage"])
        ),
        "every_intersection_nonempty": bool(all(checks_accumulator["nonempty"])),
        "every_intersection_covers_do_reward": bool(
            all(checks_accumulator["intersection_coverage"])
        ),
        "intersection_not_wider_than_components": bool(
            all(checks_accumulator["intersection_not_wider"])
        ),
        "redundant_sources_do_not_change_intersection": bool(
            checks_accumulator["redundant_unchanged"]
            and all(checks_accumulator["redundant_unchanged"])
        ),
        "duplicated_source_does_not_change_intersection": bool(
            all(checks_accumulator["duplicate_unchanged"])
        ),
        "pooled_joint_mass_aggregation_exact": bool(
            all(checks_accumulator["pooled_joint"])
        ),
        "false_action_certification_zero": bool(
            all(checks_accumulator["false_certification"])
        ),
        "independent_latents_negative_control_exact": bool(
            checks_accumulator["independent_control"]
            and all(checks_accumulator["independent_control"])
        ),
        "input_hashes_unchanged": hashes_unchanged,
        "all_arrays_and_metrics_finite": bool(
            all(checks_accumulator["finite"])
            and _finite_tree(bound_rows)
            and _finite_tree(certification_rows)
            and _finite_tree(regret_rows)
        ),
        "old_input_artifacts_unmodified": hashes_unchanged,
        "all_2048_anchors_used": len(anchor_ids) == NUM_ANCHORS,
        "oracle_support_envelope_diagnostic_only_labeled": (
            ORACLE_SUPPORT_ENVELOPE_DIAGNOSTIC_ONLY is True
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Phase8FQuickBoundError(f"hard checks failed: {failed}")

    answers = _aggregate_answer_rows(bound_rows, certification_rows, regret_rows)
    summary = {
        "stage": STAGE,
        "scenario_count": len(FIXED_KAPPAS)
        * len(FIXED_LAMBDAS)
        * len(FIXED_SOURCE_SETTINGS)
        * len(CONDITIONS),
        "anchor_count": len(anchor_ids),
        "all_hard_checks_passed": True,
        "answers": answers,
        "interpretation_boundary": (
            "controlled-population optimistic one-step diagnostic; no paper-level Go/No-Go"
        ),
    }
    if not _finite_tree(summary):
        raise Phase8FQuickBoundError("summary contains NaN or Inf")

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": STAGE,
        "artifact_schema": "phase8f_quick_bound_intersection_v1",
        "num_anchors": NUM_ANCHORS,
        "kappas": list(kappas_tuple),
        "lambda_values": list(lambdas_tuple),
        "source_settings": {
            name: source_probability_tables()[name].tolist() for name in settings_tuple
        },
        "conditions": list(CONDITIONS),
        "action_keys": list(ACTION_KEYS),
        "reward_sampling_noise": False,
        "neural_network_training": False,
        "mujoco_rollouts": False,
        "ORACLE_SUPPORT_ENVELOPE_DIAGNOSTIC_ONLY": True,
        "observational_quantity_definition": "exact joint population mass",
        "source_proportion_pooling": (
            "uniform exact source proportions from the balanced Phase 8E-Q source design"
        ),
        "phase8a_root_hard_checks_available": inputs[
            "phase8a_root_checks_available"
        ],
        "all_hard_checks_passed": True,
    }
    integrity = {
        "before": before_hashes,
        "after": after_hashes,
        "unchanged": hashes_unchanged,
        "input_count": len(before_hashes),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "input_integrity.json", integrity)
    _write_json(
        output / "hard_checks.json",
        {"all_passed": True, "checks": checks, "failed": []},
    )
    _write_csv(output / "bound_metrics.csv", bound_rows)
    _write_csv(output / "certification_metrics.csv", certification_rows)
    _write_csv(output / "regret_metrics.csv", regret_rows)
    np.savez_compressed(output / "anchor_action_bounds.npz", **saved_arrays)
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    _make_figures(output, bound_rows, certification_rows, regret_rows)

    files = [path for path in output.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    if len(files) >= 50 or total_bytes >= 100 * 1024 * 1024:
        raise Phase8FQuickBoundError(
            f"lightweight artifact budget exceeded: {len(files)} files/{total_bytes} bytes"
        )
    if {path.name for path in files} != EXPECTED_FILENAMES:
        raise Phase8FQuickBoundError("Phase 8F-Q output file set is not exact")
    return summary
