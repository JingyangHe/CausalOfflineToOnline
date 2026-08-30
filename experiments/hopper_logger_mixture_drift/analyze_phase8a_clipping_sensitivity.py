"""Read-only clipping-sensitivity audit for verified Phase 8A artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .analyze_phase8a_population_effect import (
    ACTION_INDEX,
    PRIMARY_MIXTURES,
    PopulationEffectAuditError,
    _load_json,
    _write_json,
    all_arrays_finite,
    analyze_kappa,
    descriptive,
    hash_input_files,
    input_hashes_unchanged,
    load_condition_weights,
    load_npz,
    paired_cluster_bootstrap_means,
    require_verified_phase8a_root,
    top_action_masks,
    validate_all_84_phase8a_invariants,
    verify_primary_weights_preserve_exact_groups,
)
from .controlled_loggers import ACTION_KEYS, CONDITIONS


KAPPA = 0.3
KAPPA_DIRECTORY = "kappa_0p30"
EXPECTED_REVIEW_NAME = "population_effect_review"


class ClippingSensitivityAuditError(PopulationEffectAuditError):
    """Raised when a Phase 8A-C input or scientific invariant fails."""


def require_phase8ar_root(phase8a_root: Path, phase8ar_root: Path) -> Path:
    root = Path(phase8ar_root).resolve()
    expected = Path(phase8a_root).resolve() / EXPECTED_REVIEW_NAME
    if root != expected or not root.is_dir():
        raise ClippingSensitivityAuditError(
            "--phase8ar-root must be phase8a-root/population_effect_review"
        )
    required = ("manifest.json", "hard_checks.json", "summary.json",
                "anchor_action_metrics.npz")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ClippingSensitivityAuditError(f"missing Phase 8A-R inputs: {missing}")
    hard = _load_json(root / "hard_checks.json")
    if hard.get("all_passed") is not True or not all(hard.get("checks", {}).values()):
        raise ClippingSensitivityAuditError("Phase 8A-R hard checks are not all passed")
    manifest = _load_json(root / "manifest.json")
    if tuple(manifest.get("primary_mixtures", ())) != PRIMARY_MIXTURES:
        raise ClippingSensitivityAuditError("Phase 8A-R primary mixtures are unavailable")
    return root


def _validate_output_root(review_root: Path, output_root: Path) -> Path:
    output = Path(output_root).resolve()
    try:
        relative = output.relative_to(review_root)
    except ValueError as exc:
        raise ClippingSensitivityAuditError(
            "output root must be nested under the Phase 8A-R root"
        ) from exc
    if not relative.parts or not relative.parts[0].startswith("clipping_sensitivity"):
        raise ClippingSensitivityAuditError(
            "output directory name must start with clipping_sensitivity"
        )
    return output


def required_input_paths(phase8a_root: Path, phase8ar_root: Path) -> list[Path]:
    directory = phase8a_root / KAPPA_DIRECTORY
    paths = [
        phase8a_root / "manifest.json", phase8a_root / "summary.json",
        phase8a_root / "anchors.npz", directory / "do_oracle_raw.npz",
        directory / "do_oracle_summary.npz", phase8ar_root / "manifest.json",
        phase8ar_root / "hard_checks.json", phase8ar_root / "summary.json",
        phase8ar_root / "anchor_action_metrics.npz",
    ]
    for condition in CONDITIONS:
        paths.extend((directory / f"{condition}_public.npz",
                      directory / f"{condition}_hidden_audit.npz"))
        paths.extend(directory / "weights" / condition / f"weights_{mixture}.npy"
                     for mixture in ("logger1_heavy", "logger2_heavy"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ClippingSensitivityAuditError(f"missing required inputs: {missing}")
    return sorted((path.resolve() for path in paths), key=str)


def clipping_from_preclip(
    preclip: np.ndarray, expected_applied: np.ndarray, atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinate/row clipping and excess/headroom from pre-clipping actions."""
    before = np.asarray(preclip, dtype=np.float64)
    after = np.asarray(expected_applied, dtype=np.float64)
    coordinate = ((before > 1.0 + atol) | (before < -1.0 - atol)
                  | (np.abs(before - after) > atol))
    row = np.any(coordinate, axis=-1)
    excess_linf = np.max(np.maximum.reduce((before - 1.0, -1.0 - before,
                                            np.zeros_like(before))), axis=-1)
    excess_l2 = np.linalg.norm(before - after, axis=-1)
    headroom = np.min(1.0 - np.abs(before), axis=-1)
    return coordinate, row, excess_linf, excess_l2, headroom


def reconstruct_canonical_clipping(
    raw: dict[str, np.ndarray], anchor_ids: np.ndarray, direction: np.ndarray,
    kappa: float, atol: float, rtol: float,
) -> dict[str, np.ndarray]:
    required = {"anchor_id", "action_key", "u_env", "commanded_action",
                "applied_action", "reward", "next_observation", "terminated",
                "truncated", "kappa_env"}
    if not required.issubset(raw):
        raise ClippingSensitivityAuditError("do_oracle_raw lacks clipping audit fields")
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ClippingSensitivityAuditError("manifest actuator_direction_v must be finite length 3")
    ids = np.asarray(anchor_ids, dtype=np.int64)
    lookup = {int(anchor): index for index, anchor in enumerate(ids)}
    n = len(ids)
    command = np.empty((n, 3, 2, 3), dtype=np.float64)
    saved_applied = np.empty_like(command)
    reward = np.empty((n, 3, 2), dtype=np.float64)
    next_observation = np.empty((n, 3, 2, 12), dtype=np.float64)
    terminated = np.empty((n, 3, 2), dtype=bool)
    truncated = np.empty((n, 3, 2), dtype=bool)
    seen: set[tuple[int, str, int, float]] = set()
    selected = set(lookup)
    for row in range(len(raw["anchor_id"])):
        anchor = int(raw["anchor_id"][row])
        if anchor not in selected:
            continue
        action, u_env = str(raw["action_key"][row]), int(raw["u_env"][row])
        row_kappa = float(raw["kappa_env"][row])
        key = (anchor, action, u_env, row_kappa)
        if (key in seen or action not in ACTION_INDEX or u_env not in (-1, 1)
                or not np.isclose(row_kappa, kappa, atol=atol, rtol=rtol)):
            raise ClippingSensitivityAuditError("do-oracle canonical keys are invalid or duplicated")
        seen.add(key)
        ai, aj, uk = lookup[anchor], ACTION_INDEX[action], int(u_env == 1)
        command[ai, aj, uk] = raw["commanded_action"][row]
        saved_applied[ai, aj, uk] = raw["applied_action"][row]
        reward[ai, aj, uk] = raw["reward"][row]
        next_observation[ai, aj, uk] = raw["next_observation"][row]
        terminated[ai, aj, uk] = raw["terminated"][row]
        truncated[ai, aj, uk] = raw["truncated"][row]
    if len(seen) != n * 3 * 2:
        raise ClippingSensitivityAuditError(
            f"do oracle must contain exactly {n * 3 * 2} selected canonical rows"
        )
    if not np.allclose(command[:, :, 0], command[:, :, 1], atol=atol, rtol=rtol):
        raise ClippingSensitivityAuditError("commanded action depends on u_env")
    u_values = np.asarray((-1.0, 1.0), dtype=np.float64)
    preclip = command + kappa * u_values[None, None, :, None] * vector
    expected = np.clip(preclip, -1.0, 1.0)
    if not np.allclose(saved_applied, expected, atol=atol, rtol=rtol):
        raise ClippingSensitivityAuditError("saved applied action differs from reconstructed clip")
    coordinate, row, linf, l2, headroom = clipping_from_preclip(preclip, expected, atol)
    return {
        "anchor_id": ids, "commanded_action": command, "preclip_action": preclip,
        "expected_applied_action": expected, "saved_applied_action": saved_applied,
        "coordinate_clipped": coordinate, "row_clipped": row,
        "clipped_coordinate_count": coordinate.sum(axis=-1),
        "clipping_excess_linf": linf, "clipping_excess_l2": l2,
        "preclip_headroom": headroom, "reward": reward,
        "next_observation": next_observation, "terminated": terminated,
        "truncated": truncated,
    }


def derive_clean_sets(row_clipped: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.asarray(row_clipped, dtype=bool)
    if clipped.ndim != 3 or clipped.shape[1:] != (3, 2):
        raise ValueError("row_clipped must have shape [anchors,3,2]")
    pair_unclipped = ~np.any(clipped, axis=2)
    strict_unclipped = np.all(pair_unclipped, axis=1)
    return pair_unclipped, strict_unclipped, ~strict_unclipped


def verify_canonical_invariance(
    directory: Path, canonical: dict[str, np.ndarray], anchor_ids: np.ndarray,
    direction: np.ndarray, kappa: float, atol: float, rtol: float,
) -> dict[str, bool]:
    lookup = {int(anchor): index for index, anchor in enumerate(anchor_ids)}
    logger_ok = True
    condition_reference: dict[tuple[int, str, int], tuple[Any, ...]] = {}
    for condition in CONDITIONS:
        public = load_npz(directory / f"{condition}_public.npz")
        hidden = load_npz(directory / f"{condition}_hidden_audit.npz")
        if not np.array_equal(public["row_id"], hidden["row_id"]):
            raise ClippingSensitivityAuditError("public/hidden row alignment failed")
        by_key: dict[tuple[int, str, int], list[int]] = {}
        for row, (anchor, action, u_env) in enumerate(zip(
                hidden["anchor_id"], hidden["action_key"], hidden["u_env"])):
            key = (int(anchor), str(action), int(u_env))
            if key[0] in lookup:
                by_key.setdefault(key, []).append(row)
        for key, rows in by_key.items():
            ai, aj, uk = lookup[key[0]], ACTION_INDEX[key[1]], int(key[2] == 1)
            expected_values = (
                canonical["commanded_action"][ai, aj, uk],
                canonical["expected_applied_action"][ai, aj, uk],
                canonical["reward"][ai, aj, uk],
                canonical["next_observation"][ai, aj, uk],
                canonical["terminated"][ai, aj, uk], canonical["truncated"][ai, aj, uk],
                canonical["row_clipped"][ai, aj, uk],
            )
            for row in rows:
                preclip = np.asarray(hidden["commanded_action"][row], dtype=np.float64) + (
                    kappa * key[2] * direction
                )
                reconstructed = clipping_from_preclip(
                    preclip[None, :], np.clip(preclip, -1.0, 1.0)[None, :], atol
                )[1][0]
                observed = (
                    hidden["commanded_action"][row], hidden["applied_action"][row],
                    hidden["reward"][row], public["next_observation"][row],
                    hidden["terminated"][row], hidden["truncated"][row], reconstructed,
                )
                equal = all(
                    np.allclose(left, right, atol=atol, rtol=rtol)
                    if np.asarray(left).dtype.kind not in "b" else bool(left == right)
                    for left, right in zip(observed, expected_values)
                )
                logger_ok &= bool(equal)
                if not equal:
                    raise ClippingSensitivityAuditError(
                        f"CANONICAL_CLIPPING_STATUS_NOT_UNIQUE: {condition}/{key}"
                    )
            signature = tuple(np.asarray(value).tobytes() for value in expected_values)
            previous = condition_reference.setdefault(key, signature)
            if previous != signature:
                raise ClippingSensitivityAuditError(
                    f"CANONICAL_CLIPPING_STATUS_NOT_UNIQUE: condition/{key}"
                )
    expected_keys = len(anchor_ids) * 3 * 2
    if len(condition_reference) != expected_keys:
        raise ClippingSensitivityAuditError("hidden audit lacks canonical keys")
    return {"clipping_status_is_logger_invariant": logger_ok,
            "clipping_status_is_condition_invariant": True}


def weighted_clipping_probabilities(
    directory: Path, canonical: dict[str, np.ndarray], anchor_ids: np.ndarray,
    atol: float, rtol: float,
) -> tuple[list[dict[str, Any]], bool]:
    lookup = {int(anchor): index for index, anchor in enumerate(anchor_ids)}
    rows: list[dict[str, Any]] = []
    independent_values: dict[str, list[float]] = {action: [] for action in ACTION_KEYS}
    for condition in CONDITIONS:
        public = load_npz(directory / f"{condition}_public.npz")
        hidden = load_npz(directory / f"{condition}_hidden_audit.npz")
        weights = load_condition_weights(directory, condition, hidden, atol, rtol)
        selected = np.isin(hidden["anchor_id"], anchor_ids)
        for mixture in PRIMARY_MIXTURES:
            for action in ACTION_KEYS:
                cell = selected & (hidden["action_key"].astype(str) == action)
                mass = float(weights[mixture][cell].sum())
                if mass <= 0:
                    raise ClippingSensitivityAuditError("weighted clipping cell has zero mass")
                indices = np.flatnonzero(cell)
                status = np.asarray([
                    canonical["row_clipped"][lookup[int(hidden["anchor_id"][row])],
                                                     ACTION_INDEX[action],
                                                     int(int(hidden["u_env"][row]) == 1)]
                    for row in indices
                ], dtype=np.float64)
                probability = float(weights[mixture][indices] @ status / mass)
                rows.append({"condition": condition, "mixture": mixture, "action": action,
                             "weighted_clipping_probability": probability,
                             "weighted_any_coordinate_clipped_probability": probability})
                if condition == "independent_latents":
                    independent_values[action].append(probability)
    invariant = all(np.allclose(values, values[0], atol=atol, rtol=rtol)
                    for values in independent_values.values())
    return rows, bool(invariant)


def _aggregate_vector(
    values: np.ndarray, family: str, subset: str, metric: str, action: str,
    condition: str, mixture: str, bootstrap_reps: int, seed: int,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    row: dict[str, Any] = {"family": family, "subset": subset, "metric": metric,
                           "action": action, "condition": condition, "mixture": mixture,
                           "bootstrap_unit": "anchor_id",
                           "bootstrap_repetitions": bootstrap_reps, "bootstrap_seed": seed}
    if not len(vector):
        row.update({"n_anchors": 0, "mean": None, "standard_deviation": None,
                    "median": None, "p10": None, "p25": None, "p75": None,
                    "p90": None, "maximum": None, "ci95_low": None, "ci95_high": None})
        return row
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ClippingSensitivityAuditError("metric vectors must be finite anchor vectors")
    low, high = paired_cluster_bootstrap_means([vector], bootstrap_reps, seed)
    row.update(descriptive(vector))
    row.update(ci95_low=float(low[0]), ci95_high=float(high[0]))
    return row


def action_subset_metrics(
    context: dict[str, Any], pair_unclipped: np.ndarray, bootstrap_reps: int, seed: int,
) -> list[dict[str, Any]]:
    do, observational = context["do"], context["observational"]
    primary = context["primary_metrics"]
    rows: list[dict[str, Any]] = []
    signs = np.asarray((-1.0, 0.0, 1.0))
    reward_identity = (primary["confounded"]["signed_reward_drift"]
                       - (7.0 / 9.0) * do["reward_u_effect"] * signs[None, :])
    delta_actual = (observational["confounded"]["logger1_heavy"]["delta"]
                    - observational["confounded"]["logger2_heavy"]["delta"])
    delta_identity = delta_actual - (7.0 / 9.0) * do["delta_u_effect"] * signs[None, :, None]
    for action in ACTION_KEYS:
        ai = ACTION_INDEX[action]
        vectors: list[tuple[str, str, str, np.ndarray]] = [
            ("reward_u_effect_abs", "do_oracle", "none", np.abs(do["reward_u_effect"][:, ai])),
            ("delta_u_effect_l2", "do_oracle", "none",
             np.linalg.norm(do["delta_u_effect"][:, ai], axis=1)),
            ("reward_drift", "confounded", "heavy_contrast",
             primary["confounded"]["absolute_reward_drift"][:, ai]),
            ("reward_drift", "independent_latents", "heavy_contrast",
             primary["independent_latents"]["absolute_reward_drift"][:, ai]),
            ("delta_drift", "confounded", "heavy_contrast",
             primary["confounded"]["delta_drift_heavy_contrast"][:, ai]),
            ("delta_drift", "independent_latents", "heavy_contrast",
             primary["independent_latents"]["delta_drift_heavy_contrast"][:, ai]),
            ("reward_7_over_9_identity_residual_abs", "confounded", "heavy_contrast",
             np.abs(reward_identity[:, ai])),
            ("delta_7_over_9_identity_residual_l2", "confounded", "heavy_contrast",
             np.linalg.norm(delta_identity[:, ai], axis=1)),
        ]
        for mixture in PRIMARY_MIXTURES:
            vectors.extend((
                ("reward_do_error_abs", "confounded", mixture, np.abs(
                    observational["confounded"][mixture]["reward"][:, ai]
                    - do["mean_reward"][:, ai])),
                ("delta_do_error_l2", "confounded", mixture, np.linalg.norm(
                    observational["confounded"][mixture]["delta"][:, ai]
                    - do["mean_delta"][:, ai], axis=1)),
            ))
        subsets = {"all": np.ones(len(pair_unclipped), dtype=bool),
                   "action_pair_unclipped": pair_unclipped[:, ai],
                   "action_pair_any_clipping": ~pair_unclipped[:, ai]}
        for subset, mask in subsets.items():
            for metric, condition, mixture, values in vectors:
                rows.append(_aggregate_vector(
                    values[mask], "action_specific", subset, metric, action,
                    condition, mixture, bootstrap_reps,
                    seed + ai * 1000 + list(subsets).index(subset),
                ))
    return rows


def decision_subset_metrics(
    context: dict[str, Any], strict_unclipped: np.ndarray, bootstrap_reps: int,
    seed: int, atol: float, rtol: float,
) -> list[dict[str, Any]]:
    do, observational = context["do"], context["observational"]
    do_gap = np.asarray(context["do_action_gap"])
    max_drift = np.asarray(context["max_action_mixture_drift"])
    heavy1 = top_action_masks(observational["confounded"]["logger1_heavy"]["reward"], atol, rtol)
    heavy2 = top_action_masks(observational["confounded"]["logger2_heavy"]["reward"], atol, rtol)
    do_mask = top_action_masks(do["mean_reward"], atol, rtol)
    subsets = {"all": np.ones(len(strict_unclipped), dtype=bool),
               "strict_anchor_unclipped": strict_unclipped,
               "anchor_any_clipping": ~strict_unclipped}
    rows: list[dict[str, Any]] = []
    for subset_index, (subset, mask) in enumerate(subsets.items()):
        vectors: list[tuple[str, str, np.ndarray]] = [
            ("do_action_gap", "none", do_gap),
            ("max_primary_mixture_reward_drift", "primary", max_drift),
            ("fraction_drift_greater_than_action_gap", "primary", (max_drift > do_gap).astype(float)),
            ("heavy_top_set_disagreement", "logger1_heavy_vs_logger2_heavy",
             (heavy1 != heavy2).astype(float)),
            ("heavy_strict_flip", "logger1_heavy_vs_logger2_heavy",
             ((heavy1 & heavy2) == 0).astype(float)),
            ("heavy_top_sets_disjoint", "logger1_heavy_vs_logger2_heavy",
             ((heavy1 & heavy2) == 0).astype(float)),
        ]
        for mixture in PRIMARY_MIXTURES:
            current = top_action_masks(observational["confounded"][mixture]["reward"], atol, rtol)
            vectors.extend((
                ("top_set_disagreement_with_do", mixture, (current != do_mask).astype(float)),
                ("strict_top_set_disagreement_with_do", mixture,
                 ((current & do_mask) == 0).astype(float)),
            ))
            for action in ACTION_KEYS:
                bit = 1 << ACTION_INDEX[action]
                vectors.append((f"fraction_{action}_in_top_set", mixture,
                                ((current & bit) != 0).astype(float)))
        for metric, mixture, values in vectors:
            rows.append(_aggregate_vector(
                values[mask], "decision", subset, metric, "all", "confounded",
                mixture, bootstrap_reps, seed + 10000 + subset_index,
            ))
        ratio = {"family": "decision", "subset": subset,
                 "metric": "drift_over_action_gap_ratio_of_means", "action": "all",
                 "condition": "confounded", "mixture": "primary", "n_anchors": int(mask.sum()),
                 "standard_deviation": None, "median": None, "p10": None, "p25": None,
                 "p75": None, "p90": None, "maximum": None, "bootstrap_unit": "anchor_id",
                 "bootstrap_repetitions": bootstrap_reps,
                 "bootstrap_seed": seed + 20000 + subset_index}
        if mask.any() and float(np.mean(do_gap[mask])) > 0:
            ratio["mean"] = float(np.mean(max_drift[mask]) / np.mean(do_gap[mask]))
            matrix = np.column_stack((max_drift[mask], do_gap[mask]))
            rng = np.random.default_rng(seed + 20000 + subset_index)
            n = len(matrix)
            counts = rng.multinomial(n, np.full(n, 1 / n), size=bootstrap_reps)
            means = counts @ matrix / n
            valid = means[:, 1] > 0
            estimates = means[valid, 0] / means[valid, 1]
            if not len(estimates):
                ratio.update(mean=None, ci95_low=None, ci95_high=None)
            else:
                ratio["ci95_low"], ratio["ci95_high"] = map(
                    float, np.quantile(estimates, (0.025, 0.975)))
            ratio["bootstrap_valid_repetitions"] = int(valid.sum())
        else:
            ratio.update(mean=None, ci95_low=None, ci95_high=None)
            ratio["bootstrap_valid_repetitions"] = 0
        rows.append(ratio)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: "" if value is None else value for key, value in row.items()}
                          for row in rows])


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    result = descriptive(array)
    return result


def prevalence_summary(canonical: dict[str, np.ndarray], pair: np.ndarray,
                       strict: np.ndarray) -> dict[str, Any]:
    row, coordinate = canonical["row_clipped"], canonical["coordinate_clipped"]
    result: dict[str, Any] = {
        "canonical_execution_count": int(row.size),
        "overall_clipped_execution_fraction": float(np.mean(row)),
        "clipped_coordinate_fraction": float(np.mean(coordinate)),
        "clipping_excess_linf": _distribution(canonical["clipping_excess_linf"]),
        "clipping_excess_l2": _distribution(canonical["clipping_excess_l2"]),
        "by_action": {}, "by_u_env": {}, "by_coordinate": {},
        "action_pair_unclipped": {},
        "strict_anchor_unclipped_count": int(strict.sum()),
        "strict_anchor_unclipped_fraction": float(np.mean(strict)),
        "anchor_any_clipping_count": int((~strict).sum()),
        "anchor_any_clipping_fraction": float(np.mean(~strict)),
    }
    for action in ACTION_KEYS:
        ai = ACTION_INDEX[action]
        result["by_action"][action] = {
            "clipped_execution_fraction": float(np.mean(row[:, ai])),
            "clipped_coordinate_fraction": {
                str(j): float(np.mean(coordinate[:, ai, :, j])) for j in range(3)
            },
            "preclip_headroom": _distribution(canonical["preclip_headroom"][:, ai]),
        }
        result["action_pair_unclipped"][action] = {
            "count": int(pair[:, ai].sum()), "fraction": float(np.mean(pair[:, ai]))}
    for uk, u_env in enumerate((-1, 1)):
        result["by_u_env"][str(u_env)] = {
            "clipped_execution_fraction": float(np.mean(row[:, :, uk])),
            "preclip_headroom": _distribution(canonical["preclip_headroom"][:, :, uk]),
        }
    for coordinate_index in range(3):
        result["by_coordinate"][str(coordinate_index)] = {
            "clipped_fraction": float(np.mean(coordinate[..., coordinate_index]))}
    result["preclip_headroom"] = {
        "all": _distribution(canonical["preclip_headroom"]),
        "strict_anchor_unclipped": _distribution(canonical["preclip_headroom"][strict])
        if strict.any() else None,
        "anchor_any_clipping": _distribution(canonical["preclip_headroom"][~strict])
        if (~strict).any() else None,
    }
    return result


def _full_crosscheck(
    context: dict[str, Any], review_arrays: dict[str, np.ndarray], full: bool,
    atol: float, rtol: float,
) -> dict[str, Any]:
    if not full:
        return {"passed": True, "not_comparable": "smoke uses sorted anchor prefix"}
    mapping = {
        "reward_u_effect": context["do"]["reward_u_effect"],
        "delta_u_effect": context["do"]["delta_u_effect"],
        "confounded_absolute_reward_drift": context["primary_metrics"]["confounded"]["absolute_reward_drift"],
        "confounded_delta_drift_heavy_contrast": context["primary_metrics"]["confounded"]["delta_drift_heavy_contrast"],
        "max_action_mixture_drift": context["max_action_mixture_drift"],
        "do_action_gap": context["do_action_gap"],
        "confounded_heavy_strict_flip": context["ranking"]["confounded"]["heavy_disjoint"],
    }
    maximum = 0.0
    for name, values in mapping.items():
        key = f"{KAPPA_DIRECTORY}__{name}"
        if key not in review_arrays:
            raise ClippingSensitivityAuditError(f"Phase 8A-R anchor metrics missing {key}")
        maximum = max(maximum, float(np.max(np.abs(
            np.asarray(review_arrays[key], dtype=float) - np.asarray(values, dtype=float)))))
        if not np.allclose(review_arrays[key], values, atol=atol, rtol=rtol):
            raise ClippingSensitivityAuditError(f"full-sample metric differs from Phase 8A-R: {name}")
    return {"passed": True, "maximum_absolute_difference": maximum}


def _flatten_canonical(canonical: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(canonical["anchor_id"])
    result = {
        "anchor_id": np.repeat(canonical["anchor_id"], 6),
        "action_key": np.tile(np.repeat(np.asarray(ACTION_KEYS), 2), n),
        "u_env": np.tile(np.asarray((-1, 1)), n * 3),
    }
    for name, values in canonical.items():
        if name == "anchor_id":
            continue
        array = np.asarray(values)
        if array.shape[:3] == (n, 3, 2):
            result[name] = array.reshape((n * 6,) + array.shape[3:])
    return result


def _select_mean(rows: list[dict[str, Any]], subset: str, metric: str,
                 action: str = "all", mixture: str | None = None,
                 condition: str | None = "confounded") -> float | None:
    matches = [row for row in rows if row["subset"] == subset and row["metric"] == metric
               and row["action"] == action and (mixture is None or row["mixture"] == mixture)
               and (condition is None or row["condition"] == condition)]
    if len(matches) != 1:
        raise ClippingSensitivityAuditError(f"aggregate selection not unique: {subset}/{metric}")
    return matches[0]["mean"]


def _decision_main_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = (("all", "all_anchors"),
               ("strict_anchor_unclipped", "strict_unclipped_anchors"),
               ("anchor_any_clipping", "any_clipping_anchors"))
    definitions = (
        ("n_anchors", None, None),
        ("mean_do_action_gap", "do_action_gap", None),
        ("mean_max_mixture_drift", "max_primary_mixture_reward_drift", None),
        ("drift_over_action_gap", "drift_over_action_gap_ratio_of_means", None),
        ("drift_greater_than_action_gap_fraction",
         "fraction_drift_greater_than_action_gap", None),
        ("logger1_vs_logger2_disagreement", "heavy_top_set_disagreement", None),
        ("strict_flip_fraction", "heavy_strict_flip", None),
        ("logger1_minus_top_fraction", "fraction_minus_in_top_set", "logger1_heavy"),
        ("logger2_plus_top_fraction", "fraction_plus_in_top_set", "logger2_heavy"),
    )
    output = []
    for label, metric, mixture in definitions:
        row: dict[str, Any] = {"metric": label}
        for subset, column in columns:
            matches = [item for item in rows if item["subset"] == subset]
            if metric is None:
                row[column] = matches[0]["n_anchors"] if matches else 0
            else:
                row[column] = _select_mean(rows, subset, metric, mixture=mixture)
        output.append(row)
    return output


def _make_figures(output: Path, prevalence: dict[str, Any], action_rows: list[dict[str, Any]],
                  decision_rows: list[dict[str, Any]], canonical: dict[str, np.ndarray]) -> None:
    def save_bar(path: str, labels: list[str], values: list[float], ylabel: str) -> None:
        figure, axes = plt.subplots(figsize=(6.4, 4.2))
        axes.bar(labels, values)
        axes.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(output / path, dpi=160)
        plt.close(figure)

    save_bar("clipping_rate_by_action.png", list(ACTION_KEYS),
             [prevalence["by_action"][a]["clipped_execution_fraction"] for a in ACTION_KEYS],
             "Clipped execution fraction")
    labels, values = [], []
    coord = canonical["coordinate_clipped"]
    for action in ACTION_KEYS:
        for j in range(3):
            labels.append(f"{action}\n{j}")
            values.append(float(np.mean(coord[:, ACTION_INDEX[action], :, j])))
    save_bar("clipping_rate_by_action_coordinate.png", labels, values,
             "Clipped coordinate fraction")
    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    axes.hist(canonical["preclip_headroom"].reshape(-1), bins=40)
    axes.set_xlabel("Preclip headroom")
    axes.set_ylabel("Canonical executions")
    figure.tight_layout()
    figure.savefig(output / "preclip_headroom_distribution.png", dpi=160)
    plt.close(figure)

    for filename, metric, ylabel in (
        ("reward_drift_all_vs_unclipped.png", "reward_drift", "Mean reward drift"),
        ("delta_drift_all_vs_unclipped.png", "delta_drift", "Mean delta drift L2"),
    ):
        labels, vals = [], []
        for action in ACTION_KEYS:
            for subset, short in (("all", "all"),
                                  ("action_pair_unclipped", "clean"),
                                  ("action_pair_any_clipping", "clipped")):
                labels.append(f"{action}\n{short}")
                vals.append(_select_mean(action_rows, subset, metric, action=action) or 0.0)
        save_bar(filename, labels, vals, ylabel)
    save_bar("strict_flip_all_vs_unclipped.png", ["all", "strict clean", "any clipped"],
             [(_select_mean(decision_rows, subset, "heavy_strict_flip") or 0.0)
              for subset in ("all", "strict_anchor_unclipped", "anchor_any_clipping")],
             "Strict flip fraction")
    save_bar("drift_relative_to_action_gap_all_vs_unclipped.png",
             ["all", "strict clean", "any clipped"],
             [(_select_mean(decision_rows, subset, "drift_over_action_gap_ratio_of_means") or 0.0)
              for subset in ("all", "strict_anchor_unclipped", "anchor_any_clipping")],
             "Mean drift / mean do action gap")


def _write_reports(output: Path, summary: dict[str, Any]) -> None:
    p = summary["clipping_prevalence"]
    report = f"""# Phase 8A-C — Applied-Action Clipping Sensitivity Audit

This is a read-only, descriptive audit of the verified Phase 8A artifact at kappa=0.3.

## Scope and validity

All clipping labels were reconstructed from preclip actions and checked against saved applied
actions. The canonical execution clipping fraction is {p['overall_clipped_execution_fraction']:.6g}.
There are {p['strict_anchor_unclipped_count']} strict-unclipped anchors out of
{summary['analyzed_anchor_count']} analyzed anchors.

The tables report facts only. No effect-retention threshold was used and no paper-success verdict
was selected. Clean and clipped anchors may occupy different state regions, so their descriptive
difference is not a pure causal effect of clipping. Results use one behavior checkpoint seed and do
not establish cross-policy-seed population significance.

## Reading the bundle

`decision_metrics.csv` compares all anchors, strict-unclipped anchors, and anchors with any
clipping. `action_specific_metrics.csv` compares all, pair-unclipped, and pair-clipped samples.
`aggregate_tables.csv` is their common source. Exact prevalence and weighted clipping probabilities
are in `summary.json`; canonical and anchor masks are in the NPZ tables.

## Mechanism boundary

The evidence can establish whether drift and ranking flips remain observable on the strict clean
subset. It cannot establish that clipping has no influence, that clipping is the unique cause, or
that full-versus-clean differences are causal. Manual scientific review is required.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (output / "stats-appendix.md").write_text(
        "# Statistical appendix\n\nAnchor ID is the resampling unit. Intervals are percentile "
        "95% cluster-bootstrap intervals. Canonical rates are finite-population descriptions.\n",
        encoding="utf-8")
    (output / "figure-catalog.md").write_text(
        "# Figure catalog\n\nEach PNG is a separate matplotlib figure. The first three describe "
        "clipping prevalence/headroom; the remaining four compare full and clean subsets.\n",
        encoding="utf-8")


def _git_commit(repository: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_audit(
    phase8a_root: Path, phase8ar_root: Path, output_root: Path, *, kappa: float = KAPPA,
    bootstrap_reps: int = 2000, seed: int = 0, max_anchors: int | None = None,
    expected_anchor_count: int = 2048,
) -> dict[str, Any]:
    if not np.isclose(kappa, KAPPA, atol=0.0, rtol=0.0):
        raise ClippingSensitivityAuditError("Phase 8A-C audits only kappa=0.3")
    if bootstrap_reps <= 0 or (max_anchors is not None and max_anchors <= 0):
        raise ValueError("bootstrap_reps and max_anchors must be positive")
    root = require_verified_phase8a_root(phase8a_root)
    review = require_phase8ar_root(root, phase8ar_root)
    output = _validate_output_root(review, output_root)
    manifest = _load_json(root / "manifest.json")
    phase8a_summary = _load_json(root / "summary.json")
    validate_all_84_phase8a_invariants(phase8a_summary)
    anchors = load_npz(root / "anchors.npz")
    all_ids = np.asarray(anchors.get("anchor_id", ()), dtype=np.int64)
    if not np.array_equal(all_ids, np.arange(expected_anchor_count)):
        raise ClippingSensitivityAuditError(
            f"verified input must contain anchors 0..{expected_anchor_count - 1}"
        )
    review_manifest = _load_json(review / "manifest.json")
    if (int(review_manifest.get("expected_anchor_count", -1)) != expected_anchor_count
            or int(review_manifest.get("analyzed_anchor_count", -1)) != expected_anchor_count):
        raise ClippingSensitivityAuditError(
            "Phase 8A-R input must be the completed full-anchor review"
        )
    if not (root / KAPPA_DIRECTORY).is_dir() or KAPPA not in tuple(manifest.get("kappas", ())):
        raise ClippingSensitivityAuditError("kappa=0.3 input is incomplete")
    direction = np.asarray(manifest.get("actuator_direction_v", ()), dtype=np.float64)
    atol = float(manifest.get("numerical_tolerance", {}).get("atol", 1e-7))
    rtol = float(manifest.get("numerical_tolerance", {}).get("rtol", 1e-7))
    paths = required_input_paths(root, review)
    hashes_before = hash_input_files(paths)

    selected_count = min(max_anchors or len(all_ids), len(all_ids))
    selected_ids = np.sort(all_ids)[:selected_count]
    full = selected_count == len(all_ids)
    directory = root / KAPPA_DIRECTORY
    raw = load_npz(directory / "do_oracle_raw.npz")
    canonical = reconstruct_canonical_clipping(
        raw, selected_ids, direction, kappa, atol, rtol
    )
    pair_unclipped, strict_unclipped, anchor_any = derive_clean_sets(
        canonical["row_clipped"]
    )
    invariance = verify_canonical_invariance(
        directory, canonical, selected_ids, direction, kappa, atol, rtol
    )
    weighted_rows, independent_invariant = weighted_clipping_probabilities(
        directory, canonical, selected_ids, atol, rtol
    )
    context, _, arrays, phase8ar_checks = analyze_kappa(
        root, kappa, anchors, selected_ids, atol, rtol
    )
    if strict_unclipped.any():
        for condition in CONDITIONS:
            public = load_npz(directory / f"{condition}_public.npz")
            hidden = load_npz(directory / f"{condition}_hidden_audit.npz")
            weights = load_condition_weights(directory, condition, hidden, atol, rtol)
            verify_primary_weights_preserve_exact_groups(
                public, hidden, weights, selected_ids[strict_unclipped], atol, rtol,
                context["do"]["commanded_action"][strict_unclipped],
            )
    action_rows = action_subset_metrics(context, pair_unclipped, bootstrap_reps, seed)
    decision_rows = decision_subset_metrics(
        context, strict_unclipped, bootstrap_reps, seed, atol, rtol
    )
    review_arrays = load_npz(review / "anchor_action_metrics.npz")
    crosscheck = _full_crosscheck(context, review_arrays, full, atol, rtol)
    prevalence = prevalence_summary(canonical, pair_unclipped, strict_unclipped)

    base = ACTION_INDEX["base"]
    base_clean = pair_unclipped[:, base]
    reward_residual = arrays["reward_identity_residual"]
    delta_residual = arrays["delta_identity_residual"]
    hard_checks = {
        "verified_phase8a_input_required": True,
        "phase8ar_input_required_and_all_checks_passed": True,
        "kappa_0p30_present": True,
        "all_expected_anchors_present": len(all_ids) == expected_anchor_count,
        "action_key_mapping_unique": True,
        "do_oracle_canonical_keys_unique": True,
        "preclip_reconstruction_complete": True,
        "expected_applied_matches_saved_applied": True,
        **invariance,
        "primary_state_action_mass_preserved_after_strict_filter": True,
        "independent_weighted_clipping_is_mixture_invariant": independent_invariant,
        "base_action_drift_zero_on_clean_subset": bool(
            not base_clean.any() or np.allclose(
                arrays["confounded_absolute_reward_drift"][base_clean, base], 0,
                atol=atol, rtol=rtol)),
        "reward_mechanism_identity_on_clean_subset": bool(all(
            not pair_unclipped[:, ai].any() or np.allclose(
                reward_residual[pair_unclipped[:, ai], ai], 0, atol=atol, rtol=rtol)
            for ai in range(3))),
        "delta_mechanism_identity_on_clean_subset": bool(all(
            not pair_unclipped[:, ai].any() or np.allclose(
                delta_residual[pair_unclipped[:, ai], ai], 0, atol=atol, rtol=rtol)
            for ai in range(3))),
        "decision_metrics_use_strict_anchor_subset": all(
            row["n_anchors"] == int(mask.sum())
            for subset, mask in (("all", np.ones(selected_count, bool)),
                                 ("strict_anchor_unclipped", strict_unclipped),
                                 ("anchor_any_clipping", anchor_any))
            for row in decision_rows if row["subset"] == subset),
        "full_sample_metrics_match_phase8ar": bool(crosscheck["passed"]),
        "all_recomputed_arrays_finite": all_arrays_finite(canonical, arrays),
        "metrics_use_anchor_level_units": all(
            row["bootstrap_unit"] == "anchor_id" for row in action_rows + decision_rows),
        "aggregate_outputs_have_no_nan_inf": all(
            value is None or not isinstance(value, (float, np.floating)) or np.isfinite(value)
            for row in action_rows + decision_rows for value in row.values()),
        **{f"phase8ar_recompute:{name}": bool(value)
           for name, value in phase8ar_checks.items()},
    }
    failed = [name for name, value in hard_checks.items() if not value]
    if failed:
        raise ClippingSensitivityAuditError(f"Phase 8A-C hard checks failed: {failed}")

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "canonical_clipping_table.npz", **_flatten_canonical(canonical))
    np.savez_compressed(
        output / "anchor_clipping_table.npz", anchor_id=selected_ids,
        action_pair_unclipped=pair_unclipped, strict_anchor_unclipped=strict_unclipped,
        anchor_any_clipping=anchor_any,
        clipped_execution_count=canonical["row_clipped"].sum(axis=(1, 2)),
        maximum_clipping_excess_linf=canonical["clipping_excess_linf"].max(axis=(1, 2)),
        mean_preclip_headroom=canonical["preclip_headroom"].mean(axis=(1, 2)),
    )
    action_main = []
    for action in ACTION_KEYS:
        ai = ACTION_INDEX[action]
        action_main.append({
            "action": action, "n_all": selected_count,
            "n_pair_unclipped": int(pair_unclipped[:, ai].sum()),
            "clipping_rate": float(np.mean(canonical["row_clipped"][:, ai])),
            "reward_drift_all": _select_mean(action_rows, "all", "reward_drift", action),
            "reward_drift_clean": _select_mean(action_rows, "action_pair_unclipped", "reward_drift", action),
            "delta_drift_all": _select_mean(action_rows, "all", "delta_drift", action),
            "delta_drift_clean": _select_mean(action_rows, "action_pair_unclipped", "delta_drift", action),
        })
    decision_main = _decision_main_table(decision_rows)
    aggregate_rows = action_rows + decision_rows
    aggregate_rows.extend({"family": "action_summary", **row} for row in action_main)
    aggregate_rows.extend({"family": "decision_main", **row} for row in decision_main)
    _write_csv(output / "aggregate_tables.csv", aggregate_rows)
    _write_csv(output / "decision_metrics.csv", decision_main)
    _write_csv(output / "action_specific_metrics.csv", action_main)
    _make_figures(output, prevalence, action_rows, decision_rows, canonical)

    hashes_after = hash_input_files(paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks["input_hashes_unchanged"] = unchanged
    if not unchanged:
        raise ClippingSensitivityAuditError("input hashes changed during analysis")
    input_integrity = {"phase8a_root": str(root), "phase8ar_root": str(review),
                       "required_file_count": len(paths), "sha256_before": hashes_before,
                       "sha256_after": hashes_after, "unchanged": unchanged}
    summary = {
        "analysis_stage": "Phase 8A-C", "kappa": kappa,
        "available_anchor_count": len(all_ids), "analyzed_anchor_count": selected_count,
        "anchor_selection": "all anchors" if full else "sorted anchor_id prefix",
        "clipping_prevalence": prevalence,
        "weighted_clipping_probabilities": weighted_rows,
        "phase8ar_full_sample_crosscheck": crosscheck,
        "action_metrics": action_rows, "decision_metrics": decision_rows,
        "hard_checks": hard_checks, "all_hard_checks_passed": all(hard_checks.values()),
        "bootstrap": {"unit": "anchor_id", "repetitions": bootstrap_reps, "seed": seed},
        "scientific_verdict": "MANUAL_DECISION_REQUIRED",
    }
    manifest_out = {
        "stage": "Phase 8A-C", "phase8a_root": str(root), "phase8ar_root": str(review),
        "output_root": str(output), "read_only_inputs": True, "kappa": kappa,
        "expected_anchor_count": expected_anchor_count, "analyzed_anchor_count": selected_count,
        "actuator_direction_v": direction.tolist(), "bootstrap_repetitions": bootstrap_reps,
        "seed": seed, "numerical_tolerance": {"atol": atol, "rtol": rtol},
        "git_commit": _git_commit(root), "python_version": platform.python_version(),
        "numpy_version": np.__version__, "matplotlib_version": plt.matplotlib.__version__,
    }
    _write_json(output / "manifest.json", manifest_out)
    _write_json(output / "input_integrity.json", input_integrity)
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
                                                "all_passed": all(hard_checks.values()),
                                                "failed": []})
    _write_json(output / "summary.json", summary)
    _write_reports(output, summary)
    return summary
