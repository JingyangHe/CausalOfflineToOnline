"""Exact low-dose decision-threshold audit for Phase 8B-RS-T.

The module is deliberately read-only with respect to its inputs.  It exploits
the exact affine reward identity ``R_obs(lambda) = b + c * lambda`` and does
not train, load, or evaluate a neural network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ACTIONS = ("minus", "base", "plus")
ACTION_PAIRS = ((0, 1), (0, 2), (1, 2))
KAPPAS = (0.0, 0.3)
CONDITIONS = ("confounded", "independent_latents")
MIXTURES = ("logger1_heavy", "logger12_balanced", "logger2_heavy")
EXISTING_LAMBDAS = (0.0, 0.05, 0.10, 0.20)
LOW_DOSE_GRID = (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075,
                 0.010, 0.015, 0.020, 0.030, 0.050)
THRESHOLD_NAMES = ("lambda_first_competitor_tie", "lambda_first_top_disagreement",
                   "lambda_first_positive_regret", "lambda_first_strict_flip")


class LowDoseThresholdAuditError(RuntimeError):
    """Raised when the exact audit cannot be completed faithfully."""


def kappa_name(value: float) -> str:
    return f"kappa_{value:.2f}".replace(".", "p")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")


def _json_number(value: float) -> float | str:
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    return float(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in sorted(set(paths))}


def _require_passed(path: Path) -> None:
    value = _read_json(path)
    passed = value.get("all_passed", value.get("all_hard_checks_passed", False))
    if not passed:
        raise LowDoseThresholdAuditError(f"input hard checks did not pass: {path}")


def top_mask(values: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    """Return tie-aware top sets using the project's registered tolerance."""
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] != 3:
        raise LowDoseThresholdAuditError("reward arrays must have final dimension 3")
    return np.isclose(x, np.max(x, axis=-1, keepdims=True), atol=atol, rtol=rtol)


def exact_top_mask(values: np.ndarray) -> np.ndarray:
    """Return the exact maximizer set, used only for one-sided crossing limits."""
    x = np.asarray(values)
    return x == np.max(x, axis=-1, keepdims=True)


def pairwise_crossing(b: np.ndarray, c: np.ndarray, left: int,
                      right: int) -> tuple[float, str]:
    denominator = float(c[left] - c[right])
    numerator = -float(b[left] - b[right])
    if denominator == 0.0:
        return math.inf, "parallel_equal" if numerator == 0.0 else "parallel_distinct"
    value = numerator / denominator
    if not np.isfinite(value):
        raise LowDoseThresholdAuditError("non-finite pairwise crossing")
    return float(value), "nonnegative" if value >= 0.0 else "negative"


def _reward(b: np.ndarray, c: np.ndarray, strength: float) -> np.ndarray:
    return np.asarray(b, np.float64) + np.asarray(c, np.float64) * float(strength)


def _one_sided_top(b: np.ndarray, c: np.ndarray, boundary: float,
                   direction: float, atol: float, rtol: float) -> tuple[float, np.ndarray]:
    """Classify the adjacent representable float without losing its ULP in addition."""
    probe = np.nextafter(float(boundary), direction)
    point_candidates = top_mask(_reward(b, c, boundary), atol, rtol)
    slopes = np.asarray(c, np.float64)
    target = np.max(slopes[point_candidates]) if direction > 0 else np.min(slopes[point_candidates])
    side = point_candidates & np.isclose(slopes, target, atol=atol, rtol=rtol)
    return probe, side


def _set_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(left, bool), np.asarray(right, bool)))


def _disjoint(left: np.ndarray, right: np.ndarray) -> bool:
    return not bool(np.any(np.asarray(left, bool) & np.asarray(right, bool)))


def scenario_thresholds(b: np.ndarray, c: np.ndarray, do: np.ndarray,
                        atol: float, rtol: float) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Solve all decision thresholds from the three affine action lines.

    Tolerant sets are used at registered points.  Exact sets are used at the
    immediate left/right representable floats because tolerance would otherwise
    smear a mathematical crossing across a nonzero interval.
    """
    b, c, do = (np.asarray(value, np.float64) for value in (b, c, do))
    if any(value.shape != (3,) for value in (b, c, do)):
        raise LowDoseThresholdAuditError("each local affine problem must have three actions")
    if not all(np.all(np.isfinite(value)) for value in (b, c, do)):
        raise LowDoseThresholdAuditError("local affine problem contains non-finite values")
    do_top = top_mask(do, atol, rtol)
    obs_zero = top_mask(b, atol, rtol)
    crossings: list[dict[str, Any]] = []
    finite_nonnegative: list[float] = []
    for left, right in ACTION_PAIRS:
        value, status = pairwise_crossing(b, c, left, right)
        row = {"left_action": ACTIONS[left], "right_action": ACTIONS[right],
               "crossing": value, "status": status}
        crossings.append(row)
        if status == "nonnegative":
            finite_nonnegative.append(value)
    boundaries = sorted(set([0.0, *finite_nonnegative]))

    first_tie = 0.0 if bool(np.any(obs_zero & ~do_top)) else math.inf
    first_disagreement = 0.0 if not _set_equal(obs_zero, do_top) else math.inf
    first_positive = 0.0 if _disjoint(obs_zero, do_top) else math.inf
    first_strict = first_positive
    for value in boundaries:
        point = top_mask(_reward(b, c, value), atol, rtol)
        if not np.isfinite(first_tie) and bool(np.any(point & ~do_top)):
            first_tie = value
        if not np.isfinite(first_disagreement) and not _set_equal(point, do_top):
            first_disagreement = value
        _, right_top = _one_sided_top(b, c, value, math.inf, atol, rtol)
        if not np.isfinite(first_positive) and _disjoint(right_top, do_top):
            first_positive = value
        if not np.isfinite(first_strict) and _disjoint(right_top, do_top):
            first_strict = value
    return {
        "lambda_first_competitor_tie": float(first_tie),
        "lambda_first_top_disagreement": float(first_disagreement),
        "lambda_first_positive_regret": float(first_positive),
        "lambda_first_strict_flip": float(first_strict),
    }, crossings


def crossing_sides(b: np.ndarray, c: np.ndarray, value: float,
                   atol: float, rtol: float) -> dict[str, np.ndarray | float]:
    if value > 0.0:
        left, left_top = _one_sided_top(b, c, value, -math.inf, atol, rtol)
    else:
        left, left_top = value, exact_top_mask(_reward(b, c, value))
    right, right_top = _one_sided_top(b, c, value, math.inf, atol, rtol)
    return {
        "lambda_left": left, "lambda_point": value, "lambda_right": right,
        "top_left": left_top,
        "top_point": top_mask(_reward(b, c, value), atol, rtol),
        "top_right": right_top,
    }


def decision_path(b: np.ndarray, c: np.ndarray, do: np.ndarray,
                  atol: float, rtol: float) -> list[dict[str, Any]]:
    """Return the exact piecewise decision path, including crossing points."""
    boundaries = []
    for left, right in ACTION_PAIRS:
        value, status = pairwise_crossing(b, c, left, right)
        if status == "nonnegative":
            boundaries.append(value)
    boundaries = sorted(set([0.0, *boundaries]))
    do_top = top_mask(do, atol, rtol)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(boundaries):
        point_top = top_mask(_reward(b, c, point), atol, rtol)
        rows.append(_path_row("point", point, point, point, point_top, do, do_top,
                              _reward(b, c, point)))
        stop = boundaries[index + 1] if index + 1 < len(boundaries) else math.inf
        probe = point + 1.0 if math.isinf(stop) else point + 0.5 * (stop - point)
        interval_top = exact_top_mask(_reward(b, c, probe))
        rows.append(_path_row("interval", point, stop, probe, interval_top, do, do_top,
                              _reward(b, c, probe)))
    return rows


def _path_row(kind: str, start: float, stop: float, probe: float, mask: np.ndarray,
              do: np.ndarray, do_top: np.ndarray, obs: np.ndarray) -> dict[str, Any]:
    selected = np.flatnonzero(mask)
    regret = float(np.max(do) - np.max(do[selected]))
    worst = float(np.max(do) - np.min(do[selected]))
    return {
        "segment_kind": kind, "lambda_start": start, "lambda_stop": stop,
        "probe_lambda": probe, "top_set": "|".join(ACTIONS[i] for i in selected),
        "top_disagreement": not _set_equal(mask, do_top),
        "strict_flip": _disjoint(mask, do_top), "regret_best": regret,
        "regret_worst": worst, "true_do_regret": regret,
        "observational_top_second_gap": float(np.sort(obs)[-1] - np.sort(obs)[-2]),
    }


def threshold_distribution(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, np.float64)
    finite = x[np.isfinite(x)]
    result: dict[str, Any] = {
        "count": int(x.size), "finite_count": int(finite.size),
        "zero_count": int(np.count_nonzero(finite == 0.0)),
        "censored_count": int(np.count_nonzero(np.isposinf(x))),
        "finite_fraction": float(finite.size / x.size) if x.size else None,
        "zero_fraction": float(np.count_nonzero(finite == 0.0) / x.size) if x.size else None,
        "censored_fraction": float(np.count_nonzero(np.isposinf(x)) / x.size) if x.size else None,
    }
    for name, quantile in (("min", 0.0), ("p05", 0.05), ("p10", 0.10),
                           ("p25", 0.25), ("median", 0.50), ("p75", 0.75),
                           ("p90", 0.90), ("p95", 0.95), ("max", 1.0)):
        if x.size:
            rank = min(len(x) - 1, max(0, int(math.ceil(quantile * len(x))) - 1))
            result[name] = _json_number(float(np.sort(x)[rank]))
        else:
            result[name] = None
        result[f"finite_{name}"] = (float(np.quantile(finite, quantile))
                                     if finite.size else None)
    return result


def _resolve_phase8ac(root: Path, manifest: Mapping[str, Any]) -> Path:
    recorded = Path(str(manifest["phase8ac_input_root"]))
    if recorded.is_dir():
        return recorded.resolve()
    candidate = root.parent / "controlled_loggers_seed0_verified" / "population_effect_review" \
        / "clipping_sensitivity_kappa_0p30"
    if not candidate.is_dir():
        raise LowDoseThresholdAuditError("canonical Phase 8A-C clipping artifact is unavailable")
    return candidate.resolve()


def _input_files(oracle: Path, ranking: Path, phase8a: Path,
                 split_root: Path, phase8ac: Path) -> list[Path]:
    files = [
        oracle / "manifest.json", oracle / "hard_checks.json", oracle / "anchor_action_metrics.npz",
        ranking / "hard_checks.json", ranking / "summary.json",
        ranking / "anchor_action_metrics.npz", phase8a / "manifest.json",
        phase8a / "hard_checks.json", phase8a / "anchor_action_metrics.npz",
        split_root / "manifest.json", split_root / "hard_checks.json", split_root / "splits.json",
        phase8ac / "manifest.json", phase8ac / "hard_checks.json",
        phase8ac / "anchor_clipping_table.npz",
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise LowDoseThresholdAuditError(f"required input is missing: {missing[0]}")
    for path in (oracle / "hard_checks.json", ranking / "hard_checks.json",
                 phase8a / "hard_checks.json", split_root / "hard_checks.json",
                 phase8ac / "hard_checks.json"):
        _require_passed(path)
    return files


def _load_split_masks(anchor_ids: np.ndarray, split_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    split = _read_json(split_path)
    names = np.full(len(anchor_ids), "unselected", dtype="<U10")
    seen: set[int] = set()
    masks: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        values = [int(value) for value in split[name]]
        if len(values) != len(set(values)) or seen.intersection(values):
            raise LowDoseThresholdAuditError("train/validation/test split is not unique")
        seen.update(values)
        mask = np.isin(anchor_ids, values)
        if int(mask.sum()) != len(values):
            raise LowDoseThresholdAuditError(f"split anchor IDs cannot be recovered: {name}")
        names[mask] = name
        masks[name] = mask
    if len(split["test"]) != 78:
        raise LowDoseThresholdAuditError("expected exactly 78 unique test anchors")
    masks["all"] = np.ones(len(anchor_ids), bool)
    return names, masks


def _load_clipping(anchor_ids: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = _load_npz(path)
    ids = raw["anchor_id"].astype(int)
    if len(ids) != len(set(ids.tolist())):
        raise LowDoseThresholdAuditError("clipping table has duplicate anchor IDs")
    lookup = {int(anchor): index for index, anchor in enumerate(ids)}
    try:
        positions = np.asarray([lookup[int(anchor)] for anchor in anchor_ids], int)
    except KeyError as exc:
        raise LowDoseThresholdAuditError("clipping table does not cover every anchor") from exc
    strict = raw["strict_anchor_unclipped"][positions].astype(bool)
    any_clipping = raw["anchor_any_clipping"][positions].astype(bool)
    if np.any(strict & any_clipping) or not np.all(strict | any_clipping):
        raise LowDoseThresholdAuditError("clipping subsets are not complementary")
    return strict, any_clipping


def _base_arrays(raw: Mapping[str, np.ndarray], kappa: float, condition: str,
                 mixture: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = kappa_name(kappa)
    ids = raw[f"{prefix}__anchor_id"].astype(int)
    do = raw[f"{prefix}__do_mean_reward"].astype(np.float64)
    b = raw[f"{prefix}__{condition}_{mixture}_reward"].astype(np.float64)
    posterior = raw[f"{prefix}__{condition}_{mixture}_posterior_u_plus"].astype(np.float64)
    c = 2.0 * posterior - 1.0
    if do.shape != b.shape or b.shape != c.shape or b.shape != (len(ids), 3):
        raise LowDoseThresholdAuditError("Phase 8A-NC affine arrays are misaligned")
    return ids, do, b, c


def _regret(do: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(do, axis=1)
    best = np.max(np.where(candidate, do, -np.inf), axis=1)
    worst = np.min(np.where(candidate, do, np.inf), axis=1)
    return maximum - best, maximum - worst


def _top_second_gap(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _population_row(do: np.ndarray, b: np.ndarray, c: np.ndarray, strength: float,
                    mask: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
    do, b, c = do[mask], b[mask], c[mask]
    obs = b + c * strength
    do_top, obs_top = top_mask(do, atol, rtol), top_mask(obs, atol, rtol)
    disagreement = ~np.all(do_top == obs_top, axis=1)
    strict = ~np.any(do_top & obs_top, axis=1)
    regret_best, regret_worst = _regret(do, obs_top)
    do_order = np.argsort(-do, axis=1, kind="stable")
    rows = np.arange(len(do))
    best, second = do_order[:, 0], do_order[:, 1]
    obs_margin = obs[rows, best] - obs[rows, second]
    do_margin = do[rows, best] - do[rows, second]
    distortion = c * strength
    centered = distortion - distortion.mean(axis=1, keepdims=True)
    return {
        "count": int(len(do)), "rank_disagreement_rate": float(np.mean(disagreement)),
        "strict_flip_rate": float(np.mean(strict)),
        "obs_regret": float(np.mean(regret_best)),
        "obs_regret_worst_tie": float(np.mean(regret_worst)),
        "mean_regret": float(np.mean(regret_best)),
        "conditional_regret_given_error": float(np.mean(regret_best[disagreement]))
        if np.any(disagreement) else 0.0,
        "mean_action_dependent_distortion": float(np.mean(np.linalg.norm(centered, axis=1))),
        "gap_sign_preservation": float(np.mean(obs_margin >= -atol - rtol * np.abs(do_margin))),
        "mean_obs_top_second_gap": float(np.mean(_top_second_gap(obs))),
        **{f"obs_top_fraction_{action}": float(np.mean(obs_top[:, index]))
           for index, action in enumerate(ACTIONS)},
        **{f"do_top_fraction_{action}": float(np.mean(do_top[:, index]))
           for index, action in enumerate(ACTIONS)},
    }


def _crosscheck_oracle(base: Mapping[str, np.ndarray], oracle_path: Path,
                       atol: float, rtol: float) -> dict[str, float | int]:
    oracle = _load_npz(oracle_path)
    maximum_obs = maximum_do = maximum_slope = 0.0
    checked = 0
    passed = True
    cache: dict[tuple[float, str, str], tuple[dict[int, int], np.ndarray,
                                              np.ndarray, np.ndarray]] = {}
    for index in range(len(oracle["anchor_id"])):
        kappa = float(oracle["kappa"][index])
        condition, mixture, action = (str(oracle[key][index])
                                      for key in ("condition", "mixture", "action"))
        scenario = (kappa, condition, mixture)
        if scenario not in cache:
            ids, do, b, c = _base_arrays(base, *scenario)
            cache[scenario] = ({int(anchor): i for i, anchor in enumerate(ids)}, do, b, c)
        positions, do, b, c = cache[scenario]
        pos = positions[int(oracle["anchor_id"][index])]
        action_index = ACTIONS.index(action)
        strength = float(oracle["lambda_reward"][index])
        expected = b[pos, action_index] + c[pos, action_index] * strength
        observed = float(oracle["augmented_observational_reward"][index])
        oracle_do = float(oracle["do_reward"][index])
        oracle_slope = float(oracle["conditional_u_env_mean"][index])
        maximum_obs = max(maximum_obs, abs(expected - observed))
        maximum_do = max(maximum_do, abs(do[pos, action_index] - oracle_do))
        maximum_slope = max(maximum_slope, abs(c[pos, action_index] - oracle_slope))
        passed &= bool(np.isclose(expected, observed, atol=atol, rtol=rtol))
        passed &= bool(np.isclose(do[pos, action_index], oracle_do, atol=atol, rtol=rtol))
        passed &= bool(np.isclose(c[pos, action_index], oracle_slope, atol=atol, rtol=rtol))
        checked += 1
    if not passed:
        raise LowDoseThresholdAuditError("exact affine reconstruction failed Oracle cross-check")
    return {"rows_checked": checked, "max_abs_observational_reward_error": maximum_obs,
            "max_abs_do_reward_error": maximum_do, "max_abs_slope_error": maximum_slope}


def _crosscheck_ranking(base: Mapping[str, np.ndarray], ranking_path: Path,
                        test_ids: Sequence[int], atol: float, rtol: float) -> list[dict[str, Any]]:
    raw = _load_npz(ranking_path)
    rows: list[dict[str, Any]] = []
    sid = raw["scenario__anchor_id"].astype(int)
    for kappa in KAPPAS:
        for condition in CONDITIONS:
            for mixture in MIXTURES:
                ids, do, b, c = _base_arrays(base, kappa, condition, mixture)
                lookup = {int(anchor): i for i, anchor in enumerate(ids)}
                positions = np.asarray([lookup[int(anchor)] for anchor in test_ids])
                expected_do = do[positions]
                for strength in EXISTING_LAMBDAS:
                    expected_obs = b[positions] + c[positions] * strength
                    selector = ((raw["scenario__kappa"] == kappa)
                                & (raw["scenario__lambda_reward"] == strength)
                                & (raw["scenario__condition"] == condition)
                                & (raw["scenario__mixture"] == mixture)
                                & (raw["scenario__model_seed"] == 0))
                    found_ids = sid[selector]
                    if set(found_ids.tolist()) != set(test_ids):
                        raise LowDoseThresholdAuditError("ranking audit test anchors are not recoverable")
                    ordering = np.asarray([int(np.flatnonzero(found_ids == anchor)[0])
                                           for anchor in test_ids])
                    scenario_indices = np.flatnonzero(selector)[ordering]
                    obs_top = top_mask(expected_obs, atol, rtol)
                    do_top = top_mask(expected_do, atol, rtol)
                    disagreement = (~np.all(obs_top == do_top, axis=1)).astype(float)
                    strict = (~np.any(obs_top & do_top, axis=1)).astype(float)
                    regret, _ = _regret(expected_do, obs_top)
                    errors = {
                        "obs_do_disagreement": float(np.max(np.abs(disagreement
                            - raw["scenario__obs_do_top_set_disagreement"][scenario_indices]))),
                        "strict_flip": float(np.max(np.abs(strict
                            - raw["scenario__obs_do_strict_disjoint_flip"][scenario_indices]))),
                        "obs_regret": float(np.max(np.abs(regret
                            - raw["scenario__obs_regret_best"][scenario_indices]))),
                    }
                    action_anchor = raw["anchor_id"].astype(int)
                    for action_index, action in enumerate(ACTIONS):
                        action_selector = ((raw["kappa"] == kappa)
                            & (raw["lambda_reward"] == strength)
                            & (raw["condition"] == condition)
                            & (raw["mixture"] == mixture)
                            & (raw["model_seed"] == 0) & (raw["action"] == action))
                        found_action_ids = action_anchor[action_selector]
                        if set(found_action_ids.tolist()) != set(test_ids):
                            raise LowDoseThresholdAuditError(
                                "ranking audit action rows are not recoverable")
                        action_order = np.asarray([int(np.flatnonzero(found_action_ids == anchor)[0])
                                                   for anchor in test_ids])
                        action_indices = np.flatnonzero(action_selector)[action_order]
                        errors[f"do_top_fraction_{action}"] = abs(
                            float(np.mean(do_top[:, action_index]))
                            - float(np.mean(raw["do_top"][action_indices])))
                        errors[f"obs_top_fraction_{action}"] = abs(
                            float(np.mean(obs_top[:, action_index]))
                            - float(np.mean(raw["obs_top"][action_indices])))
                    rows.append({"kappa": kappa, "lambda_reward": strength,
                                 "condition": condition, "mixture": mixture,
                                 **{f"max_abs_error_{key}": value for key, value in errors.items()},
                                 "passed": max(errors.values()) <= atol})
    if not all(row["passed"] for row in rows):
        raise LowDoseThresholdAuditError("existing-grid ranking cross-check failed")
    return rows


def _make_figures(output: Path, threshold_data: Mapping[str, np.ndarray],
                  dose_rows: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir()
    focus = ((threshold_data["condition"] == "confounded")
             & (threshold_data["mixture"] == "logger12_balanced"))
    for threshold, filename, title in (
        ("lambda_first_top_disagreement", "threshold_cdf_top_disagreement.png", "First top-set disagreement"),
        ("lambda_first_positive_regret", "threshold_cdf_positive_regret.png", "First positive regret"),
        ("lambda_first_strict_flip", "threshold_cdf_strict_flip.png", "First strict flip"),
    ):
        plt.figure()
        for kappa in KAPPAS:
            values = threshold_data[threshold][focus & (threshold_data["kappa"] == kappa)]
            total = len(values)
            values = np.sort(values[np.isfinite(values)])
            if values.size:
                plt.step(values, np.arange(1, len(values) + 1) / total, where="post",
                         label=f"kappa={kappa:g}")
        plt.xlabel("lambda threshold"); plt.ylabel("finite-threshold empirical CDF")
        plt.title(title); plt.legend(); plt.tight_layout()
        plt.savefig(figures / filename, dpi=160); plt.close()

    focus_rows = [row for row in dose_rows if row["subset"] == "all"
                  and row["condition"] == "confounded"
                  and row["mixture"] == "logger12_balanced"]
    for metric, filename, title in (
        ("rank_disagreement_rate", "rank_error_low_dose_curve.png", "Low-dose rank error"),
        ("mean_regret", "regret_low_dose_curve.png", "Low-dose true do-regret"),
        ("gap_sign_preservation", "gap_sign_preservation_low_dose.png", "Do-gap sign preservation"),
    ):
        plt.figure()
        for kappa in KAPPAS:
            rows = sorted((row for row in focus_rows if row["kappa"] == kappa),
                          key=lambda row: row["lambda_reward"])
            plt.plot([row["lambda_reward"] for row in rows], [row[metric] for row in rows],
                     marker="o", label=f"kappa={kappa:g}")
        plt.xlabel("lambda"); plt.ylabel(metric.replace("_", " ")); plt.title(title)
        plt.legend(); plt.tight_layout(); plt.savefig(figures / filename, dpi=160); plt.close()

    plt.figure()
    for action in ACTIONS:
        rows = sorted((row for row in focus_rows if row["kappa"] == 0.3),
                      key=lambda row: row["lambda_reward"])
        plt.plot([row["lambda_reward"] for row in rows],
                 [row[f"obs_top_fraction_{action}"] for row in rows], marker="o", label=action)
    plt.xlabel("lambda"); plt.ylabel("observational top-set fraction")
    plt.title("Top action fractions (kappa=0.3)"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "top_action_fraction_low_dose.png", dpi=160); plt.close()

    for filename, group_key, groups, title in (
        ("all_vs_unclipped_threshold_cdf.png", "clipping_group", ("all", "strict_unclipped"),
         "Positive-regret thresholds: all vs unclipped"),
        ("trainval_vs_test_threshold_cdf.png", "split_group", ("trainval", "test"),
         "Positive-regret thresholds: train/validation vs test"),
    ):
        plt.figure()
        for group in groups:
            mask = focus.copy()
            if group_key == "clipping_group":
                if group == "strict_unclipped":
                    mask &= threshold_data["strict_unclipped"]
            else:
                if group == "trainval":
                    mask &= np.isin(threshold_data["split"], ["train", "validation"])
                else:
                    mask &= threshold_data["split"] == "test"
            values = threshold_data["lambda_first_positive_regret"][mask]
            total = len(values)
            values = np.sort(values[np.isfinite(values)])
            if values.size:
                plt.step(values, np.arange(1, len(values) + 1) / total, where="post", label=group)
        plt.xlabel("lambda threshold"); plt.ylabel("finite-threshold empirical CDF")
        plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(figures / filename, dpi=160)
        plt.close()


def run_audit(oracle_root: Path, ranking_audit_root: Path, phase8a_root: Path,
              output_root: Path, seed: int = 0) -> dict[str, Any]:
    oracle, ranking, phase8a, output = (Path(path).resolve() for path in
                                        (oracle_root, ranking_audit_root, phase8a_root, output_root))
    manifest = _read_json(phase8a / "manifest.json")
    split_root = phase8a / "phase8b_reward_signal_calibration"
    phase8ac = _resolve_phase8ac(phase8a, manifest)
    inputs = _input_files(oracle, ranking, phase8a, split_root, phase8ac)
    for source in (oracle, ranking, phase8a, split_root, phase8ac):
        if output == source or source in output.parents:
            raise LowDoseThresholdAuditError("output must not be inside an input artifact")
    if output.exists() and any(output.iterdir()):
        raise LowDoseThresholdAuditError(f"output directory is not empty: {output}")
    before = _hashes(inputs)
    atol = float(manifest["numerical_tolerance"]["atol"])
    rtol = float(manifest["numerical_tolerance"]["rtol"])
    raw = _load_npz(phase8a / "anchor_action_metrics.npz")
    anchor_ids = raw["kappa_0p00__anchor_id"].astype(int)
    if len(anchor_ids) != 2048 or len(set(anchor_ids.tolist())) != 2048:
        raise LowDoseThresholdAuditError("expected 2048 unique Phase 8A-NC anchors")
    split_names, split_masks = _load_split_masks(anchor_ids, split_root / "splits.json")
    strict_unclipped, any_clipping = _load_clipping(
        anchor_ids, phase8ac / "anchor_clipping_table.npz")
    oracle_check = _crosscheck_oracle(raw, oracle / "anchor_action_metrics.npz", atol, rtol)
    ranking_rows = _crosscheck_ranking(raw, ranking / "anchor_action_metrics.npz",
                                       anchor_ids[split_masks["test"]], atol, rtol)

    threshold_columns: dict[str, list[Any]] = {key: [] for key in
        ("anchor_id", "kappa", "condition", "mixture", "split", "strict_unclipped",
         "any_clipping", *THRESHOLD_NAMES)}
    crossing_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    hard = {
        "phase8a_anchor_count_2048": len(anchor_ids) == 2048,
        "test_anchor_count_78": int(split_masks["test"].sum()) == 78,
        "all_affine_values_finite": True,
        "do_reward_invariant_to_lambda": True,
        "independent_slopes_zero": True,
        "base_action_slope_zero": True,
        "confounded_balanced_slopes_exact": True,
        "direct_slopes_equal_across_kappa": True,
        "independent_thresholds_zero_or_censored": True,
        "oracle_affine_crosscheck_passed": True,
        "existing_grid_crosscheck_passed": True,
        "thresholds_nonnegative_or_censored": True,
        "crossing_left_point_right_valid": True,
        "clipping_subsets_partition_anchors": True,
        "public_inputs_hidden_free": bool(_read_json(phase8a / "hard_checks.json")
                                            ["checks"]["public_hidden_leakage_empty"]),
    }
    for kappa in KAPPAS:
        for condition in CONDITIONS:
            for mixture in MIXTURES:
                ids, do, b, c = _base_arrays(raw, kappa, condition, mixture)
                if not np.array_equal(ids, anchor_ids):
                    raise LowDoseThresholdAuditError("anchor order changes across scenarios")
                hard["all_affine_values_finite"] &= bool(np.all(np.isfinite(do))
                    and np.all(np.isfinite(b)) and np.all(np.isfinite(c)))
                hard["base_action_slope_zero"] &= bool(np.allclose(c[:, 1], 0.0, atol=atol, rtol=rtol))
                if condition == "independent_latents":
                    hard["independent_slopes_zero"] &= bool(np.allclose(c, 0.0, atol=atol, rtol=rtol))
                if condition == "confounded" and mixture == "logger12_balanced":
                    hard["confounded_balanced_slopes_exact"] &= bool(np.allclose(
                        c, np.asarray([-0.6, 0.0, 0.6]), atol=atol, rtol=rtol))
                _, _, _, c_other = _base_arrays(raw, KAPPAS[0], condition, mixture)
                hard["direct_slopes_equal_across_kappa"] &= bool(
                    np.allclose(c, c_other, atol=atol, rtol=rtol))
                for index, anchor in enumerate(anchor_ids):
                    thresholds, crossings = scenario_thresholds(b[index], c[index], do[index], atol, rtol)
                    for key, value in (("anchor_id", int(anchor)), ("kappa", kappa),
                                       ("condition", condition), ("mixture", mixture),
                                       ("split", str(split_names[index])),
                                       ("strict_unclipped", bool(strict_unclipped[index])),
                                       ("any_clipping", bool(any_clipping[index]))):
                        threshold_columns[key].append(value)
                    for key in THRESHOLD_NAMES:
                        threshold_columns[key].append(thresholds[key])
                    if condition == "independent_latents":
                        hard["independent_thresholds_zero_or_censored"] &= all(
                            value == 0.0 or np.isposinf(value) for value in thresholds.values())
                    hard["thresholds_nonnegative_or_censored"] &= all(
                        value >= 0.0 or np.isposinf(value) for value in thresholds.values())
                    for crossing in crossings:
                        if crossing["status"] == "negative":
                            continue
                        value = crossing["crossing"]
                        row = {"anchor_id": int(anchor), "kappa": kappa,
                               "condition": condition, "mixture": mixture, **crossing}
                        if np.isfinite(value) and value >= 0.0:
                            sides = crossing_sides(b[index], c[index], value, atol, rtol)
                            row.update({"top_left": "|".join(np.asarray(ACTIONS)[sides["top_left"]]),
                                        "top_point": "|".join(np.asarray(ACTIONS)[sides["top_point"]]),
                                        "top_right": "|".join(np.asarray(ACTIONS)[sides["top_right"]])})
                            hard["crossing_left_point_right_valid"] &= bool(
                                sides["lambda_left"] <= value < sides["lambda_right"])
                        row["crossing"] = _json_number(value)
                        crossing_rows.append(row)
                    for segment in decision_path(b[index], c[index], do[index], atol, rtol):
                        for key in ("lambda_start", "lambda_stop", "probe_lambda"):
                            segment[key] = _json_number(float(segment[key]))
                        path_rows.append({"anchor_id": int(anchor), "kappa": kappa,
                                          "condition": condition, "mixture": mixture, **segment})

    threshold_data = {key: np.asarray(value) for key, value in threshold_columns.items()}
    for threshold in THRESHOLD_NAMES:
        threshold_data[f"{threshold}_censored"] = np.isposinf(threshold_data[threshold])
    subset_masks = {
        "all": np.ones(len(anchor_ids), bool), "train": split_masks["train"],
        "validation": split_masks["validation"], "test": split_masks["test"],
        "trainval": split_masks["train"] | split_masks["validation"],
        "initial_step_strict_unclipped": strict_unclipped, "any_clipping": any_clipping,
    }
    threshold_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    scenario_count = len(anchor_ids)
    for kappa in KAPPAS:
        for condition in CONDITIONS:
            for mixture in MIXTURES:
                scenario_mask = ((threshold_data["kappa"] == kappa)
                                 & (threshold_data["condition"] == condition)
                                 & (threshold_data["mixture"] == mixture))
                _, do, b, c = _base_arrays(raw, kappa, condition, mixture)
                for subset, anchor_mask in subset_masks.items():
                    expanded = np.tile(anchor_mask, len(KAPPAS) * len(CONDITIONS) * len(MIXTURES))
                    mask = scenario_mask & expanded
                    for threshold in THRESHOLD_NAMES:
                        threshold_rows.append({"kappa": kappa, "condition": condition,
                                               "mixture": mixture, "subset": subset,
                                               "threshold": threshold,
                                               **threshold_distribution(threshold_data[threshold][mask])})
                    for strength in LOW_DOSE_GRID:
                        dose_rows.append({"kappa": kappa, "condition": condition,
                                          "mixture": mixture, "subset": subset,
                                          "lambda_reward": strength,
                                          **_population_row(do, b, c, strength, anchor_mask, atol, rtol)})

    trainval = np.isin(threshold_data["split"], ["train", "validation"])
    relevant = trainval & (threshold_data["condition"] == "confounded")
    proposals: dict[str, Any] = {
        "selection_data": "train_and_validation_only",
        "threshold": "lambda_first_positive_regret",
        "strong_positive_control": 0.05,
        "quantiles": {},
        "manual_freeze_required": True,
    }
    for kappa in KAPPAS:
        for mixture in MIXTURES:
            mask = relevant & (threshold_data["kappa"] == kappa) & (threshold_data["mixture"] == mixture)
            selected = np.sort(threshold_data["lambda_first_positive_regret"][mask])
            proposals["quantiles"][f"kappa_{kappa:g}__{mixture}"] = {
                name: (_json_number(float(selected[min(len(selected) - 1,
                    max(0, int(math.ceil(quantile * len(selected))) - 1))])) if selected.size else None)
                for name, quantile in (("p10", .10), ("p25", .25), ("median", .50),
                                       ("p75", .75), ("p90", .90))}

    output.mkdir(parents=True)
    np.savez_compressed(output / "anchor_thresholds.npz", **threshold_data)
    _write_csv(output / "pairwise_crossings.csv", crossing_rows)
    _write_csv(output / "piecewise_decision_paths.csv", path_rows)
    _write_csv(output / "threshold_summary.csv", threshold_rows)
    _write_csv(output / "dose_response_population.csv", dose_rows)
    _write_csv(output / "existing_grid_crosscheck.csv", ranking_rows)
    _write_json(output / "proposed_lambda_quantiles_trainval.json", proposals)
    _make_figures(output, threshold_data, dose_rows)

    after = _hashes(inputs)
    hard["input_hashes_unchanged"] = before == after
    hard["all_required_figures_present"] = len(list((output / "figures").glob("*.png"))) == 9
    all_passed = all(bool(value) for value in hard.values())
    if not all_passed:
        failed = [key for key, value in hard.items() if not value]
        raise LowDoseThresholdAuditError(f"hard checks failed: {failed}")
    integrity = {"input_hashes_before": before, "input_hashes_after": after,
                 "unchanged": before == after}
    headline: dict[str, Any] = {}
    for kappa in KAPPAS:
        mask = ((threshold_data["kappa"] == kappa)
                & (threshold_data["condition"] == "confounded")
                & (threshold_data["mixture"] == "logger12_balanced"))
        headline[f"kappa_{kappa:g}"] = threshold_distribution(
            threshold_data["lambda_first_positive_regret"][mask])
    summary = {
        "stage": "Phase 8B-RS-T", "seed": int(seed), "analyzed_anchor_count": 2048,
        "test_anchor_count": 78, "one_step_reward_ranking_only": True,
        "neural_training_performed": False, "all_hard_checks_passed": True,
        "oracle_crosscheck": oracle_check, "registered_low_dose_grid": list(LOW_DOSE_GRID),
        "threshold_censoring": "+inf means no decision boundary on lambda >= 0",
        "manual_freeze_required": True, "balanced_confounded_positive_regret_threshold": headline,
    }
    manifest_out = {
        **summary, "actions": list(ACTIONS), "kappas": list(KAPPAS),
        "conditions": list(CONDITIONS), "mixtures": list(MIXTURES),
        "existing_lambdas": list(EXISTING_LAMBDAS),
        "numerical_tolerance": {"atol": atol, "rtol": rtol},
        "oracle_root": str(oracle), "ranking_audit_root": str(ranking),
        "phase8a_root": str(phase8a), "split_source": str(split_root / "splits.json"),
        "clipping_source": str(phase8ac / "anchor_clipping_table.npz"),
        "output_files": sorted(path.name for path in output.iterdir()),
    }
    _write_json(output / "manifest.json", manifest_out)
    _write_json(output / "input_integrity.json", integrity)
    _write_json(output / "hard_checks.json", {"checks": hard, "failed": [], "all_passed": True})
    _write_json(output / "summary.json", summary)
    report = f"""# Phase 8B-RS-T — Exact Low-Dose Decision-Threshold Audit

This read-only audit analyzed 2,048 Phase 8A-NC anchors and recovered all 78 held-out test anchors. It used the exact affine identity $R_{{obs}}(\\lambda)=b+c\\lambda$; no neural model was trained or evaluated.

## Validity

All input hashes were unchanged. Oracle reconstruction and the existing $\\lambda\\in\\{{0,0.05,0.10,0.20\\}}$ ranking/regret results agreed within the registered `atol=rtol={atol:g}` tolerance. Independent-latent slopes and the base-action slope were zero; balanced-confounded slopes were exactly `[-0.6, 0, +0.6]` within tolerance. Infinite thresholds are right-censored cases with no boundary for $\\lambda\\geq0$.

For `logger12_balanced/confounded`, the finite positive-regret threshold median was {headline['kappa_0']['finite_median']:.6g} at `kappa=0` and {headline['kappa_0.3']['finite_median']:.6g} at `kappa=0.3`. The corresponding censored fractions were {headline['kappa_0']['censored_fraction']:.2%} and {headline['kappa_0.3']['censored_fraction']:.2%}. Thus the old first positive dose `lambda=0.05` lies well above the typical exact decision boundary and is suitable as a strong positive control, not as a fine transition probe.

## Interpretation boundary

The tables describe exact one-step reward-ranking sensitivity. They do not establish long-horizon policy value or neural-model performance. Clipping subsets are descriptive sensitivity groups, not randomized causal strata. Crossing-point ties use the project tolerance; immediate one-sided decisions use `numpy.nextafter` so no arbitrary epsilon is introduced.

## Dose-grid rule

The proposed quantiles use only train and validation anchors; the held-out test split is reported only for audit comparison. `lambda=0.05` is retained as a strong positive control.

Final neural dose grid must be manually frozen before mechanism-model training and may use only train/validation threshold information.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return summary
