"""Audit multi-scale local public-state support for Hopper source groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_hopper_source_groups import CROSS_SEED_LABELS, STAGE_LABELS


K_GRID = np.asarray((1, 4, 16, 64, 256), dtype=np.int64)
PUBLIC_FIELDS = {
    "observation", "action", "reward", "next_observation", "terminated",
    "truncated", "source_id", "episode_id", "time_step",
}
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))


def _stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise RuntimeError("local-support metric is empty or nonfinite")
    return {
        "mean": float(np.mean(data)), "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)), "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)), "p90": float(np.percentile(data, 90)),
    }


def deterministic_indices(size: int, count: int) -> np.ndarray:
    if size <= 0 or count <= 0:
        raise ValueError("size and count must be positive")
    return np.linspace(0, size - 1, min(size, count), dtype=np.int64)


def find_public_pilot(directory: Path, group: str) -> Path:
    directory = Path(directory)
    expected = {
        "stage_group": "stage_group_public_pilot.npz",
        "cross_seed_500k_group": "cross_seed_500k_public_pilot.npz",
    }[group]
    direct = directory / expected
    if direct.is_file():
        return direct
    token = "stage" if group == "stage_group" else "cross_seed"
    candidates = [
        path for path in directory.glob("*.npz")
        if token in path.name.lower() and "public" in path.name.lower()
        and "pilot" in path.name.lower() and "hidden" not in path.name.lower()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one public pilot for {group} in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def load_public_pilot(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        if fields != PUBLIC_FIELDS:
            raise RuntimeError(f"public pilot has invalid fields: {sorted(fields)}")
        data = {field: archive[field].copy() for field in PUBLIC_FIELDS}
    if data["observation"].shape[1:] != (12,) or data["action"].shape[1:] != (3,):
        raise RuntimeError("public pilot observation or action shape is invalid")
    counts = [int(np.sum(data["source_id"] == source_id)) for source_id in (1, 2, 3)]
    if len(set(counts)) != 1 or counts[0] < int(K_GRID[-1]) + 1:
        raise RuntimeError(f"public pilot source counts are invalid: {counts}")
    return data


def kth_neighbor_radii(
    queries: np.ndarray, source_states: np.ndarray, k_grid: np.ndarray = K_GRID
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    maximum = int(np.max(k_grid))
    if source_states.shape[0] < maximum:
        raise ValueError("source has fewer states than the largest requested k")
    distances, indices = cKDTree(source_states).query(queries, k=maximum)
    if maximum == 1:
        distances, indices = distances[:, None], indices[:, None]
    return distances[:, k_grid - 1], indices


def support_geometry(
    source_radii: np.ndarray, self_radii: np.ndarray
) -> dict[str, np.ndarray]:
    ordered = np.sort(np.asarray(source_radii, dtype=np.float64), axis=1)
    all_source, best_two = ordered[:, -1, :], ordered[:, 1, :]
    self_values = np.asarray(self_radii, dtype=np.float64)
    return {
        "all_source_radii": all_source,
        "best_two_radii": best_two,
        "all_source_ratios": all_source / self_values,
        "best_two_ratios": best_two / self_values,
        "third_source_penalties": all_source - best_two,
        "third_source_ratios": all_source / (best_two + 1e-12),
    }


def normalized_action_separation(
    first_mean: np.ndarray, second_mean: np.ndarray,
    first_variation: np.ndarray, second_variation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.linalg.norm(np.asarray(first_mean) - np.asarray(second_mean), axis=-1)
    scale = 0.5 * (np.asarray(first_variation) + np.asarray(second_variation)) + 1e-12
    return raw, raw / scale


def source3_action_novelty(local_action_means: np.ndarray) -> np.ndarray:
    means = np.asarray(local_action_means)
    first = np.linalg.norm(means[:, 2] - means[:, 0], axis=-1)
    second = np.linalg.norm(means[:, 2] - means[:, 1], axis=-1)
    return np.minimum(first, second)


def self_source_radii(
    queries: np.ndarray,
    query_global_indices: np.ndarray,
    query_source_ids: np.ndarray,
    reference_states: dict[int, np.ndarray],
    reference_global_indices: dict[int, np.ndarray],
    k_grid: np.ndarray = K_GRID,
) -> np.ndarray:
    """Compute query-origin Stage radii while removing each query's own row."""
    from scipy.spatial import cKDTree

    output = np.empty((queries.shape[0], k_grid.size), dtype=np.float64)
    maximum = int(np.max(k_grid))
    for source_id in (1, 2, 3):
        rows = np.flatnonzero(query_source_ids == source_id)
        if rows.size == 0:
            continue
        distances, neighbors = cKDTree(reference_states[source_id]).query(
            queries[rows], k=maximum + 1
        )
        global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(reference_global_indices[source_id])
        }
        for local_row, query_row in enumerate(rows):
            own = global_to_local[int(query_global_indices[query_row])]
            keep = neighbors[local_row] != own
            filtered = distances[local_row][keep]
            if filtered.size < maximum:
                raise RuntimeError("could not exclude self while retaining requested neighbors")
            output[query_row] = filtered[k_grid - 1]
    return output


def _partition(data: dict[str, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    states, actions, indices = {}, {}, {}
    for source_id in (1, 2, 3):
        mask = data["source_id"] == source_id
        states[source_id] = data["observation"][mask]
        actions[source_id] = data["action"][mask]
        indices[source_id] = np.flatnonzero(mask)
    return states, actions, indices


def _closest_fraction_summary(radius: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    result = {}
    for fraction in (0.10, 0.25, 0.50, 1.00):
        count = max(1, int(np.ceil(radius.size * fraction)))
        selected = values[np.argsort(radius, kind="stable")[:count]]
        result[f"closest_{int(100 * fraction)}pct"] = {
            "mean": float(np.mean(selected)), "median": float(np.median(selected)),
        }
    return result


def _audit_group(
    group_states: dict[int, np.ndarray],
    group_actions: dict[int, np.ndarray],
    stage_reference_states: dict[int, np.ndarray],
    stage_reference_indices: dict[int, np.ndarray],
    queries_raw: np.ndarray,
    query_indices: np.ndarray,
    query_source_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pooled = np.concatenate(list(group_states.values()))
    mean = np.mean(pooled, axis=0)
    scale = np.maximum(np.std(pooled, axis=0), 1e-8)
    queries = (queries_raw - mean) / scale
    normalized = {key: (value - mean) / scale for key, value in group_states.items()}
    reference = {key: (value - mean) / scale for key, value in stage_reference_states.items()}
    source_radii = np.empty((queries.shape[0], 3, K_GRID.size))
    local_means = np.empty((queries.shape[0], 3, K_GRID.size, 3))
    local_variation = np.empty((queries.shape[0], 3, K_GRID.size))
    for source_column, source_id in enumerate((1, 2, 3)):
        radii, neighbors = kth_neighbor_radii(queries, normalized[source_id])
        source_radii[:, source_column] = radii
        for k_column, k in enumerate(K_GRID):
            nearby_actions = group_actions[source_id][neighbors[:, :k]]
            action_mean = np.mean(nearby_actions, axis=1)
            local_means[:, source_column, k_column] = action_mean
            local_variation[:, source_column, k_column] = np.mean(
                np.linalg.norm(nearby_actions - action_mean[:, None], axis=2), axis=1
            )
    self_radii = self_source_radii(
        queries, query_indices, query_source_ids, reference,
        stage_reference_indices,
    )
    geometry = support_geometry(source_radii, self_radii)
    pair_radius = np.empty((queries.shape[0], 3, K_GRID.size))
    raw_separation = np.empty_like(pair_radius)
    normalized_separation = np.empty_like(pair_radius)
    for pair_column, (first, second) in enumerate(PAIR_INDICES):
        pair_radius[:, pair_column] = np.maximum(
            source_radii[:, first], source_radii[:, second]
        )
        raw, normalized_action = normalized_action_separation(
            local_means[:, first], local_means[:, second],
            local_variation[:, first], local_variation[:, second],
        )
        raw_separation[:, pair_column] = raw
        normalized_separation[:, pair_column] = normalized_action
    arrays = {
        "source_radii": source_radii, "self_radii": self_radii,
        "local_action_means": local_means,
        "within_source_action_variation": local_variation,
        "pairwise_local_state_radius": pair_radius,
        "pairwise_raw_action_separation": raw_separation,
        "pairwise_normalized_action_separation": normalized_separation,
        **geometry,
    }
    summary = {"support_curve": {}, "action_separation_curve": {}}
    for k_column, k in enumerate(K_GRID):
        summary["support_curve"][str(k)] = {
            name: _stats(values[:, k_column])
            for name, values in geometry.items()
        }
        pair_report = {}
        for pair_column, (first, second) in enumerate(PAIR_INDICES):
            label = f"policy_{first + 1}_vs_policy_{second + 1}"
            pair_report[label] = {
                "local_state_radius": _stats(pair_radius[:, pair_column, k_column]),
                "raw_action_separation": _stats(raw_separation[:, pair_column, k_column]),
                "normalized_action_separation": _stats(
                    normalized_separation[:, pair_column, k_column]
                ),
                "closest_state_regions": _closest_fraction_summary(
                    pair_radius[:, pair_column, k_column],
                    normalized_separation[:, pair_column, k_column],
                ),
            }
        summary["action_separation_curve"][str(k)] = pair_report
    return arrays, summary


def _source3_report(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    novelty = source3_action_novelty(arrays["local_action_means"])
    radius = arrays["source_radii"][:, 2]
    report = {}
    for k_column, k in enumerate(K_GRID):
        report[str(k)] = {
            "novelty": _stats(novelty[:, k_column]),
            "source3_local_radius": _stats(radius[:, k_column]),
            "novelty_in_best_supported_regions": _closest_fraction_summary(
                radius[:, k_column], novelty[:, k_column]
            ),
        }
    return novelty, report


def _validate_labels(summary: dict[str, Any]) -> None:
    expected = {
        "stage_group": list(STAGE_LABELS),
        "cross_seed_500k_group": list(CROSS_SEED_LABELS),
    }
    for group, labels in expected.items():
        actual = summary.get("groups", {}).get(group, {}).get("checkpoint_labels")
        if actual != labels:
            raise RuntimeError(f"unexpected Phase 6C labels for {group}: {actual}")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(arguments.group_comparison_dir)
    comparison_path = source_dir / "source_group_comparison_summary.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"missing Phase 6C summary: {comparison_path}")
    phase6c = json.loads(comparison_path.read_text(encoding="utf-8"))
    _validate_labels(phase6c)
    pilot_paths = {
        group: find_public_pilot(source_dir, group)
        for group in ("stage_group", "cross_seed_500k_group")
    }
    pilots = {group: load_public_pilot(path) for group, path in pilot_paths.items()}
    stage_states, stage_actions, stage_indices = _partition(pilots["stage_group"])
    cross_states, cross_actions, _ = _partition(pilots["cross_seed_500k_group"])
    pooled_indices = deterministic_indices(pilots["stage_group"]["observation"].shape[0], arguments.query_count)
    stable_pool = stage_indices[2]
    stable_indices = stable_pool[deterministic_indices(stable_pool.size, arguments.query_count)]
    query_sets = {"pooled_queries": pooled_indices, "stable_policy_queries": stable_indices}
    representations = {"physical_11d": slice(None, -1), "public_12d": slice(None)}
    all_arrays: dict[str, np.ndarray] = {
        "k_grid": K_GRID, "pooled_query_indices": pooled_indices,
        "stable_policy_query_indices": stable_indices,
        "stage_labels": np.asarray(STAGE_LABELS), "cross_seed_labels": np.asarray(CROSS_SEED_LABELS),
        "pair_indices": np.asarray(PAIR_INDICES, dtype=np.int8),
    }
    group_summary: dict[str, Any] = {"stage_group": {}, "cross_seed_500k_group": {}}
    source3_summary: dict[str, Any] = {}
    for query_name, query_indices in query_sets.items():
        query_source_ids = pilots["stage_group"]["source_id"][query_indices]
        for representation, columns in representations.items():
            queries = pilots["stage_group"]["observation"][query_indices, columns]
            reference = {key: values[:, columns] for key, values in stage_states.items()}
            for group, (states, actions) in {
                "stage_group": (stage_states, stage_actions),
                "cross_seed_500k_group": (cross_states, cross_actions),
            }.items():
                represented = {key: values[:, columns] for key, values in states.items()}
                arrays, report = _audit_group(
                    represented, actions, reference, stage_indices,
                    queries, query_indices, query_source_ids,
                )
                group_summary[group].setdefault(query_name, {})[representation] = report
                prefix = f"{group}_{query_name}_{representation}"
                all_arrays.update({f"{prefix}_{key}": value for key, value in arrays.items()})
                if group == "stage_group":
                    novelty, novelty_report = _source3_report(arrays)
                    source3_summary.setdefault(query_name, {})[representation] = novelty_report
                    all_arrays[f"{prefix}_source3_action_novelty"] = novelty
    comparison: dict[str, Any] = {}
    for query_name in query_sets:
        comparison[query_name] = {}
        for representation in representations:
            comparison[query_name][representation] = {}
            for k in K_GRID:
                stage_curve = group_summary["stage_group"][query_name][representation]["support_curve"][str(k)]
                cross_curve = group_summary["cross_seed_500k_group"][query_name][representation]["support_curve"][str(k)]
                comparison[query_name][representation][str(k)] = {
                    "STAGE_MEDIAN_ALL_SOURCE_RADIUS": stage_curve["all_source_radii"]["median"],
                    "CROSS_SEED_MEDIAN_ALL_SOURCE_RADIUS": cross_curve["all_source_radii"]["median"],
                    "ALL_SOURCE_RADIUS_MEDIAN_DIFFERENCE": cross_curve["all_source_radii"]["median"] - stage_curve["all_source_radii"]["median"],
                    "STAGE_MEDIAN_BEST_TWO_RADIUS": stage_curve["best_two_radii"]["median"],
                    "CROSS_SEED_MEDIAN_BEST_TWO_RADIUS": cross_curve["best_two_radii"]["median"],
                    "BEST_TWO_RADIUS_MEDIAN_DIFFERENCE": cross_curve["best_two_radii"]["median"] - stage_curve["best_two_radii"]["median"],
                }
    historical = {
        group: {
            representation: phase6c["group_structure"][group][representation][
                "different_source_nearest_neighbor_fraction"
            ]
            for representation in representations
        }
        for group in group_summary
    }
    summary = {
        "phase": "6D", "audit_type": "LOCAL_SUPPORT_AUDIT",
        "ACTUAL_JOINT_TIGHTENING": "NOT_EVALUATED", "diagnostic_seed": arguments.seed,
        "k_grid": K_GRID.tolist(), "query_count": arguments.query_count,
        "input_public_pilots": {key: str(value) for key, value in pilot_paths.items()},
        "hidden_audit_loaded": False,
        "self_reference_definition": "Stage query-origin policy, self row excluded, transformed in each audited group's pooled coordinates.",
        "group_local_support": group_summary, "source3_local_contribution": source3_summary,
        "stage_vs_cross_seed": comparison,
        "historical_different_source_nn_fraction": historical,
        "method_pilot_recommendation": "NOT_AUTOMATED_REQUIRES_HUMAN_REVIEW",
        "formal_offline_dataset_generated": False,
    }
    json.dumps(summary, allow_nan=False)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "local_joint_support_audit.npz", **all_arrays)
    (output_dir / "local_joint_support_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for query_name in query_sets:
        for representation in representations:
            print(f"\n{query_name} / {representation}")
            print("k  stage_all  cross_all  stage_two  cross_two")
            for k in K_GRID:
                item = comparison[query_name][representation][str(k)]
                print(
                    f"{k:<3d} {item['STAGE_MEDIAN_ALL_SOURCE_RADIUS']:<10.4f} "
                    f"{item['CROSS_SEED_MEDIAN_ALL_SOURCE_RADIUS']:<10.4f} "
                    f"{item['STAGE_MEDIAN_BEST_TWO_RADIUS']:<10.4f} "
                    f"{item['CROSS_SEED_MEDIAN_BEST_TWO_RADIUS']:<10.4f}"
                )
    print("\nACTUAL_JOINT_TIGHTENING = NOT_EVALUATED")
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-comparison-dir", type=Path,
        default=Path("artifacts/hopper_behavior_source_audit/group_comparison"),
    )
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/hopper_behavior_source_audit/local_joint_support"),
    )
    arguments = parser.parse_args()
    if arguments.query_count <= 0:
        parser.error("--query-count must be positive")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
    print("PHASE6D_LOCAL_JOINT_SUPPORT_AUDIT_COMPLETE")
