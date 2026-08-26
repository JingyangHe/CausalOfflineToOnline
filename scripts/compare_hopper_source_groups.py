"""Compare stage-based and cross-seed 500k Confounded Hopper source groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_hopper_behavior_sources import run_source_group_audit


STAGE_LABELS = ("stage_200k_seed0", "stage_500k_seed0", "stage_1000k_seed0")
CROSS_SEED_LABELS = ("seed0_500k", "seed1_500k", "seed2_500k")
FORMAL_SOURCE_MAPPING = {"source_1": 200_000, "source_2": 500_000, "source_3": 1_000_000}
FORBIDDEN_PUBLIC_FIELDS = {"hidden_u", "applied_action", "qpos", "qvel"}


def find_unique_checkpoint_by_step(directory: Path, step: int) -> Path:
    """Find exactly one zip whose filename records the requested checkpoint step."""
    directory = Path(directory)
    matches = []
    for path in directory.glob("*.zip") if directory.is_dir() else ():
        match = re.search(r"step_(\d+)\.zip$", path.name)
        if match and int(match.group(1)) == step:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one step={step} checkpoint in {directory}, found {len(matches)}"
        )
    return matches[0]


def resolve_checkpoint_groups(
    seed0_dir: Path, seed1_dir: Path, seed2_dir: Path
) -> dict[str, dict[str, Path]]:
    """Resolve fixed group labels from seed and checkpoint step, never source index."""
    seed0_200 = find_unique_checkpoint_by_step(seed0_dir, 200_000)
    seed0_500 = find_unique_checkpoint_by_step(seed0_dir, 500_000)
    seed0_1000 = find_unique_checkpoint_by_step(seed0_dir, 1_000_000)
    seed1_500 = find_unique_checkpoint_by_step(seed1_dir, 500_000)
    seed2_500 = find_unique_checkpoint_by_step(seed2_dir, 500_000)
    return {
        "stage_group": dict(zip(STAGE_LABELS, (seed0_200, seed0_500, seed0_1000))),
        "cross_seed_500k_group": dict(
            zip(CROSS_SEED_LABELS, (seed0_500, seed1_500, seed2_500))
        ),
    }


def state_overlap_ratio_change(stage_ratio: float, cross_seed_ratio: float) -> float:
    return float(cross_seed_ratio - stage_ratio)


def action_complementarity_ratio(cross_distance: float, within_distance: float) -> float:
    if within_distance <= 0.0:
        raise ValueError("within-source action distance must be positive")
    return float(cross_distance / within_distance)


def public_artifact_is_leakage_free(fields: Iterable[str]) -> bool:
    return not bool(FORBIDDEN_PUBLIC_FIELDS & set(fields))


def _manifest(directory: Path, expected_seed: int) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing behavior-policy manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("seed") != expected_seed:
        raise RuntimeError(f"manifest seed mismatch in {directory}")
    if manifest.get("env_id") != "Hopper-v5" or manifest.get("kappa") != 0.2:
        raise RuntimeError(f"manifest environment or kappa mismatch in {directory}")
    return manifest


def _strictly_less(first: float, second: float) -> bool:
    return not np.isclose(first, second, atol=1e-12, rtol=0.0) and first < second


def _strictly_greater(first: float, second: float) -> bool:
    return not np.isclose(first, second, atol=1e-12, rtol=0.0) and first > second


def summarize_group_structure(group: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the same six directed Phase 6B comparisons without thresholds."""
    result = {}
    coverage = group["state_coverage_and_action_complementarity"]
    for representation in ("physical_11d", "public_12d"):
        report = coverage[representation]
        if report.get("status") != "AVAILABLE":
            result[representation] = {"status": report.get("status", "NOT_AVAILABLE")}
            continue
        within = list(report["within"].values())
        cross = list(report["directed_cross"].values())
        within_state_median = float(np.mean([item["state_distance"]["median"] for item in within]))
        cross_state_median = float(np.mean([item["matched_state_distance"]["median"] for item in cross]))
        ratio_mean = float(np.mean([item["cross_over_within_state_distance"]["mean"] for item in cross]))
        ratio_median = float(np.mean([item["cross_over_within_state_distance"]["median"] for item in cross]))
        within_action_mean = float(np.mean([item["matched_action_distance"]["mean"] for item in within]))
        within_action_median = float(np.mean([item["matched_action_distance"]["median"] for item in within]))
        cross_action_mean = float(np.mean([item["matched_action_distance"]["mean"] for item in cross]))
        cross_action_median = float(np.mean([item["matched_action_distance"]["median"] for item in cross]))
        result[representation] = {
            "status": "AVAILABLE",
            "within_state_distance_median": within_state_median,
            "cross_state_distance_median": cross_state_median,
            "cross_over_within_state_distance_ratio_mean": ratio_mean,
            "cross_over_within_state_distance_ratio_median": ratio_median,
            "different_source_nearest_neighbor_fraction": report[
                "nearest_neighbor_has_different_source"
            ],
            "cross_source_matched_action_distance_mean": cross_action_mean,
            "cross_source_matched_action_distance_median": cross_action_median,
            "within_source_matched_action_distance_mean": within_action_mean,
            "within_source_matched_action_distance_median": within_action_median,
            "action_complementarity_ratio_mean": action_complementarity_ratio(
                cross_action_mean, within_action_mean
            ),
            "action_complementarity_ratio_median": action_complementarity_ratio(
                cross_action_median, within_action_median
            ),
        }
    return result


def compare_group_structures(
    stage: dict[str, Any], cross_seed: dict[str, Any]
) -> dict[str, Any]:
    comparison = {}
    for representation in ("physical_11d", "public_12d"):
        first, second = stage[representation], cross_seed[representation]
        comparison[representation] = {
            "state_overlap_ratio_change_mean": state_overlap_ratio_change(
                first["cross_over_within_state_distance_ratio_mean"],
                second["cross_over_within_state_distance_ratio_mean"],
            ),
            "state_overlap_ratio_change_median": state_overlap_ratio_change(
                first["cross_over_within_state_distance_ratio_median"],
                second["cross_over_within_state_distance_ratio_median"],
            ),
            "different_source_nn_change": float(
                second["different_source_nearest_neighbor_fraction"]
                - first["different_source_nearest_neighbor_fraction"]
            ),
            "matched_action_distance_change_mean": float(
                second["cross_source_matched_action_distance_mean"]
                - first["cross_source_matched_action_distance_mean"]
            ),
            "action_complementarity_ratio_change_mean": float(
                second["action_complementarity_ratio_mean"]
                - first["action_complementarity_ratio_mean"]
            ),
            "action_complementarity_ratio_change_median": float(
                second["action_complementarity_ratio_median"]
                - first["action_complementarity_ratio_median"]
            ),
        }
    return comparison


def _spread(group: dict[str, Any], field: str, statistic: str) -> float:
    values = [report[field][statistic] for report in group["policy_quality"].values()]
    return float(max(values) - min(values))


def _termination_spread(group: dict[str, Any]) -> float:
    values = [report["early_termination_rate"] for report in group["policy_quality"].values()]
    return float(max(values) - min(values))


def _policies_use_u(group: dict[str, Any]) -> bool:
    return all(
        report["compensation_cosine"] != "NOT_AVAILABLE"
        for report in group["u_directional_compensation"].values()
    )


def _print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    strings = [[str(value) for value in row] for row in (headers, *rows)]
    widths = [max(len(row[column]) for row in strings) for column in range(len(headers))]
    for row_index, row in enumerate(strings):
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def _print_results(
    groups: dict[str, dict[str, Any]], structures: dict[str, dict[str, Any]], changes: dict[str, Any]
) -> None:
    quality_rows = []
    for group_name, group in groups.items():
        for label, quality in group["policy_quality"].items():
            directional = group["u_directional_compensation"][label]
            quality_rows.append((
                group_name, label, f'{quality["return"]["mean"]:.3f}',
                f'{quality["return"]["median"]:.3f}', f'{quality["episode_length"]["mean"]:.2f}',
                f'{quality["early_termination_rate"]:.4f}', f'{directional["projection"]["mean"]:.4f}',
                f'{directional["applied_action_residual"]["mean"]:.4f}',
                f'{quality["episode_clipped_step_proportion"]["mean"]:.4f}',
            ))
    print("\nTable 1: policy quality")
    _print_table(("group", "label", "return_mean", "return_med", "length", "term", "U_proj", "residual", "clip"), quality_rows)
    structure_rows = []
    for group_name, report in structures.items():
        physical, public = report["physical_11d"], report["public_12d"]
        structure_rows.append((
            group_name, f'{physical["cross_over_within_state_distance_ratio_mean"]:.4f}',
            f'{public["cross_over_within_state_distance_ratio_mean"]:.4f}',
            f'{physical["different_source_nearest_neighbor_fraction"]:.4f}',
            f'{public["different_source_nearest_neighbor_fraction"]:.4f}',
            f'{public["cross_source_matched_action_distance_mean"]:.4f}',
            f'{public["within_source_matched_action_distance_mean"]:.4f}',
            f'{public["action_complementarity_ratio_mean"]:.4f}',
        ))
    print("\nTable 2: group source structure")
    _print_table(("group", "phys_ratio", "pub_ratio", "phys_mix", "pub_mix", "cross_act", "within_act", "act_ratio"), structure_rows)
    print("\nTable 3: cross-seed minus stage changes")
    change_rows = []
    for representation in ("physical_11d", "public_12d"):
        item = changes[representation]
        change_rows.append((
            representation, f'{item["state_overlap_ratio_change_mean"]:.4f}',
            f'{item["different_source_nn_change"]:.4f}',
            f'{item["matched_action_distance_change_mean"]:.4f}',
            f'{item["action_complementarity_ratio_change_mean"]:.4f}',
            f'{changes["return_spread_change"]:.4f}', f'{changes["termination_spread_change"]:.4f}',
        ))
    _print_table(("space", "ratio_change", "mix_change", "action_change", "act_ratio_change", "return_spread", "term_spread"), change_rows)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    directories = (arguments.seed0_dir, arguments.seed1_dir, arguments.seed2_dir)
    manifests = [_manifest(directory, seed) for seed, directory in enumerate(directories)]
    if any(manifest["kappa"] != manifests[0]["kappa"] for manifest in manifests[1:]):
        raise RuntimeError("all source groups must use the same kappa")
    paths = resolve_checkpoint_groups(*directories)
    audit_kwargs = dict(
        kappa=float(manifests[0]["kappa"]), eval_episodes=arguments.eval_episodes,
        pilot_transitions_per_policy=arguments.pilot_transitions_per_policy,
        paired_audit_samples=arguments.paired_audit_samples, seed=arguments.seed,
        device=arguments.device,
    )
    group_bundles = {}
    for group_name, labeled_paths in paths.items():
        group_bundles[group_name] = run_source_group_audit(
            list(labeled_paths.values()), list(labeled_paths), **audit_kwargs
        )
    groups = {name: bundle[0] for name, bundle in group_bundles.items()}
    structures = {name: summarize_group_structure(group) for name, group in groups.items()}
    if any(
        report[representation].get("status") != "AVAILABLE"
        for report in structures.values()
        for representation in ("physical_11d", "public_12d")
    ):
        raise RuntimeError("SciPy cKDTree diagnostics are required for source-group comparison")
    stage_structure = structures["stage_group"]
    cross_structure = structures["cross_seed_500k_group"]
    changes = compare_group_structures(stage_structure, cross_structure)
    changes["return_spread_change"] = _spread(groups["cross_seed_500k_group"], "return", "mean") - _spread(groups["stage_group"], "return", "mean")
    changes["termination_spread_change"] = _termination_spread(groups["cross_seed_500k_group"]) - _termination_spread(groups["stage_group"])
    flags = {
        "CROSS_SEED_STATE_DISTANCE_RATIO_LOWER": all(
            _strictly_less(cross_structure[name]["cross_over_within_state_distance_ratio_mean"], stage_structure[name]["cross_over_within_state_distance_ratio_mean"])
            for name in ("physical_11d", "public_12d")
        ),
        "CROSS_SEED_DIFFERENT_SOURCE_NN_FRACTION_HIGHER": all(
            _strictly_greater(cross_structure[name]["different_source_nearest_neighbor_fraction"], stage_structure[name]["different_source_nearest_neighbor_fraction"])
            for name in ("physical_11d", "public_12d")
        ),
        "CROSS_SEED_ACTION_COMPLEMENTARITY_RATIO_GREATER_THAN_ONE": all(
            _strictly_greater(cross_structure[name][statistic], 1.0)
            for name in ("physical_11d", "public_12d")
            for statistic in ("action_complementarity_ratio_mean", "action_complementarity_ratio_median")
        ),
        "CROSS_SEED_ALL_POLICIES_USE_U": _policies_use_u(groups["cross_seed_500k_group"]),
        "CROSS_SEED_PUBLIC_DATA_LEAKAGE_FREE": all(
            public_artifact_is_leakage_free(bundle[2]) for bundle in group_bundles.values()
        ),
    }
    summary = {
        "phase": "6C", "diagnostic_seed": arguments.seed,
        "formal_source_mapping_unchanged": FORMAL_SOURCE_MAPPING,
        "groups": groups, "group_structure": structures,
        "cross_seed_minus_stage_changes": changes, "fact_flags": flags,
        "formal_source_group_selected": False, "formal_offline_dataset_generated": False,
    }
    json.dumps(summary, allow_nan=False)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_arrays = {}
    for group_name, bundle in group_bundles.items():
        audit_arrays.update({f"{group_name}_{key}": value for key, value in bundle[1].items()})
    for representation, report in changes.items():
        if isinstance(report, dict):
            audit_arrays.update({f"change_{representation}_{key}": np.asarray(value) for key, value in report.items()})
    np.savez_compressed(output_dir / "source_group_comparison_audit.npz", **audit_arrays)
    artifact_names = {
        "stage_group": ("stage_group_public_pilot.npz", "stage_group_hidden_audit.npz"),
        "cross_seed_500k_group": ("cross_seed_500k_public_pilot.npz", "cross_seed_500k_hidden_audit.npz"),
    }
    for group_name, (public_name, hidden_name) in artifact_names.items():
        _, _, public, hidden = group_bundles[group_name]
        if not public_artifact_is_leakage_free(public):
            raise RuntimeError(f"hidden field leaked into {public_name}")
        np.savez_compressed(output_dir / public_name, **public)
        np.savez_compressed(
            output_dir / hidden_name, **hidden,
            AUDIT_ONLY_DO_NOT_USE_FOR_TRAINING=np.asarray(True),
        )
    (output_dir / "source_group_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _print_results(groups, structures, changes)
    print("\nFact flags")
    for name, value in flags.items():
        print(f"{name}={value}")
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-dir", type=Path, default=Path("artifacts/hopper_behavior_policies/seed_0"))
    parser.add_argument("--seed1-dir", type=Path, default=Path("artifacts/hopper_behavior_policies/seed_1"))
    parser.add_argument("--seed2-dir", type=Path, default=Path("artifacts/hopper_behavior_policies/seed_2"))
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--pilot-transitions-per-policy", type=int, default=10_000)
    parser.add_argument("--paired-audit-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hopper_behavior_source_audit/group_comparison"))
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.eval_episodes <= 0 or arguments.pilot_transitions_per_policy < 2:
        parser.error("evaluation episodes must be positive and pilot count must be at least two")
    if not 0 < arguments.paired_audit_samples <= 3 * arguments.pilot_transitions_per_policy:
        parser.error("paired audit samples must be positive and fit within each group pilot")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
    print("PHASE6C_CROSS_SEED_AUDIT_COMPLETE")
