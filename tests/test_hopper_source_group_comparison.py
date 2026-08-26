"""Unit tests for the Phase 6C cross-seed Hopper source comparison."""

from pathlib import Path

import pytest

from scripts.compare_hopper_source_groups import (
    CROSS_SEED_LABELS,
    FORMAL_SOURCE_MAPPING,
    STAGE_LABELS,
    action_complementarity_ratio,
    find_unique_checkpoint_by_step,
    public_artifact_is_leakage_free,
    resolve_checkpoint_groups,
    state_overlap_ratio_change,
    summarize_group_structure,
)
from scripts.diagnose_hopper_behavior_sources import SOURCE_STEPS


def _checkpoint_directories(tmp_path: Path) -> tuple[Path, Path, Path]:
    directories = tuple(tmp_path / f"seed_{seed}" for seed in range(3))
    for directory in directories:
        directory.mkdir()
    for step in (200_000, 500_000, 1_000_000):
        (directories[0] / f"source_x_step_{step}.zip").touch()
    (directories[1] / "source_1_step_500000.zip").touch()
    (directories[2] / "anything_step_500000.zip").touch()
    return directories


def test_checkpoint_is_located_uniquely_by_filename_step(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    expected = directory / "arbitrary_label_step_500000.zip"
    expected.touch()
    (directory / "arbitrary_label_step_200000.zip").touch()
    assert find_unique_checkpoint_by_step(directory, 500_000) == expected


def test_missing_checkpoint_step_fails_explicitly(tmp_path):
    with pytest.raises(RuntimeError, match="found 0"):
        find_unique_checkpoint_by_step(tmp_path, 500_000)


def test_duplicate_checkpoint_step_fails_explicitly(tmp_path):
    (tmp_path / "first_step_500000.zip").touch()
    (tmp_path / "second_step_500000.zip").touch()
    with pytest.raises(RuntimeError, match="found 2"):
        find_unique_checkpoint_by_step(tmp_path, 500_000)


def test_stage_group_labels_map_to_seed0_training_stages(tmp_path):
    seed0, seed1, seed2 = _checkpoint_directories(tmp_path)
    group = resolve_checkpoint_groups(seed0, seed1, seed2)["stage_group"]
    assert tuple(group) == STAGE_LABELS
    assert [path.name for path in group.values()] == [
        "source_x_step_200000.zip", "source_x_step_500000.zip",
        "source_x_step_1000000.zip",
    ]


def test_cross_seed_group_labels_map_to_three_500k_checkpoints(tmp_path):
    seed0, seed1, seed2 = _checkpoint_directories(tmp_path)
    group = resolve_checkpoint_groups(seed0, seed1, seed2)["cross_seed_500k_group"]
    assert tuple(group) == CROSS_SEED_LABELS
    assert all("step_500000.zip" in path.name for path in group.values())
    assert [path.parent for path in group.values()] == [seed0, seed1, seed2]


def test_formal_phase6a_source_mapping_is_unchanged():
    expected = {"source_1": 200_000, "source_2": 500_000, "source_3": 1_000_000}
    assert FORMAL_SOURCE_MAPPING == expected
    assert SOURCE_STEPS == expected
    assert not any(label.startswith("source_") for label in CROSS_SEED_LABELS)


def test_state_overlap_change_is_cross_seed_minus_stage():
    assert state_overlap_ratio_change(4.25, 2.75) == pytest.approx(-1.5)
    assert state_overlap_ratio_change(2.0, 3.0) == pytest.approx(1.0)


def test_action_complementarity_ratio_is_cross_over_within():
    assert action_complementarity_ratio(0.9, 0.6) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="positive"):
        action_complementarity_ratio(0.9, 0.0)


def test_public_artifact_hidden_key_audit():
    public_fields = {
        "observation", "action", "reward", "next_observation", "terminated",
        "truncated", "source_id", "episode_id", "time_step",
    }
    assert public_artifact_is_leakage_free(public_fields)
    for hidden in ("hidden_u", "applied_action", "qpos", "qvel"):
        assert not public_artifact_is_leakage_free(public_fields | {hidden})


def test_group_structure_aggregation_accepts_non_source_labels():
    labels = CROSS_SEED_LABELS
    within = {
        label: {
            "state_distance": {"median": 1.0},
            "matched_action_distance": {"mean": 0.5, "median": 0.4},
        }
        for label in labels
    }
    cross = {
        f"{first}_to_{second}": {
            "matched_state_distance": {"median": 2.0},
            "cross_over_within_state_distance": {"mean": 2.0, "median": 1.5},
            "matched_action_distance": {"mean": 1.0, "median": 0.8},
        }
        for first in labels for second in labels if first != second
    }
    report = {
        "state_coverage_and_action_complementarity": {
            name: {
                "status": "AVAILABLE", "within": within, "directed_cross": cross,
                "nearest_neighbor_has_different_source": 0.25,
            }
            for name in ("physical_11d", "public_12d")
        }
    }
    structure = summarize_group_structure(report)["public_12d"]
    assert structure["action_complementarity_ratio_mean"] == pytest.approx(2.0)
    assert structure["action_complementarity_ratio_median"] == pytest.approx(2.0)
