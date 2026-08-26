"""Unit tests for the Phase 6D local Joint-support audit."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.compare_hopper_source_groups import CROSS_SEED_LABELS, STAGE_LABELS
from scripts.diagnose_hopper_local_joint_support import (
    _validate_labels,
    find_public_pilot,
    kth_neighbor_radii,
    load_public_pilot,
    normalized_action_separation,
    run,
    self_source_radii,
    source3_action_novelty,
    support_geometry,
)


def test_kth_nearest_neighbor_radius_on_toy_states():
    radii, _ = kth_neighbor_radii(
        np.array(((0.0,),)), np.array(((0.0,), (1.0,), (2.0,), (4.0,))),
        np.array((1, 2, 4)),
    )
    np.testing.assert_allclose(radii, ((0.0, 1.0, 4.0),))


def test_all_source_radius_is_maximum_source_radius():
    source = np.array([[[1.0, 2.0], [3.0, 4.0], [2.0, 5.0]]])
    result = support_geometry(source, np.ones((1, 2)))
    np.testing.assert_array_equal(result["all_source_radii"], ((3.0, 5.0),))


def test_best_two_radius_is_second_smallest_source_radius():
    source = np.array([[[1.0, 8.0], [4.0, 2.0], [3.0, 5.0]]])
    result = support_geometry(source, np.ones((1, 2)))
    np.testing.assert_array_equal(result["best_two_radii"], ((3.0, 5.0),))


def test_self_source_radius_excludes_query_row():
    radii = self_source_radii(
        queries=np.array(((1.0,),)),
        query_global_indices=np.array((11,)),
        query_source_ids=np.array((1,)),
        reference_states={1: np.array(((0.0,), (1.0,), (2.0,)))},
        reference_global_indices={1: np.array((10, 11, 12))},
        k_grid=np.array((1, 2)),
    )
    np.testing.assert_allclose(radii, ((1.0, 1.0),))


def test_normalized_action_separation_uses_within_source_variation():
    raw, normalized = normalized_action_separation(
        np.array(((0.0, 0.0),)), np.array(((2.0, 0.0),)),
        np.array((1.0,)), np.array((3.0,)),
    )
    np.testing.assert_allclose(raw, (2.0,))
    np.testing.assert_allclose(normalized, (1.0,), atol=1e-12)


def test_source3_novelty_uses_closer_of_first_two_action_means():
    means = np.array([[[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]]])
    np.testing.assert_allclose(source3_action_novelty(means), ((1.0,),))


def test_third_source_penalty_and_ratio_are_correct():
    source = np.array([[[1.0], [3.0], [5.0]]])
    result = support_geometry(source, np.ones((1, 1)))
    np.testing.assert_allclose(result["third_source_penalties"], ((2.0,),))
    np.testing.assert_allclose(result["third_source_ratios"], ((5.0 / 3.0,),))


def test_public_path_discovery_never_selects_hidden_audit(tmp_path):
    public = tmp_path / "stage_alternate_public_pilot_data.npz"
    hidden = tmp_path / "stage_group_hidden_audit.npz"
    np.savez(public, placeholder=np.array(1))
    np.savez(hidden, hidden_u=np.array((1,)))
    assert find_public_pilot(tmp_path, "stage_group") == public
    with pytest.raises(RuntimeError, match="invalid fields"):
        load_public_pilot(hidden)


def test_stage_and_cross_seed_label_mapping_is_fixed():
    summary = {
        "groups": {
            "stage_group": {"checkpoint_labels": list(STAGE_LABELS)},
            "cross_seed_500k_group": {"checkpoint_labels": list(CROSS_SEED_LABELS)},
        }
    }
    _validate_labels(summary)
    summary["groups"]["cross_seed_500k_group"]["checkpoint_labels"][0] = "source_1"
    with pytest.raises(RuntimeError, match="unexpected"):
        _validate_labels(summary)


def test_toy_public_artifacts_run_end_to_end_without_hidden_data(tmp_path):
    source_dir, output_dir = tmp_path / "group", tmp_path / "output"
    source_dir.mkdir()
    rng = np.random.default_rng(7)
    count_per_source = 300
    source_id = np.repeat((1, 2, 3), count_per_source)
    count = source_id.size
    public = {
        "observation": rng.normal(size=(count, 12)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(count, 3)).astype(np.float32),
        "reward": rng.normal(size=count),
        "next_observation": rng.normal(size=(count, 12)).astype(np.float32),
        "terminated": np.zeros(count, dtype=bool),
        "truncated": np.zeros(count, dtype=bool),
        "source_id": source_id,
        "episode_id": np.zeros(count, dtype=np.int64),
        "time_step": np.arange(count),
    }
    np.savez(source_dir / "stage_group_public_pilot.npz", **public)
    np.savez(source_dir / "cross_seed_500k_public_pilot.npz", **public)
    group_structure = {
        group: {
            representation: {"different_source_nearest_neighbor_fraction": 0.1}
            for representation in ("physical_11d", "public_12d")
        }
        for group in ("stage_group", "cross_seed_500k_group")
    }
    phase6c = {
        "groups": {
            "stage_group": {"checkpoint_labels": list(STAGE_LABELS)},
            "cross_seed_500k_group": {"checkpoint_labels": list(CROSS_SEED_LABELS)},
        },
        "group_structure": group_structure,
    }
    (source_dir / "source_group_comparison_summary.json").write_text(json.dumps(phase6c))
    summary = run(SimpleNamespace(
        group_comparison_dir=source_dir, query_count=9, seed=2026,
        output_dir=output_dir,
    ))
    assert summary["hidden_audit_loaded"] is False
    assert summary["ACTUAL_JOINT_TIGHTENING"] == "NOT_EVALUATED"
    with np.load(output_dir / "local_joint_support_audit.npz") as artifact:
        assert "hidden_u" not in artifact.files
        assert artifact["k_grid"].tolist() == [1, 4, 16, 64, 256]
