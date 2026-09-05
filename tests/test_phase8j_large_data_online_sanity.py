from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift.phase8j_large_data_online_sanity import (
    BACKUP_METHODS,
    FORMAL_EVAL_STEPS,
    MODEL_SEEDS,
    ONLINE_METHODS,
    POTENTIAL_METHODS,
    SAMPLES_PER_ANCHOR_SOURCE,
    TRANSITION_COUNT,
    _first_reach_step,
    _replay_training_snapshot,
    audit_rematerialized_dataset,
    common_return_level_metrics,
    run_online,
)


def test_frozen_phase8j_contract() -> None:
    assert SAMPLES_PER_ANCHOR_SOURCE == 128
    assert TRANSITION_COUNT == 196_608
    assert MODEL_SEEDS == (0, 1, 2)
    assert POTENTIAL_METHODS == ("pooled", "state_min", "action_min")
    assert len(ONLINE_METHODS) == 4
    assert FORMAL_EVAL_STEPS == (0, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000)
    assert BACKUP_METHODS == {
        "pooled": "pooled_aamas_union_full",
        "state_min": "state_min_full",
        "action_min": "action_min_full",
    }


def test_rematerialized_dataset_audit_rejects_wrong_frozen_count() -> None:
    count = 2 * 3 * 128
    anchor = np.repeat(np.arange(2), 3 * 128)
    source = np.tile(np.repeat(np.arange(1, 4), 128), 2)
    sample = np.tile(np.arange(128), 2 * 3)
    public = {
        "observation": np.zeros((count, 12), dtype=np.float32),
        "commanded_action": np.zeros((count, 3), dtype=np.float32),
        "reward": np.zeros(count, dtype=np.float32),
        "next_observation": np.zeros((count, 12), dtype=np.float32),
        "terminated": np.zeros(count, dtype=bool),
        "truncated": np.zeros(count, dtype=bool),
        "anchor_id": anchor,
        "source_id": source.astype(np.int8),
        "sample_id": sample.astype(np.int16),
    }
    hidden = {"u_behavior": np.ones(count), "u_environment": np.ones(count)}
    anchors = {"anchor_id": np.arange(2)}
    generation = {"original_d32_exact": True, "extension_protocol": "fixed"}
    with pytest.raises(Exception, match="expected 196608"):
        audit_rematerialized_dataset(public, hidden, anchors, generation)


def test_first_reach_interpolates_without_fixed_threshold() -> None:
    steps = np.asarray([0, 5_000, 10_000], dtype=float)
    returns = np.asarray([0.0, 10.0, 30.0])
    assert _first_reach_step(steps, returns, 20.0) == pytest.approx(7_500)
    assert _first_reach_step(steps, returns, 40.0) is None


def test_step_zero_training_snapshot_does_not_require_sb3_logger() -> None:
    class EmptyBuffer:
        @staticmethod
        def size() -> int:
            return 0

    class Model:
        replay_buffer = EmptyBuffer()

    snapshot = _replay_training_snapshot(Model(), None)
    assert snapshot == {
        "critic_loss": "", "actor_loss": "", "entropy_coefficient": "",
        "entropy": "", "q_value_abs_mean": "",
    }


def test_common_levels_are_observed_and_shared() -> None:
    methods = ("a", "b")
    steps = np.asarray([0.0, 1.0, 2.0])
    curves = {
        (0, "a"): (steps, np.asarray([0.0, 2.0, 4.0])),
        (0, "b"): (steps, np.asarray([1.0, 3.0, 5.0])),
    }
    rows = common_return_level_metrics(curves, (0,), methods)
    assert {row["return_level"] for row in rows} == {1.0, 2.0, 3.0, 4.0}
    assert all(row["steps_to_reach"] is not None for row in rows)


def test_online_source_declares_isolation_and_secondary_metrics() -> None:
    source = inspect.getsource(run_online)
    for text in ("replay_initially_empty", "raw_environment_reward",
                 "critic_loss", "actor_loss", "entropy", "q_value_abs_mean",
                 "potential_frozen", "progress_bar=False"):
        assert text in source
