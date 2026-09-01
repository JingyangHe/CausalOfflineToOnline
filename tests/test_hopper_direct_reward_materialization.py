from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift import direct_reward_materialization as phase
from experiments.hopper_logger_mixture_drift.reward_signal_calibration import (
    FORBIDDEN_DERIVED_PUBLIC_FIELDS,
)


def _pair(anchor_count: int = 3, kappa: float = 0.0,
          condition: str = "confounded") -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = anchor_count * 3
    anchors = np.repeat(np.arange(anchor_count), 3)
    row_ids = np.arange(rows)
    public = {
        "row_id": row_ids,
        "anchor_id": anchors,
        "observation": np.zeros((rows, 12), dtype=np.float32),
        "commanded_action": np.tile(np.asarray([[-.2, 0., 0.], [0., 0., 0.], [.2, 0., 0.]],
                                                   dtype=np.float32), (anchor_count, 1)),
        "reward": np.linspace(0.0, 1.0, rows),
        "next_observation": np.ones((rows, 12), dtype=np.float32),
        "terminated": np.zeros(rows, dtype=bool),
        "truncated": np.zeros(rows, dtype=bool),
        "logger_id": np.tile(np.arange(3, dtype=np.int8), anchor_count),
        "condition": np.full(rows, condition),
        "kappa_env": np.full(rows, kappa),
    }
    hidden = {
        "row_id": row_ids,
        "u_env": np.tile(np.asarray([-1, 1, -1], dtype=np.int8), anchor_count),
        "u_behavior": np.tile(np.asarray([-1, 1, -1], dtype=np.int8), anchor_count),
        "action_key": np.tile(np.asarray(["minus", "base", "plus"]), anchor_count),
        "logger_id": public["logger_id"].copy(),
    }
    return public, hidden


def test_materialize_scenario_uses_exact_formula():
    public, hidden = _pair()
    derived, audit = phase.materialize_scenario(public, hidden, 0.005)
    expected = public["reward"] + 0.005 * hidden["u_env"]
    assert np.array_equal(derived["reward"], expected)
    assert np.array_equal(audit["reward_bonus"], 0.005 * hidden["u_env"])


def test_public_artifact_has_no_hidden_fields():
    public, hidden = _pair()
    derived, _ = phase.materialize_scenario(public, hidden, 0.01)
    assert not FORBIDDEN_DERIVED_PUBLIC_FIELDS.intersection(derived)
    assert "u_env" not in derived and "original_reward" not in derived


def test_hidden_audit_stays_row_aligned():
    public, hidden = _pair()
    derived, audit = phase.materialize_scenario(public, hidden, 0.02)
    assert np.array_equal(derived["row_id"], audit["row_id"])
    assert np.array_equal(audit["u_env"], hidden["u_env"])


def test_lambda_zero_is_bit_exact():
    public, hidden = _pair()
    derived, _ = phase.materialize_scenario(public, hidden, 0.0)
    assert np.array_equal(derived["reward"], public["reward"])


def test_source_pair_rejects_misalignment():
    public, hidden = _pair()
    hidden["row_id"] = hidden["row_id"][::-1]
    with pytest.raises(phase.DirectRewardMaterializationError, match="not aligned"):
        phase.validate_source_pair(public, hidden, 0.0, "confounded")


def test_source_pair_requires_2048_anchors():
    public, hidden = _pair()
    with pytest.raises(phase.DirectRewardMaterializationError, match="2048 anchors"):
        phase.validate_source_pair(public, hidden, 0.0, "confounded")


def test_formal_grid_is_not_selected_in_materializer():
    source = inspect.getsource(phase.materialize_frozen_direct_reward_grid)
    assert "load_frozen_lambda_grid" in source
    assert "quantile" not in source and "threshold" not in source


def test_cli_has_block_and_completion_markers():
    text = (ROOT / "scripts/materialize_hopper_phase8c_direct_reward_grid.py").read_text(
        encoding="utf-8")
    assert "PHASE8C_DIRECT_REWARD_MATERIALIZATION_BLOCKED" in text
    assert "PHASE8C_DIRECT_REWARD_MATERIALIZATION_COMPLETE" in text
    assert "PHASE8C_FORMAL_INPUTS_READY" in text

