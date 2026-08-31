"""Focused invariants for Phase 8B-RS reward-signal calibration."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift import reward_signal_calibration as phase
from experiments.hopper_logger_mixture_drift.neural_observational_bias import (
    MODEL_INPUT_DIMENSION,
    MODEL_INPUT_FIELDS,
    RewardMeanModel,
    expected_parameter_count,
    load_checkpoint,
    make_anchor_splits,
    make_initial_state,
    parameter_count,
    save_checkpoint,
    validate_splits,
)


def _rows(n: int = 4) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    public = {
        "row_id": np.arange(n), "anchor_id": np.arange(n),
        "observation": np.zeros((n, 12), dtype=np.float32),
        "commanded_action": np.zeros((n, 3), dtype=np.float32),
        "reward": np.linspace(1.0, 2.0, n),
        "next_observation": np.ones((n, 12), dtype=np.float32),
        "terminated": np.zeros(n, dtype=bool), "truncated": np.zeros(n, dtype=bool),
        "logger_id": np.asarray([0, 1, 2, 0], dtype=np.int8)[:n],
        "condition": np.asarray(["confounded"] * n),
        "kappa_env": np.zeros(n),
    }
    hidden = {
        "row_id": np.arange(n), "u_env": np.asarray([-1, 1, -1, 1], dtype=np.int8)[:n],
        "u_behavior": np.asarray([-1, 1, 1, -1], dtype=np.int8)[:n],
        "action_key": np.asarray(["minus", "plus", "base", "minus"])[:n],
        "logger_id": public["logger_id"],
    }
    return public, hidden


def _torch_or_skip():
    if sys.platform == "win32":
        pytest.skip("local Windows PyTorch DLL loading is unavailable; exercised on Linux server")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"PyTorch unavailable in this environment: {exc}")
    return torch


def test_verified_phase8anc_required(tmp_path: Path):
    with pytest.raises(phase.RewardSignalCalibrationError, match="Phase 8A-NC"):
        phase.require_verified_inputs(tmp_path / "missing", tmp_path / "causal")


def test_verified_phase8a_oracle_required(tmp_path: Path):
    nc = tmp_path / "nc"
    nc.mkdir()
    with pytest.raises(phase.RewardSignalCalibrationError, match="Phase 8A"):
        phase.require_verified_inputs(nc, tmp_path / "missing")


def test_reward_grid_exact():
    assert phase.REWARD_STRENGTHS == (0.0, 0.05, 0.10, 0.20)
    assert phase.validate_reward_strengths((0.0, 0.1, 0.2)) == (0.0, 0.1, 0.2)
    with pytest.raises(ValueError):
        phase.validate_reward_strengths((0.0, 0.15))


def test_augmented_reward_uses_u_env():
    reward, bonus = phase.augment_reward(np.array([1.0, 1.0]), np.array([-1, 1]), 0.2)
    assert np.allclose(reward, [0.8, 1.2])
    assert np.allclose(bonus, [-0.2, 0.2])


def test_do_mean_invariant_to_lambda():
    plus, _ = phase.augment_reward(np.array([2.0]), np.array([1]), 0.2)
    minus, _ = phase.augment_reward(np.array([1.0]), np.array([-1]), 0.2)
    assert np.isclose(0.5 * (plus[0] + minus[0]), 1.5)


def test_kappa_zero_balanced_bias_identity():
    for strength in phase.REWARD_STRENGTHS:
        assert np.isclose(phase.theoretical_balanced_direct_bias("plus", strength), 0.6 * strength)
        assert np.isclose(phase.theoretical_balanced_direct_bias("minus", strength), -0.6 * strength)


def test_kappa_zero_heavy_drift_identity():
    for strength in phase.REWARD_STRENGTHS:
        assert np.isclose(phase.theoretical_heavy_direct_drift("plus", strength), 14/45*strength)
        assert np.isclose(phase.theoretical_heavy_direct_drift("minus", strength), -14/45*strength)


def test_kappa_point3_additive_bias_identity():
    original = np.asarray([0.2, -0.3])
    direct = phase.theoretical_balanced_direct_bias("plus", 0.2)
    assert np.allclose(original + direct, original + 0.12)


def test_independent_direct_bias_zero():
    for mixture in phase.PRIMARY_MIXTURE_NAMES:
        for action in phase.ACTION_KEYS:
            assert phase.direct_reward_mean("independent_latents", mixture, action, 0.2) == 0.0


def test_base_direct_bias_zero():
    for mixture in phase.PRIMARY_MIXTURE_NAMES:
        assert np.isclose(phase.direct_reward_mean("confounded", mixture, "base", 0.2), 0.0)


def test_primary_state_action_mass_unchanged():
    public, hidden = _rows()
    left, _ = phase.make_derived_artifacts(public, hidden, 0.0)
    right, _ = phase.make_derived_artifacts(public, hidden, 0.2)
    for field in ("row_id", "anchor_id", "observation", "commanded_action", "logger_id"):
        assert np.array_equal(left[field], right[field])


def test_public_hidden_leakage_empty():
    public, hidden = phase.make_derived_artifacts(*_rows(), 0.2)
    assert phase.validate_derived_artifacts(public, hidden) == set()


def test_original_and_augmented_reward_not_both_public():
    public, _ = phase.make_derived_artifacts(*_rows(), 0.2)
    assert "reward" in public and "original_reward" not in public
    assert not phase.FORBIDDEN_DERIVED_PUBLIC_FIELDS.intersection(public)


def test_all_512_pilot_anchors_available():
    source = inspect.getsource(phase.run_reward_signal_calibration)
    assert "num_anchors: int = 512" in source
    assert "2048" in source


def test_split_reused_or_fixed():
    anchors = np.arange(512)
    splits = make_anchor_splits(anchors, 0)
    assert validate_splits(splits, anchors)
    assert all(splits[name] for name in ("train", "validation", "test"))


def test_reward_model_input_15d():
    assert MODEL_INPUT_DIMENSION == 15
    assert MODEL_INPUT_FIELDS == ("observation", "commanded_action")


def test_model_excludes_lambda():
    assert "lambda_reward" not in MODEL_INPUT_FIELDS


def test_model_excludes_kappa():
    assert "kappa_env" not in MODEL_INPUT_FIELDS


def test_model_excludes_logger():
    assert "logger_id" not in MODEL_INPUT_FIELDS


def test_model_excludes_hidden_u():
    assert "u_env" not in MODEL_INPUT_FIELDS


def test_output_normalization_shared_across_lambda():
    source = inspect.getsource(phase.run_reward_signal_calibration)
    assert "output_stats[kappa]" in source
    assert "balanced_zero" in source
    assert "strength" not in inspect.signature(phase._train_reward_model).parameters


def test_same_initial_hash_across_lambda():
    source = inspect.getsource(phase.run_reward_signal_calibration)
    assert source.index("make_initial_state") < source.index("for strength in strengths", source.index("make_initial_state"))
    assert "set(observed_initial) == {initial_digest}" in source


def test_same_batch_schedule_across_lambda():
    source = inspect.getsource(phase.run_reward_signal_calibration)
    assert source.index("schedule = batch_schedule") < source.index(
        "for strength in strengths", source.index("schedule = batch_schedule"))
    assert "set(observed_schedule) == {schedule_digest}" in source


def test_best_checkpoint_roundtrip(tmp_path: Path):
    torch = _torch_or_skip()
    model = RewardMeanModel(256)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, {"target": "reward", "hidden_width": 256})
    loaded, metadata = load_checkpoint(path, "reward", "cpu", 256)
    assert metadata["target"] == "reward"
    for left, right in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(left, right)


def test_lambda_zero_crosscheck_phase8b():
    public, hidden = _rows()
    derived, audit = phase.make_derived_artifacts(public, hidden, 0.0)
    assert np.array_equal(derived["reward"], public["reward"])
    assert np.array_equal(audit["original_reward"], audit["augmented_reward"])


def test_paired_increment_alignment():
    lambdas = np.asarray(phase.REWARD_STRENGTHS)
    values = 1.7 + lambdas[:, None] * np.asarray([[0.6, -0.6]])
    slope, origin, r2 = phase.fit_lambda_slope(lambdas, values)
    assert np.allclose(slope, [0.6, -0.6])
    assert np.allclose(origin, slope)
    assert np.allclose(r2, 1.0)


def test_no_nan_inf():
    for condition in phase.CONDITIONS:
        for mixture in phase.PRIMARY_MIXTURE_NAMES:
            for action in phase.ACTION_KEYS:
                values = [phase.direct_reward_mean(condition, mixture, action, strength)
                          for strength in phase.REWARD_STRENGTHS]
                assert np.isfinite(values).all()


def test_input_hashes_unchanged(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("read only", encoding="utf-8")
    before = phase.hash_input_files([path.resolve()])
    after = phase.hash_input_files([path.resolve()])
    assert phase.input_hashes_unchanged(before, after)


def test_old_artifacts_unchanged():
    source = inspect.getsource(phase.run_reward_signal_calibration)
    assert 'hard_checks["old_artifacts_unchanged"] = unchanged' in source
    assert "hashes_before" in source and "hashes_after" in source


def test_width_256_reward_model_fixed():
    _torch_or_skip()
    model = RewardMeanModel(256)
    assert parameter_count(model) == expected_parameter_count(1, 256) == 135937


def test_initial_state_is_reproducible():
    _torch_or_skip()
    _, first = make_initial_state("reward", 17, 256)
    _, second = make_initial_state("reward", 17, 256)
    assert first == second


def test_cli_completion_markers_present():
    path = Path(__file__).parents[1] / "scripts" / "run_hopper_reward_signal_calibration.py"
    text = path.read_text(encoding="utf-8")
    assert "PHASE8B_REWARD_SIGNAL_CALIBRATION_COMPLETE" in text
    assert "READY_FOR_REWARD_SIGNAL_LEARNABILITY_REVIEW" in text
    assert "PHASE8B_REWARD_SIGNAL_CALIBRATION_BLOCKED" in text
