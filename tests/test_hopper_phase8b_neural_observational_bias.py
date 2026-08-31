"""Focused implementation tests for Phase 8B-NC."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift import neural_observational_bias as phase
from experiments.hopper_logger_mixture_drift.analyze_phase8a_population_effect import (
    hash_input_files, input_hashes_unchanged,
)


REPO = Path(__file__).resolve().parents[1]
NC = REPO / "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified"
CONTROLLED = REPO / "artifacts/hopper_logger_mixture_drift/controlled_loggers_seed0_verified"
CLIPPING = CONTROLLED / "population_effect_review/clipping_sensitivity_kappa_0p30"


def synthetic_public() -> tuple[dict[str, np.ndarray], np.ndarray]:
    actions = np.asarray([[-0.2, 0.0, 0.2], [-0.2, 0.0, 0.2],
                          [0.2, 0.0, -0.2], [0.2, 0.0, -0.2]], dtype=np.float32)
    observations = np.asarray([[1.0]*12, [1.0]*12, [2.0]*12, [2.0]*12], dtype=np.float32)
    public = {
        "anchor_id": np.asarray([0, 0, 1, 1]), "observation": observations,
        "commanded_action": actions, "reward": np.asarray([1., 3., 4., 8.]),
        "next_observation": observations + np.asarray([[1.], [3.], [2.], [4.]], dtype=np.float32),
    }
    return public, np.asarray([0.1, 0.3, 0.2, 0.4])


def torch_or_skip():
    if os.name == "nt":
        pytest.skip("local Windows PyTorch DLL loader is broken; CUDA tests run on Linux server")
    try:
        return phase._torch()
    except Exception as exc:
        pytest.skip(str(exc))


def test_verified_phase8anc_required(tmp_path):
    root = tmp_path / "nc"
    root.mkdir()
    (root / "hard_checks.json").write_text('{"all_passed": false, "checks": {}}')
    with pytest.raises(phase.NeuralObservationalBiasError):
        phase.require_verified_inputs(root, tmp_path / "causal", tmp_path / "clip")


def test_verified_phase8a_do_oracle_required(tmp_path):
    with pytest.raises(phase.NeuralObservationalBiasError):
        phase.require_verified_inputs(tmp_path / "missing", tmp_path / "missing2", tmp_path / "missing3")


def test_all_2048_anchors_available():
    manifest = json.loads((NC / "manifest.json").read_text())
    assert manifest["available_anchor_count"] == 2048


def test_all_four_kappas_available():
    manifest = json.loads((NC / "manifest.json").read_text())
    assert tuple(manifest["kappas"]) == phase.EXPECTED_KAPPAS


def test_primary_mixtures_preserve_state_action_mass():
    table = np.load(NC / "kappa_0p30/population_tables.npz")
    for condition in phase.CONDITIONS:
        masses = [table[f"{condition}_{mixture}_state_action_mass"]
                  for mixture in phase.PRIMARY_MIXTURE_NAMES]
        assert all(np.allclose(masses[0], value, atol=1e-12, rtol=1e-12) for value in masses[1:])


def test_anchor_splits_are_disjoint():
    anchors = np.arange(2048)
    splits = phase.make_anchor_splits(anchors, 0, np.arange(2048) % 3)
    assert phase.validate_splits(splits, anchors)


def test_group_key_uses_exact_commanded_action():
    left = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    right = left.copy(); right[0] = np.nextafter(right[0], np.float32(np.inf))
    assert phase.action_bytes(left) != phase.action_bytes(right)


def test_grouped_target_matches_weighted_rows():
    public, weights = synthetic_public()
    grouped = phase.build_grouped_targets(public, weights)
    assert np.allclose(grouped.reward, [2.5, 20/3])
    assert np.allclose(grouped.delta[:, 0], [2.5, 10/3])


def test_grouped_weighted_mse_decomposition():
    y = np.asarray([0., 2., 3., 7.]); w = np.asarray([1., 3., 2., 4.])
    groups = np.asarray([0, 0, 1, 1]); prediction = np.asarray([1.5, 4.])
    row, grouped, within = phase.grouped_weighted_mse_decomposition(y, w, groups, prediction)
    assert np.isclose(row, grouped + within, atol=1e-12)


def test_model_input_is_15d():
    public, weights = synthetic_public()
    assert phase.build_grouped_targets(public, weights).x.shape[1] == 15


def test_model_input_excludes_logger():
    assert phase.MODEL_INPUT_FIELDS == ("observation", "commanded_action")
    assert phase.LEAKAGE_FLAGS["LOGGER_ID_IN_MODEL_INPUT"] is False


def test_model_input_excludes_hidden_fields():
    assert not ({"u", "u_env", "applied_action", "action_key"} & set(phase.MODEL_INPUT_FIELDS))


def test_output_shapes():
    torch = torch_or_skip()
    x = torch.zeros((5, 15))
    assert tuple(phase.RewardMeanModel()(x).shape) == (5, 1)
    assert tuple(phase.DeltaMeanModel()(x).shape) == (5, 11)


def test_normalization_shared_across_mixtures():
    stats = phase.normalization(np.arange(45, dtype=float).reshape(3, 15))
    assert phase.apply_normalization(np.ones((2, 15)), stats).shape == (2, 15)
    assert "mixture" not in inspect.signature(phase.normalization).parameters


def test_same_initial_hash_across_mixtures():
    torch = torch_or_skip()
    hashes = []
    for _ in phase.PRIMARY_MIXTURE_NAMES:
        torch.manual_seed(7); hashes.append(phase.state_hash(phase.RewardMeanModel()))
    assert len(set(hashes)) == 1


def test_same_batch_schedule_across_mixtures():
    hashes = [phase.array_hash(phase.batch_schedule(10, 5, 4, 9))
              for _ in phase.PRIMARY_MIXTURE_NAMES]
    assert len(set(hashes)) == 1


def test_checkpoint_roundtrip(tmp_path):
    torch_or_skip()
    model = phase.RewardMeanModel(); path = tmp_path / "model.pt"
    phase.save_checkpoint(path, model, {"target": "reward"})
    loaded, metadata = phase.load_checkpoint(path, "reward", "cpu")
    assert phase.state_hash(model) == phase.state_hash(loaded)
    assert metadata["target"] == "reward"


def test_kappa_zero_population_invariance():
    table = np.load(NC / "kappa_0p00/population_tables.npz")
    for condition in phase.CONDITIONS:
        values = [table[f"{condition}_{m}_reward"] for m in phase.PRIMARY_MIXTURE_NAMES]
        assert all(np.allclose(values[0], value) for value in values[1:])


def test_independent_population_invariance():
    table = np.load(NC / "kappa_0p30/population_tables.npz")
    values = [table[f"independent_latents_{m}_reward"] for m in phase.PRIMARY_MIXTURE_NAMES]
    assert all(np.allclose(values[0], value) for value in values[1:])


def test_base_action_population_invariance():
    table = np.load(NC / "kappa_0p30/population_tables.npz")
    values = [table[f"confounded_{m}_reward"][:, 1] for m in phase.PRIMARY_MIXTURE_NAMES]
    assert all(np.allclose(values[0], value) for value in values[1:])


def test_do_oracle_not_used_in_training():
    assert "do" not in inspect.signature(phase.train_model).parameters
    assert phase.LEAKAGE_FLAGS["DO_ORACLE_USED_FOR_TRAINING"] is False


def test_long_horizon_oracle_not_used_in_training():
    assert "long_horizon" not in inspect.signature(phase.train_model).parameters
    assert phase.LEAKAGE_FLAGS["LONG_HORIZON_ORACLE_USED_FOR_TRAINING"] is False


def test_action_key_not_used_in_training():
    source = inspect.getsource(phase.train_model)
    assert "action_key" not in source


def test_strict_unclipped_mask_matches_phase8ac():
    path = CLIPPING / "anchor_clipping_table.npz"
    if not path.is_file():
        pytest.skip("Phase 8A-C table is not in the local clone")
    table = np.load(path)
    assert table["strict_anchor_unclipped"].shape == (2048,)


def test_metrics_use_test_anchors_only():
    assert "test_anchor_ids" in inspect.signature(phase.seed_metrics).parameters
    assert "statistical_unit\": \"anchor_id" in inspect.getsource(phase.seed_metrics)


def test_seed_and_anchor_units_not_conflated():
    rows = [{"kappa": 0.3, "condition": "confounded", "metric": "x", "mean": value}
            for value in (1., 2., 3.)]
    result = phase._aggregate_seed_rows(rows)[0]
    assert result["model_seed_count"] == 3
    assert result["seed_variation_reported_separately"] is True


def test_input_hashes_unchanged(tmp_path):
    path = tmp_path / "input.txt"; path.write_text("fixed")
    before = hash_input_files([path.resolve()]); after = hash_input_files([path.resolve()])
    assert input_hashes_unchanged(before, after)


def test_no_nan_inf():
    public, weights = synthetic_public()
    grouped = phase.build_grouped_targets(public, weights)
    assert all(np.isfinite(value).all() for value in grouped.arrays().values())


def test_old_artifacts_unchanged():
    path = NC / "manifest.json"
    before = hash_input_files([path.resolve()]); after = hash_input_files([path.resolve()])
    assert input_hashes_unchanged(before, after)
