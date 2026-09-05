from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift import phase8j_fvi_stability as fvi
from scripts.run_phase8j_fvi_stability import parse_arguments


def test_frozen_protocol_constants() -> None:
    assert fvi.METHOD == "pooled_union"
    assert fvi.BACKUP_METHOD == "pooled_aamas_union_full"
    assert fvi.MODEL_SEED == 0
    assert fvi.SAMPLES_PER_ANCHOR_SOURCE == 128
    assert fvi.COMPONENT_UPDATES == 4000
    assert fvi.VARIANTS == ("matched_moving_target", "frozen_outer_target")
    assert len(fvi.DIAGNOSTIC_EPOCHS) == 15
    assert fvi.MILESTONE_EPOCHS == (50, 100, 150, 200)


def test_outer_batches_and_rng_stream_are_deterministic_and_nested() -> None:
    rows = np.arange(2500, dtype=np.int64)
    first = fvi._outer_batches(rows, 3)
    second = fvi._outer_batches(rows, 3)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.array_equal(np.sort(np.concatenate(first)), rows)
    assert fvi._candidate_seed(3, 2) == fvi._candidate_seed(3, 2)
    assert fvi._candidate_seed(3, 2, validation=True) != fvi._candidate_seed(3, 2)


def test_pairing_digest_depends_only_on_shared_rows_and_rng() -> None:
    context = {"train_rows": np.arange(1031, dtype=np.int64)}
    assert fvi._pairing_digest(context, 2) == fvi._pairing_digest(context, 2)
    assert len(fvi._pairing_digest(context, 2)) == 2


def test_summary_and_nearest_neighbor_distances() -> None:
    train = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    query = np.asarray([[1.0, 0.0], [2.0, 0.0]])
    distances = fvi._nearest_distances(
        query, train, np.zeros((1, 2)), np.ones((1, 2)))
    assert distances.tolist() == pytest.approx([1.0, 0.0])
    summary = fvi._summary(np.asarray([-2.0, 0.0, 1.0]))
    assert summary["count"] == 3
    assert summary["mean_abs"] == pytest.approx(1.0)


def test_pooled_union_uses_exact_backup_then_outer_max(monkeypatch) -> None:
    observed = {}

    def fake_backup(models, states, candidates, continuation, *, common_noise):
        observed["models"] = tuple(models)
        observed["noise_shape"] = common_noise.shape
        # Exercise the terminal wrapper on one value per candidate.
        values = continuation(np.ones((len(states) * candidates.shape[1], 12)))
        assert values.shape == (len(states) * candidates.shape[1],)
        return np.asarray([[[1.0, 3.0], [4.0, 2.0]]])

    monkeypatch.setattr(fvi, "compute_source_aamas_backup", fake_backup)
    batch = {
        "states": np.zeros((2, 12), dtype=np.float32),
        "candidates": np.zeros((2, 2, 3), dtype=np.float32),
        "terminated": np.asarray([False, True]),
        "noise_seed": 7,
    }
    result = fvi._pooled_union_backup(batch, object(), lambda states: np.ones(len(states)))
    assert result.tolist() == [3.0, 4.0]
    assert observed["noise_shape"] == (4, fvi.CANDIDATE_ACTIONS, 3)


def test_late_instability_is_trend_based_not_absolute_value_cutoff() -> None:
    rows = []
    for index, epoch in enumerate((150, 160, 170, 180, 190, 200), start=1):
        rows.append({
            "variant": "matched_moving_target", "epoch": str(epoch),
            "training_loss": str(index), "prediction_gradient_norm_mean": str(index),
            "prediction_std": str(index), "target_std": str(index),
            "target_mean": "0",
        })
    assert fvi._late_instability(rows, "matched_moving_target")
    assert fvi._status(rows, "matched_moving_target") == "BOOTSTRAP_INSTABILITY_OBSERVED"


def test_no_divergence_label_does_not_claim_convergence() -> None:
    rows = []
    for epoch in (150, 160, 170, 180, 190, 200):
        rows.append({
            "variant": "frozen_outer_target", "epoch": str(epoch),
            "training_loss": "1", "prediction_gradient_norm_mean": "1",
            "prediction_std": "2", "target_std": "2", "target_mean": "1",
        })
    assert fvi._status(rows, "frozen_outer_target") == \
        "NO_DIVERGENCE_OBSERVED_WITHIN_BUDGET"


def test_degenerate_output_has_distinct_status() -> None:
    rows = [{
        "variant": "frozen_outer_target", "epoch": "200",
        "prediction_std": "0", "target_std": "0", "target_mean": "1",
    }]
    assert fvi._status(rows, "frozen_outer_target") == \
        "TARGET_OR_OUTPUT_DEGENERACY_REQUIRES_REVIEW"


def test_preflight_rejects_scope_expansion_before_reading_inputs(tmp_path: Path) -> None:
    with pytest.raises(fvi.Phase8JFVIError, match="scope"):
        fvi.run_preflight_and_tests(
            output_root=tmp_path, method="action_min", device="cpu")


def test_train_rejects_any_third_variant(tmp_path: Path) -> None:
    with pytest.raises(fvi.Phase8JFVIError, match="frozen paired"):
        fvi.run_train(output_root=tmp_path,
                      variants=(*fvi.VARIANTS, "sigmoid"), device="cpu")


def test_cli_examples_parse_exactly() -> None:
    preflight = parse_arguments([
        "--phase", "preflight-and-tests", "--method", "pooled_union",
        "--model-seed", "0", "--samples-per-anchor-source", "128",
        "--component-updates", "4000"])
    assert preflight.phase == "preflight-and-tests"
    train = parse_arguments([
        "--phase", "train", "--variants", "matched_moving_target",
        "frozen_outer_target", "--outer-iterations", "40", "--inner-epochs", "5",
        "--total-epochs", "200", "--device", "cuda"])
    assert train.variants == list(fvi.VARIANTS)


def test_module_has_no_online_training_or_output_clamp() -> None:
    source = inspect.getsource(fvi)
    assert "run_online(" not in source
    # torch.clamp remains in the inherited road-not-taken action post-processing;
    # the potential readout itself must be the unclamped repaired implementation.
    assert "return self.network(state)" in inspect.getsource(
        fvi.make_repaired_potential_network)
    assert "clip_grad" not in source
    assert "gradient_clip" not in source
    assert "eligible_for_sac\": False" in source
    assert "do_oracle_used\": False" in source
