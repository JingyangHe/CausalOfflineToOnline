from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.hopper_logger_mixture_drift import phase8j_bellman_mean_error as bme
from scripts.analyze_phase8j_bellman_mean_error import parse_arguments


def test_exact_weighted_decomposition() -> None:
    prediction = np.array([1.0, 1.0, 4.0, 4.0])
    target = np.array([0.0, 2.0, 2.0, 6.0])
    anchors = np.array([10, 10, 20, 20])
    weights = np.array([1.0, 1.0, 1.0, 3.0])
    result, states = bme.decompose_by_anchor(prediction, target, anchors, weights)
    assert len(states) == 2
    np.testing.assert_allclose(
        result["total_mse"],
        result["mean_bellman_mse"] + result["within_state_variance"],
        rtol=0.0,
        atol=result["identity_tolerance"],
    )
    assert result["identity_residual"] <= result["identity_tolerance"]
    assert np.isclose(result["mean_error_fraction"] + result["noise_fraction"], 1.0)


def test_decomposition_rejects_non_state_constant_predictions() -> None:
    with np.testing.assert_raises_regex(ValueError, "differ within anchor"):
        bme.decompose_by_anchor(
            np.array([1.0, 1.1]), np.array([0.0, 2.0]), np.array([3, 3])
        )


def test_nearest_checkpoint_mapping_records_actual_epoch() -> None:
    available = {0: Path("initial.pt"), 50: Path("epoch_50.pt"), 200: Path("latest.pt")}
    mapped = bme.map_requested_checkpoints(available, (0, 20, 40, 120, 200))
    assert [row["actual_epoch"] for row in mapped] == [0, 0, 50, 50, 200]
    assert [row["exact"] for row in mapped] == [True, False, False, False, True]


def _rows(real: list[float], full: list[float], within: float = 1.0):
    result = []
    for variant, values in zip(bme.VARIANTS, (real, full)):
        for epoch, value in enumerate(values):
            result.append({
                "variant": variant,
                "epoch": epoch,
                "mean_bellman_mse": value,
                "within_state_variance": within,
                "total_mse": value + within,
            })
    return result


def test_frozen_decision_labels() -> None:
    assert bme.classify_result(_rows([1, 2, 3], [1, 3, 8])) == \
        "BOTH_REAL_AND_FULL_MEAN_BELLMAN_ERROR_GROW"
    assert bme.classify_result(_rows([1, 1, 1], [1, 2, 4])) == \
        "REAL_TRANSITION_MOSTLY_STABLE_FULL_AAMAS_DIVERGES"
    assert bme.classify_result(_rows([1, 2, 4], [4, 3, 2])) == \
        "REAL_TRANSITION_MEAN_BELLMAN_ERROR_DIVERGES"
    assert bme.classify_result(_rows([1, 1], [1, 2])) == "RESULT_AMBIGUOUS"

    variance_dominated = []
    for variant in bme.VARIANTS:
        for epoch, within in enumerate((1.0, 2.0, 4.0)):
            variance_dominated.append({
                "variant": variant,
                "epoch": epoch,
                "mean_bellman_mse": 1.0,
                "within_state_variance": within,
                "total_mse": 1.0 + within,
            })
    assert bme.classify_result(variance_dominated) == \
        "MOST_RAW_MSE_GROWTH_IS_TARGET_VARIANCE"


def test_full_target_uses_one_frozen_validation_stream(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(bme, "_validation_batches", lambda rows: [np.array([0, 1])])

    def fake_target(variant, context, components, rows, network, mean, std, outer, index):
        observed.append((variant, outer, index))
        return np.array([1.0, 2.0])

    monkeypatch.setattr(bme, "_recomputed_target", fake_target)
    context = {"validation_rows": np.array([0, 1])}
    for epoch in (0, 50, 200):
        np.testing.assert_array_equal(
            bme._variant_targets(
                "full_aamas_frozen", context, {}, object(),
                np.zeros((1, 1)), np.ones((1, 1)), epoch,
            ),
            np.array([1.0, 2.0]),
        )
    assert observed == [
        ("full_aamas_frozen", 0, 0),
        ("full_aamas_frozen", 0, 0),
        ("full_aamas_frozen", 0, 0),
    ]


def test_cli_frozen_defaults() -> None:
    args = parse_arguments([])
    assert args.model_seed == 0
    assert tuple(args.epochs) == bme.REQUESTED_EPOCHS
    assert args.phase8j_root == bme.DEFAULT_SOURCE_ROOT
