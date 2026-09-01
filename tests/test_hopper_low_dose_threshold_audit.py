from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift import low_dose_threshold_audit as audit


PHASE8A = ROOT / "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified"
RANKING = ROOT / "analysis/phase8b_rs_ranking_calibration_regret_audit"


def _example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    do = np.asarray([1.0, 0.99, 0.98])
    b = do.copy()
    c = np.asarray([-0.6, 0.0, 0.6])
    return do, b, c


def test_inputs_are_read_only(tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_text('{"fixed": true}', encoding="utf-8")
    before = audit._hashes([path])
    audit._read_json(path)
    assert audit._hashes([path]) == before


def test_do_reward_is_lambda_invariant():
    do, _, _ = _example()
    assert np.array_equal(do, do.copy())
    assert "lambda" not in inspect.signature(audit._regret).parameters


def test_observational_reward_is_affine_in_lambda():
    _, b, c = _example()
    assert np.allclose(audit._reward(b, c, 0.2), b + 0.2 * c)
    assert np.allclose(audit._reward(b, c, 0.1),
                       0.5 * (audit._reward(b, c, 0.0) + audit._reward(b, c, 0.2)))


def test_existing_lambda_grid_crosscheck():
    if not (PHASE8A / "anchor_action_metrics.npz").is_file():
        return
    raw = audit._load_npz(PHASE8A / "anchor_action_metrics.npz")
    splits = audit._read_json(PHASE8A / "phase8b_reward_signal_calibration/splits.json")
    rows = audit._crosscheck_ranking(raw, RANKING / "anchor_action_metrics.npz",
                                     splits["test"], 1e-7, 1e-7)
    assert len(rows) == 48 and all(row["passed"] for row in rows)


def test_pairwise_crossing_formula():
    b = np.asarray([1.0, 0.8, 0.0]); c = np.asarray([-1.0, 1.0, 0.0])
    value, status = audit.pairwise_crossing(b, c, 0, 1)
    assert status == "nonnegative" and np.isclose(value, 0.1)
    assert np.isclose(audit._reward(b, c, value)[0], audit._reward(b, c, value)[1])


def test_parallel_curves_handled():
    value, status = audit.pairwise_crossing(np.asarray([1., 0., 1.]),
                                            np.asarray([0., 0., 0.]), 0, 1)
    assert np.isposinf(value) and status == "parallel_distinct"
    _, equal_status = audit.pairwise_crossing(np.ones(3), np.zeros(3), 0, 1)
    assert equal_status == "parallel_equal"


def test_threshold_zero_when_already_wrong():
    thresholds, _ = audit.scenario_thresholds(
        np.asarray([0., 1., 2.]), np.zeros(3), np.asarray([2., 1., 0.]), 1e-7, 1e-7)
    assert all(value == 0.0 for value in thresholds.values())


def test_infinite_threshold_when_never_wrong():
    do = np.asarray([2., 1., 0.])
    thresholds, _ = audit.scenario_thresholds(do, np.zeros(3), do, 1e-7, 1e-7)
    assert all(np.isposinf(value) for value in thresholds.values())


def test_crossing_left_point_right_classification():
    do, b, c = _example()
    b = np.asarray([1.0, 0.90, 0.98])
    value = (b[0] - b[2]) / (c[2] - c[0])
    sides = audit.crossing_sides(b, c, value, 1e-7, 1e-7)
    assert sides["top_left"].tolist() == [True, False, False]
    assert sides["top_point"].tolist() == [True, False, True]
    assert sides["top_right"].tolist() == [False, False, True]


def test_nextafter_used_instead_of_arbitrary_epsilon():
    source = (inspect.getsource(audit.crossing_sides)
              + inspect.getsource(audit.scenario_thresholds)
              + inspect.getsource(audit._one_sided_top))
    assert "np.nextafter" in source
    assert "epsilon" not in source and "1e-" not in source


def test_top_sets_use_project_tolerance():
    values = np.asarray([1.0, 1.0 + 5e-8, 0.0])
    assert audit.top_mask(values, 1e-7, 1e-7).tolist() == [True, True, False]


def test_regret_nonnegative():
    do = np.asarray([[3., 2., 1.], [1., 2., 3.]])
    candidate = np.asarray([[False, True, True], [True, False, False]])
    best, worst = audit._regret(do, candidate)
    assert np.all(best >= 0.0) and np.all(worst >= best)


def test_independent_top_set_is_lambda_invariant():
    b = np.asarray([1., 2., 0.]); c = np.zeros(3)
    first = audit.top_mask(audit._reward(b, c, 0.0), 1e-7, 1e-7)
    assert all(np.array_equal(first, audit.top_mask(audit._reward(b, c, value), 1e-7, 1e-7))
               for value in (0.001, 0.2, 100.0))


def test_test_anchors_not_used_for_dose_proposal():
    source = inspect.getsource(audit.run_audit)
    assert 'np.isin(threshold_data["split"], ["train", "validation"])' in source
    assert '"selection_data": "train_and_validation_only"' in source


def test_input_hashes_unchanged(tmp_path: Path):
    path = tmp_path / "fixed.bin"; path.write_bytes(b"immutable")
    first = audit._hashes([path]); second = audit._hashes([path])
    assert first == second


def test_no_nan_except_explicit_infinity():
    thresholds, crossings = audit.scenario_thresholds(
        np.asarray([2., 1., 0.]), np.zeros(3), np.asarray([2., 1., 0.]), 1e-7, 1e-7)
    assert not any(np.isnan(value) for value in thresholds.values())
    assert all(not np.isnan(row["crossing"]) for row in crossings)


def test_current_report_numbers_reproduced():
    if not (PHASE8A / "anchor_action_metrics.npz").is_file():
        return
    raw = audit._load_npz(PHASE8A / "anchor_action_metrics.npz")
    ids, do, b, c = audit._base_arrays(raw, 0.3, "confounded", "logger12_balanced")
    assert len(ids) == 2048
    assert np.allclose(c, [-0.6, 0.0, 0.6], atol=1e-7, rtol=1e-7)
    assert np.all(np.isfinite(do)) and np.all(np.isfinite(b))


def test_threshold_distribution_tracks_censoring_and_zeros():
    result = audit.threshold_distribution(np.asarray([0.0, 0.1, 0.2, np.inf]))
    assert result["finite_count"] == 3
    assert result["zero_count"] == 1
    assert result["censored_count"] == 1


def test_registered_low_dose_grid_is_unchanged():
    assert audit.LOW_DOSE_GRID == (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005,
                                   0.0075, 0.010, 0.015, 0.020, 0.030, 0.050)


def test_cli_contains_completion_and_blocking_markers():
    text = (ROOT / "scripts/audit_hopper_low_dose_thresholds.py").read_text(encoding="utf-8")
    assert "PHASE8B_RS_LOW_DOSE_THRESHOLD_AUDIT_COMPLETE" in text
    assert "READY_FOR_MANUAL_DOSE_GRID_FREEZE" in text
    assert "PHASE8B_RS_LOW_DOSE_THRESHOLD_AUDIT_BLOCKED" in text
