from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift import ranking_calibration_regret_audit as audit


def test_top_masks_use_existing_numeric_tolerance():
    values = np.asarray([[1.0, 1.0 + 5e-8, 0.0], [0.0, 1.0, 2.0]])
    masks = audit.top_masks(values)
    assert masks.tolist() == [[True, True, False], [False, False, True]]
    assert audit.TOP_ATOL == audit.TOP_RTOL == 1e-7


def test_failure_type_decomposition_is_exhaustive():
    do = np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], bool)
    obs = np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]], bool)
    nn = np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], bool)
    assert audit.classify_failure_types(do, obs, nn).tolist() == list("ABCDE")


def test_regret_respects_tied_top_sets():
    do = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    top = np.asarray([[False, True, False], [False, True, True]])
    best, worst = audit.top_set_regret(do, top)
    assert np.allclose(best, [1.0, 1.0])
    assert np.allclose(worst, [1.0, 2.0])


def test_bias_and_gap_decomposition():
    anchors = np.asarray([7, 21])
    do = np.asarray([[3.0, 2.0, 1.0], [1.0, 1.5, 2.0]])
    obs = do + np.asarray([[0.2, 0.2, 0.2], [-0.1, 0.0, 0.1]])
    nn = obs + np.asarray([[0.1, -0.1, 0.0], [0.2, -0.2, 0.0]])
    scenarios, actions = audit._scenario(0.0, 0.2, "confounded", "logger12_balanced",
                                          0, anchors, do, obs, nn)
    assert len(scenarios) == 2 and len(actions) == 6
    for row in actions:
        assert np.isclose(row["b_nn"], row["b_obs"] + row["e_nn"])
    assert np.isclose(scenarios[0]["c_obs"], 0.2)
    assert np.isclose(scenarios[0]["max_abs_d_obs"], 0.0)


def test_safe_correlations_return_none_for_constant_inputs():
    assert audit.safe_pearson(np.ones(4), np.arange(4)) is None
    assert audit.safe_spearman(np.ones(4), np.arange(4)) is None


def test_preflight_blocks_when_raw_predictions_are_absent(tmp_path: Path):
    neural, oracle, output = tmp_path / "neural", tmp_path / "oracle", tmp_path / "out"
    neural.mkdir(); oracle.mkdir()
    for root in (neural, oracle):
        (root / "hard_checks.json").write_text('{"all_passed": true}', encoding="utf-8")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
    (neural / "splits.json").write_text(
        '{"test": [' + ",".join(str(i) for i in range(78)) + "]}", encoding="utf-8")
    np.savez_compressed(oracle / "anchor_action_metrics.npz", placeholder=np.asarray([1]))
    with pytest.raises(audit.RankingCalibrationRegretAuditError, match="144/144 missing"):
        audit.preflight(neural, oracle, output)


def test_end_to_end_read_only_audit(tmp_path: Path):
    neural, oracle, output = tmp_path / "neural", tmp_path / "oracle", tmp_path / "output"
    neural.mkdir(); oracle.mkdir()
    for root in (neural, oracle):
        (root / "hard_checks.json").write_text('{"all_passed": true}', encoding="utf-8")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
    test_ids = list(range(78))
    (neural / "splits.json").write_text(
        '{"test": [' + ",".join(str(i) for i in test_ids) + "]}", encoding="utf-8")
    oracle_rows = {key: [] for key in ("anchor_id", "kappa", "lambda_reward", "condition",
                                        "mixture", "action", "augmented_observational_reward",
                                        "do_reward")}
    for kappa in audit.KAPPAS:
        for strength in audit.LAMBDAS:
            for condition in audit.CONDITIONS:
                for mixture in audit.MIXTURES:
                    do = np.stack([np.linspace(1.0, 2.0, 78), np.linspace(0.9, 1.9, 78),
                                   np.linspace(0.8, 1.8, 78)], axis=1)
                    direct = np.asarray([-0.6, 0.0, 0.6]) * strength \
                        if condition == "confounded" else np.zeros(3)
                    obs = do + direct
                    for seed in audit.MODEL_SEEDS:
                        path = audit.prediction_path(neural, kappa, strength, condition, mixture, seed)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, anchor_id=np.asarray(test_ids),
                                            prediction=obs + (seed - 1) * 1e-3,
                                            population_target=obs)
                    for i, anchor in enumerate(test_ids):
                        for j, action in enumerate(audit.ACTIONS):
                            values = (anchor, kappa, strength, condition, mixture, action,
                                      obs[i, j], do[i, j])
                            for key, value in zip(oracle_rows, values):
                                oracle_rows[key].append(value)
    np.savez_compressed(oracle / "anchor_action_metrics.npz",
                        **{key: np.asarray(value) for key, value in oracle_rows.items()})
    before = audit._hashes([neural / "manifest.json", oracle / "anchor_action_metrics.npz"])
    summary = audit.run_audit(neural, oracle, output)
    after = audit._hashes([neural / "manifest.json", oracle / "anchor_action_metrics.npz"])
    assert before == after
    assert summary["all_hard_checks_passed"] is True
    for name in ("REPORT.md", "summary.json", "ranking_metrics.csv", "calibration_metrics.csv",
                 "gap_metrics.csv", "regret_metrics.csv", "failure_type_metrics.csv",
                 "seed_metrics.csv", "anchor_action_metrics.npz", "hard_checks.json"):
        assert (output / name).is_file()
    assert len(list((output / "figures").glob("*.png"))) == 10


def test_cli_has_completion_and_blocking_markers():
    path = Path(__file__).parents[1] / "scripts" / "audit_hopper_reward_ranking_calibration_regret.py"
    text = path.read_text(encoding="utf-8")
    assert "PHASE8B_RS_RANKING_CALIBRATION_REGRET_AUDIT_BLOCKED" in text
    assert "PHASE8B_RS_RANKING_CALIBRATION_REGRET_AUDIT_COMPLETE" in text
