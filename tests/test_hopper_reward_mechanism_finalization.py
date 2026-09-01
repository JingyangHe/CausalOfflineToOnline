from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift import reward_mechanism_finalization as finalizer
from experiments.hopper_logger_mixture_drift import reward_mechanism_separation as phase


def test_expected_formal_result_counts():
    counts = finalizer.expected_result_counts(test_anchor_count=308, scenario_count=140)
    assert counts["models"] == 3780
    assert counts["observational_metrics"] == 18_900
    assert counts["do_metrics"] == 15_120
    assert counts["ranking_metrics"] == 3780
    assert counts["regret_metrics"] == 3780
    assert counts["composition_stability"] == 3780
    assert counts["latent_diagnostics"] == 26_040
    assert counts["seed_metrics"] == 3780
    assert counts["anchor_action_predictions"] == 3780 * 308 * 3


def test_base_figure_ignores_prediction_rows_without_mae(tmp_path: Path):
    scenario = {"kappa": 0.0, "lambda_reward": 0.0, "condition": "confounded",
                "mixture": phase.PRIMARY_MIXTURE, "seed": 0}
    do_rows = [
        {**scenario, "method": "pooled_mlp", "action": "all", "anchor_id": "",
         "prediction": "", "mae": 0.1, "rmse": 0.2, "signed_bias": 0.0},
        {**scenario, "method": "pooled_mlp", "action": "base", "anchor_id": "",
         "prediction": "", "mae": 0.1, "rmse": 0.2, "signed_bias": 0.0},
        {**scenario, "method": "pooled_mlp", "action": "base", "anchor_id": 7,
         "prediction": 1.0, "do_reward": 1.1},
    ]
    ranking = [{**scenario, "method": "pooled_mlp", "top_set_disagreement": 0.1}]
    regret = [{**scenario, "method": "pooled_mlp", "mean_regret": 0.01}]
    observational = [{**scenario, "method": "pooled_mlp", "mae": 0.2}]
    phase._make_figures(tmp_path, do_rows, ranking, regret, observational, [], [])
    assert (tmp_path / "figures/base_action_error_by_method.png").is_file()


def test_numeric_validation_rejects_nonfinite():
    try:
        finalizer._validate_numeric_rows([{"mae": "nan"}], ("mae",), "test")
    except finalizer.RewardMechanismFinalizationError:
        pass
    else:
        raise AssertionError("non-finite metric was accepted")


def test_cli_has_recovery_markers():
    text = (ROOT / "scripts/finalize_hopper_reward_mechanism_separation.py").read_text(
        encoding="utf-8")
    assert "PHASE8C_REWARD_MECHANISM_FINALIZATION_BLOCKED" in text
    assert "PHASE8C_REWARD_MECHANISM_FINALIZATION_COMPLETE" in text
    assert "READY_FOR_MECHANISM_EFFECT_REVIEW" in text

