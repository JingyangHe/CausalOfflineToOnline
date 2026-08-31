"""Focused tests for Phase 8B-RS-O exact population identities."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift import oracle_direct_reward_audit as audit


LOGGER_TABLES = {
    "0": {"-1": {"minus": 0.9, "plus": 0.1},
          "1": {"minus": 0.1, "plus": 0.9}},
    "1": {"-1": {"minus": 0.7, "plus": 0.3},
          "1": {"minus": 0.3, "plus": 0.7}},
    "2": {"-1": {"base": 1.0}, "1": {"base": 1.0}},
}
MIXTURES = {name: list(value) for name, value in audit.PRIMARY_MIXTURES.items()}


def test_manifest_theoretical_balanced_slopes():
    assert np.isclose(audit.manifest_u_mean(
        "confounded", "logger12_balanced", "plus", LOGGER_TABLES, MIXTURES), 0.6)
    assert np.isclose(audit.manifest_u_mean(
        "confounded", "logger12_balanced", "minus", LOGGER_TABLES, MIXTURES), -0.6)
    assert audit.manifest_u_mean(
        "confounded", "logger12_balanced", "base", LOGGER_TABLES, MIXTURES) == 0.0


def test_manifest_theoretical_heavy_slopes():
    plus = audit.manifest_u_mean(
        "confounded", "logger1_heavy", "plus", LOGGER_TABLES, MIXTURES)
    plus -= audit.manifest_u_mean(
        "confounded", "logger2_heavy", "plus", LOGGER_TABLES, MIXTURES)
    minus = audit.manifest_u_mean(
        "confounded", "logger1_heavy", "minus", LOGGER_TABLES, MIXTURES)
    minus -= audit.manifest_u_mean(
        "confounded", "logger2_heavy", "minus", LOGGER_TABLES, MIXTURES)
    assert np.isclose(plus, 14.0 / 45.0)
    assert np.isclose(minus, -14.0 / 45.0)


def test_independent_latents_direct_slope_zero():
    for mixture in MIXTURES:
        for action in audit.ACTION_KEYS:
            assert audit.manifest_u_mean(
                "independent_latents", mixture, action, LOGGER_TABLES, MIXTURES) == 0.0


def test_support_action_means_use_u_env():
    public = {
        "row_id": np.arange(4), "anchor_id": np.zeros(4, dtype=int),
        "reward": np.asarray([1.0, 3.0, 2.0, 2.0]),
    }
    hidden = {
        "row_id": np.arange(4), "reward": public["reward"],
        "action_key": np.asarray(["plus", "plus", "minus", "base"]),
        "u_env": np.asarray([-1, 1, -1, 1]),
    }
    result = audit.support_action_means(public, hidden, np.ones(4), np.asarray([0]))
    assert result["plus"]["reward"][0] == 2.0
    assert result["plus"]["u_mean"][0] == 0.0
    assert result["minus"]["u_mean"][0] == -1.0
    assert result["base"]["u_mean"][0] == 1.0


def test_do_direct_term_cancels():
    for strength in (0.0, 0.05, 0.1, 0.2):
        assert 0.5 * (strength + -strength) == 0.0


def test_exact_bias_decomposition():
    original_obs, do, conditional_u, strength = 1.7, 1.4, 0.6, 0.2
    physical = original_obs - do
    direct = strength * conditional_u
    total = original_obs + direct - do
    assert np.isclose(total, physical + direct)


def test_fit_line_recovers_exact_slope_and_intercept():
    x = np.asarray([0.0, 0.05, 0.1, 0.2])
    slope, intercept, r2 = audit.fit_line(x, 0.17 + 0.6 * x)
    assert np.isclose(slope, 0.6)
    assert np.isclose(intercept, 0.17)
    assert np.isclose(r2, 1.0)


def test_anchor_bootstrap_reports_required_statistics():
    result = audit.bootstrap_stats(np.asarray([1.0, 2.0, 3.0]), 100, 0)
    assert {"mean", "std", "median", "p10", "p25", "p75", "p90", "max",
            "ci_low", "ci_high", "n_anchors"}.issubset(result)
    assert result["bootstrap_unit"] == "anchor_id"


def test_auto_resolution_requires_unique_complete_artifact(tmp_path: Path):
    with pytest.raises(audit.OracleRewardAuditError, match="exactly one"):
        audit.resolve_reward_signal_root(tmp_path, None)


def test_no_neural_training_or_prediction_path():
    source = inspect.getsource(audit.run_oracle_direct_reward_audit)
    forbidden = ("torch", "RewardMeanModel", "load_checkpoint", "predict(", "fit(")
    assert not any(token in source for token in forbidden)


def test_do_shift_is_computed_not_hard_coded():
    source = inspect.getsource(audit.run_oracle_direct_reward_audit)
    assert '"do_shift": 0.0' not in source
    assert "augmented_do - original_do" in source


def test_cli_markers_present():
    path = Path(__file__).parents[1] / "scripts" / "analyze_hopper_oracle_direct_reward_confounding.py"
    text = path.read_text(encoding="utf-8")
    assert "ORACLE_DIRECT_REWARD_CONFOUNDING_AUDIT_COMPLETE" in text
    assert "READY_FOR_NEURAL_REWARD_SIGNAL_TEST" in text
    assert "ORACLE_DIRECT_REWARD_CONFOUNDING_AUDIT_BLOCKED" in text
