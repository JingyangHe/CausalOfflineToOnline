from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift.multisource_contrast_calibration import (
    ACTION_MARGINAL,
    FORBIDDEN_PUBLIC_FIELDS,
    bic_select_rank,
    calibration_features,
    closed_form_calibration,
    empirical_source_mean_matrix,
    fixed_draw_public_table,
    multisource_behavior_probabilities,
    source_action_marginals,
    svd_initialization,
)
from experiments.hopper_logger_mixture_drift.phase8e_quick_go_nogo import (
    QUICK_BUDGETS,
    QUICK_LAMBDAS,
    QUICK_METHODS,
    QUICK_SOURCE_SETTINGS,
    _comparison_rows,
    _save_scenario_rows,
    fit_quick_model,
    preflight_estimate,
    quick_scenarios,
    resolve_quick_inputs,
    run_phase8e_quick,
)
from experiments.hopper_logger_mixture_drift.multisource_contrast_calibration import (
    Phase8EMultisourceContrastError,
    budgets_are_nested,
    random_balanced_query_order,
    shuffle_source_within_anchor_action,
)


def synthetic_universe(n: int = 8):
    anchor_id = np.arange(n, dtype=np.int64)
    observation = np.arange(n * 12, dtype=np.float32).reshape(n, 12) / 100
    action = np.zeros((n, 3, 3), dtype=np.float32)
    action[:, 0, 0] = -0.2
    action[:, 2, 0] = 0.2
    center = 1 + anchor_id[:, None] * 0.01 + np.asarray((0.0, 0.1, 0.2))[None, :]
    contrast = (1 + anchor_id[:, None] / 20) * np.asarray((0.03, 0.0, 0.05))[None, :]
    branches = np.stack((center - contrast, center + contrast), axis=2)
    return anchor_id, observation, action, branches


def public_for(setting: str, budget: int = 240):
    anchor_id, observation, action, branches = synthetic_universe()
    return fixed_draw_public_table(
        anchor_id, observation, action, branches, QUICK_SOURCE_SETTINGS[setting],
        kappa=0.0, lambda_reward=0.05, sigma_reward=0.02,
        condition="confounded", sample_budget=budget, seed=11)


def test_quick_grid_is_exact():
    assert QUICK_LAMBDAS == (0.0, 0.01, 0.05)
    assert QUICK_BUDGETS == (0, 16, 64)
    assert QUICK_METHODS == (
        "pooled_rank0", "MSCSC_correct_source", "MSCSC_source_shuffle")
    assert np.array_equal(QUICK_SOURCE_SETTINGS["M2_diverse"], [0.55, 0.95])
    assert np.array_equal(QUICK_SOURCE_SETTINGS["M5_diverse"],
                          [0.55, 0.65, 0.75, 0.85, 0.95])
    assert np.array_equal(QUICK_SOURCE_SETTINGS["M5_redundant"], np.full(5, 0.75))


def test_quick_has_nine_primary_and_one_negative_control_scenarios():
    scenarios = quick_scenarios(tuple(QUICK_SOURCE_SETTINGS), QUICK_LAMBDAS)
    assert len(scenarios) == 10
    controls = [row for row in scenarios if row["condition"] == "independent_latents"]
    assert controls == [{"setting": "M5_diverse", "lambda_reward": 0.05,
                         "condition": "independent_latents"}]


def test_preflight_scales_by_scenario_method_seed():
    estimate = preflight_estimate(10, 3, 49152)
    assert estimate["model_count"] == 90
    assert estimate["estimated_file_count"] < 500
    assert estimate["estimated_disk_mib"] < 100


@pytest.mark.parametrize("setting", tuple(QUICK_SOURCE_SETTINGS))
def test_every_source_has_the_same_action_marginal(setting: str):
    behavior = multisource_behavior_probabilities(QUICK_SOURCE_SETTINGS[setting])
    assert np.allclose(source_action_marginals(behavior), ACTION_MARGINAL[None, :],
                       atol=1e-15, rtol=0)


def test_all_settings_have_equal_total_sample_budget():
    assert {len(public_for(setting)[0]["reward"])
            for setting in QUICK_SOURCE_SETTINGS} == {240}


def test_shuffle_changes_only_source_within_anchor_action():
    public, _ = public_for("M5_diverse")
    shuffled = shuffle_source_within_anchor_action(
        public["anchor_id"], public["action_index"], public["source_id"], 31)
    shuffled_public = dict(public)
    shuffled_public["source_id"] = shuffled
    assert FORBIDDEN_PUBLIC_FIELDS.isdisjoint(public)
    for name, values in public.items():
        if name != "source_id":
            assert np.array_equal(values, shuffled_public[name])
    for anchor in np.unique(public["anchor_id"]):
        for action in range(3):
            mask = ((public["anchor_id"] == anchor)
                    & (public["action_index"] == action))
            assert np.array_equal(np.sort(public["source_id"][mask]),
                                  np.sort(shuffled[mask]))


def test_compact_scenario_does_not_repeat_observation_or_action(tmp_path: Path):
    public, _ = public_for("M2_diverse")
    path = tmp_path / "scenario.npz"
    _save_scenario_rows(path, public, np.arange(8), {"setting": "M2_diverse"})
    with np.load(path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "row_index", "source_id", "reward", "weight", "configuration_json"}
        assert "observation" not in arrays and "commanded_action" not in arrays


def test_calibration_prefix_and_closed_forms():
    anchors = np.repeat(np.arange(30), 3)
    actions = np.tile(np.arange(3), 30)
    order = random_balanced_query_order(anchors, actions, 64, 0)
    assert budgets_are_nested(order, QUICK_BUDGETS)
    h = np.linspace(-1, 1, len(actions))
    x0 = calibration_features(actions, h, rank=0)
    x1 = calibration_features(actions, h, rank=1)
    theta0 = np.asarray((0.1, -0.2, 0.3))
    theta1 = np.asarray((0.1, -0.2, 0.3, 0.4, -0.1, 0.2))
    assert np.allclose(closed_form_calibration(np.zeros(len(x0)), x0 @ theta0, x0).coefficients,
                       theta0)
    assert np.allclose(closed_form_calibration(np.zeros(len(x1)), x1 @ theta1, x1).coefficients,
                       theta1)
    assert bic_select_rank(1.0, 1.0, 16) == 0


def test_quick_fit_signature_excludes_hidden_and_oracle_inputs():
    parameters = set(inspect.signature(fit_quick_model).parameters)
    assert not parameters.intersection({
        "u", "u_env", "u_behavior", "do_reward", "reward_branches"})


def test_fixed_configuration_gate_runs_before_input_resolution(tmp_path: Path):
    with pytest.raises(Phase8EMultisourceContrastError,
                       match="fixed quick configuration was changed"):
        run_phase8e_quick(
            tmp_path / "missing", tmp_path / "out", num_anchors=511,
            source_settings=tuple(QUICK_SOURCE_SETTINGS), lambda_values=QUICK_LAMBDAS,
            reward_noise_std=0.02, offline_sample_budget=49152,
            model_seeds=(0, 1, 2), gradient_updates=1000,
            calibration_budgets=QUICK_BUDGETS, calibration_replicates=5,
            device="cpu")


def test_phase8a_root_manifest_pair_is_optional_for_raw_oracle_layout(tmp_path: Path):
    phase8a = tmp_path / "controlled_loggers_seed0_verified"
    raw = phase8a / "kappa_0p00" / "do_oracle_raw.npz"
    raw.parent.mkdir(parents=True)
    np.savez(raw, placeholder=np.asarray([1]))
    direct = (tmp_path / "noncomplementary_loggers_seed0_verified"
              / "phase8c_direct_reward_public_grid")
    direct.mkdir(parents=True)
    (direct / "manifest.json").write_text("{}", encoding="utf-8")
    (direct / "hard_checks.json").write_text(
        json.dumps({"all_passed": True, "checks": {"ok": True}}), encoding="utf-8")
    (direct / "splits.json").write_text("{}", encoding="utf-8")
    (direct / "frozen_lambda_grid.json").write_text(json.dumps({
        "manually_frozen": True, "lambdas": [0.0, 0.01, 0.05]}), encoding="utf-8")
    for dose, condition in ((0.0, "confounded"), (0.01, "confounded"),
                            (0.05, "confounded"), (0.05, "independent_latents")):
        np.savez(direct / f"dose_{dose}_{condition}_public.npz",
                 kappa_env=np.asarray([0.0]), lambda_reward=np.asarray([dose]),
                 condition=np.asarray([condition]))
    resolved = resolve_quick_inputs(phase8a, QUICK_LAMBDAS)
    assert resolved["phase8a_root_checks_available"] is False
    assert raw.resolve() in resolved["required_paths"]


def test_paired_comparisons_report_seed_differences():
    definitions = (
        ("M5_diverse", "MSCSC_correct_source", 0.10),
        ("M5_diverse", "MSCSC_source_shuffle", 0.20),
        ("M2_diverse", "MSCSC_correct_source", 0.30),
        ("M5_redundant", "MSCSC_correct_source", 0.40),
    )
    seed_rows = []
    for setting, method, base in definitions:
        for budget in (0, 64):
            for seed in (0, 1, 2):
                value = base + seed * 0.01 - (0.02 if budget == 64 else 0)
                seed_rows.append({
                    "setting": setting, "lambda_reward": 0.05,
                    "condition": "confounded", "seed": seed, "method": method,
                    "calibration_budget": budget, "do_mae": value,
                    "top_set_disagreement": value, "mean_regret": value,
                    "selected_rank": 0.0,
                })
    # Add the B=0 correct cell required by the budget comparison (already present),
    # then summarize using the production aggregators.
    from experiments.hopper_logger_mixture_drift.phase8e_quick_go_nogo import (
        _summary_table,
    )
    summary = _summary_table(seed_rows)
    comparisons = _comparison_rows(summary, seed_rows)
    correct_shuffle = next(row for row in comparisons
                           if row["comparison"].startswith("Correct"))
    assert np.isclose(correct_shuffle["do_mae_difference"], -0.1)
    assert correct_shuffle["do_mae_difference_negative_seed_fraction"] == 1.0


@pytest.mark.skipif(sys.platform == "win32", reason="server PyTorch integration test")
def test_quick_model_is_width_128_and_paired_schedule():
    pytest.importorskip("torch")
    public, _ = public_for("M2_diverse")
    train_ids = np.arange(6)
    train_mask = np.isin(public["anchor_id"], train_ids)
    train_public = {name: np.asarray(value)[train_mask] for name, value in public.items()}
    initialization = svd_initialization(
        empirical_source_mean_matrix(train_public, 2, train_ids))
    first = fit_quick_model(public, initialization, train_ids, train_ids,
                            seed=0, updates=2, batch_size=16, device="cpu")
    second = fit_quick_model(public, initialization, train_ids, train_ids,
                             seed=0, updates=2, batch_size=16, device="cpu")
    assert first[0].g.network[0].out_features == 128
    assert first[2]["initial_network_hash"] == second[2]["initial_network_hash"]
    assert first[2]["minibatch_schedule_hash"] == second[2]["minibatch_schedule_hash"]


def test_cli_is_independent_and_exposes_required_arguments():
    result = subprocess.run(
        [sys.executable, "scripts/run_hopper_phase8e_quick.py", "--help"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert "--source-settings" in result.stdout
    assert "--offline-sample-budget" in result.stdout
    assert "--output-root" in result.stdout
