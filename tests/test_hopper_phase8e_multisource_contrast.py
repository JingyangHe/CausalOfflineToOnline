from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift.multisource_contrast_calibration import (
    ACTION_MARGINAL,
    FORBIDDEN_PUBLIC_FIELDS,
    PUBLIC_FIELDS,
    active_query_order,
    antithetic_reward_noise,
    audit_population_subspace,
    bic_select_rank,
    budgets_are_nested,
    calibration_features,
    closed_form_calibration,
    decision_metrics,
    deterministic_rank1_svd,
    diversity_profile,
    do_reward_mean,
    fixed_draw_public_table,
    fit_source_free_model,
    input_hashes,
    load_checkpoint,
    make_source_free_model,
    multisource_behavior_probabilities,
    normalize_loadings,
    pairwise_gap_features,
    population_source_means,
    random_balanced_query_order,
    save_checkpoint,
    shuffle_source_within_anchor_action,
    source_action_marginals,
    source_composition_state_action_mass,
    svd_initialization,
    validate_public_table,
    validate_source_free_model,
)
from experiments.hopper_logger_mixture_drift.phase8e_multisource_contrast import (
    select_anchor_splits,
)


def synthetic_universe(n: int = 8):
    ids = np.arange(n, dtype=np.int64)
    observation = np.arange(n * 12, dtype=np.float32).reshape(n, 12) / 100
    action = np.zeros((n, 3, 3), dtype=np.float32)
    action[:, 0, 0] = -0.2; action[:, 2, 0] = 0.2
    base = 1.0 + ids[:, None] * 0.01 + np.asarray((0.0, 0.1, 0.2))[None, :]
    effect = np.asarray((0.03, 0.04, 0.05))[None, :] * (1 + ids[:, None] / 20)
    branches = np.stack((base - effect, base + effect), axis=2)
    return ids, observation, action, branches


def synthetic_public(m: int = 3, diversity: float = 0.2, n: int = 8, budget: int = 288):
    ids, obs, actions, branches = synthetic_universe(n)
    return fixed_draw_public_table(ids, obs, actions, branches,
                                   diversity_profile(m, diversity), kappa=0.0,
                                   lambda_reward=0.05, sigma_reward=0.02,
                                   condition="confounded", sample_budget=budget, seed=7)


def test_multisource_behavior_probabilities():
    table = multisource_behavior_probabilities([0.55, 0.75, 0.95])
    assert table.shape == (3, 2, 3)
    assert np.allclose(table.sum(axis=2), 1.0)
    assert np.isclose(table[1, 1, 2], 0.9 * 0.75)
    assert np.isclose(table[1, 0, 0], 0.9 * 0.75)


def test_all_sources_same_action_marginal():
    table = multisource_behavior_probabilities(diversity_profile(8, 0.2))
    assert np.allclose(source_action_marginals(table), ACTION_MARGINAL[None, :], atol=1e-15)


def test_state_action_mass_source_composition_invariant():
    table = multisource_behavior_probabilities(diversity_profile(5, 0.15))
    for weights in (np.ones(5) / 5, np.arange(1, 6) / 15):
        assert np.allclose(source_composition_state_action_mass(table, weights), ACTION_MARGINAL)


def test_diversity_profiles_exact():
    assert np.array_equal(diversity_profile(2, 0.2), [0.55, 0.95])
    assert np.allclose(diversity_profile(5, 0.15), np.linspace(0.60, 0.90, 5))


def test_redundant_sources_identical():
    assert np.array_equal(diversity_profile(8, 0.0), np.full(8, 0.75))
    loading, direction, singular = deterministic_rank1_svd(np.zeros((8, 12)))
    assert np.isclose(loading.mean(), 0.0)
    assert np.isclose(np.mean(loading ** 2), 1.0)
    assert np.array_equal(direction, np.zeros(12))
    assert np.array_equal(singular, np.zeros(8))


def test_reward_noise_antithetic():
    noise = antithetic_reward_noise(101, 0.02, 9)
    assert np.array_equal(noise[:100:2], -noise[1:100:2])
    assert noise[-1] == 0.0


def test_do_mean_noise_invariant():
    *_, branches = synthetic_universe()
    assert np.allclose(do_reward_mean(branches, 0.0), do_reward_mean(branches, 0.1))
    assert np.isclose(antithetic_reward_noise(100, 0.02, 3).mean(), 0.0, atol=1e-18)


def test_population_centered_rank_one():
    *_, branches = synthetic_universe()
    means = population_source_means(branches, diversity_profile(8, 0.2), 0.05, "confounded")
    for action in range(3):
        audit = audit_population_subspace(means[:, :, action], do_reward_mean(branches)[:, action])
        assert audit.numerical_rank <= 1
        assert np.isclose(audit.rank1_explained_variance, 1.0)


def test_population_do_in_affine_span():
    *_, branches = synthetic_universe()
    means = population_source_means(branches, diversity_profile(5, 0.15), 0.05, "confounded")
    truth = do_reward_mean(branches)
    for action in (0, 2):
        assert audit_population_subspace(means[:, :, action], truth[:, action]).affine_do_projection_residual < 1e-12


def test_base_contrast_zero():
    *_, branches = synthetic_universe()
    means = population_source_means(branches, diversity_profile(8, 0.2), 0.1, "confounded")
    assert np.linalg.norm(means[:, :, 1] - means[:, :, 1].mean(axis=0)) < 1e-12


def test_independent_contrast_zero():
    *_, branches = synthetic_universe()
    means = population_source_means(branches, diversity_profile(8, 0.2), 0.1,
                                    "independent_latents")
    assert np.linalg.norm(means - means.mean(axis=0)) < 1e-12


def test_source_shuffle_within_anchor_action():
    public, _ = synthetic_public()
    shuffled = shuffle_source_within_anchor_action(public["anchor_id"], public["action_index"],
                                                   public["source_id"], 10)
    for anchor in np.unique(public["anchor_id"]):
        for action in range(3):
            mask = (public["anchor_id"] == anchor) & (public["action_index"] == action)
            assert np.array_equal(np.sort(shuffled[mask]), np.sort(public["source_id"][mask]))


def test_svd_initialization_deterministic():
    *_, branches = synthetic_universe()
    means = population_source_means(branches, diversity_profile(5, 0.2), 0.05, "confounded")
    first, second = svd_initialization(means), svd_initialization(means)
    assert np.array_equal(first.loadings, second.loadings)
    assert np.array_equal(first.contrast_targets, second.contrast_targets)
    centered = means[:, :, 2] - means[:, :, 2].mean(axis=0)
    assert np.allclose(centered, first.loadings[:, 2, None] * first.contrast_targets[None, :, 2])
    for action in (0, 2):
        loading = first.loadings[:, action]
        assert loading[np.argmax(np.abs(loading))] >= 0


def test_loading_constraints():
    loading = normalize_loadings([-3, -1, 2, 8])
    assert np.isclose(loading.mean(), 0.0)
    assert np.isclose(np.mean(loading ** 2), 1.0)


@pytest.mark.skipif(sys.platform == "win32", reason="server PyTorch integration test")
def test_source_not_in_environment_network():
    model = make_source_free_model(3, np.zeros((3, 3)), seed=0)
    assert validate_source_free_model(model)
    assert model.g.network[0].in_features == 15
    assert model.h.network[0].in_features == 15


def test_hidden_u_not_used():
    signature = inspect.signature(fit_source_free_model)
    assert not {"u", "u_env", "u_behavior"}.intersection(signature.parameters)
    public, hidden = synthetic_public()
    assert set(public) == PUBLIC_FIELDS
    assert FORBIDDEN_PUBLIC_FIELDS.isdisjoint(public)
    assert {"u_env", "u_behavior", "epsilon"}.issubset(hidden)


def test_do_not_used_offline():
    signature = inspect.signature(fit_source_free_model)
    assert "do_reward" not in signature.parameters
    assert "reward_branches" not in signature.parameters


def test_rank0_calibration_closed_form():
    action = np.tile(np.arange(3), 5)
    x = calibration_features(action, np.zeros(len(action)), rank=0)
    delta = np.asarray((0.1, -0.2, 0.3))
    fit = closed_form_calibration(np.zeros(len(action)), x @ delta, x)
    assert np.allclose(fit.coefficients, delta)


def test_rank1_calibration_closed_form():
    action = np.tile(np.arange(3), 8)
    h = np.linspace(-1, 1, len(action))
    x = calibration_features(action, h, rank=1)
    theta = np.asarray((0.1, -0.2, 0.3, 0.4, -0.1, 0.2))
    fit = closed_form_calibration(np.zeros(len(action)), x @ theta, x)
    assert np.allclose(fit.prediction, x @ theta)


def test_pseudoinverse_minimum_norm():
    x = np.asarray([[1.0, 1.0]])
    fit = closed_form_calibration([0.0], [2.0], x)
    assert np.allclose(fit.coefficients, [1.0, 1.0])
    assert np.allclose(fit.coefficients, np.linalg.pinv(x) @ np.asarray([2.0]))


def test_bic_tie_prefers_rank0():
    assert bic_select_rank(1.0, 1.0, 8) == 0


def test_active_query_does_not_read_outcome():
    action = np.tile(np.arange(3), 4)
    anchor = np.repeat(np.arange(4), 3)
    h = np.linspace(-1, 1, len(action))
    phi = calibration_features(action, h, rank=1)
    first = active_query_order(phi, anchor, action, 6)
    second = active_query_order(phi, anchor, action, 6)
    assert np.array_equal(first, second)
    assert "reward" not in inspect.signature(active_query_order).parameters


def test_active_query_design_objective():
    action = np.tile(np.arange(3), 5)
    anchor = np.repeat(np.arange(5), 3)
    h = np.linspace(-1, 1, len(action))
    phi = calibration_features(action, h, rank=1)
    order = active_query_order(phi, anchor, action, 6)
    assert np.linalg.matrix_rank(phi[order[:3]]) >= 3
    assert pairwise_gap_features(phi, anchor, action).shape == (15, 6)


def test_nested_budgets():
    action = np.tile(np.arange(3), 10)
    anchor = np.repeat(np.arange(10), 3)
    order = random_balanced_query_order(anchor, action, 16, 0)
    assert budgets_are_nested(order, (0, 8, 16))


def test_fixed_draw_budget_across_M():
    ids, obs, action, branches = synthetic_universe()
    for m in (2, 3, 5, 8):
        public, _ = fixed_draw_public_table(ids, obs, action, branches,
                                            diversity_profile(m, 0.2), kappa=0,
                                            lambda_reward=0.05, sigma_reward=0,
                                            condition="confounded", sample_budget=384, seed=0)
        assert len(public["reward"]) == 384


def test_test_split_isolated():
    splits = {"train": range(0, 60), "observational_validation": range(60, 75),
              "do_calibration_pool": range(75, 90), "test": range(90, 120)}
    selected = select_anchor_splits(splits, range(120), 100, 16)
    groups = list(map(lambda x: set(map(int, x)), selected.values()))
    assert sum(map(len, groups)) == 100
    assert not any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))


def test_regret_nonnegative():
    truth = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    prediction = np.asarray([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    metrics = decision_metrics(truth, prediction)
    assert metrics["mean_regret"] >= 0
    assert metrics["worst_tie_mean_regret"] >= metrics["mean_regret"]


@pytest.mark.skipif(sys.platform == "win32", reason="server PyTorch integration test")
def test_checkpoint_roundtrip(tmp_path: Path):
    import torch
    loading = np.stack([normalize_loadings([-1, 0, 1])] * 3, axis=1)
    model = make_source_free_model(3, loading, seed=3)
    norm = {"x_mean": np.zeros(15), "x_std": np.ones(15),
            "reward_mean": np.asarray(0.0), "reward_std": np.asarray(1.0)}
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, norm, {"rank": 1})
    loaded, loaded_norm, metadata = load_checkpoint(path)
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), loaded.state_dict()[key].cpu())
    assert metadata["rank"] == 1
    assert np.array_equal(loaded_norm["x_mean"], norm["x_mean"])


def test_input_hashes_unchanged(tmp_path: Path):
    path = tmp_path / "input.bin"; path.write_bytes(b"immutable")
    before = input_hashes([path]); after = input_hashes([path])
    assert before == after


def test_no_nan_inf():
    public, _ = synthetic_public()
    assert validate_public_table(public)
    assert all(np.all(np.isfinite(public[name])) for name in
               ("observation", "commanded_action", "reward", "row_weight"))


def test_old_artifacts_unchanged(tmp_path: Path):
    old = tmp_path / "old.json"; old.write_text('{"fixed": true}', encoding="utf-8")
    before = input_hashes([old])
    _ = diversity_profile(8, 0.2)
    assert input_hashes([old]) == before
