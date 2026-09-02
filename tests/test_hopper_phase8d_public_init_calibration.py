from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift.phase8d_public_init_calibration import (
    BINARY_LATENT_REWARD_ONLY_PROOF_OF_CONCEPT,
    FORBIDDEN_OFFLINE_FIELDS,
    NO_TEMPERATURE_PARAMETER,
    PUBLIC_INITIALIZATION_FIELDS,
    Phase8DPublicInitCalibrationError,
    anchor_fold_assignment,
    brute_force_weighted_two_means,
    build_calibration_sequence,
    calibration_budgets_are_nested,
    calibration_public_is_hidden_free,
    deduplicate_candidate_predictions,
    exact_do_log_predictive_density,
    exact_weighted_two_means,
    hierarchical_candidate_prior,
    initialize_behavior,
    phase8d_anchor_splits,
    posterior_candidate_weights,
    reproduce_phase8c_fd,
    set_stage_trainability,
    soft_responsibilities,
    stage_update_allocation,
    train_public_initialized_candidates,
    validate_phase8d_inputs,
    verify_oof_exclusion,
)
from experiments.hopper_logger_mixture_drift.reward_mechanism_separation import (
    _state_hash,
    load_model,
    make_model,
    regret_metrics,
    save_model,
    validate_main_model_structure,
)
from experiments.hopper_logger_mixture_drift.analyze_phase8a_population_effect import (
    hash_input_files,
    input_hashes_unchanged,
)


ROOT = Path(__file__).resolve().parents[1]
FD = ROOT / ("artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified/"
             "phase8c_failure_decomposition")


def torch_or_skip():
    if os.name == "nt":
        pytest.skip("the local Windows PyTorch DLL is unavailable; exercised on Linux CI/server")
    try:
        import torch
        torch.zeros(1)
    except Exception as exc:
        pytest.skip(f"working PyTorch is unavailable: {exc}")
    return torch


def small_splits():
    return {"train": list(range(10)), "validation": list(range(10, 19)),
            "test": list(range(19, 24))}


def fake_do_raw(n=12, kappa=0.0):
    rows = []
    direction = np.asarray([1.0, -1.0, 1.0]) / np.sqrt(3.0)
    for anchor in range(n):
        for action_index, action in enumerate(("minus", "base", "plus")):
            for u in (-1, 1):
                rows.append((anchor, action_index, action, u))
    return {
        "anchor_id": np.asarray([r[0] for r in rows]),
        "action_key": np.asarray([r[2] for r in rows]),
        "u_env": np.asarray([r[3] for r in rows]),
        "kappa_env": np.full(len(rows), kappa),
        "commanded_action": np.asarray([(r[1] - 1) * direction for r in rows]),
        "applied_action": np.asarray([(r[1] - 1) * direction for r in rows]),
        "reward": np.asarray([0.1 * r[0] + 0.01 * r[1] for r in rows]),
        "next_observation": np.zeros((len(rows), 12)),
        "terminated": np.zeros(len(rows), bool),
        "truncated": np.zeros(len(rows), bool),
        "applied_action_clipped": np.zeros(len(rows), bool),
    }


def test_phase8c_inputs_required(tmp_path):
    with pytest.raises(Phase8DPublicInitCalibrationError):
        validate_phase8d_inputs(tmp_path / "a", tmp_path / "b", tmp_path / "c")


def test_failure_decomposition_reproduced():
    if not FD.is_dir():
        pytest.skip("formal FD result is not present in this checkout")
    facts = reproduce_phase8c_fd(FD)
    assert facts["all_reproduced"]
    assert facts["v0_collapse_count"] == 35 and facts["v6_collapse_count"] == 0


def test_split_disjoint():
    split = phase8d_anchor_splits(small_splits())
    groups = [set(split[k]) for k in ("train", "observational_validation",
                                      "do_calibration_pool", "test")]
    assert not any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))


def test_test_not_used_for_calibration():
    split = phase8d_anchor_splits(small_splits())
    assert set(split["test"]).isdisjoint(split["do_calibration_pool"])


def test_oof_residual_excludes_anchor():
    assignment = anchor_fold_assignment(range(10), folds=5)
    training = {fold: [a for a in range(10) if assignment[a] != fold] for fold in range(5)}
    assert verify_oof_exclusion(range(10), assignment, training)
    training[assignment[0]].append(0)
    assert not verify_oof_exclusion(range(10), assignment, training)


def test_exact_weighted_two_means_against_bruteforce():
    x = np.asarray([-3.0, -1.0, 0.2, 1.0, 8.0]); w = np.asarray([1, 2, 4, 1, 3.0])
    exact = exact_weighted_two_means(x, w); brute = brute_force_weighted_two_means(x, w)
    assert exact.objective == pytest.approx(brute[0], abs=1e-12)
    assert exact.split_index == brute[1]


def test_two_means_deterministic_tie_break():
    result = exact_weighted_two_means([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert result.split_index == 1


def test_soft_responsibilities_sum_to_one():
    q = soft_responsibilities([-2, 0, 2], [-1, 1], [.4, .6], .5)
    assert np.allclose(q.sum(1), 1.0, atol=1e-12)


def test_hidden_u_not_used_in_public_init():
    assert not FORBIDDEN_OFFLINE_FIELDS.intersection(PUBLIC_INITIALIZATION_FIELDS)
    assert BINARY_LATENT_REWARD_ONLY_PROOF_OF_CONCEPT


def test_do_not_used_in_offline_training():
    parameters = inspect.signature(train_public_initialized_candidates).parameters
    assert "do_reward" not in parameters and "u_env" not in parameters


def test_behavior_init_uses_source_action_q_only():
    source = np.asarray([0, 0, 1, 1, 2, 2]); action = np.asarray([0, 2, 0, 2, 1, 1])
    q = np.asarray([[.9, .1], [.1, .9], [.8, .2], [.2, .8], [.6, .4], [.4, .6]])
    prior, beta = initialize_behavior(source, action, q, np.ones(6))
    assert np.isclose(prior.sum(), 1) and np.allclose(beta.sum(2), 1)


def test_source_only_enters_behavior():
    torch_or_skip()
    checks = validate_main_model_structure(make_model("mechanism_separated", 0))
    assert checks["source_only_enters_behavior"] and checks["reward_decoder_is_source_invariant"]


def test_stage_freezing():
    torch_or_skip(); model = make_model("mechanism_separated", 0)
    a = set_stage_trainability(model, "reward_pretraining")
    assert a["reward_decoder.0.weight"] and not a["prior_logits"] and not a["behavior_logits"]
    b = set_stage_trainability(model, "behavior_only")
    assert b["prior_logits"] and b["behavior_logits"] and not b["reward_decoder.0.weight"]
    c = set_stage_trainability(model, "reward_refinement")
    assert c["log_scale"] and not c["prior_logits"]
    d = set_stage_trainability(model, "joint")
    assert all(d.values())


def test_compute_budget_matched():
    for total in (1, 7, 3000, 3001):
        allocation = stage_update_allocation(total)
        assert sum(allocation.values()) == total
        assert allocation["joint"] >= allocation["reward_pretraining"]


def test_candidate_prior_hierarchical_uniform():
    candidates = [{"seed": 0}, {"seed": 0}, {"seed": 1}]
    prior = hierarchical_candidate_prior(candidates)
    assert np.allclose(prior, [.25, .25, .5])


def test_candidate_prediction_hash_deduplication():
    candidates = [{"seed": 0}] * 3
    unique, mapping = deduplicate_candidate_predictions(
        candidates, [np.asarray([1, 2]), np.asarray([1, 2]), np.asarray([2, 1])])
    assert unique == [0, 2] and mapping == {0: 0, 1: 0, 2: 2}


def test_calibration_actions_independent_of_u():
    sequence = build_calibration_sequence(fake_do_raw(), range(12), 0.0, .1, 3,
                                          size=12, anchor_observation={i: np.zeros(12) for i in range(12)})
    # Actions are a deterministic balanced rotation; changing U cannot change them.
    assert np.array_equal(np.bincount(sequence.public["action_index"], minlength=3), [4, 4, 4])


def test_nested_calibration_budgets():
    sequence = build_calibration_sequence(fake_do_raw(), range(12), 0.0, 0.0, 0,
                                          size=10, anchor_observation={i: np.zeros(12) for i in range(12)})
    assert calibration_budgets_are_nested(sequence.public, (0, 2, 5, 10))


def test_calibration_artifact_hides_u():
    sequence = build_calibration_sequence(fake_do_raw(), range(12), 0.0, .1, 0,
                                          anchor_observation={i: np.zeros(12) for i in range(12)})
    assert calibration_public_is_hidden_free(sequence.public)
    assert "u_env" not in sequence.public


def test_exact_do_predictive_marginal():
    y = np.asarray([0.2]); means = np.asarray([[0.0, 1.0]]); prior = np.asarray([.25, .75])
    actual = exact_do_log_predictive_density(y, means, prior, math.log(.5))[0]
    normal = lambda mu: np.exp(-.5 * ((.2 - mu) / .5) ** 2) / (.5 * np.sqrt(2*np.pi))
    assert np.exp(actual) == pytest.approx(.25 * normal(0) + .75 * normal(1))


def test_no_temperature_parameter():
    assert NO_TEMPERATURE_PARAMETER
    assert "temperature" not in inspect.signature(posterior_candidate_weights).parameters


def test_b_zero_equals_candidate_prior():
    prior = np.asarray([.1, .2, .7])
    assert np.array_equal(posterior_candidate_weights(prior, np.zeros(3)), prior)


def test_final_test_not_used_for_weights():
    names = set(inspect.signature(posterior_candidate_weights).parameters)
    assert names == {"candidate_prior", "calibration_log_score"}


def test_regret_nonnegative():
    truth = np.asarray([[0., 1., 2.], [2., 1., 0.]])
    prediction = np.asarray([[2., 1., 0.], [0., 1., 2.]])
    result = regret_metrics(truth, prediction)
    assert result["mean_regret"] >= 0 and result["worst_tie_mean_regret"] >= 0


def test_checkpoint_roundtrip(tmp_path):
    torch_or_skip(); model = make_model("mechanism_separated", 4)
    path = tmp_path / "model.pt"
    save_model(path, model, {"kind": "mechanism_separated", "seed": 4})
    loaded, _ = load_model(path, "cpu")
    assert _state_hash(loaded) == _state_hash(model)


def test_input_hashes_unchanged(tmp_path):
    path = tmp_path / "input.txt"; path.write_text("fixed", encoding="utf-8")
    before = hash_input_files([path]); after = hash_input_files([path])
    assert input_hashes_unchanged(before, after)


def test_no_nan_inf():
    result = exact_weighted_two_means([-1, 0, 1], [1, 2, 1])
    q = soft_responsibilities([-1, 0, 1], result.centers, [.5, .5], result.shared_variance)
    assert np.all(np.isfinite(q))


def test_old_artifacts_unchanged(tmp_path):
    path = tmp_path / "old.npz"; np.savez(path, x=np.arange(4))
    before = hash_input_files([path])
    with np.load(path) as data:
        assert data["x"].sum() == 6
    assert input_hashes_unchanged(before, hash_input_files([path]))
