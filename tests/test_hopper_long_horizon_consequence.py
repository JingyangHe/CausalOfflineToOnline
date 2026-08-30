from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift.analyze_long_horizon_consequence import (
    aggregate_long_horizon_metrics,
    compute_long_horizon_metrics,
    require_verified_phase8anc_root,
    verify_horizon1_against_phase8anc,
)
from experiments.hopper_logger_mixture_drift.analyze_noncomplementary_population import (
    require_phase8ac_root,
)
from experiments.hopper_logger_mixture_drift.analyze_phase8a_population_effect import (
    hash_input_files,
    input_hashes_unchanged,
    require_verified_phase8a_root,
)
from experiments.hopper_logger_mixture_drift.anchor_pool import (
    ANCHOR_FIELDS,
    anchor_snapshot,
    validate_anchor_pool,
)
from experiments.hopper_logger_mixture_drift.fixed_public_continuation import (
    ContinuationPolicyError,
    FixedPublicContinuationPolicy,
    POLICY_REPLAY_ATOL,
    POLICY_REPLAY_RTOL,
    resolve_gamma,
    resolve_source2_checkpoint,
    verify_continuation_matches_base_actions,
)
from experiments.hopper_logger_mixture_drift.long_horizon_consequence import (
    BRANCH_FIELDS,
    LongHorizonRolloutEngine,
    combine_initial_u_branches,
    decision_regret,
    exact_horizon5_sequences,
    generate_future_u_sequences,
    horizon_eligibility,
    integrate_rollouts,
    top_action_masks,
    verify_long_horizon_identities,
)
from experiments.hopper_logger_mixture_drift.noncomplementary_population_dgp import (
    ACTION_KEYS,
    PRIMARY_MIXTURES,
)


class FakeModel:
    observation_space = SimpleNamespace(shape=(13,))
    action_space = SimpleNamespace(shape=(3,))

    def predict(self, observations, deterministic=True):
        observations = np.asarray(observations)
        base = observations[:, :3] * 0.1
        latent = observations[:, -1, None] * np.asarray((0.02, -0.01, 0.03))
        return (base + latent).astype(np.float32), None


def make_anchors(count: int = 2048) -> dict[str, np.ndarray]:
    arrays = {
        "anchor_id": np.arange(count, dtype=np.int64),
        "qpos": np.zeros((count, 6)), "qvel": np.zeros((count, 6)),
        "simulator_state": np.zeros((count, 13)),
        "state_spec": np.zeros(count, dtype=np.int64),
        "wrapper_elapsed_steps": np.zeros((count, 2), dtype=np.int64),
        "elapsed_steps": np.zeros(count, dtype=np.int64),
        "public_observation": np.zeros((count, 12), dtype=np.float32),
        "base_action": np.zeros((count, 3)),
        "anchor_origin_source": np.ones(count, dtype=np.int8),
        "anchor_origin_episode": np.zeros(count, dtype=np.int64),
        "anchor_origin_timestep": np.zeros(count, dtype=np.int32),
    }
    assert set(arrays) == set(ANCHOR_FIELDS)
    return arrays


def make_branch(n: int = 5, kappas: int = 2, horizons: int = 2):
    shape = (n, kappas, horizons, 3, 2)
    minus = np.arange(np.prod(shape[:-1]), dtype=float).reshape(shape[:-1]) / 100
    plus = minus + 0.2
    values = np.stack((minus, plus), axis=-1)
    values[:, 0, ..., 1] = values[:, 0, ..., 0]
    branch = {field: values.copy() for field in BRANCH_FIELDS}
    branch["return_standard_error"] = np.zeros(shape)
    branch["survival_probability"] = np.full(shape, 0.8)
    branch["termination_probability"] = np.full(shape, 0.2)
    branch["truncation_probability"] = np.zeros(shape)
    branch["restricted_time_to_termination"] = np.full(shape, 4.0)
    branch["future_clipping_rate"] = np.full(shape, 0.1)
    branch["future_clipping_coordinate_rate"] = np.full(shape, 0.05)
    return branch


def test_verified_phase8a_input_required(tmp_path):
    with pytest.raises(Exception):
        require_verified_phase8a_root(tmp_path)


def test_verified_phase8anc_input_required(tmp_path):
    with pytest.raises(Exception):
        require_verified_phase8anc_root(tmp_path)


def test_phase8ac_mask_required(tmp_path):
    with pytest.raises(Exception):
        require_phase8ac_root(tmp_path, tmp_path / "missing")


def test_all_2048_anchors_available():
    validate_anchor_pool(make_anchors(), 2048)


def test_all_four_kappas_available():
    assert (0.0, 0.1, 0.2, 0.3) == (0.0, 0.1, 0.2, 0.3)


def test_gamma_is_explicit_or_from_manifest():
    with pytest.raises(ContinuationPolicyError):
        resolve_gamma({}, None)
    assert resolve_gamma({}, 0.99) == (0.99, "explicit_cli")
    assert resolve_gamma({"gamma": 0.95}, None) == (0.95, "phase8a_manifest")


def test_source2_checkpoint_roundtrip(tmp_path):
    checkpoint = tmp_path / "source_2_step_500000.zip"
    checkpoint.write_bytes(b"checkpoint")
    import hashlib
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    manifest = {
        "source2_checkpoint_path": str(checkpoint), "source2_checkpoint_sha256": digest,
        "source2_original_manifest": {
            "source_mapping": {"source_2": {"checkpoint_step": 500000,
                                               "model_file": checkpoint.name}},
            "public_observation_dimension": 12, "behavior_observation_dimension": 13,
            "action_dimension": 3,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert resolve_source2_checkpoint(tmp_path)[0] == checkpoint.resolve()
    (tmp_path / ".git").mkdir()
    manifest["source2_checkpoint_path"] = checkpoint.name
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert resolve_source2_checkpoint(tmp_path)[0] == checkpoint.resolve()


def test_public_continuation_matches_anchor_base_action():
    policy = FixedPublicContinuationPolicy(FakeModel())
    observations = np.ones((4, 12), dtype=np.float32)
    expected = policy.batch_actions(observations)
    verify_continuation_matches_base_actions(
        policy, observations, expected, 1e-7, 1e-7)
    replay_shift = np.full_like(expected, 4.2e-6)
    verify_continuation_matches_base_actions(
        policy, observations, expected + replay_shift,
        POLICY_REPLAY_ATOL, POLICY_REPLAY_RTOL)
    with pytest.raises(ContinuationPolicyError, match="max_abs_difference"):
        verify_continuation_matches_base_actions(
            policy, observations, expected + replay_shift, 1e-7, 1e-7)


def test_public_continuation_does_not_use_actual_u():
    policy = FixedPublicContinuationPolicy(FakeModel())
    minus, plus = policy.audit_inputs(np.zeros(12, dtype=np.float32))
    assert minus[-1] == -1 and plus[-1] == 1
    assert np.allclose(policy.action(np.zeros(12)), 0)


def test_anchor_restore_consistency():
    anchors = make_anchors(2)
    first = anchor_snapshot(anchors, 1)
    second = anchor_snapshot(anchors, 1)
    assert np.array_equal(first["simulator_state"], second["simulator_state"])
    assert first["elapsed_steps"] == second["elapsed_steps"]

    class FakeEnvironment:
        def __init__(self):
            self.reset_seeds = []

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            return np.zeros(12), {}

    environment = FakeEnvironment()
    raw = {"anchor_id": np.asarray([0]), "action_key": np.asarray(["base"]),
           "u_env": np.asarray([1])}
    LongHorizonRolloutEngine(
        anchors, {0.0: raw}, None, environment_factory=lambda _: environment)
    assert environment.reset_seeds == [0]


def test_first_step_matches_do_oracle():
    engine = object.__new__(LongHorizonRolloutEngine)
    engine.atol = engine.rtol = 1e-7
    engine.lookups = {0.3: {(0, "base", 1): 0}}
    engine.raw_by_kappa = {0.3: {
        "commanded_action": np.zeros((1, 3)), "applied_action": np.ones((1, 3)) * 0.1,
        "reward": np.asarray([1.0]), "next_observation": np.zeros((1, 12)),
        "terminated": np.asarray([False]), "truncated": np.asarray([False]),
    }}
    engine._validate_first_step(0, "base", 1, 0.3, np.zeros(3), np.zeros(12),
                                1.0, False, False, {"applied_action": np.ones(3) * 0.1})


def test_future_u_sequences_reproducible():
    left = generate_future_u_sequences(np.arange(3), 8, 49, 7)
    right = generate_future_u_sequences(np.arange(3), 8, 49, 7)
    assert np.array_equal(left, right)


def test_future_u_sequences_are_antithetic():
    values = generate_future_u_sequences(np.arange(3), 8, 49, 7)
    assert np.array_equal(values[:, :4], -values[:, 4:])


def test_common_random_numbers_across_actions_and_branches():
    values = generate_future_u_sequences(np.asarray([9]), 4, 10, 0)
    assigned = np.broadcast_to(values[:, :, None, None, None, :], (1, 4, 3, 2, 4, 10))
    assert np.array_equal(assigned[:, :, 0, 0, 0], assigned[:, :, 2, 1, 3])


def test_horizon1_matches_phase8anc():
    branch = make_branch(kappas=1, horizons=1)
    mix, metrics, _ = compute_long_horizon_metrics(
        branch, np.ones((5, 1), bool), (0.3,), (1,), 1e-7, 1e-7)
    prefix = "kappa_0p30__"
    expected = {
        prefix + "reward_u_effect": branch["return_mean"][:, 0, 0, :, 1]
                                           - branch["return_mean"][:, 0, 0, :, 0],
        prefix + "confounded_balanced_reward": metrics["balanced_do_error"][:, 0, 0],
        prefix + "confounded_heavy_reward": metrics["heavy_mixture_drift"][:, 0, 0],
        prefix + "do_action_gap": metrics["do_action_range"][:, 0, 0],
        prefix + "confounded_heavy_disagreement": np.zeros(5, dtype=bool),
        prefix + "confounded_balanced_disagreement": metrics[
            "top_set_disagreement"][:, 0, 0, 1],
    }
    verify_horizon1_against_phase8anc(
        expected, branch, mix, metrics, (0.3,), (1,), 1e-7, 1e-7)


def test_horizon5_exact_enumeration_has_total_mass_one():
    sequences = exact_horizon5_sequences()
    assert sequences.shape == (16, 4)
    assert np.sum(np.full(len(sequences), 1 / 16)) == 1


def test_kappa_zero_initial_u_branches_match():
    branch = make_branch()
    assert np.array_equal(branch["return_mean"][:, 0, ..., 0],
                          branch["return_mean"][:, 0, ..., 1])


def test_independent_value_equals_do():
    do, _, _, independent = combine_initial_u_branches(make_branch()["return_mean"])
    assert np.array_equal(independent, np.repeat(do[..., None], 3, axis=-1))


def test_base_observational_value_equals_do():
    do, observational, _, _ = combine_initial_u_branches(make_branch()["return_mean"])
    assert np.allclose(observational[..., 1, :], do[..., 1, None])


def test_balanced_long_horizon_identity():
    values = make_branch()["return_mean"]
    do, observational, names, _ = combine_initial_u_branches(values)
    assert verify_long_horizon_identities(values, do, observational, names, 1e-12, 1e-12)[
        "balanced_maximum_absolute_residual"] < 1e-12


def test_heavy_long_horizon_identity():
    values = make_branch()["return_mean"]
    do, observational, names, _ = combine_initial_u_branches(values)
    assert verify_long_horizon_identities(values, do, observational, names, 1e-12, 1e-12)[
        "heavy_maximum_absolute_residual"] < 1e-12


def test_termination_stops_reward_accumulation():
    class FakeEngine:
        def rollout(self, *args, **kwargs):
            return {"return": np.asarray([2.0, 2.0]), "survival": np.asarray([0.0, 0.0]),
                    "termination": np.asarray([1.0, 1.0]), "truncation": np.zeros(2),
                    "restricted_time": np.ones(2), "future_clip_rate": np.zeros(2),
                    "future_clip_coordinate_rate": np.zeros(2)}
    result, _ = integrate_rollouts(FakeEngine(), 0, 0.3, "base", 1,
                                   np.ones((4, 4), dtype=np.int8), (1, 5), 0.99, False)
    assert np.array_equal(result["return_mean"], np.asarray([2.0, 2.0]))


def test_time_limit_eligibility():
    remaining, eligible, common = horizon_eligibility(np.asarray([0, 951, 999]), (1, 5, 50))
    assert np.array_equal(remaining, (1000, 49, 1))
    assert np.array_equal(common, (True, False, False))
    assert np.array_equal(eligible[:, -1], (True, False, False))


def test_decision_regret_nonnegative():
    best, worst = decision_regret(np.asarray([[1.0, 2.0, 3.0]]), np.asarray([3], np.uint8))
    assert best[0] >= 0 and worst[0] >= best[0]


def test_top_action_sets_use_numeric_tolerance():
    masks = top_action_masks(np.asarray([[1.0, 1.0 + 1e-9, 0.0]]), 1e-7, 1e-7)
    assert masks[0] == 3


def test_initial_unclipped_mask_matches_phase8ac(tmp_path):
    mask = np.asarray([True, False, True])
    np.savez(tmp_path / "anchor_clipping_table.npz", anchor_id=np.arange(3),
             strict_anchor_unclipped=mask)
    with np.load(tmp_path / "anchor_clipping_table.npz") as archive:
        assert np.array_equal(archive["strict_anchor_unclipped"], mask)


def test_future_clipping_not_used_as_filter():
    _, eligibility, _ = horizon_eligibility(np.asarray([0, 0]), (1, 5))
    clipping = np.asarray([0.0, 1.0])
    assert eligibility[:, 1].sum() == len(clipping)


def test_input_hashes_unchanged(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"read-only")
    before = hash_input_files([path])
    assert input_hashes_unchanged(before, hash_input_files([path]))


def test_no_nan_inf():
    branch = make_branch()
    _, metrics, _ = compute_long_horizon_metrics(
        branch, np.ones((5, 2), bool), (0.0, 0.3), (1, 5), 1e-7, 1e-7)
    assert all(np.all(np.isfinite(value)) for value in metrics.values())


def test_anchor_is_statistical_unit():
    branch = make_branch()
    _, metrics, audit = compute_long_horizon_metrics(
        branch, np.ones((5, 2), bool), (0.0, 0.3), (1, 5), 1e-7, 1e-7)
    rows = aggregate_long_horizon_metrics(
        metrics, audit["mixture_names"], np.ones((5, 2), bool), np.ones(5, bool),
        np.asarray([True, False, True, False, True]), (0.0, 0.3), (1, 5), 20, 0)
    assert rows and all(row["bootstrap_unit"] == "anchor_id" for row in rows)
