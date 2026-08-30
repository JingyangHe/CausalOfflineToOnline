"""Focused tests for the Phase 8A logger-mixture causal-drift DGP."""

from pathlib import Path

import numpy as np

from confounded_hopper import ACTUATOR_DIRECTION
from experiments.hopper_logger_mixture_drift.anchor_pool import checkpoint_roundtrip
from experiments.hopper_logger_mixture_drift.audit import (
    _population_equals_do,
    anchor_distribution_audit,
    hard_invariants,
    latent_weighted_correlation,
    make_figures,
    population_observational_table,
    summarize_population_table,
)
from experiments.hopper_logger_mixture_drift.controlled_loggers import (
    MIXTURES,
    controlled_action,
    latent_pairs,
)
from experiments.hopper_logger_mixture_drift.generate_datasets import (
    FORBIDDEN_PUBLIC_FIELDS,
    all_arrays_finite,
    deterministic_repeat_check,
    generate_condition_dataset,
    generate_do_oracle,
    generate_mixture_weights,
    validate_public_hidden,
)


class FakeOneStepSimulator:
    def __init__(self, anchors):
        self.anchors = anchors

    def step(self, anchor_index, commanded_action, u_env, kappa_env):
        command = np.asarray(commanded_action, dtype=np.float64)
        preclip = command + kappa_env * u_env * ACTUATOR_DIRECTION
        applied = np.clip(preclip, -1.0, 1.0)
        observation = self.anchors["public_observation"][anchor_index].astype(np.float64)
        next_observation = observation.copy()
        next_observation[:3] += 0.01 * applied
        next_observation[-1] = max(0.0, observation[-1] - 0.001)
        reward = float(applied @ np.asarray((1.0, 0.5, -0.25)))
        qpos = self.anchors["qpos"][anchor_index].copy()
        qvel = self.anchors["qvel"][anchor_index].copy()
        return {
            "observation": observation, "commanded_action": command,
            "applied_action": applied, "reward": reward,
            "next_observation": next_observation, "terminated": False, "truncated": False,
            "qpos": qpos, "qvel": qvel, "next_qpos": qpos + 0.01,
            "next_qvel": qvel + applied.mean(),
            "commanded_action_clipped": bool(np.any(np.abs(command) > 1.0)),
            "applied_action_clipped": bool(np.any(np.abs(preclip) > 1.0)),
        }


class FakeSource2:
    def predict(self, observations, deterministic=True):
        values = np.asarray(observations)
        batch = values.reshape(-1, 13)
        actions = np.column_stack((0.05 * batch[:, -1], -0.02 * batch[:, -1],
                                   0.01 * batch[:, -1]))
        return (actions[0] if values.ndim == 1 else actions), None

    def save(self, path):
        Path(str(path) + ".zip").write_text("fake checkpoint", encoding="utf-8")


def _fake_loader(path, device="cpu"):
    assert Path(path).is_file()
    return FakeSource2()


def _anchors(count=3):
    elapsed = np.asarray((0, 20, 40), dtype=np.int64)[:count]
    observation = np.zeros((count, 12), dtype=np.float32)
    observation[:, 0] = np.arange(count) * 0.1
    observation[:, -1] = (1000 - elapsed) / 1000.0
    return {
        "anchor_id": np.arange(count, dtype=np.int64),
        "public_observation": observation,
        "base_action": np.zeros((count, 3), dtype=np.float64),
        "qpos": np.zeros((count, 6), dtype=np.float64),
        "qvel": np.zeros((count, 6), dtype=np.float64),
        "elapsed_steps": elapsed,
    }


def _bundle(kappa):
    anchors = _anchors(); simulator = FakeOneStepSimulator(anchors)
    do_raw, do_summary = generate_do_oracle(anchors, kappa, 0.2, simulator)
    datasets, weights, tables = {}, {}, {}
    for condition in ("confounded", "independent_latents"):
        public, hidden = generate_condition_dataset(anchors, condition, kappa, 0.2, simulator)
        condition_weights = generate_mixture_weights(hidden)
        datasets[condition] = (public, hidden); weights[condition] = condition_weights
        tables[condition] = population_observational_table(
            anchors, public, hidden, condition_weights, do_summary
        )
    return anchors, simulator, do_raw, do_summary, datasets, weights, tables


def test_controlled_logger_action_overlap():
    base = np.asarray((0.1, -0.2, 0.3))
    l1_plus, _ = controlled_action(base, 0, 1, 0.2)
    l2_minus, _ = controlled_action(base, 1, -1, 0.2)
    l1_minus, _ = controlled_action(base, 0, -1, 0.2)
    l2_plus, _ = controlled_action(base, 1, 1, 0.2)
    np.testing.assert_allclose(l1_plus, l2_minus, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(l1_minus, l2_plus, atol=1e-15, rtol=0.0)


def test_confounded_latent_pairing():
    pairs = latent_pairs("confounded")
    assert {(left, right) for left, right, _ in pairs} == {(-1, -1), (1, 1)}
    assert sum(mass for _, _, mass in pairs) == 1.0


def test_independent_latent_factorization():
    pairs = latent_pairs("independent_latents")
    assert len(pairs) == 4 and all(mass == 0.25 for _, _, mass in pairs)
    _, _, _, _, datasets, _, _ = _bundle(0.2)
    assert abs(latent_weighted_correlation(datasets["independent_latents"][1])) < 1e-15


def test_commanded_action_marginals_match_across_conditions():
    anchors, _, do_raw, _, datasets, weights, tables = _bundle(0.2)
    invariants = hard_invariants(
        anchors, datasets, weights, tables, do_raw, 0.2, 0.2, True, True
    )
    assert invariants["commanded_action_marginals_match_conditions"]


def test_anchor_restore_is_deterministic():
    anchors = _anchors(); simulator = FakeOneStepSimulator(anchors)
    result = deterministic_repeat_check(simulator, 0, np.zeros(3), 1, 0.2)
    assert result["passed"]


def test_kappa_zero_removes_u_env_effect():
    anchors, _, do_raw, _, datasets, weights, tables = _bundle(0.0)
    invariants = hard_invariants(
        anchors, datasets, weights, tables, do_raw, 0.0, 0.2, True, True
    )
    assert invariants["kappa_zero_removes_u_env_effect"]


def test_independent_population_equals_do_oracle():
    *_, tables = _bundle(0.2)
    assert _population_equals_do(tables["independent_latents"])


def test_do_oracle_accumulates_public_observations_in_float64():
    anchors = _anchors(); simulator = FakeOneStepSimulator(anchors)
    _, do_summary = generate_do_oracle(anchors, 0.2, 0.2, simulator)
    assert do_summary["do_mean_next_observation"].dtype == np.float64
    assert do_summary["do_mean_delta_observation"].dtype == np.float64


def test_confounded_kappa_zero_equals_do_oracle():
    *_, tables = _bundle(0.0)
    assert _population_equals_do(tables["confounded"])


def test_mixture_weights_keep_anchor_distribution_fixed():
    _, _, _, _, datasets, weights, _ = _bundle(0.2)
    public = datasets["confounded"][0]
    result = anchor_distribution_audit(public["anchor_id"], weights["confounded"])
    assert result["passed"]
    assert set(result["maximum_absolute_deviation"]) == set(MIXTURES)


def test_public_hidden_leakage():
    _, _, _, _, datasets, _, _ = _bundle(0.2)
    public, hidden = datasets["confounded"]
    assert validate_public_hidden(public, hidden) == set()
    assert FORBIDDEN_PUBLIC_FIELDS.isdisjoint(public)


def test_source2_checkpoint_roundtrip():
    result = checkpoint_roundtrip(FakeSource2(), _fake_loader, _anchors()["public_observation"], "cpu")
    assert result["passed"] and result["maximum_base_action_difference"] == 0.0


def test_no_nan_inf():
    anchors, _, do_raw, do_summary, datasets, weights, tables = _bundle(0.2)
    assert all_arrays_finite(anchors, do_raw, do_summary, *(
        bundle for condition in datasets.values() for bundle in condition
    ), *(weights.values()), *(tables.values()))


def test_time_to_go_consistency():
    anchors, _, do_raw, _, datasets, weights, tables = _bundle(0.2)
    invariants = hard_invariants(
        anchors, datasets, weights, tables, do_raw, 0.2, 0.2, True, True
    )
    assert invariants["time_to_go_consistent_with_existing_wrapper"]


def test_oracle_is_mixture_independent():
    _, _, _, do_summary, _, _, tables = _bundle(0.2)
    assert "mixture" not in do_summary
    for condition in tables.values():
        for anchor in np.unique(condition["anchor_id"]):
            for action_key in ("minus", "base", "plus"):
                mask = (condition["anchor_id"] == anchor) & (condition["action_key"] == action_key)
                assert np.unique(condition["do_mean_reward"][mask]).size == 1


def test_required_figures_are_individual_files(tmp_path):
    by_kappa = {}
    for kappa in (0.0, 0.2):
        *_, tables = _bundle(kappa)
        population = {condition: summarize_population_table(table)[0]
                      for condition, table in tables.items()}
        mixture_means = {
            condition: {name: float(np.mean(table["observational_mean_reward"][table["mixture"] == name]))
                        for name in MIXTURES}
            for condition, table in tables.items()
        }
        by_kappa[str(kappa)] = {"kappa_env": kappa, "population": population,
                                "mixture_mean_reward": mixture_means}
    make_figures(tmp_path, by_kappa)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "reward_prediction_vs_mixture.png", "next_state_drift_vs_kappa.png",
        "do_error_vs_kappa.png", "action_ranking_flip_vs_kappa.png",
    }
