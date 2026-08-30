"""Focused tests for the read-only Phase 8A-R population-effect review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from confounded_hopper import ACTUATOR_DIRECTION
from experiments.hopper_logger_mixture_drift.analyze_phase8a_population_effect import (
    EXPECTED_KAPPAS,
    PopulationEffectAuditError,
    all_arrays_finite,
    analyze_kappa,
    analytic_u_posterior,
    crosscheck_existing_summary,
    hash_input_files,
    input_hashes_unchanged,
    load_condition_weights,
    load_npz,
    midpoint_weight,
    recompute_do_oracle,
    recover_exact_action_groups,
    require_verified_phase8a_root,
    run_review,
    validate_all_2048_anchors,
    validate_all_84_phase8a_invariants,
    validate_all_four_kappas,
    validate_public_schema,
    validate_weight_array,
    verify_primary_weights_preserve_exact_groups,
)
from experiments.hopper_logger_mixture_drift.audit import (
    outcome_strength,
    population_observational_table,
    summarize_population_table,
)
from experiments.hopper_logger_mixture_drift.controlled_loggers import CONDITIONS, MIXTURES
from experiments.hopper_logger_mixture_drift.generate_datasets import (
    FORBIDDEN_PUBLIC_FIELDS,
    generate_condition_dataset,
    generate_do_oracle,
    generate_mixture_weights,
)


class FakeOneStepSimulator:
    def __init__(self, anchors: dict[str, np.ndarray]):
        self.anchors = anchors

    def step(self, anchor_index, commanded_action, u_env, kappa_env):
        command = np.asarray(commanded_action, dtype=np.float64)
        applied = np.clip(command + kappa_env * u_env * ACTUATOR_DIRECTION, -1.0, 1.0)
        observation = self.anchors["public_observation"][anchor_index].astype(np.float64)
        following = observation.copy()
        following[:3] += 0.05 * applied
        following[3:6] += 0.01 * applied
        following[-1] = max(0.0, observation[-1] - 0.001)
        reward = float(applied @ np.asarray((1.0, 0.5, -0.25)))
        qpos = self.anchors["qpos"][anchor_index].copy()
        qvel = self.anchors["qvel"][anchor_index].copy()
        return {
            "observation": observation,
            "commanded_action": command,
            "applied_action": applied,
            "reward": reward,
            "next_observation": following,
            "terminated": False,
            "truncated": False,
            "qpos": qpos,
            "qvel": qvel,
            "next_qpos": qpos + 0.01,
            "next_qvel": qvel + applied.mean(),
            "commanded_action_clipped": False,
            "applied_action_clipped": False,
        }


def _anchors(count: int) -> dict[str, np.ndarray]:
    observation = np.zeros((count, 12), dtype=np.float32)
    observation[:, 0] = np.arange(count, dtype=np.float32) * 0.1
    observation[:, -1] = 1.0 - np.arange(count, dtype=np.float32) * 0.001
    return {
        "anchor_id": np.arange(count, dtype=np.int64),
        "public_observation": observation,
        "base_action": np.zeros((count, 3), dtype=np.float64),
        "qpos": np.zeros((count, 6), dtype=np.float64),
        "qvel": np.zeros((count, 6), dtype=np.float64),
        "elapsed_steps": np.arange(count, dtype=np.int64),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _kappa_name(kappa: float) -> str:
    return f"kappa_{kappa:.2f}".replace(".", "p")


@pytest.fixture(scope="module")
def phase8a_artifact(tmp_path_factory):
    count = 4
    root = tmp_path_factory.mktemp("phase8a") / "controlled_loggers_seed0_verified"
    root.mkdir()
    anchors = _anchors(count)
    np.savez_compressed(root / "anchors.npz", **anchors)
    _write_json(root / "mixture_weights.json", {
        name: list(values) for name, values in MIXTURES.items()
    })
    _write_json(root / "manifest.json", {
        "phase": "8A", "kappas": list(EXPECTED_KAPPAS), "number_of_anchors": count,
        "numerical_tolerance": {"atol": 1e-7, "rtol": 1e-7},
    })
    (root / "phase8a_complete.log").write_text(
        "PHASE8A_HOPPER_LOGGER_MIXTURE_DGP_COMPLETE\n"
        "READY_FOR_POOLED_WORLD_MODEL_DRIFT_TRAINING\n", encoding="utf-8"
    )
    summary = {
        "phase": "8A", "anchor_count": count,
        "all_hard_invariants": {f"check_{index:02d}": True for index in range(84)},
        "all_hard_invariants_passed": True, "by_kappa": {},
    }
    simulator = FakeOneStepSimulator(anchors)
    for kappa in EXPECTED_KAPPAS:
        directory = root / _kappa_name(kappa)
        directory.mkdir()
        raw, stored = generate_do_oracle(anchors, kappa, 0.2, simulator)
        np.savez_compressed(directory / "do_oracle_raw.npz", **raw)
        np.savez_compressed(directory / "do_oracle_summary.npz", **stored)
        population = {}
        for condition in CONDITIONS:
            public, hidden = generate_condition_dataset(
                anchors, condition, kappa, 0.2, simulator
            )
            weights = generate_mixture_weights(hidden)
            np.savez_compressed(directory / f"{condition}_public.npz", **public)
            np.savez_compressed(directory / f"{condition}_hidden_audit.npz", **hidden)
            weight_directory = directory / "weights" / condition
            weight_directory.mkdir(parents=True)
            for mixture, values in weights.items():
                np.save(weight_directory / f"weights_{mixture}.npy", values)
            table = population_observational_table(
                anchors, public, hidden, weights, stored
            )
            population[condition] = summarize_population_table(table)[0]
        summary["by_kappa"][_kappa_name(kappa)] = {
            "outcome_strength": outcome_strength(raw), "population": population,
        }
    _write_json(root / "summary.json", summary)
    return root


@pytest.fixture(scope="module")
def analyzed(phase8a_artifact):
    anchors = load_npz(phase8a_artifact / "anchors.npz")
    anchor_ids = anchors["anchor_id"]
    contexts, arrays, checks = {}, {}, {}
    for kappa in EXPECTED_KAPPAS:
        context, specs, current_arrays, current_checks = analyze_kappa(
            phase8a_artifact, kappa, anchors, anchor_ids, 1e-7, 1e-7
        )
        contexts[kappa] = context
        arrays[kappa] = current_arrays
        checks[kappa] = current_checks
        assert specs and all(spec["values"].shape == (len(anchor_ids),) for spec in specs)
    return contexts, arrays, checks


def test_verified_phase8a_root_required(tmp_path, phase8a_artifact):
    assert require_verified_phase8a_root(phase8a_artifact) == phase8a_artifact.resolve()
    with pytest.raises(PopulationEffectAuditError):
        require_verified_phase8a_root(tmp_path)


def test_all_2048_anchors_present():
    assert len(validate_all_2048_anchors({"anchor_id": np.arange(2048)})) == 2048
    with pytest.raises(PopulationEffectAuditError):
        validate_all_2048_anchors({"anchor_id": np.arange(2047)})


def test_all_four_kappas_present(phase8a_artifact):
    validate_all_four_kappas({"kappas": list(EXPECTED_KAPPAS)}, phase8a_artifact)
    with pytest.raises(PopulationEffectAuditError):
        validate_all_four_kappas({"kappas": [0.0, 0.2]})


def test_all_84_phase8a_invariants_true():
    valid = {"all_hard_invariants": {str(i): True for i in range(84)},
             "all_hard_invariants_passed": True}
    validate_all_84_phase8a_invariants(valid)
    valid["all_hard_invariants"]["0"] = False
    with pytest.raises(PopulationEffectAuditError):
        validate_all_84_phase8a_invariants(valid)


def test_input_artifact_hashes_unchanged(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"fixed")
    before = hash_input_files([path])
    assert input_hashes_unchanged(before, hash_input_files([path]))
    path.write_bytes(b"changed")
    assert not input_hashes_unchanged(before, hash_input_files([path]))


def test_weight_arrays_align_with_public_rows():
    validate_weight_array(np.full(6, 1 / 6), 6, 1e-7, 1e-7)
    with pytest.raises(PopulationEffectAuditError):
        validate_weight_array(np.full(5, 0.2), 6, 1e-7, 1e-7)


def test_weight_arrays_sum_to_one():
    validate_weight_array(np.asarray((0.25, 0.75)), 2, 1e-7, 1e-7)
    with pytest.raises(PopulationEffectAuditError):
        validate_weight_array(np.asarray((0.2, 0.2)), 2, 1e-7, 1e-7)


def test_midpoint_weight_is_average_of_heavy_weights():
    left, right = np.asarray((0.8, 0.2)), np.asarray((0.1, 0.9))
    np.testing.assert_allclose(midpoint_weight(left, right), 0.5 * (left + right))


def test_primary_mixtures_preserve_anchor_action_mass(phase8a_artifact):
    directory = phase8a_artifact / "kappa_0p20"
    for condition in CONDITIONS:
        public = load_npz(directory / f"{condition}_public.npz")
        hidden = load_npz(directory / f"{condition}_hidden_audit.npz")
        weights = load_condition_weights(directory, condition, hidden, 1e-7, 1e-7)
        result = verify_primary_weights_preserve_exact_groups(
            public, hidden, weights, np.arange(4), 1e-7, 1e-7
        )
        assert result["passed"]


def test_exact_action_key_mapping(phase8a_artifact):
    directory = phase8a_artifact / "kappa_0p20"
    public = load_npz(directory / "confounded_public.npz")
    hidden = load_npz(directory / "confounded_hidden_audit.npz")
    groups = recover_exact_action_groups(public, hidden, np.arange(4), 1e-7, 1e-7)
    assert len(groups) == 12
    assert all({action for anchor, action in groups if anchor == current}
               == {"minus", "base", "plus"} for current in range(4))


def test_do_oracle_raw_summary_agreement(phase8a_artifact):
    anchors = load_npz(phase8a_artifact / "anchors.npz")
    directory = phase8a_artifact / "kappa_0p20"
    _, audit = recompute_do_oracle(
        load_npz(directory / "do_oracle_raw.npz"),
        load_npz(directory / "do_oracle_summary.npz"), anchors,
        anchors["anchor_id"], 0.2, 1e-7, 1e-7,
    )
    assert audit["passed"] and audit["maximum_absolute_difference"] <= 1e-7


def test_do_oracle_is_mixture_independent(analyzed):
    assert all(checks["do_oracle_mixture_and_condition_independent"]
               for checks in analyzed[2].values())


def test_confounded_u_posterior_matches_analytic_values(analyzed):
    posterior = analyzed[0][0.2]["observational"]["confounded"]
    for mixture in ("logger1_heavy", "logger12_midpoint", "logger2_heavy"):
        np.testing.assert_allclose(
            posterior[mixture]["posterior_u_plus"],
            np.broadcast_to(analytic_u_posterior("confounded", mixture), (4, 3)),
            atol=1e-7,
        )


def test_independent_u_posterior_is_half(analyzed):
    posterior = analyzed[0][0.3]["observational"]["independent_latents"]
    for mixture in ("logger1_heavy", "logger12_midpoint", "logger2_heavy"):
        np.testing.assert_allclose(posterior[mixture]["posterior_u_plus"], 0.5)


def test_kappa_zero_population_equals_do(analyzed):
    assert analyzed[2][0.0]["kappa_zero_negative_control"]


def test_independent_population_equals_do(analyzed):
    assert all(checks["independent_population_equals_do"] for checks in analyzed[2].values())


def test_base_action_is_mixture_invariant(analyzed):
    assert all(checks["base_action_is_primary_mixture_invariant_and_equals_do"]
               for checks in analyzed[2].values())


def test_midpoint_population_equals_do_in_complementary_dgp(analyzed):
    assert all(checks["midpoint_population_equals_do_in_complementary_dgp"]
               for checks in analyzed[2].values())


def test_reward_drift_identity(analyzed):
    assert all(checks["reward_drift_identity"] for checks in analyzed[2].values())


def test_delta_drift_identity(analyzed):
    assert all(checks["delta_drift_identity"] for checks in analyzed[2].values())


def test_public_schema_has_no_hidden_leakage(phase8a_artifact):
    public = load_npz(phase8a_artifact / "kappa_0p20" / "confounded_public.npz")
    validate_public_schema(public)
    assert FORBIDDEN_PUBLIC_FIELDS.isdisjoint(public)


def test_metrics_use_anchor_level_units(phase8a_artifact):
    anchors = load_npz(phase8a_artifact / "anchors.npz")
    _, specs, _, _ = analyze_kappa(
        phase8a_artifact, 0.2, anchors, anchors["anchor_id"], 1e-7, 1e-7
    )
    assert all(spec["values"].shape == (4,) for spec in specs)


def test_no_nan_inf(analyzed):
    assert all(all_arrays_finite(values) for values in analyzed[1].values())


def test_existing_summary_crosscheck_where_comparable(phase8a_artifact, analyzed):
    existing = json.loads((phase8a_artifact / "summary.json").read_text(encoding="utf-8"))
    result = crosscheck_existing_summary(existing, analyzed[0], 1e-7, 1e-7, True)
    assert result["passed"] and result["entries"]


def test_end_to_end_review_writes_complete_bundle_without_mutating_inputs(phase8a_artifact):
    output = phase8a_artifact / "population_effect_review_smoke"
    result = run_review(
        phase8a_artifact, output, bootstrap_reps=20, seed=0, max_anchors=3,
        expected_anchor_count=4,
    )
    expected = {
        "manifest.json", "input_integrity.json", "summary.json",
        "anchor_action_metrics.npz", "aggregate_tables.csv", "hard_checks.json",
        "REPORT.md", "analysis-report.md", "stats-appendix.md", "figure-catalog.md",
        "u_reward_effect_vs_kappa.png", "primary_reward_drift_vs_kappa.png",
        "primary_delta_drift_vs_kappa.png", "reward_do_error_vs_kappa.png",
        "action_ranking_difference_vs_kappa.png",
        "drift_relative_to_action_gap_vs_kappa.png", "drift_vs_u_effect_kappa_0p30.png",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert result["all_hard_checks_passed"]
    integrity = json.loads((output / "input_integrity.json").read_text(encoding="utf-8"))
    assert integrity["unchanged"] and integrity["sha256_before"] == integrity["sha256_after"]
