"""Tests for the exact Phase 8A-NC non-complementary population DGP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from confounded_hopper import ACTUATOR_DIRECTION
from experiments.hopper_logger_mixture_drift.analyze_noncomplementary_population import (
    NonComplementaryPopulationAuditError,
    load_strict_unclipped_mask,
    require_phase8ac_root,
    require_phase8ar_root,
    run_population_dgp,
)
from experiments.hopper_logger_mixture_drift.analyze_phase8a_clipping_sensitivity import (
    run_audit,
)
from experiments.hopper_logger_mixture_drift.analyze_phase8a_population_effect import (
    EXPECTED_KAPPAS,
    PopulationEffectAuditError,
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
    require_verified_phase8a_root,
    run_review,
)
from experiments.hopper_logger_mixture_drift.audit import (
    outcome_strength,
    population_observational_table,
    summarize_population_table,
)
from experiments.hopper_logger_mixture_drift.controlled_loggers import (
    CONDITIONS as OLD_CONDITIONS,
    MIXTURES as OLD_MIXTURES,
)
from experiments.hopper_logger_mixture_drift.generate_datasets import (
    generate_condition_dataset,
    generate_do_oracle,
    generate_mixture_weights,
)
from experiments.hopper_logger_mixture_drift.noncomplementary_population_dgp import (
    ACTION_KEYS,
    CONDITIONS,
    FORBIDDEN_PUBLIC_FIELDS,
    LOGGER_ACTION_PROBABILITIES,
    MIXTURES,
    PRIMARY_MIXTURES,
    analytic_u_posterior,
    build_do_lookup,
    logger_action_marginal,
    support_specification,
)


class FakeSimulator:
    def __init__(self, anchors):
        self.anchors = anchors

    def step(self, anchor_index, commanded_action, u_env, kappa_env):
        command = np.asarray(commanded_action, dtype=np.float64)
        preclip = command + kappa_env * u_env * ACTUATOR_DIRECTION
        applied = np.clip(preclip, -1.0, 1.0)
        observation = self.anchors["public_observation"][anchor_index].astype(np.float64)
        following = observation.copy()
        following[:3] += 0.05 * applied
        following[3:6] += 0.01 * applied
        following[-1] = max(0.0, observation[-1] - 0.001)
        qpos = self.anchors["qpos"][anchor_index]
        qvel = self.anchors["qvel"][anchor_index]
        return {
            "observation": observation, "commanded_action": command,
            "applied_action": applied,
            "reward": float(applied @ np.asarray((1.0, 0.5, -0.25))),
            "next_observation": following, "terminated": False, "truncated": False,
            "qpos": qpos, "qvel": qvel, "next_qpos": qpos + 0.01,
            "next_qvel": qvel + applied.mean(),
            "commanded_action_clipped": bool(np.any(np.abs(command) > 1.0)),
            "applied_action_clipped": bool(np.any(np.abs(preclip) > 1.0)),
        }


def _anchors(count):
    observation = np.zeros((count, 12), dtype=np.float32)
    observation[:, 0] = np.arange(count) * 0.1
    observation[:, -1] = 1.0
    base = np.zeros((count, 3), dtype=np.float64)
    base[count // 2:, 0] = 0.75
    return {"anchor_id": np.arange(count, dtype=np.int64),
            "public_observation": observation, "base_action": base,
            "qpos": np.zeros((count, 6)), "qvel": np.zeros((count, 6)),
            "elapsed_steps": np.arange(count, dtype=np.int64)}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _kappa_name(kappa):
    return f"kappa_{kappa:.2f}".replace(".", "p")


@pytest.fixture(scope="module")
def nc_bundle(tmp_path_factory):
    count = 8
    parent = tmp_path_factory.mktemp("phase8anc")
    root = parent / "controlled_loggers_seed0_verified"
    root.mkdir()
    anchors = _anchors(count)
    np.savez_compressed(root / "anchors.npz", **anchors)
    _write_json(root / "mixture_weights.json",
                {name: list(values) for name, values in OLD_MIXTURES.items()})
    _write_json(root / "manifest.json", {
        "phase": "8A", "environment_id": "Hopper-v5", "kappas": list(EXPECTED_KAPPAS),
        "number_of_anchors": count, "actuator_direction_v": ACTUATOR_DIRECTION.tolist(),
        "numerical_tolerance": {"atol": 1e-7, "rtol": 1e-7},
    })
    (root / "phase8a_complete.log").write_text(
        "PHASE8A_HOPPER_LOGGER_MIXTURE_DGP_COMPLETE\n"
        "READY_FOR_POOLED_WORLD_MODEL_DRIFT_TRAINING\n", encoding="utf-8")
    summary = {"phase": "8A", "anchor_count": count,
               "all_hard_invariants": {f"check_{i:02d}": True for i in range(84)},
               "all_hard_invariants_passed": True, "by_kappa": {}}
    simulator = FakeSimulator(anchors)
    for kappa in EXPECTED_KAPPAS:
        directory = root / _kappa_name(kappa)
        directory.mkdir()
        raw, stored = generate_do_oracle(anchors, kappa, 0.2, simulator)
        np.savez_compressed(directory / "do_oracle_raw.npz", **raw)
        np.savez_compressed(directory / "do_oracle_summary.npz", **stored)
        population = {}
        for condition in OLD_CONDITIONS:
            public, hidden = generate_condition_dataset(anchors, condition, kappa, 0.2,
                                                        simulator)
            weights = generate_mixture_weights(hidden)
            np.savez_compressed(directory / f"{condition}_public.npz", **public)
            np.savez_compressed(directory / f"{condition}_hidden_audit.npz", **hidden)
            weight_dir = directory / "weights" / condition
            weight_dir.mkdir(parents=True)
            for mixture, values in weights.items():
                np.save(weight_dir / f"weights_{mixture}.npy", values)
            table = population_observational_table(anchors, public, hidden, weights, stored)
            population[condition] = summarize_population_table(table)[0]
        summary["by_kappa"][_kappa_name(kappa)] = {
            "outcome_strength": outcome_strength(raw), "population": population}
    _write_json(root / "summary.json", summary)
    review = root / "population_effect_review"
    run_review(root, review, bootstrap_reps=10, seed=0, expected_anchor_count=count)
    clipping = review / "clipping_sensitivity_kappa_0p30"
    run_audit(root, review, clipping, bootstrap_reps=10, seed=0,
              expected_anchor_count=count)
    output = parent / "noncomplementary_loggers_seed0_verified"
    result = run_population_dgp(
        root, review, clipping, output, num_anchors=count, kappas=EXPECTED_KAPPAS,
        bootstrap_reps=20, seed=0, expected_anchor_count=count)
    return root, review, clipping, output, result


def test_verified_phase8a_input_required(tmp_path, nc_bundle):
    assert require_verified_phase8a_root(nc_bundle[0]) == nc_bundle[0].resolve()
    with pytest.raises(PopulationEffectAuditError):
        require_verified_phase8a_root(tmp_path)


def test_phase8ar_input_required(tmp_path, nc_bundle):
    assert require_phase8ar_root(nc_bundle[0], nc_bundle[1]) == nc_bundle[1].resolve()
    with pytest.raises(NonComplementaryPopulationAuditError):
        require_phase8ar_root(nc_bundle[0], tmp_path)


def test_phase8ac_input_required(tmp_path, nc_bundle):
    assert require_phase8ac_root(nc_bundle[1], nc_bundle[2]) == nc_bundle[2].resolve()
    with pytest.raises(NonComplementaryPopulationAuditError):
        require_phase8ac_root(nc_bundle[1], tmp_path)


def test_all_2048_anchors_reused(nc_bundle):
    assert run_population_dgp.__kwdefaults__["num_anchors"] == 2048
    assert run_population_dgp.__kwdefaults__["expected_anchor_count"] == 2048
    assert nc_bundle[4]["hard_checks"]["all_expected_anchors_reused"]


def test_all_four_kappas_reused(nc_bundle):
    assert all((nc_bundle[3] / _kappa_name(kappa)).is_dir() for kappa in EXPECTED_KAPPAS)


def test_do_oracle_lookup_unique(nc_bundle):
    raw = load_npz(nc_bundle[0] / "kappa_0p20" / "do_oracle_raw.npz")
    assert len(build_do_lookup(raw, 0.2)) == 8 * 3 * 2


def test_logger_probability_rows_sum_to_one():
    assert all(np.isclose(sum(LOGGER_ACTION_PROBABILITIES[l][u].values()), 1)
               for l in (0, 1, 2) for u in (-1, 1))


def test_logger1_action_marginal_is_half():
    assert logger_action_marginal(0, "plus") == pytest.approx(0.5)
    assert logger_action_marginal(0, "minus") == pytest.approx(0.5)


def test_logger2_action_marginal_is_half():
    assert logger_action_marginal(1, "plus") == pytest.approx(0.5)
    assert logger_action_marginal(1, "minus") == pytest.approx(0.5)


def test_loggers_are_noncomplementary(nc_bundle):
    assert nc_bundle[4]["logger_properties"]["LOGGERS_ARE_NONCOMPLEMENTARY"]


def test_loggers_have_same_confounding_direction(nc_bundle):
    assert nc_bundle[4]["logger_properties"]["LOGGERS_HAVE_SAME_CONFOUNDING_DIRECTION"]


def test_primary_mixtures_preserve_state_action_mass(nc_bundle):
    checks = nc_bundle[4]["hard_checks"]
    assert all(checks[f"{_kappa_name(k)}:primary_mixtures_preserve_state_action_mass"]
               for k in EXPECTED_KAPPAS)


def test_confounded_u_posteriors_match_analytic_values(nc_bundle):
    audit = nc_bundle[4]["per_kappa_audits"]["kappa_0p20"]["posterior"]["confounded"]
    assert audit["passed"]
    for mixture in MIXTURES:
        for action in ACTION_KEYS:
            assert audit["details"][mixture][action]["empirical_mean"] == pytest.approx(
                analytic_u_posterior("confounded", mixture, action))


def test_logger12_balanced_remains_confounded(nc_bundle):
    assert analytic_u_posterior("confounded", "logger12_balanced", "plus") == pytest.approx(0.8)
    assert nc_bundle[4]["logger_properties"]["LOGGER12_BALANCED_REMAINS_CONFOUNDED"]


def test_all_source_equal_remains_confounded(nc_bundle):
    assert analytic_u_posterior("confounded", "all_sources_equal", "minus") == pytest.approx(0.2)
    assert nc_bundle[4]["logger_properties"]["ALL_SOURCE_EQUAL_SAMPLING_REMAINS_CONFOUNDED"]


def test_independent_u_posterior_is_half(nc_bundle):
    audit = nc_bundle[4]["per_kappa_audits"]["kappa_0p30"]["posterior"]["independent_latents"]
    assert all(item["empirical_mean"] == pytest.approx(0.5)
               for values in audit["details"].values() for item in values.values())


def test_kappa_zero_population_equals_do(nc_bundle):
    assert nc_bundle[4]["hard_checks"]["kappa_0p00:kappa_zero_population_equals_do"]


def test_independent_population_equals_do(nc_bundle):
    assert all(nc_bundle[4]["hard_checks"][
        f"{_kappa_name(k)}:independent_population_equals_do"] for k in EXPECTED_KAPPAS)


def test_base_action_is_mixture_invariant(nc_bundle):
    assert all(nc_bundle[4]["hard_checks"][
        f"{_kappa_name(k)}:base_action_is_mixture_invariant_and_equals_do"]
               for k in EXPECTED_KAPPAS)


def test_reward_heavy_drift_identity(nc_bundle):
    assert all(nc_bundle[4]["hard_checks"][f"{_kappa_name(k)}:reward_heavy_drift_identity"]
               for k in EXPECTED_KAPPAS)


def test_delta_heavy_drift_identity(nc_bundle):
    assert all(nc_bundle[4]["hard_checks"][f"{_kappa_name(k)}:delta_heavy_drift_identity"]
               for k in EXPECTED_KAPPAS)


def test_balanced_reward_bias_identity(nc_bundle):
    assert all(nc_bundle[4]["hard_checks"][f"{_kappa_name(k)}:balanced_reward_bias_identity"]
               for k in EXPECTED_KAPPAS)


def test_balanced_delta_bias_identity(nc_bundle):
    assert all(nc_bundle[4]["hard_checks"][f"{_kappa_name(k)}:balanced_delta_bias_identity"]
               for k in EXPECTED_KAPPAS)


def test_strict_unclipped_mask_matches_phase8ac(nc_bundle):
    ids = load_npz(nc_bundle[0] / "anchors.npz")["anchor_id"]
    mask, _ = load_strict_unclipped_mask(nc_bundle[2], ids)
    assert nc_bundle[4]["strict_unclipped_count_selected"] == int(mask.sum())


def test_public_hidden_leakage_empty(nc_bundle):
    public = load_npz(nc_bundle[3] / "kappa_0p20" / "confounded_public.npz")
    assert FORBIDDEN_PUBLIC_FIELDS.isdisjoint(public)


def test_weight_arrays_align_with_public_rows(nc_bundle):
    public = load_npz(nc_bundle[3] / "kappa_0p20" / "confounded_public.npz")
    weight_dir = nc_bundle[3] / "kappa_0p20" / "weights" / "confounded"
    assert all(np.load(weight_dir / f"{name}.npy").shape == public["row_id"].shape
               for name in MIXTURES)


def test_weight_arrays_sum_to_one(nc_bundle):
    weight_dir = nc_bundle[3] / "kappa_0p20" / "weights" / "confounded"
    assert all(np.load(weight_dir / f"{name}.npy").sum() == pytest.approx(1.0)
               for name in MIXTURES)


def test_no_nan_inf(nc_bundle):
    arrays = load_npz(nc_bundle[3] / "anchor_action_metrics.npz")
    assert all(np.all(np.isfinite(values)) for values in arrays.values()
               if np.issubdtype(values.dtype, np.number))


def test_metrics_use_anchor_level_units(nc_bundle):
    assert all(row["bootstrap_unit"] == "anchor_id"
               for row in nc_bundle[4]["aggregate_metrics"])


def test_input_hashes_unchanged(nc_bundle):
    integrity = json.loads((nc_bundle[3] / "input_integrity.json").read_text())
    assert integrity["unchanged"] and integrity["sha256_before"] == integrity["sha256_after"]


def test_old_artifacts_unchanged(nc_bundle):
    assert nc_bundle[4]["hard_checks"]["old_artifacts_unchanged"]


def test_output_bundle_and_figures_complete(nc_bundle):
    root = nc_bundle[3]
    assert {"manifest.json", "input_integrity.json", "hard_checks.json", "summary.json",
            "anchor_action_metrics.npz", "aggregate_tables.csv", "REPORT.md",
            "analysis-report.md", "stats-appendix.md", "figure-catalog.md"}.issubset(
                {path.name for path in root.iterdir()})
    assert len(list((root / "figures").glob("*.png"))) == 9
