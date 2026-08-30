"""Focused tests for Phase 8A-C clipping sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from confounded_hopper import ACTUATOR_DIRECTION
from experiments.hopper_logger_mixture_drift.analyze_phase8a_clipping_sensitivity import (
    ClippingSensitivityAuditError,
    clipping_from_preclip,
    derive_clean_sets,
    reconstruct_canonical_clipping,
    require_phase8ar_root,
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
from experiments.hopper_logger_mixture_drift.controlled_loggers import CONDITIONS, MIXTURES
from experiments.hopper_logger_mixture_drift.generate_datasets import (
    generate_condition_dataset,
    generate_do_oracle,
    generate_mixture_weights,
)


class FakeClippingSimulator:
    def __init__(self, anchors: dict[str, np.ndarray]):
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
        reward = float(applied @ np.asarray((1.0, 0.5, -0.25)))
        qpos = self.anchors["qpos"][anchor_index].copy()
        qvel = self.anchors["qvel"][anchor_index].copy()
        return {
            "observation": observation, "commanded_action": command,
            "applied_action": applied, "reward": reward,
            "next_observation": following, "terminated": False, "truncated": False,
            "qpos": qpos, "qvel": qvel, "next_qpos": qpos + 0.01,
            "next_qvel": qvel + applied.mean(),
            "commanded_action_clipped": bool(np.any(np.abs(command) > 1.0)),
            "applied_action_clipped": bool(np.any(np.abs(preclip) > 1.0)),
        }


def _anchors(count: int) -> dict[str, np.ndarray]:
    observation = np.zeros((count, 12), dtype=np.float32)
    observation[:, 0] = np.arange(count, dtype=np.float32) * 0.1
    observation[:, -1] = 1.0
    base = np.zeros((count, 3), dtype=np.float64)
    base[count // 2:, 0] = 0.75
    return {
        "anchor_id": np.arange(count, dtype=np.int64),
        "public_observation": observation, "base_action": base,
        "qpos": np.zeros((count, 6), dtype=np.float64),
        "qvel": np.zeros((count, 6), dtype=np.float64),
        "elapsed_steps": np.arange(count, dtype=np.int64),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _kappa_name(kappa: float) -> str:
    return f"kappa_{kappa:.2f}".replace(".", "p")


@pytest.fixture(scope="module")
def phase8ac_bundle(tmp_path_factory):
    count = 8
    root = tmp_path_factory.mktemp("phase8ac") / "controlled_loggers_seed0_verified"
    root.mkdir()
    anchors = _anchors(count)
    np.savez_compressed(root / "anchors.npz", **anchors)
    _write_json(root / "mixture_weights.json", {
        name: list(values) for name, values in MIXTURES.items()
    })
    _write_json(root / "manifest.json", {
        "phase": "8A", "kappas": list(EXPECTED_KAPPAS), "number_of_anchors": count,
        "actuator_direction_v": ACTUATOR_DIRECTION.tolist(),
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
    simulator = FakeClippingSimulator(anchors)
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
            table = population_observational_table(anchors, public, hidden, weights, stored)
            population[condition] = summarize_population_table(table)[0]
        summary["by_kappa"][_kappa_name(kappa)] = {
            "outcome_strength": outcome_strength(raw), "population": population,
        }
    _write_json(root / "summary.json", summary)
    review = root / "population_effect_review"
    run_review(root, review, bootstrap_reps=10, seed=0, expected_anchor_count=count)
    output = review / "clipping_sensitivity_kappa_0p30"
    result = run_audit(root, review, output, bootstrap_reps=20, seed=0,
                       expected_anchor_count=count)
    return root, review, output, result


def test_verified_phase8a_input_required(tmp_path, phase8ac_bundle):
    assert require_verified_phase8a_root(phase8ac_bundle[0]) == phase8ac_bundle[0].resolve()
    with pytest.raises(PopulationEffectAuditError):
        require_verified_phase8a_root(tmp_path)


def test_phase8ar_input_required(tmp_path, phase8ac_bundle):
    assert require_phase8ar_root(phase8ac_bundle[0], phase8ac_bundle[1]) == phase8ac_bundle[1].resolve()
    with pytest.raises(ClippingSensitivityAuditError):
        require_phase8ar_root(phase8ac_bundle[0], tmp_path)


def test_kappa_0p30_present(phase8ac_bundle):
    assert (phase8ac_bundle[0] / "kappa_0p30" / "do_oracle_raw.npz").is_file()


def test_all_2048_anchors_present():
    ids = np.arange(2048)
    assert len(ids) == 2048 and np.array_equal(ids, np.arange(2048))


def test_action_key_mapping_unique(phase8ac_bundle):
    table = load_npz(phase8ac_bundle[2] / "canonical_clipping_table.npz")
    keys = set(zip(table["anchor_id"].tolist(), table["action_key"].tolist(),
                   table["u_env"].tolist()))
    assert len(keys) == len(table["anchor_id"])


def test_do_oracle_canonical_keys_unique(phase8ac_bundle):
    raw = load_npz(phase8ac_bundle[0] / "kappa_0p30" / "do_oracle_raw.npz")
    keys = set(zip(raw["anchor_id"].tolist(), raw["action_key"].tolist(),
                   raw["u_env"].tolist(), raw["kappa_env"].tolist()))
    assert len(keys) == 8 * 3 * 2


def test_preclip_reconstruction(phase8ac_bundle):
    table = load_npz(phase8ac_bundle[2] / "canonical_clipping_table.npz")
    expected = table["commanded_action"] + 0.3 * table["u_env"][:, None] * ACTUATOR_DIRECTION
    np.testing.assert_allclose(table["preclip_action"], expected)


def test_expected_applied_matches_saved_applied(phase8ac_bundle):
    table = load_npz(phase8ac_bundle[2] / "canonical_clipping_table.npz")
    np.testing.assert_allclose(table["expected_applied_action"], table["saved_applied_action"])


def test_clipping_definition_uses_preclip_not_boundary_equality():
    before = np.asarray([[1.0, 0.0, -1.0], [1.01, 0.0, 0.0]])
    coordinate, row, *_ = clipping_from_preclip(before, np.clip(before, -1, 1), 1e-7)
    assert not row[0] and row[1] and not coordinate[0].any()


def test_clipping_status_is_logger_invariant(phase8ac_bundle):
    assert phase8ac_bundle[3]["hard_checks"]["clipping_status_is_logger_invariant"]


def test_clipping_status_is_condition_invariant(phase8ac_bundle):
    assert phase8ac_bundle[3]["hard_checks"]["clipping_status_is_condition_invariant"]


def test_action_pair_unclipped_definition():
    clipped = np.zeros((2, 3, 2), dtype=bool)
    clipped[1, 2, 1] = True
    pair, _, _ = derive_clean_sets(clipped)
    assert pair[0].all() and not pair[1, 2] and pair[1, :2].all()


def test_strict_anchor_unclipped_requires_all_six_executions():
    clipped = np.zeros((2, 3, 2), dtype=bool)
    clipped[1, 2, 1] = True
    _, strict, any_clipping = derive_clean_sets(clipped)
    assert strict.tolist() == [True, False] and any_clipping.tolist() == [False, True]


def test_primary_state_action_mass_preserved_after_strict_anchor_filter(phase8ac_bundle):
    assert phase8ac_bundle[3]["hard_checks"][
        "primary_state_action_mass_preserved_after_strict_filter"]


def test_independent_weighted_clipping_is_mixture_invariant(phase8ac_bundle):
    assert phase8ac_bundle[3]["hard_checks"][
        "independent_weighted_clipping_is_mixture_invariant"]


def test_base_action_drift_remains_zero_on_clean_subset(phase8ac_bundle):
    assert phase8ac_bundle[3]["hard_checks"]["base_action_drift_zero_on_clean_subset"]


def test_reward_mechanism_identity_on_clean_subset(phase8ac_bundle):
    assert phase8ac_bundle[3]["hard_checks"]["reward_mechanism_identity_on_clean_subset"]


def test_delta_mechanism_identity_on_clean_subset(phase8ac_bundle):
    assert phase8ac_bundle[3]["hard_checks"]["delta_mechanism_identity_on_clean_subset"]


def test_decision_metrics_use_strict_anchor_subset(phase8ac_bundle):
    result = phase8ac_bundle[3]
    strict_n = result["clipping_prevalence"]["strict_anchor_unclipped_count"]
    rows = [row for row in result["decision_metrics"]
            if row["subset"] == "strict_anchor_unclipped"]
    assert rows and all(row["n_anchors"] == strict_n for row in rows)


def test_full_sample_metrics_match_phase8ar(phase8ac_bundle):
    assert phase8ac_bundle[3]["phase8ar_full_sample_crosscheck"]["passed"]


def test_input_hashes_unchanged(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"immutable")
    before = hash_input_files([path])
    assert input_hashes_unchanged(before, hash_input_files([path]))


def test_no_nan_inf(phase8ac_bundle):
    summary = phase8ac_bundle[3]
    for rows in (summary["action_metrics"], summary["decision_metrics"]):
        for row in rows:
            for value in row.values():
                assert value is None or not isinstance(value, float) or np.isfinite(value)


def test_metrics_use_anchor_level_units(phase8ac_bundle):
    summary = phase8ac_bundle[3]
    assert all(row["bootstrap_unit"] == "anchor_id"
               for row in summary["action_metrics"] + summary["decision_metrics"])


def test_end_to_end_bundle_contains_required_outputs(phase8ac_bundle):
    expected = {
        "manifest.json", "input_integrity.json", "hard_checks.json",
        "canonical_clipping_table.npz", "anchor_clipping_table.npz",
        "action_specific_metrics.csv", "decision_metrics.csv", "aggregate_tables.csv",
        "summary.json", "REPORT.md", "analysis-report.md", "stats-appendix.md",
        "figure-catalog.md", "clipping_rate_by_action.png",
        "clipping_rate_by_action_coordinate.png", "preclip_headroom_distribution.png",
        "reward_drift_all_vs_unclipped.png", "delta_drift_all_vs_unclipped.png",
        "strict_flip_all_vs_unclipped.png",
        "drift_relative_to_action_gap_all_vs_unclipped.png",
    }
    assert expected.issubset({path.name for path in phase8ac_bundle[2].iterdir()})
    integrity = json.loads((phase8ac_bundle[2] / "input_integrity.json").read_text())
    assert integrity["unchanged"] and integrity["sha256_before"] == integrity["sha256_after"]
