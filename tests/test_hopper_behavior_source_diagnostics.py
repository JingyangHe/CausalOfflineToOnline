"""Focused tests for the Phase 6B Hopper source-readiness audit."""

from pathlib import Path

import numpy as np
import pytest

import scripts.diagnose_hopper_behavior_sources as diagnostics
from confounded_hopper import ACTUATOR_DIRECTION
from scripts.diagnose_hopper_behavior_sources import (
    PUBLIC_FIELDS,
    SOURCE_STEPS,
    _hash,
    clipping_diagnostics,
    compensation_diagnostics,
    deterministic_indices,
    nearest_neighbor_diagnostics,
    run_paired_outcome_audit,
    validate_public_pilot,
)
from tests.test_confounded_hopper_behavior_policies import make_wrapper


def test_compensation_projection_matches_actuator_direction():
    minus = np.zeros((4, 3))
    plus = -0.4 * np.tile(ACTUATOR_DIRECTION, (4, 1))
    report, arrays = compensation_diagnostics(minus, plus, ACTUATOR_DIRECTION, 0.2)
    np.testing.assert_allclose(arrays["projection"], -0.4, atol=1e-15)
    assert report["projection_mean_absolute_deviation_from_target"] < 1e-15


def test_orthogonal_component_is_separated_from_projection():
    perpendicular = np.array((1.0, 1.0, 0.0)) / np.sqrt(2.0)
    report, arrays = compensation_diagnostics(
        np.zeros((2, 3)), np.tile(perpendicular, (2, 1)), ACTUATOR_DIRECTION, 0.2
    )
    np.testing.assert_allclose(arrays["projection"], 0.0, atol=1e-15)
    np.testing.assert_allclose(arrays["orthogonal_norm"], 1.0, atol=1e-15)
    assert report["orthogonal_norm"]["mean"] == pytest.approx(1.0)


def test_exact_compensation_has_zero_applied_action_residual():
    minus = np.tile(0.2 * ACTUATOR_DIRECTION, (3, 1))
    plus = np.tile(-0.2 * ACTUATOR_DIRECTION, (3, 1))
    _, arrays = compensation_diagnostics(minus, plus, ACTUATOR_DIRECTION, 0.2)
    np.testing.assert_allclose(arrays["applied_residual"], 0.0, atol=1e-15)


def test_clipping_diagnostics_count_only_actual_boundary_crossings():
    commands = np.array(((1.0, 0.0, -1.0), (0.0, 0.0, 0.0)))
    report = clipping_diagnostics(commands, 1, ACTUATOR_DIRECTION, 0.2)
    assert report["any_clipping_rate"] == 0.5
    np.testing.assert_allclose(report["per_dimension_clipping_rate"], (0.5, 0.0, 0.0))
    assert report["clipping_correction_norm"]["mean"] > 0.0


def _public_fixture(per_source=2):
    count = 3 * per_source
    values = {
        "observation": np.zeros((count, 12), dtype=np.float32),
        "action": np.zeros((count, 3), dtype=np.float32),
        "reward": np.zeros(count),
        "next_observation": np.zeros((count, 12), dtype=np.float32),
        "terminated": np.zeros(count, dtype=bool),
        "truncated": np.zeros(count, dtype=bool),
        "source_id": np.repeat((1, 2, 3), per_source),
        "episode_id": np.zeros(count, dtype=np.int64),
        "time_step": np.arange(count),
    }
    assert set(values) == PUBLIC_FIELDS
    return values


def test_public_pilot_schema_rejects_hidden_fields():
    public = _public_fixture()
    public["hidden_u"] = np.ones(6)
    with pytest.raises(RuntimeError, match="exactly"):
        validate_public_pilot(public, 2)


def test_public_pilot_requires_exact_equal_source_counts():
    validate_public_pilot(_public_fixture(3), 3)
    unbalanced = _public_fixture(3)
    unbalanced["source_id"][0] = 2
    with pytest.raises(RuntimeError, match="balanced"):
        validate_public_pilot(unbalanced, 3)


def test_nearest_neighbor_and_matched_action_diagnostics_on_toy_data():
    states = {
        "source_1": np.array(((0.0,), (10.0,))),
        "source_2": np.array(((0.1,), (10.1,))),
        "source_3": np.array(((0.2,), (10.2,))),
    }
    actions = {
        "source_1": np.array(((0.0,), (0.0,))),
        "source_2": np.array(((1.0,), (2.0,))),
        "source_3": np.array(((3.0,), (4.0,))),
    }
    report, arrays = nearest_neighbor_diagnostics(states, actions)
    cross = report["directed_cross"]["source_1_to_source_2"]
    assert cross["matched_action_distance"]["mean"] == pytest.approx(1.5)
    assert report["nearest_neighbor_has_different_source"] == 1.0
    assert np.all(arrays["cross_state_source_1_to_source_2"] > 0.0)


def test_source_mapping_and_equal_spaced_selection_are_fixed_and_reproducible():
    assert SOURCE_STEPS == {"source_1": 200_000, "source_2": 500_000, "source_3": 1_000_000}
    first = deterministic_indices(10_000, 512)
    second = deterministic_indices(10_000, 512)
    np.testing.assert_array_equal(first, second)
    assert first[0] == 0 and first[-1] == 9_999 and np.unique(first).size == 512


def test_same_seed_reproduces_pilot_diagnostics(monkeypatch):
    class FakeModel:
        def set_random_seed(self, seed):
            self.rng = np.random.default_rng(seed)

        def predict(self, observation, deterministic=False):
            return self.rng.uniform(-0.5, 0.5, size=3).astype(np.float32), None

    class FakeEnvironment:
        def __init__(self, kappa):
            self.kappa, self.steps = kappa, 0

        def reset(self, seed=None):
            self.rng, self.steps = np.random.default_rng(seed), 0
            self.observation = self.rng.normal(size=13).astype(np.float32)
            self.observation[-1] = self.rng.choice((-1.0, 1.0))
            return self.observation.copy(), {}

        @staticmethod
        def get_public_observation(observation):
            return np.asarray(observation)[:12].copy()

        def capture_audit_state(self):
            return {"observation": self.observation.copy(), "steps": self.steps}

        def step(self, command):
            hidden_u = int(self.observation[-1])
            applied = np.clip(command + self.kappa * hidden_u * ACTUATOR_DIRECTION, -1, 1)
            self.steps += 1
            self.observation = self.rng.normal(size=13).astype(np.float32)
            self.observation[-1] = self.rng.choice((-1.0, 1.0))
            return self.observation.copy(), float(np.sum(applied)), self.steps == 3, False, {
                "hidden_u": hidden_u, "applied_action": applied,
            }

        def close(self):
            pass

    monkeypatch.setattr(diagnostics, "_environment", FakeEnvironment)
    model_sets = [
        {source: FakeModel() for source in SOURCE_STEPS},
        {source: FakeModel() for source in SOURCE_STEPS},
    ]
    first = diagnostics._collect_pilot(model_sets[0], 0.2, 7, 3, 2026)
    second = diagnostics._collect_pilot(model_sets[1], 0.2, 7, 3, 2026)
    for first_group, second_group in zip(first[:2], second[:2]):
        for key in first_group:
            np.testing.assert_array_equal(first_group[key], second_group[key])
    assert len(first[2]) == len(second[2]) == 3


def test_paired_audit_reuses_identical_snapshot_and_command():
    class FakeAuditEnvironment:
        def __init__(self):
            self.calls = []

        def audit_step_from_state(self, snapshot, command, hidden_u):
            self.calls.append((snapshot, np.asarray(command).copy(), hidden_u))
            applied = np.asarray(command) + 0.1 * hidden_u
            public = np.array((snapshot["state"] + applied[0], 0.5))
            return public, float(applied[0]), False, False, {
                "public_observation": public, "applied_action": applied,
            }

    environment = FakeAuditEnvironment()
    snapshot = {"state": 2.0}
    command = np.array((0.2, -0.1, 0.3))
    report, arrays = run_paired_outcome_audit(
        environment, [{"snapshot": snapshot, "commanded_action": command}]
    )
    assert environment.calls[0][0] is snapshot and environment.calls[1][0] is snapshot
    np.testing.assert_array_equal(environment.calls[0][1], environment.calls[1][1])
    assert [call[2] for call in environment.calls] == [-1, 1]
    assert arrays["next_state"][0] > 0.0
    assert report["termination_disagreement_count"] == 0


def test_checkpoint_hash_helper_is_read_only(tmp_path: Path):
    checkpoint = tmp_path / "source.zip"
    checkpoint.write_bytes(b"fixed checkpoint bytes")
    before = checkpoint.read_bytes(), _hash(checkpoint)
    after = checkpoint.read_bytes(), _hash(checkpoint)
    assert before == after


def test_audit_snapshot_interface_is_unavailable_on_default_wrapper():
    environment = make_wrapper(audit=False)
    with pytest.raises(RuntimeError, match="audit_info=True"):
        environment.capture_audit_state()
    with pytest.raises(RuntimeError, match="audit_info=True"):
        environment.audit_step_from_state({}, np.zeros(3), 1)
