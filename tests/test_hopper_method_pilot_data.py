"""Focused tests for Phase 7A Hopper method-pilot data collection."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import scripts.collect_hopper_method_pilot_data as collection
from confounded_hopper import ACTUATOR_DIRECTION


class FakePolicy:
    def __init__(self):
        self.seen_u: list[int] = []
        self.deterministic_flags: list[bool] = []

    def set_random_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic=False):
        self.seen_u.append(int(observation[-1]))
        self.deterministic_flags.append(bool(deterministic))
        return self.rng.uniform(-0.95, 0.95, size=3).astype(np.float32), None


class FakeHopper:
    def __init__(self, kappa: float, episode_length: int = 3):
        self.kappa = kappa
        self.episode_length = episode_length
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.steps = 0
        self.rng = None
        self.current_u = 1
        self.transition_u: list[int] = []
        self.next_u: list[int] = []

    def _observation(self):
        public = np.concatenate(
            (self.state, np.asarray((1.0 - self.steps / 100.0,), dtype=np.float32))
        )
        return np.concatenate((public, np.asarray((self.current_u,), dtype=np.float32)))

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        assert self.rng is not None
        self.steps = 0
        self.state = self.rng.normal(size=11).astype(np.float32)
        self.current_u = int(self.rng.choice((-1, 1)))
        return self._observation(), {}

    @staticmethod
    def get_public_observation(observation):
        return np.asarray(observation, dtype=np.float32)[:12].copy()

    def step(self, command):
        transition_u = self.current_u
        command = np.asarray(command, dtype=np.float32)
        preclip = command.astype(np.float64) + self.kappa * transition_u * ACTUATOR_DIRECTION
        applied = np.clip(preclip, -1.0, 1.0).astype(np.float32)
        self.steps += 1
        self.state = self.state + 0.01 * np.pad(applied, (0, 8))
        self.current_u = -transition_u
        self.transition_u.append(transition_u)
        self.next_u.append(self.current_u)
        terminated = self.steps >= self.episode_length
        return self._observation(), float(np.sum(applied)), terminated, False, {
            "hidden_u": transition_u,
            "commanded_action": command.copy(),
            "applied_action": applied,
        }

    def close(self):
        pass


@pytest.fixture
def fake_factory(monkeypatch):
    environments = []

    def factory(kappa):
        environment = FakeHopper(kappa)
        environments.append(environment)
        return environment

    monkeypatch.setattr(collection, "_make_environment", factory)
    return environments


def _collect(seed=2027, train_count=5, audit_count=2):
    models = {source_id: FakePolicy() for source_id in collection.SOURCE_STEPS}
    result = collection.collect_datasets(
        models,
        train_transitions_per_source=train_count,
        audit_transitions_per_source=audit_count,
        seed=seed,
        kappa=0.2,
    )
    return models, result


def test_fixed_source_mapping_and_unique_step_lookup(tmp_path):
    assert collection.SOURCE_STEPS == {1: 200_000, 2: 500_000, 3: 1_000_000}
    for step in collection.SOURCE_STEPS.values():
        (tmp_path / f"arbitrary_name_step_{step}.zip").touch()
        assert collection.find_unique_checkpoint_by_step(tmp_path, step).name.endswith(
            f"step_{step}.zip"
        )


def test_checkpoint_lookup_rejects_missing_and_duplicate_steps(tmp_path):
    with pytest.raises(RuntimeError, match="found 0"):
        collection.find_unique_checkpoint_by_step(tmp_path, 500_000)
    (tmp_path / "first_step_500000.zip").touch()
    (tmp_path / "second_step_500000.zip").touch()
    with pytest.raises(RuntimeError, match="found 2"):
        collection.find_unique_checkpoint_by_step(tmp_path, 500_000)


def test_seed_plan_has_six_distinct_reproducible_streams():
    first = collection.collection_seed_plan(2027)
    second = collection.collection_seed_plan(2027)
    assert first == second
    spawn_keys = {
        tuple(first[source][split]["spawn_key"])
        for source in first for split in ("train", "audit")
    }
    assert len(spawn_keys) == 6
    for source in first:
        assert first[source]["train"] != first[source]["audit"]


def test_exact_counts_dtypes_and_source_checkpoint_association(fake_factory):
    _, (train, audit, _, _) = _collect(train_count=5, audit_count=2)
    assert train["observations"].shape == (15, 12)
    assert train["actions"].shape == (15, 3)
    assert audit["observations"].shape == (6, 12)
    assert train["observations"].dtype == train["actions"].dtype == np.float32
    assert train["rewards"].dtype == np.float32
    assert train["terminated"].dtype == train["truncated"].dtype == bool
    assert train["episode_id"].dtype == train["row_id"].dtype == np.int64
    assert train["step_in_episode"].dtype == np.int32
    for source_id, step in collection.SOURCE_STEPS.items():
        assert np.sum(train["source_id"] == source_id) == 5
        assert np.sum(audit["source_id"] == source_id) == 2
        assert np.all(train["checkpoint_step"][train["source_id"] == source_id] == step)


def test_train_and_audit_have_disjoint_episodes_and_global_rows(fake_factory):
    _, (train, audit, _, _) = _collect()
    assert not np.intersect1d(train["episode_id"], audit["episode_id"]).size
    assert not np.intersect1d(train["row_id"], audit["row_id"]).size
    assert np.unique(np.concatenate((train["row_id"], audit["row_id"]))).size == 21


def test_public_schema_contains_command_only_and_no_hidden_fields(fake_factory):
    models, (train, audit, _, _) = _collect()
    assert set(train) == set(audit) == set(collection.PUBLIC_FIELDS)
    assert not (collection.FORBIDDEN_PUBLIC_FIELDS & set(train))
    assert all(flag is False for model in models.values() for flag in model.deterministic_flags)
    assert np.all(np.abs(train["actions"]) <= 1.0)


def test_hidden_audit_aligns_rows_and_uses_pre_step_u(fake_factory):
    models, (_, audit, hidden, _) = _collect()
    assert np.array_equal(hidden["row_id"], audit["row_id"])
    assert np.array_equal(hidden["source_id"], audit["source_id"])
    assert np.array_equal(hidden["episode_id"], audit["episode_id"])
    assert bool(hidden[collection.HIDDEN_METADATA_FIELD])
    expected_u = np.concatenate([model.seen_u[-2:] for model in models.values()])
    np.testing.assert_array_equal(hidden["hidden_u"], expected_u)
    assert all(current != following for current, following in zip(
        hidden["hidden_u"], np.concatenate([env.next_u for env in fake_factory[1::2]])
    ))


def test_hidden_applied_action_formula_and_bounds(fake_factory):
    _, (_, audit, hidden, _) = _collect(train_count=4, audit_count=4)
    expected_preclip = (
        audit["actions"] + 0.2 * hidden["hidden_u"][:, None] * ACTUATOR_DIRECTION
    )
    np.testing.assert_allclose(hidden["preclip_action"], expected_preclip, atol=1e-7)
    np.testing.assert_allclose(
        hidden["applied_action"], np.clip(expected_preclip, -1.0, 1.0), atol=1e-7
    )
    assert np.all(np.abs(audit["actions"]) <= 1.0)
    assert np.all(np.abs(hidden["applied_action"]) <= 1.0)


def test_budget_boundary_marks_only_collector_truncation(monkeypatch):
    monkeypatch.setattr(collection, "_make_environment", lambda kappa: FakeHopper(kappa, 100))
    public, hidden = collection._collect_stream(
        FakePolicy(), source_id=1, split="audit", transition_count=5, kappa=0.2,
        seeds=collection.collection_seed_plan(9)["source_1"]["audit"],
    )
    assert hidden is not None
    np.testing.assert_array_equal(public["collector_truncated"], (False,) * 4 + (True,))
    assert not np.any(public["terminated"] | public["truncated"])


def test_same_seed_reproduces_every_array(fake_factory):
    _, first = _collect(seed=71)
    _, second = _collect(seed=71)
    for first_group, second_group in zip(first[:3], second[:3]):
        assert set(first_group) == set(second_group)
        for field in first_group:
            np.testing.assert_array_equal(first_group[field], second_group[field])


def test_independent_streams_start_from_different_seeded_states(fake_factory):
    _, (train, audit, _, _) = _collect(train_count=2, audit_count=2)
    starts = []
    for data in (train, audit):
        for source_id in collection.SOURCE_STEPS:
            mask = (data["source_id"] == source_id) & (data["step_in_episode"] == 0)
            starts.append(data["observations"][mask][0])
    assert len({values.tobytes() for values in starts}) == 6


def test_artifact_writer_preserves_public_hidden_separation(tmp_path, fake_factory):
    _, (train, audit, hidden, _) = _collect()
    summary = {
        "train": collection._split_summary(train, None),
        "audit": collection._split_summary(audit, hidden),
    }
    collection.write_artifacts(tmp_path, train, audit, hidden, {"phase": "7A"}, summary)
    with np.load(tmp_path / "train_public.npz", allow_pickle=False) as stored_train:
        assert set(stored_train.files) == set(collection.PUBLIC_FIELDS)
    with np.load(tmp_path / "audit_public.npz", allow_pickle=False) as stored_audit:
        assert set(stored_audit.files) == set(collection.PUBLIC_FIELDS)
    with np.load(tmp_path / "audit_hidden.npz", allow_pickle=False) as stored_hidden:
        assert set(stored_hidden.files) == set(collection.HIDDEN_ARRAY_FIELDS) | {
            collection.HIDDEN_METADATA_FIELD
        }
    assert summary["train"]["source_1"]["transition_count"] == 5
    assert set(summary["audit"]["source_1"]["hidden_u_proportions"]) == {"-1", "+1"}
    assert "clipping_rate" in summary["audit"]["source_1"]


def test_smoke_cli_sets_fixed_small_counts_and_output():
    arguments = collection.parse_arguments(["--smoke", "--seed", "123"])
    assert arguments.train_transitions_per_source == 1_600
    assert arguments.audit_transitions_per_source == 400
    assert arguments.seed == 123
    assert arguments.output_dir == Path("artifacts/_smoke/hopper_method_pilot/stage_seed0")
