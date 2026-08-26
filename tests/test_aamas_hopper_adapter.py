"""Dependency-light tests for the Phase 7B AAMAS Hopper adapter."""

import inspect
from pathlib import Path
import subprocess

import numpy as np
import pytest

import aamas_hopper_adapter as adapter
from scripts.train_aamas_hopper_potential import (
    _architecture,
    parse_arguments,
    runtime_requirement_error,
    seed_everything,
)


class FakeTorch:
    float32 = np.float32
    bool = np.bool_

    @staticmethod
    def as_tensor(values, dtype=None, device=None):
        del device
        return np.asarray(values, dtype=dtype)


def _public_data(per_source=2):
    count = 3 * per_source
    source_id = np.repeat((1, 2, 3), per_source).astype(np.int8)
    steps = np.choose(source_id - 1, (200_000, 500_000, 1_000_000)).astype(np.int64)
    observations = np.arange(count * 12, dtype=np.float32).reshape(count, 12) / 100.0
    return {
        "observations": observations,
        "actions": np.linspace(-0.8, 0.8, count * 3, dtype=np.float32).reshape(count, 3),
        "rewards": np.linspace(1.0, 3.0, count, dtype=np.float32),
        "next_observations": observations + np.float32(0.1),
        "terminated": np.asarray(([False] * (count - 1)) + [True], dtype=bool),
        "truncated": np.asarray(([False] * (count - 2)) + [True, False], dtype=bool),
        "collector_truncated": np.zeros(count, dtype=bool),
        "source_id": source_id,
        "checkpoint_step": steps,
        "episode_id": np.arange(count, dtype=np.int64) + 10,
        "step_in_episode": np.zeros(count, dtype=np.int32),
        "row_id": np.arange(count, dtype=np.int64),
    }


def test_public_dataset_schema_and_shapes_are_accepted():
    data = _public_data()
    adapter.validate_public_dataset(data)
    assert data["observations"].shape[1] == 12
    assert data["actions"].shape[1] == 3


def test_hidden_and_simulator_fields_are_rejected_immediately():
    for field in ("hidden_u", "applied_action", "qpos", "qvel"):
        data = _public_data()
        data[field] = np.zeros(data["rewards"].size)
        with pytest.raises(RuntimeError, match="forbidden"):
            adapter.validate_public_dataset(data)


def test_three_sources_pool_without_metadata_model_tensors():
    data = _public_data(3)
    tensors, _ = adapter.convert_to_official_tensors(data, "cpu", torch_module=FakeTorch)
    assert tensors["observations"].shape == (9, 12)
    assert tensors["actions"].shape == (9, 3)
    assert tensors["rewards"].shape == (9, 1)
    assert set(tensors) == adapter.MODEL_TENSOR_FIELDS
    assert not ({"source_id", "checkpoint_step", "episode_id", "row_id"} & set(tensors))


def test_source_id_remains_available_only_for_audit_grouping():
    data = _public_data(2)
    grouped = {source: data["observations"][data["source_id"] == source] for source in (1, 2, 3)}
    assert all(values.shape == (2, 12) for values in grouped.values())
    tensors, _ = adapter.convert_to_official_tensors(data, "cpu", torch_module=FakeTorch)
    assert "source_id" not in tensors


def test_collector_truncation_never_enters_done_or_truncated():
    data = _public_data()
    data["collector_truncated"][0] = True
    assert not data["terminated"][0] and not data["truncated"][0]
    tensors, _ = adapter.convert_to_official_tensors(data, "cpu", torch_module=FakeTorch)
    assert not tensors["dones"][0, 0]
    assert not tensors["truncated"][0, 0]
    np.testing.assert_array_equal(
        tensors["dones"].reshape(-1), np.logical_or(data["terminated"], data["truncated"])
    )


def test_audit_rewards_use_unchanged_train_statistics():
    train = np.asarray((1.0, 2.0, 4.0), dtype=np.float32)
    _, train_statistics = adapter.normalize_rewards_like_official(train)
    audit = np.asarray((100.0, 200.0), dtype=np.float32)
    normalized_audit, returned = adapter.normalize_rewards_like_official(audit, train_statistics)
    assert returned == train_statistics
    expected = (audit - train_statistics["reward_mean"]) / (train_statistics["reward_std"] + 1e-7)
    np.testing.assert_array_equal(normalized_audit.reshape(-1), expected.astype(np.float32))


def test_reward_normalization_matches_official_toy_preprocessing():
    rewards = np.asarray((1.0, 2.0, 4.0), dtype=np.float32)
    normalized, statistics = adapter.normalize_rewards_like_official(rewards)
    official_mean = np.mean(rewards, dtype=np.float32)
    official_std = np.std(rewards, ddof=1, dtype=np.float32)
    official = (rewards - official_mean) / np.float32(official_std + 1e-7)
    np.testing.assert_array_equal(normalized.reshape(-1), official)
    assert statistics["normalized_b"] == float(np.max(official))
    assert statistics["calculation_rule"] == adapter.REWARD_RULE


def test_cli_gamma_default_is_fixed_to_point_99():
    arguments = parse_arguments([])
    assert arguments.gamma == 0.99


def test_reward_mode_and_behavior_width_are_fixed():
    assert adapter.REWARD_MODE == "b_norm"
    assert adapter.RELEASED_BEHAVIOR_HIDDEN_DIM == 1
    assert adapter.BEHAVIOR_HIDDEN_DIM == 128
    assert _architecture()["behavior"]["hidden_dim"] == 128


def test_checkpoint_metadata_contract_is_complete_and_enforced(tmp_path):
    expected = {
        "observation_dim", "action_dim", "gamma", "reward_mode", "seed",
        "architecture", "external_repo_path", "external_commit",
    }
    assert adapter.CHECKPOINT_METADATA_FIELDS == expected
    with pytest.raises(RuntimeError, match="incomplete"):
        adapter.save_aamas_checkpoint(tmp_path, object(), {"gamma": 0.99}, object())


def test_state_potential_is_state_only_and_shape_preserving():
    critic = lambda values: np.sum(values, axis=1, keepdims=True)
    phi = adapter._FrozenPotential(critic, np.zeros((1, 12)), np.ones((1, 12)), "cpu", None)
    assert list(inspect.signature(phi.__call__).parameters) == ["observations"]
    assert isinstance(phi(np.ones(12, dtype=np.float32)), float)
    assert phi(np.ones((4, 12), dtype=np.float32)).shape == (4,)
    with pytest.raises(ValueError, match="shape"):
        phi(np.ones(13, dtype=np.float32))
    with pytest.raises(TypeError):
        phi(np.ones(12), np.zeros(3))


def test_external_repository_commit_and_clean_gate(tmp_path):
    repository = tmp_path / "official"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=repository, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=repository, check=True)
    tracked = repository / "fin_train_value_state_new_continuous.py"
    tracked.write_text("fixed = True\n", encoding="utf-8")
    subprocess.run(("git", "add", tracked.name), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-q", "-m", "fixed"), cwd=repository, check=True)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert adapter.validate_external_repo(repository, commit)["clean"] is True
    assert adapter._import_official_module(repository).fixed is True
    assert not (repository / "__pycache__").exists()
    tracked.write_text("fixed = False\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        adapter.validate_external_repo(repository, commit)


def test_public_loader_never_opens_hidden_sidecar(tmp_path, monkeypatch):
    data = _public_data()
    np.savez_compressed(tmp_path / "train_public.npz", **data)
    forbidden = tmp_path / "audit_hidden.npz"
    forbidden.write_bytes(b"must not be read")
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        assert Path(path).name != "audit_hidden.npz"
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    loaded = adapter.load_hopper_aamas_data(tmp_path, "train")
    assert loaded["rewards"].shape == data["rewards"].shape


def test_runtime_checks_python_and_lists_missing_dependencies(monkeypatch):
    assert "Python >= 3.12" in runtime_requirement_error((3, 11, 9), lambda _: object())
    missing = runtime_requirement_error((3, 12, 0), lambda _: None)
    assert missing == "MISSING_DEPENDENCIES: torch, torchrl, tensordict, minari"

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeDeterministicTorch:
        cuda = FakeCuda()
        backends = type("FakeBackends", (), {})()

        @staticmethod
        def manual_seed(seed):
            assert seed == 7

        @staticmethod
        def use_deterministic_algorithms(enabled, warn_only=False):
            assert enabled is True
            assert warn_only is True

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    warnings = seed_everything(7, FakeDeterministicTorch(), cuda_training=True)
    assert any("CUBLAS_WORKSPACE_CONFIG" in warning for warning in warnings)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert seed_everything(7, FakeDeterministicTorch(), cuda_training=True) == []
