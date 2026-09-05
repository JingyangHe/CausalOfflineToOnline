from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift.phase8h_compute_matched_online_quick import (
    COMPONENT_UPDATES,
    GAMMA,
    ONLINE_METHODS,
    PBRS_BETA,
    POTENTIAL_EPOCHS,
    POTENTIAL_METHODS,
    SAC_CONFIG,
    TARGET_UPDATE_INTERVAL,
    _TerminalMaskedValue,
    _curve_diagnostics,
    _make_online_environment,
    _preflight_record,
    aggregate_dynamic_backup,
    commanded_replay_action,
    discounted_shaping_sum,
    file_fingerprint,
    normalized_auc,
    normalized_positive_area,
    pbrs_increment,
    resolve_artifact_root,
    run_online,
    strip_private_info,
)


def test_pbrs_terminal_and_truncation_semantics() -> None:
    assert pbrs_increment(2.0, 5.0, True) == pytest.approx(-2.0)
    assert pbrs_increment(2.0, 5.0, False) == pytest.approx(GAMMA * 5.0 - 2.0)
    # A time-limit truncation passes terminated=False and therefore bootstraps.
    assert pbrs_increment(2.0, 5.0, False) != pbrs_increment(2.0, 5.0, True)


def test_pbrs_discounted_telescope() -> None:
    phi = [2.0, 3.0, 5.0, 7.0]
    assert discounted_shaping_sum(phi, [False, False, False]) == pytest.approx(
        PBRS_BETA * (-phi[0] + GAMMA**3 * phi[-1]))
    assert discounted_shaping_sum(phi, [False, False, True]) == pytest.approx(
        -PBRS_BETA * phi[0])


def test_exact_positive_area_handles_zero_crossings() -> None:
    # The positive triangle has base 1 and height 1; normalize over width 2.
    assert normalized_positive_area([0, 2], [1.0, -1.0]) == pytest.approx(0.25)
    assert normalized_auc([0, 1, 2], [0.0, 2.0, 0.0]) == pytest.approx(1.0)


def test_dynamic_aggregation_definitions() -> None:
    source = np.asarray([
        [[1.0, 4.0], [5.0, 2.0]],
        [[2.0, 3.0], [4.0, 3.0]],
        [[0.0, 5.0], [6.0, 1.0]],
    ])
    pooled = np.asarray([[2.0, 7.0], [1.0, 4.0]])
    assert np.array_equal(aggregate_dynamic_backup("action_min_full", source), [3.0, 4.0])
    assert np.array_equal(aggregate_dynamic_backup("state_min_full", source), [3.0, 4.0])
    assert np.array_equal(
        aggregate_dynamic_backup("pooled_aamas_union_full", source, pooled), [7.0, 4.0])
    native = aggregate_dynamic_backup(
        "pooled_aamas_native_full", source, np.asarray([[2.0], [1.0]]))
    assert np.array_equal(native, [2.0, 1.0])


def test_terminal_mask_applies_to_every_candidate_and_alternative() -> None:
    value = _TerminalMaskedValue(lambda states: np.ones(len(states)), [False, True])
    assert np.array_equal(value(np.zeros((6, 12))), [1, 1, 1, 0, 0, 0])


def test_private_info_and_commanded_action_contract() -> None:
    private = {"hidden_u": 1, "applied_action": [1, 2, 3], "source_id": 2,
               "safe": "kept"}
    assert strip_private_info(private) == {"safe": "kept"}
    command = np.asarray([.1, -.2, .3], dtype=np.float64)
    replay = commanded_replay_action(command)
    assert replay.dtype == np.float32 and np.allclose(replay, command)


class _FakeHopper(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    spec = SimpleNamespace(max_episode_steps=2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(11, dtype=np.float32), {"base": True}

    def step(self, action):
        return np.ones(11, dtype=np.float32), 2.0, False, True, {"base": True}


def test_online_wrapper_hides_u_and_uses_final_truncated_observation(monkeypatch) -> None:
    monkeypatch.setattr(gym, "make", lambda _identifier: _FakeHopper())
    potential = lambda states: np.asarray(states)[:, 0]
    environment = _make_online_environment(
        potential, kappa=.2, lambda_reward=.01, beta=1.0, gamma=GAMMA)
    observation, info = environment.reset(seed=4)
    environment.env._hidden_u = 1
    following, reward, terminated, truncated, info = environment.step(np.zeros(3))
    assert observation.shape == following.shape == (12,)
    assert terminated is False and truncated is True
    assert info["pbrs_truncation_bootstraps"] is True
    assert info["raw_environment_reward"] == pytest.approx(2.01)
    assert reward == pytest.approx(2.01 + GAMMA)
    assert not {"hidden_u", "commanded_action", "applied_action"}.intersection(info)
    environment.close()


def test_artifact_resolver_uses_passed_manifest(tmp_path: Path) -> None:
    requested = tmp_path / "nominal"
    actual = tmp_path / "retry"
    actual.mkdir()
    (actual / "manifest.json").write_text(json.dumps({"stage": "Phase 8H-Q"}))
    (actual / "hard_checks.json").write_text(json.dumps({"all_passed": True}))
    assert resolve_artifact_root(requested, "Phase 8H-Q") == actual.resolve()


def test_lightweight_integrity_record_uses_blake2(tmp_path: Path) -> None:
    path = tmp_path / "x"; path.write_bytes(b"phase8h")
    record = file_fingerprint(path)
    assert set(record) == {"path", "size_bytes", "modified_time_ns", "blake2b_128"}
    assert len(record["blake2b_128"]) == 32


def test_preflight_frozen_counts(tmp_path: Path) -> None:
    phase8h = tmp_path / "phase8h"; scaling = tmp_path / "scaling"
    phase8h.mkdir(); scaling.mkdir()
    record = _preflight_record(phase8h, scaling, tmp_path / "out", [1, 2])
    assert record["missing_component_models"] == 8
    assert record["missing_required_existing_component_models"] == 16
    assert record["potential_count"] == 12
    assert record["online_run_count"] == 15
    assert record["online_training_environment_steps"] == 750_000
    assert record["estimated_evaluation_episodes"] == 450


def test_frozen_budgets_and_methods() -> None:
    assert COMPONENT_UPDATES == 4000 and POTENTIAL_EPOCHS == 200
    assert TARGET_UPDATE_INTERVAL == 3
    assert len(POTENTIAL_METHODS) == 4 and len(ONLINE_METHODS) == 5
    assert SAC_CONFIG["gamma"] == GAMMA and SAC_CONFIG["buffer_size"] == 1_000_000
    assert SAC_CONFIG["learning_starts"] == 100


def test_curve_audit_does_not_equate_late_best_with_nonconvergence() -> None:
    rows = [{"condition": "confounded", "data_label": "n32", "seed": 0,
             "model": "source_1", "step": step, "validation_loss": loss}
            for step, loss in ((1, 2.0), (10, 1.5), (20, 1.4), (30, 1.39))]
    result = _curve_diagnostics(rows)[0]
    assert result["best_step"] == 30
    assert result["descriptive_status"] in {"plateau_like", "validation_still_improving"}


def test_stage_c_source_declares_required_isolation() -> None:
    source = inspect.getsource(run_online)
    assert "replay_initial_size" in source
    assert "raw_environment_return" in source
    assert "parameter_fingerprint" in source
    assert "progress_bar=False" in source
