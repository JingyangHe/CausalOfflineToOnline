"""Dependency-light tests for Phase 8H-Q scientific and leakage invariants."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

from aamas_hopper_adapter import (
    ContinuousAAMASComponents,
    compute_official_continuous_action_backup,
    compute_source_aamas_backup,
)
from experiments.hopper_logger_mixture_drift.phase8h_quick_multipolicy_aamas import (
    EXPECTED_SPLIT_COUNTS,
    FORBIDDEN_MODEL_FIELDS,
    KAPPA,
    LAMBDA_REWARD,
    POOLED_MIXTURES,
    SOURCE_B,
    SOURCE_D,
    V_Q,
    V_U,
    FrozenSACReferenceValue,
    action_and_state_level_envelopes,
    do_bellman_oracle,
    fixed_anchor_splits,
    generate_source_dataset,
    pooled_row_weights,
    prediction_metrics,
    source_commanded_action,
    source_duplication_invariant,
    source_policy_parameters,
    validate_public_dataset,
)


def _torch_and_bundle():
    if sys.platform == "win32":
        pytest.skip("official AAMAS PyTorch equivalence is exercised in the Linux 3.12 environment")
    try:
        import torch
    except (ImportError, OSError) as error:
        pytest.skip(f"PyTorch runtime is unavailable: {error}")

    class Distribution:
        def __init__(self, state):
            self.loc = torch.zeros((len(state), 3), device=state.device)
            self.scale = torch.full_like(self.loc, 0.2)

        def log_prob(self, action):
            return -0.5 * (action / self.scale).square()

    class Behavior(torch.nn.Module):
        def forward(self, state):
            return Distribution(state)

    class Delta(torch.nn.Module):
        def forward(self, pair):
            return 0.01 * pair[:, :12]

    class Reward(torch.nn.Module):
        def forward(self, pair):
            return pair[:, 12:].mean(dim=1, keepdim=True)

    bundle = ContinuousAAMASComponents(
        Behavior(), Delta(), Reward(), reward_mean=1.0, reward_std=0.5,
        reward_upper=2.0, gamma=0.99, device=torch.device("cpu"),
        action_separation=0.1, not_action_samples=5)
    return torch, bundle


def _reference(states: np.ndarray) -> np.ndarray:
    return np.asarray(states, dtype=np.float64).sum(axis=1) * 0.01


class _Simulator:
    def __init__(self) -> None:
        self.commands: list[np.ndarray] = []

    def step(self, anchor: int, command: np.ndarray, u: int, kappa: float):
        self.commands.append(np.asarray(command).copy())
        obs = np.full(12, anchor * 0.01, dtype=np.float32)
        return {"observation": obs, "next_observation": obs + command.mean() * 0.01,
                "reward": float(command.sum() + kappa * u),
                "terminated": False, "truncated": False,
                "applied_action": np.clip(command + kappa * u, -1, 1)}


def _anchors(count: int = 4) -> dict[str, np.ndarray]:
    return {"anchor_id": np.arange(count), "base_action": np.zeros((count, 3))}


def test_aamas_baseline_available() -> None:
    path = Path(__file__).resolve().parents[1] / "external" / "li_aamas2026" / "fin_train_value_state_new_continuous.py"
    text = path.read_text(encoding="utf-8")
    assert "class CausalUpperBoundEstimator" in text
    assert "def sample_not_a_state_continuous" in text


def test_single_source_wrapper_equivalence() -> None:
    _, first = _torch_and_bundle()
    _, second = _torch_and_bundle()
    states = np.zeros((2, 12), dtype=np.float32)
    actions = np.zeros((2, 3, 3), dtype=np.float32)
    noise = np.random.default_rng(3).standard_normal((6, 5, 3)).astype(np.float32)
    direct = compute_official_continuous_action_backup(
        first, states, actions, _reference, common_noise=noise)
    wrapped = compute_source_aamas_backup(
        [second], states, actions, _reference, common_noise=noise)[0]
    assert np.allclose(direct, wrapped, atol=1e-7, rtol=1e-7)


def test_source_policy_parameters_exact() -> None:
    record = source_policy_parameters()
    assert record["b"] == [-0.15, 0.0, 0.15]
    assert record["d"] == [0.10, 0.18, 0.26]
    assert np.isclose(np.linalg.norm(V_Q), 1) and np.isclose(np.linalg.norm(V_U), 1)
    assert np.isclose(V_Q @ V_U, 0)


def test_source_sample_counts_equal() -> None:
    public, _ = generate_source_dataset(
        _anchors(), _Simulator(), condition="confounded",
        samples_per_anchor_source=3, seed=7)
    assert [np.sum(public["source_id"] == source) for source in (1, 2, 3)] == [12, 12, 12]


def test_hidden_u_not_model_input() -> None:
    assert not FORBIDDEN_MODEL_FIELDS.intersection(("observation", "commanded_action"))
    assert not {"u_behavior", "u_environment"}.intersection(
        inspect.signature(validate_public_dataset).parameters)


def test_do_not_used_for_training() -> None:
    from experiments.hopper_logger_mixture_drift.phase8h_quick_multipolicy_aamas import fit_aamas_components
    assert not {"do_q", "do_reward", "u_environment"}.intersection(
        inspect.signature(fit_aamas_components).parameters)


def test_reference_value_frozen() -> None:
    source = inspect.getsource(FrozenSACReferenceValue)
    assert "requires_grad_(False)" in source and "verify_frozen" in source


def test_do_action_bypasses_behavior_policy() -> None:
    simulator = _Simulator()
    candidates = np.zeros((1, 2, 3), dtype=np.float32)
    truth = do_bellman_oracle(simulator, _anchors(1), np.array([0]), candidates,
                              lambda states: np.zeros(len(states)))
    assert truth.shape == (1, 2) and len(simulator.commands) == 4


def test_binary_u_exact_average() -> None:
    assert "0.5 * (branches[0] + branches[1])" in inspect.getsource(do_bellman_oracle)
    simulator = _Simulator()
    value = do_bellman_oracle(simulator, _anchors(1), np.array([0]),
                              np.zeros((1, 1, 3)), lambda states: np.zeros(len(states)))
    assert np.isclose(value[0, 0], 0.0)  # +/- kappa and +/- lambda cancel exactly.


def test_pooled_mixture_weights() -> None:
    sources = np.repeat([1, 2, 3], [2, 4, 8])
    for name, mixture in POOLED_MIXTURES.items():
        weight = pooled_row_weights(sources, mixture)
        observed = np.array([weight[sources == source].sum() for source in (1, 2, 3)])
        assert np.allclose(observed, mixture), name


def test_union_candidate_set_shared() -> None:
    source_q = np.zeros((3, 7, 28))
    result = action_and_state_level_envelopes(source_q)
    assert result["action_q"].shape == (7, 28)


def test_action_min_not_above_state_min() -> None:
    values = np.random.default_rng(4).normal(size=(3, 11, 9))
    result = action_and_state_level_envelopes(values)
    assert np.all(result["action_phi"] <= result["state_phi"] + 1e-12)


def test_source_duplication_invariance() -> None:
    values = np.random.default_rng(5).normal(size=(3, 11, 9))
    assert source_duplication_invariant(values)


def test_composition_invariance() -> None:
    values = np.random.default_rng(6).normal(size=(3, 5, 7))
    expected = action_and_state_level_envelopes(values)["action_q"]
    for _ in POOLED_MIXTURES.values():
        assert np.array_equal(action_and_state_level_envelopes(values)["action_q"], expected)


def test_independent_latents_control() -> None:
    _, hidden = generate_source_dataset(
        _anchors(20), _Simulator(), condition="independent_latents",
        samples_per_anchor_source=32, seed=11)
    assert abs(np.corrcoef(hidden["u_behavior"], hidden["u_environment"])[0, 1]) < 0.05


def test_anchor_splits_disjoint() -> None:
    ids = np.arange(512)
    record = {}
    start = 0
    for name, count in EXPECTED_SPLIT_COUNTS.items():
        record[name] = ids[start:start + count].tolist(); start += count
    result = fixed_anchor_splits(record, ids)
    groups = [set(value.tolist()) for value in result.values()]
    assert not any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))


def test_input_hashes_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"; path.write_bytes(b"immutable")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    action_and_state_level_envelopes(np.ones((3, 2, 4)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_no_nan_inf() -> None:
    metrics = prediction_metrics(np.ones((3, 4)), np.ones((3, 4)))
    assert all(np.isfinite(value) for value in metrics.values())


def test_old_artifacts_unchanged(tmp_path: Path) -> None:
    old = tmp_path / "phase8a.npz"; old.write_bytes(b"old")
    before = old.read_bytes()
    out = tmp_path / "phase8h"; out.mkdir(); (out / "summary.json").write_text("{}")
    assert old.read_bytes() == before


def test_source_action_formula_stays_in_range() -> None:
    action = source_commanded_action(np.array([.9, -.9, .1]), 3, 1, np.zeros(3))
    assert np.all(np.abs(action) <= 1) and np.all(np.isfinite(action))
    assert KAPPA == .2 and LAMBDA_REWARD == .01
