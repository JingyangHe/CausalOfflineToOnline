from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift import phase8c_failure_decomposition as phase


def _torch_or_skip():
    if sys.platform == "win32":
        pytest.skip("Phase 8C-FD torch invariants run in the Linux training environment")
    try:
        return phase._torch()
    except Exception as exc:
        pytest.skip(str(exc))


def _behavior_manifest(path: Path) -> Path:
    value = {
        "action_keys": ["minus", "base", "plus"],
        "logger_probability_tables": {
            "0": {"-1": {"minus": .9, "plus": .1},
                  "1": {"minus": .1, "plus": .9}},
            "1": {"-1": {"minus": .7, "plus": .3},
                  "1": {"minus": .3, "plus": .7}},
            "2": {"-1": {"base": 1.0}, "1": {"base": 1.0}},
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_phase8c_inputs_required(tmp_path: Path):
    with pytest.raises(phase.FailureDecompositionError, match="input root is unavailable"):
        phase.validate_phase8c_inputs(*(tmp_path / name for name in ("a", "b", "c", "d")))


def test_existing_results_reproduced():
    artifact = ROOT / "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified"
    facts = phase.reproduce_existing_results(
        artifact / "phase8c_reward_mechanism_separation",
        ROOT / "analysis/phase8c_reward_mechanism_strict_analysis")
    assert facts["all_reproduced"] and facts["mechanism_model_count"] == 35


def test_true_behavior_loaded_from_manifest(tmp_path: Path):
    table = phase.load_true_behavior_table(_behavior_manifest(tmp_path / "manifest.json"))
    assert table.shape == (3, 2, 3)
    assert np.allclose(table.sum(axis=2), 1.0)
    assert table[0, 1, 2] == .9 and table[2, 0, 1] == 1.0


def test_true_behavior_fixed_has_no_grad(tmp_path: Path):
    _torch_or_skip()
    model = phase.make_true_behavior_fixed_model(
        0, phase.load_true_behavior_table(_behavior_manifest(tmp_path / "manifest.json")))
    assert not model.prior_logits.requires_grad
    assert not model.behavior_logits.requires_grad
    assert any(parameter.requires_grad for parameter in model.reward_decoder.parameters())


def test_public_only_variants_do_not_read_u():
    for function in (phase.train_public_observational, phase.train_true_behavior_em):
        assert not {"u", "u_env", "train_u", "test_u"}.intersection(
            inspect.signature(function).parameters)
    assert set(phase.PUBLIC_ONLY_VARIANTS) == set(phase.VARIANTS[:4])


def test_oracle_variants_are_isolated():
    registry = phase.variant_registry()
    assert registry["oracle_information_used_for_diagnosis_only"] is True
    assert registry["oracle_variants_are_not_deployable"] is True
    assert all(registry["variants"][variant]["data"].startswith("oracle")
               for variant in phase.ORACLE_VARIANTS)


def test_exact_latent_marginalization():
    prior = np.asarray([.4, .6])
    behavior = np.asarray([[[.8, .1, .1], [.2, .1, .7]]] * 3)
    means = np.asarray([[0., 1.], [1., 2.]])
    reward = np.asarray([.25, 1.75])
    source = np.asarray([0, 2]); action = np.asarray([0, 2]); weight = np.asarray([1., 1.])
    value = phase.exact_observed_nll_numpy(
        prior, behavior, means, reward, source, action, weight, 0.0)
    densities = []
    for index in range(2):
        densities.append(sum(prior[z] * behavior[source[index], z, action[index]]
                         * np.exp(-.5 * (reward[index] - means[index, z]) ** 2)
                         / np.sqrt(2 * np.pi) for z in range(2)))
    assert np.isclose(value, -np.mean(np.log(densities)))


def test_collapsed_model_has_identical_branches():
    torch = _torch_or_skip()
    model = phase.make_collapsed_reference(0)
    x = torch.randn(8, 15)
    source = torch.arange(8) % 3
    means = model.latent_means(x).detach().numpy()
    beta = model.behavior_probabilities().detach().numpy()
    assert np.array_equal(means[:, 0], means[:, 1])
    assert np.array_equal(beta[:, 0], beta[:, 1])
    assert torch.equal(model.plain_mean(x), model.plain_mean(x, source))


def test_em_responsibilities_sum_to_one():
    prior = np.asarray([.5, .5]); behavior = np.full((3, 2, 3), 1 / 3)
    q = phase.exact_responsibilities_numpy(
        prior, behavior, np.asarray([[0., 1.], [1., 2.]]), np.asarray([.2, 1.8]),
        np.asarray([0, 1]), np.asarray([0, 2]), -1.0)
    assert np.allclose(q.sum(axis=1), 1.0)


def test_em_observed_nll_finite():
    value = phase.exact_observed_nll_numpy(
        np.asarray([.5, .5]), np.full((3, 2, 3), 1 / 3),
        np.asarray([[0., 1.], [1., 2.]]), np.asarray([.2, 1.8]),
        np.asarray([0, 1]), np.asarray([0, 2]), np.asarray([.4, .6]), -1.0)
    assert np.isfinite(value)


def test_oracle_plugin_uses_true_prior_and_behavior(tmp_path: Path):
    torch = _torch_or_skip()
    truth = phase.load_true_behavior_table(_behavior_manifest(tmp_path / "manifest.json"))
    oracle = phase.make_model("oracle_u_aware", 0)
    plugin = phase.make_oracle_plugin(oracle, 0, truth)
    assert np.allclose(torch.softmax(plugin.prior_logits, 0).detach().numpy(), [.5, .5])
    assert np.allclose(torch.softmax(plugin.behavior_logits, 2).detach().numpy(), truth)
    assert not any(parameter.requires_grad for parameter in plugin.parameters())


def test_oracle_init_update_zero_matches_plugin(tmp_path: Path):
    _torch_or_skip()
    truth = phase.load_true_behavior_table(_behavior_manifest(tmp_path / "manifest.json"))
    plugin = phase.make_oracle_plugin(phase.make_model("oracle_u_aware", 0), 0, truth)
    initialized = phase.make_oracle_initialized_joint(plugin, 0)
    assert phase._state_hash(plugin) == phase._state_hash(initialized)


def test_label_permutation_invariance(tmp_path: Path):
    truth = phase.load_true_behavior_table(_behavior_manifest(tmp_path / "manifest.json"))
    direct, _ = phase.best_label_permutation_behavior_error(truth, truth)
    flipped, permutation = phase.best_label_permutation_behavior_error(truth[:, ::-1], truth)
    assert direct == 0.0 and flipped == 0.0 and permutation == (1, 0)


def test_reward_profile_endpoints():
    left, right = np.asarray([1., 2.]), np.asarray([3., 6.])
    collapsed = phase.reward_profile_means(left, right, 0.0)
    oracle = phase.reward_profile_means(left, right, 1.0)
    assert np.array_equal(collapsed[:, 0], collapsed[:, 1])
    assert np.allclose(oracle, np.stack((left, right), axis=1))


def test_behavior_profile_endpoints(tmp_path: Path):
    truth = phase.load_true_behavior_table(_behavior_manifest(tmp_path / "manifest.json"))
    collapsed = phase.behavior_profile_table(truth, 0.0)
    assert np.array_equal(collapsed[:, 0], collapsed[:, 1])
    assert np.allclose(phase.behavior_profile_table(truth, 1.0), truth)


def test_do_not_select_checkpoint_by_do():
    for function in (phase.train_public_observational, phase.train_true_behavior_em):
        source = inspect.getsource(function)
        assert "do_oracle_used_for_selection" in source
        assert '"observational_validation_nll"' in source
        assert "do_reward" not in inspect.signature(function).parameters


def test_same_split_all_variants():
    source = inspect.getsource(phase.run_phase8c_failure_decomposition)
    assert 'rows.subset(splits["train"])' in source
    assert 'rows.subset(splits["validation"])' in source
    assert 'rows.subset(splits["test"])' in source
    assert '"all_variants_same_public_split"' in source


def test_input_hashes_unchanged(tmp_path: Path):
    path = tmp_path / "input.bin"; path.write_bytes(b"immutable")
    before = phase.hash_input_files([path])
    assert phase.input_hashes_unchanged(before, phase.hash_input_files([path]))


def test_no_nan_inf():
    rows = [{"a": 1.0, "b": "", "c": True}, {"a": -2.5, "b": "scope"}]
    assert phase._all_numeric_finite(rows)
    assert not phase._all_numeric_finite([{"a": np.inf}])


def test_old_artifacts_unchanged():
    source = inspect.getsource(phase.run_phase8c_failure_decomposition)
    assert "hashes_before" in source and "hashes_after" in source
    assert '"old_artifacts_unchanged": unchanged' in source
