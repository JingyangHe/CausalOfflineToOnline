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

from confounded_hopper import ACTUATOR_DIRECTION
from experiments.hopper_logger_mixture_drift import reward_mechanism_separation as phase


def _raw(anchor_count: int = 3, condition: str = "confounded") -> dict[str, np.ndarray]:
    values: dict[str, list] = {key: [] for key in
        ("row_id", "anchor_id", "observation", "commanded_action", "reward",
         "logger_id", "condition", "kappa_env", "lambda_reward")}
    row = 0
    for anchor in range(anchor_count):
        observation = np.linspace(0.0, 1.1, 12, dtype=np.float32) + anchor
        base = np.asarray([0.1, -0.1, 0.2], dtype=np.float32)
        minus = (base - 0.2 * ACTUATOR_DIRECTION).astype(np.float32)
        plus = (base + 0.2 * ACTUATOR_DIRECTION).astype(np.float32)
        for logger, actions in ((0, (minus, plus, minus, plus)),
                                (1, (minus, plus, minus, plus)), (2, (base, base))):
            for action in actions:
                values["row_id"].append(row); values["anchor_id"].append(anchor)
                values["observation"].append(observation); values["commanded_action"].append(action)
                values["reward"].append(float(anchor + np.sum(action)))
                values["logger_id"].append(logger); values["condition"].append(condition)
                values["kappa_env"].append(0.0); values["lambda_reward"].append(0.005)
                row += 1
    return {key: np.asarray(value) for key, value in values.items()}


def _rows() -> phase.PublicRows:
    raw = _raw()
    return phase.make_public_rows(raw, np.ones(len(raw["row_id"])), (0.45, 0.45, 0.10))


def _torch_or_skip():
    if sys.platform == "win32":
        pytest.skip("Phase 8C torch roundtrip is exercised in the Linux training environment")
    try:
        return phase._torch()
    except phase.RewardMechanismSeparationError as exc:
        pytest.skip(str(exc))


def test_frozen_lambda_grid_required(tmp_path: Path):
    with pytest.raises(phase.LambdaGridNotFrozenError, match="LAMBDA_GRID_NOT_FROZEN"):
        phase.load_frozen_lambda_grid(tmp_path / "missing.json")


def test_test_thresholds_not_used_for_grid(tmp_path: Path):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps({"manually_frozen": True, "lambdas": [0, .005, .05],
                                "selection_basis": ["test_threshold_quantiles"]}), encoding="utf-8")
    with pytest.raises(phase.LambdaGridNotFrozenError, match="test-threshold"):
        phase.load_frozen_lambda_grid(path)


def test_public_only_main_training():
    assert not phase.FORBIDDEN_MAIN_FIELDS.intersection(phase.PublicRows.__dataclass_fields__)
    signature = inspect.signature(phase.train_reward_model)
    assert not phase.FORBIDDEN_MAIN_FIELDS.intersection(signature.parameters)


def test_source_only_enters_behavior():
    source = inspect.getsource(phase.make_model)
    assert "behavior_logits" in source
    assert "decoder_input = 20 if kind == \"source_dependent_reward\" else 17" in source
    assert "mechanism_separated" in source


def test_reward_decoder_is_source_invariant():
    _torch_or_skip()
    model = phase.make_model("mechanism_separated", 0)
    assert model.reward_decoder[0].in_features == 17
    assert phase.validate_main_model_structure(model)["reward_decoder_is_source_invariant"]


def test_prior_is_source_invariant():
    _torch_or_skip()
    model = phase.make_model("mechanism_separated", 0)
    assert tuple(model.prior_logits.shape) == (2,)
    assert phase.validate_main_model_structure(model)["prior_is_source_invariant"]


def test_action_index_behavior_target_only():
    source = inspect.getsource(phase.make_model)
    assert "behavior_log_probs()[source, :, action_index]" in source
    assert phase.ACTION_INDEX_USED_ONLY_AS_BEHAVIOR_TARGET is True
    assert "action_index" not in inspect.signature(phase.predict_do).parameters


def test_exact_latent_marginalization():
    prior = np.log([.4, .6]); behavior = np.log(np.asarray([
        [[.7, .2, .1], [.1, .2, .7]], [[.6, .3, .1], [.2, .3, .5]],
        [[.1, .8, .1], [.1, .8, .1]]]))
    means = np.asarray([[0., 1.], [1., 2.]])
    result = phase.exact_joint_log_likelihood_numpy(
        prior, behavior, means, np.asarray([.5, 1.5]), np.asarray([0, 1]),
        np.asarray([0, 2]), 0.0)
    components = []
    for i, (source, action) in enumerate(((0, 0), (1, 2))):
        density = sum(np.exp(prior[z] + behavior[source, z, action])
                      * np.exp(-.5 * (([.5, 1.5][i] - means[i, z]) ** 2))
                      / np.sqrt(2 * np.pi) for z in range(2))
        components.append(np.log(density))
    assert np.allclose(result, components)


def test_joint_likelihood_finite():
    result = phase.exact_joint_log_likelihood_numpy(
        np.zeros(2), np.zeros((3, 2, 3)), np.zeros((4, 2)), np.zeros(4),
        np.asarray([0, 1, 2, 0]), np.asarray([0, 1, 2, 1]), -1.0)
    assert np.all(np.isfinite(result))


def test_row_weight_normalization():
    raw = _raw()
    result = phase.normalized_logger_row_weights(
        np.arange(1, len(raw["row_id"]) + 1), raw["logger_id"], (.45, .45, .10))
    assert np.isclose(result.sum(), 1.0)
    assert np.allclose([result[raw["logger_id"] == logger].sum() for logger in range(3)],
                       [.45, .45, .10])


def test_anchor_split_disjoint():
    selected = np.arange(30)
    existing = {"train": [0, 1], "validation": [2], "test": [3]}
    split = phase.extend_or_reuse_splits(existing, selected, 7)
    sets = [set(split[name]) for name in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(selected)
    assert all(anchor in split[name] for name in existing for anchor in existing[name])


def test_do_oracle_not_used_for_selection():
    signature = inspect.signature(phase.train_reward_model)
    assert "do_oracle" not in signature.parameters
    source = inspect.getsource(phase.train_reward_model)
    assert "best_validation_nll" in source and "do_oracle_used_for_selection" in source


def test_checkpoint_roundtrip(tmp_path: Path):
    _torch_or_skip()
    model = phase.make_model("mechanism_separated", 11)
    path = tmp_path / "model.pt"
    phase.save_model(path, model, {"kind": "mechanism_separated", "seed": 11})
    loaded, metadata = phase.load_model(path, "cpu")
    assert metadata["kind"] == "mechanism_separated"
    assert phase._state_hash(model) == phase._state_hash(loaded)


def test_source_shuffle_changes_labels():
    source = np.asarray([0, 0, 1, 1, 2, 2])
    shuffled = phase.shuffled_sources(source, 0)
    assert sorted(shuffled.tolist()) == sorted(source.tolist())
    assert not np.array_equal(shuffled, source)


def test_oracle_u_model_is_isolated():
    main_signature = inspect.signature(phase.train_reward_model)
    oracle_signature = inspect.signature(phase.train_oracle_u_model)
    assert "train_u_env" not in main_signature.parameters
    assert "train_u_env" in oracle_signature.parameters
    assert "Oracle/evaluation" in (phase.__doc__ or "")


def test_independent_negative_control_available():
    assert "independent_latents" in phase.CONDITIONS
    assert "conditions" in inspect.signature(phase.run_reward_mechanism_separation).parameters


def test_base_action_control_available():
    raw = _raw(1)
    indices = phase.commanded_action_indices(raw)
    assert set(indices.tolist()) == {0, 1, 2}
    assert np.all(indices[raw["logger_id"] == 2] == 1)


def test_all_methods_evaluated_on_same_test_anchors():
    raw = _raw(3)
    ids, observations, actions = phase.derive_test_action_table(raw, [0, 2])
    assert ids.tolist() == [0, 2]
    assert observations.shape == (2, 12) and actions.shape == (2, 3, 3)
    source = inspect.getsource(phase.run_reward_mechanism_separation)
    assert "for method in METHODS" in source and "test_anchor_ids" in source


def test_regret_nonnegative():
    do = np.asarray([[3., 2., 1.], [1., 2., 3.]])
    prediction = np.asarray([[1., 3., 2.], [3., 2., 1.]])
    result = phase.regret_metrics(do, prediction)
    assert result["mean_regret"] >= 0 and result["max_regret"] >= 0


def test_input_hashes_unchanged(tmp_path: Path):
    path = tmp_path / "input.bin"; path.write_bytes(b"read-only")
    before = phase.hash_input_files([path]); phase._sha256(path)
    assert phase.input_hashes_unchanged(before, phase.hash_input_files([path]))


def test_no_nan_inf():
    rows = _rows(); stats = phase.fit_normalization(rows)
    assert all(np.all(np.isfinite(value)) for value in
               (stats.x_mean, stats.x_std, stats.reward_mean, stats.reward_std,
                phase.normalized_x(rows, stats)))


def test_old_artifacts_unchanged():
    source = inspect.getsource(phase.run_reward_mechanism_separation)
    assert "hashes_before" in source and "hashes_after" in source
    assert '"old_artifacts_unchanged": unchanged' in source


def test_cli_completion_and_blocking_markers():
    text = (ROOT / "scripts/run_hopper_reward_mechanism_separation.py").read_text(encoding="utf-8")
    assert "LAMBDA_GRID_NOT_FROZEN" in text
    assert "PHASE8C_REWARD_MECHANISM_SEPARATION_COMPLETE" in text
    assert "READY_FOR_MECHANISM_EFFECT_REVIEW" in text
    assert "PHASE8C_REWARD_MECHANISM_SEPARATION_BLOCKED" in text
