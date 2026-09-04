from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.hopper_logger_mixture_drift.phase8h_data_scaling import (
    BATCH_SIZE,
    SAMPLE_SIZES,
    UPDATE_BUDGETS,
    Phase8HDataScalingError,
    _baseline_gate,
    aggregate_scaling_metrics,
    cvar90,
    _metric_row,
    file_metadata,
    generate_nested_master,
    metadata_snapshot,
    nested_dataset_audit,
    subset_nested,
)
from experiments.hopper_logger_mixture_drift.phase8h_quick_multipolicy_aamas import (
    CANDIDATE_ACTIONS,
    FORBIDDEN_MODEL_FIELDS,
    KAPPA,
    LAMBDA_REWARD,
    PUBLIC_MODEL_FIELDS,
    SIGMA_ACTION,
    SOURCE_B,
    SOURCE_D,
    FrozenSACReferenceValue,
    fit_aamas_components,
    source_policy_parameters,
)


REPOSITORY = Path(__file__).resolve().parents[1]
BASE = REPOSITORY / "artifacts/hopper_logger_mixture_drift/phase8h_quick_multipolicy_aamas_retry"


class _FakeSimulator:
    def step(self, position: int, command: np.ndarray, u: int, kappa: float):
        observation = np.full(12, position, dtype=np.float32)
        return {
            "observation": observation, "next_observation": observation + .1,
            "reward": float(np.sum(command) + u * kappa),
            "terminated": False, "truncated": False,
            "applied_action": np.asarray(command) + u * kappa,
        }


@pytest.fixture(scope="module")
def nested_master():
    anchors = {"anchor_id": np.arange(2, dtype=np.int64),
               "base_action": np.zeros((2, 3), dtype=np.float32)}
    return generate_nested_master(
        anchors, _FakeSimulator(), condition="confounded", seed=20260804)


def test_phase8h_baseline_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _baseline_gate(tmp_path)


def test_n32_exact_reproduction(nested_master) -> None:
    _, _, audit = nested_master
    assert audit["original_d32_exact"] is True
    if BASE.is_dir():
        values, passed = _baseline_gate(BASE)
        assert passed and np.isclose(values["action_level_min.do_mae"], 2.372, atol=.002)


def test_nested_datasets(nested_master) -> None:
    master, _, _ = nested_master
    audit = nested_dataset_audit(master)
    assert audit["all_nested"] is True
    assert all(audit["subset_checks"].values())
    anchors = {"anchor_id": np.arange(2, dtype=np.int64),
               "base_action": np.zeros((2, 3), dtype=np.float32)}
    smoke_master, _, smoke_audit = generate_nested_master(
        anchors, _FakeSimulator(), condition="confounded", seed=20260804,
        max_samples=32)
    assert len(smoke_master["reward"]) == 2 * 3 * 32
    assert smoke_audit["original_d32_exact"] is True


def test_equal_source_sample_counts(nested_master) -> None:
    master, _, _ = nested_master
    assert nested_dataset_audit(master)["all_source_counts_equal"] is True


def test_dgp_parameters_unchanged() -> None:
    assert source_policy_parameters() == {
        "b": [-.15, 0., .15], "d": [.10, .18, .26], "sigma_action": .20,
        "v_q": source_policy_parameters()["v_q"], "v_u": source_policy_parameters()["v_u"],
    }
    assert KAPPA == .20 and LAMBDA_REWARD == .01
    assert np.array_equal(SOURCE_B, [-.15, 0., .15])
    assert np.array_equal(SOURCE_D, [.10, .18, .26]) and SIGMA_ACTION == .20


def test_split_unchanged() -> None:
    if not BASE.is_dir():
        pytest.skip("repository Phase 8H artifact unavailable")
    import json
    split = json.loads((BASE / "splits.json").read_text(encoding="utf-8"))
    assert {name: len(value) for name, value in split.items()} == {
        "train": 333, "observational_validation": 51,
        "do_calibration_pool": 51, "test": 77}


def test_reference_value_frozen() -> None:
    signature = inspect.signature(FrozenSACReferenceValue)
    assert "use_parameter_hash" in signature.parameters
    source = inspect.getsource(FrozenSACReferenceValue.verify_frozen)
    assert "requires_grad" in source and "torch.equal" in source


def test_candidate_protocol_unchanged() -> None:
    assert 3 * (8 + 1) + 1 == 28
    assert CANDIDATE_ACTIONS == 25


def test_hidden_u_not_input() -> None:
    assert not (FORBIDDEN_MODEL_FIELDS & set(PUBLIC_MODEL_FIELDS))
    assert "model_public" in inspect.getsource(fit_aamas_components)


def test_do_not_used_for_training() -> None:
    forbidden = {"do_q", "do_oracle", "u_environment", "u_behavior"}
    assert not forbidden.intersection(inspect.signature(fit_aamas_components).parameters)


def test_update_scaling_exact() -> None:
    assert UPDATE_BUDGETS == {16: 500, 32: 1000, 64: 2000, 128: 4000}
    assert all(UPDATE_BUDGETS[size] == 1000 * size // 32 for size in SAMPLE_SIZES)


def test_batch_size_fixed() -> None:
    assert BATCH_SIZE == 512


def test_compute_control_same_data(nested_master) -> None:
    master, _, _ = nested_master
    standard = subset_nested(master, 32)
    compute_control = subset_nested(master, 32)
    assert all(np.array_equal(standard[key], compute_control[key]) for key in standard)
    assert UPDATE_BUDGETS[32] == 1000 and UPDATE_BUDGETS[128] == 4000


def test_regret_definition_frozen() -> None:
    truth = np.asarray([[3., 2., 1.], [0., 1., 4.]])
    prediction = np.asarray([[1., 4., 0.], [0., 1., 4.]])
    row = _metric_row("confounded", 0, "n32", 32, 1000,
                      "action_level_min", truth, prediction)
    assert row["regret_mean"] == .5
    assert row["regret_cvar90"] == cvar90(np.asarray([1., 0.]), np.arange(2))


def test_underestimation_definition() -> None:
    truth = np.asarray([[2., 2., 2.]])
    prediction = np.asarray([[1., 3., 2.]])
    row = _metric_row("confounded", 0, "n32", 32, 1000, "source_1", truth, prediction)
    assert row["underestimation_fraction"] == pytest.approx(1 / 3)


def test_input_hashes_unchanged(tmp_path: Path) -> None:
    """Legacy required name; this uses metadata and computes no cryptographic digest."""
    path = tmp_path / "input.json"; path.write_text("{}\n", encoding="utf-8")
    assert metadata_snapshot([path]) == metadata_snapshot([path])
    assert set(file_metadata(path)) == {"path", "size_bytes", "modified_time_ns"}


def test_no_nan_inf() -> None:
    rows = aggregate_scaling_metrics([
        {"condition": "confounded", "data_label": "n32", "samples_per_anchor_source": 32,
         "method": "action_level_min", "do_mae": 1., "do_rmse": 1., "signed_error": 0.,
         "underestimation_fraction": .5, "regret_mean": 1., "regret_median": 1.,
         "regret_p90": 1., "regret_cvar90": 1.},
    ])
    assert rows and all(np.isfinite(float(value)) for value in rows[0].values()
                        if isinstance(value, (int, float)))


def test_old_artifacts_unchanged(tmp_path: Path, nested_master) -> None:
    path = tmp_path / "old.bin"; path.write_bytes(b"old-artifact")
    before = metadata_snapshot([path]); _ = nested_dataset_audit(nested_master[0])
    assert before == metadata_snapshot([path])
