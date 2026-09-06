from __future__ import annotations

import numpy as np

from experiments.hopper_logger_mixture_drift import phase8j_core_bootstrap_isolation as bi
from scripts.run_phase8j_core_bootstrap_isolation import parse_arguments


def _curve(unstable: set[str]) -> list[dict[str, str]]:
    rows = []
    for variant in bi.VARIANTS:
        for epoch in (0, 150, 160, 170, 180, 190, 200):
            growing = variant in unstable
            value = 1.0 + (epoch - 140) / 10 if growing and epoch else 1.0
            rows.append({
                "variant": variant,
                "epoch": str(epoch),
                "training_loss": str(value),
                "validation_loss": str(value),
                "prediction_std": str(value),
                "prediction_p99": str(value),
                "gradient_norm_mean": str(value),
                "target_std": str(value),
            })
    return rows


def test_frozen_contract_and_cli() -> None:
    assert bi.VARIANTS == (
        "reward_only", "real_transition_bootstrap", "full_aamas_frozen"
    )
    assert (bi.MODEL_SEED, bi.SAMPLES_PER_ANCHOR_SOURCE, bi.COMPONENT_UPDATES) == (0, 128, 4000)
    assert (bi.OUTER_ITERATIONS, bi.INNER_EPOCHS, bi.TOTAL_EPOCHS) == (40, 5, 200)
    assert bi.GAMMA == 0.99
    args = parse_arguments([
        "--phase", "train", "--variants", *bi.VARIANTS,
        "--model-seed", "0", "--outer-iterations", "40",
        "--inner-epochs", "5", "--total-epochs", "200",
    ])
    assert args.variants == list(bi.VARIANTS)


def test_reward_only_target_is_fixed_logged_reward() -> None:
    public = {"reward": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_array_equal(
        bi.reward_only_target(public, np.array([2, 0])), np.array([3.0, 1.0])
    )


def test_real_transition_target_uses_terminated_but_not_truncated() -> None:
    public = {
        "reward": np.array([2.0, 2.0, 2.0]),
        "next_observation": np.zeros((3, 12), dtype=np.float32),
        "terminated": np.array([False, True, False]),
        "truncated": np.array([False, False, True]),
    }
    target = bi.real_transition_target(
        public, np.arange(3), lambda states: np.full(len(states), 7.0)
    )
    np.testing.assert_allclose(
        target, np.array([2.0 + 0.99 * 7.0, 2.0, 2.0 + 0.99 * 7.0])
    )


def test_termination_audit_returns_complete_parent_semantics() -> None:
    audit = bi._termination_audit()
    assert audit["terminated_zeroes_continuation"] is True
    assert audit["truncated_retains_continuation"] is True
    assert audit["matches_parent_semantics"] is True


def test_minibatch_rows_are_deterministic_and_outer_specific() -> None:
    rows = np.arange(4097)
    first = bi._minibatch_audit(rows)
    second = bi._minibatch_audit(rows)
    assert first == second
    digests = [row["row_sequence_blake2b_128"] for row in first["outer_row_sequences"]]
    assert len(set(digests)) == bi.OUTER_ITERATIONS
    assert first["identical_across_variants"] is True


def test_root_layer_classification_cases() -> None:
    assert bi.classify_root_layer(_curve({"reward_only"}))[0] == "A_BASIC_SUPERVISED_REGRESSION"
    root, labels = bi.classify_root_layer(_curve({
        "real_transition_bootstrap", "full_aamas_frozen"
    }))
    assert root == "B_REAL_TRANSITION_BOOTSTRAP"
    assert "CORE_BOOTSTRAP_VALUE_FITTING_INSTABILITY" in labels
    assert bi.classify_root_layer(_curve({"full_aamas_frozen"}))[0] == \
        "C_AAMAS_SPECIFIC_BACKUP"
    assert bi.classify_root_layer(_curve(set()))[0] == "ROOT_CAUSE_LAYER_NOT_IDENTIFIED"


def test_case_d_detects_mild_core_growth_amplified_by_full_backup() -> None:
    rows = _curve(set())
    for row in rows:
        epoch = int(row["epoch"])
        if epoch < 150:
            continue
        if row["variant"] == "real_transition_bootstrap":
            row["prediction_std"] = str(1.0 + (epoch - 140) / 20.0)
        elif row["variant"] == "full_aamas_frozen":
            value = str(1.0 + (epoch - 140) / 10.0)
            for key in (
                "training_loss", "validation_loss", "prediction_std",
                "prediction_p99", "gradient_norm_mean", "target_std",
            ):
                row[key] = value
    root, labels = bi.classify_root_layer(rows)
    assert root == "B_REAL_TRANSITION_BOOTSTRAP"
    assert "CORE_BOOTSTRAP_INSTABILITY_AMPLIFIED_BY_AAMAS_BACKUP" in labels


def test_finite_metric_check_accepts_csv_boolean_fields() -> None:
    rows = [[{
        "variant": "real_transition_bootstrap",
        "epoch": "5",
        "value_drift_rmse": "0.1",
        "target_hash_constant_for_all_inner_epochs": "True",
    }]]
    assert bi._all_scalar_metrics_finite(rows)
    rows[0][0]["value_drift_rmse"] = "nan"
    assert not bi._all_scalar_metrics_finite(rows)


def test_recomputed_target_dispatches_with_current_network(monkeypatch) -> None:
    observed = {}

    def fake(*args, **kwargs):
        observed["network"] = args[4]
        observed["outer"] = args[7]
        observed["validation"] = kwargs["validation"]
        return np.array([1.0])

    monkeypatch.setattr(bi, "_target_for_rows", fake)
    current = object()
    result = bi._recomputed_target(
        "reward_only", {}, {}, np.array([0]), current,
        np.zeros((1, 12)), np.ones((1, 12)), 7, 3,
    )
    np.testing.assert_array_equal(result, np.array([1.0]))
    assert observed == {"network": current, "outer": 7, "validation": True}
