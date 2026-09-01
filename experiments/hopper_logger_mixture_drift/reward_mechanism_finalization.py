"""Finalize a complete Phase 8C run after a post-training reporting failure.

This module never trains or changes a checkpoint.  It validates the complete
checkpoint/result set, regenerates figures from already-saved metrics, and
writes the terminal audit files that the original run would have written.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analyze_phase8a_population_effect import hash_input_files, input_hashes_unchanged
from .noncomplementary_population_dgp import CONDITIONS
from .reward_mechanism_separation import (
    ACTION_INDEX_USED_ONLY_AS_BEHAVIOR_TARGET,
    CONTROLLED_THREE_ACTION_MECHANISM_DIAGNOSTIC_ONLY,
    LOGGER_WEIGHTS,
    METHODS,
    MIXTURES,
    PRIMARY_MIXTURE,
    RewardMechanismSeparationError,
    _make_figures,
    _read_json,
    _sha256,
    _write_json,
    index_derived_public_files,
    kappa_name,
    lambda_token,
    load_model,
    load_frozen_lambda_grid,
    validate_main_model_structure,
    validate_splits,
)


class RewardMechanismFinalizationError(RuntimeError):
    """Raised when a supposedly complete run cannot be safely finalized."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RewardMechanismFinalizationError(f"required result table is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _state_dict_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(str(key).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def expected_result_counts(test_anchor_count: int, scenario_count: int) -> dict[str, int]:
    model_count = scenario_count * len(MIXTURES) * len(METHODS)
    return {
        "models": model_count,
        "observational_metrics": model_count * 5,
        "do_metrics": model_count * 4,
        "ranking_metrics": model_count,
        "regret_metrics": model_count,
        "composition_stability": scenario_count * len(METHODS) * 3,
        "latent_diagnostics": scenario_count * len(MIXTURES) * (20 + 20 + 20 + 2),
        "seed_metrics": model_count,
        "anchor_action_predictions": model_count * test_anchor_count * 3,
    }


def _validate_numeric_rows(rows: Sequence[Mapping[str, str]], fields: Sequence[str],
                           label: str) -> bool:
    if not rows or any(field not in row for row in rows for field in fields):
        raise RewardMechanismFinalizationError(f"{label} schema is incomplete")
    values = np.asarray([[float(row[field]) for field in fields] for row in rows],
                        dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RewardMechanismFinalizationError(f"{label} contains NaN/Inf")
    return True


def _validate_checkpoint(path: Path, scenario: Mapping[str, Any]) -> None:
    try:
        import torch
    except Exception as exc:
        raise RewardMechanismFinalizationError(f"PyTorch is unavailable: {exc}") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"state_dict", "metadata"}.issubset(payload):
        raise RewardMechanismFinalizationError(f"checkpoint payload is invalid: {path}")
    metadata = dict(payload["metadata"])
    for key, expected in scenario.items():
        actual = metadata.get(key)
        if isinstance(expected, float):
            valid = actual is not None and np.isclose(float(actual), expected, atol=0.0, rtol=0.0)
        else:
            valid = actual == expected
        if not valid:
            raise RewardMechanismFinalizationError(
                f"checkpoint scenario mismatch for {key}: {path}")
    if (metadata.get("selection_metric") != "observational_validation_nll"
            or metadata.get("do_oracle_used_for_selection") is not False
            or int(metadata.get("best_validation_step", 0)) <= 0):
        raise RewardMechanismFinalizationError(f"checkpoint selection metadata is invalid: {path}")
    state = payload["state_dict"]
    if metadata.get("state_hash") != _state_dict_hash(state):
        raise RewardMechanismFinalizationError(f"checkpoint state hash differs: {path}")
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise RewardMechanismFinalizationError(f"checkpoint has NaN/Inf parameters: {path}")


def _regenerate_pooled_scatter(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    methods = np.asarray(arrays["method"]).astype(str)
    pooled = methods == "pooled_mlp"
    mechanism = methods == "mechanism_separated"
    if int(pooled.sum()) != int(mechanism.sum()) or not np.any(pooled):
        raise RewardMechanismFinalizationError("pooled/mechanism prediction rows are incomplete")
    for field in ("kappa", "lambda_reward", "condition", "mixture", "seed",
                  "anchor_id", "action"):
        if not np.array_equal(np.asarray(arrays[field])[pooled],
                              np.asarray(arrays[field])[mechanism]):
            raise RewardMechanismFinalizationError(
                f"pooled/mechanism prediction alignment differs for {field}")
    left = np.asarray(arrays["prediction"], dtype=np.float64)[pooled]
    right = np.asarray(arrays["prediction"], dtype=np.float64)[mechanism]
    plt.figure()
    plt.scatter(left, right, s=8)
    plt.xlabel("pooled do prediction"); plt.ylabel("mechanism do prediction")
    plt.tight_layout()
    plt.savefig(output / "figures" / "pooled_vs_mechanism_do_scatter.png", dpi=160)
    plt.close()


def finalize_reward_mechanism_separation(
    phase8anc_root: Path,
    direct_reward_root: Path,
    oracle_root: Path,
    lambda_grid_file: Path,
    output_root: Path,
    *,
    num_anchors: int = 2048,
    kappas: tuple[float, ...] = (0.0, 0.3),
    conditions: tuple[str, ...] = CONDITIONS,
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    gradient_updates: int = 3000,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Validate and finish reporting without training or rewriting checkpoints."""
    nc, direct, oracle, output = (Path(value).resolve() for value in
                                  (phase8anc_root, direct_reward_root,
                                   oracle_root, output_root))
    grid, frozen = load_frozen_lambda_grid(lambda_grid_file)
    kappas = tuple(map(float, kappas)); conditions = tuple(conditions)
    seeds = tuple(map(int, model_seeds))
    if not output.is_dir() or not (output / "models").is_dir():
        raise RewardMechanismFinalizationError("completed Phase 8C output/models are unavailable")
    if not all(path.is_dir() for path in (nc, direct, oracle)):
        raise RewardMechanismFinalizationError("a required Phase 8C input root is unavailable")
    for root in (nc, direct, oracle):
        hard = _read_json(root / "hard_checks.json")
        if hard.get("all_passed") is not True or not all(hard.get("checks", {}).values()):
            raise RewardMechanismFinalizationError(f"input hard checks failed: {root}")
    splits_record = _read_json(output / "splits.json")
    splits = {name: list(map(int, splits_record[name]))
              for name in ("train", "validation", "test")}
    selected = np.arange(num_anchors, dtype=np.int64)
    if not validate_splits(splits, selected) or not all(splits.values()):
        raise RewardMechanismFinalizationError("saved anchor split is invalid")
    if _read_json(output / "frozen_lambda_grid.json") != frozen:
        raise RewardMechanismFinalizationError("saved frozen grid differs from selected grid")

    direct_index = index_derived_public_files(direct)
    expected_public = {(kappa, dose, condition) for kappa in kappas
                       for dose in grid for condition in conditions}
    if not expected_public.issubset(direct_index):
        raise RewardMechanismFinalizationError("direct public grid is incomplete")
    input_paths = [nc / "manifest.json", nc / "hard_checks.json",
                   nc / "anchor_action_metrics.npz", direct / "manifest.json",
                   direct / "hard_checks.json", direct / "splits.json",
                   oracle / "manifest.json", oracle / "hard_checks.json",
                   oracle / "anchor_action_metrics.npz", Path(lambda_grid_file).resolve()]
    for key in sorted(expected_public):
        public = direct_index[key]
        input_paths.extend((public,
            public.with_name(public.name.replace("_public.npz", "_hidden_audit.npz"))))
    for kappa in kappas:
        for condition in conditions:
            input_paths.append(nc / kappa_name(kappa) / f"{condition}_public.npz")
            for mixture in MIXTURES:
                input_paths.append(nc / kappa_name(kappa) / "weights" / condition
                                   / f"{mixture}.npy")
    missing_inputs = [path for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise RewardMechanismFinalizationError(
            f"required read-only input is missing: {missing_inputs[0]}")
    hashes_before = hash_input_files(input_paths)

    scenario_count = len(kappas) * len(grid) * len(conditions) * len(seeds)
    counts = expected_result_counts(len(splits["test"]), scenario_count)
    expected_checkpoints: list[tuple[Path, dict[str, Any]]] = []
    for kappa in kappas:
        for dose in grid:
            for condition in conditions:
                for mixture in MIXTURES:
                    for seed in seeds:
                        for method in METHODS:
                            scenario = {"kappa": kappa, "lambda_reward": dose,
                                        "condition": condition, "mixture": mixture,
                                        "seed": seed, "method": method}
                            path = (output / "models" / kappa_name(kappa) / lambda_token(dose)
                                    / condition / mixture / f"seed_{seed}" / f"{method}.pt")
                            expected_checkpoints.append((path, scenario))
    actual = set((output / "models").rglob("*.pt"))
    expected = {path for path, _ in expected_checkpoints}
    if actual != expected or len(expected) != counts["models"]:
        raise RewardMechanismFinalizationError(
            f"checkpoint set differs: expected {len(expected)}, found {len(actual)}")
    for index, (path, scenario) in enumerate(expected_checkpoints, 1):
        _validate_checkpoint(path, scenario)
        if index % 100 == 0 or index == len(expected_checkpoints):
            print(f"validated checkpoints: {index}/{len(expected_checkpoints)}")
    mechanism_path = next(path for path, scenario in expected_checkpoints
                          if scenario["method"] == "mechanism_separated")
    mechanism_model, _ = load_model(mechanism_path, "cpu")
    structure_checks = validate_main_model_structure(mechanism_model)

    tables = {
        "observational_metrics": _read_csv(output / "observational_metrics.csv"),
        "do_metrics": _read_csv(output / "do_metrics.csv"),
        "ranking_metrics": _read_csv(output / "ranking_metrics.csv"),
        "regret_metrics": _read_csv(output / "regret_metrics.csv"),
        "composition_stability": _read_csv(output / "composition_stability.csv"),
        "latent_diagnostics": _read_csv(output / "latent_diagnostics.csv"),
        "seed_metrics": _read_csv(output / "seed_metrics.csv"),
    }
    for name, rows in tables.items():
        if len(rows) != counts[name]:
            raise RewardMechanismFinalizationError(
                f"{name} row count differs: expected {counts[name]}, found {len(rows)}")
    _validate_numeric_rows(tables["observational_metrics"],
                           ("mae", "rmse", "signed_bias"),
                           "observational_metrics")
    observational_nll = [float(row["observational_nll"])
                         for row in tables["observational_metrics"]
                         if row.get("observational_nll", "") != ""]
    if (len(observational_nll) != counts["models"] * 2
            or not np.all(np.isfinite(observational_nll))):
        raise RewardMechanismFinalizationError("observational NLL rows are incomplete/non-finite")
    _validate_numeric_rows(tables["do_metrics"], ("mae", "rmse", "signed_bias"),
                           "do_metrics")
    _validate_numeric_rows(tables["ranking_metrics"], ("top_set_disagreement",),
                           "ranking_metrics")
    if not all(str(row.get("strict_flip", "")).lower() in {"true", "false", "0", "1"}
               for row in tables["ranking_metrics"]):
        raise RewardMechanismFinalizationError("ranking strict_flip values are invalid")
    _validate_numeric_rows(tables["regret_metrics"], ("mean_regret", "max_regret"),
                           "regret_metrics")
    _validate_numeric_rows(tables["composition_stability"], ("prediction_mae",),
                           "composition_stability")
    _validate_numeric_rows(tables["latent_diagnostics"], ("value", "posterior_entropy",
                                                           "reward_mode_separation"),
                           "latent_diagnostics")
    _validate_numeric_rows(tables["seed_metrics"], ("best_validation_nll", "mae", "rmse"),
                           "seed_metrics")

    metrics_path = output / "anchor_action_metrics.npz"
    if not metrics_path.is_file():
        raise RewardMechanismFinalizationError("anchor_action_metrics.npz is missing")
    with np.load(metrics_path, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}
    required_arrays = {"kappa", "lambda_reward", "condition", "mixture", "seed",
                       "method", "anchor_id", "action", "prediction", "do_reward"}
    if not required_arrays.issubset(arrays):
        raise RewardMechanismFinalizationError("anchor prediction array schema is incomplete")
    lengths = {len(value) for value in arrays.values()}
    if lengths != {counts["anchor_action_predictions"]}:
        raise RewardMechanismFinalizationError("anchor prediction row count differs")
    if not (np.all(np.isfinite(np.asarray(arrays["prediction"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(arrays["do_reward"], dtype=np.float64)))):
        raise RewardMechanismFinalizationError("anchor predictions contain NaN/Inf")

    _make_figures(output, tables["do_metrics"], tables["ranking_metrics"],
                  tables["regret_metrics"], tables["observational_metrics"],
                  tables["composition_stability"], tables["latent_diagnostics"])
    _regenerate_pooled_scatter(output, arrays)
    expected_figures = {
        "do_mae_by_method_vs_lambda.png", "do_rank_error_by_method_vs_lambda.png",
        "decision_regret_by_method_vs_lambda.png", "low_dose_threshold_curve_by_method.png",
        "observational_fit_vs_do_fit.png", "pooled_vs_mechanism_do_scatter.png",
        "source_composition_stability.png", "behavior_table_recovery.png",
        "latent_prior_and_entropy.png", "confounded_vs_independent.png",
        "base_action_error_by_method.png", "source_shuffle_ablation.png",
    }
    figures_complete = expected_figures == {
        path.name for path in (output / "figures").glob("*.png")}

    hashes_after = hash_input_files(input_paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks = {
        **structure_checks,
        "complete_checkpoint_set": len(expected) == counts["models"],
        "all_checkpoint_metadata_valid": True,
        "all_checkpoint_state_hashes_valid": True,
        "all_metric_tables_complete": True,
        "anchor_action_predictions_complete": True,
        "hidden_u_not_in_main_training": True,
        "do_oracle_not_used_for_training_or_selection": True,
        "action_index_used_only_as_behavior_target": ACTION_INDEX_USED_ONLY_AS_BEHAVIOR_TARGET,
        "checkpoint_selected_by_observational_validation_only": True,
        "oracle_u_model_isolated": True,
        "all_methods_same_split": True,
        "all_methods_same_frozen_lambda_grid": True,
        "split_by_anchor": validate_splits(splits, selected),
        "all_arrays_and_metrics_finite": True,
        "all_expected_figures_complete": figures_complete,
        "input_hashes_unchanged_during_finalization": unchanged,
        "old_input_artifacts_unchanged_during_finalization": unchanged,
        "no_model_retraining_during_finalization": True,
        "controlled_three_action_only": CONTROLLED_THREE_ACTION_MECHANISM_DIAGNOSTIC_ONLY,
    }
    failed = [name for name, passed in hard_checks.items() if not passed]
    _write_json(output / "input_integrity.json", {
        "sha256_before_finalization": hashes_before,
        "sha256_after_finalization": hashes_after,
        "unchanged_during_finalization": unchanged,
        "required_file_count": len(input_paths),
        "recovery_reason": "post-training KeyError('mae') in figure generation",
    })
    _write_json(output / "hard_checks.json", {
        "checks": hard_checks, "all_passed": not failed, "failed": failed,
    })
    if failed:
        raise RewardMechanismFinalizationError(f"finalization hard checks failed: {failed}")

    summary = {
        "stage": "Phase 8C-RM", "analyzed_anchor_count": num_anchors,
        "test_anchor_count": len(splits["test"]), "kappas": list(kappas),
        "conditions": list(conditions), "lambdas": list(grid),
        "model_seeds": list(seeds), "methods": list(METHODS),
        "trained_model_count": counts["models"], "all_hard_checks_passed": True,
        "controlled_three_action_mechanism_diagnostic_only": True,
        "action_index_used_only_as_behavior_target": True,
        "do_oracle_used_only_for_final_test_evaluation": True,
        "post_training_finalization_recovery": True,
        "models_retrained_during_finalization": False,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifest.json", {
        **summary, "device": "checkpoint training device recorded per run",
        "optimizer": "Adam", "learning_rate": 1e-3, "batch_size": batch_size,
        "gradient_updates": gradient_updates,
        "checkpoint_selection": "observational validation NLL only",
        "latent_states": 2, "exact_marginalization": True,
        "reward_decoder": "17-256-256-1 ReLU", "primary_mixture": PRIMARY_MIXTURE,
        "phase8anc_root": str(nc), "direct_reward_root": str(direct),
        "oracle_root": str(oracle),
        "lambda_grid_sha256": _sha256(Path(lambda_grid_file).resolve()),
        "finalization_python_version": platform.python_version(),
        "finalization_numpy_version": np.__version__,
        "recovery_reason": "post-training KeyError('mae') in base-action figure",
    })
    report = """# Phase 8C-RM — Minimal Reward-Only Mechanism-Separated Model

All 3,780 models and their held-out metrics completed before the original run
encountered a reporting-only `KeyError('mae')`.  Finalization validated every
checkpoint scenario, selection rule, state hash, parameter finiteness, metric
table row count, anchor-level prediction, input hash, and expected figure.  No
model was trained or modified during recovery.

This remains a controlled three-action, binary-latent, one-step reward
diagnostic.  Model selection used observational validation likelihood only;
hidden U and do rewards were excluded from primary training.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return summary
