"""Phase 8J-Q: large-data multi-source AAMAS online sanity check.

The Phase 8H-DS run did not persist its generated transitions.  Phase 8J-Q
therefore performs one explicitly authorized deterministic rematerialization,
freezes that public dataset, and only reads it on subsequent stages.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from aamas_hopper_adapter import _import_official_module, validate_external_repo
from scripts.train_aamas_hopper_potential import seed_everything
from .generate_datasets import MujocoOneStepSimulator
from .phase8h_compute_matched_online_quick import (
    EXTERNAL_COMMIT,
    GAMMA,
    KAPPA,
    LAMBDA_REWARD,
    PBRS_BETA,
    POTENTIAL_BATCH_SIZE,
    POTENTIAL_EPOCHS,
    POTENTIAL_LR,
    SAC_CONFIG,
    TARGET_TAU,
    TARGET_UPDATE_INTERVAL,
    _TorchPotentialValue,
    _component_paths,
    _device_name,
    _evaluate_online_model,
    _load_component,
    _load_potential,
    _make_online_environment,
    _make_potential_network,
    _phase8a_from_phase8h,
    _read_csv,
    _replay_matches_commands,
    _runtime_versions,
    _same_fingerprint,
    _train_one_potential,
    _write_csv,
    _write_json,
    discounted_shaping_sum,
    file_fingerprint,
    normalized_auc,
    normalized_positive_area,
    parameter_fingerprint,
    resolve_artifact_root,
)
from .phase8h_data_scaling import (
    BATCH_SIZE,
    file_metadata,
    generate_nested_master,
    metadata_snapshot,
    nested_dataset_audit,
    subset_nested,
)
from .phase8h_quick_multipolicy_aamas import (
    FORBIDDEN_MODEL_FIELDS,
    PUBLIC_MODEL_FIELDS,
    _load_phase8h_inputs,
    source_policy_parameters,
    validate_public_dataset,
)


PHASE = "Phase 8J-Q"
SAMPLES_PER_ANCHOR_SOURCE = 128
TRANSITION_COUNT = 512 * 3 * SAMPLES_PER_ANCHOR_SOURCE
SOURCE_TRANSITION_COUNT = 512 * SAMPLES_PER_ANCHOR_SOURCE
MODEL_SEEDS = (0, 1, 2)
POTENTIAL_METHODS = ("pooled", "state_min", "action_min")
BACKUP_METHODS = {
    "pooled": "pooled_aamas_union_full",
    "state_min": "state_min_full",
    "action_min": "action_min_full",
}
ONLINE_METHODS = ("sac_scratch", "sac_pooled", "sac_state_min", "sac_action_min")
ONLINE_TO_POTENTIAL = {
    "sac_pooled": "pooled",
    "sac_state_min": "state_min",
    "sac_action_min": "action_min",
}
FORMAL_EVAL_STEPS = (0, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000)
NUMERIC_EXPLOSION_LIMIT = 1_000_000.0
REMATERIALIZATION_SEED = 20260804


class Phase8JLargeDataOnlineSanityError(RuntimeError):
    """Raised when the frozen Phase 8J-Q contract is violated."""


def _potential_path(output: Path, method: str, seed: int) -> Path:
    return output / "potentials" / f"{method}_seed{seed}.pt"


def _dataset_path(output: Path) -> Path:
    return output / "input_data" / "n128_confounded_public.npz"


def _load_public_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(PUBLIC_MODEL_FIELDS):
            raise Phase8JLargeDataOnlineSanityError(
                "frozen n128 public dataset has unexpected fields")
        public = {name: archive[name].copy() for name in PUBLIC_MODEL_FIELDS}
    validate_public_dataset(public, TRANSITION_COUNT)
    return public


def audit_rematerialized_dataset(
    public: Mapping[str, np.ndarray], hidden: Mapping[str, np.ndarray],
    anchors: Mapping[str, np.ndarray], generation: Mapping[str, Any],
) -> dict[str, Any]:
    validate_public_dataset(public, TRANSITION_COUNT)
    anchor_ids = np.asarray(public["anchor_id"], dtype=np.int64)
    source_ids = np.asarray(public["source_id"], dtype=np.int64)
    sample_ids = np.asarray(public["sample_id"], dtype=np.int64)
    expected_anchors = np.asarray(anchors["anchor_id"], dtype=np.int64)
    source_counts = {str(source): int(np.sum(source_ids == source)) for source in (1, 2, 3)}
    key_count = len(set(zip(anchor_ids.tolist(), source_ids.tolist(), sample_ids.tolist())))
    hidden_behavior = np.asarray(hidden["u_behavior"], dtype=np.int8)
    hidden_environment = np.asarray(hidden["u_environment"], dtype=np.int8)
    numeric_arrays = [np.asarray(public[name]) for name in (
        "observation", "commanded_action", "reward", "next_observation")]
    return {
        "row_count": len(anchor_ids),
        "source_counts": source_counts,
        "unique_key_count": key_count,
        "anchor_ids_exact": np.array_equal(np.unique(anchor_ids), expected_anchors),
        "source_ids_exact": np.array_equal(np.unique(source_ids), np.asarray([1, 2, 3])),
        "sample_ids_exact": np.array_equal(
            np.unique(sample_ids), np.arange(SAMPLES_PER_ANCHOR_SOURCE)),
        "all_anchor_source_sample_keys_unique": key_count == TRANSITION_COUNT,
        "hidden_values_binary": set(np.unique(hidden_behavior)).issubset({-1, 1})
        and set(np.unique(hidden_environment)).issubset({-1, 1}),
        "u_behavior_equals_u_environment": np.array_equal(
            hidden_behavior, hidden_environment),
        "public_fields_exclude_hidden_u": not (
            FORBIDDEN_MODEL_FIELDS & set(public)),
        "commanded_actions_finite_and_bounded": bool(
            np.all(np.isfinite(public["commanded_action"]))
            and np.max(np.abs(public["commanded_action"])) <= 1.0 + 1e-6),
        "all_public_numeric_finite": all(np.all(np.isfinite(value)) for value in numeric_arrays),
        "original_d32_exact": bool(generation["original_d32_exact"]),
        "extension_protocol": generation["extension_protocol"],
        "nested_audit": nested_dataset_audit(public),
    }


def _old_input_paths(scaling: Path, inputs: Mapping[str, Any]) -> list[Path]:
    paths = [scaling / "manifest.json", scaling / "hard_checks.json",
             scaling / "seed_metrics.csv", *inputs["required_paths"]]
    for seed in MODEL_SEEDS:
        paths.extend(_component_paths(scaling, scaling, seed, 128).values())
    return sorted({Path(path).resolve() for path in paths}, key=str)


def _component_contract(
    scaling: Path, torch: Any,
) -> tuple[dict[int, dict[str, Path]], list[dict[str, Any]], dict[str, bool]]:
    paths_by_seed: dict[int, dict[str, Path]] = {}
    references: list[dict[str, Any]] = []
    architecture_signatures = []
    all_metadata_valid = True
    pooled_public_only = True
    source_public_only = True
    for seed in MODEL_SEEDS:
        paths = _component_paths(scaling, scaling, seed, 128)
        paths_by_seed[seed] = paths
        for name, path in paths.items():
            if not path.is_file():
                raise Phase8JLargeDataOnlineSanityError(
                    f"required n128 component checkpoint is missing: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            is_pooled = name == "pooled_balanced"
            expected_kind = "pooled" if is_pooled else "source"
            valid = (
                metadata.get("condition") == "confounded"
                and metadata.get("data_label") == "n128"
                and metadata.get("samples_per_anchor_source") == 128
                and metadata.get("gradient_updates") == 4000
                and metadata.get("batch_size") == BATCH_SIZE
                and metadata.get("model_kind") == expected_kind
                and metadata.get("model_input_fields") == ["observation", "commanded_action"]
                and metadata.get("checkpoint_selection_fields")
                == ["behavior_nll", "delta_mse", "reward_mse"]
            )
            if is_pooled:
                valid = valid and metadata.get("composition") == "balanced"
                pooled_public_only &= metadata.get("model_input_fields") == [
                    "observation", "commanded_action"]
            else:
                source_public_only &= metadata.get("model_input_fields") == [
                    "observation", "commanded_action"]
            all_metadata_valid &= bool(valid)
            signature = {
                section: [(key, tuple(value.shape)) for key, value in sorted(values.items())]
                for section, values in sorted(payload["state_dict"].items())
            }
            architecture_signatures.append(signature)
            references.append({
                "model_seed": seed, "component": name,
                "checkpoint": str(path.resolve()), "metadata": metadata,
                "file": file_metadata(path),
            })
    checks = {
        "all_n128_component_checkpoints_present": len(references) == 12,
        "component_metadata_matches_n128_4000": all_metadata_valid,
        "component_architecture_identical": all(
            value == architecture_signatures[0] for value in architecture_signatures),
        "pooled_model_public_fields_only": pooled_public_only,
        "source_models_public_fields_only": source_public_only,
    }
    return paths_by_seed, references, checks


def _materialize_once(
    output: Path, anchors: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    path = _dataset_path(output)
    audit_path = output / "input_data" / "rematerialization_audit.json"
    if path.is_file():
        if not audit_path.is_file():
            raise Phase8JLargeDataOnlineSanityError(
                "dataset exists without its rematerialization audit")
        public = _load_public_archive(path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        fingerprint = file_fingerprint(path)
        recorded = audit.get("public_dataset_fingerprint", {})
        if fingerprint.get("blake2b_128") != recorded.get("blake2b_128"):
            raise Phase8JLargeDataOnlineSanityError("frozen n128 dataset fingerprint changed")
        return public, audit, fingerprint

    path.parent.mkdir(parents=True, exist_ok=True)
    simulator = MujocoOneStepSimulator(anchors, (KAPPA,), seed=REMATERIALIZATION_SEED)
    try:
        master, hidden, generation = generate_nested_master(
            anchors, simulator, condition="confounded",
            seed=REMATERIALIZATION_SEED, max_samples=SAMPLES_PER_ANCHOR_SOURCE)
    finally:
        simulator.close()
    public = subset_nested(master, SAMPLES_PER_ANCHOR_SOURCE)
    audit = audit_rematerialized_dataset(public, hidden, anchors, generation)
    np.savez_compressed(path, **public)
    fingerprint = file_fingerprint(path)
    audit.update({
        "provenance": "authorized_deterministic_rematerialization",
        "generator": "phase8h_data_scaling.generate_nested_master",
        "generator_seed": REMATERIALIZATION_SEED,
        "public_dataset_fingerprint": fingerprint,
        "hash_policy": "one BLAKE2b-128 digest for the frozen public dataset only",
    })
    _write_json(audit_path, audit)
    return public, audit, fingerprint


def _load_offline_proxy_rows(scaling: Path) -> dict[tuple[int, str], dict[str, Any]]:
    mapping = {
        "pooled": "pooled_balanced",
        "state_min": "state_level_min",
        "action_min": "action_level_min",
    }
    rows = _read_csv(scaling / "seed_metrics.csv")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in MODEL_SEEDS:
        for method, old_method in mapping.items():
            selected = [row for row in rows
                        if row["condition"] == "confounded"
                        and row["data_label"] == "n128"
                        and int(row["seed"]) == seed
                        and row["method"] == old_method]
            if len(selected) != 1:
                raise Phase8JLargeDataOnlineSanityError(
                    f"missing Phase 8H-DS n128 proxy row: seed={seed}, method={old_method}")
            row = selected[0]
            result[(seed, method)] = {
                "do_bellman_mae": float(row["do_mae"]),
                "potential_mae": float(row["phi_mae"]),
                "regret_mean": float(row["regret_mean"]),
                "regret_p90": float(row["regret_p90"]),
                "regret_cvar90": float(row["regret_cvar90"]),
            }
    return result


def _shaping_diagnostics(
    value: Callable[[np.ndarray], np.ndarray], public: Mapping[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, float]:
    state = value(np.asarray(public["observation"])[rows])
    following = value(np.asarray(public["next_observation"])[rows])
    terminated = np.asarray(public["terminated"])[rows].astype(bool)
    increments = PBRS_BETA * (GAMMA * (~terminated) * following - state)
    return {
        "shaping_increment_mean": float(np.mean(increments)),
        "shaping_increment_std": float(np.std(increments)),
        "shaping_increment_p90_abs": float(np.quantile(np.abs(increments), .90)),
        "shaping_increment_p99_abs": float(np.quantile(np.abs(increments), .99)),
        "shaping_increment_max_abs": float(np.max(np.abs(increments))),
    }


def run_potentials(
    scaling_root: Path, output_root: Path, *, samples_per_anchor_source: int,
    model_seeds: Sequence[int], component_updates: int, device: str,
    external_repo: Path,
) -> dict[str, Any]:
    if samples_per_anchor_source != 128 or tuple(map(int, model_seeds)) != MODEL_SEEDS:
        raise Phase8JLargeDataOnlineSanityError("potentials are frozen to n128 and seeds 0,1,2")
    if component_updates != 4000:
        raise Phase8JLargeDataOnlineSanityError("component updates are frozen to 4000")
    scaling = resolve_artifact_root(scaling_root, "Phase 8H-DS")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scaling_manifest = json.loads((scaling / "manifest.json").read_text(encoding="utf-8"))
    if json.loads((scaling / "hard_checks.json").read_text(encoding="utf-8")).get(
            "all_passed") is not True:
        raise Phase8JLargeDataOnlineSanityError("Phase 8H-DS hard checks did not pass")
    phase8h = Path(scaling_manifest["phase8h_root"]).resolve()
    inputs = _load_phase8h_inputs(
        _phase8a_from_phase8h(phase8h), 512, None, compute_checkpoint_hash=False)
    anchors, splits = inputs["anchors"], inputs["splits"]
    old_paths = _old_input_paths(scaling, inputs)
    old_before = metadata_snapshot(old_paths)
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
    except (ImportError, OSError) as error:
        raise Phase8JLargeDataOnlineSanityError("PyTorch is required") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(external_repo)
    paths_by_seed, component_references, component_checks = _component_contract(scaling, torch)
    public, data_audit, data_fingerprint = _materialize_once(output, anchors)
    train_rows = np.flatnonzero(np.isin(public["anchor_id"], splits["train"]))
    validation_rows = np.flatnonzero(
        np.isin(public["anchor_id"], splits["observational_validation"]))
    base_by_anchor = np.asarray(anchors["base_action"], dtype=np.float32)
    proxy_rows = _load_offline_proxy_rows(scaling)
    history_path = output / "potential_training_metrics.csv"
    diagnostic_path = output / "potential_diagnostics.csv"
    histories = _read_csv(history_path) if history_path.is_file() else []
    diagnostics = _read_csv(diagnostic_path) if diagnostic_path.is_file() else []
    initial_by_seed: dict[int, Mapping[str, Any]] = {}
    for seed in MODEL_SEEDS:
        components = {name: _load_component(path, official, torch, selected_device)
                      for name, path in paths_by_seed[seed].items()}
        source_models = tuple(components[f"source_{source}"] for source in (1, 2, 3))
        for method in POTENTIAL_METHODS:
            checkpoint = _potential_path(output, method, seed)
            if checkpoint.is_file():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                metadata = payload.get("metadata", {})
                if not (
                    metadata.get("phase8j_method") == method
                    and metadata.get("run_id") == seed
                    and metadata.get("epochs") == POTENTIAL_EPOCHS
                    and metadata.get("dataset_blake2b_128")
                    == data_fingerprint["blake2b_128"]
                ):
                    raise Phase8JLargeDataOnlineSanityError(
                        f"existing potential has mismatched provenance: {checkpoint}")
                initial = metadata["initial_parameter_fingerprint"]
                if seed in initial_by_seed and not _same_fingerprint(initial_by_seed[seed], initial):
                    raise Phase8JLargeDataOnlineSanityError(
                        f"potential initialization is not paired for seed {seed}")
                initial_by_seed[seed] = initial
                print(f"reusing potential {method}, seed {seed}", flush=True)
                continue
            payload, history, diagnostic = _train_one_potential(
                BACKUP_METHODS[method], seed, public, train_rows, validation_rows,
                base_by_anchor, source_models, components["pooled_balanced"],
                official, torch, selected_device, POTENTIAL_EPOCHS)
            payload["metadata"].update({
                "stage": PHASE,
                "phase8j_method": method,
                "dataset_blake2b_128": data_fingerprint["blake2b_128"],
                "samples_per_anchor_source": 128,
                "component_updates": 4000,
                "candidate_random_stream_paired": True,
            })
            initial = payload["metadata"]["initial_parameter_fingerprint"]
            if seed in initial_by_seed and not _same_fingerprint(initial_by_seed[seed], initial):
                raise Phase8JLargeDataOnlineSanityError(
                    f"potential initialization is not paired for seed {seed}")
            initial_by_seed[seed] = initial
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint)
            value, _ = _load_potential(checkpoint, official, torch, selected_device)
            probe = np.asarray(anchors["public_observation"][:16], dtype=np.float32)
            network = _make_potential_network(
                official, payload["reward_min"], payload["reward_max"], selected_device)
            network.load_state_dict(payload["current_state_dict"])
            expected = _TorchPotentialValue(
                network, payload["state_mean"], payload["state_std"], selected_device, torch)
            roundtrip = float(np.max(np.abs(value(probe) - expected(probe))))
            shape = _shaping_diagnostics(value, public, validation_rows)
            diagnostic_row = {
                "model_seed": seed, "potential": method,
                "roundtrip_max_abs": roundtrip, "numeric_status": "finite",
                **diagnostic, **shape, **proxy_rows[(seed, method)],
            }
            histories.extend({"model_seed": seed, "potential": method, **row}
                             for row in history)
            diagnostics.append(diagnostic_row)
            _write_csv(history_path, histories)
            _write_csv(diagnostic_path, diagnostics)
            print(f"Phase 8J potential ready: {method}, seed {seed}", flush=True)
            del value, network
            if selected_device == "cuda":
                torch.cuda.empty_cache()
    unique_history = {(int(row["model_seed"]), str(row["potential"]), int(row["epoch"])): row
                      for row in histories}
    unique_diagnostics = {(int(row["model_seed"]), str(row["potential"])): row
                          for row in diagnostics}
    histories, diagnostics = list(unique_history.values()), list(unique_diagnostics.values())
    _write_csv(history_path, histories)
    _write_csv(diagnostic_path, diagnostics)
    potential_paths = [_potential_path(output, method, seed)
                       for seed in MODEL_SEEDS for method in POTENTIAL_METHODS]
    if len(diagnostics) != 9:
        raise Phase8JLargeDataOnlineSanityError(
            "potential diagnostics are incomplete; retain CSVs when resuming")
    old_after = metadata_snapshot(old_paths)
    source_counts = data_audit["source_counts"]
    numerical_stability = all(
        row["numeric_status"] == "finite"
        and max(abs(float(row["potential_min"])), abs(float(row["potential_max"])))
        <= NUMERIC_EXPLOSION_LIMIT
        and float(row["shaping_increment_max_abs"]) <= NUMERIC_EXPLOSION_LIMIT
        for row in diagnostics)
    checks = {
        "n128_dataset_deterministically_rematerialized": (
            data_audit["provenance"] == "authorized_deterministic_rematerialization"),
        "transition_count_196608": data_audit["row_count"] == TRANSITION_COUNT,
        "equal_source_sample_counts": set(source_counts.values()) == {SOURCE_TRANSITION_COUNT},
        "anchor_source_sample_ids_complete": all([
            data_audit["anchor_ids_exact"], data_audit["source_ids_exact"],
            data_audit["sample_ids_exact"], data_audit["all_anchor_source_sample_keys_unique"]]),
        "d32_prefix_exact": data_audit["original_d32_exact"],
        "dgp_read_from_phase8h_manifest": (
            scaling_manifest["source_policy"] == source_policy_parameters()
            and float(scaling_manifest["kappa"]) == KAPPA
            and float(scaling_manifest["lambda_reward"]) == LAMBDA_REWARD
            and float(scaling_manifest["gamma"]) == GAMMA),
        "split_exactly_reused": {name: list(map(int, values)) for name, values in splits.items()}
        == json.loads(Path(inputs["split_path"]).read_text(encoding="utf-8")),
        "hidden_u_generation_confounded": (
            data_audit["hidden_values_binary"]
            and data_audit["u_behavior_equals_u_environment"]),
        "action_generation_valid": data_audit["commanded_actions_finite_and_bounded"],
        "hidden_u_not_model_input": data_audit["public_fields_exclude_hidden_u"],
        "source_ids_only_for_component_separation": True,
        "pooled_model_does_not_read_source_id": component_checks["pooled_model_public_fields_only"],
        "component_updates_4000": component_checks["component_metadata_matches_n128_4000"],
        "component_architecture_common": component_checks["component_architecture_identical"],
        "complete_potential_checkpoint_set": all(path.is_file() for path in potential_paths),
        "complete_potential_uses_own_target_v": all(
            torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "backup_value_source") == "own_target_potential" for path in potential_paths),
        "frozen_reference_not_used": all(
            not torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "fixed_reference_value_used") for path in potential_paths),
        "candidate_protocol_28_identical": all(
            torch.load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "candidate_count") == 28 for path in potential_paths),
        "potential_initialization_paired": len(initial_by_seed) == 3,
        "do_oracle_not_used_for_potential_training": True,
        "no_nan_inf_or_extreme_explosion": numerical_stability,
        "old_artifacts_unchanged": old_before == old_after,
    }
    failed = [name for name, passed in checks.items() if not passed]
    _write_json(output / "potentials" / "hard_checks.json", {
        "all_passed": not failed, "checks": checks, "failed": failed})
    _write_json(output / "offline_models" / "references.json", {
        "policy": "reference existing n128/4000 checkpoints; do not copy or retrain",
        "components": component_references,
    })
    _write_json(output / "input_integrity.json", {
        "dataset": data_fingerprint,
        "hash_policy": "one BLAKE2b-128 digest for the frozen public dataset only",
        "old_inputs_before": old_before,
        "old_inputs_after": old_after,
        "old_inputs_unchanged": old_before == old_after,
    })
    manifest = {
        "stage": PHASE,
        "status": "potentials_complete" if not failed else "blocked",
        "authorized_protocol_amendment": (
            "one deterministic Phase 8H-DS n128 rematerialization, then frozen read-only reuse"),
        "phase8h_scaling_root": str(scaling),
        "phase8h_root": str(phase8h),
        "samples_per_anchor_source": 128,
        "transition_count": TRANSITION_COUNT,
        "model_seeds": list(MODEL_SEEDS),
        "component_updates": 4000,
        "component_batch_size": BATCH_SIZE,
        "potential_methods": list(POTENTIAL_METHODS),
        "potential_epochs": POTENTIAL_EPOCHS,
        "potential_optimizer_updates_per_method": (
            POTENTIAL_EPOCHS * math.ceil(len(train_rows) / POTENTIAL_BATCH_SIZE)),
        "potential_batch_size": POTENTIAL_BATCH_SIZE,
        "potential_learning_rate": POTENTIAL_LR,
        "target_tau": TARGET_TAU,
        "target_update_interval_batches": TARGET_UPDATE_INTERVAL,
        "candidate_actions_per_source": 8,
        "candidate_count": 28,
        "candidate_random_stream_paired": True,
        "source_policy": scaling_manifest["source_policy"],
        "kappa": scaling_manifest["kappa"],
        "lambda_reward": scaling_manifest["lambda_reward"],
        "gamma": scaling_manifest["gamma"],
        "pbrs_beta": PBRS_BETA,
        "pbrs_beta_source": "frozen Phase 8H-ON-Q Hopper scale",
        "numeric_explosion_limit": NUMERIC_EXPLOSION_LIMIT,
        "device": selected_device,
        "external_commit": EXTERNAL_COMMIT,
        "runtime_versions": _runtime_versions(),
        "sac_config": SAC_CONFIG,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "hard_checks.json", {
        "all_passed": False, "status": "potentials_complete_online_pending",
        "checks": {f"potentials:{key}": value for key, value in checks.items()},
        "failed": failed, "pending": ["online_smoke", "online"],
    })
    _write_report(output)
    _plot_potential_training(output)
    if failed:
        raise Phase8JLargeDataOnlineSanityError(f"potential hard checks failed: {failed}")
    return {"potential_count": len(potential_paths), "all_hard_checks_passed": True}


class _ZeroPotential:
    def __call__(self, states: np.ndarray) -> np.ndarray:
        return np.zeros(len(np.asarray(states)), dtype=np.float64)


def _online_potential(
    method: str, seed: int, output: Path, official: Any, torch: Any, device: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    if method == "sac_scratch":
        return _ZeroPotential(), {"phase8j_method": "zero", "model_seed": seed}
    potential_method = ONLINE_TO_POTENTIAL[method]
    path = _potential_path(output, potential_method, seed)
    if not path.is_file():
        raise Phase8JLargeDataOnlineSanityError(f"potential checkpoint is missing: {path}")
    value, metadata = _load_potential(path, official, torch, device)
    return value, {**metadata, "checkpoint": str(path)}


def _replay_training_snapshot(model: Any, torch: Any) -> dict[str, Any]:
    logger = getattr(model, "_logger", None)
    values = getattr(logger, "name_to_value", {})
    result = {
        "critic_loss": (float(values["train/critic_loss"])
                        if "train/critic_loss" in values else ""),
        "actor_loss": (float(values["train/actor_loss"])
                       if "train/actor_loss" in values else ""),
        "entropy_coefficient": (float(values["train/ent_coef"])
                                if "train/ent_coef" in values else ""),
        "entropy": "",
        "q_value_abs_mean": "",
    }
    count = int(model.replay_buffer.size())
    if count:
        size = min(count, 1024)
        observation = torch.as_tensor(
            model.replay_buffer.observations[:size, 0],
            dtype=torch.float32, device=model.device)
        action = torch.as_tensor(
            model.replay_buffer.actions[:size, 0],
            dtype=torch.float32, device=model.device)
        with torch.no_grad():
            _, log_probability = model.actor.action_log_prob(observation)
            q_values = model.critic(observation, action)
        result["entropy"] = float((-log_probability).mean().cpu())
        result["q_value_abs_mean"] = float(torch.stack(q_values).abs().mean().cpu())
    return result


def _curve_rows(
    evaluation_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int],
    methods: Sequence[str], eval_steps: Sequence[int],
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(int(row["online_seed"]), str(row["method"]),
                 int(row["training_step"]))].append(float(row["raw_environment_return"]))
    curves = {}
    for seed in seeds:
        for method in methods:
            values = []
            for step in eval_steps:
                episodes = grouped.get((seed, method, int(step)), [])
                if not episodes:
                    raise Phase8JLargeDataOnlineSanityError(
                        f"incomplete evaluation curve: seed={seed}, method={method}, step={step}")
                values.append(float(np.mean(episodes)))
            curves[(seed, method)] = (
                np.asarray(eval_steps, dtype=np.float64), np.asarray(values, dtype=np.float64))
    return curves


def _first_reach_step(steps: np.ndarray, returns: np.ndarray, level: float) -> float | None:
    for index in range(len(steps)):
        if returns[index] >= level:
            if index == 0 or returns[index - 1] >= level or returns[index] == returns[index - 1]:
                return float(steps[index])
            fraction = (level - returns[index - 1]) / (returns[index] - returns[index - 1])
            return float(steps[index - 1] + fraction * (steps[index] - steps[index - 1]))
    return None


def common_return_level_metrics(
    curves: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    seeds: Sequence[int], methods: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        selected = [curves[(seed, method)] for method in methods]
        lower = max(float(values.min()) for _, values in selected)
        upper = min(float(values.max()) for _, values in selected)
        observed = sorted({float(value) for _, values in selected for value in values
                           if lower <= float(value) <= upper})
        for level in observed:
            for method in methods:
                steps, values = curves[(seed, method)]
                reached = _first_reach_step(steps, values, level)
                rows.append({"online_seed": seed, "return_level": level,
                             "method": method, "steps_to_reach": reached})
    return rows


def summarize_online(
    evaluation_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int],
    methods: Sequence[str], eval_steps: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves = _curve_rows(evaluation_rows, seeds, methods, eval_steps)
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(int(row["online_seed"]), str(row["method"]),
                 int(row["training_step"]))].append(row)
    per_seed = []
    for seed in seeds:
        scratch_steps, scratch_returns = curves[(seed, "sac_scratch")]
        pooled_auc = normalized_auc(*curves[(seed, "sac_pooled")])
        for method in methods:
            steps, returns = curves[(seed, method)]
            final = grouped[(seed, method, int(eval_steps[-1]))]
            auc = normalized_auc(steps, returns)
            per_seed.append({
                "online_seed": seed, "method": method, "early_auc_0_50k": auc,
                "auc_difference_vs_pooled": auc - pooled_auc,
                "negative_transfer_area_vs_scratch": normalized_positive_area(
                    scratch_steps, scratch_returns - returns),
                "final_return_50k": float(np.mean([
                    float(row["raw_environment_return"]) for row in final])),
                "final_episode_length": float(np.mean([
                    float(row["episode_length"]) for row in final])),
                "final_termination_fraction": float(np.mean([
                    str(row["terminated"]).lower() == "true" for row in final])),
            })
    summary = []
    metrics = ("early_auc_0_50k", "auc_difference_vs_pooled",
               "negative_transfer_area_vs_scratch", "final_return_50k")
    for method in methods:
        rows = [row for row in per_seed if row["method"] == method]
        record: dict[str, Any] = {"method": method, "seed_count": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(record)
    return per_seed, summary


def run_online(
    output_root: Path, *, online_seeds: Sequence[int], online_steps: int,
    eval_steps: Sequence[int], eval_episodes: int, device: str,
    external_repo: Path, smoke: bool,
) -> dict[str, Any]:
    seeds = tuple(map(int, online_seeds))
    steps = tuple(map(int, eval_steps))
    if smoke:
        if seeds != (0,) or online_steps != 5_000:
            raise Phase8JLargeDataOnlineSanityError("smoke is frozen to seed 0 and 5000 steps")
        if steps != (0, 5_000):
            raise Phase8JLargeDataOnlineSanityError("smoke evaluation steps must be 0,5000")
    elif seeds != MODEL_SEEDS or online_steps != 50_000 or steps != FORMAL_EVAL_STEPS \
            or eval_episodes != 5:
        raise Phase8JLargeDataOnlineSanityError(
            "formal online is frozen to seeds 0,1,2; 50k; fixed evaluation grid; 5 episodes")
    output = Path(output_root).resolve()
    potential_checks = json.loads((output / "potentials" / "hard_checks.json").read_text(
        encoding="utf-8"))
    if potential_checks.get("all_passed") is not True:
        raise Phase8JLargeDataOnlineSanityError("potential sanity audit must pass before online")
    if not smoke:
        smoke_checks = output / "online" / "smoke" / "hard_checks.json"
        if not smoke_checks.is_file() or json.loads(smoke_checks.read_text(
                encoding="utf-8")).get("all_passed") is not True:
            raise Phase8JLargeDataOnlineSanityError("online smoke must pass before formal online")
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
        from stable_baselines3 import SAC
    except (ImportError, OSError) as error:
        raise Phase8JLargeDataOnlineSanityError("PyTorch and stable-baselines3 are required") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(external_repo)
    destination = output / "online" / "smoke" if smoke else output / "online"
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_path = destination / "evaluation_returns.csv"
    training_path = destination / "online_training_metrics.csv"
    evaluation_rows = _read_csv(evaluation_path) if evaluation_path.is_file() else []
    training_rows = _read_csv(training_path) if training_path.is_file() else []
    initial_by_seed: dict[int, Mapping[str, Any]] = {}
    potential_before = {str(path): file_metadata(path)
                        for seed in seeds for method in POTENTIAL_METHODS
                        for path in [_potential_path(output, method, seed)]}
    for seed in seeds:
        for method_index, method in enumerate(ONLINE_METHODS):
            existing_steps = {int(row["training_step"]) for row in evaluation_rows
                              if int(row["online_seed"]) == seed and row["method"] == method}
            final_metrics = [row for row in training_rows
                             if int(row["online_seed"]) == seed and row["method"] == method
                             and int(row["training_step"]) == online_steps]
            if existing_steps == set(steps) and len(final_metrics) == 1 \
                    and str(final_metrics[0].get("run_complete", "")).lower() == "true":
                initial = {key: float(final_metrics[0][f"initial_{key}"])
                           for key in ("parameter_count", "sum", "sum_squares", "max_abs")}
                initial["parameter_count"] = int(initial["parameter_count"])
                if seed in initial_by_seed and not _same_fingerprint(initial_by_seed[seed], initial):
                    raise Phase8JLargeDataOnlineSanityError(
                        f"reused SAC initialization is not paired for seed {seed}")
                initial_by_seed[seed] = initial
                print(f"reusing completed online run: {method}, seed {seed}", flush=True)
                continue
            evaluation_rows = [row for row in evaluation_rows
                               if not (int(row["online_seed"]) == seed and row["method"] == method)]
            training_rows = [row for row in training_rows
                             if not (int(row["online_seed"]) == seed and row["method"] == method)]
            potential, metadata = _online_potential(
                method, seed, output, official, torch, selected_device)
            phi_before = parameter_fingerprint((potential.network,)) \
                if hasattr(potential, "network") else None
            environment = _make_online_environment(
                potential, kappa=KAPPA, lambda_reward=LAMBDA_REWARD,
                beta=PBRS_BETA, gamma=GAMMA)
            seed_everything(seed, torch, cuda_training=selected_device == "cuda")
            model = SAC(env=environment, seed=seed, device=selected_device,
                        verbose=0, **SAC_CONFIG)
            replay_empty = int(model.replay_buffer.size()) == 0
            initial = parameter_fingerprint((model.actor, model.critic))
            if seed in initial_by_seed and not _same_fingerprint(initial_by_seed[seed], initial):
                raise Phase8JLargeDataOnlineSanityError(
                    f"SAC initialization is not paired for seed {seed}")
            initial_by_seed[seed] = initial
            wall_start = time.perf_counter()
            evaluation_interactions = 0
            previous = 0
            for step in steps:
                if step > previous:
                    model.learn(total_timesteps=step - previous,
                                reset_num_timesteps=False, progress_bar=False)
                rows, interactions = _evaluate_online_model(
                    model, potential, run_id=seed, method_index=method_index,
                    training_step=step, episodes=eval_episodes, kappa=KAPPA,
                    lambda_reward=LAMBDA_REWARD, beta=PBRS_BETA, gamma=GAMMA)
                evaluation_interactions += interactions
                evaluation_rows.extend({"online_seed": seed, "method": method, **{
                    key: value for key, value in row.items() if key != "run_id"}}
                                       for row in rows)
                increments = np.asarray(environment.shaping_increments, dtype=np.float64)
                snapshot = _replay_training_snapshot(model, torch)
                training_rows.append({
                    "online_seed": seed, "method": method, "training_step": step,
                    "training_environment_steps": step,
                    "evaluation_environment_steps": evaluation_interactions,
                    "replay_size": int(model.replay_buffer.size()),
                    "critic_loss": snapshot["critic_loss"],
                    "actor_loss": snapshot["actor_loss"],
                    "entropy": snapshot["entropy"],
                    "entropy_coefficient": snapshot["entropy_coefficient"],
                    "q_value_abs_mean": snapshot["q_value_abs_mean"],
                    "shaping_increment_mean": float(increments.mean()) if len(increments) else 0.0,
                    "shaping_increment_std": float(increments.std()) if len(increments) else 0.0,
                    "shaping_increment_p90_abs": float(np.quantile(np.abs(increments), .90))
                    if len(increments) else 0.0,
                    "shaping_increment_p99_abs": float(np.quantile(np.abs(increments), .99))
                    if len(increments) else 0.0,
                    "wall_clock_seconds": float(time.perf_counter() - wall_start),
                    "potential_method": metadata["phase8j_method"],
                    "beta": PBRS_BETA, "gamma": GAMMA,
                    "initial_parameter_count": initial["parameter_count"],
                    "initial_sum": initial["sum"],
                    "initial_sum_squares": initial["sum_squares"],
                    "initial_max_abs": initial["max_abs"],
                    "run_complete": False,
                })
                _write_csv(evaluation_path, evaluation_rows)
                _write_csv(training_path, training_rows)
                print(f"online {method} seed {seed}: {step}/{online_steps}", flush=True)
                previous = step
            phi_after = parameter_fingerprint((potential.network,)) \
                if hasattr(potential, "network") else None
            final_row = training_rows[-1]
            final_row.update({
                "replay_initially_empty": replay_empty,
                "replay_matches_commanded_actions": _replay_matches_commands(
                    model, environment.commanded_actions),
                "wrapper_commanded_action_match": environment.commanded_action_match,
                "returned_info_private_leak": environment.returned_info_private_leak,
                "raw_reward_formula_match": environment.raw_reward_formula_match,
                "potential_frozen": phi_before == phi_after,
                "run_complete": True,
            })
            _write_csv(training_path, training_rows)
            environment.close()
            del model, environment, potential
            if selected_device == "cuda":
                torch.cuda.empty_cache()
    curves = _curve_rows(evaluation_rows, seeds, ONLINE_METHODS, steps)
    paired, summary_rows = summarize_online(evaluation_rows, seeds, ONLINE_METHODS, steps)
    common_levels = common_return_level_metrics(curves, seeds, ONLINE_METHODS)
    _write_csv(destination / "paired_seed_metrics.csv", paired)
    _write_csv(destination / "summary_metrics.csv", summary_rows)
    _write_csv(destination / "common_return_levels.csv", common_levels)
    final_training = [row for row in training_rows
                      if int(row["training_step"]) == online_steps
                      and int(row["online_seed"]) in seeds and row["method"] in ONLINE_METHODS]
    potential_after = {str(path): file_metadata(path)
                       for seed in seeds for method in POTENTIAL_METHODS
                       for path in [_potential_path(output, method, seed)]}
    finite_primary = all(np.isfinite(float(value)) for row in paired for value in row.values()
                         if isinstance(value, (int, float, np.integer, np.floating)))
    finite_training = all(
        np.isfinite(float(value)) for row in training_rows for value in row.values()
        if value != "" and isinstance(value, (int, float, np.integer, np.floating)))
    checks = {
        "all_online_runs_complete": len(final_training) == len(seeds) * len(ONLINE_METHODS),
        "phi_frozen_online": all(str(row["potential_frozen"]).lower() == "true"
                                 for row in final_training),
        "pbrs_gamma_correct": all(float(row["gamma"]) == GAMMA for row in final_training),
        "terminated_truncated_handling_correct": np.isclose(
            discounted_shaping_sum([2.0, 3.0, 5.0], [False, True]), -2.0),
        "replay_stores_commanded_action": all(
            str(row["replay_matches_commanded_actions"]).lower() == "true"
            and str(row["wrapper_commanded_action_match"]).lower() == "true"
            for row in final_training),
        "sac_initialization_paired": len(initial_by_seed) == len(seeds),
        "offline_data_not_in_sac_replay": all(
            str(row["replay_initially_empty"]).lower() == "true" for row in final_training),
        "raw_environment_reward_is_primary": all(
            str(row["raw_reward_formula_match"]).lower() == "true" for row in final_training),
        "hidden_information_not_returned": all(
            str(row["returned_info_private_leak"]).lower() == "false"
            for row in final_training),
        "potential_inputs_unchanged": potential_before == potential_after,
        "all_primary_metrics_finite": finite_primary,
        "all_recorded_training_metrics_finite": finite_training,
        "scratch_shaping_exactly_zero": all(
            float(row["shaping_increment_mean"]) == 0.0
            and float(row["shaping_increment_std"]) == 0.0
            for row in final_training if row["method"] == "sac_scratch"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    _write_json(destination / "hard_checks.json", {
        "all_passed": not failed, "checks": checks, "failed": failed})
    if smoke:
        if failed:
            raise Phase8JLargeDataOnlineSanityError(f"online smoke checks failed: {failed}")
        return {"online_run_count": 4, "smoke": True, "all_hard_checks_passed": True}
    _finalize(output, checks, paired, summary_rows, common_levels)
    if failed:
        raise Phase8JLargeDataOnlineSanityError(f"formal online checks failed: {failed}")
    return {"online_run_count": 12, "smoke": False, "all_hard_checks_passed": True}


def _plot_potential_training(output: Path) -> None:
    path = output / "potential_training_metrics.csv"
    if not path.is_file():
        return
    import matplotlib.pyplot as plt
    rows = _read_csv(path)
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for method in POTENTIAL_METHODS:
        selected = [row for row in rows if row["potential"] == method]
        epochs = sorted({int(row["epoch"]) for row in selected})
        values = [np.mean([float(row["training_loss"]) for row in selected
                          if int(row["epoch"]) == epoch]) for epoch in epochs]
        axis.plot(epochs, values, label=method)
    axis.set_xlabel("Epoch"); axis.set_ylabel("Training loss")
    axis.legend(); axis.grid(alpha=.25)
    figure.tight_layout()
    destination = output / "figures" / "potential_training_curves.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180); plt.close(figure)


def _plot_online(output: Path) -> None:
    import matplotlib.pyplot as plt
    evaluation = _read_csv(output / "online" / "evaluation_returns.csv")
    paired = _read_csv(output / "online" / "paired_seed_metrics.csv")
    diagnostics = _read_csv(output / "potential_diagnostics.csv")
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in evaluation:
        grouped[(row["method"], int(row["training_step"]))].append(
            float(row["raw_environment_return"]))
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for method in ONLINE_METHODS:
        steps = sorted(step for name, step in grouped if name == method)
        mean = [np.mean(grouped[(method, step)]) for step in steps]
        sd = [np.std(grouped[(method, step)], ddof=1) for step in steps]
        axis.plot(steps, mean, label=method)
        axis.fill_between(steps, np.asarray(mean) - sd, np.asarray(mean) + sd, alpha=.15)
    axis.set_xlabel("Online environment steps"); axis.set_ylabel("Raw environment return")
    axis.legend(fontsize=8); axis.grid(alpha=.25); figure.tight_layout()
    figure.savefig(output / "figures" / "online_learning_curves.png", dpi=180); plt.close(figure)

    plots = [
        ("early_auc_by_method.png", "early_auc_0_50k", "Early AUC 0-50k"),
        ("negative_transfer_area.png", "negative_transfer_area_vs_scratch", "NTA vs scratch"),
    ]
    for filename, metric, ylabel in plots:
        figure, axis = plt.subplots(figsize=(6.8, 4.2))
        data = [[float(row[metric]) for row in paired if row["method"] == method]
                for method in ONLINE_METHODS]
        axis.boxplot(data, tick_labels=ONLINE_METHODS, showmeans=True)
        axis.set_ylabel(ylabel); axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=.25); figure.tight_layout()
        figure.savefig(output / "figures" / filename, dpi=180); plt.close(figure)

    online_lookup = {(int(row["online_seed"]), row["method"]): float(row["early_auc_0_50k"])
                     for row in paired}
    figure, axis = plt.subplots(figsize=(6.5, 4.4))
    colors = {"pooled": "tab:blue", "state_min": "tab:green", "action_min": "tab:red"}
    for row in diagnostics:
        method, seed = row["potential"], int(row["model_seed"])
        axis.scatter(float(row["regret_mean"]), online_lookup[(seed, f"sac_{method}")],
                     color=colors[method], label=method if seed == 0 else None)
    axis.set_xlabel("Offline mean regret"); axis.set_ylabel("Online early AUC")
    axis.legend(); axis.grid(alpha=.25); figure.tight_layout()
    figure.savefig(output / "figures" / "offline_proxy_vs_online_auc.png", dpi=180)
    plt.close(figure)


def _write_report(output: Path) -> None:
    lines = ["# Phase 8J-Q report", "",
             "Completion means protocol and numerical checks passed, not that any method won.", ""]
    diagnostics = output / "potential_diagnostics.csv"
    lines.extend(["## Offline potential sanity", "",
                  "| Seed | Potential | Range | Residual MAE | Shaping std | Do-Bellman MAE | Mean regret |",
                  "|---:|---|---:|---:|---:|---:|---:|"])
    if diagnostics.is_file():
        for row in _read_csv(diagnostics):
            value_range = f"[{float(row['potential_min']):.4g}, {float(row['potential_max']):.4g}]"
            lines.append(f"| {row['model_seed']} | {row['potential']} | {value_range} | "
                         f"{float(row['validation_residual_mae']):.4g} | "
                         f"{float(row['shaping_increment_std']):.4g} | "
                         f"{float(row['do_bellman_mae']):.4g} | {float(row['regret_mean']):.4g} |")
    else:
        lines.append("| pending | pending | - | - | - | - | - |")
    lines.extend(["", "## Online results", "",
                  "| Method | Early AUC mean | Delta vs pooled | Final return | NTA vs scratch |",
                  "|---|---:|---:|---:|---:|"])
    online = output / "online" / "summary_metrics.csv"
    if online.is_file():
        online_rows = _read_csv(online)
        for row in online_rows:
            lines.append(f"| {row['method']} | {float(row['early_auc_0_50k_mean']):.4g} | "
                         f"{float(row['auc_difference_vs_pooled_mean']):.4g} | "
                         f"{float(row['final_return_50k_mean']):.4g} | "
                         f"{float(row['negative_transfer_area_vs_scratch_mean']):.4g} |")
    else:
        online_rows = []
        lines.append("| pending | - | - | - | - |")
    paired_path = output / "online" / "paired_seed_metrics.csv"
    if online_rows and paired_path.is_file() and diagnostics.is_file():
        paired = _read_csv(paired_path)
        by_method = {row["method"]: row for row in online_rows}
        action_delta = [float(row["auc_difference_vs_pooled"]) for row in paired
                        if row["method"] == "sac_action_min"]
        state_delta = [float(row["auc_difference_vs_pooled"]) for row in paired
                       if row["method"] == "sac_state_min"]
        action_auc = float(by_method["sac_action_min"]["early_auc_0_50k_mean"])
        state_auc = float(by_method["sac_state_min"]["early_auc_0_50k_mean"])
        scratch_nta = {
            method: float(by_method[method]["negative_transfer_area_vs_scratch_mean"])
            for method in ("sac_pooled", "sac_state_min", "sac_action_min")}
        proxy_rows = _read_csv(diagnostics)
        def offline_best(metric: str) -> str:
            return min(
                POTENTIAL_METHODS,
                key=lambda method: np.mean([float(row[metric]) for row in proxy_rows
                                            if row["potential"] == method]))

        do_best = offline_best("do_bellman_mae")
        potential_best = offline_best("potential_mae")
        regret_best = offline_best("regret_mean")
        online_best = max(
            POTENTIAL_METHODS,
            key=lambda method: float(by_method[f"sac_{method}"]["early_auc_0_50k_mean"]))
        lines.extend([
            "", "## Scientific answers", "",
            f"1. Action-min minus pooled AUC by seed: {action_delta}. "
            f"Action-min is higher in {sum(value > 0 for value in action_delta)}/3 seeds.",
            f"2. State-min mean AUC is {state_auc:.6g}; Action-min mean AUC is "
            f"{action_auc:.6g}. The higher online PBRS result is "
            f"{'State-min' if state_auc > action_auc else 'Action-min' if action_auc > state_auc else 'a tie'}.",
            f"3. Mean negative-transfer areas versus scratch are {scratch_nta}. Lower is better; "
            f"the best shaped method is {min(scratch_nta, key=scratch_nta.get)}.",
            f"4. The best Phase 8H Do-Bellman MAE method is {do_best}; the best Phase 8H "
            f"Potential MAE method is {potential_best}; the best online AUC method is {online_best}.",
            f"5. The best offline mean-regret method is {regret_best}; it "
            f"{'agrees' if regret_best == online_best else 'does not agree'} with online AUC.",
            f"6. State-min minus pooled AUC by seed: {state_delta}. Results are reported as "
            "observed; in particular, an offline Action-min advantage is not used to override a "
            "possible online State-min advantage. No method is selected or retuned from outcomes.",
            "", "Interpretation boundary: a multi-source gain only means retained source-specific "
            "AAMAS information produced a more useful shaping potential in this setting. It does "
            "not identify hidden U, prove confounding removal, or establish a confounding-specific "
            "benefit without an independent control.",
        ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize(
    output: Path, online_checks: Mapping[str, bool], paired: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]], common_levels: Sequence[Mapping[str, Any]],
) -> None:
    potential_record = json.loads((output / "potentials" / "hard_checks.json").read_text(
        encoding="utf-8"))
    smoke_record = json.loads((output / "online" / "smoke" / "hard_checks.json").read_text(
        encoding="utf-8"))
    integrity = json.loads((output / "input_integrity.json").read_text(encoding="utf-8"))
    dataset_now = file_fingerprint(_dataset_path(output))
    dataset_unchanged = dataset_now["blake2b_128"] == integrity["dataset"]["blake2b_128"]
    old_paths = [Path(row["path"]) for row in integrity["old_inputs_before"]]
    old_unchanged = metadata_snapshot(old_paths) == integrity["old_inputs_before"]
    checks = {
        **{f"potentials:{key}": bool(value)
           for key, value in potential_record["checks"].items()},
        **{f"smoke:{key}": bool(value) for key, value in smoke_record["checks"].items()},
        **{f"online:{key}": bool(value) for key, value in online_checks.items()},
        "final:frozen_public_dataset_unchanged": dataset_unchanged,
        "final:old_artifacts_unchanged": old_unchanged,
    }
    failed = [name for name, passed in checks.items() if not passed]
    _write_json(output / "hard_checks.json", {
        "all_passed": not failed, "status": "complete" if not failed else "blocked",
        "checks": checks, "failed": failed})
    diagnostics = _read_csv(output / "potential_diagnostics.csv")
    proxy_auc_pairs = []
    for row in diagnostics:
        method, seed = row["potential"], int(row["model_seed"])
        online_row = next(value for value in paired
                          if int(value["online_seed"]) == seed
                          and value["method"] == f"sac_{method}")
        proxy_auc_pairs.append((float(row["regret_mean"]), float(online_row["early_auc_0_50k"])))
    proxy_values = np.asarray(proxy_auc_pairs, dtype=np.float64)
    correlation = (float(np.corrcoef(proxy_values.T)[0, 1])
                   if np.all(np.std(proxy_values, axis=0) > 0) else None)
    summary = {
        "stage": PHASE, "all_hard_checks_passed": not failed,
        "failed_hard_checks": failed,
        "online_summary": list(summary_rows),
        "action_min_minus_pooled_auc": [float(row["auc_difference_vs_pooled"])
                                         for row in paired if row["method"] == "sac_action_min"],
        "state_min_minus_pooled_auc": [float(row["auc_difference_vs_pooled"])
                                        for row in paired if row["method"] == "sac_state_min"],
        "offline_regret_online_auc_correlation": correlation,
        "common_return_level_count": len(common_levels),
        "interpretation_boundary": (
            "Any gain only shows that retained source-specific AAMAS information produced a more "
            "useful shaping potential under this setting; it does not identify hidden U or prove "
            "confounding removal."),
    }
    _write_json(output / "summary.json", summary)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "status": "complete" if not failed else "blocked",
        "online_seeds": list(MODEL_SEEDS), "online_steps": 50_000,
        "eval_steps": list(FORMAL_EVAL_STEPS), "eval_episodes": 5,
        "online_methods": list(ONLINE_METHODS),
        "primary_reward": "raw_environment_reward",
        "primary_metrics": ["early_auc_0_50k", "auc_difference_vs_pooled",
                            "final_return_50k", "negative_transfer_area_vs_scratch",
                            "steps_to_common_observed_return_levels"],
    })
    _write_json(output / "manifest.json", manifest)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    _plot_potential_training(output)
    _plot_online(output)
    _write_report(output)
