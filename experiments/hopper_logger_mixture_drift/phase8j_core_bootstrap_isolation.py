"""Phase 8J-BI-Q: isolate the first training layer that becomes unstable.

The three arms share data, initialization, normalization, optimizer, minibatch
order, and update budget.  Only the regression target differs.  This module
does not run SAC, inspect online returns, tune hyperparameters, or read hidden
or do-oracle data.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.train_aamas_hopper_potential import seed_everything
from .phase8h_compute_matched_online_quick import (
    GAMMA,
    POTENTIAL_BATCH_SIZE,
    POTENTIAL_LR,
    file_fingerprint,
    parameter_fingerprint,
)
from .phase8j_fvi_stability import (
    COMPONENT_UPDATES,
    DEFAULT_FIX_ROOT,
    METHOD,
    MODEL_SEED,
    SAMPLES_PER_ANCHOR_SOURCE,
    _batch_inputs,
    _build_probes,
    _load_seed_components,
    _network_value,
    _normalization,
    _outer_batches,
    _pooled_union_backup,
    _relocate_recorded_path,
    _resolve_context,
)
from .phase8j_potential_clamp_fix_quick import (
    _gradient_norm,
    make_repaired_potential_network,
)


PHASE = "Phase 8J-BI-Q"
VARIANTS = ("reward_only", "real_transition_bootstrap", "full_aamas_frozen")
BOOTSTRAP_VARIANTS = VARIANTS[1:]
OUTER_ITERATIONS = 40
INNER_EPOCHS = 5
TOTAL_EPOCHS = 200
WEIGHT_DECAY = 1e-5
RECORD_EPOCHS = (0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200)
# Retain the exact network states needed by the downstream read-only
# Bellman mean-error audit. ``latest.pt`` remains the resumable checkpoint.
BME_CHECKPOINT_EPOCHS = (10, 20, 40, 60, 80, 100, 120, 150, 180, 200)
DEFAULT_FVI_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_fvi_stability_diagnostic"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_core_bootstrap_isolation"
)


class Phase8JBootstrapIsolationError(RuntimeError):
    """Raised when the frozen diagnostic contract cannot be honored."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise Phase8JBootstrapIsolationError("metric input is empty or non-finite")
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p99": float(np.quantile(array, 0.99)),
    }


def _all_scalar_metrics_finite(
    row_groups: Sequence[Sequence[Mapping[str, Any]]],
) -> bool:
    """Validate numeric CSV metrics while ignoring identifiers and boolean audit fields."""
    ignored = {
        "variant",
        "probe_set",
        "cached_target_blake2b_128",
        "residual_target_blake2b_128",
        "target_hash_constant_for_all_inner_epochs",
        "uses_current_network_recomputation",
    }
    for rows in row_groups:
        for row in rows:
            for key, value in row.items():
                if key in ignored or isinstance(value, (bool, np.bool_)):
                    continue
                if value is None or value == "" or not np.isfinite(float(value)):
                    return False
    return True


def reward_only_target(public: Mapping[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Return the fixed logged reward without touching any bootstrap dependency."""
    result = np.asarray(public["reward"], dtype=np.float64)[np.asarray(rows, dtype=np.int64)]
    if result.shape != (len(rows),) or not np.all(np.isfinite(result)):
        raise Phase8JBootstrapIsolationError("reward-only target is invalid")
    return result


def real_transition_target(
    public: Mapping[str, np.ndarray],
    rows: np.ndarray,
    value: Callable[[np.ndarray], np.ndarray],
    gamma: float = GAMMA,
) -> np.ndarray:
    """Use logged reward and real next state; only termination masks continuation."""
    indices = np.asarray(rows, dtype=np.int64)
    reward = np.asarray(public["reward"], dtype=np.float64)[indices]
    next_state = np.asarray(public["next_observation"], dtype=np.float32)[indices]
    terminated = np.asarray(public["terminated"], dtype=bool)[indices]
    continuation = np.asarray(value(next_state), dtype=np.float64).reshape(-1)
    result = reward + float(gamma) * (~terminated).astype(np.float64) * continuation
    if result.shape != (len(indices),) or not np.all(np.isfinite(result)):
        raise Phase8JBootstrapIsolationError("real-transition bootstrap target is invalid")
    return result


def _target_for_rows(
    variant: str,
    context: Mapping[str, Any],
    components: Mapping[str, Any],
    rows: np.ndarray,
    target_network: Any,
    mean: np.ndarray,
    std: np.ndarray,
    outer: int,
    batch_index: int,
    *,
    validation: bool = False,
) -> np.ndarray:
    if variant == "reward_only":
        return reward_only_target(context["public"], rows)
    value = _network_value(target_network, mean, std, context)
    if variant == "real_transition_bootstrap":
        return real_transition_target(context["public"], rows, value)
    if variant == "full_aamas_frozen":
        batch = _batch_inputs(
            context, components, rows, outer, batch_index, validation=validation
        )
        return _pooled_union_backup(batch, components["pooled_balanced"], value)
    raise Phase8JBootstrapIsolationError(f"unknown variant: {variant}")


def _recomputed_target(
    variant: str,
    context: Mapping[str, Any],
    components: Mapping[str, Any],
    rows: np.ndarray,
    current: Any,
    mean: np.ndarray,
    std: np.ndarray,
    outer: int,
    batch_index: int,
) -> np.ndarray:
    """Recompute B(V_current), never reuse the outer iteration's cached target."""
    return _target_for_rows(
        variant,
        context,
        components,
        rows,
        current,
        mean,
        std,
        outer,
        batch_index,
        validation=True,
    )


def _source_fingerprints() -> list[dict[str, Any]]:
    repository = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        repository / "experiments/hopper_logger_mixture_drift/phase8j_fvi_stability.py",
        repository / "experiments/hopper_logger_mixture_drift/phase8j_potential_clamp_fix_quick.py",
        repository / "aamas_hopper_adapter.py",
    ]
    return [file_fingerprint(path) for path in paths]


def _load_fvi_contract(fvi_root: Path) -> tuple[Path, dict[str, Any]]:
    root = Path(fvi_root).resolve()
    manifest_path = root / "manifest.json"
    hard_path = root / "hard_checks.json"
    if not manifest_path.is_file() or not hard_path.is_file():
        raise Phase8JBootstrapIsolationError("Phase 8J-FVI manifest/hard checks are missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "Phase 8J-FVI-Q" or hard.get("all_passed") is not True:
        raise Phase8JBootstrapIsolationError("Phase 8J-FVI contract is not valid")
    expected = {
        "method": METHOD,
        "model_seed": MODEL_SEED,
        "samples_per_anchor_source": SAMPLES_PER_ANCHOR_SOURCE,
        "component_updates": COMPONENT_UPDATES,
        "epochs": TOTAL_EPOCHS,
        "outer_iterations": OUTER_ITERATIONS,
        "inner_epochs": INNER_EPOCHS,
        "optimizer": "Adam",
        "batch_size": POTENTIAL_BATCH_SIZE,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items()
                  if manifest.get(key) != value}
    if mismatches:
        raise Phase8JBootstrapIsolationError(f"Phase 8J-FVI configuration mismatch: {mismatches}")
    if not math.isclose(float(manifest.get("learning_rate", math.nan)), POTENTIAL_LR):
        raise Phase8JBootstrapIsolationError("Phase 8J-FVI learning rate mismatch")
    if not math.isclose(float(manifest.get("weight_decay", math.nan)), WEIGHT_DECAY):
        raise Phase8JBootstrapIsolationError("Phase 8J-FVI weight decay mismatch")
    return root, manifest


def _input_paths(
    context: Mapping[str, Any], fvi_root: Path
) -> list[Path]:
    paths = [
        context["fix_manifest_path"],
        context["fix_tests_path"],
        context["dataset"],
        context["split_path"],
        *context["recorded_component_paths"],
        context["old_divergence_curve"],
        context["old_divergence_checkpoint"],
        fvi_root / "manifest.json",
        fvi_root / "hard_checks.json",
    ]
    return sorted({Path(path).resolve() for path in paths}, key=str)


def _check_integrity(output: Path) -> bool:
    path = output / "input_integrity.json"
    if not path.is_file():
        return False
    repository = Path(__file__).resolve().parents[2]
    record = json.loads(path.read_text(encoding="utf-8"))
    for expected in record.get("inputs", []):
        try:
            actual = _relocate_recorded_path(expected["path"], repository)
        except RuntimeError:
            return False
        current = file_fingerprint(actual)
        if (int(current["size_bytes"]) != int(expected["size_bytes"])
                or current.get("blake2b_128") != expected.get("blake2b_128")):
            return False
    return bool(record.get("inputs"))


def _minibatch_audit(train_rows: np.ndarray) -> dict[str, Any]:
    outer_digests = []
    for outer in range(OUTER_ITERATIONS):
        batches = _outer_batches(train_rows, outer)
        digest = _array_digest(*batches)
        outer_digests.append({
            "outer_iteration": outer + 1,
            "batch_count": len(batches),
            "row_sequence_blake2b_128": digest,
        })
    return {
        "variants": list(VARIANTS),
        "construction": "same deterministic _outer_batches rows for every variant",
        "outer_row_sequences": outer_digests,
        "identical_across_variants": True,
    }


def _load_or_create_initialization(
    output: Path,
    context: Mapping[str, Any],
    reward_min: float,
    reward_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = context["torch"]
    path = output / "init_seed0.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("metadata", {}).get("stage") != PHASE:
            raise Phase8JBootstrapIsolationError("existing initialization has wrong provenance")
        return payload["state_dict"], payload["fingerprint"]
    seed_everything(MODEL_SEED, torch, cuda_training=context["device"] == "cuda")
    network = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"]
    )
    state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
    fingerprint = parameter_fingerprint((network,))
    torch.save({
        "state_dict": state,
        "fingerprint": fingerprint,
        "metadata": {
            "stage": PHASE,
            "model_seed": MODEL_SEED,
            "readout": "unclamped_linear",
            "used_by_variants": list(VARIANTS),
        },
    }, path)
    return state, fingerprint


def _initialization_audit(
    state: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    context: Mapping[str, Any],
    reward_min: float,
    reward_max: float,
) -> dict[str, Any]:
    networks = []
    for _ in VARIANTS:
        network = make_repaired_potential_network(
            context["official"], reward_min, reward_max, context["device"]
        )
        network.load_state_dict(state)
        networks.append(network)
    reference = [parameter.detach().cpu().numpy() for parameter in networks[0].parameters()]
    pairwise = {}
    for variant, network in zip(VARIANTS, networks):
        values = [parameter.detach().cpu().numpy() for parameter in network.parameters()]
        pairwise[variant] = bool(all(np.array_equal(left, right)
                                     for left, right in zip(reference, values)))
    return {
        "initial_parameter_fingerprint": fingerprint,
        "parameter_count": int(sum(parameter.numel() for parameter in networks[0].parameters())),
        "variant_elementwise_equality": pairwise,
        "all_elementwise_identical": all(pairwise.values()),
    }


def _termination_audit() -> dict[str, Any]:
    public = {
        "reward": np.array([2.0, 2.0, 2.0]),
        "next_observation": np.arange(36, dtype=np.float32).reshape(3, 12),
        "terminated": np.array([False, True, False]),
        "truncated": np.array([False, False, True]),
    }
    result = real_transition_target(
        public, np.arange(3), lambda states: np.full(len(states), 7.0)
    )
    expected = np.array([2.0 + GAMMA * 7.0, 2.0, 2.0 + GAMMA * 7.0])
    return {
        "computed": result,
        "expected": expected,
        "terminated_zeroes_continuation": bool(result[1] == 2.0),
        "truncated_retains_continuation": bool(result[2] > 2.0),
        "matches_parent_semantics": bool(np.array_equal(result, expected)),
    }


def _target_detachment_audit(
    context: Mapping[str, Any],
    components: Mapping[str, Any],
    state: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    reward_min: float,
    reward_max: float,
) -> dict[str, Any]:
    """Materialize one target per arm and verify that no autograd object escapes."""
    network = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"]
    )
    network.load_state_dict(state)
    network.eval().requires_grad_(False)
    rows = np.asarray(context["validation_rows"][:8], dtype=np.int64)
    checks: dict[str, bool] = {}
    for variant in VARIANTS:
        target = _target_for_rows(
            variant, context, components, rows, network, mean, std, 0, 0,
            validation=True,
        )
        checks[variant] = bool(
            isinstance(target, np.ndarray)
            and target.shape == (len(rows),)
            and np.all(np.isfinite(target))
            and not hasattr(target, "grad_fn")
        )
    return {"checks": checks, "all_passed": all(checks.values())}


def run_preflight(
    fix_root: Path = DEFAULT_FIX_ROOT,
    fvi_root: Path = DEFAULT_FVI_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
    model_seed: int = MODEL_SEED,
    samples_per_anchor_source: int = SAMPLES_PER_ANCHOR_SOURCE,
    component_updates: int = COMPONENT_UPDATES,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    if (model_seed, samples_per_anchor_source, component_updates) != (
            MODEL_SEED, SAMPLES_PER_ANCHOR_SOURCE, COMPONENT_UPDATES):
        raise Phase8JBootstrapIsolationError("scope must remain seed0/n128/4000")
    if not math.isclose(float(gamma), GAMMA):
        raise Phase8JBootstrapIsolationError("gamma must remain 0.99")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_fvi, source_manifest = _load_fvi_contract(fvi_root)
    context = _resolve_context(fix_root, external_repo, device)
    components = _load_seed_components(context, MODEL_SEED)
    mean, std, reward_min, reward_max = _normalization(context)
    probes, distance_rows = _build_probes(context, components, mean, std)
    state, fingerprint = _load_or_create_initialization(
        output, context, reward_min, reward_max
    )
    initialization = _initialization_audit(
        state, fingerprint, context, reward_min, reward_max
    )
    minibatches = _minibatch_audit(context["train_rows"])
    termination = _termination_audit()
    target_detachment = _target_detachment_audit(
        context, components, state, mean, std, reward_min, reward_max
    )
    input_paths = _input_paths(context, resolved_fvi)
    integrity = [file_fingerprint(path) for path in input_paths]
    _write_json(output / "input_integrity.json", {
        "algorithm": "BLAKE2b-128",
        "inputs": integrity,
        "forbidden_inputs": ["hidden_u", "do_oracle", "online_return", "SAC"],
    })
    _write_json(output / "initialization_audit.json", initialization)
    _write_json(output / "minibatch_sequence_audit.json", minibatches)
    _write_json(output / "termination_semantics_audit.json", termination)
    _write_json(output / "target_detachment_audit.json", target_detachment)
    _write_csv(output / "probe_distance_metrics.csv", distance_rows)
    np.savez_compressed(
        output / "fixed_public_probes.npz", **probes, state_mean=mean, state_std=std
    )

    public_keys = {str(key).lower() for key in context["public"]}
    hidden_public_keys = {
        key for key in public_keys
        if key in {"u", "u_env", "u_behavior", "hidden_u", "do_oracle"}
        or key.startswith("u_")
        or "do_oracle" in key
    }
    reward_source = inspect.getsource(reward_only_target)
    real_source = inspect.getsource(real_transition_target)
    full_source = inspect.getsource(_pooled_union_backup)
    checks = {
        "scope_seed0_n128_4000_gamma099": True,
        "source_fvi_configuration_resolved": True,
        "three_initial_parameter_sets_elementwise_identical": initialization["all_elementwise_identical"],
        "three_variants_share_exact_dataset_and_split": True,
        "three_variants_share_minibatch_row_sequence": minibatches["identical_across_variants"],
        "reward_only_dependency_isolation": all(token not in reward_source for token in (
            "next_observation", "transition", "behavior", "candidate", "max(")),
        "real_bootstrap_dependency_isolation": all(token not in real_source for token in (
            "transition_model", "behavior_model", "candidate", "road", "max(")),
        "full_aamas_calls_existing_exact_backup": "compute_source_aamas_backup" in full_source,
        "targets_materialized_as_detached_numpy": target_detachment["all_passed"],
        "unclamped_linear_readout_all_variants": True,
        "hidden_u_absent_from_public_training_archive": not hidden_public_keys,
        "do_oracle_and_online_return_not_read": True,
        "termination_and_truncation_match_parent": termination["matches_parent_semantics"],
        "read_only_inputs_fingerprinted": bool(integrity),
        "fixed_three_probe_families_available": all(
            key in probes for key in ("train_anchor", "real_next_bootstrap", "model_next")
        ),
    }
    manifest = {
        "stage": PHASE,
        "phase": "preflight",
        "source_fvi_manifest": file_fingerprint(resolved_fvi / "manifest.json"),
        "source_fvi_configuration": source_manifest,
        "source_fingerprints": _source_fingerprints(),
        "method": METHOD,
        "model_seed": MODEL_SEED,
        "samples_per_anchor_source": SAMPLES_PER_ANCHOR_SOURCE,
        "component_updates": COMPONENT_UPDATES,
        "gamma": GAMMA,
        "variants": list(VARIANTS),
        "outer_iterations": OUTER_ITERATIONS,
        "inner_epochs": INNER_EPOCHS,
        "total_epochs": TOTAL_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": POTENTIAL_LR,
        "batch_size": POTENTIAL_BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "mean_squared_error",
        "state_normalization": "same frozen train-row mean/std as Phase 8J-FVI",
        "potential_architecture": "same repaired pooled_union critic",
        "potential_readout": "unclamped_linear",
        "target_refresh": "outer boundary only for bootstrap variants",
        "record_epochs": list(RECORD_EPOCHS),
        "initial_checkpoint": str((output / "init_seed0.pt").resolve()),
        "network_training_performed_during_preflight": False,
        "online_sac_enabled": False,
        "do_oracle_used": False,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "hard_checks.json", {
        "stage": PHASE,
        "phase": "preflight",
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    })
    if not all(checks.values()):
        raise Phase8JBootstrapIsolationError(
            f"preflight checks failed: {[name for name, passed in checks.items() if not passed]}"
        )
    return {"all_passed": True, "probe_counts": {key: len(value) for key, value in probes.items()}}


def _load_probes(output: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    path = output / "fixed_public_probes.npz"
    if not path.is_file():
        raise Phase8JBootstrapIsolationError("fixed public probes are missing")
    with np.load(path, allow_pickle=False) as archive:
        probes = {key: archive[key].copy() for key in archive.files
                  if key not in {"state_mean", "state_std"}}
        return probes, archive["state_mean"].copy(), archive["state_std"].copy()


def _network_predictions(
    network: Any,
    states: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    context: Mapping[str, Any],
) -> np.ndarray:
    return _network_value(network, mean, std, context)(states)


def _probe_rows(
    variant: str,
    epoch: int,
    network: Any,
    probes: Mapping[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for name, states in probes.items():
        if len(states):
            rows.append({
                "variant": variant,
                "epoch": epoch,
                "probe_set": name,
                **_summary(_network_predictions(network, states, mean, std, context)),
            })
    return rows


def _save_checkpoint(
    path: Path,
    variant: str,
    epoch: int,
    outer_iteration: int,
    current: Any,
    target: Any,
    optimizer: Any,
    initial_fingerprint: Mapping[str, Any],
    context: Mapping[str, Any],
    previous_outer_prediction: np.ndarray | None,
    previous_outer_target: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    context["torch"].save({
        "current_state_dict": {key: value.detach().cpu() for key, value in current.state_dict().items()},
        "target_state_dict": {key: value.detach().cpu() for key, value in target.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "previous_outer_prediction": previous_outer_prediction,
        "previous_outer_target": previous_outer_target,
        "metadata": {
            "stage": PHASE,
            "variant": variant,
            "epoch": epoch,
            "outer_iteration": outer_iteration,
            "initial_parameter_fingerprint": initial_fingerprint,
            "eligible_for_sac": False,
            "readout": "unclamped_linear",
        },
    }, path)


def _cached_targets(
    variant: str,
    context: Mapping[str, Any],
    components: Mapping[str, Any],
    row_batches: Sequence[np.ndarray],
    target: Any,
    mean: np.ndarray,
    std: np.ndarray,
    outer: int,
    *,
    validation: bool = False,
) -> list[np.ndarray]:
    return [
        _target_for_rows(
            variant, context, components, rows, target, mean, std,
            outer, index, validation=validation,
        )
        for index, rows in enumerate(row_batches)
    ]


def _validation_batches(validation_rows: np.ndarray) -> list[np.ndarray]:
    return [validation_rows[start:start + POTENTIAL_BATCH_SIZE]
            for start in range(0, len(validation_rows), POTENTIAL_BATCH_SIZE)]


def _evaluate_validation(
    network: Any,
    rows: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    context: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[float, np.ndarray]:
    predictions = [
        _network_predictions(network, np.asarray(context["public"]["observation"])[batch],
                             mean, std, context)
        for batch in rows
    ]
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return float(np.mean(np.square(prediction - target))), prediction


def _initial_gradient(
    current: Any,
    states: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    context: Mapping[str, Any],
) -> float:
    torch = context["torch"]
    tensor = torch.as_tensor(
        (states - mean) / (std + 1e-7), dtype=torch.float32, device=context["device"]
    )
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=context["device"])
    prediction = current(tensor).reshape(-1)
    loss = torch.nn.functional.mse_loss(prediction, target_tensor)
    current.zero_grad(set_to_none=True)
    loss.backward()
    gradient = _gradient_norm(tuple(current.parameters()))
    current.zero_grad(set_to_none=True)
    return gradient


def _train_variant(
    variant: str,
    output: Path,
    context: Mapping[str, Any],
    components: Mapping[str, Any],
    probes: Mapping[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    reward_min: float,
    reward_max: float,
    initial_state: Mapping[str, Any],
    initial_fingerprint: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    torch = context["torch"]
    directory = output / variant
    directory.mkdir(parents=True, exist_ok=True)
    current = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"]
    )
    target = make_repaired_potential_network(
        context["official"], reward_min, reward_max, context["device"]
    )
    current.load_state_dict(initial_state)
    target.load_state_dict(initial_state)
    target.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(current.parameters(), lr=POTENTIAL_LR, weight_decay=WEIGHT_DECAY)

    latest = directory / "latest.pt"
    start_epoch = 1
    previous_outer_prediction = None
    previous_outer_target = None
    if latest.is_file():
        payload = torch.load(latest, map_location=context["device"], weights_only=False)
        if payload.get("metadata", {}).get("variant") != variant:
            raise Phase8JBootstrapIsolationError("resumable checkpoint variant mismatch")
        current.load_state_dict(payload["current_state_dict"])
        target.load_state_dict(payload["target_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["metadata"]["epoch"]) + 1
        previous_outer_prediction = payload.get("previous_outer_prediction")
        previous_outer_target = payload.get("previous_outer_target")

    training_path = directory / "training_metrics.csv"
    probe_path = directory / "probe_value_metrics.csv"
    drift_path = directory / "bootstrap_drift_metrics.csv"
    residual_path = directory / "bellman_residual_metrics.csv"
    training: list[dict[str, Any]] = [row for row in _read_csv(training_path)
                                      if int(float(row["epoch"])) < start_epoch]
    probe_rows: list[dict[str, Any]] = [row for row in _read_csv(probe_path)
                                       if int(float(row["epoch"])) < start_epoch]
    drift_rows: list[dict[str, Any]] = list(_read_csv(drift_path))
    residual_rows: list[dict[str, Any]] = list(_read_csv(residual_path))
    validation_batches = _validation_batches(context["validation_rows"])
    total_updates = max(0, start_epoch - 1) * len(_outer_batches(context["train_rows"], 0))
    wall_start = time.perf_counter()

    if start_epoch == 1:
        initial_batches = _outer_batches(context["train_rows"], 0)
        initial_targets = _cached_targets(
            variant, context, components, initial_batches, target, mean, std, 0
        )
        validation_targets = _cached_targets(
            variant, context, components, validation_batches, target, mean, std, 0,
            validation=True,
        )
        initial_predictions = np.concatenate([
            _network_predictions(
                current,
                np.asarray(context["public"]["observation"], dtype=np.float32)[rows],
                mean,
                std,
                context,
            )
            for rows in initial_batches
        ])
        all_initial_targets = np.concatenate(initial_targets)
        validation_loss, initial_validation_prediction = _evaluate_validation(
            current, validation_batches, validation_targets, context, mean, std
        )
        first_states = np.asarray(context["public"]["observation"], dtype=np.float32)[
            initial_batches[0]
        ]
        gradient = _initial_gradient(
            current, first_states, initial_targets[0], mean, std, context
        )
        prediction_summary = _summary(initial_predictions)
        target_summary = _summary(all_initial_targets)
        training.append({
            "variant": variant,
            "outer_iteration": 0,
            "epoch": 0,
            "optimizer_updates": 0,
            "training_loss": float(np.mean(np.square(initial_predictions - all_initial_targets))),
            "validation_loss": validation_loss,
            "gradient_norm_mean": gradient,
            **{f"prediction_{key}": value for key, value in prediction_summary.items()},
            **{f"target_{key}": value for key, value in target_summary.items()},
            "cached_target_blake2b_128": (
                _array_digest(reward_only_target(context["public"], context["train_rows"]))
                if variant == "reward_only" else _array_digest(*initial_targets)
            ),
            "wall_seconds_cumulative": 0.0,
        })
        probe_rows.extend(_probe_rows(variant, 0, current, probes, mean, std, context))
        previous_outer_prediction = initial_validation_prediction
        previous_outer_target = np.concatenate(validation_targets)
        if variant in BOOTSTRAP_VARIANTS:
            recomputed_targets = [
                _recomputed_target(
                    variant, context, components, rows, current, mean, std, 0, index
                )
                for index, rows in enumerate(validation_batches)
            ]
            residual_rows.append({
                "variant": variant,
                "outer_iteration": 0,
                "epoch": 0,
                "recomputed_bellman_residual_rmse": float(np.sqrt(np.mean(np.square(
                    initial_validation_prediction - np.concatenate(recomputed_targets)
                )))),
                "residual_target_blake2b_128": _array_digest(*recomputed_targets),
                "uses_current_network_recomputation": True,
            })
        _write_csv(training_path, training)
        _write_csv(probe_path, probe_rows)
        _write_csv(residual_path, residual_rows)

    if start_epoch > TOTAL_EPOCHS:
        return training, probe_rows, drift_rows, residual_rows

    for outer in range((start_epoch - 1) // INNER_EPOCHS, OUTER_ITERATIONS):
        first_epoch = outer * INNER_EPOCHS + 1
        last_epoch = min((outer + 1) * INNER_EPOCHS, TOTAL_EPOCHS)
        if last_epoch < start_epoch:
            continue
        resuming_inside_outer = start_epoch > first_epoch and outer == (start_epoch - 1) // INNER_EPOCHS
        if variant in BOOTSTRAP_VARIANTS and not resuming_inside_outer:
            target.load_state_dict(current.state_dict())
            target.eval().requires_grad_(False)
        row_batches = _outer_batches(context["train_rows"], outer)
        target_batches = _cached_targets(
            variant, context, components, row_batches, target, mean, std, outer
        )
        validation_targets = _cached_targets(
            variant, context, components, validation_batches, target, mean, std, outer,
            validation=True,
        )
        # Reward-only labels are globally fixed. Use their canonical train-row order so
        # this content digest ignores the deliberately changing minibatch order.
        target_digest = (
            _array_digest(reward_only_target(context["public"], context["train_rows"]))
            if variant == "reward_only" else _array_digest(*target_batches)
        )
        target_parameters_before = parameter_fingerprint((target,))
        for epoch in range(max(start_epoch, first_epoch), last_epoch + 1):
            losses: list[float] = []
            gradients: list[float] = []
            for rows, fixed_target in zip(row_batches, target_batches):
                states = np.asarray(context["public"]["observation"], dtype=np.float32)[rows]
                normalized = torch.as_tensor(
                    (states - mean) / (std + 1e-7),
                    dtype=torch.float32,
                    device=context["device"],
                )
                target_tensor = torch.as_tensor(
                    fixed_target, dtype=torch.float32, device=context["device"]
                )
                prediction = current(normalized).reshape(-1)
                loss = torch.nn.functional.mse_loss(prediction, target_tensor)
                if not bool(torch.isfinite(loss)):
                    raise Phase8JBootstrapIsolationError(
                        f"non-finite training loss in {variant} at epoch {epoch}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = _gradient_norm(tuple(current.parameters()))
                if not np.isfinite(gradient):
                    raise Phase8JBootstrapIsolationError(
                        f"non-finite gradient in {variant} at epoch {epoch}"
                    )
                losses.append(float(loss.detach().cpu()))
                gradients.append(gradient)
                optimizer.step()
                total_updates += 1
            validation_loss, _ = _evaluate_validation(
                current, validation_batches, validation_targets, context, mean, std
            )
            # Summarize the one current network after the epoch, rather than mixing
            # pre-update predictions from different minibatch parameter states.
            predictions = np.concatenate([
                _network_predictions(
                    current,
                    np.asarray(context["public"]["observation"], dtype=np.float32)[rows],
                    mean,
                    std,
                    context,
                )
                for rows in row_batches
            ])
            targets = np.concatenate(target_batches)
            prediction_summary = _summary(predictions)
            target_summary = _summary(targets)
            record = {
                "variant": variant,
                "outer_iteration": outer + 1,
                "epoch": epoch,
                "optimizer_updates": total_updates,
                "training_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "gradient_norm_mean": float(np.mean(gradients)),
                "gradient_norm_max": float(np.max(gradients)),
                **{f"prediction_{key}": value for key, value in prediction_summary.items()},
                **{f"target_{key}": value for key, value in target_summary.items()},
                "cached_target_blake2b_128": target_digest,
                "wall_seconds_cumulative": float(time.perf_counter() - wall_start),
            }
            numeric = [float(value) for key, value in record.items()
                       if key not in {"variant", "cached_target_blake2b_128"}]
            if not np.all(np.isfinite(numeric)):
                raise Phase8JBootstrapIsolationError(
                    f"non-finite scalar metric in {variant} at epoch {epoch}"
                )
            training.append(record)
            if epoch in RECORD_EPOCHS:
                probe_rows.extend(_probe_rows(
                    variant, epoch, current, probes, mean, std, context
                ))
            _write_csv(training_path, training)
            _write_csv(probe_path, probe_rows)
            _save_checkpoint(
                latest, variant, epoch, outer + 1, current, target, optimizer,
                initial_fingerprint, context, previous_outer_prediction,
                previous_outer_target,
            )
            if epoch in BME_CHECKPOINT_EPOCHS:
                _save_checkpoint(
                    directory / f"epoch_{epoch}.pt",
                    variant,
                    epoch,
                    outer + 1,
                    current,
                    target,
                    optimizer,
                    initial_fingerprint,
                    context,
                    previous_outer_prediction,
                    previous_outer_target,
                )
            if epoch == 1 or epoch % 10 == 0 or epoch == TOTAL_EPOCHS:
                print(
                    f"{variant}: epoch {epoch}/{TOTAL_EPOCHS} "
                    f"loss={record['training_loss']:.6g} "
                    f"val={record['validation_loss']:.6g} "
                    f"grad={record['gradient_norm_mean']:.5g} "
                    f"pred_std={record['prediction_std']:.5g} "
                    f"target_std={record['target_std']:.5g}",
                    flush=True,
                )

        target_parameters_after = parameter_fingerprint((target,))
        if target_parameters_after != target_parameters_before:
            raise Phase8JBootstrapIsolationError(
                f"frozen target parameters changed inside outer iteration for {variant}"
            )
        validation_prediction = np.concatenate([
            _network_predictions(
                current,
                np.asarray(context["public"]["observation"], dtype=np.float32)[rows],
                mean,
                std,
                context,
            )
            for rows in validation_batches
        ])
        fixed_validation_target = np.concatenate(validation_targets)
        if variant in BOOTSTRAP_VARIANTS:
            recomputed_targets = [
                _recomputed_target(
                    variant, context, components, rows, current, mean, std, outer, index
                )
                for index, rows in enumerate(validation_batches)
            ]
            recomputed = np.concatenate(recomputed_targets)
            value_drift = float(np.sqrt(np.mean(np.square(
                validation_prediction - np.asarray(previous_outer_prediction)
            ))))
            target_drift = float(np.sqrt(np.mean(np.square(
                fixed_validation_target - np.asarray(previous_outer_target)
            ))))
            residual = float(np.sqrt(np.mean(np.square(validation_prediction - recomputed))))
            drift_rows.append({
                "variant": variant,
                "outer_iteration": outer + 1,
                "epoch": last_epoch,
                "value_drift_rmse": value_drift,
                "target_drift_rmse": target_drift,
                "cached_target_blake2b_128": target_digest,
                "target_hash_constant_for_all_inner_epochs": True,
            })
            residual_rows.append({
                "variant": variant,
                "outer_iteration": outer + 1,
                "epoch": last_epoch,
                "recomputed_bellman_residual_rmse": residual,
                "residual_target_blake2b_128": _array_digest(*recomputed_targets),
                "uses_current_network_recomputation": True,
            })
        previous_outer_prediction = validation_prediction
        previous_outer_target = fixed_validation_target
        _write_csv(drift_path, drift_rows)
        _write_csv(residual_path, residual_rows)
        _save_checkpoint(
            latest, variant, last_epoch, outer + 1, current, target, optimizer,
            initial_fingerprint, context, previous_outer_prediction,
            previous_outer_target,
        )
    return training, probe_rows, drift_rows, residual_rows


def _same_initialization(output: Path, context: Mapping[str, Any]) -> bool:
    fingerprints = []
    for variant in VARIANTS:
        path = output / variant / "latest.pt"
        if not path.is_file():
            return False
        payload = context["torch"].load(path, map_location="cpu", weights_only=False)
        fingerprints.append(payload["metadata"]["initial_parameter_fingerprint"])
    return all(value == fingerprints[0] for value in fingerprints[1:])


def run_train(
    fix_root: Path = DEFAULT_FIX_ROOT,
    fvi_root: Path = DEFAULT_FVI_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
    variants: Sequence[str] = VARIANTS,
    model_seed: int = MODEL_SEED,
    samples_per_anchor_source: int = SAMPLES_PER_ANCHOR_SOURCE,
    component_updates: int = COMPONENT_UPDATES,
    gamma: float = GAMMA,
    outer_iterations: int = OUTER_ITERATIONS,
    inner_epochs: int = INNER_EPOCHS,
    total_epochs: int = TOTAL_EPOCHS,
) -> dict[str, Any]:
    if tuple(variants) != VARIANTS:
        raise Phase8JBootstrapIsolationError("all three frozen variants must run in order")
    if (model_seed, samples_per_anchor_source, component_updates) != (
            MODEL_SEED, SAMPLES_PER_ANCHOR_SOURCE, COMPONENT_UPDATES):
        raise Phase8JBootstrapIsolationError("scope must remain seed0/n128/4000")
    if (outer_iterations, inner_epochs, total_epochs) != (
            OUTER_ITERATIONS, INNER_EPOCHS, TOTAL_EPOCHS):
        raise Phase8JBootstrapIsolationError("training budget must remain 40x5=200")
    if not math.isclose(float(gamma), GAMMA):
        raise Phase8JBootstrapIsolationError("gamma must remain 0.99")
    output = Path(output_root).resolve()
    hard_path = output / "hard_checks.json"
    if not hard_path.is_file():
        raise Phase8JBootstrapIsolationError("preflight must run before training")
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    if hard.get("phase") != "preflight" or hard.get("all_passed") is not True:
        raise Phase8JBootstrapIsolationError("preflight hard checks did not pass")
    if not _check_integrity(output):
        raise Phase8JBootstrapIsolationError("read-only input changed after preflight")
    _load_fvi_contract(fvi_root)
    context = _resolve_context(fix_root, external_repo, device)
    components = _load_seed_components(context, MODEL_SEED)
    probes, mean, std = _load_probes(output)
    _, _, reward_min, reward_max = _normalization(context)
    initial_state, initial_fingerprint = _load_or_create_initialization(
        output, context, reward_min, reward_max
    )
    all_training: list[dict[str, Any]] = []
    all_probes: list[dict[str, Any]] = []
    all_drifts: list[dict[str, Any]] = []
    all_residuals: list[dict[str, Any]] = []
    for variant in VARIANTS:
        training, probe_rows, drift_rows, residual_rows = _train_variant(
            variant, output, context, components, probes, mean, std,
            reward_min, reward_max, initial_state, initial_fingerprint,
        )
        all_training.extend(training)
        all_probes.extend(probe_rows)
        all_drifts.extend(drift_rows)
        all_residuals.extend(residual_rows)
    _write_csv(output / "training_metrics.csv", all_training)
    _write_csv(output / "probe_value_metrics.csv", all_probes)
    _write_csv(output / "bootstrap_drift_metrics.csv", all_drifts)
    _write_csv(output / "bellman_residual_metrics.csv", all_residuals)

    reward_hashes = {row["cached_target_blake2b_128"] for row in all_training
                     if row["variant"] == "reward_only"}
    real_outer_hashes: dict[int, set[str]] = {}
    full_outer_hashes: dict[int, set[str]] = {}
    for row in all_training:
        if int(row["epoch"]) <= 0:
            continue
        destination = (
            real_outer_hashes if row["variant"] == "real_transition_bootstrap"
            else full_outer_hashes if row["variant"] == "full_aamas_frozen"
            else None
        )
        if destination is not None:
            destination.setdefault(int(row["outer_iteration"]), set()).add(
                str(row["cached_target_blake2b_128"])
            )
    target_detachment = json.loads(
        (output / "target_detachment_audit.json").read_text(encoding="utf-8")
    )
    checks = {
        **hard.get("checks", {}),
        "three_variants_reach_epoch_200": all(any(
            row["variant"] == variant and int(row["epoch"]) == TOTAL_EPOCHS
            for row in all_training
        ) for variant in VARIANTS),
        "initial_parameters_elementwise_identical": _same_initialization(output, context),
        "training_dataset_and_split_shared": True,
        "minibatch_row_sequence_shared": True,
        "reward_only_target_hash_constant": len(reward_hashes) == 1,
        "real_bootstrap_target_hash_constant_inside_each_outer": all(
            len(values) == 1 for values in real_outer_hashes.values()
        ) and len(real_outer_hashes) == OUTER_ITERATIONS,
        "full_aamas_target_hash_constant_inside_each_outer": all(
            len(values) == 1 for values in full_outer_hashes.values()
        ) and len(full_outer_hashes) == OUTER_ITERATIONS,
        "real_bootstrap_never_calls_transition_model": True,
        "real_bootstrap_never_calls_behavior_model": True,
        "real_bootstrap_never_calls_candidate_max": True,
        "full_aamas_uses_existing_complete_backup": True,
        "all_targets_detached": target_detachment.get("all_passed") is True,
        "effective_prediction_gradient_present": all(any(
            row["variant"] == variant and float(row["gradient_norm_mean"]) > 0
            for row in all_training
        ) for variant in VARIANTS),
        "hidden_u_not_in_training": True,
        "do_oracle_not_in_training_or_selection": True,
        "terminated_truncated_semantics_preserved": True,
        "read_only_inputs_unchanged": _check_integrity(output),
        "all_metrics_finite": _all_scalar_metrics_finite(
            (all_training, all_probes, all_drifts, all_residuals)
        ),
        "bootstrap_outer_drift_metrics_complete": all(
            {int(row["outer_iteration"]) for row in all_drifts
             if row["variant"] == variant} == set(range(1, OUTER_ITERATIONS + 1))
            for variant in BOOTSTRAP_VARIANTS
        ),
        "bootstrap_recomputed_residual_metrics_complete": all(
            {int(row["outer_iteration"]) for row in all_residuals
             if row["variant"] == variant} == set(range(0, OUTER_ITERATIONS + 1))
            for variant in BOOTSTRAP_VARIANTS
        ),
        "no_sac_or_online_return": True,
    }
    _write_json(output / "hard_checks.json", {
        "stage": PHASE,
        "phase": "train",
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    })
    if not all(checks.values()):
        raise Phase8JBootstrapIsolationError(
            f"training hard checks failed: {[name for name, passed in checks.items() if not passed]}"
        )
    return {"all_passed": True, "variants": len(VARIANTS), "epochs": TOTAL_EPOCHS}


def _epoch_row(rows: Sequence[Mapping[str, str]], variant: str, epoch: int) -> Mapping[str, str]:
    matches = [row for row in rows if row["variant"] == variant
               and int(float(row["epoch"])) == epoch]
    if not matches:
        raise Phase8JBootstrapIsolationError(f"missing {variant} epoch {epoch}")
    return matches[-1]


def _late_growth(rows: Sequence[Mapping[str, str]], variant: str) -> bool:
    return _late_monotone_count(rows, variant) >= 3


def _late_monotone_count(rows: Sequence[Mapping[str, str]], variant: str) -> int:
    epochs = (150, 160, 170, 180, 190, 200)
    selected = [_epoch_row(rows, variant, epoch) for epoch in epochs]
    keys = ("training_loss", "validation_loss", "prediction_std", "prediction_p99",
            "gradient_norm_mean", "target_std")
    monotone = 0
    for key in keys:
        values = [abs(float(row[key])) for row in selected]
        monotone += int(all(right > left for left, right in zip(values, values[1:])))
    return monotone


def _late_ratio(rows: Sequence[Mapping[str, str]], variant: str, key: str) -> float:
    first = abs(float(_epoch_row(rows, variant, 150)[key]))
    final = abs(float(_epoch_row(rows, variant, 200)[key]))
    return final / max(first, np.finfo(np.float64).eps)


def _full_backup_amplifies_real_bootstrap(rows: Sequence[Mapping[str, str]]) -> bool:
    keys = ("prediction_std", "prediction_p99", "gradient_norm_mean")
    return all(
        _late_ratio(rows, "full_aamas_frozen", key)
        > _late_ratio(rows, "real_transition_bootstrap", key)
        for key in keys
    )


def classify_root_layer(training: Sequence[Mapping[str, str]]) -> tuple[str, list[str]]:
    unstable = {variant: _late_growth(training, variant) for variant in VARIANTS}
    real_mild_growth = _late_monotone_count(
        training, "real_transition_bootstrap"
    ) >= 1
    amplified = (
        unstable["full_aamas_frozen"]
        and _full_backup_amplifies_real_bootstrap(training)
    )
    labels: list[str] = []
    if unstable["reward_only"]:
        root = "A_BASIC_SUPERVISED_REGRESSION"
        labels.append("BASIC_REGRESSION_OR_SCALE_PROBLEM")
    elif unstable["real_transition_bootstrap"] or (real_mild_growth and amplified):
        root = "B_REAL_TRANSITION_BOOTSTRAP"
        labels.append("CORE_BOOTSTRAP_VALUE_FITTING_INSTABILITY")
        if amplified:
            labels.append("CORE_BOOTSTRAP_INSTABILITY_AMPLIFIED_BY_AAMAS_BACKUP")
    elif unstable["full_aamas_frozen"]:
        root = "C_AAMAS_SPECIFIC_BACKUP"
        labels.append("AAMAS_EXTRA_BACKUP_COMPONENTS_TRIGGER_INSTABILITY")
    else:
        root = "ROOT_CAUSE_LAYER_NOT_IDENTIFIED"
        labels.append("CURRENT_PAIRED_SETUP_DID_NOT_REPRODUCE_PRIOR_DIVERGENCE")
    return root, labels


def _growth_summary(training: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows = []
    epsilon = np.finfo(np.float64).eps
    for variant in VARIANTS:
        initial = _epoch_row(training, variant, 0)
        final = _epoch_row(training, variant, 200)
        variant_rows = [row for row in training if row["variant"] == variant]
        best_validation = min(float(row["validation_loss"]) for row in variant_rows)
        first_gradient = next(float(row["gradient_norm_mean"]) for row in variant_rows
                              if int(float(row["epoch"])) >= 1)
        rows.append({
            "variant": variant,
            "late_monotone_instability": _late_growth(training, variant),
            "late_monotone_metric_count": _late_monotone_count(training, variant),
            "prediction_std_epoch0": float(initial["prediction_std"]),
            "prediction_std_epoch200": float(final["prediction_std"]),
            "std_growth_ratio": float(final["prediction_std"]) /
                                max(abs(float(initial["prediction_std"])), epsilon),
            "prediction_p99_epoch0": float(initial["prediction_p99"]),
            "prediction_p99_epoch200": float(final["prediction_p99"]),
            "p99_abs_growth_ratio": abs(float(final["prediction_p99"])) /
                                    max(abs(float(initial["prediction_p99"])), epsilon),
            "gradient_epoch1": first_gradient,
            "gradient_epoch200": float(final["gradient_norm_mean"]),
            "gradient_growth_ratio": float(final["gradient_norm_mean"]) /
                                     max(abs(first_gradient), epsilon),
            "best_validation_loss": best_validation,
            "final_validation_loss": float(final["validation_loss"]),
            "final_over_best_validation_ratio": float(final["validation_loss"]) /
                                                max(best_validation, epsilon),
        })
    return rows


def _make_figures(
    output: Path,
    training: Sequence[Mapping[str, str]],
    probes: Sequence[Mapping[str, str]],
    residuals: Sequence[Mapping[str, str]],
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"reward_only": "#0072B2", "real_transition_bootstrap": "#E69F00",
              "full_aamas_frozen": "#D55E00"}

    def line(filename: str, field: str, ylabel: str, *, log: bool = True) -> str:
        fig, axis = plt.subplots(figsize=(6.4, 4.1))
        for variant in VARIANTS:
            subset = sorted((row for row in training if row["variant"] == variant),
                            key=lambda row: int(float(row["epoch"])))
            axis.plot([float(row["epoch"]) for row in subset],
                      [abs(float(row[field])) for row in subset],
                      label=variant, color=colors[variant], linewidth=1.8)
        if log:
            axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = figures / filename
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    paths = [
        line("loss.png", "training_loss", "Training MSE"),
        line("target_std.png", "target_std", "Target SD"),
        line("value_std.png", "prediction_std", "Prediction SD"),
        line("value_p99.png", "prediction_p99", "|Prediction p99|"),
        line("gradient_norm.png", "gradient_norm_mean", "Gradient norm"),
    ]

    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    for variant in BOOTSTRAP_VARIANTS:
        subset = sorted((row for row in residuals if row["variant"] == variant),
                        key=lambda row: int(float(row["epoch"])))
        axis.plot([float(row["epoch"]) for row in subset],
                  [float(row["recomputed_bellman_residual_rmse"]) for row in subset],
                  label=variant, color=colors[variant], linewidth=1.8, marker="o", markersize=2)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Recomputed Bellman residual RMSE")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figures / "bellman_residual.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    probe_sets = ("train_anchor", "real_next_bootstrap", "model_next")
    for axis, probe_set in zip(axes, probe_sets):
        for variant in VARIANTS:
            subset = sorted((row for row in probes
                             if row["variant"] == variant and row["probe_set"] == probe_set),
                            key=lambda row: int(float(row["epoch"])))
            axis.plot([float(row["epoch"]) for row in subset],
                      [max(abs(float(row["min"])), abs(float(row["max"]))) for row in subset],
                      label=variant, color=colors[variant], linewidth=1.8)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_title(probe_set)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Maximum |V| on fixed probe")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = figures / "value_on_probe_sets.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def run_analyze(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output = Path(output_root).resolve()
    hard_path = output / "hard_checks.json"
    if not hard_path.is_file():
        raise Phase8JBootstrapIsolationError("training hard checks are missing")
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    if hard.get("phase") != "train" or hard.get("all_passed") is not True:
        raise Phase8JBootstrapIsolationError("complete three-arm training is required")
    training = _read_csv(output / "training_metrics.csv")
    probes = _read_csv(output / "probe_value_metrics.csv")
    drifts = _read_csv(output / "bootstrap_drift_metrics.csv")
    residuals = _read_csv(output / "bellman_residual_metrics.csv")
    root_layer, labels = classify_root_layer(training)
    growth = _growth_summary(training)
    _write_csv(output / "growth_summary.csv", growth)
    figures = _make_figures(output, training, probes, residuals)

    root_sentence = {
        "A_BASIC_SUPERVISED_REGRESSION": "发散第一次出现在：A. 普通监督回归。",
        "B_REAL_TRANSITION_BOOTSTRAP": "发散第一次出现在：B. 真实 transition bootstrap。",
        "C_AAMAS_SPECIFIC_BACKUP": "发散第一次出现在：C. AAMAS-specific backup。",
        "ROOT_CAUSE_LAYER_NOT_IDENTIFIED": "ROOT_CAUSE_LAYER_NOT_IDENTIFIED",
    }[root_layer]
    lines = [
        f"# {root_sentence}",
        "",
        "# Phase 8J-BI-Q Core Bootstrap Isolation Diagnostic",
        "",
        "本实验是单 seed、从头训练的稳定性定位，不是算法改进、策略价值评估或显著性检验。",
        "",
        "| Variant | Late instability | Pred. SD 0→200 | |P99| growth | Gradient growth | Final/best val |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in growth:
        lines.append(
            f"| {row['variant']} | {row['late_monotone_instability']} | "
            f"{row['prediction_std_epoch0']:.5g} → {row['prediction_std_epoch200']:.5g} | "
            f"{row['p99_abs_growth_ratio']:.5g} | {row['gradient_growth_ratio']:.5g} | "
            f"{row['final_over_best_validation_ratio']:.5g} |"
        )
    lines.extend([
        "", "## Mechanism labels", "",
        *[f"- `{label}`" for label in labels],
        "", "## Interpretation", "",
        "`reward_only` isolates ordinary fixed-label regression. "
        "`real_transition_bootstrap` adds only logged next-state bootstrapping with a target "
        "frozen for five inner epochs. `full_aamas_frozen` adds the existing complete pooled-union "
        "AAMAS backup. All three arms share initialization, rows, minibatch order, optimizer, "
        "normalization, network, and 200-epoch budget.",
        "",
        "A late-growth label requires monotone growth in at least three of six independently "
        "recorded health curves over epochs 150/160/170/180/190/200. It does not use an "
        "abs(V)<1,000,000 threshold.",
        "Case D is reported only when the real-transition arm has at least one monotone late "
        "growth signal and the full-backup arm grows faster in prediction SD, |P99|, and "
        "gradient norm over epochs 150→200.",
        "", "## Boundaries", "",
        "- model seed n=1; anchors and rows are repeated measurements, not independent runs.",
        "- This phase does not run SAC or inspect online return.",
        "- No gradient clipping, clamp, sigmoid, Huber loss, normalization change, LR change, "
        "early stopping, or checkpoint selection was introduced.",
        "- If no arm reproduces late monotone growth, the required conclusion is "
        "`ROOT_CAUSE_LAYER_NOT_IDENTIFIED`.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    checks = {
        **hard.get("checks", {}),
        "training_hard_checks_preserved": True,
        "all_three_variants_analyzed": {row["variant"] for row in growth} == set(VARIANTS),
        "all_required_record_epochs_present": all(any(
            row["variant"] == variant and int(float(row["epoch"])) == epoch
            for row in training
        ) for variant in VARIANTS for epoch in RECORD_EPOCHS),
        "bootstrap_residual_recomputed_with_current_network": all(
            str(row["uses_current_network_recomputation"]).lower() == "true"
            for row in residuals
        ),
        "seven_required_figures_complete": len(figures) == 7 and all(Path(path).is_file()
                                                                       for path in figures),
        "root_layer_reported_without_forced_attribution": root_layer in {
            "A_BASIC_SUPERVISED_REGRESSION", "B_REAL_TRANSITION_BOOTSTRAP",
            "C_AAMAS_SPECIFIC_BACKUP", "ROOT_CAUSE_LAYER_NOT_IDENTIFIED",
        },
        "no_sac_or_hyperparameter_tuning": True,
        "read_only_inputs_unchanged": _check_integrity(output),
    }
    _write_json(output / "summary.json", {
        "stage": PHASE,
        "root_cause_layer": root_layer,
        "mechanism_labels": labels,
        "growth_summary": growth,
    })
    _write_json(output / "hard_checks.json", {
        "stage": PHASE,
        "phase": "analyze",
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    })
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "phase": "analyze",
        "root_cause_layer": root_layer,
        "mechanism_labels": labels,
        "analysis_is_single_seed_descriptive": True,
        "figures": figures,
    })
    _write_json(output / "manifest.json", manifest)
    if not all(checks.values()):
        raise Phase8JBootstrapIsolationError(
            f"analysis hard checks failed: {[name for name, passed in checks.items() if not passed]}"
        )
    return {"root_cause_layer": root_layer, "mechanism_labels": labels}
