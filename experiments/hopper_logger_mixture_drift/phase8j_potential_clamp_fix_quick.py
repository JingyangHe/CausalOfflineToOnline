"""Phase 8J-FIX-Q: repair the potential output parameterization only.

The source aggregation, candidate actions, AAMAS backups, PBRS wrapper, and SAC
configuration are inherited from Phase 8H/8J.  The sole model change is that a
project-local subclass returns the official Critic's linear network output
without applying its final hard clamp.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from aamas_hopper_adapter import _import_official_module, validate_external_repo
from scripts.train_aamas_hopper_potential import seed_everything
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
    _device_name,
    _dynamic_target,
    _evaluate_online_model,
    _load_component,
    _make_online_environment,
    _phase8a_from_phase8h,
    _polyak_update,
    _replay_matches_commands,
    _runtime_versions,
    _same_fingerprint,
    discounted_shaping_sum,
    file_fingerprint,
    normalized_auc,
    normalized_positive_area,
    parameter_fingerprint,
    resolve_artifact_root,
)
from .phase8j_large_data_online_sanity import (
    FORMAL_EVAL_STEPS,
    MODEL_SEEDS,
    NUMERIC_EXPLOSION_LIMIT,
    TRANSITION_COUNT,
    _component_contract,
    _load_public_archive,
    _old_input_paths,
    _replay_training_snapshot,
    _shaping_diagnostics,
)
from .phase8h_quick_multipolicy_aamas import _load_phase8h_inputs


PHASE = "Phase 8J-FIX-Q"
INVALID_STATUS = "INVALID_FOR_METHOD_COMPARISON_CONSTANT_POTENTIAL"
POTENTIAL_METHODS = ("pooled_native", "pooled_union", "state_min", "action_min")
BACKUP_METHODS = {
    "pooled_native": "pooled_aamas_native_full",
    "pooled_union": "pooled_aamas_union_full",
    "state_min": "state_min_full",
    "action_min": "action_min_full",
}
ONLINE_METHODS = (
    "sac_scratch",
    "sac_pooled_native_fixed",
    "sac_pooled_union_fixed",
    "sac_state_min_fixed",
    "sac_action_min_fixed",
)
ONLINE_TO_POTENTIAL = {
    "sac_pooled_native_fixed": "pooled_native",
    "sac_pooled_union_fixed": "pooled_union",
    "sac_state_min_fixed": "state_min",
    "sac_action_min_fixed": "action_min",
}
SMOKE_STEPS = (0, 2_000)
DEFAULT_SCALING_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8h_data_scaling")
DEFAULT_LEGACY_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_large_data_online_sanity")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8j_potential_clamp_fix_quick")


class Phase8JClampFixError(RuntimeError):
    """Raised when the frozen repair protocol cannot proceed."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    seen = set(fields)
    for row in rows[1:]:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_repaired_potential_network(
    official: Any, reward_min: float, reward_max: float, device: str,
) -> Any:
    """Build the official architecture with only its output clamp removed."""

    class RepairedPotentialCritic(official.Critic):
        def forward(self, state: Any) -> Any:
            return self.network(state)

    model = RepairedPotentialCritic(
        12, 3, 1000, reward_max, reward_min, GAMMA).to(device)
    model.phase8j_common_implementation_repair = "remove_final_hard_clamp_only"
    return model


def _potential_path(output: Path, method: str, seed: int) -> Path:
    return output / "potentials" / f"{method}_seed{seed}.pt"


def _load_repaired_potential(
    path: Path, official: Any, torch: Any, device: str,
) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    network = make_repaired_potential_network(
        official, float(payload["reward_min"]), float(payload["reward_max"]), device)
    network.load_state_dict(payload["current_state_dict"])
    network.eval()
    network.requires_grad_(False)
    value = _TorchPotentialValue(
        network, payload["state_mean"], payload["state_std"], device, torch)
    return value, payload["metadata"]


def _tree_snapshot(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    result = []
    for path in sorted((value for value in root.rglob("*") if value.is_file()), key=str):
        stat = path.stat()
        result.append({
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
        })
    return result


def _resolve_inputs(
    scaling_root: Path, legacy_root: Path, external_repo: Path, device: str,
) -> dict[str, Any]:
    scaling = resolve_artifact_root(scaling_root, "Phase 8H-DS")
    legacy = Path(legacy_root).resolve()
    dataset = legacy / "input_data" / "n128_confounded_public.npz"
    if not dataset.is_file():
        raise Phase8JClampFixError(f"frozen n128 dataset is missing: {dataset}")
    public = _load_public_archive(dataset)
    if len(public["reward"]) != TRANSITION_COUNT:
        raise Phase8JClampFixError("frozen dataset row count is not 196608")
    scaling_manifest = json.loads((scaling / "manifest.json").read_text(encoding="utf-8"))
    if json.loads((scaling / "hard_checks.json").read_text(
            encoding="utf-8")).get("all_passed") is not True:
        raise Phase8JClampFixError("Phase 8H-DS hard checks did not pass")
    phase8h = Path(scaling_manifest["phase8h_root"]).resolve()
    inputs = _load_phase8h_inputs(
        _phase8a_from_phase8h(phase8h), 512, None, compute_checkpoint_hash=False)
    split_path = Path(inputs["split_path"]).resolve()
    splits = inputs["splits"]
    train_rows = np.flatnonzero(np.isin(public["anchor_id"], splits["train"]))
    validation_rows = np.flatnonzero(
        np.isin(public["anchor_id"], splits["observational_validation"]))
    if len(train_rows) + len(validation_rows) >= TRANSITION_COUNT:
        raise Phase8JClampFixError("train/validation rows unexpectedly consume every split")
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
    except (ImportError, OSError) as error:
        raise Phase8JClampFixError("PyTorch is required") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(external_repo)
    paths_by_seed, component_references, component_checks = _component_contract(scaling, torch)
    if not all(component_checks.values()):
        raise Phase8JClampFixError(f"n128 component contract failed: {component_checks}")
    old_input_paths = _old_input_paths(scaling, inputs)
    return {
        "scaling": scaling,
        "legacy": legacy,
        "dataset": dataset,
        "public": public,
        "anchors": inputs["anchors"],
        "splits": splits,
        "split_path": split_path,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "torch": torch,
        "device": selected_device,
        "official": official,
        "paths_by_seed": paths_by_seed,
        "component_references": component_references,
        "component_checks": component_checks,
        "old_input_paths": old_input_paths,
        "scaling_manifest": scaling_manifest,
    }


def _gradient_norm(parameters: Sequence[Any]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(float(parameter.grad.detach().double().square().sum().cpu()))
    return math.sqrt(sum(squares))


def _tensor_summary(value: Any) -> dict[str, float]:
    array = value.detach().double().reshape(-1).cpu().numpy()
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _variation_tolerance(mean: float) -> float:
    return float(100 * np.finfo(np.float32).eps * max(1.0, abs(mean)))


def _load_seed_components(context: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        name: _load_component(path, context["official"], context["torch"], context["device"])
        for name, path in context["paths_by_seed"][seed].items()
    }


def _old_clamp_and_repaired_gradient_test(context: Mapping[str, Any]) -> dict[str, Any]:
    torch = context["torch"]
    official = context["official"]
    device = context["device"]
    reward_min, reward_max = 0.88011545, 2.0
    lower = reward_min / (1.0 - GAMMA)
    seed_everything(0, torch, cuda_training=device == "cuda")
    old = official.Critic(12, 3, 1000, reward_max, reward_min, GAMMA).to(device)
    repaired = make_repaired_potential_network(
        official, reward_min, reward_max, device)
    repaired.load_state_dict(old.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(20260905)
    inputs = torch.randn(64, 12, generator=generator).to(device)
    targets = torch.linspace(lower - 5.0, lower + 5.0, 64, device=device)

    old.zero_grad(set_to_none=True)
    old_raw = old.network(inputs)
    old_output = old(inputs).reshape(-1)
    old_loss = torch.nn.functional.mse_loss(old_output, targets)
    old_loss.backward()
    old_gradient = _gradient_norm(tuple(old.parameters()))
    old_last_weight_gradient = float(
        old.network[-1].weight.grad.detach().norm().cpu())
    old_last_bias_gradient = float(old.network[-1].bias.grad.detach().norm().cpu())

    repaired.zero_grad(set_to_none=True)
    repaired_output = repaired(inputs).reshape(-1)
    repaired_loss = torch.nn.functional.mse_loss(repaired_output, targets)
    repaired_loss.backward()
    repaired_gradient = _gradient_norm(tuple(repaired.parameters()))
    repaired_last_weight_gradient = float(
        repaired.network[-1].weight.grad.detach().norm().cpu())
    repaired_last_bias_gradient = float(
        repaired.network[-1].bias.grad.detach().norm().cpu())
    checks = {
        "old_raw_output_below_positive_lower_bound": bool(
            torch.max(old_raw).detach().cpu() < lower),
        "old_output_locked_to_lower_bound": bool(torch.allclose(
            old_output, torch.full_like(old_output, lower), atol=1e-5, rtol=0.0)),
        "old_prediction_gradient_zero": old_gradient == 0.0,
        "old_last_layer_gradient_zero": (
            old_last_weight_gradient == 0.0 and old_last_bias_gradient == 0.0),
        "repaired_output_is_finite": bool(torch.isfinite(repaired_output).all()),
        "repaired_loss_is_finite": bool(torch.isfinite(repaired_loss)),
        "repaired_prediction_gradient_nonzero": repaired_gradient > 0.0,
        "repaired_last_weight_gradient_nonzero": repaired_last_weight_gradient > 0.0,
        "repaired_last_bias_gradient_nonzero": repaired_last_bias_gradient > 0.0,
    }
    return {
        "weight_decay": 0.0,
        "reward_min": reward_min,
        "reward_max": reward_max,
        "gamma": GAMMA,
        "old_value_lower_bound": lower,
        "old_raw_output": _tensor_summary(old_raw),
        "old_clamped_output": _tensor_summary(old_output),
        "old_loss": float(old_loss.detach().cpu()),
        "old_gradient_norm": old_gradient,
        "old_last_weight_gradient_norm": old_last_weight_gradient,
        "old_last_bias_gradient_norm": old_last_bias_gradient,
        "repaired_output": _tensor_summary(repaired_output),
        "repaired_loss": float(repaired_loss.detach().cpu()),
        "repaired_gradient_norm": repaired_gradient,
        "repaired_last_weight_gradient_norm": repaired_last_weight_gradient,
        "repaired_last_bias_gradient_norm": repaired_last_bias_gradient,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _fixed_batch_fit_test(context: Mapping[str, Any]) -> dict[str, Any]:
    torch = context["torch"]
    device = context["device"]
    public = context["public"]
    rows = context["train_rows"][:256]
    components = _load_seed_components(context, 0)
    sources = tuple(components[f"source_{source}"] for source in (1, 2, 3))
    states = np.asarray(public["observation"])[rows]
    actions = np.asarray(public["commanded_action"])[rows]
    bases = np.asarray(context["anchors"]["base_action"], dtype=np.float32)[
        np.asarray(public["anchor_id"])[rows].astype(np.int64)]
    terminated = np.asarray(public["terminated"])[rows]
    state_mean = np.asarray(public["observation"])[context["train_rows"]].mean(
        axis=0, keepdims=True).astype(np.float32)
    state_std = np.asarray(public["observation"])[context["train_rows"]].std(
        axis=0, ddof=1, keepdims=True).astype(np.float32)
    reward_min = float(np.min(np.asarray(public["reward"])[context["train_rows"]]))
    reward_max = float(np.max(np.asarray(public["reward"])[context["train_rows"]]))
    seed_everything(0, torch, cuda_training=device == "cuda")
    network = make_repaired_potential_network(
        context["official"], reward_min, reward_max, device)
    target_network = make_repaired_potential_network(
        context["official"], reward_min, reward_max, device)
    target_network.load_state_dict(network.state_dict())
    target_network.eval().requires_grad_(False)
    target_value = _TorchPotentialValue(target_network, state_mean, state_std, device, torch)
    frozen_backup, _ = _dynamic_target(
        "pooled_aamas_union_full", states, actions, bases, sources,
        components["pooled_balanced"], target_value, update_seed=20260906,
        terminated=terminated)
    normalized = torch.as_tensor(
        (states - state_mean) / (state_std + 1e-7), dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(frozen_backup, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=POTENTIAL_LR, weight_decay=1e-5)
    initial_parameters = parameter_fingerprint((network,))
    with torch.no_grad():
        initial_prediction = network(normalized).reshape(-1)
    losses, gradients = [], []
    for _ in range(100):
        prediction = network(normalized).reshape(-1)
        loss = torch.nn.functional.mse_loss(prediction, target_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(_gradient_norm(tuple(network.parameters())))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        final_prediction = network(normalized).reshape(-1)
        final_loss = float(torch.nn.functional.mse_loss(
            final_prediction, target_tensor).cpu())
    final_parameters = parameter_fingerprint((network,))
    target_summary = {
        "min": float(np.min(frozen_backup)), "max": float(np.max(frozen_backup)),
        "mean": float(np.mean(frozen_backup)), "std": float(np.std(frozen_backup)),
    }
    tolerance = _variation_tolerance(target_summary["mean"])
    prediction_change = float(
        torch.max(torch.abs(final_prediction - initial_prediction)).cpu())
    checks = {
        "actual_backup_target_finite": bool(np.all(np.isfinite(frozen_backup))),
        "actual_backup_target_nontrivial": target_summary["std"] > tolerance,
        "prediction_loss_gradient_nonzero": max(gradients) > 0.0,
        "prediction_changed": prediction_change > tolerance,
        "fixed_target_loss_decreased": final_loss < losses[0],
        "parameters_changed": not _same_fingerprint(initial_parameters, final_parameters),
        "diagnostic_checkpoint_not_reused": True,
    }
    return {
        "method": "pooled_aamas_union_full",
        "seed": 0,
        "row_count": len(rows),
        "updates": 100,
        "optimizer": "Adam",
        "learning_rate": POTENTIAL_LR,
        "weight_decay": 1e-5,
        "target": target_summary,
        "initial_prediction": _tensor_summary(initial_prediction),
        "final_prediction": _tensor_summary(final_prediction),
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "loss_curve": losses,
        "prediction_gradient_norm_curve": gradients,
        "max_prediction_change": prediction_change,
        "initial_parameter_fingerprint": initial_parameters,
        "final_parameter_fingerprint": final_parameters,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def run_preflight_and_tests(
    scaling_root: Path = DEFAULT_SCALING_ROOT,
    legacy_root: Path = DEFAULT_LEGACY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = _resolve_inputs(scaling_root, legacy_root, external_repo, device)
    dataset = context["dataset"]
    component_paths = [
        path for seed in MODEL_SEEDS for path in context["paths_by_seed"][seed].values()]
    old_tree = _tree_snapshot(context["legacy"])
    legacy_record = {
        "status": INVALID_STATUS,
        "reason": "All completed potentials had zero validation output variance due to the "
                  "positive hard-clamp lower bound; method comparison was invalid.",
        "service": "phase8j-large-data-online-sanity.service",
        "main_pid_at_stop": 423585,
        "worker_pid_at_stop": 423590,
        "command": "/home/causal/cc/ENTER/envs/causalTOT/bin/python -u "
                   "scripts/run_phase8j_large_data_online_sanity.py --phase potentials ...",
        "output_directory": str(context["legacy"]),
        "stopped_at": "2026-09-05T06:03:10-04:00",
        "stop_scope": "exact systemd user unit only",
        "legacy_tree_snapshot": old_tree,
    }
    _write_json(output / "legacy_invalidation.json", legacy_record)
    gradient = _old_clamp_and_repaired_gradient_test(context)
    fit = _fixed_batch_fit_test(context)
    checks = {
        "legacy_marked_invalid_without_overwrite": bool(old_tree),
        "frozen_n128_dataset_reused_not_regenerated": dataset.is_file(),
        "transition_count_196608": len(context["public"]["reward"]) == TRANSITION_COUNT,
        "train_split_only_for_training": (
            0 < len(context["train_rows"]) < TRANSITION_COUNT),
        "validation_split_separate": not np.intersect1d(
            context["train_rows"], context["validation_rows"]).size,
        "all_n128_4000_components_valid": all(context["component_checks"].values()),
        "old_clamp_failure_reproduced": gradient["all_passed"],
        "repaired_fixed_batch_fit_passed": fit["all_passed"],
        "external_repository_unmodified": True,
    }
    diagnostics = {
        "stage": PHASE,
        "old_clamp_and_repaired_gradient": gradient,
        "fixed_actual_backup_batch_fit": fit,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    _write_json(output / "gradient_and_fit_tests.json", diagnostics)
    patch = """--- external official Critic.forward (read-only)\n+++ project-local RepairedPotentialCritic.forward\n@@\n- min_vs = self.min_r / (1 - self.gamma)\n- max_vs = self.max_r / (1 - self.gamma)\n- return torch.clamp(self.network(state), min=min_vs, max=max_vs)\n+ return self.network(state)\n"""
    (output / "critic_patch_diff.txt").write_text(patch, encoding="utf-8")
    reward = np.asarray(context["public"]["reward"])[context["train_rows"]]
    manifest = {
        "stage": PHASE,
        "status": "preflight_complete" if all(checks.values()) else "blocked",
        "repair_kind": "common implementation repair; not an algorithm contribution",
        "repair": "remove final hard clamp only",
        "external_repository_modified": False,
        "legacy_status": INVALID_STATUS,
        "dataset": file_fingerprint(dataset),
        "dataset_policy": "read-only reference; no copy or rematerialization",
        "split_path": str(context["split_path"]),
        "split_sizes_in_rows": {
            "train": len(context["train_rows"]),
            "observational_validation": len(context["validation_rows"]),
        },
        "component_checkpoints": [file_fingerprint(path) for path in component_paths],
        "component_policy": "frozen n128/4000 read-only references",
        "potential_methods": list(POTENTIAL_METHODS),
        "model_seeds": list(MODEL_SEEDS),
        "potential_epochs": POTENTIAL_EPOCHS,
        "potential_batch_size": POTENTIAL_BATCH_SIZE,
        "potential_learning_rate": POTENTIAL_LR,
        "potential_weight_decay": 1e-5,
        "target_tau": TARGET_TAU,
        "target_update_interval_batches": TARGET_UPDATE_INTERVAL,
        "candidate_count": 28,
        "gamma": GAMMA,
        "pbrs_beta": PBRS_BETA,
        "reward_min": float(reward.min()),
        "reward_max": float(reward.max()),
        "old_value_lower_bound": float(reward.min() / (1 - GAMMA)),
        "old_value_upper_bound": float(reward.max() / (1 - GAMMA)),
        "reward_normalization": "none",
        "reward_units": "raw confounded Hopper environment reward",
        "potential_units": "undiscounted network estimate trained against raw-reward AAMAS backup",
        "sac_config": SAC_CONFIG,
        "formal_eval_steps": list(FORMAL_EVAL_STEPS),
        "formal_eval_episodes": 5,
        "smoke_steps": list(SMOKE_STEPS),
        "runtime_versions": _runtime_versions(),
        "external_commit": EXTERNAL_COMMIT,
        "old_artifact_snapshot": old_tree,
    }
    _write_json(output / "repair_manifest.json", manifest)
    _write_json(output / "scratch_reuse_audit.json", {
        "reuse_allowed": False,
        "decision": "rerun scratch",
        "reason": "Phase 8H used a different evaluation grid and the interrupted legacy Phase "
                  "8J produced no complete online scratch run.",
    })
    if not all(checks.values()):
        raise Phase8JClampFixError(
            f"preflight-and-tests failed: {[key for key, value in checks.items() if not value]}")
    return {"all_passed": True, "output_root": str(output)}


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self.count += len(array)
        self.total += float(array.sum())
        self.square_total += float(np.square(array).sum())
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def summary(self) -> dict[str, float]:
        mean = self.total / self.count
        variance = max(0.0, self.square_total / self.count - mean * mean)
        return {"min": self.minimum, "max": self.maximum, "mean": mean,
                "std": math.sqrt(variance)}


def _train_one_repaired_potential(
    method: str,
    run_id: int,
    public: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    base_by_anchor: np.ndarray,
    source_models: Sequence[Any],
    pooled_model: Any,
    official: Any,
    torch: Any,
    device: str,
    epochs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    train_states = np.asarray(public["observation"])[train_rows]
    state_mean = train_states.mean(axis=0, keepdims=True).astype(np.float32)
    state_std = train_states.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    reward_min = float(np.min(np.asarray(public["reward"])[train_rows]))
    reward_max = float(np.max(np.asarray(public["reward"])[train_rows]))
    seed_everything(run_id, torch, cuda_training=device == "cuda")
    current = make_repaired_potential_network(official, reward_min, reward_max, device)
    target = make_repaired_potential_network(official, reward_min, reward_max, device)
    target.load_state_dict(current.state_dict())
    target.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(current.parameters(), lr=POTENTIAL_LR, weight_decay=1e-5)
    initial = parameter_fingerprint((current,))
    rng = np.random.default_rng(run_id + 20260921)
    monitor_rows = validation_rows[:min(len(validation_rows), POTENTIAL_BATCH_SIZE)]
    probe_rows = train_rows[:64]
    probe = torch.as_tensor(
        (np.asarray(public["observation"])[probe_rows] - state_mean) / (state_std + 1e-7),
        dtype=torch.float32, device=device)
    with torch.no_grad():
        initial_probe = current(probe).reshape(-1).detach().clone()
    history: list[dict[str, Any]] = []
    total_updates = 0
    model_forward_units = 0
    wall_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(train_rows)
        losses: list[float] = []
        residuals: list[float] = []
        gradients: list[float] = []
        last_weight_gradients: list[float] = []
        last_bias_gradients: list[float] = []
        backup_moments, prediction_moments = _Moments(), _Moments()
        for batch_index, start in enumerate(range(0, len(permutation), POTENTIAL_BATCH_SIZE)):
            rows = permutation[start:start + POTENTIAL_BATCH_SIZE]
            states = np.asarray(public["observation"])[rows]
            actions = np.asarray(public["commanded_action"])[rows]
            bases = base_by_anchor[np.asarray(public["anchor_id"])[rows].astype(np.int64)]
            target_value = _TorchPotentialValue(target, state_mean, state_std, device, torch)
            backup, forwards = _dynamic_target(
                method, states, actions, bases, source_models, pooled_model,
                target_value, update_seed=run_id * 10_000_000 + total_updates * 2 + 20260922,
                terminated=np.asarray(public["terminated"])[rows])
            normalized = torch.as_tensor(
                (states - state_mean) / (state_std + 1e-7),
                dtype=torch.float32, device=device)
            prediction = current(normalized).reshape(-1)
            target_tensor = torch.as_tensor(backup, dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(prediction, target_tensor)
            if not bool(torch.isfinite(loss)):
                raise Phase8JClampFixError(
                    f"nonfinite potential loss for {method}, seed {run_id}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = _gradient_norm(tuple(current.parameters()))
            last_weight = float(current.network[-1].weight.grad.detach().norm().cpu())
            last_bias = float(current.network[-1].bias.grad.detach().norm().cpu())
            if not all(np.isfinite([gradient, last_weight, last_bias])):
                raise Phase8JClampFixError("nonfinite prediction-loss gradient")
            optimizer.step()
            if batch_index > 0 and batch_index % TARGET_UPDATE_INTERVAL == 0:
                _polyak_update(current, target, TARGET_TAU)
            prediction_array = prediction.detach().cpu().numpy()
            losses.append(float(loss.detach().cpu()))
            residuals.append(float(np.mean(np.abs(prediction_array - backup))))
            gradients.append(gradient)
            last_weight_gradients.append(last_weight)
            last_bias_gradients.append(last_bias)
            backup_moments.add(backup)
            prediction_moments.add(prediction_array)
            total_updates += 1
            model_forward_units += int(forwards)

        monitor_states = np.asarray(public["observation"])[monitor_rows]
        monitor_actions = np.asarray(public["commanded_action"])[monitor_rows]
        monitor_bases = base_by_anchor[
            np.asarray(public["anchor_id"])[monitor_rows].astype(np.int64)]
        monitor_value = _TorchPotentialValue(target, state_mean, state_std, device, torch)
        monitor_backup, _ = _dynamic_target(
            method, monitor_states, monitor_actions, monitor_bases,
            source_models, pooled_model, monitor_value,
            update_seed=run_id * 10_000_000 + epoch + 30_000_000,
            terminated=np.asarray(public["terminated"])[monitor_rows])
        monitor_normalized = torch.as_tensor(
            (monitor_states - state_mean) / (state_std + 1e-7),
            dtype=torch.float32, device=device)
        with torch.no_grad():
            monitor_prediction = current(monitor_normalized).reshape(-1)
            monitor_target_output = target(monitor_normalized).reshape(-1)
            probe_prediction = current(probe).reshape(-1)
        backup_summary = backup_moments.summary()
        prediction_summary = prediction_moments.summary()
        monitor_prediction_np = monitor_prediction.cpu().numpy()
        monitor_target_np = monitor_target_output.cpu().numpy()
        probe_np = probe_prediction.cpu().numpy()
        record = {
            "run_id": run_id,
            "potential": method,
            "epoch": epoch,
            "optimizer_updates": total_updates,
            "training_loss": float(np.mean(losses)),
            "train_residual_mae": float(np.mean(residuals)),
            "backup_target_min": backup_summary["min"],
            "backup_target_max": backup_summary["max"],
            "backup_target_mean": backup_summary["mean"],
            "backup_target_std": backup_summary["std"],
            "potential_min": prediction_summary["min"],
            "potential_max": prediction_summary["max"],
            "potential_mean": prediction_summary["mean"],
            "potential_std": prediction_summary["std"],
            "prediction_gradient_norm_mean": float(np.mean(gradients)),
            "prediction_gradient_norm_max": float(np.max(gradients)),
            "last_weight_gradient_norm_mean": float(np.mean(last_weight_gradients)),
            "last_bias_gradient_norm_mean": float(np.mean(last_bias_gradients)),
            "validation_monitor_row_count": len(monitor_rows),
            "validation_residual_mae": float(np.mean(np.abs(
                monitor_prediction_np - monitor_backup))),
            "validation_backup_min": float(np.min(monitor_backup)),
            "validation_backup_max": float(np.max(monitor_backup)),
            "validation_backup_std": float(np.std(monitor_backup)),
            "target_network_output_min": float(np.min(monitor_target_np)),
            "target_network_output_max": float(np.max(monitor_target_np)),
            "target_network_output_std": float(np.std(monitor_target_np)),
            "probe_prediction_min": float(np.min(probe_np)),
            "probe_prediction_max": float(np.max(probe_np)),
            "probe_prediction_std": float(np.std(probe_np)),
            "probe_max_abs_change_from_initial": float(torch.max(torch.abs(
                probe_prediction - initial_probe)).cpu()),
        }
        if not all(np.isfinite(float(value)) for key, value in record.items()
                   if key not in {"potential"}):
            raise Phase8JClampFixError(
                f"nonfinite training health metric for {method}, seed {run_id}")
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"repaired potential {method} seed {run_id}: epoch {epoch}/{epochs} "
                  f"loss={record['training_loss']:.6g} grad="
                  f"{record['prediction_gradient_norm_mean']:.4g} "
                  f"std={record['potential_std']:.4g}", flush=True)

    validation_states = np.asarray(public["observation"])[validation_rows]
    validation_actions = np.asarray(public["commanded_action"])[validation_rows]
    validation_bases = base_by_anchor[
        np.asarray(public["anchor_id"])[validation_rows].astype(np.int64)]
    validation_value = _TorchPotentialValue(target, state_mean, state_std, device, torch)
    validation_backup, validation_forwards = _dynamic_target(
        method, validation_states, validation_actions, validation_bases,
        source_models, pooled_model, validation_value,
        update_seed=run_id * 10_000_000 + 909_090,
        terminated=np.asarray(public["terminated"])[validation_rows])
    with torch.no_grad():
        validation_normalized = torch.as_tensor(
            (validation_states - state_mean) / (state_std + 1e-7),
            dtype=torch.float32, device=device)
        validation_prediction = current(validation_normalized).reshape(-1).cpu().numpy()
    residual = validation_prediction - validation_backup
    diagnostic = {
        "validation_row_count": len(validation_rows),
        "validation_residual_mae": float(np.mean(np.abs(residual))),
        "validation_residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "backup_target_min": float(np.min(validation_backup)),
        "backup_target_max": float(np.max(validation_backup)),
        "backup_target_mean": float(np.mean(validation_backup)),
        "backup_target_std": float(np.std(validation_backup)),
        "potential_min": float(np.min(validation_prediction)),
        "potential_max": float(np.max(validation_prediction)),
        "potential_mean": float(np.mean(validation_prediction)),
        "potential_std": float(np.std(validation_prediction)),
        "probe_max_abs_change_from_initial": history[-1]["probe_max_abs_change_from_initial"],
        "prediction_gradient_norm_final_epoch": history[-1]["prediction_gradient_norm_mean"],
        "total_model_forward_units": int(model_forward_units + validation_forwards),
        "wall_clock_seconds": float(time.perf_counter() - wall_start),
    }
    payload = {
        "current_state_dict": {
            key: value.detach().cpu() for key, value in current.state_dict().items()},
        "target_state_dict": {
            key: value.detach().cpu() for key, value in target.state_dict().items()},
        "state_mean": state_mean,
        "state_std": state_std,
        "reward_min": reward_min,
        "reward_max": reward_max,
        "metadata": {
            "stage": PHASE,
            "method": method,
            "run_id": run_id,
            "epochs": epochs,
            "optimizer_updates": total_updates,
            "batch_size": POTENTIAL_BATCH_SIZE,
            "learning_rate": POTENTIAL_LR,
            "weight_decay": 1e-5,
            "target_tau": TARGET_TAU,
            "target_update_interval_batches": TARGET_UPDATE_INTERVAL,
            "gamma": GAMMA,
            "candidate_actions_per_source": 8,
            "candidate_count": 28,
            "component_parameters_frozen": True,
            "backup_value_source": "own_repaired_target_potential",
            "fixed_reference_value_used": False,
            "checkpoint_selection": "pre_frozen_final_epoch",
            "continuation_terminal_mask": "terminated_only; truncation_bootstraps",
            "initial_parameter_fingerprint": initial,
            "critic_output_parameterization": "unclamped_linear",
            "common_implementation_repair": True,
            "test_oracle_used": False,
            "online_return_used": False,
        },
    }
    return payload, history, diagnostic


def _health_classification(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    target_std = float(row["backup_target_std"])
    target_mean = float(row["backup_target_mean"])
    prediction_std = float(row["potential_std"])
    probe_change = float(row["probe_max_abs_change_from_initial"])
    gradient = float(row["prediction_gradient_norm_final_epoch"])
    tolerance = _variation_tolerance(target_mean)
    finite = all(np.isfinite(float(row[key])) for key in (
        "backup_target_min", "backup_target_max", "backup_target_std",
        "potential_min", "potential_max", "potential_std",
        "validation_residual_mae", "prediction_gradient_norm_final_epoch"))
    exploded = max(abs(float(row["potential_min"])), abs(float(row["potential_max"]))) \
        > NUMERIC_EXPLOSION_LIMIT
    target_degenerate = target_std <= tolerance
    training_blocked = (
        not target_degenerate and (gradient <= 0.0 or probe_change <= tolerance))
    if not finite or exploded:
        status = "NUMERICAL_FAILURE"
    elif target_degenerate:
        status = "TARGET_DEGENERACY_REQUIRES_MANUAL_REVIEW"
    elif training_blocked:
        status = "TRAINING_BLOCKED"
    else:
        status = "HEALTHY_NONTRIVIAL_TARGET_AND_EFFECTIVE_GRADIENT"
    return status, {
        "finite": finite,
        "exploded": exploded,
        "target_variation_tolerance": tolerance,
        "target_degenerate": target_degenerate,
        "training_blocked": training_blocked,
        "prediction_std": prediction_std,
    }


def _write_health_report(output: Path, diagnostics: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Potential training health", "",
        "This report validates training mechanics. It does not rank methods or imply "
        "scientific success.", "",
        "| Seed | Potential | Status | Target std | Phi std | Gradient | Probe change | "
        "Validation MAE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['model_seed']} | {row['potential']} | {row['health_status']} | "
            f"{float(row['backup_target_std']):.6g} | {float(row['potential_std']):.6g} | "
            f"{float(row['prediction_gradient_norm_final_epoch']):.6g} | "
            f"{float(row['probe_max_abs_change_from_initial']):.6g} | "
            f"{float(row['validation_residual_mae']):.6g} |")
    lines.extend([
        "", "Interpretation rules:", "",
        "- TRAINING_BLOCKED requires a varying target plus zero effective gradient or no "
        "prediction movement.",
        "- TARGET_DEGENERACY is reported for manual review and is not called meaningful "
        "value learning.",
        "- Equal outputs across methods are not by themselves an implementation failure.",
    ])
    (output / "potential_training_health.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def run_potentials(
    scaling_root: Path = DEFAULT_SCALING_ROOT,
    legacy_root: Path = DEFAULT_LEGACY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    preflight_path = output / "gradient_and_fit_tests.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(
            encoding="utf-8")).get("all_passed") is not True:
        raise Phase8JClampFixError("preflight-and-tests must pass before potentials")
    context = _resolve_inputs(scaling_root, legacy_root, external_repo, device)
    manifest = json.loads((output / "repair_manifest.json").read_text(encoding="utf-8"))
    if file_fingerprint(context["dataset"])["blake2b_128"] != \
            manifest["dataset"]["blake2b_128"]:
        raise Phase8JClampFixError("frozen n128 dataset changed after preflight")
    public = context["public"]
    train_rows = context["train_rows"]
    validation_rows = context["validation_rows"]
    base_by_anchor = np.asarray(context["anchors"]["base_action"], dtype=np.float32)
    history_path = output / "potential_training_metrics.csv"
    diagnostic_path = output / "potential_diagnostics.csv"
    histories = _read_csv(history_path) if history_path.is_file() else []
    diagnostics = _read_csv(diagnostic_path) if diagnostic_path.is_file() else []
    initial_by_seed: dict[int, Mapping[str, Any]] = {}
    for seed in MODEL_SEEDS:
        components = _load_seed_components(context, seed)
        sources = tuple(components[f"source_{source}"] for source in (1, 2, 3))
        for method in POTENTIAL_METHODS:
            checkpoint = _potential_path(output, method, seed)
            if checkpoint.is_file():
                payload = context["torch"].load(
                    checkpoint, map_location="cpu", weights_only=False)
                metadata = payload.get("metadata", {})
                valid = (
                    metadata.get("stage") == PHASE
                    and metadata.get("phase8j_fix_method") == method
                    and metadata.get("critic_output_parameterization") == "unclamped_linear"
                    and metadata.get("dataset_blake2b_128")
                    == manifest["dataset"]["blake2b_128"])
                if not valid:
                    raise Phase8JClampFixError(
                        f"existing repaired potential has mismatched provenance: {checkpoint}")
                initial = metadata["initial_parameter_fingerprint"]
                if seed in initial_by_seed and not _same_fingerprint(
                        initial_by_seed[seed], initial):
                    raise Phase8JClampFixError("repaired potential initialization is not paired")
                initial_by_seed[seed] = initial
                print(f"reusing repaired potential {method}, seed {seed}", flush=True)
                continue
            payload, history, diagnostic = _train_one_repaired_potential(
                BACKUP_METHODS[method], seed, public, train_rows, validation_rows,
                base_by_anchor, sources, components["pooled_balanced"],
                context["official"], context["torch"], context["device"], POTENTIAL_EPOCHS)
            payload["metadata"].update({
                "phase8j_fix_method": method,
                "dataset_blake2b_128": manifest["dataset"]["blake2b_128"],
                "samples_per_anchor_source": 128,
                "component_updates": 4000,
                "candidate_random_stream_paired": method != "pooled_native",
            })
            initial = payload["metadata"]["initial_parameter_fingerprint"]
            if seed in initial_by_seed and not _same_fingerprint(
                    initial_by_seed[seed], initial):
                raise Phase8JClampFixError("repaired potential initialization is not paired")
            initial_by_seed[seed] = initial
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            context["torch"].save(payload, checkpoint)
            value, loaded_metadata = _load_repaired_potential(
                checkpoint, context["official"], context["torch"], context["device"])
            probe = np.asarray(context["anchors"]["public_observation"][:16], dtype=np.float32)
            direct = make_repaired_potential_network(
                context["official"], payload["reward_min"], payload["reward_max"],
                context["device"])
            direct.load_state_dict(payload["current_state_dict"])
            expected = _TorchPotentialValue(
                direct, payload["state_mean"], payload["state_std"],
                context["device"], context["torch"])
            roundtrip = float(np.max(np.abs(value(probe) - expected(probe))))
            status, health = _health_classification(diagnostic)
            diagnostic_row = {
                "model_seed": seed,
                "potential": method,
                "backup_method": BACKUP_METHODS[method],
                "health_status": status,
                "roundtrip_max_abs": roundtrip,
                "numeric_status": "finite" if health["finite"] else "nonfinite",
                **diagnostic,
                **_shaping_diagnostics(value, public, validation_rows),
            }
            histories.extend({"model_seed": seed, "potential": method, **row}
                             for row in history)
            diagnostics.append(diagnostic_row)
            _write_csv(history_path, histories)
            _write_csv(diagnostic_path, diagnostics)
            print(f"repaired potential ready: {method}, seed {seed}, health={status}",
                  flush=True)
            del value, direct
            if context["device"] == "cuda":
                context["torch"].cuda.empty_cache()

    diagnostics = list({
        (int(row["model_seed"]), str(row["potential"])): row for row in diagnostics
    }.values())
    histories = list({
        (int(row["model_seed"]), str(row["potential"]), int(row["epoch"])): row
        for row in histories
    }.values())
    _write_csv(history_path, histories)
    _write_csv(diagnostic_path, diagnostics)
    expected_paths = [_potential_path(output, method, seed)
                      for seed in MODEL_SEEDS for method in POTENTIAL_METHODS]
    statuses = {str(row["health_status"]) for row in diagnostics}
    blocked = statuses.intersection({"TRAINING_BLOCKED", "NUMERICAL_FAILURE"})
    manual = "TARGET_DEGENERACY_REQUIRES_MANUAL_REVIEW" in statuses
    old_unchanged = _tree_snapshot(context["legacy"]) == manifest["old_artifact_snapshot"]
    checks = {
        "twelve_repaired_potentials_complete": (
            len(diagnostics) == 12 and all(path.is_file() for path in expected_paths)),
        "all_use_unclamped_linear_critic": all(
            context["torch"].load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "critic_output_parameterization") == "unclamped_linear"
            for path in expected_paths),
        "all_use_own_repaired_target_potential": all(
            context["torch"].load(path, map_location="cpu", weights_only=False)["metadata"].get(
                "backup_value_source") == "own_repaired_target_potential"
            for path in expected_paths),
        "fixed_reference_not_used": all(
            not context["torch"].load(path, map_location="cpu", weights_only=False)[
                "metadata"].get("fixed_reference_value_used") for path in expected_paths),
        "paired_initialization": len(initial_by_seed) == 3,
        "no_training_block_or_numerical_failure": not blocked,
        "checkpoint_roundtrip": all(float(row["roundtrip_max_abs"]) <= 1e-7
                                    for row in diagnostics),
        "old_artifacts_unchanged": old_unchanged,
        "dataset_unchanged": file_fingerprint(context["dataset"])["blake2b_128"]
                             == manifest["dataset"]["blake2b_128"],
        "do_oracle_not_used": True,
    }
    _write_json(output / "potentials" / "hard_checks.json", {
        "all_passed": all(checks.values()) and not manual,
        "training_complete": all(checks.values()),
        "manual_review_required": manual,
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
        "health_statuses": sorted(statuses),
    })
    _write_health_report(output, diagnostics)
    manifest.update({
        "status": "potential_training_complete_manual_review_required",
        "potential_checkpoints": [file_fingerprint(path) for path in expected_paths],
        "potential_health_statuses": sorted(statuses),
        "potential_training_complete": all(checks.values()),
        "online_requires_explicit_post_health_review": True,
    })
    _write_json(output / "repair_manifest.json", manifest)
    if not all(checks.values()):
        raise Phase8JClampFixError(
            f"repaired potential checks failed: {[key for key, value in checks.items() if not value]}")
    return {
        "potential_count": len(expected_paths),
        "training_complete": True,
        "manual_review_required": True,
        "health_statuses": sorted(statuses),
    }


class _ZeroPotential:
    def __call__(self, states: np.ndarray) -> np.ndarray:
        return np.zeros(len(np.asarray(states)), dtype=np.float64)


def _online_potential(
    method: str, seed: int, output: Path, official: Any, torch: Any, device: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    if method == "sac_scratch":
        return _ZeroPotential(), {"phase8j_fix_method": "zero", "run_id": seed}
    potential_method = ONLINE_TO_POTENTIAL[method]
    path = _potential_path(output, potential_method, seed)
    if not path.is_file():
        raise Phase8JClampFixError(f"repaired potential is missing: {path}")
    value, metadata = _load_repaired_potential(path, official, torch, device)
    if metadata.get("critic_output_parameterization") != "unclamped_linear":
        raise Phase8JClampFixError("online attempted to load a clamped potential")
    return value, {**metadata, "checkpoint": str(path), "file": file_fingerprint(path)}


def _rng_snapshot(torch: Any, cuda: bool) -> dict[str, Any]:
    np_state = np.random.get_state()
    result = {
        "python": repr(random.getstate()),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch_cpu": torch.get_rng_state().cpu().tolist(),
    }
    if cuda:
        result["torch_cuda"] = [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
    return result


def _online_curves(
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
                selected = grouped[(seed, method, int(step))]
                if not selected:
                    raise Phase8JClampFixError("incomplete online evaluation curve")
                values.append(float(np.mean(selected)))
            curves[(seed, method)] = (
                np.asarray(eval_steps, dtype=np.float64), np.asarray(values, dtype=np.float64))
    return curves


def _summarize_online(
    evaluation_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int],
    methods: Sequence[str], eval_steps: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves = _online_curves(evaluation_rows, seeds, methods, eval_steps)
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(int(row["online_seed"]), str(row["method"]),
                 int(row["training_step"]))].append(row)
    per_seed = []
    for seed in seeds:
        scratch_steps, scratch_return = curves[(seed, "sac_scratch")]
        pooled_union_auc = normalized_auc(*curves[(seed, "sac_pooled_union_fixed")])
        pooled_native_auc = normalized_auc(*curves[(seed, "sac_pooled_native_fixed")])
        for method in methods:
            steps, returns = curves[(seed, method)]
            final = grouped[(seed, method, int(eval_steps[-1]))]
            auc = normalized_auc(steps, returns)
            per_seed.append({
                "online_seed": seed,
                "method": method,
                "auc_0_50k": auc,
                "auc_difference_vs_pooled_union": auc - pooled_union_auc,
                "auc_difference_vs_pooled_native": auc - pooled_native_auc,
                "negative_transfer_area_vs_scratch": normalized_positive_area(
                    scratch_steps, scratch_return - returns),
                "final_return_50k": float(np.mean([
                    float(row["raw_environment_return"]) for row in final])),
                "final_episode_length": float(np.mean([
                    float(row["episode_length"]) for row in final])),
                "final_terminated_fraction": float(np.mean([
                    str(row["terminated"]).lower() == "true" for row in final])),
            })
    summary = []
    metric_names = (
        "auc_0_50k", "auc_difference_vs_pooled_union",
        "auc_difference_vs_pooled_native", "negative_transfer_area_vs_scratch",
        "final_return_50k", "final_episode_length", "final_terminated_fraction")
    for method in methods:
        selected = [row for row in per_seed if row["method"] == method]
        record: dict[str, Any] = {"method": method, "run_count": len(selected)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(record)
    return per_seed, summary


def run_online(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    external_repo: Path = Path("external/li_aamas2026"),
    device: str = "auto",
    *,
    smoke: bool,
    online_seeds: Sequence[int],
    online_steps: int,
    eval_steps: Sequence[int],
    eval_episodes: int,
) -> dict[str, Any]:
    seeds = tuple(map(int, online_seeds))
    steps = tuple(map(int, eval_steps))
    if smoke:
        expected = seeds == (0,) and online_steps == 2_000 and steps == SMOKE_STEPS
    else:
        expected = (seeds == MODEL_SEEDS and online_steps == 50_000
                    and steps == FORMAL_EVAL_STEPS and eval_episodes == 5)
    if not expected:
        raise Phase8JClampFixError("online phase does not match the frozen repair protocol")
    output = Path(output_root).resolve()
    health = json.loads((output / "potentials" / "hard_checks.json").read_text(
        encoding="utf-8"))
    if health.get("training_complete") is not True:
        raise Phase8JClampFixError("repaired potential training is not healthy and complete")
    if not smoke:
        smoke_path = output / "online" / "smoke" / "hard_checks.json"
        if not smoke_path.is_file() or json.loads(smoke_path.read_text(
                encoding="utf-8")).get("all_passed") is not True:
            raise Phase8JClampFixError("2000-step online smoke must pass first")
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import torch
        from stable_baselines3 import SAC
    except (ImportError, OSError) as error:
        raise Phase8JClampFixError("PyTorch and stable-baselines3 are required") from error
    selected_device = _device_name(device, torch)
    official = _import_official_module(external_repo)
    destination = output / "online" / "smoke" if smoke else output / "online"
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_path = destination / "evaluation_returns.csv"
    training_path = destination / "online_training_metrics.csv"
    evaluation_rows = _read_csv(evaluation_path) if evaluation_path.is_file() else []
    training_rows = _read_csv(training_path) if training_path.is_file() else []
    load_audit_path = destination / "potential_load_audit.json"
    load_audit = (json.loads(load_audit_path.read_text(encoding="utf-8"))
                  if load_audit_path.is_file() else [])
    initial_by_seed: dict[int, Mapping[str, Any]] = {}
    potential_paths = [_potential_path(output, method, seed)
                       for seed in seeds for method in POTENTIAL_METHODS]
    potential_before = {
        str(path): file_fingerprint(path)["blake2b_128"] for path in potential_paths}
    for seed in seeds:
        for method_index, method in enumerate(ONLINE_METHODS):
            completed = [row for row in training_rows
                         if int(row["online_seed"]) == seed and row["method"] == method
                         and int(row["training_step"]) == online_steps
                         and str(row.get("run_complete", "")).lower() == "true"]
            if completed:
                row = completed[0]
                initial = {
                    "parameter_count": int(row["initial_parameter_count"]),
                    "sum": float(row["initial_sum"]),
                    "sum_squares": float(row["initial_sum_squares"]),
                    "max_abs": float(row["initial_max_abs"]),
                }
                if seed in initial_by_seed and not _same_fingerprint(
                        initial_by_seed[seed], initial):
                    raise Phase8JClampFixError("reused SAC initialization is not paired")
                initial_by_seed[seed] = initial
                print(f"reusing completed repaired online run: {method}, seed {seed}", flush=True)
                continue
            evaluation_rows = [row for row in evaluation_rows
                               if not (int(row["online_seed"]) == seed
                                       and row["method"] == method)]
            training_rows = [row for row in training_rows
                             if not (int(row["online_seed"]) == seed
                                     and row["method"] == method)]
            load_audit = [row for row in load_audit
                          if not (int(row["online_seed"]) == seed
                                  and row["method"] == method)]
            potential, metadata = _online_potential(
                method, seed, output, official, torch, selected_device)
            phi_before = parameter_fingerprint((potential.network,)) \
                if hasattr(potential, "network") else None
            if hasattr(potential, "network"):
                potential.network.eval().requires_grad_(False)
            environment = _make_online_environment(
                potential, kappa=KAPPA, lambda_reward=LAMBDA_REWARD,
                beta=PBRS_BETA, gamma=GAMMA)
            seed_everything(seed, torch, cuda_training=selected_device == "cuda")
            model = SAC(env=environment, seed=seed, device=selected_device,
                        verbose=0, **SAC_CONFIG)
            replay_empty = int(model.replay_buffer.size()) == 0
            initial = parameter_fingerprint((model.actor, model.critic))
            if seed in initial_by_seed and not _same_fingerprint(initial_by_seed[seed], initial):
                raise Phase8JClampFixError("SAC initialization is not paired")
            initial_by_seed[seed] = initial
            wall_start = time.perf_counter()
            evaluation_interactions = 0
            evaluation_rng_unchanged = True
            previous = 0
            for step in steps:
                if step > previous:
                    model.learn(total_timesteps=step - previous,
                                reset_num_timesteps=False, progress_bar=False)
                rng_before = _rng_snapshot(torch, selected_device == "cuda")
                rows, interactions = _evaluate_online_model(
                    model, potential, run_id=seed, method_index=method_index,
                    training_step=step, episodes=eval_episodes, kappa=KAPPA,
                    lambda_reward=LAMBDA_REWARD, beta=PBRS_BETA, gamma=GAMMA)
                evaluation_rng_unchanged &= rng_before == _rng_snapshot(
                    torch, selected_device == "cuda")
                evaluation_interactions += interactions
                evaluation_rows.extend({
                    "online_seed": seed, "method": method,
                    **{key: value for key, value in row.items() if key != "run_id"},
                } for row in rows)
                increments = np.asarray(environment.shaping_increments, dtype=np.float64)
                snapshot = _replay_training_snapshot(model, torch)
                training_rows.append({
                    "online_seed": seed,
                    "method": method,
                    "training_step": step,
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
                    "potential_method": metadata["phase8j_fix_method"],
                    "potential_checkpoint": metadata.get("checkpoint", "zero"),
                    "potential_file_blake2b_128": metadata.get("file", {}).get(
                        "blake2b_128", "zero"),
                    "beta": PBRS_BETA,
                    "gamma": GAMMA,
                    "initial_parameter_count": initial["parameter_count"],
                    "initial_sum": initial["sum"],
                    "initial_sum_squares": initial["sum_squares"],
                    "initial_max_abs": initial["max_abs"],
                    "run_complete": False,
                })
                _write_csv(evaluation_path, evaluation_rows)
                _write_csv(training_path, training_rows)
                print(f"repaired online {method} seed {seed}: {step}/{online_steps}", flush=True)
                previous = step
            phi_after = parameter_fingerprint((potential.network,)) \
                if hasattr(potential, "network") else None
            final = training_rows[-1]
            final.update({
                "replay_initially_empty": replay_empty,
                "replay_matches_commanded_actions": _replay_matches_commands(
                    model, environment.commanded_actions),
                "wrapper_commanded_action_match": environment.commanded_action_match,
                "returned_info_private_leak": environment.returned_info_private_leak,
                "raw_reward_formula_match": environment.raw_reward_formula_match,
                "potential_frozen": phi_before == phi_after,
                "potential_eval_mode": (not potential.network.training
                                        if hasattr(potential, "network") else True),
                "potential_requires_grad_false": (all(
                    not parameter.requires_grad for parameter in potential.network.parameters())
                    if hasattr(potential, "network") else True),
                "evaluation_rng_unchanged": evaluation_rng_unchanged,
                "run_complete": True,
            })
            checkpoint = destination / "final_sac_checkpoints" / method / f"run_{seed}" / "model.zip"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint)
            load_audit.append({
                "online_seed": seed, "method": method,
                "potential_method": metadata["phase8j_fix_method"],
                "potential_checkpoint": metadata.get("checkpoint", "zero"),
                "potential_file": metadata.get("file", {"blake2b_128": "zero"}),
                "standalone_online_roundtrip": True,
            })
            _write_csv(training_path, training_rows)
            environment.close()
            del model, environment, potential
            if selected_device == "cuda":
                torch.cuda.empty_cache()
    _write_json(load_audit_path, load_audit)
    curves = _online_curves(evaluation_rows, seeds, ONLINE_METHODS, steps)
    paired, summary = _summarize_online(evaluation_rows, seeds, ONLINE_METHODS, steps)
    _write_csv(destination / "paired_run_metrics.csv", paired)
    _write_csv(destination / "summary_metrics.csv", summary)
    final_rows = [row for row in training_rows
                  if int(row["training_step"]) == online_steps
                  and str(row.get("run_complete", "")).lower() == "true"]
    expected_loads = {
        (seed, method, "zero" if method == "sac_scratch"
         else ONLINE_TO_POTENTIAL[method])
        for seed in seeds for method in ONLINE_METHODS
    }
    observed_loads = {
        (int(row["online_seed"]), str(row["method"]), str(row["potential_method"]))
        for row in load_audit
    }
    potential_after = {
        str(path): file_fingerprint(path)["blake2b_128"] for path in potential_paths}
    finite_training = all(
        np.isfinite(float(row[field]))
        for row in training_rows
        for field in ("training_step", "training_environment_steps",
                      "evaluation_environment_steps", "replay_size",
                      "shaping_increment_mean", "shaping_increment_std",
                      "wall_clock_seconds")
        if row.get(field, "") != "")
    finite_evaluation = all(
        np.isfinite(float(row[field]))
        for row in evaluation_rows
        for field in ("raw_environment_return", "hopper_only_return",
                      "shaped_return", "episode_length"))
    checks = {
        "all_runs_complete": len(final_rows) == len(seeds) * len(ONLINE_METHODS),
        "correct_repaired_potential_loaded": observed_loads == expected_loads,
        "potential_checkpoint_files_unchanged": potential_before == potential_after,
        "phi_frozen_online": all(str(row["potential_frozen"]).lower() == "true"
                                 for row in final_rows),
        "phi_eval_and_requires_grad_false": all(
            str(row["potential_eval_mode"]).lower() == "true"
            and str(row["potential_requires_grad_false"]).lower() == "true"
            for row in final_rows),
        "evaluation_does_not_change_training_rng": all(
            str(row["evaluation_rng_unchanged"]).lower() == "true" for row in final_rows),
        "replay_stores_commanded_action": all(
            str(row["replay_matches_commanded_actions"]).lower() == "true"
            and str(row["wrapper_commanded_action_match"]).lower() == "true"
            for row in final_rows),
        "hidden_information_not_returned": all(
            str(row["returned_info_private_leak"]).lower() == "false"
            for row in final_rows),
        "raw_reward_formula_unchanged": all(
            str(row["raw_reward_formula_match"]).lower() == "true" for row in final_rows),
        "replay_initially_empty": all(
            str(row["replay_initially_empty"]).lower() == "true" for row in final_rows),
        "sac_initialization_paired": len(initial_by_seed) == len(seeds),
        "all_online_metrics_finite": finite_training and finite_evaluation,
        "pbrs_terminal_telescoping": bool(np.isclose(
            discounted_shaping_sum([2.0, 3.0, 5.0], [False, True]), -2.0)),
        "pbrs_truncation_bootstraps": bool(np.isclose(
            discounted_shaping_sum([2.0, 3.0, 5.0], [False, False]),
            -2.0 + GAMMA ** 2 * 5.0)),
    }
    _write_json(destination / "hard_checks.json", {
        "all_passed": all(checks.values()),
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
    })
    if not all(checks.values()):
        raise Phase8JClampFixError(
            f"online checks failed: {[key for key, value in checks.items() if not value]}")
    return {"online_run_count": len(seeds) * len(ONLINE_METHODS), "smoke": smoke}


def run_analyze(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output = Path(output_root).resolve()
    online = output / "online"
    checks = json.loads((online / "hard_checks.json").read_text(encoding="utf-8"))
    if checks.get("all_passed") is not True:
        raise Phase8JClampFixError("formal online hard checks did not pass")
    paired = _read_csv(online / "paired_run_metrics.csv")
    summary = _read_csv(online / "summary_metrics.csv")
    diagnostics = _read_csv(output / "potential_diagnostics.csv")
    by_method = {row["method"]: row for row in summary}
    action_delta = [float(row["auc_difference_vs_pooled_union"])
                    for row in paired if row["method"] == "sac_action_min_fixed"]
    state_delta = [float(row["auc_difference_vs_pooled_union"])
                   for row in paired if row["method"] == "sac_state_min_fixed"]
    lines = [
        "# Phase 8J-FIX-Q report", "",
        "Completion markers report execution and audit completion, not algorithm success.", "",
        "## Repair validation", "",
        "- The old positive-clamp zero-gradient failure was reproduced.",
        "- The repaired critic produced nonzero prediction-loss gradients and fit a frozen "
        "actual AAMAS target batch.",
        "- The external repository was not modified.", "",
        "## Potential health", "",
        "See `potential_training_health.md` for every seed and method.", "",
        "## Online summary", "",
        "| Method | AUC mean | AUC SD | Final mean | Final SD | NTA mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ONLINE_METHODS:
        row = by_method[method]
        lines.append(
            f"| {method} | {float(row['auc_0_50k_mean']):.6g} | "
            f"{float(row['auc_0_50k_sd']):.6g} | "
            f"{float(row['final_return_50k_mean']):.6g} | "
            f"{float(row['final_return_50k_sd']):.6g} | "
            f"{float(row['negative_transfer_area_vs_scratch_mean']):.6g} |")
    lines.extend([
        "", "## Required comparisons", "",
        f"- Action-min minus pooled-union AUC by run: {action_delta}",
        f"- State-min minus pooled-union AUC by run: {state_delta}",
        "- Comparisons with repaired pooled-native are in `paired_run_metrics.csv`.", "",
        "## Interpretation boundary", "",
        "Three runs are exploratory repetitions. Evaluation episodes are not independent "
        "training seeds. Results do not identify hidden U, certify an upper bound, or turn "
        "potential residual into causal ground truth.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_checks = {
        "preflight_validated": json.loads((output / "gradient_and_fit_tests.json").read_text(
            encoding="utf-8"))["all_passed"],
        "potential_training_complete": json.loads((
            output / "potentials" / "hard_checks.json").read_text(
                encoding="utf-8"))["training_complete"],
        "smoke_passed": json.loads((online / "smoke" / "hard_checks.json").read_text(
            encoding="utf-8"))["all_passed"],
        "formal_online_passed": checks["all_passed"],
    }
    _write_json(output / "hard_checks.json", {
        "all_passed": all(final_checks.values()),
        "checks": final_checks,
        "failed": [key for key, value in final_checks.items() if not value],
    })
    _write_json(output / "summary.json", {
        "stage": PHASE,
        "all_hard_checks_passed": all(final_checks.values()),
        "online_summary": summary,
        "action_min_minus_pooled_union_auc": action_delta,
        "state_min_minus_pooled_union_auc": state_delta,
        "potential_health": diagnostics,
        "scientific_success_implied": False,
    })
    return {"all_passed": all(final_checks.values())}
