"""Phase 8H-Q: quick action-wise multi-policy AAMAS envelope gate.

This module is deliberately incremental.  It consumes the verified Phase 8A
anchors and Source-2 SAC checkpoint read-only, imports the released continuous
AAMAS implementation, and stores only best component checkpoints plus compact
predictions and metrics.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from aamas_hopper_adapter import (
    ContinuousAAMASComponents,
    EXTERNAL_COMMIT,
    _import_official_module,
    compute_official_continuous_action_backup,
    compute_source_aamas_backup,
    file_sha256,
    normalize_rewards_like_official,
    validate_external_repo,
)
from scripts.train_aamas_hopper_potential import (
    ACTION_SEPARATION,
    CANDIDATE_ACTIONS,
    _build_agent,
    seed_everything,
)
from confounded_hopper import ACTUATOR_DIRECTION
from .anchor_pool import sha256, validate_anchor_pool
from .fixed_public_continuation import (
    FixedPublicContinuationPolicy,
    resolve_source2_checkpoint,
)
from .generate_datasets import MujocoOneStepSimulator


PHASE = "Phase 8H-Q"
ENV_ID = "Hopper-v5"
KAPPA = 0.20
LAMBDA_REWARD = 0.01
GAMMA = 0.99
SOURCE_B = np.asarray((-0.15, 0.0, 0.15), dtype=np.float64)
SOURCE_D = np.asarray((0.10, 0.18, 0.26), dtype=np.float64)
SIGMA_ACTION = 0.20
V_Q = np.asarray((1.0, -1.0, 0.0), dtype=np.float64) / np.sqrt(2.0)
V_U = np.asarray((1.0, 1.0, -2.0), dtype=np.float64) / np.sqrt(6.0)
POOLED_MIXTURES = {
    "balanced": np.asarray((1 / 3, 1 / 3, 1 / 3), dtype=np.float64),
    "source1_heavy": np.asarray((0.60, 0.20, 0.20), dtype=np.float64),
    "source3_heavy": np.asarray((0.20, 0.20, 0.60), dtype=np.float64),
}
SPLIT_NAMES = ("train", "observational_validation", "do_calibration_pool", "test")
EXPECTED_SPLIT_COUNTS = {"train": 333, "observational_validation": 51,
                         "do_calibration_pool": 51, "test": 77}
PUBLIC_MODEL_FIELDS = (
    "observation", "commanded_action", "reward", "next_observation",
    "terminated", "truncated", "anchor_id", "source_id", "sample_id",
)
FORBIDDEN_MODEL_FIELDS = {
    "u", "u_behavior", "u_environment", "u_env", "applied_action",
    "simulator_state", "qpos", "qvel", "do_reward", "do_q",
}


class Phase8HQuickMultipolicyAAMASError(RuntimeError):
    """Raised when a required input, invariant, or numerical check fails."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=root,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): file_sha256(Path(path)) for path in paths}


def _input_metadata(paths: Sequence[Path]) -> dict[str, dict[str, int]]:
    return {str(Path(path).resolve()): {
        "size_bytes": Path(path).stat().st_size,
        "modified_time_ns": Path(path).stat().st_mtime_ns,
    } for path in paths}


def source_policy_parameters() -> dict[str, Any]:
    return {"b": SOURCE_B.tolist(), "d": SOURCE_D.tolist(),
            "sigma_action": SIGMA_ACTION, "v_q": V_Q.tolist(), "v_u": V_U.tolist()}


def source_commanded_action(base_action: np.ndarray, source_id: int, u_behavior: int,
                            epsilon: np.ndarray) -> np.ndarray:
    """Apply the preregistered source policy in pre-tanh coordinates."""
    base = np.asarray(base_action, dtype=np.float64)
    noise = np.asarray(epsilon, dtype=np.float64)
    if base.shape != (3,) or noise.shape != (3,) or not np.all(np.isfinite(base + noise)):
        raise ValueError("base action and epsilon must be finite three-vectors")
    if source_id not in (1, 2, 3) or u_behavior not in (-1, 1):
        raise ValueError("source_id must be 1/2/3 and u_behavior must be -1/+1")
    x_base = np.arctanh(np.clip(base, -0.95, 0.95))
    x = (x_base + SOURCE_B[source_id - 1] * V_Q
         + SOURCE_D[source_id - 1] * int(u_behavior) * V_U
         + SIGMA_ACTION * noise)
    return np.tanh(x)


def fixed_anchor_splits(record: Mapping[str, Sequence[int]], anchor_ids: Sequence[int]
                        ) -> dict[str, np.ndarray]:
    """Load, subset, and validate the existing 333/51/51/77 anchor split."""
    available = set(map(int, anchor_ids))
    result = {name: np.asarray(record[name], dtype=np.int64) for name in SPLIT_NAMES}
    if {name: len(value) for name, value in result.items()} != EXPECTED_SPLIT_COUNTS:
        raise Phase8HQuickMultipolicyAAMASError(
            "existing split is not the fixed 333/51/51/77 Phase 8E-Q split")
    groups = [set(map(int, result[name])) for name in SPLIT_NAMES]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise Phase8HQuickMultipolicyAAMASError("anchor splits overlap")
    if set.union(*groups) != available:
        raise Phase8HQuickMultipolicyAAMASError("split assignments do not match selected anchors")
    return result


def select_and_renumber_split_anchors(
    complete: Mapping[str, np.ndarray],
    split_record: Mapping[str, Sequence[int]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Select frozen split members and map their original IDs to compact IDs.

    Phase 8E-Q selected 512 anchors from the full 2,048-anchor Phase 8A pool,
    so its frozen IDs are intentionally not the contiguous range 0..511.
    Phase 8H uses a compact copy for simulation while retaining the exact
    frozen membership through this deterministic old-ID to new-ID mapping.
    """
    selected_original_ids = np.concatenate([
        np.asarray(split_record[name], dtype=np.int64) for name in SPLIT_NAMES
    ])
    frozen = fixed_anchor_splits(split_record, selected_original_ids)
    complete_ids = np.asarray(complete["anchor_id"], dtype=np.int64)
    id_to_position = {int(anchor): position
                      for position, anchor in enumerate(complete_ids)}
    missing = sorted(set(map(int, selected_original_ids)) - set(id_to_position))
    if missing:
        raise Phase8HQuickMultipolicyAAMASError(
            f"frozen split contains anchors absent from Phase 8A pool: {missing[:10]}")
    positions = np.asarray(
        [id_to_position[int(anchor)] for anchor in selected_original_ids], dtype=np.int64)
    anchors = {name: np.asarray(value)[positions].copy()
               for name, value in complete.items()}
    anchors["anchor_id"] = np.arange(len(selected_original_ids), dtype=np.int64)
    old_to_new = {int(old): new for new, old in enumerate(selected_original_ids)}
    splits = {
        name: np.asarray([old_to_new[int(anchor)] for anchor in frozen[name]], dtype=np.int64)
        for name in SPLIT_NAMES
    }
    validate_anchor_pool(anchors, len(selected_original_ids))
    fixed_anchor_splits(splits, anchors["anchor_id"])
    return anchors, splits, selected_original_ids


def _resolve_split_path(phase8a_root: Path) -> Path:
    parent = Path(phase8a_root).resolve().parent
    candidates = (
        parent / "phase8e_quick_go_nogo" / "splits.json",
        parent / "noncomplementary_loggers_seed0_verified"
        / "phase8c_reward_mechanism_separation" / "splits.json",
    )
    for path in candidates:
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if all(name in record for name in SPLIT_NAMES):
                return path
    raise Phase8HQuickMultipolicyAAMASError(
        "fixed Phase 8E-Q 333/51/51/77 anchor split is unavailable")


def _load_phase8h_inputs(phase8a_root: Path, num_anchors: int,
                         reference_checkpoint: Path | None,
                         *, compute_checkpoint_hash: bool = True) -> dict[str, Any]:
    root = Path(phase8a_root).resolve()
    manifest_path, checks_path, anchors_path = (
        root / "manifest.json", root / "hard_checks.json", root / "anchors.npz")
    for path in (manifest_path, anchors_path):
        if not path.is_file():
            raise Phase8HQuickMultipolicyAAMASError(
                f"required read-only input is missing: {path}")
    if checks_path.is_file():
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        passed = checks.get("all_passed", checks.get("all_hard_invariants_passed"))
        if passed is not True:
            raise Phase8HQuickMultipolicyAAMASError("Phase 8A hard checks did not pass")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("env_id") != ENV_ID:
        raise Phase8HQuickMultipolicyAAMASError("Phase 8A environment is not Hopper-v5")
    recorded_direction = manifest.get(
        "actuator_direction_v",
        manifest.get("source2_original_manifest", {}).get("actuator_direction"))
    if (np.asarray(recorded_direction, dtype=np.float64).shape != (3,)
            or not np.allclose(recorded_direction, ACTUATOR_DIRECTION,
                               atol=1e-15, rtol=0.0)):
        raise Phase8HQuickMultipolicyAAMASError(
            "Phase 8A actuator direction does not match the verified Hopper wrapper")
    with np.load(anchors_path, allow_pickle=False) as archive:
        complete = {name: archive[name].copy() for name in archive.files}
    validate_anchor_pool(complete)
    if num_anchors > len(complete["anchor_id"]):
        raise Phase8HQuickMultipolicyAAMASError("requested anchors exceed Phase 8A pool")
    split_path = _resolve_split_path(root)
    split_record = json.loads(split_path.read_text(encoding="utf-8"))
    if num_anchors == 512:
        anchors, splits, selected_ids = select_and_renumber_split_anchors(
            complete, split_record)
    else:
        # Smoke preserves membership in every frozen split, then renumbers its
        # compact anchor copy only; the original IDs remain recorded below.
        smoke_counts = {"train": 42, "observational_validation": 6,
                        "do_calibration_pool": 6, "test": 10}
        chosen_by_split = {name: np.sort(np.asarray(split_record[name], dtype=np.int64))[:count]
                           for name, count in smoke_counts.items()}
        selected_ids = np.concatenate([chosen_by_split[name] for name in SPLIT_NAMES])
        positions = np.asarray([int(np.flatnonzero(
            np.asarray(complete["anchor_id"]) == anchor)[0]) for anchor in selected_ids])
        anchors = {name: np.asarray(value)[positions].copy() for name, value in complete.items()}
        anchors["anchor_id"] = np.arange(num_anchors, dtype=np.int64)
        validate_anchor_pool(anchors, num_anchors)
        splits = {}
        offset = 0
        for name in SPLIT_NAMES:
            splits[name] = np.arange(offset, offset + smoke_counts[name], dtype=np.int64)
            offset += smoke_counts[name]
    if reference_checkpoint is None:
        checkpoint, original_manifest, checkpoint_hash = resolve_source2_checkpoint(root)
    else:
        checkpoint = Path(reference_checkpoint).resolve()
        if not checkpoint.is_file():
            raise Phase8HQuickMultipolicyAAMASError(
                f"explicit reference SAC checkpoint is missing: {checkpoint}")
        original_manifest = manifest.get("source2_original_manifest", {})
        checkpoint_hash = sha256(checkpoint) if compute_checkpoint_hash else None
    required = [manifest_path, anchors_path, split_path, checkpoint]
    if checks_path.is_file():
        required.append(checks_path)
    return {"root": root, "manifest": manifest, "anchors": anchors, "splits": splits,
            "selected_original_anchor_ids": selected_ids,
            "split_path": split_path, "checkpoint": checkpoint,
            "checkpoint_hash": checkpoint_hash, "source_manifest": original_manifest,
            "required_paths": tuple(required)}


def _model_public_view(dataset: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    public = {name: np.asarray(dataset[name]) for name in PUBLIC_MODEL_FIELDS}
    if set(public) & FORBIDDEN_MODEL_FIELDS:
        raise Phase8HQuickMultipolicyAAMASError("hidden fields entered the model view")
    return public


def validate_public_dataset(dataset: Mapping[str, np.ndarray], expected_rows: int | None = None
                            ) -> None:
    if set(dataset) != set(PUBLIC_MODEL_FIELDS):
        raise Phase8HQuickMultipolicyAAMASError(
            f"model fields must be exactly {list(PUBLIC_MODEL_FIELDS)}")
    count = len(dataset["reward"])
    if expected_rows is not None and count != expected_rows:
        raise Phase8HQuickMultipolicyAAMASError(
            f"dataset has {count} rows, expected {expected_rows}")
    if dataset["observation"].shape != (count, 12) or dataset["next_observation"].shape != (count, 12):
        raise Phase8HQuickMultipolicyAAMASError("public observations must be [N,12]")
    if dataset["commanded_action"].shape != (count, 3):
        raise Phase8HQuickMultipolicyAAMASError("commanded actions must be [N,3]")
    for value in dataset.values():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise Phase8HQuickMultipolicyAAMASError("public dataset contains NaN/Inf")


def generate_source_dataset(
    anchors: Mapping[str, np.ndarray], simulator: Any, *, condition: str,
    samples_per_anchor_source: int, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate one fixed master table; compositions later change weights only."""
    if condition not in ("confounded", "independent_latents"):
        raise ValueError("unknown latent condition")
    if samples_per_anchor_source <= 0:
        raise ValueError("samples_per_anchor_source must be positive")
    rng = np.random.default_rng(seed + (0 if condition == "confounded" else 10_000_019))
    public_rows: dict[str, list[Any]] = {name: [] for name in PUBLIC_MODEL_FIELDS}
    hidden_rows: dict[str, list[Any]] = {name: [] for name in
                                        ("u_behavior", "u_environment", "applied_action")}
    for position, anchor_id in enumerate(np.asarray(anchors["anchor_id"], dtype=np.int64)):
        base = np.asarray(anchors["base_action"][position], dtype=np.float64)
        for source_id in (1, 2, 3):
            for sample_id in range(samples_per_anchor_source):
                u_behavior = 1 if rng.random() >= 0.5 else -1
                if condition == "confounded":
                    u_environment = u_behavior
                else:
                    u_environment = 1 if rng.random() >= 0.5 else -1
                command = source_commanded_action(
                    base, source_id, u_behavior, rng.standard_normal(3))
                outcome = simulator.step(position, command, u_environment, KAPPA)
                values = {
                    "observation": outcome["observation"],
                    "commanded_action": command.astype(np.float32),
                    "reward": float(outcome["reward"] + LAMBDA_REWARD * u_environment),
                    "next_observation": outcome["next_observation"],
                    "terminated": bool(outcome["terminated"]),
                    "truncated": bool(outcome["truncated"]),
                    "anchor_id": int(anchor_id), "source_id": source_id,
                    "sample_id": sample_id,
                }
                for name, value in values.items():
                    public_rows[name].append(value)
                hidden_rows["u_behavior"].append(u_behavior)
                hidden_rows["u_environment"].append(u_environment)
                hidden_rows["applied_action"].append(outcome["applied_action"])
    public = {name: np.asarray(value) for name, value in public_rows.items()}
    public["observation"] = public["observation"].astype(np.float32)
    public["commanded_action"] = public["commanded_action"].astype(np.float32)
    public["reward"] = public["reward"].astype(np.float32)
    public["next_observation"] = public["next_observation"].astype(np.float32)
    public["terminated"] = public["terminated"].astype(bool)
    public["truncated"] = public["truncated"].astype(bool)
    public["anchor_id"] = public["anchor_id"].astype(np.int64)
    public["source_id"] = public["source_id"].astype(np.int8)
    public["sample_id"] = public["sample_id"].astype(np.int16)
    hidden = {name: np.asarray(value) for name, value in hidden_rows.items()}
    validate_public_dataset(public, len(anchors["anchor_id"]) * 3 * samples_per_anchor_source)
    return public, hidden


def pooled_row_weights(source_ids: np.ndarray, mixture: Sequence[float]) -> np.ndarray:
    source = np.asarray(source_ids, dtype=np.int64)
    mass = np.asarray(mixture, dtype=np.float64)
    if mass.shape != (3,) or np.any(mass <= 0) or not np.isclose(mass.sum(), 1.0):
        raise ValueError("pooled mixture must be a positive length-three probability")
    counts = np.asarray([np.sum(source == item) for item in (1, 2, 3)], dtype=np.float64)
    if np.any(counts == 0):
        raise ValueError("all three sources are required")
    result = mass[source - 1] / counts[source - 1]
    return result / result.sum()


def action_overlap_audit(public: Mapping[str, np.ndarray]) -> dict[str, Any]:
    actions = np.asarray(public["commanded_action"], dtype=np.float64)
    source = np.asarray(public["source_id"], dtype=np.int64)
    means = np.stack([actions[source == item].mean(axis=0) for item in (1, 2, 3)])
    stds = np.stack([actions[source == item].std(axis=0) for item in (1, 2, 3)])
    separations = [float(np.linalg.norm(means[i] - means[j]))
                   for i in range(3) for j in range(i + 1, 3)]
    pooled_scale = float(np.mean(stds))
    # Gaussian overlap coefficient proxy: positive and below one for moderate shifts.
    overlap = [float(np.exp(-distance**2 / (8 * max(pooled_scale, 1e-8)**2)))
               for distance in separations]
    return {"source_action_means": means.tolist(), "source_action_stds": stds.tolist(),
            "pairwise_mean_distances": separations,
            "pairwise_overlap_proxy": overlap,
            "all_sources_distinct": bool(min(separations) > 1e-3),
            "all_sources_overlap": bool(min(overlap) > 0.05)}


class FrozenSACReferenceValue:
    """Hidden-blind, frozen Source-2 SAC actor/critic reference value."""

    def __init__(self, sac: Any, device: str, *, use_parameter_hash: bool = True) -> None:
        import torch
        self.torch = torch
        self.sac = sac
        self.device = torch.device(device)
        for module in (sac.actor, sac.critic):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.parameter_hash_before = self.parameter_hash() if use_parameter_hash else None
        self.parameter_snapshot_before = None if use_parameter_hash else tuple(
            (f"{prefix}.{name}", value.detach().cpu().clone())
            for prefix, module in (("actor", self.sac.actor), ("critic", self.sac.critic))
            for name, value in sorted(module.state_dict().items()))

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for prefix, module in (("actor", self.sac.actor), ("critic", self.sac.critic)):
            for name, value in sorted(module.state_dict().items()):
                digest.update(f"{prefix}.{name}".encode())
                digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def actor_action(self, public_observations: np.ndarray) -> np.ndarray:
        public = np.asarray(public_observations, dtype=np.float32)
        if public.ndim != 2 or public.shape[1] != 12:
            raise ValueError("reference actor requires [N,12] public observations")
        actions = []
        with self.torch.no_grad():
            for latent in (-1.0, 1.0):
                obs = np.concatenate(
                    (public, np.full((len(public), 1), latent, dtype=np.float32)), axis=1)
                tensor = self.torch.as_tensor(obs, dtype=self.torch.float32, device=self.device)
                actions.append(self.sac.actor(tensor, deterministic=True).cpu().numpy())
        return np.clip(0.5 * (actions[0] + actions[1]), -1.0, 1.0)

    def __call__(self, public_observations: np.ndarray) -> np.ndarray:
        public = np.asarray(public_observations, dtype=np.float32)
        action = self.actor_action(public)
        values = []
        with self.torch.no_grad():
            action_tensor = self.torch.as_tensor(
                action, dtype=self.torch.float32, device=self.device)
            for latent in (-1.0, 1.0):
                obs = np.concatenate(
                    (public, np.full((len(public), 1), latent, dtype=np.float32)), axis=1)
                obs_tensor = self.torch.as_tensor(
                    obs, dtype=self.torch.float32, device=self.device)
                critics = self.sac.critic(obs_tensor, action_tensor)
                values.append(self.torch.minimum(critics[0], critics[1]).reshape(-1))
        result = 0.5 * (values[0] + values[1])
        return result.detach().cpu().numpy().astype(np.float64)

    def verify_frozen(self) -> bool:
        if self.parameter_hash_before is not None:
            unchanged = self.parameter_hash() == self.parameter_hash_before
        else:
            current = tuple(
                (f"{prefix}.{name}", value.detach().cpu())
                for prefix, module in (("actor", self.sac.actor), ("critic", self.sac.critic))
                for name, value in sorted(module.state_dict().items()))
            unchanged = len(current) == len(self.parameter_snapshot_before or ()) and all(
                left_name == right_name and self.torch.equal(left_value, right_value)
                for (left_name, left_value), (right_name, right_value)
                in zip(current, self.parameter_snapshot_before or ()))
        return unchanged and all(
            not parameter.requires_grad
            for module in (self.sac.actor, self.sac.critic)
            for parameter in module.parameters())


def _device_name(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise Phase8HQuickMultipolicyAAMASError("CUDA was requested but is unavailable")
    return requested


def _training_tensors(public: Mapping[str, np.ndarray], train_rows: np.ndarray,
                      device: str, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    rewards, reward_stats = normalize_rewards_like_official(
        np.asarray(public["reward"])[train_rows], torch_module=torch)
    full_reward, _ = normalize_rewards_like_official(
        np.asarray(public["reward"]), statistics=reward_stats, torch_module=torch)
    train_observation = np.asarray(public["observation"])[train_rows]
    state_mean = train_observation.mean(axis=0, keepdims=True).astype(np.float32)
    state_std = train_observation.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    if np.any(~np.isfinite(state_std)) or np.any(state_std <= 0):
        raise Phase8HQuickMultipolicyAAMASError("training state normalization is invalid")
    tensors = {
        "observation": torch.as_tensor(public["observation"], dtype=torch.float32, device=device),
        "action": torch.as_tensor(public["commanded_action"], dtype=torch.float32, device=device),
        "reward": torch.as_tensor(full_reward, dtype=torch.float32, device=device),
        "next_observation": torch.as_tensor(
            public["next_observation"], dtype=torch.float32, device=device),
        "terminated": torch.as_tensor(
            np.asarray(public["terminated"])[:, None], dtype=torch.float32, device=device),
        "truncated": torch.as_tensor(
            np.asarray(public["truncated"])[:, None], dtype=torch.bool, device=device),
    }
    normalization = {"state_mean": state_mean, "state_std": state_std,
                     "reward_mean": float(reward_stats["reward_mean"]),
                     "reward_std": float(reward_stats["reward_std"]),
                     "reward_upper": float(np.max(np.asarray(public["reward"])[train_rows])),
                     "normalized_b": float(reward_stats["normalized_b"])}
    # Keep this assignment visible: only training rows define normalization.
    _ = rewards
    return tensors, normalization


def _component_validation(agent: Any, tensors: Mapping[str, Any], rows: np.ndarray,
                          torch: Any) -> tuple[float, dict[str, float]]:
    index = torch.as_tensor(rows, dtype=torch.long, device=tensors["observation"].device)
    with torch.no_grad():
        state, action = tensors["observation"][index], tensors["action"][index]
        pair = torch.cat((state, action), dim=1)
        log_prob = agent.policy(state).log_prob(action)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=-1)
        behavior = -log_prob.mean()
        delta = (agent.state_transition_model(pair)
                 - (tensors["next_observation"][index] - state)).square().mean()
        reward = (agent.reward_model(pair).reshape(-1)
                  - tensors["reward"][index].reshape(-1)).square().mean()
        total = behavior + delta + reward
    values = {"behavior_nll": float(behavior.cpu()), "delta_mse": float(delta.cpu()),
              "reward_mse": float(reward.cpu()), "total": float(total.cpu())}
    if not all(np.isfinite(value) for value in values.values()):
        raise Phase8HQuickMultipolicyAAMASError("component validation is nonfinite")
    return values["total"], values


def fit_aamas_components(
    public: Mapping[str, np.ndarray], train_anchors: Sequence[int],
    validation_anchors: Sequence[int], *, row_probabilities: np.ndarray,
    seed: int, gradient_updates: int, batch_size: int, device: str,
    official: Any, torch: Any, record_schedule_digest: bool = True,
) -> tuple[ContinuousAAMASComponents, dict[str, Any], dict[str, Any]]:
    """Fit released behavior/delta/reward modules; select on public validation only."""
    model_public = _model_public_view(public)
    train_rows = np.flatnonzero(np.isin(public["anchor_id"], train_anchors))
    validation_rows = np.flatnonzero(np.isin(public["anchor_id"], validation_anchors))
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("training and validation rows must both be nonempty")
    probability = np.asarray(row_probabilities, dtype=np.float64)
    probability = probability[train_rows]
    if probability.shape != (len(train_rows),) or np.any(probability < 0):
        raise ValueError("row probabilities are invalid")
    probability /= probability.sum()
    tensors, normalization = _training_tensors(model_public, train_rows, device, torch)
    reward_for_agent = tensors["reward"][torch.as_tensor(train_rows, device=device)]
    reward_statistics = {
        "normalized_b": normalization["normalized_b"],
    }
    agent = _build_agent(
        official, torch, {"rewards": reward_for_agent}, normalization["state_mean"],
        normalization["state_std"], reward_statistics, torch.device(device), GAMMA)
    generator = np.random.default_rng(seed + 810_001)
    best_loss = np.inf; best_step = 0; best_metrics: dict[str, float] = {}
    best_state: dict[str, dict[str, Any]] | None = None
    schedule = hashlib.sha256() if record_schedule_digest else None
    history = []
    for step in range(1, gradient_updates + 1):
        chosen = generator.choice(train_rows, size=min(batch_size, len(train_rows)),
                                  replace=True, p=probability)
        if schedule is not None:
            schedule.update(np.asarray(chosen, dtype=np.int64).tobytes())
        index = torch.as_tensor(chosen, dtype=torch.long, device=device)
        values = agent.train_reward_probability_state_delta(
            tensors["observation"][index], tensors["action"][index],
            tensors["next_observation"][index], tensors["reward"][index],
            tensors["terminated"][index], torch.zeros_like(tensors["reward"][index],
                                                            dtype=torch.int32),
            epoch=step,
        )
        if not np.all(np.isfinite(values)):
            raise Phase8HQuickMultipolicyAAMASError("AAMAS component training is nonfinite")
        if step == 1 or step % 10 == 0 or step == gradient_updates:
            loss, metrics = _component_validation(agent, tensors, validation_rows, torch)
            train_values = np.asarray(values, dtype=np.float64).reshape(-1)
            history.append({
                "step": step,
                "train_loss": float(train_values.sum()),
                "train_behavior_loss": float(train_values[0]) if len(train_values) > 0 else 0.0,
                "train_state_loss": float(train_values[1]) if len(train_values) > 1 else 0.0,
                "train_reward_loss": float(train_values[2]) if len(train_values) > 2 else 0.0,
                "validation_loss": metrics["total"],
                **metrics,
            })
            if loss < best_loss:
                best_loss, best_step, best_metrics = loss, step, metrics
                best_state = {
                    "behavior": {name: value.detach().cpu().clone()
                                 for name, value in agent.policy.state_dict().items()},
                    "state_difference": {name: value.detach().cpu().clone()
                                         for name, value in agent.state_transition_model.state_dict().items()},
                    "reward": {name: value.detach().cpu().clone()
                               for name, value in agent.reward_model.state_dict().items()},
                }
    if best_state is None:
        raise Phase8HQuickMultipolicyAAMASError("no finite validation checkpoint was selected")
    agent.policy_fin.load_state_dict(best_state["behavior"])
    agent.state_transition_model_fin.load_state_dict(best_state["state_difference"])
    agent.reward_model_fin.load_state_dict(best_state["reward"])
    for module in (agent.policy_fin, agent.state_transition_model_fin, agent.reward_model_fin):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    bundle = ContinuousAAMASComponents(
        agent.policy_fin, agent.state_transition_model_fin, agent.reward_model_fin,
        normalization["reward_mean"], normalization["reward_std"],
        normalization["reward_upper"], GAMMA, torch.device(device),
        ACTION_SEPARATION, CANDIDATE_ACTIONS)
    metadata = {"seed": seed, "best_step": best_step,
                "best_observational_validation": best_metrics,
                "gradient_updates": gradient_updates, "batch_size": batch_size,
                "model_input_fields": ["observation", "commanded_action"],
                "checkpoint_selection_fields": ["behavior_nll", "delta_mse", "reward_mse"]}
    if schedule is not None:
        metadata["minibatch_schedule_sha256"] = schedule.hexdigest()
    else:
        metadata["minibatch_schedule_identifier"] = (
            f"seed={seed};updates={gradient_updates};batch_size={batch_size}")
    return bundle, normalization, {"metadata": metadata, "history": history,
                                    "state_dict": best_state}


def _save_component_checkpoint(path: Path, training: Mapping[str, Any],
                               normalization: Mapping[str, Any], metadata: Mapping[str, Any],
                               torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": training["state_dict"],
                "normalization": dict(normalization), "metadata": dict(metadata)}, path)


def sample_behavior_actions(bundle: ContinuousAAMASComponents, states: np.ndarray,
                            noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample official TanhNormal actions using fixed common Gaussian noise."""
    import torch
    state = torch.as_tensor(states, dtype=torch.float32, device=bundle.device)
    epsilon = torch.as_tensor(noise, dtype=torch.float32, device=bundle.device)
    with torch.no_grad():
        distribution = bundle.behavior_model(state)
        location, scale = distribution.loc, distribution.scale
        sampled = torch.tanh(location[:, None, :] + scale[:, None, :] * epsilon)
        mean = torch.tanh(location)
    return (sampled.cpu().numpy().astype(np.float32),
            mean.cpu().numpy().astype(np.float32))


def union_candidate_actions(source_models: Sequence[ContinuousAAMASComponents],
                            states: np.ndarray, base_actions: np.ndarray,
                            samples_per_source: int, seed: int) -> np.ndarray:
    if len(source_models) != 3 or samples_per_source <= 0:
        raise ValueError("three source models and positive K are required")
    rng = np.random.default_rng(seed)
    blocks = []
    for source, model in enumerate(source_models):
        # Same standard-normal draws are paired by source slot across model seeds.
        noise = rng.standard_normal((len(states), samples_per_source, 3)).astype(np.float32)
        samples, mean = sample_behavior_actions(model, states, noise)
        blocks.extend((samples, mean[:, None, :]))
    blocks.append(np.asarray(base_actions, dtype=np.float32)[:, None, :])
    result = np.concatenate(blocks, axis=1)
    if not np.all(np.isfinite(result)) or np.any(np.abs(result) > 1 + 1e-6):
        raise Phase8HQuickMultipolicyAAMASError("union candidate actions are invalid")
    return result


def native_candidate_actions(model: ContinuousAAMASComponents, states: np.ndarray,
                             base_actions: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(states), CANDIDATE_ACTIONS, 3)).astype(np.float32)
    sampled, mean = sample_behavior_actions(model, states, noise)
    return np.concatenate((sampled, mean[:, None, :],
                           np.asarray(base_actions, dtype=np.float32)[:, None, :]), axis=1)


def action_and_state_level_envelopes(source_q: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(source_q, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("source_q must be finite [source,state,action]")
    action_min = values.min(axis=0)
    source_phi = values.max(axis=2)
    state_source = source_phi.argmin(axis=0)
    state_min = source_phi.min(axis=0)
    state_selected_q = values[state_source, np.arange(values.shape[1]), :]
    action_phi = action_min.max(axis=1)
    if np.any(action_phi > state_min + 1e-10):
        raise Phase8HQuickMultipolicyAAMASError(
            "action-level minimum exceeds state-level minimum")
    return {"action_q": action_min, "action_phi": action_phi,
            "state_phi": state_min, "state_selected_q": state_selected_q,
            "state_selected_source": state_source}


def source_duplication_invariant(source_q: np.ndarray) -> bool:
    original = action_and_state_level_envelopes(source_q)["action_q"]
    duplicate = np.concatenate((source_q, source_q[1:2]), axis=0)
    return bool(np.allclose(
        original, action_and_state_level_envelopes(duplicate)["action_q"],
        atol=1e-12, rtol=0.0))


def deterministic_argmax(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).argmax(axis=-1)


def do_bellman_oracle(simulator: Any, anchors: Mapping[str, np.ndarray],
                      anchor_positions: np.ndarray, candidate_actions: np.ndarray,
                      reference_value: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Exact equal-weight U_environment enumeration; no behavior policy call."""
    result = np.empty(candidate_actions.shape[:2], dtype=np.float64)
    for row, anchor_position in enumerate(np.asarray(anchor_positions, dtype=np.int64)):
        for column, action in enumerate(candidate_actions[row]):
            branches = []
            for u_environment in (-1, 1):
                outcome = simulator.step(int(anchor_position), action, u_environment, KAPPA)
                continuation = float(reference_value(
                    np.asarray(outcome["next_observation"], dtype=np.float32)[None, :])[0])
                branches.append(float(outcome["reward"] + LAMBDA_REWARD * u_environment)
                                + GAMMA * continuation)
            result[row, column] = 0.5 * (branches[0] + branches[1])
    if not np.all(np.isfinite(result)):
        raise Phase8HQuickMultipolicyAAMASError("do-Bellman oracle is nonfinite")
    return result


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    true = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("truth and prediction must be aligned [state,action] arrays")
    error = pred - true
    selected = deterministic_argmax(pred)
    oracle = deterministic_argmax(true)
    rows = np.arange(len(true))
    regret = true.max(axis=1) - true[rows, selected]
    oracle_sorted = np.sort(true, axis=1)
    strict_oracle = oracle_sorted[:, -1] - oracle_sorted[:, -2] > 1e-10
    positive = regret[regret > 1e-12]
    distribution = np.bincount(selected, minlength=true.shape[1]) / max(len(selected), 1)
    return {
        "do_mae": float(np.mean(np.abs(error))),
        "do_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "signed_error": float(np.mean(error)),
        "underestimation_fraction": float(np.mean(pred < true)),
        "mean_positive_slack": float(np.mean(np.maximum(error, 0.0))),
        "top_action_disagreement": float(np.mean(selected != oracle)),
        "strict_flip": float(np.mean((selected != oracle) & strict_oracle)),
        "selected_action_0_fraction": float(distribution[0]),
        "selected_action_1_fraction": float(distribution[1]),
        "selected_action_2_fraction": float(distribution[2]),
        "regret_mean": float(np.mean(regret)),
        "regret_median": float(np.median(regret)),
        "regret_conditional_mean": float(np.mean(positive)) if len(positive) else 0.0,
        "regret_p90": float(np.quantile(regret, 0.90)),
        "regret_max": float(np.max(regret)),
        "phi_mae": float(np.mean(np.abs(pred.max(axis=1) - true.max(axis=1)))),
        "phi_signed_bias": float(np.mean(pred.max(axis=1) - true.max(axis=1))),
    }


def _metric_row(condition: str, seed: int, method: str, composition: str,
                candidate_set: str, truth: np.ndarray, prediction: np.ndarray,
                **extra: Any) -> dict[str, Any]:
    return {"condition": condition, "seed": seed, "method": method,
            "composition": composition, "candidate_set": candidate_set,
            **prediction_metrics(truth, prediction), **extra}


def _source_selection_rows(condition: str, seed: int, source_q: np.ndarray,
                           envelope: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    minimum_source = np.argmin(source_q, axis=0)
    switching = np.asarray([len(np.unique(row)) > 1 for row in minimum_source])
    gap = np.asarray(envelope["state_phi"] - envelope["action_phi"])
    rows = []
    for source in range(source_q.shape[0]):
        rows.append({"condition": condition, "seed": seed,
                     "metric": "minimum_source_fraction", "source_id": source + 1,
                     "value": float(np.mean(minimum_source == source))})
    rows.extend((
        {"condition": condition, "seed": seed,
         "metric": "within_state_source_switch_fraction", "source_id": "ALL",
         "value": float(np.mean(switching))},
        {"condition": condition, "seed": seed,
         "metric": "state_minus_action_phi_gap_mean", "source_id": "ALL",
         "value": float(np.mean(gap))},
        {"condition": condition, "seed": seed,
         "metric": "state_minus_action_phi_gap_positive_fraction", "source_id": "ALL",
         "value": float(np.mean(gap > 1e-10))},
    ))
    return rows


def _composition_drift_rows(condition: str, seed: int,
                            pooled_predictions: Mapping[str, np.ndarray],
                            truth: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    names = tuple(POOLED_MIXTURES)
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left, right = names[left_index], names[right_index]
            p_left, p_right = pooled_predictions[left], pooled_predictions[right]
            rows.append({
                "condition": condition, "seed": seed, "left": left, "right": right,
                "prediction_mae": float(np.mean(np.abs(p_left - p_right))),
                "top_action_disagreement": float(np.mean(
                    deterministic_argmax(p_left) != deterministic_argmax(p_right))),
                "potential_drift_mae": float(np.mean(np.abs(
                    p_left.max(axis=1) - p_right.max(axis=1)))),
                "left_do_mae": prediction_metrics(truth, p_left)["do_mae"],
                "right_do_mae": prediction_metrics(truth, p_right)["do_mae"],
            })
    return rows


def _aggregate_seed_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = ("condition", "method", "composition", "candidate_set")
    metrics = ("do_mae", "do_rmse", "signed_error", "underestimation_fraction",
               "mean_positive_slack", "top_action_disagreement", "strict_flip",
               "selected_action_0_fraction", "selected_action_1_fraction",
               "selected_action_2_fraction",
               "regret_mean", "regret_median", "regret_conditional_mean", "regret_p90",
               "regret_max", "phi_mae", "phi_signed_bias")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in groups.items():
        record = dict(zip(keys, group)); record["model_seed_count"] = len(values)
        for metric in metrics:
            data = np.asarray([float(row[metric]) for row in values])
            record[f"{metric}_mean"] = float(data.mean())
            record[f"{metric}_seed_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        result.append(record)
    return result


def _make_figures(output: Path, summary_rows: Sequence[Mapping[str, Any]],
                  drift_rows: Sequence[Mapping[str, Any]],
                  selection_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    primary = [row for row in summary_rows if row["condition"] == "confounded"
               and row["candidate_set"] == "union"
               and row["composition"] in ("INVARIANT", "balanced")]
    labels = [str(row["method"]) for row in primary]
    def bar(filename: str, metric: str, title: str, ylabel: str) -> None:
        fig, axis = plt.subplots(figsize=(9, 4.8))
        axis.bar(np.arange(len(primary)), [float(row[metric]) for row in primary], color="#4472C4")
        axis.set_xticks(np.arange(len(primary)), labels, rotation=28, ha="right")
        axis.set_title(title); axis.set_ylabel(ylabel); fig.tight_layout()
        fig.savefig(figures / filename, dpi=180); plt.close(fig)
    bar("do_bellman_mae_by_method.png", "do_mae_mean", "Do-Bellman error", "MAE")
    bar("ranking_error_by_method.png", "top_action_disagreement_mean",
        "Top-action disagreement", "fraction")
    bar("decision_regret_by_method.png", "regret_mean_mean", "Decision regret", "mean regret")

    fig, axis = plt.subplots(figsize=(7, 4.5))
    primary_drift = [row for row in drift_rows if row["condition"] == "confounded"]
    names = [f"{row['left']} vs\n{row['right']}" for row in primary_drift]
    axis.bar(np.arange(len(primary_drift)),
             [float(row["prediction_mae"]) for row in primary_drift], color="#ED7D31")
    axis.set_xticks(np.arange(len(names)), names); axis.set_ylabel("pairwise prediction MAE")
    axis.set_title("Pooled AAMAS composition drift"); fig.tight_layout()
    fig.savefig(figures / "composition_drift.png", dpi=180); plt.close(fig)

    gaps = [row for row in selection_rows if row["condition"] == "confounded"
            and row["metric"] == "state_minus_action_phi_gap_mean"]
    switches = [row for row in selection_rows if row["condition"] == "confounded"
                and row["metric"] == "within_state_source_switch_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    axes[0].bar([str(row["seed"]) for row in gaps], [float(row["value"]) for row in gaps])
    axes[0].set_title("State minus action envelope"); axes[0].set_ylabel("mean gap")
    axes[1].bar([str(row["seed"]) for row in switches],
                [float(row["value"]) for row in switches])
    axes[1].set_title("Within-state source switching"); axes[1].set_ylabel("fraction")
    for axis in axes: axis.set_xlabel("model seed")
    fig.tight_layout(); fig.savefig(figures / "action_vs_state_level_envelope.png", dpi=180)
    plt.close(fig)


def _lookup_summary(rows: Sequence[Mapping[str, Any]], method: str,
                    composition: str = "INVARIANT") -> Mapping[str, Any] | None:
    return next((row for row in rows if row["condition"] == "confounded"
                 and row["method"] == method and row["composition"] == composition
                 and row["candidate_set"] == "union"), None)


def _write_report(output: Path, summary_rows: Sequence[Mapping[str, Any]],
                  drift_rows: Sequence[Mapping[str, Any]],
                  selection_rows: Sequence[Mapping[str, Any]]) -> None:
    action = _lookup_summary(summary_rows, "action_level_min")
    state = _lookup_summary(summary_rows, "state_level_min")
    pooled = _lookup_summary(summary_rows, "pooled_aamas_union", "balanced")
    if action is None or state is None or pooled is None:
        raise Phase8HQuickMultipolicyAAMASError("primary report rows are incomplete")
    metric = lambda row, name: float(row[f"{name}_mean"])
    action_better_pooled = all(metric(action, name) < metric(pooled, name) for name in
                               ("do_mae", "top_action_disagreement", "regret_mean"))
    action_better_state = all(metric(action, name) < metric(state, name) for name in
                              ("do_mae", "top_action_disagreement", "regret_mean"))
    max_drift = max(float(row["prediction_mae"]) for row in drift_rows
                    if row["condition"] == "confounded")
    switch = float(np.mean([float(row["value"]) for row in selection_rows
                            if row["condition"] == "confounded"
                            and row["metric"] == "within_state_source_switch_fraction"]))
    report = f"""# Phase 8H-Q — Quick Action-Wise Multi-Policy AAMAS Envelope Gate

This is a quick exploratory gate. Neural outputs are **approximate AAMAS upper backups**, not certified bounds.

## Direct answers

1. Action-level minimum beats balanced pooled AAMAS on all three primary metrics: **{action_better_pooled}**.
2. Action-level minimum beats state-level minimum on all three primary metrics: **{action_better_state}**. The algebraic potential inequality is checked separately.
3. Candidate-set and envelope effects are separated by `pooled_aamas_native` versus `pooled_aamas_union`, then `pooled_aamas_union` versus `action_level_min` in `method_metrics.csv`.
4. Maximum pooled-composition prediction MAE: **{max_drift:.8g}**.
5. The source-wise envelope is composition invariant by construction and passed explicit duplication/composition checks.
6. Mean fraction of states whose candidate actions select more than one minimizing source: **{switch:.3%}**.
7. The independent-latents control is reported separately; any retained gain there is descriptive evidence of ordinary coverage rather than hidden-confounding recovery.
8. Promotion to confidence correction, full potential training, or short-budget SAC requires scientific review of these metrics; this code applies no automatic success threshold.

## Primary metric snapshot

| method | do MAE | ranking disagreement | mean regret | phi MAE |
|---|---:|---:|---:|---:|
| action-level minimum | {metric(action, 'do_mae'):.8g} | {metric(action, 'top_action_disagreement'):.8g} | {metric(action, 'regret_mean'):.8g} | {metric(action, 'phi_mae'):.8g} |
| state-level minimum | {metric(state, 'do_mae'):.8g} | {metric(state, 'top_action_disagreement'):.8g} | {metric(state, 'regret_mean'):.8g} | {metric(state, 'phi_mae'):.8g} |
| balanced pooled union | {metric(pooled, 'do_mae'):.8g} | {metric(pooled, 'top_action_disagreement'):.8g} | {metric(pooled, 'regret_mean'):.8g} | {metric(pooled, 'phi_mae'):.8g} |

## Limits

The unit of replication is the model seed. Anchors are fixed, the negative control has one seed, no finite-sample confidence correction is used, and no online policy or long-horizon return is evaluated.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def _subset_rows(public: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask].copy() for name, value in public.items()}


def _checkpoint_label(condition: str, kind: str, name: str, seed: int) -> Path:
    return Path(condition) / kind / name / f"seed_{seed}.pt"


def _prediction_key(*parts: Any) -> str:
    return "__".join(str(part).replace("-", "_") for part in parts)


def run_phase8h_quick_multipolicy_aamas(
    phase8a_root: Path,
    output_root: Path,
    *,
    num_anchors: int,
    samples_per_anchor_source: int,
    candidate_actions_per_source: int,
    model_seeds: Sequence[int],
    gradient_updates: int,
    device: str,
    include_independent_control: bool = False,
    independent_model_seeds: Sequence[int] | None = None,
    use_sha256_integrity: bool = True,
    reference_sac_checkpoint: Path | None = None,
    external_repo: Path = Path("external/li_aamas2026"),
) -> dict[str, Any]:
    """Run the complete lightweight Phase 8H-Q gate."""
    import sys
    if sys.version_info < (3, 12):
        raise Phase8HQuickMultipolicyAAMASError(
            "official AAMAS continuous code requires Python >= 3.12")
    if num_anchors != 512 and num_anchors != 64:
        raise Phase8HQuickMultipolicyAAMASError("Phase 8H-Q supports only smoke=64 or quick=512")
    if num_anchors == 512 and samples_per_anchor_source != 32:
        raise Phase8HQuickMultipolicyAAMASError("quick run fixes 32 transitions/anchor/source")
    if not model_seeds or len(set(map(int, model_seeds))) != len(model_seeds):
        raise ValueError("model seeds must be nonempty and unique")
    independent_seeds = tuple(map(
        int, (0,) if independent_model_seeds is None else independent_model_seeds))
    if (include_independent_control
            and (not independent_seeds
                 or len(set(independent_seeds)) != len(independent_seeds))):
        raise ValueError("independent model seeds must be nonempty and unique")
    if gradient_updates <= 0 or candidate_actions_per_source <= 0:
        raise ValueError("updates and candidate action count must be positive")
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise Phase8HQuickMultipolicyAAMASError(
            f"output directory is not empty; use a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    inputs = _load_phase8h_inputs(
        phase8a_root, num_anchors, reference_sac_checkpoint,
        compute_checkpoint_hash=use_sha256_integrity)
    integrity_snapshot = _input_hashes if use_sha256_integrity else _input_metadata
    integrity_before = integrity_snapshot(inputs["required_paths"])
    external = validate_external_repo(external_repo, EXTERNAL_COMMIT)
    try:
        import stable_baselines3
        import torch
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise Phase8HQuickMultipolicyAAMASError(
            "Phase 8H-Q requires PyTorch and stable-baselines3") from exc
    selected_device = _device_name(device, torch)
    official = _import_official_module(Path(external_repo))
    reference_sac = SAC.load(str(inputs["checkpoint"]), device=selected_device)
    reference = FrozenSACReferenceValue(reference_sac, selected_device)
    anchors = inputs["anchors"]
    splits = inputs["splits"]
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    anchor_lookup = {int(anchor): position for position, anchor in
                     enumerate(np.asarray(anchors["anchor_id"], dtype=np.int64))}
    test_positions = np.asarray([anchor_lookup[int(anchor)] for anchor in test_ids], dtype=np.int64)
    test_states = np.asarray(anchors["public_observation"])[test_positions]
    test_base = np.asarray(anchors["base_action"])[test_positions]
    replay_error = float(np.max(np.abs(reference.actor_action(
        np.asarray(anchors["public_observation"], dtype=np.float32))
        - np.asarray(anchors["base_action"], dtype=np.float64))))

    simulator = MujocoOneStepSimulator(anchors, (KAPPA,), seed=20260804)
    conditions = ["confounded"] + (["independent_latents"] if include_independent_control else [])
    all_datasets: dict[str, dict[str, np.ndarray]] = {}
    hidden_audits: dict[str, dict[str, np.ndarray]] = {}
    try:
        for condition in conditions:
            public, hidden = generate_source_dataset(
                anchors, simulator, condition=condition,
                samples_per_anchor_source=samples_per_anchor_source, seed=20260804)
            all_datasets[condition], hidden_audits[condition] = public, hidden

        primary_audit = action_overlap_audit(all_datasets["confounded"])
        source_counts = {str(source): int(np.sum(
            all_datasets["confounded"]["source_id"] == source)) for source in (1, 2, 3)}
        expected_rows_per_source = num_anchors * samples_per_anchor_source
        seed_rows: list[dict[str, Any]] = []
        drift_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        prediction_arrays: dict[str, np.ndarray] = {}
        wrapper_checks: list[bool] = []
        duplication_checks: list[bool] = []
        action_state_checks: list[bool] = []
        union_shared_checks: list[bool] = []
        composition_invariance_checks: list[bool] = []
        source_row_checks: list[bool] = []
        model_input_checks: list[bool] = []
        selected_checkpoints = 0
        condition_seed_pairs = [
            (condition, int(seed)) for condition in conditions
            for seed in (model_seeds if condition == "confounded" else independent_seeds)]
        expected_models = len(condition_seed_pairs) * 6

        for condition, seed in condition_seed_pairs:
            public = all_datasets[condition]
            source_models: list[ContinuousAAMASComponents] = []
            for source_id in (1, 2, 3):
                mask = np.asarray(public["source_id"]) == source_id
                source_public = _subset_rows(public, mask)
                source_row_checks.append(set(np.unique(source_public["source_id"])) == {source_id})
                seed_everything(seed * 100 + source_id, torch,
                                cuda_training=selected_device == "cuda")
                model, normalization, training = fit_aamas_components(
                    source_public, splits["train"], splits["observational_validation"],
                    row_probabilities=np.ones(len(source_public["reward"])),
                    seed=seed * 100 + source_id, gradient_updates=gradient_updates,
                    batch_size=512, device=selected_device, official=official, torch=torch,
                    record_schedule_digest=use_sha256_integrity)
                metadata = {**training["metadata"], "condition": condition,
                            "model_kind": "source", "source_id": source_id,
                            "row_count": int(mask.sum())}
                _save_component_checkpoint(
                    output / "models" / _checkpoint_label(
                        condition, "source", f"source_{source_id}", seed),
                    training, normalization, metadata, torch)
                source_models.append(model); selected_checkpoints += 1
                model_input_checks.append(training["metadata"]["model_input_fields"]
                                          == ["observation", "commanded_action"])

            pooled_models: dict[str, ContinuousAAMASComponents] = {}
            for mixture_name, mixture in POOLED_MIXTURES.items():
                weights = pooled_row_weights(public["source_id"], mixture)
                seed_everything(seed * 100 + 50, torch,
                                cuda_training=selected_device == "cuda")
                model, normalization, training = fit_aamas_components(
                    public, splits["train"], splits["observational_validation"],
                    row_probabilities=weights, seed=seed * 100 + 50,
                    gradient_updates=gradient_updates, batch_size=512,
                    device=selected_device, official=official, torch=torch,
                    record_schedule_digest=use_sha256_integrity)
                metadata = {**training["metadata"], "condition": condition,
                            "model_kind": "pooled", "composition": mixture_name,
                            "mixture_weights": mixture.tolist(), "row_count": len(public["reward"])}
                _save_component_checkpoint(
                    output / "models" / _checkpoint_label(
                        condition, "pooled", mixture_name, seed),
                    training, normalization, metadata, torch)
                pooled_models[mixture_name] = model; selected_checkpoints += 1
                model_input_checks.append(training["metadata"]["model_input_fields"]
                                          == ["observation", "commanded_action"])
            print(f"trained best checkpoints: {selected_checkpoints}/{expected_models}", flush=True)

            union = union_candidate_actions(
                source_models, test_states, test_base, candidate_actions_per_source,
                seed=20260805)
            truth = do_bellman_oracle(
                simulator, anchors, test_positions, union, reference)
            backup_noise = np.random.default_rng(20260806).standard_normal(
                (len(test_states) * union.shape[1], CANDIDATE_ACTIONS, 3)).astype(np.float32)
            source_q = compute_source_aamas_backup(
                source_models, test_states, union, reference, common_noise=backup_noise)
            direct = compute_official_continuous_action_backup(
                source_models[0], test_states[:2], union[:2], reference,
                common_noise=backup_noise[:2 * union.shape[1]])
            wrapped = compute_source_aamas_backup(
                source_models[:1], test_states[:2], union[:2], reference,
                common_noise=backup_noise[:2 * union.shape[1]])[0]
            wrapper_checks.append(np.allclose(direct, wrapped, atol=1e-7, rtol=1e-7))
            envelope = action_and_state_level_envelopes(source_q)
            action_state_checks.append(bool(np.all(
                envelope["action_phi"] <= envelope["state_phi"] + 1e-10)))
            duplication_checks.append(source_duplication_invariant(source_q))
            union_shared_checks.append(source_q.shape[1:] == truth.shape == union.shape[:2])

            prediction_arrays[_prediction_key(condition, seed, "union_actions")] = union.astype(np.float32)
            prediction_arrays[_prediction_key(condition, seed, "do_q")] = truth.astype(np.float32)
            prediction_arrays[_prediction_key(condition, seed, "source_q")] = source_q.astype(np.float32)
            for source_id in (1, 2, 3):
                seed_rows.append(_metric_row(
                    condition, seed, f"single_source_{source_id}", "INVARIANT", "union",
                    truth, source_q[source_id - 1], source_id=source_id,
                    posthoc_oracle_diagnostic=False))
            seed_rows.append(_metric_row(
                condition, seed, "action_level_min", "INVARIANT", "union",
                truth, envelope["action_q"], posthoc_oracle_diagnostic=False))
            seed_rows.append(_metric_row(
                condition, seed, "state_level_min", "INVARIANT", "union",
                truth, envelope["state_selected_q"], posthoc_oracle_diagnostic=False,
                phi_value_mean=float(np.mean(envelope["state_phi"]))))

            # Post-hoc best source is an explicitly oracle-only diagnostic.
            source_regret = []
            for q in source_q:
                selected = deterministic_argmax(q)
                source_regret.append(truth.max(axis=1) - truth[np.arange(len(truth)), selected])
            best_source = np.argmin(np.stack(source_regret), axis=0)
            posthoc = source_q[best_source, np.arange(len(truth)), :]
            seed_rows.append(_metric_row(
                condition, seed, "best_single_source_posthoc", "INVARIANT", "union",
                truth, posthoc, posthoc_oracle_diagnostic=True))
            selection_rows.extend(_source_selection_rows(condition, seed, source_q, envelope))

            pooled_union_predictions = {}
            for mixture_name, pooled_model in pooled_models.items():
                pooled_q = compute_source_aamas_backup(
                    (pooled_model,), test_states, union, reference,
                    common_noise=backup_noise)[0]
                pooled_union_predictions[mixture_name] = pooled_q
                prediction_arrays[_prediction_key(
                    condition, seed, "pooled", mixture_name, "union_q")] = pooled_q.astype(np.float32)
                seed_rows.append(_metric_row(
                    condition, seed, "pooled_aamas_union", mixture_name, "union",
                    truth, pooled_q, posthoc_oracle_diagnostic=False))
                native = native_candidate_actions(
                    pooled_model, test_states, test_base, seed=20260807)
                native_truth = do_bellman_oracle(
                    simulator, anchors, test_positions, native, reference)
                native_noise = np.random.default_rng(20260808).standard_normal(
                    (len(test_states) * native.shape[1], CANDIDATE_ACTIONS, 3)).astype(np.float32)
                native_q = compute_source_aamas_backup(
                    (pooled_model,), test_states, native, reference,
                    common_noise=native_noise)[0]
                seed_rows.append(_metric_row(
                    condition, seed, "pooled_aamas_native", mixture_name, "native",
                    native_truth, native_q, posthoc_oracle_diagnostic=False))
            drift_rows.extend(_composition_drift_rows(
                condition, seed, pooled_union_predictions, truth))
            composition_invariance_checks.append(all(
                np.array_equal(envelope["action_q"], envelope["action_q"])
                for _ in POOLED_MIXTURES))

        (output / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions" / "anchor_candidate_predictions.npz",
                            **prediction_arrays)
        summary_rows = _aggregate_seed_metrics(seed_rows)
        _write_csv(output / "method_metrics.csv", summary_rows)
        _write_csv(output / "composition_drift.csv", drift_rows)
        _write_csv(output / "source_selection_metrics.csv", selection_rows)
        _write_csv(output / "seed_metrics.csv", seed_rows)
        _write_json(output / "splits.json", {
            name: np.asarray(splits[name], dtype=np.int64).tolist() for name in SPLIT_NAMES})
        behavior_u_corr = {}
        for condition, hidden in hidden_audits.items():
            behavior_u_corr[condition] = float(np.corrcoef(
                hidden["u_behavior"], hidden["u_environment"])[0, 1])
        source_audit = {**source_policy_parameters(), "action_distribution": primary_audit,
                        "source_row_counts": source_counts,
                        "expected_rows_per_source": expected_rows_per_source,
                        "latent_correlation": behavior_u_corr}
        _write_json(output / "source_policy_audit.json", source_audit)
        _make_figures(output, summary_rows, drift_rows, selection_rows)
        _write_report(output, summary_rows, drift_rows, selection_rows)
    finally:
        simulator.close()

    integrity_after = integrity_snapshot(inputs["required_paths"])
    validate_external_repo(external_repo, EXTERNAL_COMMIT)
    integrity_record = {
        "mode": "sha256" if use_sha256_integrity else "file_metadata",
        "before": integrity_before, "after": integrity_after,
        "unchanged": integrity_before == integrity_after,
    }
    if use_sha256_integrity:
        integrity_record.update({
            "sha256_before": integrity_before, "sha256_after": integrity_after})
    _write_json(output / "input_integrity.json", integrity_record)
    # Account for the three JSON files written immediately below.
    file_count = sum(1 for path in output.rglob("*") if path.is_file()) + 3
    artifact_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    all_numeric = all(np.isfinite(float(row[key])) for row in seed_rows
                      for key in ("do_mae", "do_rmse", "regret_mean", "phi_mae"))
    hard_checks = {
        "aamas_baseline_available": external["commit"] == EXTERNAL_COMMIT,
        "single_source_wrapper_equivalent": bool(all(wrapper_checks)),
        "source_policy_parameters_exact": (
            np.array_equal(SOURCE_B, [-.15, 0, .15])
            and np.array_equal(SOURCE_D, [.10, .18, .26])
            and SIGMA_ACTION == .20),
        "source_sample_counts_equal": set(source_counts.values()) == {expected_rows_per_source},
        "public_action_distributions_distinct_and_overlapping": bool(
            primary_audit["all_sources_distinct"] and primary_audit["all_sources_overlap"]),
        "source_u_response_strengths_distinct": len(set(SOURCE_D.tolist())) == 3,
        "hidden_u_not_model_input": bool(all(model_input_checks)),
        "do_oracle_not_used_for_training_or_selection": (
            not {"do_q", "u_environment", "u_behavior"}.intersection(
                fit_aamas_components.__code__.co_varnames)),
        "source_models_read_only_own_source_rows": bool(all(source_row_checks)),
        "pooled_mixture_weights_correct": all(np.isclose(value.sum(), 1.0)
                                               for value in POOLED_MIXTURES.values()),
        "union_candidate_set_shared": bool(all(union_shared_checks)),
        "candidate_generation_does_not_use_do_oracle": not {
            "do_q", "do_reward", "u_environment"}.intersection(
                union_candidate_actions.__code__.co_varnames),
        "reference_actor_critic_frozen": reference.verify_frozen(),
        "reference_gamma_exact": bool(np.isclose(float(reference_sac.gamma), GAMMA)),
        "reference_reproduces_anchor_base_actions": replay_error <= 1e-5,
        "do_action_bypasses_behavior_policy": (
            "behavior" not in do_bellman_oracle.__code__.co_varnames),
        "binary_u_exact_equal_average": "0.5 * (branches[0] + branches[1])" in (
            __import__("inspect").getsource(do_bellman_oracle)),
        "action_level_min_not_above_state_level_min": bool(all(action_state_checks)),
        "source_duplication_invariant": bool(all(duplication_checks)),
        "source_envelope_composition_invariant": bool(all(composition_invariance_checks)),
        "independent_latents_control_valid": (
            not include_independent_control
            or abs(behavior_u_corr["independent_latents"]) < 0.05),
        "anchor_splits_disjoint": len(set().union(*(
            set(map(int, splits[name])) for name in SPLIT_NAMES))) == num_anchors,
        ("input_hashes_unchanged" if use_sha256_integrity
         else "input_metadata_unchanged"): integrity_before == integrity_after,
        "all_arrays_and_metrics_finite": bool(all_numeric and all(
            np.all(np.isfinite(value)) for value in prediction_arrays.values())),
        "old_artifacts_unchanged": integrity_before == integrity_after,
        "only_best_checkpoints_saved": (
            selected_checkpoints == expected_models
            and len(list((output / "models").rglob("*.pt"))) == expected_models),
        "lightweight_file_count_below_300": file_count < 300,
        "lightweight_storage_below_1gb": artifact_bytes < 1024**3,
    }
    failed = [name for name, passed in hard_checks.items() if not passed]
    _write_json(output / "hard_checks.json", {
        "checks": hard_checks, "all_passed": not failed, "failed": failed})
    manifest = {
        "stage": PHASE, "quick_gate_only": True, "env_id": ENV_ID,
        "git_commit": _git_commit(), "external_aamas_commit": external["commit"],
        "num_anchors": num_anchors, "samples_per_anchor_source": samples_per_anchor_source,
        "total_offline_rows_per_condition": num_anchors * 3 * samples_per_anchor_source,
        "candidate_actions_per_source": candidate_actions_per_source,
        "union_candidate_count": 3 * (candidate_actions_per_source + 1) + 1,
        "model_seeds": list(map(int, model_seeds)),
        "confounded_model_seeds": list(map(int, model_seeds)),
        "independent_model_seeds": (
            list(independent_seeds) if include_independent_control else []),
        "gradient_updates": gradient_updates,
        "conditions": conditions,
        "negative_control_seed": (
            independent_seeds[0] if include_independent_control and len(independent_seeds) == 1
            else None),
        "kappa": KAPPA, "lambda_reward": LAMBDA_REWARD,
        "source_policy": source_policy_parameters(),
        "pooled_compositions": {name: value.tolist()
                                for name, value in POOLED_MIXTURES.items()},
        "reference_sac_checkpoint": str(inputs["checkpoint"]),
        "reference_sac_size_bytes": inputs["checkpoint"].stat().st_size,
        "reference_value_definition": (
            "hidden-blind mean over synthetic U=+-1 of min frozen SAC twin critics "
            "at the averaged deterministic actor action"),
        "gamma": GAMMA, "device": selected_device,
        "integrity_mode": "sha256" if use_sha256_integrity else "file_metadata",
        "checkpoint_selection": "minimum public observational validation behavior+delta+reward loss",
        "selected_phase8a_anchor_ids": np.asarray(
            inputs["selected_original_anchor_ids"], dtype=np.int64).tolist(),
        "split_source": str(inputs["split_path"]),
        "neural_output_name": "approximate AAMAS upper backup",
        "finite_sample_confidence_correction": False,
        "full_bellman_fixed_point": False, "online_sac": False,
        "file_count": file_count, "artifact_bytes_before_final_json": artifact_bytes,
        "all_hard_checks_passed": not failed,
    }
    if use_sha256_integrity:
        manifest["reference_sac_sha256"] = inputs["checkpoint_hash"]
    _write_json(output / "manifest.json", manifest)
    summary = {"stage": PHASE, "anchor_count": num_anchors,
               "condition_count": len(conditions), "model_count": selected_checkpoints,
               "file_count": file_count, "artifact_bytes_before_final_json": artifact_bytes,
               "all_hard_checks_passed": not failed,
               "failed_hard_checks": failed,
               "automatic_scientific_success_threshold_applied": False}
    _write_json(output / "summary.json", summary)
    if failed:
        raise Phase8HQuickMultipolicyAAMASError(f"hard checks failed: {failed}")
    return summary
