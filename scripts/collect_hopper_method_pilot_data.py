"""Collect the fixed Stage Hopper data for the Phase 7A method pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confounded_hopper import ACTUATOR_DIRECTION, ConfoundedHopperWrapper
from scripts.compare_hopper_source_groups import find_unique_checkpoint_by_step


ENV_ID = "Hopper-v5"
SOURCE_STEPS = {1: 200_000, 2: 500_000, 3: 1_000_000}
PUBLIC_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
    "collector_truncated",
    "source_id",
    "checkpoint_step",
    "episode_id",
    "step_in_episode",
    "row_id",
)
HIDDEN_ARRAY_FIELDS = (
    "row_id",
    "source_id",
    "episode_id",
    "step_in_episode",
    "hidden_u",
    "applied_action",
    "preclip_action",
    "clipping_indicator",
)
HIDDEN_METADATA_FIELD = "AUDIT_ONLY_DO_NOT_USE_FOR_TRAINING"
FORBIDDEN_PUBLIC_FIELDS = {
    "hidden_u",
    "applied_action",
    "preclip_action",
    "qpos",
    "qvel",
    "behavior_observation",
    HIDDEN_METADATA_FIELD,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collection_seed_plan(seed: int) -> dict[str, dict[str, dict[str, Any]]]:
    """Spawn six independent and reproducible source/split random streams."""
    root = np.random.SeedSequence(seed)
    children = root.spawn(6)
    plan: dict[str, dict[str, dict[str, Any]]] = {
        f"source_{source_id}": {} for source_id in SOURCE_STEPS
    }
    for child, (source_id, split) in zip(
        children,
        ((source, split) for source in SOURCE_STEPS for split in ("train", "audit")),
    ):
        env_seed, policy_seed, numpy_seed, python_seed = (
            int(value) for value in child.generate_state(4, dtype=np.uint32)
        )
        plan[f"source_{source_id}"][split] = {
            "spawn_key": list(child.spawn_key),
            "environment_seed": env_seed,
            "policy_seed": policy_seed,
            "numpy_seed": numpy_seed,
            "python_seed": python_seed,
            "torch_seed": policy_seed,
        }
    signatures = {
        tuple(entry[key] for key in ("environment_seed", "policy_seed", "numpy_seed", "python_seed"))
        for source in plan.values()
        for entry in source.values()
    }
    if len(signatures) != 6:
        raise RuntimeError("SeedSequence did not produce six distinct collection streams")
    return plan


def _seed_policy_stream(model: Any, seeds: dict[str, Any]) -> None:
    if hasattr(model, "set_random_seed"):
        model.set_random_seed(int(seeds["policy_seed"]))
    random.seed(int(seeds["python_seed"]))
    np.random.seed(int(seeds["numpy_seed"]))
    torch = sys.modules.get("torch")
    if torch is None:  # Fake policies in unit tests do not load PyTorch.
        return
    torch.manual_seed(int(seeds["torch_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seeds["torch_seed"]))


def _make_environment(kappa: float) -> ConfoundedHopperWrapper:
    return ConfoundedHopperWrapper(
        gym.make(ENV_ID),
        kappa=kappa,
        expose_confounder=True,
        audit_info=True,
    )


def _episode_id(source_id: int, split: str, local_episode: int) -> int:
    split_offset = 0 if split == "train" else 100_000_000
    return source_id * 1_000_000_000 + split_offset + local_episode


def _collect_stream(
    model: Any,
    *,
    source_id: int,
    split: str,
    transition_count: int,
    kappa: float,
    seeds: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    if source_id not in SOURCE_STEPS or split not in ("train", "audit"):
        raise ValueError("unknown source or split")
    if transition_count <= 0:
        raise ValueError("transition_count must be positive")
    _seed_policy_stream(model, seeds)
    environment = _make_environment(kappa)
    public: dict[str, list[Any]] = {
        field: [] for field in PUBLIC_FIELDS if field != "row_id"
    }
    hidden: dict[str, list[Any]] | None = None
    if split == "audit":
        hidden = {
            field: [] for field in HIDDEN_ARRAY_FIELDS if field != "row_id"
        }
    local_episode = 0
    step_in_episode = 0
    observation, _ = environment.reset(seed=int(seeds["environment_seed"]))
    last_terminated = last_truncated = False
    try:
        for index in range(transition_count):
            behavior_observation = np.asarray(observation, dtype=np.float32)
            if behavior_observation.shape != (13,):
                raise RuntimeError("behavior observation must have shape (13,)")
            current_u = int(behavior_observation[-1])
            if current_u not in (-1, 1):
                raise RuntimeError("behavior observation contains an invalid hidden U")
            public_observation = environment.get_public_observation(behavior_observation)
            action, _ = model.predict(behavior_observation, deterministic=False)
            command = np.asarray(action, dtype=np.float32)
            if command.shape != (3,) or not np.all(np.isfinite(command)):
                raise RuntimeError("policy produced an invalid commanded action")
            if np.any(command < -1.0) or np.any(command > 1.0):
                raise RuntimeError("policy commanded action is outside [-1, 1]")

            next_observation, reward, terminated, truncated, info = environment.step(command)
            if int(info["hidden_u"]) != current_u:
                raise RuntimeError("wrapper hidden U is not aligned with the pre-step observation")
            if not np.allclose(info["commanded_action"], command, rtol=0.0, atol=1e-7):
                raise RuntimeError("wrapper changed the commanded action semantics")
            preclip = command.astype(np.float64) + kappa * current_u * ACTUATOR_DIRECTION
            applied = np.asarray(info["applied_action"], dtype=np.float32)
            expected_applied = np.clip(preclip, -1.0, 1.0).astype(np.float32)
            if not np.allclose(applied, expected_applied, rtol=0.0, atol=1e-7):
                raise RuntimeError("wrapper applied action differs from the fixed formula")
            if not np.isfinite(reward):
                raise RuntimeError("environment produced a nonfinite reward")

            episode_id = _episode_id(source_id, split, local_episode)
            values = {
                "observations": public_observation,
                "actions": command.copy(),
                "rewards": reward,
                "next_observations": environment.get_public_observation(next_observation),
                "terminated": terminated,
                "truncated": truncated,
                "collector_truncated": False,
                "source_id": source_id,
                "checkpoint_step": SOURCE_STEPS[source_id],
                "episode_id": episode_id,
                "step_in_episode": step_in_episode,
            }
            for field, value in values.items():
                public[field].append(value)
            if hidden is not None:
                hidden_values = {
                    "source_id": source_id,
                    "episode_id": episode_id,
                    "step_in_episode": step_in_episode,
                    "hidden_u": current_u,
                    "applied_action": applied.copy(),
                    "preclip_action": preclip.astype(np.float32),
                    "clipping_indicator": bool(np.any(np.abs(preclip) > 1.0)),
                }
                for field, value in hidden_values.items():
                    hidden[field].append(value)

            observation = next_observation
            last_terminated, last_truncated = bool(terminated), bool(truncated)
            step_in_episode += 1
            if (terminated or truncated) and index + 1 < transition_count:
                local_episode += 1
                step_in_episode = 0
                observation, _ = environment.reset()
        if not (last_terminated or last_truncated):
            public["collector_truncated"][-1] = True
    finally:
        environment.close()

    public_arrays = {
        "observations": np.asarray(public["observations"], dtype=np.float32),
        "actions": np.asarray(public["actions"], dtype=np.float32),
        "rewards": np.asarray(public["rewards"], dtype=np.float32),
        "next_observations": np.asarray(public["next_observations"], dtype=np.float32),
        "terminated": np.asarray(public["terminated"], dtype=bool),
        "truncated": np.asarray(public["truncated"], dtype=bool),
        "collector_truncated": np.asarray(public["collector_truncated"], dtype=bool),
        "source_id": np.asarray(public["source_id"], dtype=np.int8),
        "checkpoint_step": np.asarray(public["checkpoint_step"], dtype=np.int64),
        "episode_id": np.asarray(public["episode_id"], dtype=np.int64),
        "step_in_episode": np.asarray(public["step_in_episode"], dtype=np.int32),
    }
    if hidden is None:
        return public_arrays, None
    hidden_arrays = {
        "source_id": np.asarray(hidden["source_id"], dtype=np.int8),
        "episode_id": np.asarray(hidden["episode_id"], dtype=np.int64),
        "step_in_episode": np.asarray(hidden["step_in_episode"], dtype=np.int32),
        "hidden_u": np.asarray(hidden["hidden_u"], dtype=np.int8),
        "applied_action": np.asarray(hidden["applied_action"], dtype=np.float32),
        "preclip_action": np.asarray(hidden["preclip_action"], dtype=np.float32),
        "clipping_indicator": np.asarray(hidden["clipping_indicator"], dtype=bool),
    }
    return public_arrays, hidden_arrays


def _concatenate(parts: list[dict[str, np.ndarray]], fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {field: np.concatenate([part[field] for part in parts]) for field in fields}


def collect_datasets(
    models: dict[int, Any],
    *,
    train_transitions_per_source: int,
    audit_transitions_per_source: int,
    seed: int,
    kappa: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    if set(models) != set(SOURCE_STEPS):
        raise ValueError("models must contain fixed sources 1, 2, and 3")
    seeds = collection_seed_plan(seed)
    train_parts, audit_parts, hidden_parts = [], [], []
    for source_id in SOURCE_STEPS:
        source_name = f"source_{source_id}"
        train, _ = _collect_stream(
            models[source_id], source_id=source_id, split="train",
            transition_count=train_transitions_per_source, kappa=kappa,
            seeds=seeds[source_name]["train"],
        )
        audit, hidden = _collect_stream(
            models[source_id], source_id=source_id, split="audit",
            transition_count=audit_transitions_per_source, kappa=kappa,
            seeds=seeds[source_name]["audit"],
        )
        assert hidden is not None
        train_parts.append(train)
        audit_parts.append(audit)
        hidden_parts.append(hidden)
        print(
            f"collected {source_name}: train={train_transitions_per_source}, "
            f"audit={audit_transitions_per_source}"
        )

    public_without_row = tuple(field for field in PUBLIC_FIELDS if field != "row_id")
    hidden_without_row = tuple(field for field in HIDDEN_ARRAY_FIELDS if field != "row_id")
    train = _concatenate(train_parts, public_without_row)
    audit = _concatenate(audit_parts, public_without_row)
    train["row_id"] = np.arange(train["rewards"].size, dtype=np.int64)
    audit["row_id"] = np.arange(
        train["rewards"].size,
        train["rewards"].size + audit["rewards"].size,
        dtype=np.int64,
    )
    hidden = _concatenate(hidden_parts, hidden_without_row)
    hidden["row_id"] = audit["row_id"].copy()
    hidden[HIDDEN_METADATA_FIELD] = np.asarray(True, dtype=bool)

    validate_public_dataset(train, train_transitions_per_source)
    validate_public_dataset(audit, audit_transitions_per_source)
    validate_split_isolation(train, audit)
    validate_hidden_audit(audit, hidden, kappa)
    return train, audit, hidden, seeds


def validate_public_dataset(data: dict[str, np.ndarray], per_source: int) -> None:
    if set(data) != set(PUBLIC_FIELDS):
        raise RuntimeError(f"public fields must be exactly {sorted(PUBLIC_FIELDS)}")
    if FORBIDDEN_PUBLIC_FIELDS & set(data):
        raise RuntimeError("public data contains a hidden or simulator field")
    count = 3 * per_source
    expected_shapes = {
        "observations": (count, 12), "actions": (count, 3),
        "rewards": (count,), "next_observations": (count, 12),
        **{field: (count,) for field in PUBLIC_FIELDS[4:]},
    }
    for field, shape in expected_shapes.items():
        if data[field].shape != shape:
            raise RuntimeError(f"{field} has shape {data[field].shape}, expected {shape}")
    expected_dtypes = {
        "observations": np.dtype(np.float32), "actions": np.dtype(np.float32),
        "rewards": np.dtype(np.float32), "next_observations": np.dtype(np.float32),
        "terminated": np.dtype(bool), "truncated": np.dtype(bool),
        "collector_truncated": np.dtype(bool), "episode_id": np.dtype(np.int64),
        "step_in_episode": np.dtype(np.int32), "row_id": np.dtype(np.int64),
    }
    for field, dtype in expected_dtypes.items():
        if data[field].dtype != dtype:
            raise RuntimeError(f"{field} has dtype {data[field].dtype}, expected {dtype}")
    for field in ("observations", "actions", "rewards", "next_observations"):
        if not np.all(np.isfinite(data[field])):
            raise RuntimeError(f"{field} contains a nonfinite value")
    if np.any(np.abs(data["actions"]) > 1.0):
        raise RuntimeError("public commanded action is outside [-1, 1]")
    if np.unique(data["row_id"]).size != count:
        raise RuntimeError("public row_id is not unique")
    for source_id, step in SOURCE_STEPS.items():
        mask = data["source_id"] == source_id
        if int(np.sum(mask)) != per_source or not np.all(data["checkpoint_step"][mask] == step):
            raise RuntimeError("source count or checkpoint mapping is invalid")


def validate_split_isolation(
    train: dict[str, np.ndarray], audit: dict[str, np.ndarray]
) -> None:
    if np.intersect1d(train["episode_id"], audit["episode_id"]).size:
        raise RuntimeError("train and audit episode_id spaces overlap")
    if np.intersect1d(train["row_id"], audit["row_id"]).size:
        raise RuntimeError("train and audit row_id spaces overlap")


def validate_hidden_audit(
    audit: dict[str, np.ndarray], hidden: dict[str, np.ndarray], kappa: float
) -> None:
    expected_fields = set(HIDDEN_ARRAY_FIELDS) | {HIDDEN_METADATA_FIELD}
    if set(hidden) != expected_fields:
        raise RuntimeError(f"hidden audit fields must be exactly {sorted(expected_fields)}")
    if hidden[HIDDEN_METADATA_FIELD].shape != () or not bool(hidden[HIDDEN_METADATA_FIELD]):
        raise RuntimeError("hidden audit warning metadata is missing")
    count = audit["row_id"].size
    for field in HIDDEN_ARRAY_FIELDS:
        expected_shape = (count, 3) if field in ("applied_action", "preclip_action") else (count,)
        if hidden[field].shape != expected_shape:
            raise RuntimeError(f"hidden audit field {field} has an invalid shape")
    for field in ("row_id", "source_id", "episode_id", "step_in_episode"):
        if not np.array_equal(hidden[field], audit[field]):
            raise RuntimeError(f"hidden audit {field} is not aligned with audit public data")
    if not np.all(np.isin(hidden["hidden_u"], (-1, 1))):
        raise RuntimeError("hidden audit U is not binary")
    expected_preclip = (
        audit["actions"].astype(np.float64)
        + kappa * hidden["hidden_u"][:, None] * ACTUATOR_DIRECTION
    )
    expected_applied = np.clip(expected_preclip, -1.0, 1.0)
    expected_clipping = np.any(np.abs(expected_preclip) > 1.0, axis=1)
    if not np.allclose(hidden["preclip_action"], expected_preclip, rtol=0.0, atol=1e-7):
        raise RuntimeError("hidden preclip action does not match the fixed formula")
    if not np.allclose(hidden["applied_action"], expected_applied, rtol=0.0, atol=1e-7):
        raise RuntimeError("hidden applied action does not match the fixed formula")
    if not np.array_equal(hidden["clipping_indicator"], expected_clipping):
        raise RuntimeError("hidden clipping indicator is invalid")
    if not np.all(np.isfinite(hidden["preclip_action"])) or not np.all(
        np.isfinite(hidden["applied_action"])
    ):
        raise RuntimeError("hidden action audit contains a nonfinite value")


def _split_summary(
    public: dict[str, np.ndarray], hidden: dict[str, np.ndarray] | None
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for source_id in SOURCE_STEPS:
        mask = public["source_id"] == source_id
        source = {field: values[mask] for field, values in public.items()}
        command_norm = np.linalg.norm(source["actions"], axis=1)
        collector_episodes = np.unique(
            source["episode_id"][source["collector_truncated"]]
        )
        entry: dict[str, Any] = {
            "transition_count": int(np.sum(mask)),
            "episode_count": int(np.unique(source["episode_id"]).size),
            "collector_truncated_episode_count": int(collector_episodes.size),
            "mean_reward": float(np.mean(source["rewards"])),
            "reward_std": float(np.std(source["rewards"])),
            "termination_rate": float(np.mean(source["terminated"])),
            "truncation_rate": float(np.mean(source["truncated"])),
            "mean_commanded_action_norm": float(np.mean(command_norm)),
            "commanded_action_min": np.min(source["actions"], axis=0).tolist(),
            "commanded_action_max": np.max(source["actions"], axis=0).tolist(),
            "public_observation_min": np.min(source["observations"], axis=0).tolist(),
            "public_observation_max": np.max(source["observations"], axis=0).tolist(),
        }
        if hidden is not None:
            hidden_mask = hidden["source_id"] == source_id
            hidden_u = hidden["hidden_u"][hidden_mask]
            entry["hidden_u_proportions"] = {
                "-1": float(np.mean(hidden_u == -1)),
                "+1": float(np.mean(hidden_u == 1)),
            }
            entry["clipping_rate"] = float(
                np.mean(hidden["clipping_indicator"][hidden_mask])
            )
        report[f"source_{source_id}"] = entry
    return report


def write_artifacts(
    output_dir: Path,
    train: dict[str, np.ndarray],
    audit: dict[str, np.ndarray],
    hidden: dict[str, np.ndarray],
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train_public.npz", **train)
    np.savez_compressed(output_dir / "audit_public.npz", **audit)
    np.savez_compressed(output_dir / "audit_hidden.npz", **hidden)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(arguments: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if arguments.train_transitions_per_source <= 0 or arguments.audit_transitions_per_source <= 0:
        raise ValueError("train and audit transition counts must be positive")
    if float(arguments.kappa) != 0.2:
        raise ValueError("Phase 7A requires the fixed kappa=0.2")
    checkpoint_dir = Path(arguments.checkpoint_dir)
    paths = {
        source_id: find_unique_checkpoint_by_step(checkpoint_dir, step)
        for source_id, step in SOURCE_STEPS.items()
    }
    before_hashes = {source_id: _sha256(path) for source_id, path in paths.items()}
    try:
        import mujoco
        import stable_baselines3
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo and stable_baselines3 are required for Phase 7A collection"
        ) from exc
    models = {
        source_id: SAC.load(str(path), device=arguments.device)
        for source_id, path in paths.items()
    }
    train, audit, hidden, seeds = collect_datasets(
        models,
        train_transitions_per_source=arguments.train_transitions_per_source,
        audit_transitions_per_source=arguments.audit_transitions_per_source,
        seed=arguments.seed,
        kappa=arguments.kappa,
    )
    after_hashes = {source_id: _sha256(path) for source_id, path in paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("a fixed behavior-policy checkpoint changed during collection")

    source_mapping = {
        f"source_{source_id}": {
            "checkpoint_step": SOURCE_STEPS[source_id],
            "checkpoint_file_path": str(paths[source_id]),
            "checkpoint_sha256": before_hashes[source_id],
        }
        for source_id in SOURCE_STEPS
    }
    manifest = {
        "phase": "7A",
        "env_id": ENV_ID,
        "public_observation_dim": 12,
        "behavior_observation_dim": 13,
        "action_dim": 3,
        "kappa": float(arguments.kappa),
        "actuator_direction": ACTUATOR_DIRECTION.tolist(),
        "hidden_u_distribution": {"values": [-1, 1], "probabilities": [0.5, 0.5]},
        "source_mapping": source_mapping,
        "train_transitions_per_source": int(arguments.train_transitions_per_source),
        "audit_transitions_per_source": int(arguments.audit_transitions_per_source),
        "collection_seed": int(arguments.seed),
        "collection_seeds": seeds,
        "stochastic_policy": True,
        "stored_action_semantics": "commanded_action",
        "applied_action_location": "audit_hidden_only",
        "train_audit_collected_independently": True,
        "checkpoint_files_unchanged": True,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gymnasium_version": gym.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "mujoco_version": mujoco.__version__,
        "git_commit": _git_commit(),
    }
    summary = {
        "phase": "7A",
        "train_total_transitions": int(train["row_id"].size),
        "audit_total_transitions": int(audit["row_id"].size),
        "overall_transitions": int(train["row_id"].size + audit["row_id"].size),
        "train": _split_summary(train, None),
        "audit": _split_summary(audit, hidden),
    }
    json.dumps(manifest, allow_nan=False)
    json.dumps(summary, allow_nan=False)
    write_artifacts(Path(arguments.output_dir), train, audit, hidden, manifest, summary)
    return manifest, summary


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("artifacts/hopper_behavior_policies/seed_0"),
    )
    parser.add_argument("--train-transitions-per-source", type=int, default=16_000)
    parser.add_argument("--audit-transitions-per-source", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--kappa", type=float, default=0.2)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/hopper_method_pilot/stage_seed0"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.smoke:
        arguments.train_transitions_per_source = 1_600
        arguments.audit_transitions_per_source = 400
        arguments.output_dir = Path("artifacts/_smoke/hopper_method_pilot/stage_seed0")
    return arguments


if __name__ == "__main__":
    parsed = parse_arguments()
    completed_manifest, completed_summary = run(parsed)
    print("train transitions:", completed_summary["train_total_transitions"])
    print("audit transitions:", completed_summary["audit_total_transitions"])
    print("output:", parsed.output_dir)
    if parsed.smoke:
        print("PHASE7A_SMOKE_COMPLETE")
    else:
        print("PHASE7A_HOPPER_METHOD_PILOT_DATA_READY")
