"""Fixed Hopper simulator anchors and Phase 6A checkpoint resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from confounded_hopper import ACTUATOR_DIRECTION, ConfoundedHopperWrapper
from .controlled_loggers import base_actions_from_source2, headroom_mask


ENV_ID = "Hopper-v5"
SOURCE_STEPS = {1: 200_000, 2: 500_000, 3: 1_000_000}
ANCHOR_FIELDS = (
    "anchor_id", "qpos", "qvel", "simulator_state", "state_spec",
    "wrapper_elapsed_steps", "elapsed_steps", "public_observation", "base_action",
    "anchor_origin_source", "anchor_origin_episode", "anchor_origin_timestep",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_phase6a_checkpoints(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[int, Path]]:
    """Resolve the fixed Source 1/2/3 mapping exclusively from its manifest."""
    directory = Path(checkpoint_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Phase 6A manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("env_id") != ENV_ID:
        raise RuntimeError("Phase 6A manifest is not Hopper-v5")
    if (manifest.get("public_observation_dimension") != 12
            or manifest.get("behavior_observation_dimension") != 13
            or manifest.get("action_dimension") != 3):
        raise RuntimeError("Phase 6A observation/action schema is incompatible")
    direction = np.asarray(manifest.get("actuator_direction", ()), dtype=np.float64)
    if direction.shape != (3,) or not np.allclose(direction, ACTUATOR_DIRECTION,
                                                  atol=1e-15, rtol=0.0):
        raise RuntimeError("Phase 6A actuator direction is incompatible")
    paths = {}
    for source_id, step in SOURCE_STEPS.items():
        mapping = manifest.get("source_mapping", {}).get(f"source_{source_id}", {})
        if mapping.get("checkpoint_step") != step:
            raise RuntimeError("Phase 6A Source 1/2/3 mapping has changed")
        filename = mapping.get("model_file")
        if not isinstance(filename, str) or not filename:
            raise RuntimeError(f"Phase 6A source_{source_id} model_file is missing")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase 6A checkpoint: {path}")
        paths[source_id] = path
    return manifest, paths


def balanced_source_quotas(total: int) -> dict[int, int]:
    """Use the closest possible balance while preserving the requested total."""
    if total < 3:
        raise ValueError("num_anchors must be at least three")
    return {source: total // 3 + int(source <= total % 3) for source in (1, 2, 3)}


def make_anchor_environment() -> ConfoundedHopperWrapper:
    return ConfoundedHopperWrapper(
        gym.make(ENV_ID), kappa=0.2, expose_confounder=True, audit_info=True
    )


def _encoded_wrapper_steps(values: tuple[int | None, ...]) -> np.ndarray:
    return np.asarray([-1 if value is None else int(value) for value in values], dtype=np.int64)


def validate_anchor_pool(anchors: dict[str, np.ndarray], expected_count: int | None = None) -> None:
    if set(anchors) != set(ANCHOR_FIELDS):
        raise RuntimeError(f"anchor fields must be exactly {list(ANCHOR_FIELDS)}")
    count = len(anchors["anchor_id"])
    if expected_count is not None and count != expected_count:
        raise RuntimeError(f"anchor count {count} differs from requested {expected_count}")
    if count == 0 or not np.array_equal(anchors["anchor_id"], np.arange(count)):
        raise RuntimeError("anchor_id must be contiguous and deterministic")
    expected_shapes = {"qpos": (count, 6), "qvel": (count, 6),
                       "public_observation": (count, 12), "base_action": (count, 3)}
    for field, shape in expected_shapes.items():
        if anchors[field].shape != shape:
            raise RuntimeError(f"{field} must have shape {shape}")
    lengths = {len(np.asarray(value)) for value in anchors.values()}
    if lengths != {count}:
        raise RuntimeError("anchor arrays are not row-aligned")
    for field, values in anchors.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise RuntimeError(f"anchor field {field} contains NaN or Inf")
    if np.any(anchors["elapsed_steps"] < 0) or np.any(anchors["elapsed_steps"] >= 1000):
        raise RuntimeError("anchor elapsed_steps are outside the Hopper time limit")


def collect_anchor_pool(
    models: dict[int, Any], source2_model: Any, num_anchors: int, behavior_offset: float,
    seed: int, environment_factory: Callable[[], ConfoundedHopperWrapper] = make_anchor_environment,
    minimum_episode_spacing: int = 20,
) -> dict[str, np.ndarray]:
    """Collect pre-step states using fixed Stage policies and outcome-blind selection."""
    if set(models) != {1, 2, 3}:
        raise ValueError("anchor collection requires the fixed Stage Source 1/2/3 models")
    quotas = balanced_source_quotas(num_anchors)
    rows: list[dict[str, Any]] = []
    for source_id in (1, 2, 3):
        model, environment = models[source_id], environment_factory()
        episode = timestep = accepted = 0
        last_accepted_timestep = -minimum_episode_spacing
        observation, _ = environment.reset(seed=seed + source_id * 1_000_000)
        maximum_steps = max(20_000, quotas[source_id] * 500)
        try:
            for _ in range(maximum_steps):
                public = environment.get_public_observation(observation)
                base = base_actions_from_source2(source2_model, public[None, :])[0]
                spaced = timestep - last_accepted_timestep >= minimum_episode_spacing
                if spaced and bool(headroom_mask(base[None, :], behavior_offset)[0]):
                    snapshot = environment.capture_audit_state()
                    unwrapped = environment.env.unwrapped
                    rows.append({
                        "qpos": np.asarray(unwrapped.data.qpos, dtype=np.float64).copy(),
                        "qvel": np.asarray(unwrapped.data.qvel, dtype=np.float64).copy(),
                        "simulator_state": np.asarray(snapshot["simulator_state"], dtype=np.float64).copy(),
                        "state_spec": int(snapshot["state_spec"]),
                        "wrapper_elapsed_steps": _encoded_wrapper_steps(snapshot["wrapper_elapsed_steps"]),
                        "elapsed_steps": int(snapshot["elapsed_steps"]),
                        "public_observation": public.astype(np.float32),
                        "base_action": base.astype(np.float64),
                        "anchor_origin_source": source_id,
                        "anchor_origin_episode": episode,
                        "anchor_origin_timestep": timestep,
                    })
                    accepted += 1
                    last_accepted_timestep = timestep
                    if accepted == quotas[source_id]:
                        break
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = environment.step(action)
                timestep += 1
                if terminated or truncated:
                    episode += 1; timestep = 0; last_accepted_timestep = -minimum_episode_spacing
                    observation, _ = environment.reset(seed=seed + source_id * 1_000_000 + episode)
            if accepted != quotas[source_id]:
                raise RuntimeError(
                    f"source_{source_id} produced only {accepted}/{quotas[source_id]} eligible anchors; "
                    "explicitly reduce --behavior-offset or inspect the fixed checkpoint"
                )
        finally:
            environment.close()
    anchors = {
        "anchor_id": np.arange(len(rows), dtype=np.int64),
        "qpos": np.asarray([row["qpos"] for row in rows], dtype=np.float64),
        "qvel": np.asarray([row["qvel"] for row in rows], dtype=np.float64),
        "simulator_state": np.asarray([row["simulator_state"] for row in rows], dtype=np.float64),
        "state_spec": np.asarray([row["state_spec"] for row in rows], dtype=np.int64),
        "wrapper_elapsed_steps": np.asarray([row["wrapper_elapsed_steps"] for row in rows], dtype=np.int64),
        "elapsed_steps": np.asarray([row["elapsed_steps"] for row in rows], dtype=np.int64),
        "public_observation": np.asarray([row["public_observation"] for row in rows], dtype=np.float32),
        "base_action": np.asarray([row["base_action"] for row in rows], dtype=np.float64),
        "anchor_origin_source": np.asarray([row["anchor_origin_source"] for row in rows], dtype=np.int8),
        "anchor_origin_episode": np.asarray([row["anchor_origin_episode"] for row in rows], dtype=np.int64),
        "anchor_origin_timestep": np.asarray([row["anchor_origin_timestep"] for row in rows], dtype=np.int32),
    }
    validate_anchor_pool(anchors, num_anchors)
    return anchors


def anchor_snapshot(anchors: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    encoded = np.asarray(anchors["wrapper_elapsed_steps"][index], dtype=np.int64)
    return {
        "simulator_state": np.asarray(anchors["simulator_state"][index], dtype=np.float64),
        "state_spec": int(anchors["state_spec"][index]),
        "elapsed_steps": int(anchors["elapsed_steps"][index]),
        "wrapper_elapsed_steps": tuple(None if value < 0 else int(value) for value in encoded),
    }


def restore_anchor(
    environment: ConfoundedHopperWrapper, anchors: dict[str, np.ndarray], index: int,
    atol: float = 1e-7, rtol: float = 1e-7,
) -> np.ndarray:
    """Restore without stepping and return the exact existing 12D public observation."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("mujoco is required for Hopper anchor restoration") from exc
    snapshot = anchor_snapshot(anchors, index)
    base = environment.env.unwrapped
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    if snapshot["state_spec"] != int(state_spec):
        raise ValueError("anchor state specification is incompatible")
    mujoco.mj_setState(base.model, base.data, snapshot["simulator_state"], state_spec)
    mujoco.mj_forward(base.model, base.data)
    current: Any = environment.env
    elapsed_values = snapshot["wrapper_elapsed_steps"]
    wrapper_index = 0
    while isinstance(current, gym.Wrapper):
        if wrapper_index >= len(elapsed_values):
            raise ValueError("anchor wrapper stack is incompatible")
        if elapsed_values[wrapper_index] is not None:
            current._elapsed_steps = int(elapsed_values[wrapper_index])
        current = current.env; wrapper_index += 1
    if wrapper_index != len(elapsed_values):
        raise ValueError("anchor wrapper stack is incompatible")
    environment.elapsed_steps = snapshot["elapsed_steps"]
    raw = np.asarray(base._get_obs(), dtype=np.float32)
    public = environment._public_observation(raw)
    if not np.allclose(public, anchors["public_observation"][index], atol=atol, rtol=rtol):
        raise RuntimeError("restored public observation differs from the stored anchor")
    if not np.allclose(base.data.qpos, anchors["qpos"][index], atol=atol, rtol=rtol):
        raise RuntimeError("restored qpos differs from the stored anchor")
    if not np.allclose(base.data.qvel, anchors["qvel"][index], atol=atol, rtol=rtol):
        raise RuntimeError("restored qvel differs from the stored anchor")
    return public


def checkpoint_roundtrip(
    model: Any, loader: Callable[..., Any], public_observations: np.ndarray,
    device: str, atol: float = 1e-7, rtol: float = 1e-7,
) -> dict[str, Any]:
    """Save a temporary copy, reload it, and compare deterministic base actions."""
    before = base_actions_from_source2(model, public_observations)
    with tempfile.TemporaryDirectory(prefix="phase8a_source2_") as directory:
        path = Path(directory) / "source2_roundtrip"
        model.save(str(path))
        reloaded = loader(str(path) + ".zip", device=device)
        after = base_actions_from_source2(reloaded, public_observations)
    maximum = float(np.max(np.abs(before - after)))
    return {"maximum_base_action_difference": maximum,
            "atol": atol, "rtol": rtol,
            "passed": bool(np.allclose(before, after, atol=atol, rtol=rtol))}
