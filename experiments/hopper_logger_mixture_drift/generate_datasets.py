"""One-step master datasets, sample weights, and mixture-independent do oracle."""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from confounded_hopper import ACTUATOR_DIRECTION, ConfoundedHopperWrapper
from .anchor_pool import anchor_snapshot, restore_anchor
from .controlled_loggers import (
    ACTION_KEYS,
    CONDITIONS,
    MIXTURES,
    controlled_action,
    latent_pairs,
    mixture_sample_weights,
    target_action,
)


PUBLIC_FIELDS = (
    "row_id", "anchor_id", "observation", "action", "reward", "next_observation",
    "terminated", "truncated", "logger_id", "kappa_env",
)
HIDDEN_FIELDS = (
    "row_id", "anchor_id", "logger_id", "action_key", "u_behavior", "u_env",
    "pair_mass", "commanded_action", "applied_action", "qpos", "qvel",
    "next_qpos", "next_qvel", "kappa_env", "reward", "terminated", "truncated",
    "commanded_action_clipped", "applied_action_clipped",
)
FORBIDDEN_PUBLIC_FIELDS = {
    "u_behavior", "u_env", "pair_mass", "applied_action", "qpos", "qvel",
    "next_qpos", "next_qvel", "action_key", "base_action", "hidden_u",
    "simulator_state", "full_mujoco_state",
}
DO_RAW_FIELDS = (
    "anchor_id", "action_key", "u_env", "commanded_action", "applied_action",
    "reward", "next_observation", "delta_observation", "terminated", "truncated",
    "kappa_env", "applied_action_clipped",
)
DO_SUMMARY_FIELDS = (
    "anchor_id", "action_key", "kappa_env", "do_mean_reward",
    "do_mean_next_observation", "do_mean_delta_observation",
    "do_termination_probability", "do_truncation_probability",
)


class OneStepSimulator(Protocol):
    def step(self, anchor_index: int, commanded_action: np.ndarray,
             u_env: int, kappa_env: float) -> dict[str, Any]: ...


class MujocoOneStepSimulator:
    """Restore the same anchor before every original Hopper-v5 transition."""

    def __init__(self, anchors: dict[str, np.ndarray], kappas: tuple[float, ...], seed: int) -> None:
        self.anchors = anchors
        self.environments: dict[float, ConfoundedHopperWrapper] = {}
        for index, kappa in enumerate(kappas):
            environment = ConfoundedHopperWrapper(
                gym.make("Hopper-v5"), kappa=kappa, expose_confounder=True, audit_info=True
            )
            environment.reset(seed=seed + 10_000 + index)
            self.environments[float(kappa)] = environment

    def step(self, anchor_index: int, commanded_action: np.ndarray,
             u_env: int, kappa_env: float) -> dict[str, Any]:
        environment = self.environments[float(kappa_env)]
        public = restore_anchor(environment, self.anchors, anchor_index)
        transition = environment.audit_step_from_state(
            anchor_snapshot(self.anchors, anchor_index), commanded_action, u_env
        )
        _, reward, terminated, truncated, info = transition
        next_public = np.asarray(info["public_observation"], dtype=np.float32)
        base = environment.env.unwrapped
        preclip = np.asarray(commanded_action, dtype=np.float64) + (
            float(kappa_env) * int(u_env) * ACTUATOR_DIRECTION
        )
        return {
            "observation": public.copy(),
            "commanded_action": np.asarray(commanded_action, dtype=np.float64).copy(),
            "applied_action": np.asarray(info["applied_action"], dtype=np.float64).copy(),
            "reward": float(reward), "next_observation": next_public,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "qpos": np.asarray(self.anchors["qpos"][anchor_index], dtype=np.float64).copy(),
            "qvel": np.asarray(self.anchors["qvel"][anchor_index], dtype=np.float64).copy(),
            "next_qpos": np.asarray(base.data.qpos, dtype=np.float64).copy(),
            "next_qvel": np.asarray(base.data.qvel, dtype=np.float64).copy(),
            "commanded_action_clipped": bool(np.any(np.abs(commanded_action) > 1.0)),
            "applied_action_clipped": bool(np.any(np.abs(preclip) > 1.0)),
        }

    def close(self) -> None:
        for environment in self.environments.values():
            environment.close()


def _as_arrays(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    result = {field: np.asarray([row[field] for row in rows]) for field in fields}
    for field in ("row_id", "anchor_id"):
        if field in result:
            result[field] = result[field].astype(np.int64)
    for field in ("logger_id", "u_behavior", "u_env"):
        if field in result:
            result[field] = result[field].astype(np.int8)
    for field in ("terminated", "truncated", "commanded_action_clipped", "applied_action_clipped"):
        if field in result:
            result[field] = result[field].astype(bool)
    for field in ("observation", "next_observation", "action"):
        if field in result:
            result[field] = result[field].astype(np.float32)
    return result


def validate_public_hidden(public: dict[str, np.ndarray], hidden: dict[str, np.ndarray]) -> set[str]:
    if set(public) != set(PUBLIC_FIELDS):
        raise RuntimeError(f"public fields must be exactly {list(PUBLIC_FIELDS)}")
    if set(hidden) != set(HIDDEN_FIELDS):
        raise RuntimeError(f"hidden fields must be exactly {list(HIDDEN_FIELDS)}")
    leakage = FORBIDDEN_PUBLIC_FIELDS.intersection(public)
    if leakage:
        raise RuntimeError(f"hidden fields leaked into public data: {sorted(leakage)}")
    count = len(public["row_id"])
    if any(len(values) != count for values in public.values()) or any(
        len(values) != count for values in hidden.values()
    ):
        raise RuntimeError("public and hidden arrays are not row-aligned")
    if not np.array_equal(public["row_id"], hidden["row_id"]):
        raise RuntimeError("public and hidden row_id alignment failed")
    if public["observation"].shape != (count, 12) or public["next_observation"].shape != (count, 12):
        raise RuntimeError("public Hopper observations must have shape [N, 12]")
    if public["action"].shape != (count, 3):
        raise RuntimeError("public commanded actions must have shape [N, 3]")
    if not np.allclose(public["action"], hidden["commanded_action"], atol=1e-7, rtol=1e-7):
        raise RuntimeError("public commanded actions are misaligned with hidden audit")
    return leakage


def generate_condition_dataset(
    anchors: dict[str, np.ndarray], condition: str, kappa_env: float,
    behavior_offset: float, simulator: OneStepSimulator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Enumerate every anchor, diagnostic logger, and latent pair."""
    rows_public, rows_hidden = [], []
    row_id = 0
    for anchor_index, anchor_id in enumerate(anchors["anchor_id"]):
        base_action = anchors["base_action"][anchor_index]
        for logger_id in (0, 1, 2):
            for u_behavior, u_env, pair_mass in latent_pairs(condition):
                command, action_key = controlled_action(
                    base_action, logger_id, u_behavior, behavior_offset
                )
                outcome = simulator.step(anchor_index, command, u_env, kappa_env)
                rows_public.append({
                    "row_id": row_id, "anchor_id": anchor_id,
                    "observation": outcome["observation"], "action": command,
                    "reward": outcome["reward"], "next_observation": outcome["next_observation"],
                    "terminated": outcome["terminated"], "truncated": outcome["truncated"],
                    "logger_id": logger_id, "kappa_env": kappa_env,
                })
                rows_hidden.append({
                    "row_id": row_id, "anchor_id": anchor_id, "logger_id": logger_id,
                    "action_key": action_key, "u_behavior": u_behavior, "u_env": u_env,
                    "pair_mass": pair_mass, "commanded_action": command,
                    "applied_action": outcome["applied_action"], "qpos": outcome["qpos"],
                    "qvel": outcome["qvel"], "next_qpos": outcome["next_qpos"],
                    "next_qvel": outcome["next_qvel"], "kappa_env": kappa_env,
                    "reward": outcome["reward"], "terminated": outcome["terminated"],
                    "truncated": outcome["truncated"],
                    "commanded_action_clipped": outcome["commanded_action_clipped"],
                    "applied_action_clipped": outcome["applied_action_clipped"],
                })
                row_id += 1
    public, hidden = _as_arrays(rows_public, PUBLIC_FIELDS), _as_arrays(rows_hidden, HIDDEN_FIELDS)
    validate_public_hidden(public, hidden)
    return public, hidden


def generate_mixture_weights(hidden: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: mixture_sample_weights(
            hidden["logger_id"], hidden["pair_mass"], hidden["anchor_id"], mixture
        )
        for name, mixture in MIXTURES.items()
    }


def generate_do_oracle(
    anchors: dict[str, np.ndarray], kappa_env: float, behavior_offset: float,
    simulator: OneStepSimulator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Enumerate the mixture-independent two-point do(action) distribution."""
    raw_rows = []
    for anchor_index, anchor_id in enumerate(anchors["anchor_id"]):
        observation = np.asarray(anchors["public_observation"][anchor_index], dtype=np.float64)
        for action_key in ACTION_KEYS:
            command = target_action(anchors["base_action"][anchor_index], action_key, behavior_offset)
            for u_env in (-1, 1):
                outcome = simulator.step(anchor_index, command, u_env, kappa_env)
                raw_rows.append({
                    "anchor_id": anchor_id, "action_key": action_key, "u_env": u_env,
                    "commanded_action": command, "applied_action": outcome["applied_action"],
                    "reward": outcome["reward"], "next_observation": outcome["next_observation"],
                    "delta_observation": np.asarray(outcome["next_observation"], dtype=np.float64) - observation,
                    "terminated": outcome["terminated"], "truncated": outcome["truncated"],
                    "kappa_env": kappa_env,
                    "applied_action_clipped": outcome["applied_action_clipped"],
                })
    raw = _as_arrays(raw_rows, DO_RAW_FIELDS)
    summary_rows = []
    for anchor_id in anchors["anchor_id"]:
        for action_key in ACTION_KEYS:
            mask = (raw["anchor_id"] == anchor_id) & (raw["action_key"] == action_key)
            if int(np.sum(mask)) != 2 or set(raw["u_env"][mask].tolist()) != {-1, 1}:
                raise RuntimeError("do oracle does not contain the exact two-point environment latent law")
            # Public observations are stored as float32, but population targets
            # are accumulated in float64.  Accumulate the oracle in the same
            # precision so the exact independent-latent identity is not made
            # dependent on float32 reduction rounding for large Hopper states.
            next_observations = np.asarray(
                raw["next_observation"][mask], dtype=np.float64
            )
            delta_observations = np.asarray(
                raw["delta_observation"][mask], dtype=np.float64
            )
            summary_rows.append({
                "anchor_id": anchor_id, "action_key": action_key, "kappa_env": kappa_env,
                "do_mean_reward": float(np.mean(raw["reward"][mask])),
                "do_mean_next_observation": np.mean(next_observations, axis=0),
                "do_mean_delta_observation": np.mean(delta_observations, axis=0),
                "do_termination_probability": float(np.mean(raw["terminated"][mask])),
                "do_truncation_probability": float(np.mean(raw["truncated"][mask])),
            })
    summary = _as_arrays(summary_rows, DO_SUMMARY_FIELDS)
    return raw, summary


def deterministic_repeat_check(
    simulator: OneStepSimulator, anchor_index: int, command: np.ndarray,
    u_env: int, kappa_env: float, atol: float = 1e-7, rtol: float = 1e-7,
) -> dict[str, Any]:
    first = simulator.step(anchor_index, command, u_env, kappa_env)
    second = simulator.step(anchor_index, command, u_env, kappa_env)
    array_fields = ("applied_action", "next_observation", "next_qpos", "next_qvel")
    scalar_fields = ("reward", "terminated", "truncated")
    passed = all(np.allclose(first[field], second[field], atol=atol, rtol=rtol)
                 for field in array_fields)
    passed = passed and all(first[field] == second[field] for field in scalar_fields)
    return {"passed": bool(passed), "atol": atol, "rtol": rtol,
            "maximum_next_observation_difference": float(np.max(np.abs(
                first["next_observation"] - second["next_observation"]))) }


def all_arrays_finite(*bundles: dict[str, np.ndarray]) -> bool:
    for bundle in bundles:
        for values in bundle.values():
            array = np.asarray(values)
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                return False
    return True
