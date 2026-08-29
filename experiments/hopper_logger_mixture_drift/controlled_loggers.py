"""Pure definitions for the three controlled diagnostic loggers."""

from __future__ import annotations

from typing import Any

import numpy as np

from confounded_hopper import ACTUATOR_DIRECTION


LOGGER_NAMES = (
    "diagnostic_logger_1",
    "diagnostic_logger_2",
    "diagnostic_logger_3",
)
ACTION_KEYS = ("minus", "base", "plus")
CONDITIONS = ("confounded", "independent_latents")
MIXTURES = {
    "balanced": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "logger1_heavy": (0.8, 0.1, 0.1),
    "logger2_heavy": (0.1, 0.8, 0.1),
    "logger3_heavy": (0.1, 0.1, 0.8),
}


def latent_pairs(condition: str) -> tuple[tuple[int, int, float], ...]:
    """Return the completely enumerated latent law for one condition."""
    if condition == "confounded":
        return ((-1, -1, 0.5), (1, 1, 0.5))
    if condition == "independent_latents":
        return ((-1, -1, 0.25), (-1, 1, 0.25),
                (1, -1, 0.25), (1, 1, 0.25))
    raise ValueError(f"unknown condition: {condition}")


def policy_observations(public_observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    public = np.asarray(public_observations, dtype=np.float32)
    if public.ndim != 2 or public.shape[1] != 12 or not np.all(np.isfinite(public)):
        raise ValueError("public observations must be a finite [N, 12] array")
    minus = np.concatenate((public, -np.ones((len(public), 1), dtype=np.float32)), axis=1)
    plus = np.concatenate((public, np.ones((len(public), 1), dtype=np.float32)), axis=1)
    return minus, plus


def base_actions_from_source2(model: Any, public_observations: np.ndarray) -> np.ndarray:
    """Average deterministic Source-2 actions under the two behavior latents."""
    minus_observations, plus_observations = policy_observations(public_observations)
    minus, _ = model.predict(minus_observations, deterministic=True)
    plus, _ = model.predict(plus_observations, deterministic=True)
    minus = np.asarray(minus, dtype=np.float64)
    plus = np.asarray(plus, dtype=np.float64)
    if minus.shape != (len(minus_observations), 3) or plus.shape != minus.shape:
        raise RuntimeError("Source 2 checkpoint produced an incompatible action schema")
    result = 0.5 * (minus + plus)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Source 2 checkpoint produced nonfinite base actions")
    return result


def headroom_mask(base_actions: np.ndarray, behavior_offset: float, tolerance: float = 1e-6) -> np.ndarray:
    base = np.asarray(base_actions, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base actions must have shape [N, 3]")
    if not np.isfinite(behavior_offset) or behavior_offset < 0.0:
        raise ValueError("behavior_offset must be finite and nonnegative")
    return np.all(np.abs(base) + behavior_offset * np.abs(ACTUATOR_DIRECTION)
                  <= 1.0 - tolerance, axis=1)


def controlled_action(
    base_action: np.ndarray, logger_id: int, u_behavior: int, behavior_offset: float,
) -> tuple[np.ndarray, str]:
    """Return an unclipped commanded action and its hidden audit key."""
    base = np.asarray(base_action, dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("base_action must be a finite three-vector")
    if logger_id not in (0, 1, 2) or u_behavior not in (-1, 1):
        raise ValueError("logger_id must be 0/1/2 and u_behavior must be -1/+1")
    signs = (u_behavior, -u_behavior, 0)
    signed = signs[logger_id]
    action = base + float(behavior_offset) * signed * ACTUATOR_DIRECTION
    action_key = "base" if signed == 0 else ("plus" if signed == 1 else "minus")
    if np.any(np.abs(action) > 1.0 + 1e-12):
        raise RuntimeError("controlled logger action lacks commanded-action headroom")
    return action, action_key


def target_action(base_action: np.ndarray, action_key: str, behavior_offset: float) -> np.ndarray:
    if action_key not in ACTION_KEYS:
        raise ValueError(f"unknown action key: {action_key}")
    sign = {"minus": -1, "base": 0, "plus": 1}[action_key]
    return np.asarray(base_action, dtype=np.float64) + sign * behavior_offset * ACTUATOR_DIRECTION


def mixture_sample_weights(
    logger_ids: np.ndarray, pair_masses: np.ndarray, anchor_ids: np.ndarray,
    mixture: tuple[float, float, float],
) -> np.ndarray:
    """Reweight one complete master table without deleting or copying rows."""
    logger = np.asarray(logger_ids, dtype=np.int64)
    pair_mass = np.asarray(pair_masses, dtype=np.float64)
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    weights = np.asarray(mixture, dtype=np.float64)
    if logger.shape != pair_mass.shape or logger.shape != anchors.shape or logger.ndim != 1:
        raise ValueError("logger, pair-mass, and anchor arrays must be aligned vectors")
    if weights.shape != (3,) or np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("mixture must be a positive probability vector of length three")
    unique_anchors = np.unique(anchors)
    if not unique_anchors.size or set(np.unique(logger).tolist()) != {0, 1, 2}:
        raise ValueError("master data must contain anchors and all three loggers")
    result = (1.0 / unique_anchors.size) * weights[logger] * pair_mass
    total = float(result.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("mixture weights have invalid total mass")
    return result / total
