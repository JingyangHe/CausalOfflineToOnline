"""Exact probability-mass DGP for non-complementary Hopper loggers."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_KEYS = ("minus", "base", "plus")
CONDITIONS = ("confounded", "independent_latents")
LOGGER_NAMES = ("strong_same_direction", "weak_same_direction", "base_control")
LOGGER_ACTION_PROBABILITIES = {
    0: {-1: {"minus": 0.9, "plus": 0.1},
        1: {"minus": 0.1, "plus": 0.9}},
    1: {-1: {"minus": 0.7, "plus": 0.3},
        1: {"minus": 0.3, "plus": 0.7}},
    2: {-1: {"base": 1.0}, 1: {"base": 1.0}},
}
PRIMARY_MIXTURES = {
    "logger1_heavy": (0.8, 0.1, 0.1),
    "logger12_balanced": (0.45, 0.45, 0.1),
    "logger2_heavy": (0.1, 0.8, 0.1),
}
SECONDARY_MIXTURES = {"all_sources_equal": (1 / 3, 1 / 3, 1 / 3)}
MIXTURES = {**PRIMARY_MIXTURES, **SECONDARY_MIXTURES}

PUBLIC_FIELDS = (
    "row_id", "anchor_id", "observation", "commanded_action", "reward",
    "next_observation", "terminated", "truncated", "logger_id", "condition",
    "kappa_env",
)
HIDDEN_FIELDS = (
    "row_id", "anchor_id", "logger_id", "condition", "action_key",
    "u_behavior", "u_env", "action_probability_given_u", "base_mass",
    "commanded_action", "applied_action", "applied_action_clipped", "reward",
    "next_observation", "terminated", "truncated", "kappa_env",
)
FORBIDDEN_PUBLIC_FIELDS = set(HIDDEN_FIELDS) - {
    "row_id", "anchor_id", "logger_id", "condition", "commanded_action",
    "reward", "next_observation", "terminated", "truncated", "kappa_env",
}


class NonComplementaryDGPError(RuntimeError):
    """Raised when an exact support or probability invariant fails."""


def logger_action_probability(logger_id: int, u_behavior: int, action_key: str) -> float:
    try:
        return float(LOGGER_ACTION_PROBABILITIES[logger_id][u_behavior].get(action_key, 0.0))
    except KeyError as exc:
        raise ValueError("logger_id and u_behavior must be valid") from exc


def logger_action_marginal(logger_id: int, action_key: str) -> float:
    return 0.5 * sum(
        logger_action_probability(logger_id, u, action_key) for u in (-1, 1)
    )


def analytic_u_posterior(condition: str, mixture: str, action_key: str) -> float:
    if condition == "independent_latents":
        return 0.5
    if condition != "confounded" or mixture not in MIXTURES or action_key not in ACTION_KEYS:
        raise ValueError("unknown condition, mixture, or action")
    if action_key == "base":
        return 0.5
    weights = MIXTURES[mixture]
    numerator = sum(
        weights[logger] * 0.5 * logger_action_probability(logger, 1, action_key)
        for logger in (0, 1)
    )
    denominator = sum(
        weights[logger] * logger_action_marginal(logger, action_key)
        for logger in (0, 1)
    )
    return float(numerator / denominator)


def support_specification(condition: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    for logger in (0, 1):
        for u_behavior in (-1, 1):
            u_env_values = (u_behavior,) if condition == "confounded" else (-1, 1)
            latent_mass = 0.5 if condition == "confounded" else 0.25
            for u_env in u_env_values:
                for action in ("minus", "plus"):
                    probability = logger_action_probability(logger, u_behavior, action)
                    rows.append({
                        "logger_id": logger, "u_behavior": u_behavior, "u_env": u_env,
                        "action_key": action, "action_probability_given_u": probability,
                        "base_mass": latent_mass * probability,
                    })
    pairs = ((-1, -1), (1, 1)) if condition == "confounded" else (
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    )
    latent_mass = 0.5 if condition == "confounded" else 0.25
    for u_behavior, u_env in pairs:
        rows.append({
            "logger_id": 2, "u_behavior": u_behavior, "u_env": u_env,
            "action_key": "base", "action_probability_given_u": 1.0,
            "base_mass": latent_mass,
        })
    return tuple(rows)


def build_do_lookup(raw: dict[str, np.ndarray], kappa: float) -> dict[tuple[int, str, int], int]:
    required = {"anchor_id", "action_key", "u_env", "kappa_env", "commanded_action",
                "applied_action", "reward", "next_observation", "terminated", "truncated",
                "applied_action_clipped"}
    if not required.issubset(raw):
        raise NonComplementaryDGPError("Phase 8A do_oracle_raw lacks required fields")
    lookup: dict[tuple[int, str, int], int] = {}
    for row in range(len(raw["anchor_id"])):
        if not np.isclose(float(raw["kappa_env"][row]), kappa, atol=1e-12, rtol=0):
            raise NonComplementaryDGPError("do-oracle row has wrong kappa")
        key = (int(raw["anchor_id"][row]), str(raw["action_key"][row]),
               int(raw["u_env"][row]))
        if key in lookup or key[1] not in ACTION_KEYS or key[2] not in (-1, 1):
            raise NonComplementaryDGPError("do-oracle lookup key is invalid or duplicated")
        lookup[key] = row
    return lookup


def _as_arrays(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    arrays = {field: np.asarray([row[field] for row in rows]) for field in fields}
    for field in ("row_id", "anchor_id"):
        arrays[field] = arrays[field].astype(np.int64)
    for field in ("logger_id", "u_behavior", "u_env"):
        if field in arrays:
            arrays[field] = arrays[field].astype(np.int8)
    for field in ("terminated", "truncated", "applied_action_clipped"):
        if field in arrays:
            arrays[field] = arrays[field].astype(bool)
    for field in ("observation", "next_observation", "commanded_action"):
        if field in arrays:
            arrays[field] = arrays[field].astype(np.float32)
    return arrays


def validate_public_hidden(public: dict[str, np.ndarray], hidden: dict[str, np.ndarray]) -> set[str]:
    if set(public) != set(PUBLIC_FIELDS) or set(hidden) != set(HIDDEN_FIELDS):
        raise NonComplementaryDGPError("public or hidden support schema is invalid")
    leakage = FORBIDDEN_PUBLIC_FIELDS.intersection(public)
    if leakage:
        raise NonComplementaryDGPError(f"hidden fields leaked into public: {sorted(leakage)}")
    n = len(public["row_id"])
    if any(len(value) != n for value in public.values()) or any(
            len(value) != n for value in hidden.values()):
        raise NonComplementaryDGPError("support arrays are not row aligned")
    if not np.array_equal(public["row_id"], hidden["row_id"]):
        raise NonComplementaryDGPError("public and hidden row IDs differ")
    if public["observation"].shape != (n, 12) or public["next_observation"].shape != (n, 12):
        raise NonComplementaryDGPError("public observations must be 12D")
    if public["commanded_action"].shape != (n, 3):
        raise NonComplementaryDGPError("public commanded actions must be 3D")
    if not np.allclose(public["commanded_action"], hidden["commanded_action"]):
        raise NonComplementaryDGPError("public and hidden commanded actions differ")
    return leakage


def generate_support(
    anchors: dict[str, np.ndarray], raw: dict[str, np.ndarray], anchor_ids: np.ndarray,
    condition: str, kappa: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build support only by copying fixed Phase 8A do-oracle outcomes."""
    lookup = build_do_lookup(raw, kappa)
    anchor_lookup = {int(anchor): index for index, anchor in enumerate(anchors["anchor_id"])}
    rows_public: list[dict[str, Any]] = []
    rows_hidden: list[dict[str, Any]] = []
    row_id = 0
    for anchor in np.asarray(anchor_ids, dtype=np.int64):
        if int(anchor) not in anchor_lookup:
            raise NonComplementaryDGPError("requested anchor is absent from Phase 8A")
        observation = np.asarray(
            anchors["public_observation"][anchor_lookup[int(anchor)]], dtype=np.float32
        )
        for spec in support_specification(condition):
            key = (int(anchor), spec["action_key"], spec["u_env"])
            if key not in lookup:
                raise NonComplementaryDGPError(f"do-oracle key is missing: {key}")
            source = lookup[key]
            common = {
                "row_id": row_id, "anchor_id": int(anchor),
                "logger_id": spec["logger_id"], "condition": condition,
                "commanded_action": raw["commanded_action"][source],
                "reward": raw["reward"][source],
                "next_observation": raw["next_observation"][source],
                "terminated": raw["terminated"][source],
                "truncated": raw["truncated"][source], "kappa_env": kappa,
            }
            rows_public.append({"observation": observation, **common})
            rows_hidden.append({
                **common, "action_key": spec["action_key"],
                "u_behavior": spec["u_behavior"], "u_env": spec["u_env"],
                "action_probability_given_u": spec["action_probability_given_u"],
                "base_mass": spec["base_mass"],
                "applied_action": raw["applied_action"][source],
                "applied_action_clipped": raw["applied_action_clipped"][source],
            })
            row_id += 1
    public = _as_arrays(rows_public, PUBLIC_FIELDS)
    hidden = _as_arrays(rows_hidden, HIDDEN_FIELDS)
    validate_public_hidden(public, hidden)
    return public, hidden


def generate_weights(hidden: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    anchors = np.asarray(hidden["anchor_id"], dtype=np.int64)
    logger = np.asarray(hidden["logger_id"], dtype=np.int64)
    base_mass = np.asarray(hidden["base_mass"], dtype=np.float64)
    n_anchors = len(np.unique(anchors))
    if n_anchors == 0:
        raise NonComplementaryDGPError("support contains no anchors")
    result = {}
    for name, mixture in MIXTURES.items():
        values = np.asarray(mixture, dtype=np.float64)[logger] * base_mass / n_anchors
        values /= values.sum()
        result[name] = values
    return result


def weighted_latent_correlation(hidden: dict[str, np.ndarray], weights: np.ndarray) -> float:
    left = np.asarray(hidden["u_behavior"], dtype=np.float64)
    right = np.asarray(hidden["u_env"], dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    left -= mass @ left
    right -= mass @ right
    denominator = np.sqrt((mass @ (left * left)) * (mass @ (right * right)))
    return float((mass @ (left * right)) / denominator)

