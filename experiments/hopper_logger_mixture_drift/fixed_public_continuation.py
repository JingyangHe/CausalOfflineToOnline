"""Fixed hidden-blind continuation policy for the Phase 8A-NC-LH audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .anchor_pool import sha256
from .controlled_loggers import base_actions_from_source2, policy_observations


class ContinuationPolicyError(RuntimeError):
    """Raised when the fixed Source-2 policy cannot be resolved exactly."""


class FixedPublicContinuationPolicy:
    """Average deterministic Source-2 actions at synthetic U=-1 and U=+1."""

    def __init__(self, model: Any) -> None:
        self.model = model
        if getattr(getattr(model, "observation_space", None), "shape", None) != (13,):
            raise ContinuationPolicyError("Source 2 must accept 13D [public observation, U]")
        if getattr(getattr(model, "action_space", None), "shape", None) != (3,):
            raise ContinuationPolicyError("Source 2 must produce 3D actions")

    def batch_actions(self, public_observations: np.ndarray) -> np.ndarray:
        actions = base_actions_from_source2(self.model, public_observations)
        return np.clip(actions, -1.0, 1.0)

    def action(self, public_observation: np.ndarray) -> np.ndarray:
        public = np.asarray(public_observation, dtype=np.float32)
        if public.shape != (12,) or not np.all(np.isfinite(public)):
            raise ValueError("continuation policy requires one finite 12D public observation")
        return self.batch_actions(public[None, :])[0]

    def audit_inputs(self, public_observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expose only the two fixed synthetic policy inputs for tests and auditing."""
        public = np.asarray(public_observation, dtype=np.float32)
        minus, plus = policy_observations(public[None, :])
        return minus[0], plus[0]


def resolve_source2_checkpoint(phase8a_root: Path) -> tuple[Path, dict[str, Any], str]:
    """Resolve the 500k Source-2 checkpoint only from the verified Phase 8A manifest."""
    root = Path(phase8a_root).resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Phase 8A manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    original = manifest.get("source2_original_manifest")
    if not isinstance(original, dict):
        raise ContinuationPolicyError("Phase 8A manifest lacks source2_original_manifest")
    mapping = original.get("source_mapping", {}).get("source_2", {})
    if mapping.get("checkpoint_step") != 500_000:
        raise ContinuationPolicyError("Phase 8A Source 2 is not the fixed 500k checkpoint")
    filename = mapping.get("model_file")
    recorded = manifest.get("source2_checkpoint_path")
    expected_hash = manifest.get("source2_checkpoint_sha256")
    if not isinstance(filename, str) or not filename or not isinstance(recorded, str):
        raise ContinuationPolicyError("Phase 8A Source-2 checkpoint mapping is incomplete")
    checkpoint = Path(recorded)
    if not checkpoint.is_absolute():
        repository = next((parent for parent in (root, *root.parents)
                           if (parent / ".git").exists()), None)
        if repository is None:
            raise ContinuationPolicyError(
                "relative Source-2 checkpoint path requires an identifiable repository root")
        checkpoint = (repository / checkpoint).resolve()
    if checkpoint.name != filename or not checkpoint.is_file():
        raise FileNotFoundError(f"recorded Source-2 checkpoint is unavailable: {checkpoint}")
    actual_hash = sha256(checkpoint)
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise ContinuationPolicyError("Source-2 checkpoint SHA256 differs from Phase 8A")
    if (original.get("public_observation_dimension") != 12
            or original.get("behavior_observation_dimension") != 13
            or original.get("action_dimension") != 3):
        raise ContinuationPolicyError("Source-2 observation/action schema is incompatible")
    return checkpoint.resolve(), original, actual_hash


def resolve_gamma(
    phase8a_manifest: dict[str, Any], explicit_gamma: float | None,
) -> tuple[float, str]:
    """Use one uniquely recorded gamma, otherwise require an explicit CLI value."""
    candidates: list[tuple[str, float]] = []
    for label, source in (("phase8a_manifest", phase8a_manifest),
                          ("source2_original_manifest",
                           phase8a_manifest.get("source2_original_manifest", {}))):
        if isinstance(source, dict) and "gamma" in source:
            try:
                value = float(source["gamma"])
            except (TypeError, ValueError) as exc:
                raise ContinuationPolicyError(f"invalid gamma in {label}") from exc
            candidates.append((label, value))
    unique = {value for _, value in candidates}
    if len(unique) > 1:
        raise ContinuationPolicyError("training manifests contain conflicting gamma values")
    recorded = next(iter(unique)) if unique else None
    if explicit_gamma is None:
        if recorded is None:
            raise ContinuationPolicyError(
                "gamma is absent from the training manifest; pass --gamma explicitly")
        gamma, source = recorded, candidates[0][0]
    else:
        gamma, source = float(explicit_gamma), "explicit_cli"
        if recorded is not None and not np.isclose(gamma, recorded, atol=0.0, rtol=0.0):
            raise ContinuationPolicyError("explicit gamma conflicts with the training manifest")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ContinuationPolicyError("gamma must be finite and in (0, 1]")
    return gamma, source


def verify_continuation_matches_base_actions(
    policy: FixedPublicContinuationPolicy, public_observations: np.ndarray,
    stored_base_actions: np.ndarray, atol: float, rtol: float,
) -> dict[str, Any]:
    predicted = policy.batch_actions(public_observations)
    stored = np.asarray(stored_base_actions, dtype=np.float64)
    if predicted.shape != stored.shape:
        raise ContinuationPolicyError("continuation/base-action arrays have different shapes")
    difference = np.abs(predicted - stored)
    passed = bool(np.allclose(predicted, stored, atol=atol, rtol=rtol))
    if not passed:
        raise ContinuationPolicyError("public continuation does not reproduce anchor base actions")
    return {"passed": True, "maximum_absolute_difference": float(np.max(difference)),
            "rows": int(len(predicted)), "actual_u_used": False,
            "logger_id_used": False}
