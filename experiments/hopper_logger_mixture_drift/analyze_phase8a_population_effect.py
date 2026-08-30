"""Read-only population-effect review for a verified Phase 8A artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .controlled_loggers import ACTION_KEYS, CONDITIONS, MIXTURES
from .generate_datasets import FORBIDDEN_PUBLIC_FIELDS, PUBLIC_FIELDS


EXPECTED_ROOT_NAME = "controlled_loggers_seed0_verified"
EXPECTED_KAPPAS = (0.0, 0.1, 0.2, 0.3)
PRIMARY_MIXTURES = ("logger1_heavy", "logger12_midpoint", "logger2_heavy")
SECONDARY_MIXTURES = ("balanced", "logger3_heavy")
ORIGINAL_MIXTURES = tuple(MIXTURES)
ANALYSIS_MIXTURES = PRIMARY_MIXTURES + SECONDARY_MIXTURES
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_KEYS)}
COMPLETION_MARKERS = (
    "PHASE8A_HOPPER_LOGGER_MIXTURE_DGP_COMPLETE",
    "READY_FOR_POOLED_WORLD_MODEL_DRIFT_TRAINING",
)


class PopulationEffectAuditError(RuntimeError):
    """Raised when a required input or scientific hard check fails."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PopulationEffectAuditError(f"missing required JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PopulationEffectAuditError(f"JSON input must contain an object: {path}")
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise PopulationEffectAuditError(f"missing required NPZ input: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _kappa_directory(kappa: float) -> str:
    return f"kappa_{kappa:.2f}".replace(".", "p")


def require_verified_phase8a_root(root: Path) -> Path:
    resolved = Path(root).resolve()
    if resolved.name != EXPECTED_ROOT_NAME or not resolved.is_dir():
        raise PopulationEffectAuditError(
            f"--phase8a-root must be an existing {EXPECTED_ROOT_NAME} directory"
        )
    return resolved


def validate_all_2048_anchors(anchors: dict[str, np.ndarray]) -> np.ndarray:
    if "anchor_id" not in anchors:
        raise PopulationEffectAuditError("anchors.npz has no anchor_id")
    anchor_ids = np.asarray(anchors["anchor_id"], dtype=np.int64)
    if len(anchor_ids) != 2048 or not np.array_equal(anchor_ids, np.arange(2048)):
        raise PopulationEffectAuditError("verified Phase 8A input must contain anchors 0..2047")
    return anchor_ids


def validate_all_four_kappas(manifest: dict[str, Any], root: Path | None = None) -> None:
    kappas = tuple(float(value) for value in manifest.get("kappas", ()))
    if kappas != EXPECTED_KAPPAS:
        raise PopulationEffectAuditError(f"Phase 8A kappas must be exactly {EXPECTED_KAPPAS}")
    if root is not None:
        missing = [name for name in map(_kappa_directory, EXPECTED_KAPPAS)
                   if not (root / name).is_dir()]
        if missing:
            raise PopulationEffectAuditError(f"missing kappa directories: {missing}")


def validate_all_84_phase8a_invariants(summary: dict[str, Any]) -> None:
    invariants = summary.get("all_hard_invariants")
    if not isinstance(invariants, dict) or len(invariants) != 84:
        raise PopulationEffectAuditError("Phase 8A summary must contain exactly 84 hard invariants")
    failed = [name for name, value in invariants.items() if value is not True]
    if failed or summary.get("all_hard_invariants_passed") is not True:
        raise PopulationEffectAuditError(f"Phase 8A hard invariants are not all true: {failed}")


def _find_completion_log(root: Path) -> Path:
    candidates = list(root.glob("*.log"))
    repository = next((path for path in (root, *root.parents) if (path / ".git").exists()), None)
    if repository is not None:
        candidates.extend(repository.glob("phase8a*.log"))
    for path in sorted(set(candidates)):
        text = path.read_text(encoding="utf-8", errors="replace")
        if all(marker in text for marker in COMPLETION_MARKERS):
            return path
    raise PopulationEffectAuditError(
        "could not locate a Phase 8A log containing both required completion markers"
    )


def required_input_paths(root: Path) -> list[Path]:
    paths = [root / "manifest.json", root / "summary.json", root / "anchors.npz",
             root / "mixture_weights.json", _find_completion_log(root)]
    for kappa in EXPECTED_KAPPAS:
        directory = root / _kappa_directory(kappa)
        paths.extend((directory / "do_oracle_raw.npz", directory / "do_oracle_summary.npz"))
        audit = directory / "population_audit.json"
        if audit.is_file():
            paths.append(audit)
        for condition in CONDITIONS:
            paths.extend((directory / f"{condition}_public.npz",
                          directory / f"{condition}_hidden_audit.npz"))
            paths.extend(
                directory / "weights" / condition / f"weights_{mixture}.npy"
                for mixture in ORIGINAL_MIXTURES
            )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PopulationEffectAuditError(f"missing required Phase 8A inputs: {missing}")
    return sorted(set(path.resolve() for path in paths), key=str)


def hash_input_files(paths: Iterable[Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): sha256(Path(path)) for path in paths}


def input_hashes_unchanged(before: dict[str, str], after: dict[str, str]) -> bool:
    return before == after


def all_arrays_finite(*bundles: dict[str, np.ndarray]) -> bool:
    for bundle in bundles:
        for values in bundle.values():
            array = np.asarray(values)
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                return False
    return True


def validate_public_schema(public: dict[str, np.ndarray]) -> None:
    if set(public) != set(PUBLIC_FIELDS):
        raise PopulationEffectAuditError(
            f"public schema differs from Phase 8A: {sorted(public)}"
        )
    if FORBIDDEN_PUBLIC_FIELDS.intersection(public):
        raise PopulationEffectAuditError("public data contains hidden audit fields")
    count = len(public["row_id"])
    if public["observation"].shape != (count, 12):
        raise PopulationEffectAuditError("public observations must have shape [N,12]")
    if public["next_observation"].shape != (count, 12):
        raise PopulationEffectAuditError("public next observations must have shape [N,12]")
    if public["action"].shape != (count, 3):
        raise PopulationEffectAuditError("public commanded actions must have shape [N,3]")
    if any(len(values) != count for values in public.values()):
        raise PopulationEffectAuditError("public arrays are not row aligned")


def validate_weight_array(weights: np.ndarray, row_count: int, atol: float, rtol: float) -> None:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (row_count,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise PopulationEffectAuditError("weight array is invalid or misaligned with public rows")
    if not np.isclose(values.sum(), 1.0, atol=atol, rtol=rtol):
        raise PopulationEffectAuditError("weight array does not sum to one")


def midpoint_weight(
    logger1_heavy: np.ndarray, logger2_heavy: np.ndarray,
) -> np.ndarray:
    left = np.asarray(logger1_heavy, dtype=np.float64)
    right = np.asarray(logger2_heavy, dtype=np.float64)
    if left.shape != right.shape:
        raise PopulationEffectAuditError("heavy-mixture weight arrays are not aligned")
    result = 0.5 * left + 0.5 * right
    mass = float(result.sum())
    if not np.isfinite(mass) or mass <= 0.0:
        raise PopulationEffectAuditError("midpoint weights have invalid mass")
    return result / mass


def logger_masses(weights: np.ndarray, logger_ids: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.asarray(weights, dtype=np.float64)[np.asarray(logger_ids) == logger].sum()
        for logger in (0, 1, 2)
    ])


def _action_bytes(action: np.ndarray) -> bytes:
    return np.ascontiguousarray(action).tobytes()


def recover_exact_action_groups(
    public: dict[str, np.ndarray], hidden: dict[str, np.ndarray], anchor_ids: np.ndarray,
    atol: float, rtol: float, action_reference: np.ndarray | None = None,
) -> dict[tuple[int, str], np.ndarray]:
    count = len(public["row_id"])
    required_hidden = {"row_id", "anchor_id", "action_key", "commanded_action", "u_env",
                       "logger_id"}
    if not required_hidden.issubset(hidden):
        raise PopulationEffectAuditError("hidden audit lacks action-label recovery fields")
    if any(len(hidden[field]) != count for field in required_hidden):
        raise PopulationEffectAuditError("public and hidden audit row counts differ")
    if not np.array_equal(public["row_id"], hidden["row_id"]):
        raise PopulationEffectAuditError("public and hidden row_id alignment failed")
    if not np.allclose(public["action"], hidden["commanded_action"], atol=atol, rtol=rtol):
        raise PopulationEffectAuditError("public action and hidden commanded_action differ")

    selected = set(np.asarray(anchor_ids, dtype=np.int64).tolist())
    exact_groups: dict[tuple[int, bytes], list[int]] = {}
    for index, (anchor, action) in enumerate(zip(public["anchor_id"], public["action"])):
        anchor_int = int(anchor)
        if anchor_int in selected:
            exact_groups.setdefault((anchor_int, _action_bytes(action)), []).append(index)

    reference_labels: dict[tuple[int, bytes], str] = {}
    reference_actions: dict[int, list[tuple[str, np.ndarray]]] = {}
    if action_reference is not None:
        reference = np.asarray(action_reference)
        if reference.shape != (len(anchor_ids), len(ACTION_KEYS), 3):
            raise PopulationEffectAuditError("do-oracle action reference has the wrong shape")
        for anchor_index, anchor in enumerate(anchor_ids):
            for action_index, label in enumerate(ACTION_KEYS):
                key = (int(anchor), _action_bytes(reference[anchor_index, action_index]))
                if key in reference_labels:
                    raise PopulationEffectAuditError("do-oracle commanded actions are not unique")
                reference_labels[key] = label
                reference_actions.setdefault(int(anchor), []).append(
                    (label, reference[anchor_index, action_index])
                )

    labeled: dict[tuple[int, str], np.ndarray] = {}
    per_anchor: dict[int, set[str]] = {anchor: set() for anchor in selected}
    for (anchor, action_blob), indices in exact_groups.items():
        labels = set(np.asarray(hidden["action_key"])[indices].astype(str).tolist())
        if len(labels) != 1:
            raise PopulationEffectAuditError("an exact commanded-action group has multiple action keys")
        hidden_label = labels.pop()
        if reference_labels:
            label = reference_labels.get((anchor, action_blob))
            if label is None:
                public_action = np.asarray(public["action"][indices[0]])
                matches = [candidate for candidate, action in reference_actions[anchor]
                           if np.allclose(public_action, action, atol=atol, rtol=rtol)]
                if len(matches) != 1:
                    raise PopulationEffectAuditError(
                        "public commanded action cannot be uniquely labeled by the do oracle"
                    )
                label = matches[0]
            if label != hidden_label:
                raise PopulationEffectAuditError("do-oracle and hidden action labels disagree")
        else:
            label = hidden_label
        if label not in ACTION_INDEX or (anchor, label) in labeled:
            raise PopulationEffectAuditError("action-key mapping is not unique within anchor")
        labeled[(anchor, label)] = np.asarray(indices, dtype=np.int64)
        per_anchor[anchor].add(label)
    if any(labels != set(ACTION_KEYS) for labels in per_anchor.values()):
        raise PopulationEffectAuditError("each anchor must have exactly minus/base/plus action groups")
    return labeled


def load_condition_weights(
    directory: Path, condition: str, hidden: dict[str, np.ndarray], atol: float, rtol: float,
) -> dict[str, np.ndarray]:
    result = {}
    for mixture in ORIGINAL_MIXTURES:
        path = directory / "weights" / condition / f"weights_{mixture}.npy"
        if not path.is_file():
            raise PopulationEffectAuditError(f"missing weight array: {path}")
        values = np.load(path, allow_pickle=False)
        validate_weight_array(values, len(hidden["row_id"]), atol, rtol)
        result[mixture] = np.asarray(values, dtype=np.float64)
    result["logger12_midpoint"] = midpoint_weight(
        result["logger1_heavy"], result["logger2_heavy"]
    )
    midpoint_mass = logger_masses(result["logger12_midpoint"], hidden["logger_id"])
    if not np.allclose(midpoint_mass, (0.45, 0.45, 0.1), atol=atol, rtol=rtol):
        raise PopulationEffectAuditError(
            f"midpoint logger mass differs from (0.45,0.45,0.1): {midpoint_mass}"
        )
    return result


def recompute_observational_response(
    public: dict[str, np.ndarray], hidden: dict[str, np.ndarray],
    weights: dict[str, np.ndarray], anchor_ids: np.ndarray, atol: float, rtol: float,
    action_reference: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    groups = recover_exact_action_groups(
        public, hidden, anchor_ids, atol, rtol, action_reference
    )
    anchor_lookup = {int(anchor): index for index, anchor in enumerate(anchor_ids)}
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for mixture in ANALYSIS_MIXTURES:
        mixture_weights = np.asarray(weights[mixture], dtype=np.float64)
        selected_mask = np.isin(public["anchor_id"], anchor_ids)
        selected_total = float(mixture_weights[selected_mask].sum())
        if selected_total <= 0.0:
            raise PopulationEffectAuditError("selected anchors have zero mixture mass")
        reward = np.empty((len(anchor_ids), 3), dtype=np.float64)
        next_observation = np.empty((len(anchor_ids), 3, 12), dtype=np.float64)
        delta = np.empty((len(anchor_ids), 3, 11), dtype=np.float64)
        terminated = np.empty((len(anchor_ids), 3), dtype=np.float64)
        truncated = np.empty((len(anchor_ids), 3), dtype=np.float64)
        group_mass = np.empty((len(anchor_ids), 3), dtype=np.float64)
        posterior = np.empty((len(anchor_ids), 3), dtype=np.float64)
        for (anchor, action_key), indices in groups.items():
            anchor_index, action_index = anchor_lookup[anchor], ACTION_INDEX[action_key]
            local = mixture_weights[indices]
            mass = float(local.sum())
            if mass <= 0.0:
                raise PopulationEffectAuditError("an exact anchor/action group has zero mass")
            normalized = local / mass
            current = np.asarray(public["observation"][indices], dtype=np.float64)
            following = np.asarray(public["next_observation"][indices], dtype=np.float64)
            reward[anchor_index, action_index] = normalized @ np.asarray(
                public["reward"][indices], dtype=np.float64
            )
            next_observation[anchor_index, action_index] = np.tensordot(
                normalized, following, axes=(0, 0)
            )
            delta[anchor_index, action_index] = np.tensordot(
                normalized, following[:, :11] - current[:, :11], axes=(0, 0)
            )
            terminated[anchor_index, action_index] = normalized @ np.asarray(
                public["terminated"][indices], dtype=np.float64
            )
            truncated[anchor_index, action_index] = normalized @ np.asarray(
                public["truncated"][indices], dtype=np.float64
            )
            group_mass[anchor_index, action_index] = mass / selected_total
            posterior[anchor_index, action_index] = normalized @ (
                np.asarray(hidden["u_env"])[indices] == 1
            ).astype(np.float64)
        outputs[mixture] = {
            "reward": reward, "next_observation": next_observation, "delta": delta,
            "terminated": terminated, "truncated": truncated, "group_mass": group_mass,
            "posterior_u_plus": posterior,
        }
    return outputs


def verify_primary_state_action_mass(
    observational: dict[str, dict[str, np.ndarray]], atol: float, rtol: float,
) -> dict[str, Any]:
    reference = observational[PRIMARY_MIXTURES[0]]["group_mass"]
    maximum = 0.0
    for mixture in PRIMARY_MIXTURES[1:]:
        difference = float(np.max(np.abs(observational[mixture]["group_mass"] - reference)))
        maximum = max(maximum, difference)
        if not np.allclose(observational[mixture]["group_mass"], reference,
                           atol=atol, rtol=rtol):
            raise PopulationEffectAuditError(
                "PHASE8A_PRIMARY_MIXTURES_FAIL_TO_PRESERVE_STATE_ACTION_MASS"
            )
    conditional = reference / reference.sum(axis=1, keepdims=True)
    expected = np.asarray((0.45, 0.10, 0.45), dtype=np.float64)
    if not np.allclose(conditional, expected, atol=atol, rtol=rtol):
        raise PopulationEffectAuditError(
            "PHASE8A_PRIMARY_MIXTURES_FAIL_TO_PRESERVE_STATE_ACTION_MASS"
        )
    return {"passed": True, "maximum_absolute_key_mass_difference": maximum,
            "action_mass_order": list(ACTION_KEYS),
            "conditional_action_mass": expected.tolist()}


def verify_primary_weights_preserve_exact_groups(
    public: dict[str, np.ndarray], hidden: dict[str, np.ndarray],
    weights: dict[str, np.ndarray], anchor_ids: np.ndarray, atol: float, rtol: float,
    action_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    groups = recover_exact_action_groups(
        public, hidden, anchor_ids, atol, rtol, action_reference
    )
    lookup = {int(anchor): index for index, anchor in enumerate(anchor_ids)}
    observational = {}
    for mixture in PRIMARY_MIXTURES:
        values = np.asarray(weights[mixture], dtype=np.float64)
        mass = np.empty((len(anchor_ids), 3), dtype=np.float64)
        for (anchor, action), indices in groups.items():
            mass[lookup[anchor], ACTION_INDEX[action]] = values[indices].sum()
        mass /= mass.sum()
        observational[mixture] = {"group_mass": mass}
    return verify_primary_state_action_mass(observational, atol, rtol)


def recompute_do_oracle(
    raw: dict[str, np.ndarray], stored: dict[str, np.ndarray],
    anchors: dict[str, np.ndarray], anchor_ids: np.ndarray, kappa: float,
    atol: float, rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {"anchor_id", "action_key", "u_env", "commanded_action", "applied_action",
                "reward", "next_observation", "terminated", "truncated", "kappa_env"}
    if not required.issubset(raw):
        raise PopulationEffectAuditError("do_oracle_raw lacks required fields")
    if "mixture" in raw or "condition" in raw:
        raise PopulationEffectAuditError("do oracle must not depend on mixture or condition")
    if "mixture" in stored or "condition" in stored:
        raise PopulationEffectAuditError("stored do oracle must not depend on mixture or condition")
    anchor_lookup = {int(anchor): index for index, anchor in enumerate(anchor_ids)}
    stored_lookup = {(int(anchor), str(action)): index for index, (anchor, action) in enumerate(
        zip(stored["anchor_id"], stored["action_key"])
    )}
    if len(stored_lookup) != len(stored["anchor_id"]):
        raise PopulationEffectAuditError("do oracle summary keys are not unique")
    n = len(anchor_ids)
    reward_u = np.empty((n, 3, 2), dtype=np.float64)
    next_u = np.empty((n, 3, 2, 12), dtype=np.float64)
    terminated_u = np.empty((n, 3, 2), dtype=np.float64)
    truncated_u = np.empty((n, 3, 2), dtype=np.float64)
    applied_u = np.empty((n, 3, 2, 3), dtype=np.float64)
    command = np.empty((n, 3, 3), dtype=np.float64)
    selected = set(anchor_lookup)
    seen: set[tuple[int, str, int, float]] = set()
    for index in range(len(raw["anchor_id"])):
        anchor = int(raw["anchor_id"][index])
        if anchor not in selected:
            continue
        action_key, u_env = str(raw["action_key"][index]), int(raw["u_env"][index])
        row_kappa = float(raw["kappa_env"][index])
        key = (anchor, action_key, u_env, row_kappa)
        if key in seen or action_key not in ACTION_INDEX or u_env not in (-1, 1):
            raise PopulationEffectAuditError("do oracle raw key is invalid or duplicated")
        if not np.isclose(row_kappa, kappa, atol=atol, rtol=rtol):
            raise PopulationEffectAuditError("do oracle row has the wrong kappa")
        seen.add(key)
        ai, aj, uk = anchor_lookup[anchor], ACTION_INDEX[action_key], int(u_env == 1)
        reward_u[ai, aj, uk] = float(raw["reward"][index])
        next_u[ai, aj, uk] = np.asarray(raw["next_observation"][index], dtype=np.float64)
        terminated_u[ai, aj, uk] = float(raw["terminated"][index])
        truncated_u[ai, aj, uk] = float(raw["truncated"][index])
        applied_u[ai, aj, uk] = np.asarray(raw["applied_action"][index], dtype=np.float64)
        command[ai, aj] = np.asarray(raw["commanded_action"][index], dtype=np.float64)
    expected_keys = n * 3 * 2
    if len(seen) != expected_keys:
        raise PopulationEffectAuditError(
            f"do oracle must uniquely contain {expected_keys} selected anchor/action/U rows"
        )
    anchor_observation = np.asarray(anchors["public_observation"], dtype=np.float64)[anchor_ids]
    delta_u = next_u[..., :11] - anchor_observation[:, None, None, :11]
    result = {
        "reward_u": reward_u, "next_observation_u": next_u, "delta_u": delta_u,
        "terminated_u": terminated_u, "truncated_u": truncated_u,
        "applied_action_u": applied_u,
        "commanded_action": command,
        "mean_reward": reward_u.mean(axis=2),
        "mean_next_observation": next_u.mean(axis=2),
        "mean_delta": delta_u.mean(axis=2),
        "termination_probability": terminated_u.mean(axis=2),
        "truncation_probability": truncated_u.mean(axis=2),
        "reward_u_effect": reward_u[:, :, 1] - reward_u[:, :, 0],
        "delta_u_effect": delta_u[:, :, 1] - delta_u[:, :, 0],
        "termination_u_disagreement": terminated_u[:, :, 1] != terminated_u[:, :, 0],
    }
    maximum = 0.0
    for anchor in anchor_ids:
        for action_key in ACTION_KEYS:
            old_index = stored_lookup.get((int(anchor), action_key))
            if old_index is None:
                raise PopulationEffectAuditError("do oracle summary is missing a selected key")
            ai, aj = anchor_lookup[int(anchor)], ACTION_INDEX[action_key]
            comparisons = (
                (result["mean_reward"][ai, aj], stored["do_mean_reward"][old_index]),
                (result["mean_next_observation"][ai, aj],
                 stored["do_mean_next_observation"][old_index]),
                (result["mean_delta"][ai, aj],
                 stored["do_mean_delta_observation"][old_index][:11]),
                (result["termination_probability"][ai, aj],
                 stored["do_termination_probability"][old_index]),
                (result["truncation_probability"][ai, aj],
                 stored["do_truncation_probability"][old_index]),
            )
            for new, old in comparisons:
                difference = float(np.max(np.abs(
                    np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)
                )))
                maximum = max(maximum, difference)
                if not np.allclose(new, old, atol=atol, rtol=rtol):
                    raise PopulationEffectAuditError("raw and stored do oracle summaries disagree")
    return result, {"passed": True, "maximum_absolute_difference": maximum,
                    "oracle_depends_on_condition": False,
                    "oracle_depends_on_mixture": False}


def analytic_u_posterior(condition: str, mixture: str) -> np.ndarray:
    if condition == "independent_latents":
        return np.asarray((0.5, 0.5, 0.5), dtype=np.float64)
    if condition != "confounded":
        raise ValueError(f"unknown condition: {condition}")
    if mixture == "logger1_heavy":
        return np.asarray((1.0 / 9.0, 0.5, 8.0 / 9.0))
    if mixture == "logger12_midpoint":
        return np.asarray((0.5, 0.5, 0.5))
    if mixture == "logger2_heavy":
        return np.asarray((8.0 / 9.0, 0.5, 1.0 / 9.0))
    raise ValueError(f"no primary analytic posterior for {mixture}")


def verify_u_posteriors(
    observational_by_condition: dict[str, dict[str, dict[str, np.ndarray]]],
    atol: float, rtol: float,
) -> dict[str, Any]:
    maximum = 0.0
    details = {}
    for condition in CONDITIONS:
        details[condition] = {}
        for mixture in PRIMARY_MIXTURES:
            empirical = observational_by_condition[condition][mixture]["posterior_u_plus"]
            expected = analytic_u_posterior(condition, mixture)
            difference = float(np.max(np.abs(empirical - expected[None, :])))
            maximum = max(maximum, difference)
            details[condition][mixture] = {
                action: {"analytic": float(expected[ACTION_INDEX[action]]),
                         "empirical_mean": float(np.mean(empirical[:, ACTION_INDEX[action]])),
                         "maximum_absolute_difference": float(np.max(np.abs(
                             empirical[:, ACTION_INDEX[action]] - expected[ACTION_INDEX[action]]
                         )))}
                for action in ACTION_KEYS
            }
            if not np.allclose(empirical, expected[None, :], atol=atol, rtol=rtol):
                raise PopulationEffectAuditError(
                    f"weighted U posterior disagrees with analytic value: {condition}/{mixture}"
                )
    return {"passed": True, "maximum_absolute_difference": maximum, "details": details}


def _maximum_pairwise_l2(values: np.ndarray) -> np.ndarray:
    distances = [np.linalg.norm(values[:, left] - values[:, right], axis=-1)
                 for left in range(values.shape[1]) for right in range(left + 1, values.shape[1])]
    return np.max(np.stack(distances, axis=1), axis=1)


def top_action_masks(rewards: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    maximum = np.max(values, axis=1, keepdims=True)
    tied = np.isclose(values, maximum, atol=atol, rtol=rtol)
    return np.sum(tied.astype(np.uint8) * (1 << np.arange(3, dtype=np.uint8)), axis=1)


def descriptive(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise PopulationEffectAuditError("descriptive statistic input must be a finite vector")
    quantiles = np.quantile(array, (0.10, 0.25, 0.50, 0.75, 0.90))
    return {
        "n_anchors": int(len(array)), "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(quantiles[2]), "p10": float(quantiles[0]),
        "p25": float(quantiles[1]), "p75": float(quantiles[3]),
        "p90": float(quantiles[4]), "maximum": float(np.max(array)),
    }


def paired_cluster_bootstrap_means(
    metric_vectors: list[np.ndarray], repetitions: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    matrix = np.column_stack([np.asarray(values, dtype=np.float64) for values in metric_vectors])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise PopulationEffectAuditError("bootstrap metrics must form a finite anchor matrix")
    n = matrix.shape[0]
    if n == 0:
        raise PopulationEffectAuditError("bootstrap requires at least one anchor")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=repetitions)
    estimates = counts @ matrix / n
    return np.quantile(estimates, 0.025, axis=0), np.quantile(estimates, 0.975, axis=0)


def _metric_spec(
    family: str, kappa: float, condition: str, action: str, mixture: str,
    metric: str, values: np.ndarray,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise PopulationEffectAuditError("aggregate metrics must be one value per anchor")
    return {"family": family, "kappa": float(kappa), "condition": condition,
            "action": action, "mixture": mixture, "metric": metric, "values": array}


def aggregate_metric_specs(
    specs: list[dict[str, Any]], bootstrap_reps: int, seed: int,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    lows, highs = paired_cluster_bootstrap_means(
        [spec["values"] for spec in specs], bootstrap_reps, seed
    )
    rows = []
    for index, spec in enumerate(specs):
        row = {key: value for key, value in spec.items() if key != "values"}
        row.update(descriptive(spec["values"]))
        row.update(ci95_low=float(lows[index]), ci95_high=float(highs[index]),
                   bootstrap_unit="anchor_id", bootstrap_repetitions=bootstrap_reps,
                   bootstrap_seed=seed)
        rows.append(row)
    return rows


def _all_actions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.mean(array, axis=1)


def analyze_kappa(
    root: Path, kappa: float, anchors: dict[str, np.ndarray], anchor_ids: np.ndarray,
    atol: float, rtol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, bool]]:
    directory = root / _kappa_directory(kappa)
    raw, stored = load_npz(directory / "do_oracle_raw.npz"), load_npz(
        directory / "do_oracle_summary.npz"
    )
    do, do_audit = recompute_do_oracle(raw, stored, anchors, anchor_ids, kappa, atol, rtol)
    observational_by_condition = {}
    mass_audits, public_bundles, hidden_bundles = {}, {}, {}
    for condition in CONDITIONS:
        public = load_npz(directory / f"{condition}_public.npz")
        hidden = load_npz(directory / f"{condition}_hidden_audit.npz")
        validate_public_schema(public)
        if not all_arrays_finite(public, hidden):
            raise PopulationEffectAuditError(f"nonfinite public/hidden data: {kappa}/{condition}")
        weights = load_condition_weights(directory, condition, hidden, atol, rtol)
        observational = recompute_observational_response(
            public, hidden, weights, anchor_ids, atol, rtol, do["commanded_action"]
        )
        mass_audits[condition] = verify_primary_state_action_mass(observational, atol, rtol)
        observational_by_condition[condition] = observational
        public_bundles[condition], hidden_bundles[condition] = public, hidden
    posterior_audit = verify_u_posteriors(observational_by_condition, atol, rtol)

    reward_effect = do["reward_u_effect"]
    reward_effect_abs = np.abs(reward_effect)
    delta_effect_l2 = np.linalg.norm(do["delta_u_effect"], axis=2)
    termination_disagreement = do["termination_u_disagreement"].astype(np.float64)
    specs: list[dict[str, Any]] = []
    for action in ACTION_KEYS:
        ai = ACTION_INDEX[action]
        specs.extend((
            _metric_spec("u_outcome_effect", kappa, "do_oracle", action, "none",
                         "reward_u_effect_abs", reward_effect_abs[:, ai]),
            _metric_spec("u_outcome_effect", kappa, "do_oracle", action, "none",
                         "delta_u_effect_l2", delta_effect_l2[:, ai]),
            _metric_spec("u_outcome_effect", kappa, "do_oracle", action, "none",
                         "termination_u_disagreement", termination_disagreement[:, ai]),
        ))
    specs.extend((
        _metric_spec("u_outcome_effect", kappa, "do_oracle", "all", "none",
                     "reward_u_effect_abs", _all_actions(reward_effect_abs)),
        _metric_spec("u_outcome_effect", kappa, "do_oracle", "all", "none",
                     "delta_u_effect_l2", _all_actions(delta_effect_l2)),
        _metric_spec("u_outcome_effect", kappa, "do_oracle", "all", "none",
                     "termination_u_disagreement", _all_actions(termination_disagreement)),
    ))

    primary_metrics = {}
    ranking = {}
    for condition in CONDITIONS:
        obs = observational_by_condition[condition]
        rewards = np.stack([obs[m]["reward"] for m in PRIMARY_MIXTURES], axis=1)
        deltas = np.stack([obs[m]["delta"] for m in PRIMARY_MIXTURES], axis=1)
        signed_reward = obs["logger1_heavy"]["reward"] - obs["logger2_heavy"]["reward"]
        absolute_reward = np.abs(signed_reward)
        reward_range = np.max(rewards, axis=1) - np.min(rewards, axis=1)
        delta_heavy = np.linalg.norm(
            obs["logger1_heavy"]["delta"] - obs["logger2_heavy"]["delta"], axis=2
        )
        delta_range = np.empty((len(anchor_ids), 3), dtype=np.float64)
        for action_index in range(3):
            delta_range[:, action_index] = _maximum_pairwise_l2(
                deltas[:, :, action_index, :]
            )
        primary_metrics[condition] = {
            "signed_reward_drift": signed_reward,
            "absolute_reward_drift": absolute_reward,
            "three_mixture_reward_range": reward_range,
            "delta_drift_heavy_contrast": delta_heavy,
            "three_mixture_delta_drift": delta_range,
        }
        for action in (*ACTION_KEYS, "all"):
            if action == "all":
                selectors = {name: _all_actions(values) for name, values in (
                    ("signed_reward_drift", signed_reward),
                    ("absolute_reward_drift", absolute_reward),
                    ("three_mixture_reward_range", reward_range),
                    ("delta_drift_heavy_contrast", delta_heavy),
                    ("three_mixture_delta_drift", delta_range),
                )}
            else:
                ai = ACTION_INDEX[action]
                selectors = {name: values[:, ai] for name, values in (
                    ("signed_reward_drift", signed_reward),
                    ("absolute_reward_drift", absolute_reward),
                    ("three_mixture_reward_range", reward_range),
                    ("delta_drift_heavy_contrast", delta_heavy),
                    ("three_mixture_delta_drift", delta_range),
                )}
            for metric, values in selectors.items():
                specs.append(_metric_spec(
                    "primary_mixture_drift", kappa, condition, action,
                    "logger1_heavy_vs_logger2_heavy", metric, values
                ))

        for mixture in PRIMARY_MIXTURES:
            reward_error = obs[mixture]["reward"] - do["mean_reward"]
            delta_error = np.linalg.norm(obs[mixture]["delta"] - do["mean_delta"], axis=2)
            termination_error = np.abs(
                obs[mixture]["terminated"] - do["termination_probability"]
            )
            for action in (*ACTION_KEYS, "all"):
                if action == "all":
                    values_by_metric = {
                        "reward_do_error_signed": _all_actions(reward_error),
                        "reward_do_error_abs": _all_actions(np.abs(reward_error)),
                        "delta_do_error_l2": _all_actions(delta_error),
                        "termination_do_error_abs": _all_actions(termination_error),
                    }
                else:
                    ai = ACTION_INDEX[action]
                    values_by_metric = {
                        "reward_do_error_signed": reward_error[:, ai],
                        "reward_do_error_abs": np.abs(reward_error[:, ai]),
                        "delta_do_error_l2": delta_error[:, ai],
                        "termination_do_error_abs": termination_error[:, ai],
                    }
                for metric, values in values_by_metric.items():
                    specs.append(_metric_spec(
                        "do_error", kappa, condition, action, mixture, metric, values
                    ))

        masks = {mixture: top_action_masks(obs[mixture]["reward"], atol, rtol)
                 for mixture in PRIMARY_MIXTURES}
        oracle_masks = top_action_masks(do["mean_reward"], atol, rtol)
        heavy_different = masks["logger1_heavy"] != masks["logger2_heavy"]
        heavy_disjoint = (masks["logger1_heavy"] & masks["logger2_heavy"]) == 0
        ranking[condition] = {"top_action_mask": masks, "do_top_action_mask": oracle_masks,
                              "heavy_different": heavy_different,
                              "heavy_disjoint": heavy_disjoint}
        specs.extend((
            _metric_spec("action_ranking", kappa, condition, "all",
                         "logger1_heavy_vs_logger2_heavy", "top_set_different",
                         heavy_different.astype(np.float64)),
            _metric_spec("action_ranking", kappa, condition, "all",
                         "logger1_heavy_vs_logger2_heavy", "strict_flip",
                         heavy_disjoint.astype(np.float64)),
        ))
        for mixture in PRIMARY_MIXTURES:
            specs.append(_metric_spec(
                "action_ranking", kappa, condition, "all", mixture,
                "top_set_differs_from_do", (masks[mixture] != oracle_masks).astype(np.float64)
            ))
            for action in ACTION_KEYS:
                bit = 1 << ACTION_INDEX[action]
                specs.append(_metric_spec(
                    "action_ranking", kappa, condition, action, mixture,
                    "fraction_action_in_top_set", ((masks[mixture] & bit) != 0).astype(np.float64)
                ))

    reward_excess = (primary_metrics["confounded"]["absolute_reward_drift"]
                     - primary_metrics["independent_latents"]["absolute_reward_drift"])
    delta_excess = (primary_metrics["confounded"]["delta_drift_heavy_contrast"]
                    - primary_metrics["independent_latents"]["delta_drift_heavy_contrast"])
    for action in (*ACTION_KEYS, "all"):
        if action == "all":
            reward_values, delta_values = _all_actions(reward_excess), _all_actions(delta_excess)
        else:
            ai = ACTION_INDEX[action]
            reward_values, delta_values = reward_excess[:, ai], delta_excess[:, ai]
        specs.extend((
            _metric_spec("confounding_excess", kappa, "confounded_minus_independent",
                         action, "heavy_contrast", "reward_excess_drift",
                         reward_values),
            _metric_spec("confounding_excess", kappa, "confounded_minus_independent",
                         action, "heavy_contrast", "delta_excess_drift",
                         delta_values),
        ))
    specs.extend((
        _metric_spec(
            "action_ranking", kappa, "confounded_minus_independent", "all",
            "logger1_heavy_vs_logger2_heavy", "top_set_difference_rate_change",
            ranking["confounded"]["heavy_different"].astype(float)
            - ranking["independent_latents"]["heavy_different"].astype(float),
        ),
        _metric_spec(
            "action_ranking", kappa, "confounded_minus_independent", "all",
            "logger1_heavy_vs_logger2_heavy", "strict_flip_rate_change",
            ranking["confounded"]["heavy_disjoint"].astype(float)
            - ranking["independent_latents"]["heavy_disjoint"].astype(float),
        ),
    ))

    for condition in CONDITIONS:
        original_reward = np.stack(
            [observational_by_condition[condition][m]["reward"] for m in ORIGINAL_MIXTURES],
            axis=1,
        )
        original_delta = np.stack(
            [observational_by_condition[condition][m]["delta"] for m in ORIGINAL_MIXTURES],
            axis=1,
        )
        original_reward_range = np.max(original_reward, axis=1) - np.min(original_reward, axis=1)
        original_delta_range = np.empty((len(anchor_ids), 3), dtype=np.float64)
        for action_index in range(3):
            original_delta_range[:, action_index] = _maximum_pairwise_l2(
                original_delta[:, :, action_index, :]
            )
        specs.extend((
            _metric_spec("secondary_logger_and_action_mixture_shift", kappa, condition,
                         "all", "balanced/logger3_heavy included",
                         "original_four_mixture_reward_range", _all_actions(original_reward_range)),
            _metric_spec("secondary_logger_and_action_mixture_shift", kappa, condition,
                         "all", "balanced/logger3_heavy included",
                         "original_four_mixture_delta_range", _all_actions(original_delta_range)),
        ))
        for mixture in SECONDARY_MIXTURES:
            reward_error = np.abs(
                observational_by_condition[condition][mixture]["reward"] - do["mean_reward"]
            )
            delta_error = np.linalg.norm(
                observational_by_condition[condition][mixture]["delta"] - do["mean_delta"],
                axis=2,
            )
            specs.extend((
                _metric_spec("secondary_logger_and_action_mixture_shift", kappa, condition,
                             "all", mixture, "reward_do_error_abs", _all_actions(reward_error)),
                _metric_spec("secondary_logger_and_action_mixture_shift", kappa, condition,
                             "all", mixture, "delta_do_error_l2", _all_actions(delta_error)),
            ))

    signs = np.asarray((-1.0, 0.0, 1.0), dtype=np.float64)
    expected_reward_identity = (7.0 / 9.0) * reward_effect * signs[None, :]
    expected_delta_identity = (7.0 / 9.0) * do["delta_u_effect"] * signs[None, :, None]
    actual_reward_identity = primary_metrics["confounded"]["signed_reward_drift"]
    actual_delta_identity = (
        observational_by_condition["confounded"]["logger1_heavy"]["delta"]
        - observational_by_condition["confounded"]["logger2_heavy"]["delta"]
    )
    reward_identity_residual = actual_reward_identity - expected_reward_identity
    delta_identity_residual = actual_delta_identity - expected_delta_identity

    max_drift = np.max(
        primary_metrics["confounded"]["three_mixture_reward_range"], axis=1
    )
    action_gap = np.max(do["mean_reward"], axis=1) - np.min(do["mean_reward"], axis=1)
    specs.extend((
        _metric_spec("decision_scale", kappa, "confounded", "all", "primary",
                     "max_action_mixture_drift", max_drift),
        _metric_spec("decision_scale", kappa, "do_oracle", "all", "none",
                     "do_action_gap", action_gap),
        _metric_spec("decision_scale", kappa, "confounded", "all", "primary",
                     "fraction_drift_greater_than_action_gap", (max_drift > action_gap).astype(float)),
    ))

    midpoint = observational_by_condition["confounded"]["logger12_midpoint"]
    independent_checks = []
    for mixture in PRIMARY_MIXTURES:
        current = observational_by_condition["independent_latents"][mixture]
        independent_checks.extend((
            np.allclose(current["reward"], do["mean_reward"], atol=atol, rtol=rtol),
            np.allclose(current["delta"], do["mean_delta"], atol=atol, rtol=rtol),
            np.allclose(current["terminated"], do["termination_probability"], atol=atol, rtol=rtol),
            np.allclose(current["truncated"], do["truncation_probability"], atol=atol, rtol=rtol),
        ))
        independent_checks.extend((
            not np.any(ranking["independent_latents"]["heavy_different"]),
            np.array_equal(
                ranking["independent_latents"]["top_action_mask"][mixture],
                ranking["independent_latents"]["do_top_action_mask"],
            ),
        ))
    base_index = ACTION_INDEX["base"]
    base_checks = []
    for condition in CONDITIONS:
        for mixture in PRIMARY_MIXTURES:
            current = observational_by_condition[condition][mixture]
            base_checks.extend((
                np.allclose(current["reward"][:, base_index], do["mean_reward"][:, base_index],
                            atol=atol, rtol=rtol),
                np.allclose(current["delta"][:, base_index], do["mean_delta"][:, base_index],
                            atol=atol, rtol=rtol),
                np.allclose(current["terminated"][:, base_index],
                            do["termination_probability"][:, base_index], atol=atol, rtol=rtol),
                np.allclose(current["truncated"][:, base_index],
                            do["truncation_probability"][:, base_index], atol=atol, rtol=rtol),
            ))
    kappa_zero = True
    if kappa == 0.0:
        kappa_zero = bool(
            np.allclose(do["applied_action_u"][:, :, 0], do["applied_action_u"][:, :, 1],
                        atol=atol, rtol=rtol)
            and np.allclose(do["next_observation_u"][:, :, 0],
                            do["next_observation_u"][:, :, 1], atol=atol, rtol=rtol)
            and np.allclose(do["reward_u_effect"], 0.0, atol=atol, rtol=rtol)
            and np.allclose(do["delta_u_effect"], 0.0, atol=atol, rtol=rtol)
            and np.all(do["terminated_u"][:, :, 0] == do["terminated_u"][:, :, 1])
            and np.all(do["truncated_u"][:, :, 0] == do["truncated_u"][:, :, 1])
            and all(
                np.allclose(observational_by_condition[c][m]["reward"], do["mean_reward"],
                            atol=atol, rtol=rtol)
                and np.allclose(observational_by_condition[c][m]["delta"], do["mean_delta"],
                                atol=atol, rtol=rtol)
                and np.allclose(observational_by_condition[c][m]["terminated"],
                                do["termination_probability"], atol=atol, rtol=rtol)
                and np.allclose(observational_by_condition[c][m]["truncated"],
                                do["truncation_probability"], atol=atol, rtol=rtol)
                for c in CONDITIONS for m in PRIMARY_MIXTURES
            )
            and all(
                np.array_equal(
                    ranking[c]["top_action_mask"][m], ranking[c]["do_top_action_mask"]
                )
                for c in CONDITIONS for m in PRIMARY_MIXTURES
            )
        )
    checks = {
        "primary_state_action_mass_preserved": all(item["passed"] for item in mass_audits.values()),
        "do_raw_summary_agreement": do_audit["passed"],
        "do_oracle_mixture_and_condition_independent": (
            not do_audit["oracle_depends_on_condition"] and not do_audit["oracle_depends_on_mixture"]),
        "u_posterior_matches_analytic_values": posterior_audit["passed"],
        "kappa_zero_negative_control": kappa_zero,
        "independent_population_equals_do": all(independent_checks),
        "base_action_is_primary_mixture_invariant_and_equals_do": all(base_checks),
        "midpoint_population_equals_do_in_complementary_dgp": bool(
            np.allclose(midpoint["reward"], do["mean_reward"], atol=atol, rtol=rtol)
            and np.allclose(midpoint["delta"], do["mean_delta"], atol=atol, rtol=rtol)
            and np.allclose(midpoint["terminated"], do["termination_probability"],
                            atol=atol, rtol=rtol)
            and np.allclose(midpoint["truncated"], do["truncation_probability"],
                            atol=atol, rtol=rtol)),
        "reward_drift_identity": bool(np.allclose(
            actual_reward_identity, expected_reward_identity, atol=atol, rtol=rtol)),
        "delta_drift_identity": bool(np.allclose(
            actual_delta_identity, expected_delta_identity, atol=atol, rtol=rtol)),
        "public_schema_has_no_hidden_leakage": all(
            not FORBIDDEN_PUBLIC_FIELDS.intersection(public_bundles[c]) for c in CONDITIONS),
        "all_arrays_finite": bool(all_arrays_finite(raw, stored, *public_bundles.values(),
                                                     *hidden_bundles.values())),
    }
    arrays = {
        "anchor_id": np.asarray(anchor_ids, dtype=np.int64),
        "reward_u_effect": reward_effect, "reward_u_effect_abs": reward_effect_abs,
        "delta_u_effect": do["delta_u_effect"], "delta_u_effect_l2": delta_effect_l2,
        "termination_u_disagreement": do["termination_u_disagreement"],
        "do_mean_reward": do["mean_reward"], "do_mean_delta": do["mean_delta"],
        "do_termination_probability": do["termination_probability"],
        "do_truncation_probability": do["truncation_probability"],
        "confounded_signed_reward_drift": primary_metrics["confounded"]["signed_reward_drift"],
        "confounded_absolute_reward_drift": primary_metrics["confounded"]["absolute_reward_drift"],
        "confounded_three_mixture_reward_range": primary_metrics["confounded"]["three_mixture_reward_range"],
        "confounded_delta_drift_heavy_contrast": primary_metrics["confounded"]["delta_drift_heavy_contrast"],
        "independent_absolute_reward_drift": primary_metrics["independent_latents"]["absolute_reward_drift"],
        "independent_delta_drift_heavy_contrast": primary_metrics["independent_latents"]["delta_drift_heavy_contrast"],
        "reward_identity_residual": reward_identity_residual,
        "delta_identity_residual": delta_identity_residual,
        "max_action_mixture_drift": max_drift, "do_action_gap": action_gap,
    }
    for condition in CONDITIONS:
        for mixture in ANALYSIS_MIXTURES:
            prefix = f"{condition}_{mixture}"
            arrays[f"{prefix}_reward"] = observational_by_condition[condition][mixture]["reward"]
            arrays[f"{prefix}_delta"] = observational_by_condition[condition][mixture]["delta"]
            arrays[f"{prefix}_termination_probability"] = observational_by_condition[condition][mixture]["terminated"]
            arrays[f"{prefix}_truncation_probability"] = observational_by_condition[condition][mixture]["truncated"]
            arrays[f"{prefix}_state_action_mass"] = observational_by_condition[condition][mixture]["group_mass"]
            arrays[f"{prefix}_posterior_u_plus"] = observational_by_condition[condition][mixture]["posterior_u_plus"]
        for mixture in PRIMARY_MIXTURES:
            prefix = f"{condition}_{mixture}"
            arrays[f"{prefix}_top_action_mask"] = ranking[condition]["top_action_mask"][mixture]
        arrays[f"{condition}_do_top_action_mask"] = ranking[condition]["do_top_action_mask"]
        arrays[f"{condition}_heavy_top_set_different"] = ranking[condition]["heavy_different"]
        arrays[f"{condition}_heavy_strict_flip"] = ranking[condition]["heavy_disjoint"]
    context = {"do": do, "observational": observational_by_condition,
               "mass_audits": mass_audits, "posterior_audit": posterior_audit,
               "primary_metrics": primary_metrics, "ranking": ranking,
               "max_action_mixture_drift": max_drift, "do_action_gap": action_gap}
    return context, specs, arrays, checks


def crosscheck_existing_summary(
    existing: dict[str, Any], contexts: dict[float, dict[str, Any]],
    atol: float, rtol: float, full_anchor_analysis: bool,
) -> dict[str, Any]:
    if not full_anchor_analysis:
        return {
            "passed": True, "entries": [],
            "not_comparable": ["ALL_PHASE8A_SUMMARY_METRICS: smoke uses an anchor subset"],
        }
    entries = []

    def add(path: str, old: float, new: float) -> None:
        difference = abs(float(old) - float(new))
        entries.append({"metric": path, "old_value": float(old), "new_value": float(new),
                        "absolute_difference": difference,
                        "passed": bool(np.isclose(old, new, atol=atol, rtol=rtol))})

    for kappa, context in contexts.items():
        key = _kappa_directory(kappa)
        old_kappa = existing.get("by_kappa", {}).get(key)
        if not isinstance(old_kappa, dict):
            raise PopulationEffectAuditError(f"existing summary lacks {key}")
        do = context["do"]
        add(f"{key}.outcome_strength.mean_absolute_reward_u_difference",
            old_kappa["outcome_strength"]["mean_absolute_reward_u_difference"],
            np.mean(np.abs(do["reward_u_effect"])))
        add(f"{key}.outcome_strength.mean_next_observation_u_difference_l2",
            old_kappa["outcome_strength"]["mean_next_observation_u_difference_l2"],
            np.mean(np.linalg.norm(
                do["next_observation_u"][:, :, 1] - do["next_observation_u"][:, :, 0], axis=2
            )))
        add(f"{key}.outcome_strength.termination_disagreement_rate",
            old_kappa["outcome_strength"]["termination_disagreement_rate"],
            np.mean(do["termination_u_disagreement"]))
        for condition in CONDITIONS:
            obs = context["observational"][condition]
            old_population = old_kappa["population"][condition]
            rewards = np.stack([obs[m]["reward"] for m in ORIGINAL_MIXTURES], axis=1)
            reward_drift = (np.max(rewards, axis=1) - np.min(rewards, axis=1)).reshape(-1)
            next_values = np.stack(
                [obs[m]["next_observation"] for m in ORIGINAL_MIXTURES], axis=1
            )
            next_drift = np.empty((len(next_values), 3), dtype=np.float64)
            for action_index in range(3):
                next_drift[:, action_index] = _maximum_pairwise_l2(
                    next_values[:, :, action_index, :]
                )
            next_drift = next_drift.reshape(-1)
            reward_do_errors = np.stack([
                np.abs(obs[m]["reward"] - do["mean_reward"]) for m in ORIGINAL_MIXTURES
            ], axis=1).mean(axis=1).reshape(-1)
            next_do_errors = np.stack([
                np.linalg.norm(obs[m]["next_observation"] - do["mean_next_observation"], axis=2)
                for m in ORIGINAL_MIXTURES
            ], axis=1).mean(axis=1).reshape(-1)
            termination_do_errors = np.stack([
                np.abs(obs[m]["terminated"] - do["termination_probability"])
                for m in ORIGINAL_MIXTURES
            ], axis=1).mean(axis=1).reshape(-1)
            for statistic, function in (("mean", np.mean), ("median", np.median),
                                        ("maximum", np.max)):
                add(f"{key}.{condition}.reward_mixture_drift.{statistic}",
                    old_population["reward_mixture_drift"][statistic], function(reward_drift))
                add(f"{key}.{condition}.next_state_delta_mixture_drift_l2.{statistic}",
                    old_population["next_state_delta_mixture_drift_l2"][statistic],
                    function(next_drift))
            for statistic, function in (("mean", np.mean), ("maximum", np.max)):
                add(f"{key}.{condition}.reward_do_error_absolute.{statistic}",
                    old_population["reward_do_error_absolute"][statistic],
                    function(reward_do_errors))
                add(f"{key}.{condition}.next_state_delta_do_error_l2.{statistic}",
                    old_population["next_state_delta_do_error_l2"][statistic],
                    function(next_do_errors))
                add(f"{key}.{condition}.termination_probability_do_error_absolute.{statistic}",
                    old_population["termination_probability_do_error_absolute"][statistic],
                    function(termination_do_errors))
            old_argmax = np.stack(
                [np.argmax(obs[m]["reward"], axis=1) for m in ORIGINAL_MIXTURES], axis=1
            )
            do_argmax = np.argmax(do["mean_reward"], axis=1)
            mixture_flip = np.any(old_argmax != old_argmax[:, :1], axis=1)
            oracle_flip = np.any(old_argmax != do_argmax[:, None], axis=1)
            add(f"{key}.{condition}.mixture_action_ranking_flip_rate",
                old_population["mixture_action_ranking_flip_rate"], np.mean(mixture_flip))
            add(f"{key}.{condition}.any_mixture_vs_do_action_ranking_flip_rate",
                old_population["any_mixture_vs_do_action_ranking_flip_rate"], np.mean(oracle_flip))
    return {"passed": all(entry["passed"] for entry in entries), "entries": entries,
            "not_comparable": []}


def _bootstrap_ratio(
    numerator: np.ndarray, denominator: np.ndarray, repetitions: int, seed: int,
) -> tuple[float, float]:
    left, right = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(left)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=repetitions)
    numerator_means = counts @ left / n
    denominator_means = counts @ right / n
    if np.any(np.isclose(denominator_means, 0.0)):
        raise PopulationEffectAuditError("bootstrap action-gap mean reached numerical zero")
    ratios = numerator_means / denominator_means
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def decision_scale_ratio_row(
    kappa: float, numerator: np.ndarray, denominator: np.ndarray,
    bootstrap_reps: int, seed: int,
) -> dict[str, Any]:
    numerator_mean, denominator_mean = float(np.mean(numerator)), float(np.mean(denominator))
    if np.isclose(denominator_mean, 0.0):
        raise PopulationEffectAuditError("mean do action gap is numerically zero")
    low, high = _bootstrap_ratio(numerator, denominator, bootstrap_reps, seed)
    return {
        "family": "decision_scale", "kappa": float(kappa), "condition": "confounded",
        "action": "all", "mixture": "primary", "metric": "drift_to_action_gap_ratio_of_means",
        "n_anchors": len(numerator), "mean": numerator_mean / denominator_mean,
        "standard_deviation": None, "median": None, "p10": None, "p25": None,
        "p75": None, "p90": None, "maximum": None,
        "ci95_low": low, "ci95_high": high, "bootstrap_unit": "anchor_id",
        "bootstrap_repetitions": bootstrap_reps, "bootstrap_seed": seed,
        "numerator_mean": numerator_mean, "denominator_mean": denominator_mean,
        "pointwise_ratios_computed": False,
    }


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "family", "kappa", "condition", "action", "mixture", "metric", "n_anchors",
        "mean", "standard_deviation", "median", "p10", "p25", "p75", "p90", "maximum",
        "ci95_low", "ci95_high", "bootstrap_unit", "bootstrap_repetitions", "bootstrap_seed",
        "numerator_mean", "denominator_mean", "pointwise_ratios_computed",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _select_row(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise PopulationEffectAuditError(f"aggregate row selection is not unique: {criteria}")
    return matches[0]


def _plot_series(
    output: Path, filename: str, ylabel: str,
    series: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    plt.figure(figsize=(7, 4.5))
    for label, rows in series:
        ordered = sorted(rows, key=lambda row: float(row["kappa"]))
        x = np.asarray([row["kappa"] for row in ordered], dtype=np.float64)
        y = np.asarray([row["mean"] for row in ordered], dtype=np.float64)
        low = np.asarray([row["ci95_low"] for row in ordered], dtype=np.float64)
        high = np.asarray([row["ci95_high"] for row in ordered], dtype=np.float64)
        line, = plt.plot(x, y, marker="o", label=label)
        plt.fill_between(x, low, high, alpha=0.2, color=line.get_color())
    plt.xlabel("kappa_env")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / filename, dpi=300)
    plt.close()


def make_review_figures(
    output: Path, aggregate_rows: list[dict[str, Any]], contexts: dict[float, dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    kappas = sorted(contexts)
    _plot_series(output, "u_reward_effect_vs_kappa.png", "Mean absolute reward U effect", [
        ("do oracle", [_select_row(
            aggregate_rows, family="u_outcome_effect", kappa=kappa, condition="do_oracle",
            action="all", mixture="none", metric="reward_u_effect_abs"
        ) for kappa in kappas])
    ])
    _plot_series(output, "primary_reward_drift_vs_kappa.png",
                 "Mean absolute primary reward heavy contrast", [
        (condition, [_select_row(
            aggregate_rows, family="primary_mixture_drift", kappa=kappa,
            condition=condition, action="all", mixture="logger1_heavy_vs_logger2_heavy",
            metric="absolute_reward_drift"
        ) for kappa in kappas]) for condition in CONDITIONS
    ])
    _plot_series(output, "primary_delta_drift_vs_kappa.png",
                 "Mean primary delta heavy contrast L2", [
        (condition, [_select_row(
            aggregate_rows, family="primary_mixture_drift", kappa=kappa,
            condition=condition, action="all", mixture="logger1_heavy_vs_logger2_heavy",
            metric="delta_drift_heavy_contrast"
        ) for kappa in kappas]) for condition in CONDITIONS
    ])
    _plot_series(output, "reward_do_error_vs_kappa.png", "Mean absolute reward do-error", [
        (f"{condition}:{mixture}", [_select_row(
            aggregate_rows, family="do_error", kappa=kappa, condition=condition,
            action="all", mixture=mixture, metric="reward_do_error_abs"
        ) for kappa in kappas])
        for condition in CONDITIONS for mixture in PRIMARY_MIXTURES
    ])
    ranking_series = []
    for condition in CONDITIONS:
        for metric in ("top_set_different", "strict_flip"):
            ranking_series.append((f"{condition}:{metric}", [_select_row(
                aggregate_rows, family="action_ranking", kappa=kappa, condition=condition,
                action="all", mixture="logger1_heavy_vs_logger2_heavy", metric=metric
            ) for kappa in kappas]))
        ranking_series.append((f"{condition}:midpoint_vs_do", [_select_row(
            aggregate_rows, family="action_ranking", kappa=kappa, condition=condition,
            action="all", mixture="logger12_midpoint", metric="top_set_differs_from_do"
        ) for kappa in kappas]))
    _plot_series(output, "action_ranking_difference_vs_kappa.png", "Anchor fraction",
                 ranking_series)
    _plot_series(output, "drift_relative_to_action_gap_vs_kappa.png", "Dimensionless ratio/fraction", [
        ("ratio of means", [_select_row(
            aggregate_rows, family="decision_scale", kappa=kappa, condition="confounded",
            action="all", mixture="primary", metric="drift_to_action_gap_ratio_of_means"
        ) for kappa in kappas]),
        ("fraction drift > gap", [_select_row(
            aggregate_rows, family="decision_scale", kappa=kappa, condition="confounded",
            action="all", mixture="primary", metric="fraction_drift_greater_than_action_gap"
        ) for kappa in kappas]),
    ])
    context = contexts[0.3]
    x = np.abs(context["do"]["reward_u_effect"]).reshape(-1)
    y = context["primary_metrics"]["confounded"]["three_mixture_reward_range"].reshape(-1)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(x, y, alpha=0.35, s=10)
    plt.xlabel("Absolute reward U effect at kappa=0.3")
    plt.ylabel("Primary three-mixture reward range")
    plt.tight_layout()
    plt.savefig(output / "drift_vs_u_effect_kappa_0p30.png", dpi=300)
    plt.close()


def _format_number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.8g}"


def _main_table(rows: list[dict[str, Any]], kappas: Iterable[float]) -> str:
    lines = [
        "| kappa | reward U effect | delta U effect | conf reward drift | ind reward drift | "
        "conf delta drift | ind delta drift | heavy ranking diff | strict flip | drift/action gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kappa in kappas:
        def value(family: str, condition: str, metric: str,
                  mixture: str = "logger1_heavy_vs_logger2_heavy") -> float:
            return float(_select_row(
                rows, family=family, kappa=kappa, condition=condition, action="all",
                mixture=mixture, metric=metric
            )["mean"])
        numbers = (
            value("u_outcome_effect", "do_oracle", "reward_u_effect_abs", "none"),
            value("u_outcome_effect", "do_oracle", "delta_u_effect_l2", "none"),
            value("primary_mixture_drift", "confounded", "absolute_reward_drift"),
            value("primary_mixture_drift", "independent_latents", "absolute_reward_drift"),
            value("primary_mixture_drift", "confounded", "delta_drift_heavy_contrast"),
            value("primary_mixture_drift", "independent_latents", "delta_drift_heavy_contrast"),
            value("action_ranking", "confounded", "top_set_different"),
            value("action_ranking", "confounded", "strict_flip"),
            value("decision_scale", "confounded", "drift_to_action_gap_ratio_of_means", "primary"),
        )
        lines.append("| " + f"{kappa:.2f}" + " | " + " | ".join(
            _format_number(number) for number in numbers
        ) + " |")
    return "\n".join(lines)


def _do_and_secondary_table(rows: list[dict[str, Any]], kappas: Iterable[float]) -> str:
    lines = [
        "| kappa | conf do-error L1 | conf do-error midpoint | conf do-error L2 | "
        "ind do-error max(primary) | secondary reward range conf | secondary reward range ind |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kappa in kappas:
        def value(family: str, condition: str, mixture: str, metric: str) -> float:
            return float(_select_row(
                rows, family=family, kappa=kappa, condition=condition, action="all",
                mixture=mixture, metric=metric,
            )["mean"])
        confounded_errors = [
            value("do_error", "confounded", mixture, "reward_do_error_abs")
            for mixture in PRIMARY_MIXTURES
        ]
        independent_errors = [
            value("do_error", "independent_latents", mixture, "reward_do_error_abs")
            for mixture in PRIMARY_MIXTURES
        ]
        numbers = (
            confounded_errors[0], confounded_errors[1], confounded_errors[2],
            max(independent_errors),
            value("secondary_logger_and_action_mixture_shift", "confounded",
                  "balanced/logger3_heavy included", "original_four_mixture_reward_range"),
            value("secondary_logger_and_action_mixture_shift", "independent_latents",
                  "balanced/logger3_heavy included", "original_four_mixture_reward_range"),
        )
        lines.append("| " + f"{kappa:.2f}" + " | " + " | ".join(
            _format_number(number) for number in numbers
        ) + " |")
    return "\n".join(lines)


def write_reports(
    output: Path, summary: dict[str, Any], aggregate_rows: list[dict[str, Any]],
    crosscheck: dict[str, Any], bootstrap_reps: int, seed: int,
) -> None:
    table = _main_table(aggregate_rows, EXPECTED_KAPPAS)
    do_secondary_table = _do_and_secondary_table(aggregate_rows, EXPECTED_KAPPAS)
    hard = summary["hard_checks"]
    hard_lines = "\n".join(f"- `{name}`: {value}" for name, value in hard.items())
    report = f"""# Phase 8A-R — Hopper Logger-Mixture Population Effect Review

## Input artifact integrity

The verified Phase 8A input contains 2048 anchors and kappa values 0.0, 0.1, 0.2, and 0.3.
All 84 upstream hard invariants were true. Input hashes were recorded before analysis and matched
after analysis. Population effects were recomputed from public NPZ rows, aligned sample weights,
hidden audit labels used only for action/U mechanism auditing, and the raw two-point do oracle.

## Comparison definitions

`PRIMARY_FIXED_STATE_ACTION_MIXTURES` are logger1_heavy, logger12_midpoint, and logger2_heavy.
Their exact `(anchor_id, commanded_action bytes)` probability masses match, with conditional action
mass `(minus, base, plus) = (0.45, 0.10, 0.45)` at every anchor.

`SECONDARY_LOGGER_AND_ACTION_MIXTURE_SHIFT` contains balanced and logger3_heavy. These change the
base-action mass and cannot support the fixed-P(S,A) primary interpretation.

## Main anchor-level summary

{table}

All displayed values are descriptive. Full mean, SD, median, P10/P25/P75/P90, maximum, and paired
anchor-bootstrap intervals are in `aggregate_tables.csv`.

## U posterior mechanism audit

Weighted empirical posteriors matched the analytic complementary-logger values: confounded plus
uses `(8/9, 1/2, 1/9)` across logger1-heavy/midpoint/logger2-heavy; confounded minus reverses this;
base and every independent-latent cell equal 1/2.

## Negative controls and mechanism identity

The kappa=0, independent-latent, and base-action controls are recorded in `hard_checks.json`.
The reward and physical-delta heavy contrasts were checked anchor by anchor against the signed
`(7/9) * U-effect` identity. The midpoint was checked against the do response in the complementary
DGP. No scientific-effect threshold was applied.

## do-error, decision scale, and ranking

Mixture/action/condition-specific do-errors and tie-aware top-action-set comparisons are in the CSV.
The drift/action-gap quantity is a ratio of aggregate means; pointwise division by near-zero action
gaps was not performed. Strict flips require disjoint top-action sets under the two heavy mixtures.

{do_secondary_table}

## Secondary logger-and-action mixture shift

The final two columns above summarize the original four-mixture range and are explicitly secondary:
balanced and logger3-heavy change the base-action mass, so these values cannot establish drift under
fixed P(S,A). Per-mixture secondary do-errors remain available in `aggregate_tables.csv`.

## Existing Phase 8A cross-check

Comparable legacy metrics passed: `{crosscheck['passed']}`. Old value, recomputed value, and absolute
difference are stored in `summary.json`. Items with different definitions are labeled
`NOT_COMPARABLE_DUE_TO_DIFFERENT_METRIC_DEFINITION` rather than coerced.

## Statistical scope and unsupported conclusions

The unit is `anchor_id` (N={summary['analyzed_anchor_count']}). Intervals use {bootstrap_reps}
paired cluster-bootstrap repetitions with seed {seed}. They describe anchor-level uncertainty only.
There is one behavior-policy training seed, so this review does not establish cross-policy-seed
generalization or population significance. Transition rows are not treated as independent samples.
The midpoint result is a mechanism-positive-control property of this complementary construction;
it does not show that general source balancing always removes confounding, nor that it is generally
ineffective. This stage does not train or evaluate a pooled neural world model.

## Hard checks

{hard_lines}

## Manual decision required

Review the continuous effect sizes, bootstrap intervals, ranking changes, and drift/action-gap scale.
The program intentionally does not emit a supported/not-supported scientific verdict.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    analysis_report = f"""# Analysis report

## Question

Does changing logger composition alter the population observational response while fixed anchors,
exact commanded-action mass, and the two-point do response remain unchanged?

## Evidence

{table}

The exact mechanism identities and all negative controls are listed in `hard_checks.json`. Detailed
effect distributions and intervals are in `aggregate_tables.csv`.

## Boundary

This is a deterministic population review over one behavior-policy seed. It supports no p-value or
cross-seed generalization claim and applies no automatic scientific-success threshold.
"""
    (output / "analysis-report.md").write_text(analysis_report, encoding="utf-8")
    appendix = f"""# Statistical appendix

- Unit of analysis and bootstrap cluster: anchor_id.
- Analyzed anchors: {summary['analyzed_anchor_count']} of 2048, selected by sorted ID prefix only.
- Descriptives: mean, sample SD, median, P10, P25, P75, P90, maximum.
- Uncertainty: 95% percentile interval from {bootstrap_reps} paired anchor bootstraps, seed {seed}.
- Transition rows are repeated support cells, not independent replicates.
- No hypothesis test, p-value, or multiple-comparison claim is made because the DGP is fully
  enumerated at fixed anchors and only one behavior-policy training seed is available.
- The drift/action-gap ratio divides bootstrap aggregate means, never individual anchor gaps.
"""
    (output / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    catalog = """# Figure catalog

- `u_reward_effect_vs_kappa.png`: Purpose—measure the physical reward effect of U. Notice the
  anchor-bootstrap trend over kappa. Implication—sets the outcome scale against which drift is read.
- `primary_reward_drift_vs_kappa.png`: Purpose—compare fixed-P(S,A) reward drift in confounded and
  independent conditions. Notice the negative-control curve. Implication—separates composition
  confounding from action/state frequency change.
- `primary_delta_drift_vs_kappa.png`: Purpose—repeat the primary contrast for physical 11D delta.
  Notice condition-specific trends. Implication—checks that the mechanism is not reward-only.
- `reward_do_error_vs_kappa.png`: Purpose—compare each primary observational target with the fixed
  do oracle. Notice midpoint and independent controls. Implication—locates observational bias.
- `action_ranking_difference_vs_kappa.png`: Purpose—measure decision changes with tie-aware action
  sets. Notice ordinary disagreement versus disjoint strict flips. Implication—connects drift to choice.
- `drift_relative_to_action_gap_vs_kappa.png`: Purpose—scale drift against true one-step action gaps.
  Notice the ratio of means and fraction exceeding the gap. Implication—quantifies decision relevance.
- `drift_vs_u_effect_kappa_0p30.png`: Purpose—inspect the anchor/action mechanism relation at kappa
  0.3. Notice how drift co-varies with absolute U effect. Implication—audits the predicted identity.

All line ribbons are 95% paired anchor-bootstrap percentile intervals. Scatter points are
anchor/action cells; they are not treated as independent experimental repetitions.
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def _git_commit(repository: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True,
        text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _validate_anchor_ids(
    anchors: dict[str, np.ndarray], expected_anchor_count: int,
) -> np.ndarray:
    if expected_anchor_count == 2048:
        return validate_all_2048_anchors(anchors)
    anchor_ids = np.asarray(anchors.get("anchor_id", ()), dtype=np.int64)
    expected = np.arange(expected_anchor_count, dtype=np.int64)
    if not np.array_equal(anchor_ids, expected):
        raise PopulationEffectAuditError(
            f"test fixture must contain contiguous anchors 0..{expected_anchor_count - 1}"
        )
    return anchor_ids


def validate_complete_input_data(
    root: Path, anchors: dict[str, np.ndarray], anchor_ids: np.ndarray,
    atol: float, rtol: float,
) -> dict[str, bool]:
    """Validate every formal row even when --max-anchors requests a smoke subset."""
    checks = {
        "all_kappa_public_anchor_sets_match": True,
        "weight_arrays_align_with_public_rows": True,
        "weight_arrays_sum_to_one": True,
        "midpoint_is_average_of_heavy_weights": True,
        "primary_mixtures_preserve_exact_state_action_mass": True,
        "exact_action_key_mapping": True,
        "do_oracle_raw_keys_complete_and_unique": True,
        "public_schema_has_no_hidden_leakage": True,
        "all_input_arrays_finite": True,
    }
    expected_set = set(np.asarray(anchor_ids, dtype=np.int64).tolist())
    for kappa in EXPECTED_KAPPAS:
        directory = root / _kappa_directory(kappa)
        raw = load_npz(directory / "do_oracle_raw.npz")
        stored = load_npz(directory / "do_oracle_summary.npz")
        if not all_arrays_finite(raw, stored):
            checks["all_input_arrays_finite"] = False
        # This recomputation proves raw key completeness and uniqueness over all anchors.
        do, _ = recompute_do_oracle(raw, stored, anchors, anchor_ids, kappa, atol, rtol)
        for condition in CONDITIONS:
            public = load_npz(directory / f"{condition}_public.npz")
            hidden = load_npz(directory / f"{condition}_hidden_audit.npz")
            validate_public_schema(public)
            if set(np.asarray(public["anchor_id"], dtype=np.int64).tolist()) != expected_set:
                checks["all_kappa_public_anchor_sets_match"] = False
            if not all_arrays_finite(public, hidden):
                checks["all_input_arrays_finite"] = False
            weights = load_condition_weights(directory, condition, hidden, atol, rtol)
            expected_midpoint = 0.5 * (
                weights["logger1_heavy"] + weights["logger2_heavy"]
            )
            expected_midpoint /= expected_midpoint.sum()
            if not np.allclose(weights["logger12_midpoint"], expected_midpoint,
                               atol=atol, rtol=rtol):
                checks["midpoint_is_average_of_heavy_weights"] = False
            verify_primary_weights_preserve_exact_groups(
                public, hidden, weights, anchor_ids, atol, rtol,
                do["commanded_action"],
            )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PopulationEffectAuditError(f"complete Phase 8A input validation failed: {failed}")
    return checks


def _validate_output_root(input_root: Path, output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    if resolved == input_root:
        raise PopulationEffectAuditError("output root cannot equal the Phase 8A input root")
    try:
        relative = resolved.relative_to(input_root)
    except ValueError as exc:
        raise PopulationEffectAuditError(
            "output root must be a nested directory of the verified Phase 8A root"
        ) from exc
    if not relative.parts or not relative.parts[0].startswith("population_effect_review"):
        raise PopulationEffectAuditError(
            "output must be nested under phase8a-root/population_effect_review"
        )
    return resolved


def _assert_metric_units(specs: list[dict[str, Any]], anchor_count: int) -> bool:
    return bool(specs) and all(
        np.asarray(spec["values"]).shape == (anchor_count,) for spec in specs
    )


def run_review(
    phase8a_root: Path,
    output_root: Path | None = None,
    bootstrap_reps: int = 2000,
    seed: int = 0,
    max_anchors: int | None = None,
    *,
    expected_anchor_count: int = 2048,
) -> dict[str, Any]:
    """Run the read-only Phase 8A-R audit and write only the nested review bundle."""
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    if max_anchors is not None and max_anchors <= 0:
        raise ValueError("max_anchors must be positive")

    root = require_verified_phase8a_root(phase8a_root)
    output = _validate_output_root(
        root, output_root or (root / "population_effect_review")
    )
    manifest = _load_json(root / "manifest.json")
    existing_summary = _load_json(root / "summary.json")
    anchors = load_npz(root / "anchors.npz")
    mixture_configuration = _load_json(root / "mixture_weights.json")
    validate_all_four_kappas(manifest, root)
    validate_all_84_phase8a_invariants(existing_summary)
    all_anchor_ids = _validate_anchor_ids(anchors, expected_anchor_count)
    if set(mixture_configuration) != set(ORIGINAL_MIXTURES):
        raise PopulationEffectAuditError("mixture_weights.json does not define Phase 8A mixtures")

    input_paths = required_input_paths(root)
    hashes_before = hash_input_files(input_paths)
    atol = float(manifest.get("numerical_tolerance", {}).get("atol", 1e-7))
    rtol = float(manifest.get("numerical_tolerance", {}).get("rtol", 1e-7))
    complete_checks = validate_complete_input_data(
        root, anchors, all_anchor_ids, atol, rtol
    )

    selected_count = min(max_anchors or len(all_anchor_ids), len(all_anchor_ids))
    selected_ids = np.sort(all_anchor_ids)[:selected_count]
    contexts: dict[float, dict[str, Any]] = {}
    metric_specs: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}
    per_kappa_checks: dict[str, dict[str, bool]] = {}
    for kappa in EXPECTED_KAPPAS:
        context, specs, arrays, checks = analyze_kappa(
            root, kappa, anchors, selected_ids, atol, rtol
        )
        contexts[kappa] = context
        metric_specs.extend(specs)
        key = _kappa_directory(kappa)
        per_kappa_checks[key] = checks
        saved_arrays.update({f"{key}__{name}": values for name, values in arrays.items()})

    hard_checks: dict[str, bool] = {
        "verified_phase8a_root_required": True,
        "all_expected_anchors_present": len(all_anchor_ids) == expected_anchor_count,
        "all_four_kappas_present": True,
        "all_84_phase8a_invariants_true": True,
        "both_phase8a_completion_markers_present": True,
        **complete_checks,
        "metrics_use_anchor_level_units": _assert_metric_units(metric_specs, selected_count),
    }
    for kappa_key, checks in per_kappa_checks.items():
        for name, value in checks.items():
            hard_checks[f"{kappa_key}:{name}"] = bool(value)

    full_analysis = selected_count == len(all_anchor_ids)
    crosscheck = crosscheck_existing_summary(
        existing_summary, contexts, atol, rtol, full_analysis
    )
    hard_checks["existing_summary_crosscheck_where_comparable"] = bool(crosscheck["passed"])
    hard_checks["all_recomputed_arrays_finite"] = all_arrays_finite(saved_arrays)
    if not all(hard_checks.values()):
        failed = [name for name, passed in hard_checks.items() if not passed]
        raise PopulationEffectAuditError(f"Phase 8A-R hard checks failed: {failed}")

    aggregate_rows = aggregate_metric_specs(metric_specs, bootstrap_reps, seed)
    for index, kappa in enumerate(EXPECTED_KAPPAS):
        context = contexts[kappa]
        aggregate_rows.append(decision_scale_ratio_row(
            kappa, context["max_action_mixture_drift"], context["do_action_gap"],
            bootstrap_reps, seed + 1000 + index,
        ))
    if not all(
        row["mean"] is not None and np.isfinite(float(row["mean"]))
        and np.isfinite(float(row["ci95_low"]))
        and np.isfinite(float(row["ci95_high"]))
        for row in aggregate_rows
    ):
        raise PopulationEffectAuditError("aggregate table contains NaN or infinity")
    hard_checks["aggregate_outputs_have_no_nan_inf"] = True

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "anchor_action_metrics.npz", **saved_arrays)
    write_aggregate_csv(output / "aggregate_tables.csv", aggregate_rows)
    make_review_figures(output, aggregate_rows, contexts)

    hashes_after = hash_input_files(input_paths)
    hard_checks["input_artifact_hashes_unchanged"] = input_hashes_unchanged(
        hashes_before, hashes_after
    )
    if not hard_checks["input_artifact_hashes_unchanged"]:
        raise PopulationEffectAuditError("Phase 8A input hashes changed during analysis")

    input_integrity = {
        "phase8a_root": str(root),
        "required_file_count": len(input_paths),
        "sha256_before": hashes_before,
        "sha256_after": hashes_after,
        "unchanged": True,
    }
    review_summary = {
        "analysis_stage": "Phase 8A-R",
        "comparison_labels": {
            "primary": "PRIMARY_FIXED_STATE_ACTION_MIXTURES",
            "secondary": "SECONDARY_LOGGER_AND_ACTION_MIXTURE_SHIFT",
        },
        "available_anchor_count": len(all_anchor_ids),
        "analyzed_anchor_count": selected_count,
        "anchor_selection": "sorted anchor_id prefix" if not full_analysis else "all anchors",
        "kappas": list(EXPECTED_KAPPAS),
        "hard_checks": hard_checks,
        "all_hard_checks_passed": all(hard_checks.values()),
        "existing_phase8a_summary_crosscheck": crosscheck,
        "aggregate_metrics": aggregate_rows,
        "bootstrap": {"unit": "anchor_id", "repetitions": bootstrap_reps, "seed": seed},
        "scientific_verdict": "MANUAL_DECISION_REQUIRED",
    }
    repository = next((path for path in (root, *root.parents) if (path / ".git").exists()), root)
    review_manifest = {
        "stage": "Phase 8A-R",
        "input_root": str(root),
        "output_root": str(output),
        "read_only_inputs": True,
        "expected_anchor_count": expected_anchor_count,
        "analyzed_anchor_count": selected_count,
        "kappas": list(EXPECTED_KAPPAS),
        "primary_mixtures": list(PRIMARY_MIXTURES),
        "secondary_mixtures": list(SECONDARY_MIXTURES),
        "bootstrap_repetitions": bootstrap_reps,
        "seed": seed,
        "numerical_tolerance": {"atol": atol, "rtol": rtol},
        "git_commit": _git_commit(repository),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
    }
    _write_json(output / "manifest.json", review_manifest)
    _write_json(output / "input_integrity.json", input_integrity)
    _write_json(output / "hard_checks.json", {
        "checks": hard_checks,
        "all_passed": all(hard_checks.values()),
        "failed": [name for name, value in hard_checks.items() if not value],
    })
    _write_json(output / "summary.json", review_summary)
    write_reports(output, review_summary, aggregate_rows, crosscheck, bootstrap_reps, seed)
    return review_summary
