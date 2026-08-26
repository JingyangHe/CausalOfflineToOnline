"""Audit the three fixed Phase 6A Hopper behavior-policy checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confounded_hopper import ACTUATOR_DIRECTION, ConfoundedHopperWrapper


SOURCE_STEPS = {"source_1": 200_000, "source_2": 500_000, "source_3": 1_000_000}
PUBLIC_FIELDS = {
    "observation", "action", "reward", "next_observation", "terminated",
    "truncated", "source_id", "episode_id", "time_step",
}


def _summary(values: np.ndarray, quantiles: tuple[int, ...]) -> dict[str, float] | str:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return "NOT_AVAILABLE"
    if not np.all(np.isfinite(data)):
        raise RuntimeError("diagnostic array contains a nonfinite value")
    result = {"mean": float(np.mean(data)), "median": float(np.median(data))}
    for quantile in quantiles:
        result[f"p{quantile:02d}"] = float(np.percentile(data, quantile))
    return result


def deterministic_indices(size: int, maximum: int) -> np.ndarray:
    """Return reproducible, approximately equal-spaced unique indices."""
    if size <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(size, maximum)
    return np.linspace(0, size - 1, count, dtype=np.int64)


def compensation_diagnostics(
    minus_actions: np.ndarray,
    plus_actions: np.ndarray,
    direction: np.ndarray,
    kappa: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    minus = np.asarray(minus_actions, dtype=np.float64)
    plus = np.asarray(plus_actions, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    if minus.shape != plus.shape or minus.ndim != 2 or minus.shape[1] != vector.size:
        raise ValueError("conditioned action arrays have incompatible shapes")
    vector = vector / np.linalg.norm(vector)
    difference = plus - minus
    projection = difference @ vector
    orthogonal_norm = np.linalg.norm(difference - projection[:, None] * vector, axis=1)
    difference_norm = np.linalg.norm(difference, axis=1)
    nonzero = difference_norm > 1e-12
    cosine = (-difference[nonzero] @ vector) / difference_norm[nonzero]
    applied_plus = np.clip(plus + kappa * vector, -1.0, 1.0)
    applied_minus = np.clip(minus - kappa * vector, -1.0, 1.0)
    applied_residual = np.linalg.norm(applied_plus - applied_minus, axis=1)
    report = {
        "projection": _summary(projection, (10, 90)),
        "projection_target": float(-2.0 * kappa),
        "projection_mean_absolute_deviation_from_target": float(
            np.mean(np.abs(projection + 2.0 * kappa))
        ),
        "orthogonal_norm": _summary(orthogonal_norm, (90,)),
        "compensation_cosine": _summary(cosine, (10,)),
        "applied_action_residual": _summary(applied_residual, (90,)),
    }
    arrays = {
        "projection": projection, "orthogonal_norm": orthogonal_norm,
        "cosine": cosine, "applied_residual": applied_residual,
    }
    return report, arrays


def clipping_diagnostics(
    commands: np.ndarray,
    hidden_u: int,
    direction: np.ndarray,
    kappa: float,
) -> dict[str, Any]:
    command = np.asarray(commands, dtype=np.float64)
    preclip = command + kappa * int(hidden_u) * np.asarray(direction, dtype=np.float64)
    clipped = np.clip(preclip, -1.0, 1.0)
    changed = np.abs(preclip) > 1.0
    correction = np.linalg.norm(clipped - preclip, axis=1)
    def dimension_distributions(values: np.ndarray) -> list[dict[str, float]]:
        distributions = []
        for column in values.T:
            entry = _summary(column, (1, 10, 50, 90, 99))
            assert isinstance(entry, dict)
            entry.update(
                std=float(np.std(column)), min=float(np.min(column)), max=float(np.max(column))
            )
            distributions.append(entry)
        return distributions

    return {
        "commanded_action_by_dimension": dimension_distributions(command),
        "preclip_action_by_dimension": dimension_distributions(preclip),
        "any_clipping_rate": float(np.mean(np.any(changed, axis=1))),
        "per_dimension_clipping_rate": np.mean(changed, axis=0).tolist(),
        "clipping_correction_norm": _summary(correction, (90,)),
    }


def _standardize(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pooled = np.concatenate(list(states.values()))
    mean, scale = np.mean(pooled, axis=0), np.maximum(np.std(pooled, axis=0), 1e-8)
    return {source: (values - mean) / scale for source, values in states.items()}


def nearest_neighbor_diagnostics(
    states: dict[str, np.ndarray], actions: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute within/cross-source state and matched-action diagnostics."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {"status": "SCIPY_NOT_AVAILABLE"}, {}
    normalized = _standardize(states)
    arrays: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"status": "AVAILABLE", "within": {}, "directed_cross": {}}
    within_distance: dict[str, np.ndarray] = {}
    for source, values in normalized.items():
        if values.shape[0] < 2:
            raise ValueError("each source needs at least two states")
        distances, indices = cKDTree(values).query(values, k=2)
        within_distance[source] = distances[:, 1]
        within_action = np.linalg.norm(actions[source] - actions[source][indices[:, 1]], axis=1)
        report["within"][source] = {
            "state_distance": _summary(distances[:, 1], (90,)),
            "matched_action_distance": _summary(within_action, (90,)),
        }
        arrays[f"within_state_{source}"] = distances[:, 1]
        arrays[f"within_action_{source}"] = within_action
    for first, first_values in normalized.items():
        for second, second_values in normalized.items():
            if first == second:
                continue
            distances, indices = cKDTree(second_values).query(first_values, k=1)
            action_delta = actions[first] - actions[second][indices]
            action_distance = np.linalg.norm(action_delta, axis=1)
            ratios = np.divide(
                distances, within_distance[first],
                out=np.full_like(distances, np.nan), where=within_distance[first] > 0.0,
            )
            finite_ratios = ratios[np.isfinite(ratios)]
            key = f"{first}_to_{second}"
            report["directed_cross"][key] = {
                "matched_state_distance": _summary(distances, (90,)),
                "cross_over_within_state_distance": _summary(finite_ratios, ()),
                "matched_action_distance": _summary(action_distance, (90,)),
                "absolute_action_difference_by_dimension": [
                    _summary(np.abs(action_delta[:, column]), (90,))
                    for column in range(action_delta.shape[1])
                ],
            }
            arrays[f"cross_state_{key}"] = distances
            arrays[f"cross_action_{key}"] = action_distance
    pooled = np.concatenate(list(normalized.values()))
    labels = np.concatenate([
        np.full(values.shape[0], index, dtype=np.int8)
        for index, values in enumerate(normalized.values())
    ])
    _, indices = cKDTree(pooled).query(pooled, k=2)
    report["nearest_neighbor_has_different_source"] = float(
        np.mean(labels != labels[indices[:, 1]])
    )
    return report, arrays


def validate_public_pilot(public: dict[str, np.ndarray], per_source: int) -> None:
    if set(public) != PUBLIC_FIELDS:
        raise RuntimeError(f"public pilot fields must be exactly {sorted(PUBLIC_FIELDS)}")
    forbidden = {"hidden_u", "applied_action", "qpos", "qvel"}
    if forbidden & set(public):
        raise RuntimeError("public pilot artifact contains hidden audit data")
    for source_id in (1, 2, 3):
        if int(np.sum(public["source_id"] == source_id)) != per_source:
            raise RuntimeError("pilot transition counts are not exactly balanced")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(checkpoint_dir: Path, device: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Phase 6A manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("env_id") != "Hopper-v5" or manifest.get("kappa") != 0.2:
        raise RuntimeError("manifest environment or kappa differs from fixed Phase 6A definition")
    manifest_direction = np.asarray(
        manifest.get("actuator_direction", ()), dtype=np.float64
    )
    if manifest_direction.shape != (3,) or not np.allclose(
        manifest_direction, ACTUATOR_DIRECTION, rtol=0.0, atol=1e-15
    ):
        raise RuntimeError("manifest actuator direction differs from the existing wrapper")
    if (
        manifest.get("public_observation_dimension") != 12
        or manifest.get("behavior_observation_dimension") != 13
        or manifest.get("action_dimension") != 3
    ):
        raise RuntimeError("manifest observation or action dimensions are invalid")
    paths: dict[str, Path] = {}
    for source, expected_step in SOURCE_STEPS.items():
        mapping = manifest.get("source_mapping", {}).get(source, {})
        if mapping.get("checkpoint_step") != expected_step:
            raise RuntimeError("Source 1/2/3 checkpoint mapping has changed")
        paths[source] = checkpoint_dir / mapping.get("model_file", "")
        if not paths[source].is_file():
            raise FileNotFoundError(f"missing fixed checkpoint: {paths[source]}")
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError("stable_baselines3 is required for checkpoint audit") from exc
    return manifest, {source: SAC.load(str(path), device=device) for source, path in paths.items()}, paths


def _environment(kappa: float) -> ConfoundedHopperWrapper:
    return ConfoundedHopperWrapper(
        gym.make("Hopper-v5"), kappa=kappa, expose_confounder=True, audit_info=True
    )


def _evaluate(model: Any, kappa: float, episodes: int, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = _environment(kappa)
    arrays: dict[str, list[Any]] = {
        key: [] for key in (
            "return", "length", "terminated", "truncated", "return_per_step",
            "command_norm", "applied_norm", "clipping_proportion",
        )
    }
    explicit: dict[str, list[float]] = {}
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total, length, command_norms, applied_norms, clips = 0.0, 0, [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                command = np.asarray(info["commanded_action"], dtype=np.float64)
                applied = np.asarray(info["applied_action"], dtype=np.float64)
                preclip = command + kappa * info["hidden_u"] * ACTUATOR_DIRECTION
                command_norms.append(np.linalg.norm(command))
                applied_norms.append(np.linalg.norm(applied))
                clips.append(np.any(np.abs(preclip) > 1.0))
                for key, value in info.items():
                    lower = key.lower()
                    if (lower.startswith("reward_") or "velocity" in lower) and np.isscalar(value):
                        explicit.setdefault(key, []).append(float(value))
                total, length = total + reward, length + 1
            arrays["return"].append(total)
            arrays["length"].append(length)
            arrays["terminated"].append(terminated)
            arrays["truncated"].append(truncated)
            arrays["return_per_step"].append(total / length)
            arrays["command_norm"].append(np.mean(command_norms))
            arrays["applied_norm"].append(np.mean(applied_norms))
            arrays["clipping_proportion"].append(np.mean(clips))
    finally:
        env.close()
    result = {key: np.asarray(value) for key, value in arrays.items()}
    times = np.arange(1, env.max_episode_steps + 1)
    result["survival"] = np.mean(
        (result["length"][:, None] > times) | result["truncated"][:, None], axis=0
    )
    report = {
        "return": {**_summary(result["return"], (10, 25, 75, 90)), "std": float(np.std(result["return"]))},
        "episode_length": _summary(result["length"], (10, 25, 75, 90)),
        "early_termination_rate": float(np.mean(result["terminated"] & ~result["truncated"])),
        "return_per_episode_step": _summary(result["return_per_step"], (10, 25, 50, 75, 90)),
        "episode_mean_commanded_action_norm": _summary(result["command_norm"], (10, 50, 90)),
        "episode_mean_applied_action_norm": _summary(result["applied_norm"], (10, 50, 90)),
        "episode_clipped_step_proportion": _summary(result["clipping_proportion"], (10, 50, 90)),
        "explicit_reward_or_speed_info": (
            {key: _summary(np.asarray(values), (10, 50, 90)) for key, values in explicit.items()}
            if explicit else "NOT_AVAILABLE"
        ),
    }
    return report, result


def _balanced_counts(total: int) -> tuple[int, int, int]:
    return tuple(total // 3 + int(index < total % 3) for index in range(3))


def _collect_pilot(
    models: dict[str, Any], kappa: float, per_source: int, paired_count: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    public_lists = {field: [] for field in PUBLIC_FIELDS}
    hidden_lists = {field: [] for field in ("hidden_u", "applied_action", "clipping_indicator", "source_id")}
    paired_samples: list[dict[str, Any]] = []
    allocations = _balanced_counts(paired_count)
    for source_index, (source, model) in enumerate(models.items(), start=1):
        if hasattr(model, "set_random_seed"):
            model.set_random_seed(seed + 100_000 * source_index)
        env = _environment(kappa)
        targets = set(deterministic_indices(per_source, allocations[source_index - 1]).tolist())
        episode, time_step = 0, 0
        observation, _ = env.reset(seed=seed + 1_000_000 * source_index)
        try:
            for index in range(per_source):
                public_observation = env.get_public_observation(observation)
                action, _ = model.predict(observation, deterministic=False)
                if index in targets:
                    paired_samples.append({
                        "source_id": source_index,
                        "snapshot": env.capture_audit_state(),
                        "commanded_action": np.asarray(action, dtype=np.float32).copy(),
                    })
                next_observation, reward, terminated, truncated, info = env.step(action)
                preclip = np.asarray(action) + kappa * info["hidden_u"] * ACTUATOR_DIRECTION
                values = {
                    "observation": public_observation, "action": np.asarray(action).copy(),
                    "reward": reward, "next_observation": env.get_public_observation(next_observation),
                    "terminated": terminated, "truncated": truncated, "source_id": source_index,
                    "episode_id": episode, "time_step": time_step,
                }
                for field, value in values.items():
                    public_lists[field].append(value)
                hidden_lists["hidden_u"].append(info["hidden_u"])
                hidden_lists["applied_action"].append(info["applied_action"])
                hidden_lists["clipping_indicator"].append(np.any(np.abs(preclip) > 1.0))
                hidden_lists["source_id"].append(source_index)
                observation, time_step = next_observation, time_step + 1
                if terminated or truncated:
                    episode, time_step = episode + 1, 0
                    observation, _ = env.reset(
                        seed=seed + 1_000_000 * source_index + episode
                    )
        finally:
            env.close()
    public = {field: np.asarray(values) for field, values in public_lists.items()}
    hidden = {field: np.asarray(values) for field, values in hidden_lists.items()}
    validate_public_pilot(public, per_source)
    if len(paired_samples) != paired_count:
        raise RuntimeError("paired audit allocation is not exact")
    return public, hidden, paired_samples


def run_paired_outcome_audit(
    environment: Any, samples: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    deltas = {"next_state": [], "reward": [], "applied_action": []}
    disagreements = 0
    for sample in samples:
        command = sample["commanded_action"]
        minus = environment.audit_step_from_state(sample["snapshot"], command, -1)
        plus = environment.audit_step_from_state(sample["snapshot"], command, 1)
        minus_physical = np.asarray(minus[4]["public_observation"][:-1])
        plus_physical = np.asarray(plus[4]["public_observation"][:-1])
        deltas["next_state"].append(np.linalg.norm(plus_physical - minus_physical))
        deltas["reward"].append(abs(plus[1] - minus[1]))
        deltas["applied_action"].append(
            np.linalg.norm(plus[4]["applied_action"] - minus[4]["applied_action"])
        )
        disagreements += int(bool(minus[2]) != bool(plus[2]))
    arrays = {key: np.asarray(values, dtype=np.float64) for key, values in deltas.items()}
    report = {
        f"delta_{key}": {
            **_summary(values, (90,)), "max": float(np.max(values)),
            "proportion_greater_than_1e-12": float(np.mean(values > 1e-12)),
        }
        for key, values in arrays.items()
    }
    report["termination_disagreement_count"] = disagreements
    return report, arrays


def _u_balance(hidden: dict[str, np.ndarray]) -> dict[str, Any]:
    report = {}
    for source_id in (1, 2, 3):
        values = hidden["hidden_u"][hidden["source_id"] == source_id]
        minus, plus = int(np.sum(values == -1)), int(np.sum(values == 1))
        report[f"source_{source_id}"] = {
            "minus_count": minus, "plus_count": plus, "empirical_mean": float(np.mean(values)),
            "minus_proportion": minus / values.size, "plus_proportion": plus / values.size,
        }
    return report


def _source3_unique(
    public: dict[str, np.ndarray], hidden: dict[str, np.ndarray]
) -> dict[str, Any]:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    magnitude_and_clipping = {}
    for source_id, source in enumerate(SOURCE_STEPS, 1):
        public_mask = public["source_id"] == source_id
        hidden_mask = hidden["source_id"] == source_id
        magnitude_and_clipping[source] = {
            "commanded_action_norm": _summary(
                np.linalg.norm(public["action"][public_mask], axis=1), (90,)
            ),
            "applied_action_norm": _summary(
                np.linalg.norm(hidden["applied_action"][hidden_mask], axis=1), (90,)
            ),
            "clipping_rate": float(np.mean(hidden["clipping_indicator"][hidden_mask])),
        }
    if cKDTree is None:
        return {
            "status": "SCIPY_NOT_AVAILABLE",
            "source_action_magnitude_and_clipping": magnitude_and_clipping,
        }
    states = {source: public["observation"][public["source_id"] == index] for index, source in enumerate(SOURCE_STEPS, 1)}
    actions = {source: public["action"][public["source_id"] == index] for index, source in enumerate(SOURCE_STEPS, 1)}
    normalized = _standardize(states)
    union_states = np.concatenate((normalized["source_1"], normalized["source_2"]))
    union_actions = np.concatenate((actions["source_1"], actions["source_2"]))
    distance, index = cKDTree(union_states).query(normalized["source_3"], k=1)
    action_distance = np.linalg.norm(actions["source_3"] - union_actions[index], axis=1)
    separate_matches = {}
    for comparison in ("source_1", "source_2"):
        comparison_distance, comparison_index = cKDTree(normalized[comparison]).query(
            normalized["source_3"], k=1
        )
        separate_matches[comparison] = {
            "state_distance": _summary(comparison_distance, (90,)),
            "matched_action_distance": _summary(
                np.linalg.norm(actions["source_3"] - actions[comparison][comparison_index], axis=1),
                (90,),
            ),
        }
    terminal_mask = (public["source_id"] == 3) & public["terminated"]
    terminal_states = public["observation"][terminal_mask]
    terminal_report: dict[str, Any] | str = "NOT_AVAILABLE"
    if terminal_states.size:
        pooled = np.concatenate(list(states.values()))
        mean, scale = np.mean(pooled, axis=0), np.maximum(np.std(pooled, axis=0), 1e-8)
        terminal_distance, _ = cKDTree(union_states).query((terminal_states - mean) / scale, k=1)
        terminal_report = _summary(terminal_distance, (90,))
    return {
        "status": "AVAILABLE",
        "source_3_to_source_1_union_source_2_state_distance": _summary(distance, (90,)),
        "source_3_matched_action_distance": _summary(action_distance, (90,)),
        "source_3_separate_matches": separate_matches,
        "source_3_terminal_pre_state_distance": terminal_report,
        "source_action_magnitude_and_clipping": magnitude_and_clipping,
    }


def _conclusions(quality: dict[str, Any], directional: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    second, third = quality["source_2"], quality["source_3"]
    evidence = {
        "source_3_command_norm_higher": third["episode_mean_commanded_action_norm"]["mean"] > second["episode_mean_commanded_action_norm"]["mean"],
        "source_3_applied_norm_higher": third["episode_mean_applied_action_norm"]["mean"] > second["episode_mean_applied_action_norm"]["mean"],
        "source_3_clipping_higher": third["episode_clipped_step_proportion"]["mean"] > second["episode_clipped_step_proportion"]["mean"],
        "source_3_per_step_return_higher": third["return_per_episode_step"]["mean"] > second["return_per_episode_step"]["mean"],
        "source_3_survival_shorter": third["episode_length"]["mean"] < second["episode_length"]["mean"],
        "source_3_early_termination_higher": third["early_termination_rate"] > second["early_termination_rate"],
        "source_3_compensation_residual_higher": directional["source_3"]["applied_action_residual"]["mean"] > directional["source_2"]["applied_action_residual"]["mean"],
        "speed_comparison": "NOT_AVAILABLE",
    }
    second_info, third_info = (
        second["explicit_reward_or_speed_info"], third["explicit_reward_or_speed_info"]
    )
    if isinstance(second_info, dict) and isinstance(third_info, dict):
        velocity_fields = sorted(
            key for key in second_info.keys() & third_info.keys() if "velocity" in key.lower()
        )
        if velocity_fields:
            evidence["speed_comparison"] = {
                key: {
                    "source_2_mean": second_info[key]["mean"],
                    "source_3_mean": third_info[key]["mean"],
                    "source_3_higher": third_info[key]["mean"] > second_info[key]["mean"],
                }
                for key in velocity_fields
            }
    aggressive = evidence["source_3_command_norm_higher"] and (
        evidence["source_3_applied_norm_higher"] or evidence["source_3_clipping_higher"]
    )
    unstable = evidence["source_3_survival_shorter"] and evidence["source_3_early_termination_higher"]
    if aggressive and unstable and evidence["source_3_per_step_return_higher"]:
        conclusion = "AGGRESSIVE_INSTABILITY_SUPPORTED"
    elif not aggressive and not unstable:
        conclusion = "AGGRESSIVE_INSTABILITY_NOT_SUPPORTED"
    else:
        conclusion = "AGGRESSIVE_INSTABILITY_UNRESOLVED"
    return conclusion, evidence


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir, output_dir = Path(arguments.checkpoint_dir), Path(arguments.output_dir)
    manifest, models, paths = _load_inputs(checkpoint_dir, arguments.device)
    before_hashes = {source: _hash(path) for source, path in paths.items()}
    kappa = float(manifest["kappa"])
    quality, audit_arrays = {}, {}
    for source, model in models.items():
        quality[source], arrays = _evaluate(model, kappa, arguments.eval_episodes, arguments.seed)
        audit_arrays.update({f"evaluation_{source}_{key}": value for key, value in arrays.items()})
        print(f"evaluated {source}: {arguments.eval_episodes} episodes")

    public, hidden, paired_samples = _collect_pilot(
        models, kappa, arguments.pilot_transitions_per_source,
        arguments.paired_audit_samples, arguments.seed,
    )
    print(
        "collected audit pilot:",
        f"{arguments.pilot_transitions_per_source} transitions/source",
    )
    selected = public["observation"][deterministic_indices(public["observation"].shape[0], 4096)]
    directional, action_report = {}, {}
    for source, model in models.items():
        signs = np.ones((selected.shape[0], 1), dtype=selected.dtype)
        minus_observation = np.concatenate((selected, -signs), axis=1)
        plus_observation = np.concatenate((selected, signs), axis=1)
        minus, _ = model.predict(minus_observation, deterministic=True)
        plus, _ = model.predict(plus_observation, deterministic=True)
        directional[source], arrays = compensation_diagnostics(minus, plus, ACTUATOR_DIRECTION, kappa)
        audit_arrays.update({f"u_{source}_{key}": value for key, value in arrays.items()})
        action_report[source] = {
            "u_minus": clipping_diagnostics(minus, -1, ACTUATOR_DIRECTION, kappa),
            "u_plus": clipping_diagnostics(plus, 1, ACTUATOR_DIRECTION, kappa),
        }

    states = {source: public["observation"][public["source_id"] == index] for index, source in enumerate(models, 1)}
    actions = {source: public["action"][public["source_id"] == index] for index, source in enumerate(models, 1)}
    coverage = {}
    for name, state_values in (("physical_11d", {key: value[:, :-1] for key, value in states.items()}), ("public_12d", states)):
        coverage[name], arrays = nearest_neighbor_diagnostics(state_values, actions)
        audit_arrays.update({f"coverage_{name}_{key}": value for key, value in arrays.items()})
    paired_environment = _environment(kappa)
    try:
        paired_report, paired_arrays = run_paired_outcome_audit(paired_environment, paired_samples)
    finally:
        paired_environment.close()
    audit_arrays.update({f"paired_{key}": value for key, value in paired_arrays.items()})
    print(f"completed paired U audit: {arguments.paired_audit_samples} snapshots")
    source3_unique = _source3_unique(public, hidden)
    aggressive, evidence = _conclusions(quality, directional)
    paired_report["sample_source_counts"] = {
        source: int(sum(sample["source_id"] == index for sample in paired_samples))
        for index, source in enumerate(SOURCE_STEPS, 1)
    }
    after_hashes = {source: _hash(path) for source, path in paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("a fixed Phase 6A checkpoint was modified during audit")

    summary = {
        "phase": "6B", "seed": arguments.seed,
        "source_mapping": SOURCE_STEPS, "checkpoint_sha256": before_hashes,
        "checkpoint_files_unchanged": True,
        "policy_quality": quality, "u_directional_compensation": directional,
        "action_and_clipping": action_report, "pilot_u_balance": _u_balance(hidden),
        "state_coverage_and_action_complementarity": coverage,
        "source_3_unique_information": source3_unique,
        "paired_u_outcome_audit": paired_report,
        "source_3_degradation_conclusion": aggressive,
        "source_3_degradation_evidence": evidence,
        "source_readiness_recommendation": "TRAIN_ADDITIONAL_BEHAVIOR_SEEDS_BEFORE_FORMAL_DATA",
        "recommendation_basis": "The current seed may be used for pilot work, but formal data should not depend on one behavior-training seed.",
        "readiness_evidence": {
            "behavior_heterogeneity": {
                key: value["matched_action_distance"]["mean"]
                for key, value in coverage["public_12d"].get("directed_cross", {}).items()
            },
            "state_coverage_overlap": {
                name: report.get("nearest_neighbor_has_different_source", "NOT_AVAILABLE")
                for name, report in coverage.items()
            },
            "similar_state_action_complementarity": "reported by every directed matched-state/action comparison",
            "hidden_u_policy_usage_projection_means": {
                source: report["projection"]["mean"] for source, report in directional.items()
            },
            "hidden_u_outcome_effect_means": {
                key: value["mean"] for key, value in paired_report.items()
                if key.startswith("delta_")
            },
            "source_3_unique_contribution": {
                key: source3_unique.get(key, "NOT_AVAILABLE") for key in (
                    "source_3_to_source_1_union_source_2_state_distance",
                    "source_3_matched_action_distance",
                )
            },
            "non_monotone_policy_quality_mean_return": {
                source: report["return"]["mean"] for source, report in quality.items()
            },
            "actual_clipping_rates": {
                source: report["clipping_rate"]
                for source, report in source3_unique["source_action_magnitude_and_clipping"].items()
            },
        },
        "non_monotone_quality_note": "More training steps do not guarantee monotonically better policy quality.",
        "formal_offline_dataset_generated": False,
        "pilot_artifact_warning": "AUDIT PILOT ONLY; DO NOT USE FOR AAMAS OR JOINT TRAINING",
    }
    json.dumps(summary, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "source_readiness_audit.npz", **audit_arrays)
    np.savez_compressed(output_dir / "pilot_public_transitions.npz", **public)
    np.savez_compressed(
        output_dir / "pilot_hidden_audit.npz", **hidden,
        AUDIT_ONLY_DO_NOT_USE_FOR_TRAINING=np.asarray(True),
    )
    (output_dir / "source_readiness_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--pilot-transitions-per-source", type=int, default=10_000)
    parser.add_argument("--paired-audit-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.eval_episodes <= 0 or arguments.pilot_transitions_per_source < 2:
        parser.error("evaluation episodes must be positive and pilot count must be at least two")
    if not 0 < arguments.paired_audit_samples <= 3 * arguments.pilot_transitions_per_source:
        parser.error("paired audit samples must be positive and fit within the pilot")
    return arguments


if __name__ == "__main__":
    completed_summary = run(parse_arguments())
    print("Source 3 conclusion:", completed_summary["source_3_degradation_conclusion"])
    print("Recommendation:", completed_summary["source_readiness_recommendation"])
    print("PHASE6B_SOURCE_READINESS_AUDIT_COMPLETE")
