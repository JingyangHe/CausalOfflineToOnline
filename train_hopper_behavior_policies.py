"""Train one SAC history and save exact Confounded Hopper checkpoints."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import gymnasium as gym
import numpy as np

from confounded_hopper import ACTUATOR_DIRECTION, ConfoundedHopperWrapper


ENV_ID = "Hopper-v5"
DEFAULT_CHECKPOINTS = (200_000, 500_000, 1_000_000)
SMOKE_CHECKPOINTS = (500, 1_000, 2_000)


class ExactCheckpointCallback:
    """SB3-compatible callable that saves once at each exact single-env step."""

    def __init__(self, checkpoint_steps: tuple[int, ...], output_dir: Path) -> None:
        self.checkpoint_steps = checkpoint_steps
        self.output_dir = Path(output_dir)
        self.paths = {
            step: self.output_dir / f"source_{index}_step_{step}.zip"
            for index, step in enumerate(checkpoint_steps, start=1)
        }
        self.saved_steps: set[int] = set()

    def __call__(self, local_variables: dict[str, Any], _: dict[str, Any]) -> bool:
        model = local_variables.get("self")
        if model is None or not hasattr(model, "num_timesteps"):
            raise RuntimeError("SB3 callback locals do not contain the training model")
        step = int(model.num_timesteps)
        if step in self.paths and step not in self.saved_steps:
            model.save(str(self.paths[step]))
            self.saved_steps.add(step)
        return True


def _validate_schedule(total_steps: int, checkpoint_steps: tuple[int, ...]) -> None:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not checkpoint_steps:
        raise ValueError("at least one checkpoint step is required")
    if any(step <= 0 for step in checkpoint_steps):
        raise ValueError("checkpoint steps must be positive")
    if tuple(sorted(set(checkpoint_steps))) != checkpoint_steps:
        raise ValueError("checkpoint steps must be unique and strictly increasing")
    if checkpoint_steps[-1] != total_steps:
        raise ValueError("the final checkpoint must equal total_steps")


def _git_commit() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _make_environment(kappa: float, monitor: bool) -> tuple[gym.Env, ConfoundedHopperWrapper]:
    raw_environment = gym.make(ENV_ID)
    wrapped = ConfoundedHopperWrapper(
        raw_environment,
        kappa=kappa,
        expose_confounder=True,
        audit_info=False,
    )
    if not monitor:
        return wrapped, wrapped
    from stable_baselines3.common.monitor import Monitor

    return Monitor(wrapped), wrapped


def _evaluate_model(
    model: Any,
    kappa: float,
    episodes: int,
    seed: int,
    collect_public_states: bool,
) -> tuple[dict[str, float], np.ndarray]:
    environment, wrapper = _make_environment(kappa, monitor=False)
    returns: list[float] = []
    lengths: list[int] = []
    early_terminations: list[float] = []
    public_states: list[np.ndarray] = []
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=seed + episode)
            episode_return = 0.0
            episode_length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                if collect_public_states:
                    public_states.append(wrapper.get_public_observation(observation))
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += float(reward)
                episode_length += 1
            returns.append(episode_return)
            lengths.append(episode_length)
            early_terminations.append(float(terminated and not truncated))
    finally:
        environment.close()
    metrics = {
        "mean_original_return": float(np.mean(returns)),
        "return_standard_deviation": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "early_termination_rate": float(np.mean(early_terminations)),
    }
    states = np.asarray(public_states, dtype=np.float32)
    return metrics, states


def _predict_actions(model: Any, observations: np.ndarray) -> np.ndarray:
    actions, _ = model.predict(observations, deterministic=True)
    result = np.asarray(actions, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise RuntimeError("checkpoint produced invalid deterministic actions")
    return result


def _audit_policy_actions(
    models: dict[str, Any], public_states: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    if public_states.ndim != 2 or public_states.shape[0] == 0:
        raise RuntimeError("evaluation produced no public states for policy audit")
    minus_observations = np.concatenate(
        (public_states, -np.ones((public_states.shape[0], 1), dtype=np.float32)), axis=1
    )
    plus_observations = np.concatenate(
        (public_states, np.ones((public_states.shape[0], 1), dtype=np.float32)), axis=1
    )
    conditioned_observations = np.concatenate((minus_observations, plus_observations))
    sensitivity: dict[str, float] = {}
    conditioned_actions: dict[str, np.ndarray] = {}
    for source, model in models.items():
        minus_actions = _predict_actions(model, minus_observations)
        plus_actions = _predict_actions(model, plus_observations)
        sensitivity[source] = float(
            np.mean(np.linalg.norm(plus_actions - minus_actions, axis=1))
        )
        conditioned_actions[source] = _predict_actions(model, conditioned_observations)

    differences: dict[str, float] = {}
    for first, second in combinations(models, 2):
        differences[f"{first}_vs_{second}"] = float(
            np.mean(
                np.linalg.norm(
                    conditioned_actions[first] - conditioned_actions[second], axis=1
                )
            )
        )
    return sensitivity, differences


def train_and_evaluate(arguments: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import stable_baselines3
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError(
            "stable_baselines3 is required for Hopper behavior-policy training"
        ) from exc

    checkpoint_steps = tuple(int(step) for step in arguments.checkpoint_steps)
    _validate_schedule(int(arguments.total_steps), checkpoint_steps)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_commit = _git_commit()
    training_environment, wrapper = _make_environment(arguments.kappa, monitor=True)
    callback = ExactCheckpointCallback(checkpoint_steps, output_dir)
    try:
        model = SAC(
            "MlpPolicy",
            training_environment,
            seed=arguments.seed,
            device=arguments.device,
            verbose=1,
        )
        model.learn(total_timesteps=arguments.total_steps, callback=callback)
    finally:
        training_environment.close()
    missing = [step for step in checkpoint_steps if not callback.paths[step].is_file()]
    if missing:
        raise RuntimeError(f"missing exact checkpoints after training: {missing}")

    models: dict[str, Any] = {}
    evaluations: dict[str, dict[str, float]] = {}
    audit_states = np.empty((0, wrapper.public_observation_dimension), dtype=np.float32)
    for source_index, step in enumerate(checkpoint_steps, start=1):
        source = f"source_{source_index}"
        models[source] = SAC.load(str(callback.paths[step]), device=arguments.device)
        metrics, states = _evaluate_model(
            models[source], arguments.kappa, arguments.eval_episodes,
            arguments.seed + 10_000 * source_index,
            collect_public_states=source_index == 1,
        )
        evaluations[source] = metrics
        if source_index == 1:
            audit_states = states
    sensitivity, policy_differences = _audit_policy_actions(models, audit_states)
    evaluation = {
        "checkpoint_evaluation": evaluations,
        "mean_u_action_difference": sensitivity,
        "mean_policy_action_difference": policy_differences,
        "audit_public_state_count": int(audit_states.shape[0]),
        "heterogeneity_success_threshold_applied": False,
    }
    try:
        json.dumps(evaluation, allow_nan=False)
    except ValueError as exc:
        raise RuntimeError("evaluation contains a nonfinite number") from exc

    source_mapping = {
        f"source_{index}": {
            "checkpoint_step": step,
            "model_file": callback.paths[step].name,
        }
        for index, step in enumerate(checkpoint_steps, start=1)
    }
    manifest = {
        "env_id": ENV_ID,
        "seed": arguments.seed,
        "kappa": arguments.kappa,
        "actuator_direction": ACTUATOR_DIRECTION.tolist(),
        "total_steps": arguments.total_steps,
        "checkpoint_steps": list(checkpoint_steps),
        "source_mapping": source_mapping,
        "public_observation_dimension": wrapper.public_observation_dimension,
        "behavior_observation_dimension": int(wrapper.observation_space.shape[0]),
        "action_dimension": int(wrapper.action_space.shape[0]),
        "hidden_u_distribution": {"values": [-1, 1], "probabilities": [0.5, 0.5]},
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gym.__version__,
        "python_version": platform.python_version(),
        "git_commit_at_training_start": training_commit,
        "recommended_later_data_collection_mode": {"deterministic": False},
        "action_semantics": "stored action must be commanded_action",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as stream:
        json.dump(evaluation, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest, evaluation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kappa", type=float, default=0.2)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.total_steps = 2_000
        arguments.checkpoint_steps = SMOKE_CHECKPOINTS
        arguments.eval_episodes = 1
        arguments.device = "cpu"
    if arguments.output_dir is None:
        arguments.output_dir = Path("artifacts/hopper_behavior_policies") / f"seed_{arguments.seed}"
    if not np.isfinite(arguments.kappa) or arguments.kappa < 0.0:
        parser.error("--kappa must be finite and nonnegative")
    if arguments.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    try:
        _validate_schedule(arguments.total_steps, tuple(arguments.checkpoint_steps))
    except ValueError as exc:
        parser.error(str(exc))
    return arguments


if __name__ == "__main__":
    train_and_evaluate(parse_arguments())
