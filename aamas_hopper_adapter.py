"""Small public-data adapter for the official AAMAS26 continuous causal core."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


METHOD_NAME = "AAMAS26_PAPER_FAITHFUL_HOPPER_ADAPTER"
EXTERNAL_COMMIT = "840604ba5d593279c96ed673349a019af260b1ee"
REWARD_MODE = "b_norm"
BEHAVIOR_HIDDEN_DIM = 128
RELEASED_BEHAVIOR_HIDDEN_DIM = 1
REWARD_RULE = "(r - train_mean) / (train_sample_std + 1e-7); b_norm=max(train_normalized_reward)"
PUBLIC_FIELDS = {
    "observations", "actions", "rewards", "next_observations", "terminated",
    "truncated", "collector_truncated", "source_id", "checkpoint_step",
    "episode_id", "step_in_episode", "row_id",
}
MODEL_TENSOR_FIELDS = {
    "observations", "actions", "rewards", "next_observations",
    "terminated", "truncated", "dones",
}
FORBIDDEN_FIELDS = {
    "hidden_u", "applied_action", "preclip_action", "qpos", "qvel",
    "behavior_observation", "policy_observation", "simulator_state",
}
CHECKPOINT_METADATA_FIELDS = {
    "observation_dim", "action_dim", "gamma", "reward_mode", "seed",
    "architecture", "external_repo_path", "external_commit",
}


@dataclass(frozen=True)
class ContinuousAAMASComponents:
    """Frozen official continuous components needed for one action backup.

    The networks are instances of the released ``GaussianNN`` and
    ``RegressionNN`` classes.  Reward values stored in the model are in the
    released z-normalized training scale; the wrapper returns backups in the
    original reward scale so they can be compared with a frozen SAC critic.
    """

    behavior_model: Any
    state_difference_model: Any
    reward_model: Any
    reward_mean: float
    reward_std: float
    reward_upper: float
    gamma: float
    device: Any
    action_separation: float = 0.1
    not_action_samples: int = 25


def _joint_log_probability(distribution: Any, actions: Any) -> Any:
    """Match the released continuous implementation's joint action density."""
    value = distribution.log_prob(actions)
    return value.sum(dim=-1) if value.ndim > 1 else value


def _normal_action_samples(distribution: Any, noise: Any, torch: Any) -> Any:
    """Reparameterized TanhNormal samples with caller-owned common noise."""
    location = getattr(distribution, "loc", None)
    scale = getattr(distribution, "scale", None)
    if location is None or scale is None:
        base = getattr(distribution, "base_dist", None)
        location = getattr(base, "loc", None)
        scale = getattr(base, "scale", None)
    if location is None or scale is None:
        raise RuntimeError("official behavior distribution does not expose loc/scale")
    return torch.tanh(location.unsqueeze(1) + scale.unsqueeze(1) * noise)


def compute_official_continuous_action_backup(
    source_model: ContinuousAAMASComponents,
    states: np.ndarray,
    candidate_actions: np.ndarray,
    reference_value: Any,
    *,
    common_noise: np.ndarray | None = None,
    rng_seed: int = 0,
) -> np.ndarray:
    """Evaluate the released AAMAS continuous causal target for fixed actions.

    This is a side-effect-free extraction of the target construction in
    ``CausalUpperBoundEstimator.train_critic`` and
    ``sample_not_a_state_continuous``.  The learned AAMAS critic is replaced by
    the explicitly frozen ``reference_value`` requested by Phase 8H; behavior,
    state-difference, reward handling, negative-action sampling, clipping,
    density weighting, and max backup retain the released implementation's
    semantics.
    """
    torch = importlib.import_module("torch")
    state = np.asarray(states, dtype=np.float32)
    action = np.asarray(candidate_actions, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 12 or not np.all(np.isfinite(state)):
        raise ValueError("states must be finite with shape [N, 12]")
    if action.ndim != 3 or action.shape[0] != len(state) or action.shape[2] != 3:
        raise ValueError("candidate_actions must have shape [N, K, 3]")
    if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1e-6):
        raise ValueError("candidate actions must be finite and lie in [-1, 1]")
    if source_model.not_action_samples <= 0:
        raise ValueError("not_action_samples must be positive")

    count, candidates = action.shape[:2]
    flat_state = np.repeat(state, candidates, axis=0)
    flat_action = action.reshape(-1, 3)
    tensor_state = torch.as_tensor(flat_state, dtype=torch.float32, device=source_model.device)
    tensor_action = torch.as_tensor(flat_action, dtype=torch.float32, device=source_model.device)
    batch = len(flat_state)
    sample_count = int(source_model.not_action_samples)
    if common_noise is None:
        noise = np.random.default_rng(rng_seed).standard_normal((batch, sample_count, 3))
    else:
        noise = np.asarray(common_noise, dtype=np.float32)
        if noise.shape != (batch, sample_count, 3) or not np.all(np.isfinite(noise)):
            raise ValueError(
                f"common_noise must have shape {(batch, sample_count, 3)}")
    tensor_noise = torch.as_tensor(noise, dtype=torch.float32, device=source_model.device)

    with torch.no_grad():
        distribution = source_model.behavior_model(tensor_state)
        action_log_probability = _joint_log_probability(distribution, tensor_action)
        state_action = torch.cat((tensor_state, tensor_action), dim=1)
        predicted_next = tensor_state + source_model.state_difference_model(state_action)
        predicted_reward_z = source_model.reward_model(state_action).reshape(-1)

        sampled = _normal_action_samples(distribution, tensor_noise, torch)
        original = tensor_action.unsqueeze(1).expand(-1, sample_count, -1)
        positive = original - torch.clamp(
            original + float(source_model.action_separation), min=-1.0, max=1.0)
        negative = original - torch.clamp(
            original - float(source_model.action_separation), min=-1.0, max=1.0)
        actual = original - sampled
        stacked = torch.stack((positive, negative, actual), dim=2)
        choice = torch.abs(stacked).argmax(dim=2, keepdim=True)
        selected_delta = torch.gather(stacked, 2, choice).squeeze(2)
        clean_not_action = original - selected_delta
        expanded_state = tensor_state.unsqueeze(1).expand(-1, sample_count, -1)
        alternative_pair = torch.cat(
            (expanded_state.reshape(-1, 12), clean_not_action.reshape(-1, 3)), dim=1)
        alternative_next = (
            expanded_state.reshape(-1, 12)
            + source_model.state_difference_model(alternative_pair)
        )
        alternative_distribution = source_model.behavior_model(
            expanded_state.reshape(-1, 12))
        alternative_log_probability = _joint_log_probability(
            alternative_distribution, clean_not_action.reshape(-1, 3)
        ).reshape(batch, sample_count).mean(dim=1)

    current_value = np.asarray(
        reference_value(predicted_next.detach().cpu().numpy()), dtype=np.float64
    ).reshape(-1)
    alternative_value = np.asarray(
        reference_value(alternative_next.detach().cpu().numpy()), dtype=np.float64
    ).reshape(batch, sample_count)
    if current_value.shape != (batch,) or not np.all(np.isfinite(current_value)):
        raise RuntimeError("reference_value returned invalid current-action values")
    if not np.all(np.isfinite(alternative_value)):
        raise RuntimeError("reference_value returned invalid alternative-action values")

    reward = (
        predicted_reward_z.detach().cpu().numpy().astype(np.float64)
        * float(source_model.reward_std)
        + float(source_model.reward_mean)
    )
    value_not_taken = np.maximum(alternative_value.max(axis=1), current_value)
    taken = reward + float(source_model.gamma) * current_value
    not_taken = float(source_model.reward_upper) + float(source_model.gamma) * value_not_taken
    log_taken = np.clip(
        action_log_probability.detach().cpu().numpy().reshape(-1), -50.0, -0.01)
    log_not = np.clip(
        alternative_log_probability.detach().cpu().numpy().reshape(-1), -50.0, -0.01)
    taken_weight = np.exp(log_taken) / (np.exp(log_taken) + np.exp(log_not))
    result = taken_weight * taken + (1.0 - taken_weight) * not_taken
    if not np.all(np.isfinite(result)):
        raise RuntimeError("official continuous AAMAS backup produced NaN or Inf")
    return result.reshape(count, candidates)


def compute_source_aamas_backup(
    source_models: Any,
    states: np.ndarray,
    candidate_actions: np.ndarray,
    reference_value: Any,
    *,
    common_noise: np.ndarray | None = None,
    rng_seed: int = 0,
) -> np.ndarray:
    """Thin multi-source wrapper returning ``Q_plus_e(s, a)``.

    The returned shape is ``[source, state, candidate]``.  No pooling or
    envelope operation is performed here, which makes the M=1 equivalence to
    the official continuous target directly auditable.
    """
    models = tuple(source_models)
    if not models:
        raise ValueError("at least one source model is required")
    return np.stack([
        compute_official_continuous_action_backup(
            model, states, candidate_actions, reference_value,
            common_noise=common_noise, rng_seed=rng_seed,
        )
        for model in models
    ], axis=0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_external_repo(
    repository: Path, expected_commit: str = EXTERNAL_COMMIT
) -> dict[str, Any]:
    repository = Path(repository).resolve()
    if not repository.is_dir():
        raise RuntimeError(f"external AAMAS repository is missing: {repository}")
    head = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ("git", "-C", str(repository), "status", "--short"),
        capture_output=True, text=True, check=False,
    )
    if head.returncode or status.returncode:
        raise RuntimeError("unable to audit the external AAMAS repository")
    commit, dirty = head.stdout.strip(), status.stdout.strip()
    if commit != expected_commit:
        raise RuntimeError(f"external commit mismatch: expected {expected_commit}, found {commit}")
    if dirty:
        raise RuntimeError(f"external AAMAS repository is not clean:\n{dirty}")
    return {"path": str(repository), "commit": commit, "clean": True}


def validate_public_dataset(data: dict[str, np.ndarray]) -> None:
    hidden = FORBIDDEN_FIELDS & set(data)
    if hidden:
        raise RuntimeError(f"public dataset contains forbidden fields: {sorted(hidden)}")
    if set(data) != PUBLIC_FIELDS:
        raise RuntimeError(f"public dataset fields must be exactly {sorted(PUBLIC_FIELDS)}")
    count = data["rewards"].shape[0]
    shapes = {
        "observations": (count, 12), "actions": (count, 3),
        "rewards": (count,), "next_observations": (count, 12),
        **{field: (count,) for field in PUBLIC_FIELDS if field not in {
            "observations", "actions", "rewards", "next_observations"
        }},
    }
    for field, shape in shapes.items():
        if data[field].shape != shape:
            raise RuntimeError(f"{field} has shape {data[field].shape}, expected {shape}")
    for field in ("observations", "actions", "rewards", "next_observations"):
        if data[field].dtype != np.float32 or not np.all(np.isfinite(data[field])):
            raise RuntimeError(f"{field} must be finite float32")
    for field in ("terminated", "truncated", "collector_truncated"):
        if data[field].dtype != bool:
            raise RuntimeError(f"{field} must have bool dtype")
    if count == 0 or set(np.unique(data["source_id"])) != {1, 2, 3}:
        raise RuntimeError("public dataset must contain all three fixed sources")
    mapping = {1: 200_000, 2: 500_000, 3: 1_000_000}
    for source_id, step in mapping.items():
        mask = data["source_id"] == source_id
        if not np.all(data["checkpoint_step"][mask] == step):
            raise RuntimeError("source/checkpoint mapping is invalid")
    if np.unique(data["row_id"]).size != count or np.any(np.abs(data["actions"]) > 1.0):
        raise RuntimeError("row ids are not unique or actions leave [-1, 1]")


def load_hopper_aamas_data(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    if split not in ("train", "audit"):
        raise ValueError("split must be train or audit")
    path = Path(data_dir) / f"{split}_public.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        data = {field: stored[field].copy() for field in stored.files}
    validate_public_dataset(data)
    return data


def normalize_rewards_like_official(
    rewards: np.ndarray,
    statistics: dict[str, float] | None = None,
    torch_module: Any | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    values = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("reward normalization needs at least two finite rewards")
    if statistics is None:
        if torch_module is not None and hasattr(torch_module, "std"):
            official = torch_module.as_tensor(values, dtype=torch_module.float32, device="cpu")
            mean, std = float(torch_module.mean(official)), float(torch_module.std(official))
        else:
            mean = float(np.mean(values, dtype=np.float32))
            std = float(np.std(values, ddof=1, dtype=np.float32))
        if not np.isfinite(std) or std <= 0.0:
            raise RuntimeError("train reward sample standard deviation must be positive")
        normalized = (values - np.float32(mean)) / np.float32(std + 1e-7)
        result: dict[str, float | str] = {
            "reward_mean": mean,
            "reward_std": std,
            "normalized_b": float(np.max(normalized)),
            "calculation_rule": REWARD_RULE,
        }
    else:
        result = dict(statistics)
        normalized = (
            values - np.float32(result["reward_mean"])
        ) / np.float32(float(result["reward_std"]) + 1e-7)
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("normalized rewards are nonfinite")
    return normalized.astype(np.float32).reshape(-1, 1), result


def convert_to_official_tensors(
    data: dict[str, np.ndarray],
    device: Any,
    reward_statistics: dict[str, float] | None = None,
    torch_module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, float | str]]:
    validate_public_dataset(data)
    torch = torch_module or importlib.import_module("torch")
    rewards, statistics = normalize_rewards_like_official(
        data["rewards"], reward_statistics, torch_module=torch
    )
    arrays = {
        "observations": data["observations"], "actions": data["actions"],
        "rewards": rewards, "next_observations": data["next_observations"],
        "terminated": data["terminated"][:, None],
        "truncated": data["truncated"][:, None],
        "dones": np.logical_or(data["terminated"], data["truncated"])[:, None],
    }
    tensors = {
        field: torch.as_tensor(
            values,
            dtype=torch.float32 if field not in {"terminated", "truncated", "dones"} else torch.bool,
            device=device,
        )
        for field, values in arrays.items()
    }
    if set(tensors) != MODEL_TENSOR_FIELDS:
        raise RuntimeError("model tensor adapter leaked metadata")
    return tensors, statistics


def save_aamas_checkpoint(
    checkpoint_dir: Path, agent: Any, metadata: dict[str, Any], torch_module: Any
) -> Path:
    missing = CHECKPOINT_METADATA_FIELDS - set(metadata)
    if missing:
        raise RuntimeError(f"checkpoint metadata is incomplete: {sorted(missing)}")
    payload = {
        "behavior_model_state_dict": agent.policy_fin.state_dict(),
        "state_difference_model_state_dict": agent.state_transition_model_fin.state_dict(),
        "reward_model_state_dict": agent.reward_model_fin.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "twin_critic_state_dict": agent.critic_twin.state_dict(),
        "target_critic_state_dict": agent.critic_target.state_dict(),
        "target_twin_critic_state_dict": agent.critic_target_twin.state_dict(),
        "metadata": metadata,
    }
    path = Path(checkpoint_dir) / "aamas_checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(payload, path)
    return path


def _import_official_module(repository: Path) -> Any:
    path = Path(repository).resolve() / "fin_train_value_state_new_continuous.py"
    module_name = "aamas26_official_continuous"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official AAMAS module: {path}")
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.path.insert(0, str(path.parent))
    sys.dont_write_bytecode = True
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(path.parent))
    return module


class _FrozenPotential:
    def __init__(self, critic: Any, state_mean: np.ndarray, state_std: np.ndarray, device: Any, torch: Any | None):
        self.critic, self.device, self.torch = critic, device, torch
        self.state_mean = np.asarray(state_mean, dtype=np.float32).reshape(1, -1)
        self.state_std = np.asarray(state_std, dtype=np.float32).reshape(1, -1)

    def __call__(self, observations: np.ndarray) -> np.ndarray | float:
        values = np.asarray(observations, dtype=np.float32)
        scalar = values.ndim == 1
        if scalar:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 12 or not np.all(np.isfinite(values)):
            raise ValueError("phi expects finite observations with shape [12] or [N, 12]")
        normalized = (values - self.state_mean) / (self.state_std + 1e-7)
        if self.torch is None:
            result = np.asarray(self.critic(normalized), dtype=np.float32).reshape(-1)
        else:
            tensor = self.torch.as_tensor(normalized, dtype=self.torch.float32, device=self.device)
            with self.torch.no_grad():
                result = self.critic(tensor).reshape(-1).detach().cpu().numpy()
        if not np.all(np.isfinite(result)):
            raise RuntimeError("AAMAS potential produced a nonfinite value")
        return float(result[0]) if scalar else result


def load_aamas_potential(checkpoint_dir: Path, device: str = "cpu") -> _FrozenPotential:
    checkpoint_dir = Path(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    repository = Path(manifest["external_repository_path"])
    if not repository.is_dir():
        repository = Path(__file__).resolve().parent / "external" / "li_aamas2026"
    validate_external_repo(repository)
    torch = importlib.import_module("torch")
    official = _import_official_module(repository)
    payload = torch.load(checkpoint_dir / "aamas_checkpoint.pt", map_location=device, weights_only=False)
    metadata = payload["metadata"]
    critic = official.Critic(
        metadata["observation_dim"], metadata["action_dim"],
        metadata["architecture"]["max_h"], metadata["normalized_reward_max"],
        metadata["normalized_reward_min"], metadata["gamma"],
    ).to(device)
    critic.load_state_dict(payload["critic_state_dict"])
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    with np.load(checkpoint_dir / "normalization.npz", allow_pickle=False) as normalization:
        mean, std = normalization["state_mean"].copy(), normalization["state_std"].copy()
    return _FrozenPotential(critic, mean, std, device, torch)


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"nan_count": int(np.isnan(values).sum()), "inf_count": int(np.isinf(values).sum())}
    return {
        "mean": float(np.mean(finite)), "std": float(np.std(finite)),
        "min": float(np.min(finite)), "p01": float(np.percentile(finite, 1)),
        "p10": float(np.percentile(finite, 10)), "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)), "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)), "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def evaluate_aamas_models(
    agent: Any,
    public: dict[str, np.ndarray],
    tensors: dict[str, Any],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    torch_module: Any,
) -> dict[str, Any]:
    torch = torch_module
    with torch.no_grad():
        states, actions = tensors["observations"], tensors["actions"]
        log_prob = agent.policy_fin(states).log_prob(actions)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=-1)
        nll = (-log_prob).detach().cpu().numpy().reshape(-1)
        state_pair = torch.cat((states, actions), dim=1)
        delta_error = (
            agent.state_transition_model_fin(state_pair)
            - (tensors["next_observations"] - states)
        ).detach().cpu().numpy()
        reward_error = (
            agent.reward_model_fin(state_pair) - tensors["rewards"]
        ).detach().cpu().numpy().reshape(-1)
        mean = torch.as_tensor(state_mean, dtype=torch.float32, device=states.device)
        std = torch.as_tensor(state_std, dtype=torch.float32, device=states.device)
        potential = agent.critic((states - mean) / (std + 1e-7)).detach().cpu().numpy().reshape(-1)

    source_id = public["source_id"]
    def nll_summary(values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        return {
            "mean": float(np.mean(finite)) if finite.size else "NOT_AVAILABLE",
            "median": float(np.median(finite)) if finite.size else "NOT_AVAILABLE",
            "p90": float(np.percentile(finite, 90)) if finite.size else "NOT_AVAILABLE",
            "nonfinite_count": int(np.sum(~np.isfinite(values))),
        }

    if not np.all(np.isfinite(delta_error)) or not np.all(np.isfinite(reward_error)):
        raise RuntimeError("state-difference or reward audit error is nonfinite")
    mse = lambda values: float(np.mean(np.square(values))) if values.size else "NOT_AVAILABLE"
    def source_potential(values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        return {
            "mean": float(np.mean(finite)) if finite.size else "NOT_AVAILABLE",
            "std": float(np.std(finite)) if finite.size else "NOT_AVAILABLE",
            "median": float(np.median(finite)) if finite.size else "NOT_AVAILABLE",
            "nonfinite_count": int(values.size - finite.size),
        }
    report: dict[str, Any] = {
        "behavior_model": {
            "pooled": nll_summary(nll),
            "per_source": {f"source_{i}": nll_summary(nll[source_id == i]) for i in (1, 2, 3)},
        },
        "state_difference_model": {
            "pooled_mse": mse(delta_error),
            "pooled_mae": float(np.mean(np.abs(delta_error))),
            "per_dimension_mse": np.mean(np.square(delta_error), axis=0).tolist(),
            "per_source_mse": {f"source_{i}": mse(delta_error[source_id == i]) for i in (1, 2, 3)},
            "terminated_mse": mse(delta_error[public["terminated"]]),
            "nonterminated_mse": mse(delta_error[~public["terminated"]]),
            "truncated_mse": mse(delta_error[public["truncated"]]),
            "nontruncated_mse": mse(delta_error[~public["truncated"]]),
        },
        "reward_model": {
            "pooled_mse": mse(reward_error),
            "per_source_mse": {f"source_{i}": mse(reward_error[source_id == i]) for i in (1, 2, 3)},
            "target_space": "train_z_normalized_reward",
            "causal_target_usage": "reward model is not used by the b_norm causal target",
        },
        "potential": {
            "pooled": _distribution(potential),
            "per_source": {
                f"source_{i}": source_potential(potential[source_id == i])
                for i in (1, 2, 3)
            },
        },
    }
    return report
