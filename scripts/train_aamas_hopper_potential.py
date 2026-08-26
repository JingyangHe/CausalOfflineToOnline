"""Train the paper-faithful AAMAS26 Hopper state-potential adapter."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aamas_hopper_adapter import (
    BEHAVIOR_HIDDEN_DIM,
    EXTERNAL_COMMIT,
    METHOD_NAME,
    RELEASED_BEHAVIOR_HIDDEN_DIM,
    REWARD_MODE,
    REWARD_RULE,
    _FrozenPotential,
    _import_official_module,
    convert_to_official_tensors,
    evaluate_aamas_models,
    file_sha256,
    load_aamas_potential,
    load_hopper_aamas_data,
    save_aamas_checkpoint,
    validate_external_repo,
)


PRETRAIN_EPOCHS = 50
VALUE_EPOCHS = 200
BATCH_SIZE = 1028
TAU = 0.005
TARGET_UPDATE_INTERVAL = 3
CANDIDATE_ACTIONS = 25
ACTION_SEPARATION = 0.1
MAX_H = 998
LEARNING_RATES = {
    "behavior": 1e-4, "state_difference": 1e-5,
    "critic": 1e-4, "reward": 1e-4,
}
REQUIRED_DEPENDENCIES = ("torch", "torchrl", "tensordict", "minari")


def _git_commit() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def runtime_requirement_error(
    version: tuple[int, ...] | None = None,
    finder: Any = importlib.util.find_spec,
) -> str | None:
    version = version or tuple(sys.version_info[:3])
    if version < (3, 12):
        return "AAMAS26 official continuous code requires Python >= 3.12; use a dedicated AAMAS environment"
    missing = [name for name in REQUIRED_DEPENDENCIES if finder(name) is None]
    return f"MISSING_DEPENDENCIES: {', '.join(missing)}" if missing else None


def seed_everything(seed: int, torch: Any) -> list[str]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    warnings = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except (AttributeError, RuntimeError) as exc:
        warnings.append(f"complete CUDA determinism could not be enabled: {exc}")
    return warnings


def _build_agent(
    official: Any,
    torch: Any,
    tensors: dict[str, Any],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    reward_statistics: dict[str, Any],
    device: Any,
    gamma: float,
) -> Any:
    rewards = tensors["rewards"]
    args = SimpleNamespace(loss_function="mse")
    agent = official.CausalUpperBoundEstimator(
        state_dim=12, action_dim=3,
        state_mean=torch.as_tensor(state_mean, dtype=torch.float32, device=device),
        state_std=torch.as_tensor(state_std, dtype=torch.float32, device=device),
        max_action=1, max_location=1, max_h=MAX_H,
        max_reward=float(reward_statistics["normalized_b"]),
        min_reward=float(torch.min(rewards).item()),
        mean_reward=float(torch.mean(rewards).item()),
        std_reward=float(torch.std(rewards).item()),
        device=device, args=args, gamma=gamma, tau=TAU,
        observed_policy_lr=LEARNING_RATES["behavior"],
        state_delta_lr=LEARNING_RATES["state_difference"],
        q_critic_lr=LEARNING_RATES["critic"], actor_lr=LEARNING_RATES["behavior"],
        reward_lr=LEARNING_RATES["reward"], num_sample_neg_a=CANDIDATE_ACTIONS,
        neg_action_thres=ACTION_SEPARATION,
    )
    # Paper-faithful correction: released positional max_action creates width=1.
    agent.policy = official.GaussianNN(12, 3, hidden_dim=BEHAVIOR_HIDDEN_DIM).to(device)
    agent.policy_fin = official.GaussianNN(12, 3, hidden_dim=BEHAVIOR_HIDDEN_DIM).to(device)
    agent.policy_optimizer = torch.optim.Adam(
        agent.policy.parameters(), lr=LEARNING_RATES["behavior"], weight_decay=1e-5
    )
    return agent


def _finite_epoch(name: str, values: list[float]) -> None:
    if not values or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} produced a nonfinite or empty epoch")


def train_official_core(
    agent: Any,
    tensors: dict[str, Any],
    torch: Any,
    pretraining_epochs: int,
    value_epochs: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    order = torch.randperm(tensors["observations"].shape[0], device=tensors["observations"].device)
    data = {field: values[order] for field, values in tensors.items() if field != "dones"}
    legacy_steps = torch.zeros_like(data["rewards"], dtype=torch.int32)
    metrics: dict[str, list[float]] = {
        "behavior_nll": [], "state_difference_loss": [], "reward_loss": [],
        "critic_loss": [], "critic_mean": [], "target_mean": [], "road_value_mean": [],
    }
    for epoch in range(pretraining_epochs):
        losses = {key: [] for key in ("behavior_nll", "state_difference_loss", "reward_loss")}
        for start in range(0, data["observations"].shape[0], batch_size):
            stop = start + batch_size
            mask = ~data["truncated"][start:stop].squeeze(-1)
            if not bool(mask.any()):
                continue
            values = agent.train_reward_probability_state_delta(
                data["observations"][start:stop][mask], data["actions"][start:stop][mask],
                data["next_observations"][start:stop][mask], data["rewards"][start:stop][mask],
                data["terminated"][start:stop][mask].float(), legacy_steps[start:stop][mask],
                epoch=epoch,
            )
            for key, value in zip(losses, values):
                losses[key].append(float(value))
        averages = {key: float(np.mean(value)) for key, value in losses.items()}
        for key, value in losses.items():
            _finite_epoch(key, value)
            metrics[key].append(averages[key])
        if epoch == 0 or (0 < averages["behavior_nll"] < agent.prob_loss_min):
            agent.prob_loss_min = averages["behavior_nll"]
            agent.copy_model(agent.policy, agent.policy_fin)
        if epoch == 0 or (0.001 < averages["reward_loss"] < agent.reward_loss_min):
            agent.reward_loss_min = averages["reward_loss"]
            agent.copy_model(agent.reward_model, agent.reward_model_fin)
        if epoch == 0 or (0.005 < averages["state_difference_loss"] < agent.state_loss_min):
            agent.state_loss_min = averages["state_difference_loss"]
            agent.copy_model(agent.state_transition_model, agent.state_transition_model_fin)
        print(
            f"pretrain epoch {epoch + 1}/{pretraining_epochs}: "
            f"behavior={averages['behavior_nll']:.6g} "
            f"state={averages['state_difference_loss']:.6g} reward={averages['reward_loss']:.6g}"
        )

    for epoch in range(value_epochs):
        losses = {key: [] for key in ("critic_loss", "critic_mean", "target_mean", "road_value_mean")}
        for batch_index, start in enumerate(range(0, data["observations"].shape[0], batch_size)):
            stop = start + batch_size
            result = agent.train_critic(
                data["observations"][start:stop], data["actions"][start:stop],
                data["next_observations"][start:stop], data["rewards"][start:stop],
                data["terminated"][start:stop].float(), legacy_steps[start:stop],
                data["truncated"][start:stop], soft_update=True,
                overall_step_count=batch_index, target_function="causal", reward_type=REWARD_MODE,
                policy_delay=TARGET_UPDATE_INTERVAL, max_best_state=True, max_reward=True,
            )
            values = (result[0].item(), result[1], result[2], float(result[3]))
            for key, value in zip(losses, values):
                losses[key].append(float(value))
        for key, value in losses.items():
            _finite_epoch(key, value)
            metrics[key].append(float(np.mean(value)))
        print(
            f"value epoch {epoch + 1}/{value_epochs}: loss={metrics['critic_loss'][-1]:.6g} "
            f"value={metrics['critic_mean'][-1]:.6g} target={metrics['target_mean'][-1]:.6g}"
        )
    return {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}


def _architecture() -> dict[str, Any]:
    return {
        "behavior": {"class": "GaussianNN", "distribution": "TanhNormal", "hidden_dim": 128, "hidden_layers": 3},
        "state_difference": {"class": "RegressionNN", "hidden_dim": 256, "hidden_layers": 3},
        "reward": {"class": "RegressionNN", "hidden_dim": 256, "hidden_layers": 3},
        "critic": {"class": "Critic", "twin": True, "hidden_dim": 128, "hidden_layers": 4},
        "max_h": MAX_H,
    }


def _dependency_versions() -> dict[str, str]:
    result = {}
    for name in REQUIRED_DEPENDENCIES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_AVAILABLE"
    result["numpy"] = np.__version__
    return result


def run(arguments: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if float(arguments.gamma) != 0.99:
        raise RuntimeError("Phase 7B fixes gamma=0.99; it is not a tuning parameter")
    external = validate_external_repo(arguments.external_repo)
    requirement_error = runtime_requirement_error()
    if requirement_error:
        raise RuntimeError(requirement_error)
    torch = importlib.import_module("torch")
    warnings = seed_everything(arguments.seed, torch)
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data_dir, output_dir = Path(arguments.data_dir), Path(arguments.output_dir)
    train_path, audit_path = data_dir / "train_public.npz", data_dir / "audit_public.npz"
    phase7a_manifest_path = data_dir / "manifest.json"
    train = load_hopper_aamas_data(data_dir, "train")
    formal_train_rows = train["rewards"].size
    if not arguments.smoke and formal_train_rows != 48_000:
        raise RuntimeError(f"formal Phase 7B requires 48000 train rows, found {formal_train_rows}")
    if arguments.smoke:
        train = {field: values[:4096].copy() for field, values in train.items()}
    tensors, reward_statistics = convert_to_official_tensors(train, device, torch_module=torch)
    state_mean = torch.mean(tensors["observations"], dim=0, keepdim=True).detach().cpu().numpy()
    state_std = torch.std(tensors["observations"], dim=0, keepdim=True).detach().cpu().numpy()
    if np.any(~np.isfinite(state_std)) or np.any(state_std <= 0.0):
        raise RuntimeError("train-only state standard deviation is invalid")

    official = _import_official_module(arguments.external_repo)
    agent = _build_agent(
        official, torch, tensors, state_mean, state_std, reward_statistics, device, arguments.gamma
    )
    pretrain_epochs = 1 if arguments.smoke else PRETRAIN_EPOCHS
    value_epochs = 1 if arguments.smoke else VALUE_EPOCHS
    batch_size = min(BATCH_SIZE, train["rewards"].size)
    training_metrics = train_official_core(
        agent, tensors, torch, pretrain_epochs, value_epochs, batch_size
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "training_metrics.npz", **training_metrics)
    np.savez_compressed(
        output_dir / "normalization.npz",
        state_mean=state_mean.astype(np.float32), state_std=state_std.astype(np.float32),
        reward_mean=np.asarray(reward_statistics["reward_mean"], dtype=np.float32),
        reward_std=np.asarray(reward_statistics["reward_std"], dtype=np.float32),
        normalized_b=np.asarray(reward_statistics["normalized_b"], dtype=np.float32),
    )

    architecture = _architecture()
    checkpoint_metadata = {
        "observation_dim": 12, "action_dim": 3, "gamma": arguments.gamma,
        "reward_mode": REWARD_MODE, "seed": arguments.seed, "architecture": architecture,
        "external_repo_path": external["path"], "external_commit": external["commit"],
        "normalized_reward_min": float(torch.min(tensors["rewards"]).item()),
        "normalized_reward_max": float(reward_statistics["normalized_b"]),
    }
    save_aamas_checkpoint(output_dir, agent, checkpoint_metadata, torch)

    audit = load_hopper_aamas_data(data_dir, "audit")
    if not arguments.smoke and audit["rewards"].size != 12_000:
        raise RuntimeError(f"formal Phase 7B requires 12000 audit rows, found {audit['rewards'].size}")
    audit_tensors, _ = convert_to_official_tensors(
        audit, device, reward_statistics=reward_statistics, torch_module=torch
    )
    train_report = evaluate_aamas_models(agent, train, tensors, state_mean, state_std, torch)
    audit_report = evaluate_aamas_models(agent, audit, audit_tensors, state_mean, state_std, torch)
    manifest = {
        "phase": "7B", "method_name": METHOD_NAME,
        "method_description": "Official AAMAS26 offline causal core with a small Hopper public-data and horizon adapter",
        "external_repository_path": external["path"], "external_commit": external["commit"],
        "external_git_clean": True, "our_git_commit": _git_commit(),
        "dataset_paths": {
            "train_public": str(train_path), "audit_public": str(audit_path),
            "phase7a_manifest": str(phase7a_manifest_path),
        },
        "dataset_sha256": {
            "train_public": file_sha256(train_path), "audit_public": file_sha256(audit_path),
            "phase7a_manifest": file_sha256(phase7a_manifest_path),
        },
        "train_row_count": int(train["rewards"].size), "audit_row_count": int(audit["rewards"].size),
        "pooled_source_handling": "three public train sources pooled row-wise",
        "source_id_used_by_model": False,
        "public_observation_schema": "12D Hopper physical observation plus time_to_go",
        "gamma": arguments.gamma, "reward_mode": REWARD_MODE,
        "reward_normalization": {**reward_statistics, "statistics_source": "train_public_only"},
        "candidate_action_count": CANDIDATE_ACTIONS, "action_separation": ACTION_SEPARATION,
        "training_epochs": {"model_pretraining": pretrain_epochs, "state_value": value_epochs},
        "batch_size": batch_size, "tau": TAU, "target_update_interval_batches": TARGET_UPDATE_INTERVAL,
        "network_architecture": architecture, "learning_rates": LEARNING_RATES,
        "behavior_width_correction": {
            "released_code_behavior_hidden_dim": RELEASED_BEHAVIOR_HIDDEN_DIM,
            "paper_faithful_behavior_hidden_dim": BEHAVIOR_HIDDEN_DIM,
            "correction_applied": True,
        },
        "dependency_versions": _dependency_versions(), "python_version": platform.python_version(),
        "device": str(device), "seed": arguments.seed, "determinism_warnings": warnings,
        "no_hidden_data_access": True, "online_sac_included": False,
        "certified_upper_bound": False, "reward_model_used_by_b_norm_target": False,
        "collector_truncated_used_by_model": False,
        "legacy_step_argument": "all-zero placeholder; official method leaves it unused and time_to_go remains in observation",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    direct_phi = _FrozenPotential(agent.critic, state_mean, state_std, device, torch)
    audit_roundtrip_rows = audit["observations"][:256]
    before_reload = np.asarray(direct_phi(audit_roundtrip_rows))
    after_reload = np.asarray(load_aamas_potential(output_dir, str(device))(audit_roundtrip_rows))
    reload_error = float(np.max(np.abs(before_reload - after_reload)))
    audit_metrics = {
        "behavior_model": audit_report["behavior_model"],
        "state_difference_model": audit_report["state_difference_model"],
        "reward_model": audit_report["reward_model"],
        "potential": {"train": train_report["potential"]["pooled"], **audit_report["potential"]},
        "weakness_diagnostics": {
            "behavior_nll_by_source": audit_report["behavior_model"]["per_source"],
            "state_difference_mse_by_source": audit_report["state_difference_model"]["per_source_mse"],
            "terminated_vs_nonterminated_mse": {
                "terminated": audit_report["state_difference_model"]["terminated_mse"],
                "nonterminated": audit_report["state_difference_model"]["nonterminated_mse"],
            },
            "potential_by_source": audit_report["potential"]["per_source"],
        },
        "checkpoint_roundtrip_max_absolute_difference": reload_error,
        "checkpoint_roundtrip_rows": int(audit_roundtrip_rows.shape[0]),
    }
    (output_dir / "audit_metrics.json").write_text(
        json.dumps(audit_metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if reload_error > 1e-6:
        raise RuntimeError(f"checkpoint round-trip error {reload_error} exceeds 1e-6")
    if audit_report["behavior_model"]["pooled"]["nonfinite_count"]:
        raise RuntimeError("audit behavior log probability contains nonfinite values")
    if any(audit_report["potential"]["pooled"][key] for key in ("nan_count", "inf_count")):
        raise RuntimeError("audit potential contains NaN or Inf")
    validate_external_repo(arguments.external_repo)
    return manifest, audit_metrics


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/hopper_method_pilot/stage_seed0"))
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/aamas_hopper_pilot/stage_seed0/seed_0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.smoke and arguments.output_dir == Path("artifacts/aamas_hopper_pilot/stage_seed0/seed_0"):
        arguments.output_dir = Path("artifacts/_smoke/aamas_hopper_pilot/seed_0")
    return arguments


if __name__ == "__main__":
    parsed = parse_arguments()
    try:
        completed_manifest, completed_metrics = run(parsed)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print("method:", completed_manifest["method_name"])
    print("output:", parsed.output_dir)
    print("PHASE7B_AAMAS_HOPPER_SMOKE_PASSED" if parsed.smoke else "PHASE7B_AAMAS_HOPPER_POTENTIAL_READY")
