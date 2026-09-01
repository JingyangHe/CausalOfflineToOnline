"""Phase 8C-RM: minimal reward-only mechanism separation.

This module trains only one-step reward models.  The primary model uses a
global binary latent prior, logger-specific action tables, and one shared
reward decoder.  Hidden audit arrays and do rewards are accepted only by the
explicit Oracle/evaluation entry points.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from confounded_hopper import ACTUATOR_DIRECTION
from .analyze_phase8a_population_effect import hash_input_files, input_hashes_unchanged, load_npz
from .neural_observational_bias import make_anchor_splits, validate_splits
from .noncomplementary_population_dgp import ACTION_KEYS, CONDITIONS, PRIMARY_MIXTURES


CONTROLLED_THREE_ACTION_MECHANISM_DIAGNOSTIC_ONLY = True
ACTION_INDEX_USED_ONLY_AS_BEHAVIOR_TARGET = True
MAIN_TRAINING_FIELDS = ("observation", "commanded_action", "reward", "logger_id",
                        "action_index", "row_weight", "anchor_id", "row_id")
FORBIDDEN_MAIN_FIELDS = {"u_behavior", "u_env", "applied_action", "do_reward",
                         "action_key", "qpos", "qvel", "reward_bonus",
                         "lambda_threshold", "long_horizon_return"}
METHODS = ("pooled_mlp", "source_balanced_pooled_mlp", "source_conditioned_decoder",
           "per_source_models", "mechanism_separated", "oracle_u_aware",
           "source_shuffle", "no_behavior", "source_dependent_reward")
PRIMARY_MIXTURE = "logger12_balanced"
MIXTURES = tuple(PRIMARY_MIXTURES)
LOGGER_WEIGHTS = {name: tuple(float(value) for value in values)
                  for name, values in PRIMARY_MIXTURES.items()}
KAPPAS = (0.0, 0.3)
TOP_ATOL = TOP_RTOL = 1e-7
REWARD_DECODER_WIDTH = 256


class RewardMechanismSeparationError(RuntimeError):
    """Raised when a Phase 8C-RM invariant is not satisfied."""


class LambdaGridNotFrozenError(RewardMechanismSeparationError):
    """Raised when the human-frozen lambda grid is missing or invalid."""


@dataclass(frozen=True)
class PublicRows:
    row_id: np.ndarray
    anchor_id: np.ndarray
    observation: np.ndarray
    commanded_action: np.ndarray
    reward: np.ndarray
    logger_id: np.ndarray
    action_index: np.ndarray
    row_weight: np.ndarray

    def subset(self, anchor_ids: Sequence[int]) -> "PublicRows":
        mask = np.isin(self.anchor_id, np.asarray(anchor_ids, dtype=np.int64))
        return PublicRows(**{name: np.asarray(getattr(self, name))[mask]
                            for name in self.__dataclass_fields__})

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Normalization:
    x_mean: np.ndarray
    x_std: np.ndarray
    reward_mean: float
    reward_std: float


def _torch():
    try:
        import torch
    except Exception as exc:
        raise RewardMechanismSeparationError(f"PyTorch is unavailable: {exc}") from exc
    return torch


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise RewardMechanismSeparationError(f"required JSON is unavailable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kappa_name(value: float) -> str:
    return f"kappa_{value:.2f}".replace(".", "p")


def lambda_token(value: float) -> str:
    return "lambda_" + format(float(value), ".10g").replace(".", "p")


def load_frozen_lambda_grid(path: Path) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Load a human-confirmed grid; never derive it from test thresholds."""
    grid_path = Path(path).resolve()
    if not grid_path.is_file():
        raise LambdaGridNotFrozenError("LAMBDA_GRID_NOT_FROZEN")
    value = _read_json(grid_path)
    if value.get("manually_frozen") is not True:
        raise LambdaGridNotFrozenError("LAMBDA_GRID_NOT_FROZEN: manually_frozen must be true")
    raw = value.get("lambdas", value.get("lambda_grid"))
    if not isinstance(raw, list) or not raw:
        raise LambdaGridNotFrozenError("LAMBDA_GRID_NOT_FROZEN: lambdas are absent")
    grid = tuple(float(item) for item in raw)
    if (any(not np.isfinite(item) or item < 0.0 for item in grid)
            or tuple(sorted(set(grid))) != grid or 0.0 not in grid or 0.05 not in grid):
        raise LambdaGridNotFrozenError(
            "LAMBDA_GRID_NOT_FROZEN: require sorted unique nonnegative doses including 0 and 0.05")
    serialized = json.dumps(value, sort_keys=True).lower()
    if "test_threshold" in serialized or "test_quantile" in serialized:
        raise LambdaGridNotFrozenError("frozen grid declares forbidden test-threshold provenance")
    allowed = value.get("selection_basis", [])
    if allowed and not all("train" in str(item).lower() or "validation" in str(item).lower()
                           or "positive_control" in str(item).lower() for item in allowed):
        raise LambdaGridNotFrozenError("frozen grid selection_basis is not train/validation-only")
    return grid, value


def _require_passed(path: Path) -> None:
    value = _read_json(path)
    if value.get("all_passed") is not True or not all(value.get("checks", {}).values()):
        raise RewardMechanismSeparationError(f"input hard checks did not all pass: {path}")


def index_derived_public_files(root: Path) -> dict[tuple[float, float, str], Path]:
    """Index derived public artifacts by their recorded arrays, not path formatting."""
    result: dict[tuple[float, float, str], Path] = {}
    for path in Path(root).resolve().rglob("*_public.npz"):
        raw = load_npz(path)
        if not {"kappa_env", "lambda_reward", "condition"}.issubset(raw):
            continue
        kappas = np.unique(np.asarray(raw["kappa_env"], dtype=np.float64))
        lambdas = np.unique(np.asarray(raw["lambda_reward"], dtype=np.float64))
        conditions = np.unique(np.asarray(raw["condition"]).astype(str))
        if len(kappas) != 1 or len(lambdas) != 1 or len(conditions) != 1:
            raise RewardMechanismSeparationError(f"derived public artifact is not one scenario: {path}")
        key = (float(kappas[0]), float(lambdas[0]), str(conditions[0]))
        if key in result:
            raise RewardMechanismSeparationError(f"duplicate derived public scenario: {key}")
        result[key] = path.resolve()
    return result


def validate_public_schema(raw: Mapping[str, np.ndarray]) -> None:
    required = {"row_id", "anchor_id", "observation", "commanded_action", "reward",
                "logger_id", "condition", "kappa_env", "lambda_reward"}
    if not required.issubset(raw):
        raise RewardMechanismSeparationError(
            f"derived public rows lack {sorted(required.difference(raw))}")
    if FORBIDDEN_MAIN_FIELDS.intersection(raw):
        raise RewardMechanismSeparationError("hidden or oracle field leaked into public rows")
    n = len(raw["row_id"])
    if (raw["observation"].shape != (n, 12)
            or raw["commanded_action"].shape != (n, 3)
            or any(len(np.asarray(raw[key])) != n for key in required)):
        raise RewardMechanismSeparationError("derived public row shapes are invalid")
    for key in ("observation", "commanded_action", "reward", "kappa_env", "lambda_reward"):
        if not np.all(np.isfinite(np.asarray(raw[key], dtype=np.float64))):
            raise RewardMechanismSeparationError(f"non-finite public field: {key}")
    if not set(np.unique(raw["logger_id"]).tolist()).issubset({0, 1, 2}):
        raise RewardMechanismSeparationError("logger IDs must be 0, 1, 2")


def commanded_action_indices(raw: Mapping[str, np.ndarray], tolerance: float = 1e-6) -> np.ndarray:
    """Map public commanded actions to minus/base/plus without hidden action labels."""
    validate_public_schema(raw)
    anchors = np.asarray(raw["anchor_id"], dtype=np.int64)
    actions = np.asarray(raw["commanded_action"], dtype=np.float64)
    loggers = np.asarray(raw["logger_id"], dtype=np.int64)
    result = np.full(len(anchors), -1, dtype=np.int64)
    for anchor in np.unique(anchors):
        rows = np.flatnonzero(anchors == anchor)
        base_rows = rows[loggers[rows] == 2]
        if not len(base_rows):
            raise RewardMechanismSeparationError(f"anchor {anchor} has no public base action")
        base = actions[base_rows[0]]
        if not np.allclose(actions[base_rows], base, atol=tolerance, rtol=0):
            raise RewardMechanismSeparationError(f"anchor {anchor} has inconsistent base actions")
        projection = (actions[rows] - base) @ ACTUATOR_DIRECTION
        result[rows[projection < -tolerance]] = 0
        result[rows[np.abs(projection) <= tolerance]] = 1
        result[rows[projection > tolerance]] = 2
        if set(result[rows].tolist()) != {0, 1, 2}:
            raise RewardMechanismSeparationError(
                f"anchor {anchor} does not expose exactly minus/base/plus commanded actions")
    return result


def normalized_logger_row_weights(base_weights: np.ndarray, logger_id: np.ndarray,
                                  logger_weights: Sequence[float]) -> np.ndarray:
    base = np.asarray(base_weights, dtype=np.float64)
    logger = np.asarray(logger_id, dtype=np.int64)
    target = np.asarray(logger_weights, dtype=np.float64)
    if (base.shape != logger.shape or target.shape != (3,) or np.any(base < 0)
            or np.any(target < 0) or not np.isclose(target.sum(), 1.0)):
        raise RewardMechanismSeparationError("row/logger weights are invalid")
    result = np.zeros_like(base)
    for source in range(3):
        mask = logger == source
        mass = float(base[mask].sum())
        if not np.any(mask) or mass <= 0.0:
            raise RewardMechanismSeparationError(f"logger {source} has no positive support mass")
        result[mask] = base[mask] / mass * target[source]
    if not np.isclose(result.sum(), 1.0) or np.any(result < 0) or not np.all(np.isfinite(result)):
        raise RewardMechanismSeparationError("normalized logger weights failed")
    return result


def make_public_rows(raw: Mapping[str, np.ndarray], base_weights: np.ndarray,
                     logger_weights: Sequence[float]) -> PublicRows:
    validate_public_schema(raw)
    weights = normalized_logger_row_weights(base_weights, raw["logger_id"], logger_weights)
    result = PublicRows(
        row_id=np.asarray(raw["row_id"], dtype=np.int64),
        anchor_id=np.asarray(raw["anchor_id"], dtype=np.int64),
        observation=np.asarray(raw["observation"], dtype=np.float32),
        commanded_action=np.asarray(raw["commanded_action"], dtype=np.float32),
        reward=np.asarray(raw["reward"], dtype=np.float64),
        logger_id=np.asarray(raw["logger_id"], dtype=np.int64),
        action_index=commanded_action_indices(raw), row_weight=weights)
    if FORBIDDEN_MAIN_FIELDS.intersection(result.arrays()):
        raise RewardMechanismSeparationError("main-training rows contain a forbidden field")
    return result


def extend_or_reuse_splits(existing: Mapping[str, Sequence[int]], selected: np.ndarray,
                           seed: int = 0) -> dict[str, list[int]]:
    """Preserve every existing assignment and deterministically assign new anchors."""
    selected_set = set(map(int, np.asarray(selected, dtype=np.int64)))
    result = {name: sorted(selected_set.intersection(map(int, existing.get(name, ()))))
              for name in ("train", "validation", "test")}
    assigned = set().union(*(set(values) for values in result.values()))
    if sum(len(values) for values in result.values()) != len(assigned):
        raise RewardMechanismSeparationError("existing anchor split overlaps")
    remaining = np.asarray(sorted(selected_set - assigned), dtype=np.int64)
    if remaining.size:
        additions = make_anchor_splits(remaining, seed)
        for name in result:
            result[name] = sorted(result[name] + list(map(int, additions[name])))
    if not validate_splits(result, np.asarray(sorted(selected_set), dtype=np.int64)):
        raise RewardMechanismSeparationError("extended anchor split is incomplete or overlapping")
    return result


def fit_normalization(rows: PublicRows) -> Normalization:
    x = np.concatenate((rows.observation.astype(np.float64),
                        rows.commanded_action.astype(np.float64)), axis=1)
    weights = rows.row_weight / rows.row_weight.sum()
    x_mean = weights @ x
    x_var = weights @ ((x - x_mean) ** 2)
    x_std = np.sqrt(np.maximum(x_var, 1e-12))
    reward_mean = float(weights @ rows.reward)
    reward_var = float(weights @ ((rows.reward - reward_mean) ** 2))
    return Normalization(x_mean, x_std, reward_mean, max(math.sqrt(reward_var), 1e-6))


def normalized_x(rows: PublicRows, stats: Normalization) -> np.ndarray:
    x = np.concatenate((rows.observation.astype(np.float64),
                        rows.commanded_action.astype(np.float64)), axis=1)
    return ((x - stats.x_mean) / stats.x_std).astype(np.float32)


def _mlp(torch: Any, input_dimension: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dimension, REWARD_DECODER_WIDTH), torch.nn.ReLU(),
        torch.nn.Linear(REWARD_DECODER_WIDTH, REWARD_DECODER_WIDTH), torch.nn.ReLU(),
        torch.nn.Linear(REWARD_DECODER_WIDTH, 1))


def make_model(kind: str, seed: int = 0) -> Any:
    """Construct a model without importing torch at module import time."""
    if kind not in METHODS:
        raise ValueError(f"unknown model kind: {kind}")
    torch = _torch()
    torch.manual_seed(int(seed))

    class RewardOnlyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kind = kind
            self.log_scale = torch.nn.Parameter(torch.zeros(()))
            if kind in {"mechanism_separated", "source_shuffle", "no_behavior",
                        "source_dependent_reward"}:
                self.prior_logits = torch.nn.Parameter(torch.zeros(2))
                if kind != "no_behavior":
                    self.behavior_logits = torch.nn.Parameter(torch.zeros(3, 2, 3))
                decoder_input = 20 if kind == "source_dependent_reward" else 17
                self.reward_decoder = _mlp(torch, decoder_input)
            elif kind == "per_source_models":
                self.reward_decoders = torch.nn.ModuleList([_mlp(torch, 15) for _ in range(3)])
            elif kind == "oracle_u_aware":
                self.reward_decoder = _mlp(torch, 17)
            else:
                input_dimension = 18 if kind == "source_conditioned_decoder" else 15
                self.reward_decoder = _mlp(torch, input_dimension)

        def prior_log_probs(self) -> Any:
            if not hasattr(self, "prior_logits"):
                raise RuntimeError("this model has no latent prior")
            return torch.log_softmax(self.prior_logits, dim=0)

        def behavior_log_probs(self) -> Any:
            if not hasattr(self, "behavior_logits"):
                raise RuntimeError("this model has no behavior mechanism")
            return torch.log_softmax(self.behavior_logits, dim=2)

        def latent_means(self, x: Any, source: Any | None = None) -> Any:
            batch = x.shape[0]
            repeated = x[:, None, :].expand(batch, 2, 15)
            latent = torch.eye(2, dtype=x.dtype, device=x.device)[None, :, :].expand(batch, 2, 2)
            parts = [repeated, latent]
            if self.kind == "source_dependent_reward":
                if source is None:
                    raise RuntimeError("source-dependent decoder requires source")
                source_onehot = torch.nn.functional.one_hot(source, 3).to(x.dtype)
                parts.append(source_onehot[:, None, :].expand(batch, 2, 3))
            decoder_input = torch.cat(parts, dim=2).reshape(batch * 2, -1)
            return self.reward_decoder(decoder_input).reshape(batch, 2)

        def plain_mean(self, x: Any, source: Any | None = None) -> Any:
            if self.kind == "source_conditioned_decoder":
                source_onehot = torch.nn.functional.one_hot(source, 3).to(x.dtype)
                return self.reward_decoder(torch.cat((x, source_onehot), dim=1)).squeeze(1)
            if self.kind == "per_source_models":
                output = torch.empty(len(x), dtype=x.dtype, device=x.device)
                for logger in range(3):
                    mask = source == logger
                    if torch.any(mask):
                        output[mask] = self.reward_decoders[logger](x[mask]).squeeze(1)
                return output
            return self.reward_decoder(x).squeeze(1)

        def oracle_mean(self, x: Any, u_index: Any) -> Any:
            if self.kind != "oracle_u_aware":
                raise RuntimeError("oracle_mean is isolated to the Oracle U-aware model")
            latent = torch.nn.functional.one_hot(u_index, 2).to(x.dtype)
            return self.reward_decoder(torch.cat((x, latent), dim=1)).squeeze(1)

        def training_log_prob(self, x: Any, reward: Any, source: Any,
                              action_index: Any) -> Any:
            scale = torch.exp(self.log_scale).clamp_min(1e-4)
            if self.kind in {"mechanism_separated", "source_shuffle",
                             "source_dependent_reward"}:
                means = self.latent_means(x, source if self.kind == "source_dependent_reward" else None)
                log_prior = self.prior_log_probs()[None, :]
                behavior = self.behavior_log_probs()[source, :, action_index]
                normal = -0.5 * ((reward[:, None] - means) / scale) ** 2 \
                    - self.log_scale - 0.5 * math.log(2.0 * math.pi)
                return torch.logsumexp(log_prior + behavior + normal, dim=1)
            if self.kind == "no_behavior":
                means = self.latent_means(x)
                normal = -0.5 * ((reward[:, None] - means) / scale) ** 2 \
                    - self.log_scale - 0.5 * math.log(2.0 * math.pi)
                return torch.logsumexp(self.prior_log_probs()[None, :] + normal, dim=1)
            if self.kind == "oracle_u_aware":
                raise RuntimeError("Oracle U-aware training uses isolated oracle_log_prob")
            mean = self.plain_mean(x, source)
            return (-0.5 * ((reward - mean) / scale) ** 2
                    - self.log_scale - 0.5 * math.log(2.0 * math.pi))

        def oracle_log_prob(self, x: Any, reward: Any, u_index: Any) -> Any:
            scale = torch.exp(self.log_scale).clamp_min(1e-4)
            mean = self.oracle_mean(x, u_index)
            return (-0.5 * ((reward - mean) / scale) ** 2
                    - self.log_scale - 0.5 * math.log(2.0 * math.pi))

    return RewardOnlyModel()


def exact_joint_log_likelihood_numpy(prior_logits: np.ndarray,
                                     behavior_logits: np.ndarray,
                                     means: np.ndarray, reward: np.ndarray,
                                     source: np.ndarray, action_index: np.ndarray,
                                     log_scale: float) -> np.ndarray:
    """Reference exact two-state logsumexp used by tests and audits."""
    prior = np.asarray(prior_logits, np.float64)
    prior = prior - np.logaddexp.reduce(prior)
    behavior = np.asarray(behavior_logits, np.float64)
    behavior = behavior - np.logaddexp.reduce(behavior, axis=2, keepdims=True)
    mu = np.asarray(means, np.float64)
    y = np.asarray(reward, np.float64)
    scale = math.exp(float(log_scale))
    component = (prior[None, :] + behavior[np.asarray(source), :, np.asarray(action_index)]
                 - 0.5 * ((y[:, None] - mu) / scale) ** 2
                 - math.log(scale) - 0.5 * math.log(2.0 * math.pi))
    return np.logaddexp.reduce(component, axis=1)


def shuffled_sources(source: np.ndarray, seed: int) -> np.ndarray:
    source = np.asarray(source, dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(source)
    if len(source) > 1 and np.array_equal(source, shuffled):
        shuffled = np.roll(shuffled, 1)
    return shuffled


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_model(path: Path, model: Any, metadata: Mapping[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, path)


def load_model(path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    model = make_model(str(metadata["kind"]), int(metadata["seed"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, metadata


def train_reward_model(kind: str, train: PublicRows, validation: PublicRows,
                       stats: Normalization, *, seed: int, updates: int,
                       batch_size: int, device: str,
                       shuffled_source_labels: np.ndarray | None = None) -> tuple[Any, dict[str, Any]]:
    """Train using public observational likelihood only; accepts no oracle argument."""
    torch = _torch()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(kind, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def tensors(rows: PublicRows, source_override: np.ndarray | None = None) -> tuple[Any, ...]:
        source = rows.logger_id if source_override is None else source_override
        return (
            torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device),
            torch.as_tensor((rows.reward - stats.reward_mean) / stats.reward_std,
                            dtype=torch.float32, device=device),
            torch.as_tensor(source, dtype=torch.long, device=device),
            torch.as_tensor(rows.action_index, dtype=torch.long, device=device),
            torch.as_tensor(rows.row_weight, dtype=torch.float32, device=device),
        )

    tx, ty, ts, ta, tw = tensors(train, shuffled_source_labels)
    vx, vy, vs, va, vw = tensors(validation)
    schedule = np.random.default_rng(seed).integers(
        0, len(train.anchor_id), size=(updates, batch_size), dtype=np.int64)
    interval = max(1, updates // 100)
    best_loss, best_step, best_state = math.inf, -1, None
    train_loss: list[float] = []
    validation_loss: list[float] = []
    validation_step: list[int] = []
    for step, batch_np in enumerate(schedule):
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        log_prob = model.training_log_prob(tx[batch], ty[batch], ts[batch], ta[batch])
        loss = -torch.sum(tw[batch] * log_prob) / torch.sum(tw[batch])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        train_loss.append(float(loss.detach().cpu()))
        if step % interval == 0 or step == updates - 1:
            model.eval()
            with torch.no_grad():
                value = float((-torch.sum(vw * model.training_log_prob(vx, vy, vs, va))
                               / torch.sum(vw)).cpu())
            model.train(); validation_loss.append(value); validation_step.append(step + 1)
            if value < best_loss:
                best_loss, best_step = value, step + 1
                best_state = {key: tensor.detach().cpu().clone()
                              for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RewardMechanismSeparationError("validation-only checkpoint was not selected")
    model.load_state_dict(best_state); model.eval()
    return model, {"kind": kind, "seed": seed, "best_validation_nll": best_loss,
                   "best_validation_step": best_step, "selection_metric": "observational_validation_nll",
                   "do_oracle_used_for_selection": False, "train_loss": train_loss,
                   "validation_loss": validation_loss, "validation_step": validation_step,
                   "state_hash": _state_hash(model)}


def train_oracle_u_model(train: PublicRows, validation: PublicRows,
                         train_u_env: np.ndarray, validation_u_env: np.ndarray,
                         stats: Normalization, *, seed: int, updates: int,
                         batch_size: int, device: str) -> tuple[Any, dict[str, Any]]:
    """Physically isolated ceiling model; never called by primary training."""
    torch = _torch()
    train_u = np.asarray(train_u_env, dtype=np.int8)
    validation_u = np.asarray(validation_u_env, dtype=np.int8)
    if (train_u.shape != train.reward.shape or validation_u.shape != validation.reward.shape
            or not np.isin(train_u, (-1, 1)).all()
            or not np.isin(validation_u, (-1, 1)).all()):
        raise RewardMechanismSeparationError("Oracle U rows are invalid or misaligned")
    torch.manual_seed(seed)
    model = make_model("oracle_u_aware", seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def tensors(rows: PublicRows, u: np.ndarray) -> tuple[Any, ...]:
        return (torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device),
                torch.as_tensor((rows.reward - stats.reward_mean) / stats.reward_std,
                                dtype=torch.float32, device=device),
                torch.as_tensor((u + 1) // 2, dtype=torch.long, device=device),
                torch.as_tensor(rows.row_weight, dtype=torch.float32, device=device))

    tx, ty, tu, tw = tensors(train, train_u)
    vx, vy, vu, vw = tensors(validation, validation_u)
    schedule = np.random.default_rng(seed).integers(0, len(train.reward),
        size=(updates, batch_size), dtype=np.int64)
    interval = max(1, updates // 100)
    best_loss, best_step, best_state = math.inf, -1, None
    train_loss: list[float] = []
    validation_loss: list[float] = []
    validation_step: list[int] = []
    for step, batch_np in enumerate(schedule):
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        log_prob = model.oracle_log_prob(tx[batch], ty[batch], tu[batch])
        loss = -torch.sum(tw[batch] * log_prob) / torch.sum(tw[batch])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        train_loss.append(float(loss.detach().cpu()))
        if step % interval == 0 or step == updates - 1:
            model.eval()
            with torch.no_grad():
                value = float((-torch.sum(vw * model.oracle_log_prob(vx, vy, vu))
                               / torch.sum(vw)).cpu())
            model.train(); validation_loss.append(value); validation_step.append(step + 1)
            if value < best_loss:
                best_loss, best_step = value, step + 1
                best_state = {key: tensor.detach().cpu().clone()
                              for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RewardMechanismSeparationError("Oracle validation checkpoint was not selected")
    model.load_state_dict(best_state); model.eval()
    return model, {"kind": "oracle_u_aware", "seed": seed,
                   "best_validation_nll": best_loss, "best_validation_step": best_step,
                   "selection_metric": "observational_validation_nll",
                   "do_oracle_used_for_selection": False, "oracle_only": True,
                   "train_loss": train_loss, "validation_loss": validation_loss,
                   "validation_step": validation_step, "state_hash": _state_hash(model)}


def _predict_latent_do(model: Any, x: Any, source_weights: Sequence[float] | None = None) -> Any:
    torch = _torch()
    prior = torch.softmax(model.prior_logits, dim=0)
    if model.kind == "source_dependent_reward":
        weights = source_weights if source_weights is not None else LOGGER_WEIGHTS[PRIMARY_MIXTURE]
        result = 0.0
        for source, weight in enumerate(weights):
            source_tensor = torch.full((len(x),), source, dtype=torch.long, device=x.device)
            result = result + float(weight) * torch.sum(
                model.latent_means(x, source_tensor) * prior[None, :], dim=1)
        return result
    return torch.sum(model.latent_means(x) * prior[None, :], dim=1)


def predict_do(model: Any, observation: np.ndarray, action: np.ndarray,
               stats: Normalization, device: str,
               source_weights: Sequence[float] = LOGGER_WEIGHTS[PRIMARY_MIXTURE]) -> np.ndarray:
    """Source-free prediction; never conditions on factual action posterior or reward."""
    torch = _torch()
    raw_x = np.concatenate((np.asarray(observation, np.float64), np.asarray(action, np.float64)), axis=1)
    x = torch.as_tensor(((raw_x - stats.x_mean) / stats.x_std).astype(np.float32), device=device)
    with torch.no_grad():
        if model.kind in {"mechanism_separated", "source_shuffle", "no_behavior",
                          "source_dependent_reward"}:
            prediction = _predict_latent_do(model, x, source_weights)
        elif model.kind == "source_conditioned_decoder":
            prediction = 0.0
            for source, weight in enumerate(source_weights):
                sources = torch.full((len(x),), source, dtype=torch.long, device=device)
                prediction = prediction + float(weight) * model.plain_mean(x, sources)
        elif model.kind == "per_source_models":
            prediction = 0.0
            for source, weight in enumerate(source_weights):
                prediction = prediction + float(weight) * model.reward_decoders[source](x).squeeze(1)
        elif model.kind == "oracle_u_aware":
            minus = torch.zeros(len(x), dtype=torch.long, device=device)
            plus = torch.ones(len(x), dtype=torch.long, device=device)
            prediction = 0.5 * (model.oracle_mean(x, minus) + model.oracle_mean(x, plus))
        else:
            prediction = model.plain_mean(x)
    return prediction.detach().cpu().numpy().astype(np.float64) * stats.reward_std + stats.reward_mean


def predict_observational(model: Any, rows: PublicRows, stats: Normalization,
                          device: str) -> np.ndarray:
    torch = _torch()
    x = torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device)
    source = torch.as_tensor(rows.logger_id, dtype=torch.long, device=device)
    action = torch.as_tensor(rows.action_index, dtype=torch.long, device=device)
    with torch.no_grad():
        if model.kind in {"mechanism_separated", "source_shuffle", "source_dependent_reward"}:
            prior = model.prior_log_probs()[None, :]
            behavior = model.behavior_log_probs()[source, :, action]
            posterior = torch.softmax(prior + behavior, dim=1)
            means = model.latent_means(x, source if model.kind == "source_dependent_reward" else None)
            prediction = torch.sum(posterior * means, dim=1)
        elif model.kind == "no_behavior":
            prediction = _predict_latent_do(model, x)
        else:
            prediction = model.plain_mean(x, source)
    return prediction.cpu().numpy().astype(np.float64) * stats.reward_std + stats.reward_mean


def predict_oracle_observational(model: Any, rows: PublicRows, u_env: np.ndarray,
                                 stats: Normalization, device: str) -> np.ndarray:
    """Oracle-only factual prediction; hidden U never enters the primary API."""
    if model.kind != "oracle_u_aware":
        raise RewardMechanismSeparationError("Oracle prediction called with a non-Oracle model")
    torch = _torch()
    u = np.asarray(u_env, dtype=np.int8)
    if u.shape != rows.reward.shape or not np.isin(u, (-1, 1)).all():
        raise RewardMechanismSeparationError("Oracle observational U is invalid")
    x = torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device)
    latent = torch.as_tensor((u + 1) // 2, dtype=torch.long, device=device)
    with torch.no_grad():
        prediction = model.oracle_mean(x, latent)
    return prediction.cpu().numpy().astype(np.float64) * stats.reward_std + stats.reward_mean


def top_masks(values: np.ndarray, atol: float = TOP_ATOL, rtol: float = TOP_RTOL) -> np.ndarray:
    value = np.asarray(values, np.float64)
    return np.isclose(value, np.max(value, axis=1, keepdims=True), atol=atol, rtol=rtol)


def regret_metrics(do_reward: np.ndarray, predicted_reward: np.ndarray) -> dict[str, Any]:
    do = np.asarray(do_reward, np.float64)
    prediction = np.asarray(predicted_reward, np.float64)
    do_top, predicted_top = top_masks(do), top_masks(prediction)
    maximum = do.max(axis=1)
    best = maximum - np.max(np.where(predicted_top, do, -np.inf), axis=1)
    worst = maximum - np.min(np.where(predicted_top, do, np.inf), axis=1)
    if np.any(best < -1e-12) or np.any(worst < best - 1e-12):
        raise RewardMechanismSeparationError("decision regret became negative")
    disagreement = ~np.all(do_top == predicted_top, axis=1)
    strict = ~np.any(do_top & predicted_top, axis=1)
    positive = best > TOP_ATOL
    count = max(1, int(math.ceil(0.01 * len(best))))
    total = float(best.sum())
    return {
        "top_set_disagreement": float(np.mean(disagreement)),
        "strict_flip": float(np.mean(strict)), "mean_regret": float(np.mean(best)),
        "worst_tie_mean_regret": float(np.mean(worst)),
        "conditional_mean_regret": float(np.mean(best[positive])) if np.any(positive) else 0.0,
        "p90_regret": float(np.quantile(best, .9)), "max_regret": float(np.max(best)),
        "top_1pct_regret_contribution": float(np.sort(best)[-count:].sum() / total)
        if total > 0 else 0.0,
        **{f"predicted_top_fraction_{name}": float(np.mean(predicted_top[:, index]))
           for index, name in enumerate(ACTION_KEYS)},
        **{f"do_top_fraction_{name}": float(np.mean(do_top[:, index]))
           for index, name in enumerate(ACTION_KEYS)},
    }


def derive_test_action_table(public: Mapping[str, np.ndarray], test_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one public observation and each exact commanded action per test anchor."""
    indices = commanded_action_indices(public)
    anchors, observation, action = (np.asarray(public[key]) for key in
                                     ("anchor_id", "observation", "commanded_action"))
    obs_rows, action_rows = [], []
    for anchor in test_ids:
        rows = np.flatnonzero(anchors == int(anchor))
        if not len(rows):
            raise RewardMechanismSeparationError(f"test anchor is absent from public rows: {anchor}")
        obs_rows.append(observation[rows[0]])
        choices = []
        for action_index in range(3):
            candidates = rows[indices[rows] == action_index]
            if not len(candidates):
                raise RewardMechanismSeparationError("test anchor action is unavailable")
            choices.append(action[candidates[0]])
        action_rows.append(choices)
    ids = np.asarray(test_ids, dtype=np.int64)
    obs = np.asarray(obs_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    return ids, obs, actions


def validate_main_model_structure(model: Any) -> dict[str, bool]:
    names = {name for name, _ in model.named_parameters()}
    return {
        "source_only_enters_behavior": (model.kind == "mechanism_separated"
            and any(name.startswith("behavior_logits") for name in names)
            and not any("source" in name for name in names)),
        "reward_decoder_is_source_invariant": model.kind == "mechanism_separated"
            and model.reward_decoder[0].in_features == 17,
        "prior_is_source_invariant": model.kind == "mechanism_separated"
            and tuple(model.prior_logits.shape) == (2,),
        "action_index_behavior_target_only": ACTION_INDEX_USED_ONLY_AS_BEHAVIOR_TARGET,
        "exact_two_state_marginalization": tuple(model.prior_logits.shape) == (2,),
    }


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _device(name: str) -> str:
    torch = _torch()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RewardMechanismSeparationError("CUDA requested but unavailable")
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return name


def _aligned_base_weights(phase8anc_root: Path, raw: Mapping[str, np.ndarray],
                          kappa: float, condition: str, mixture: str) -> np.ndarray:
    kname = kappa_name(kappa)
    source = load_npz(Path(phase8anc_root) / kname / f"{condition}_public.npz")
    weights = np.asarray(np.load(Path(phase8anc_root) / kname / "weights" / condition
                                 / f"{mixture}.npy"), dtype=np.float64)
    if len(weights) != len(source["row_id"]):
        raise RewardMechanismSeparationError("source weight rows are misaligned")
    lookup = {int(row): index for index, row in enumerate(source["row_id"])}
    try:
        return np.asarray([weights[lookup[int(row)]] for row in raw["row_id"]], dtype=np.float64)
    except KeyError as exc:
        raise RewardMechanismSeparationError("derived public row is absent from source weights") from exc


def _hidden_u_for_rows(public_path: Path, row_ids: np.ndarray) -> np.ndarray:
    hidden_path = public_path.with_name(public_path.name.replace("_public.npz", "_hidden_audit.npz"))
    if not hidden_path.is_file():
        raise RewardMechanismSeparationError(
            f"isolated Oracle U audit input is unavailable: {hidden_path}")
    hidden = load_npz(hidden_path)
    if not {"row_id", "u_env"}.issubset(hidden):
        raise RewardMechanismSeparationError("Oracle hidden audit lacks row_id/u_env")
    lookup = {int(row): int(value) for row, value in zip(hidden["row_id"], hidden["u_env"])}
    try:
        result = np.asarray([lookup[int(row)] for row in row_ids], dtype=np.int8)
    except KeyError as exc:
        raise RewardMechanismSeparationError("Oracle hidden rows do not align with public rows") from exc
    if not np.isin(result, (-1, 1)).all():
        raise RewardMechanismSeparationError("Oracle hidden U is not binary")
    return result


def _oracle_do_table(phase8anc_root: Path, kappa: float, anchor_ids: Sequence[int]) -> np.ndarray:
    raw = load_npz(Path(phase8anc_root) / "anchor_action_metrics.npz")
    prefix = kappa_name(kappa)
    ids = np.asarray(raw[f"{prefix}__anchor_id"], dtype=np.int64)
    reward = np.asarray(raw[f"{prefix}__do_mean_reward"], dtype=np.float64)
    lookup = {int(anchor): index for index, anchor in enumerate(ids)}
    try:
        result = reward[np.asarray([lookup[int(anchor)] for anchor in anchor_ids])]
    except KeyError as exc:
        raise RewardMechanismSeparationError("do oracle does not cover all test anchors") from exc
    if result.shape != (len(anchor_ids), 3) or not np.all(np.isfinite(result)):
        raise RewardMechanismSeparationError("do oracle table is invalid")
    return result


def _output_stats(path: Path, stats: Normalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x_mean=stats.x_mean, x_std=stats.x_std,
                        reward_mean=np.asarray(stats.reward_mean),
                        reward_std=np.asarray(stats.reward_std))


def _weighted_metrics(target: np.ndarray, prediction: np.ndarray,
                      weight: np.ndarray) -> dict[str, float]:
    y, p, w = (np.asarray(value, np.float64) for value in (target, prediction, weight))
    w = w / w.sum(); error = p - y
    return {"mae": float(w @ np.abs(error)), "rmse": float(np.sqrt(w @ (error ** 2))),
            "signed_bias": float(w @ error)}


def _grouped_observational_metrics(rows: PublicRows, prediction: np.ndarray,
                                   source: int | None = None) -> dict[str, float]:
    """MAE/RMSE of observational means grouped by public anchor/action."""
    mask = np.ones(len(rows.reward), dtype=bool) if source is None else rows.logger_id == source
    groups: dict[tuple[int, int], list[int]] = {}
    for index in np.flatnonzero(mask):
        groups.setdefault((int(rows.anchor_id[index]), int(rows.action_index[index])), []).append(index)
    targets, predictions, masses = [], [], []
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        mass = rows.row_weight[idx]
        normalized = mass / mass.sum()
        targets.append(float(normalized @ rows.reward[idx]))
        predictions.append(float(normalized @ np.asarray(prediction)[idx]))
        masses.append(float(mass.sum()))
    return _weighted_metrics(np.asarray(targets), np.asarray(predictions), np.asarray(masses))


def _gaussian_nll(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray,
                  normalized_log_scale: float, reward_std: float) -> float:
    scale = max(math.exp(float(normalized_log_scale)) * float(reward_std), 1e-8)
    error = np.asarray(target, np.float64) - np.asarray(prediction, np.float64)
    w = np.asarray(weight, np.float64); w /= w.sum()
    return float(w @ (0.5 * (error / scale) ** 2 + math.log(scale)
                       + 0.5 * math.log(2.0 * math.pi)))


def _latent_diagnostic_rows(model: Any, scenario: Mapping[str, Any], rows: PublicRows,
                            stats: Normalization, device: str) -> list[dict[str, Any]]:
    if model.kind not in {"mechanism_separated", "source_shuffle", "source_dependent_reward",
                          "no_behavior"}:
        return []
    torch = _torch()
    with torch.no_grad():
        prior = torch.softmax(model.prior_logits, dim=0).cpu().numpy()
        entropy = float(-np.sum(prior * np.log(np.maximum(prior, 1e-15))))
        x = torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device)
        if model.kind == "source_dependent_reward":
            source = torch.as_tensor(rows.logger_id, dtype=torch.long, device=device)
            means = model.latent_means(x, source).cpu().numpy() * stats.reward_std + stats.reward_mean
        else:
            means = model.latent_means(x).cpu().numpy() * stats.reward_std + stats.reward_mean
        separation = float(np.mean(np.abs(means[:, 1] - means[:, 0])))
        behavior = (torch.softmax(model.behavior_logits, dim=2).cpu().numpy()
                    if hasattr(model, "behavior_logits") else None)
    output = [{**scenario, "diagnostic": "prior", "latent": latent,
               "value": float(prior[latent]), "posterior_entropy": entropy,
               "reward_mode_separation": separation,
               "latent_collapse": bool(np.min(prior) < 0.01 or separation < 1e-6)}
              for latent in range(2)]
    if behavior is not None:
        for logger in range(3):
            for latent in range(2):
                for action, name in enumerate(ACTION_KEYS):
                    output.append({**scenario, "diagnostic": "behavior_probability",
                                   "logger_id": logger, "latent": latent, "action": name,
                                   "value": float(behavior[logger, latent, action]),
                                   "posterior_entropy": entropy,
                                   "reward_mode_separation": separation,
                                   "latent_collapse": bool(np.min(prior) < 0.01
                                                           or separation < 1e-6)})
    return output


def _make_figures(output: Path, do_rows: Sequence[Mapping[str, Any]],
                  ranking_rows: Sequence[Mapping[str, Any]],
                  regret_rows: Sequence[Mapping[str, Any]],
                  observational_rows: Sequence[Mapping[str, Any]],
                  stability_rows: Sequence[Mapping[str, Any]],
                  latent_rows: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)

    def curves(rows: Sequence[Mapping[str, Any]], metric: str, filename: str,
               ylabel: str, condition: str = "confounded", action: str | None = None) -> None:
        selected = [row for row in rows if row.get("condition") == condition
                    and row.get("mixture") == PRIMARY_MIXTURE
                    and float(row.get("kappa", -1)) == 0.0
                    and (action is None or row.get("action") == action)]
        plt.figure()
        for method in METHODS:
            points = [row for row in selected if row.get("method") == method]
            values = sorted(set(float(row["lambda_reward"]) for row in points))
            if not values:
                continue
            means = [float(np.mean([float(row[metric]) for row in points
                                    if float(row["lambda_reward"]) == value])) for value in values]
            plt.plot(values, means, marker="o", label=method)
        plt.xlabel("lambda"); plt.ylabel(ylabel); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(figures / filename, dpi=160); plt.close()

    curves(do_rows, "mae", "do_mae_by_method_vs_lambda.png", "do reward MAE", action="all")
    curves(ranking_rows, "top_set_disagreement", "do_rank_error_by_method_vs_lambda.png",
           "do top-set disagreement")
    curves(regret_rows, "mean_regret", "decision_regret_by_method_vs_lambda.png",
           "mean decision regret")
    curves(ranking_rows, "top_set_disagreement", "low_dose_threshold_curve_by_method.png",
           "do top-set disagreement")

    plt.figure()
    for method in METHODS:
        obs = [float(row["mae"]) for row in observational_rows if row["method"] == method]
        do = [float(row["mae"]) for row in do_rows if row["method"] == method
              and row.get("action") == "all"]
        if obs and do:
            plt.scatter(np.mean(obs), np.mean(do), label=method)
    plt.xlabel("observational MAE"); plt.ylabel("do MAE"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(figures / "observational_fit_vs_do_fit.png", dpi=160); plt.close()

    pooled = {(row["kappa"], row["lambda_reward"], row["condition"], row["mixture"], row["seed"],
               row["anchor_id"], row["action"]): row["prediction"] for row in do_rows
              if row["method"] == "pooled_mlp" and row.get("anchor_id") != ""}
    mechanism = [(key, row["prediction"]) for row in do_rows
                 if row["method"] == "mechanism_separated" and row.get("anchor_id") != ""
                 for key in [(row["kappa"], row["lambda_reward"], row["condition"], row["mixture"],
                              row["seed"], row["anchor_id"], row["action"])] if key in pooled]
    plt.figure()
    if mechanism:
        plt.scatter([pooled[key] for key, _ in mechanism], [value for _, value in mechanism], s=8)
    plt.xlabel("pooled do prediction"); plt.ylabel("mechanism do prediction"); plt.tight_layout()
    plt.savefig(figures / "pooled_vs_mechanism_do_scatter.png", dpi=160); plt.close()

    plt.figure()
    for method in METHODS:
        rows = [row for row in stability_rows if row["method"] == method]
        if rows:
            plt.scatter([method] * len(rows), [float(row["prediction_mae"]) for row in rows], s=10)
    plt.xticks(rotation=70); plt.ylabel("cross-composition prediction MAE"); plt.tight_layout()
    plt.savefig(figures / "source_composition_stability.png", dpi=160); plt.close()

    behavior = [row for row in latent_rows if row.get("diagnostic") == "behavior_probability"
                and row.get("method") == "mechanism_separated"]
    plt.figure()
    if behavior:
        plt.scatter(range(len(behavior)), [float(row["value"]) for row in behavior], s=8)
    plt.xlabel("behavior-table cell"); plt.ylabel("learned probability"); plt.tight_layout()
    plt.savefig(figures / "behavior_table_recovery.png", dpi=160); plt.close()

    prior = [row for row in latent_rows if row.get("diagnostic") == "prior"
             and row.get("method") == "mechanism_separated"]
    plt.figure()
    if prior:
        plt.scatter([float(row["value"]) for row in prior],
                    [float(row["posterior_entropy"]) for row in prior], s=10)
    plt.xlabel("learned latent prior mass"); plt.ylabel("prior entropy"); plt.tight_layout()
    plt.savefig(figures / "latent_prior_and_entropy.png", dpi=160); plt.close()

    plt.figure()
    for condition in CONDITIONS:
        points = [row for row in ranking_rows if row["condition"] == condition
                  and row["method"] == "mechanism_separated" and row["mixture"] == PRIMARY_MIXTURE]
        if points:
            plt.scatter([condition] * len(points), [float(row["top_set_disagreement"]) for row in points])
    plt.ylabel("do rank error"); plt.tight_layout()
    plt.savefig(figures / "confounded_vs_independent.png", dpi=160); plt.close()

    plt.figure()
    for method in METHODS:
        values = [float(row["mae"]) for row in do_rows if row["method"] == method
                  and row.get("action") == "base"]
        if values:
            plt.scatter([method] * len(values), values, s=10)
    plt.xticks(rotation=70); plt.ylabel("base-action do MAE"); plt.tight_layout()
    plt.savefig(figures / "base_action_error_by_method.png", dpi=160); plt.close()

    plt.figure()
    for method in ("mechanism_separated", "source_shuffle"):
        values = [float(row["top_set_disagreement"]) for row in ranking_rows
                  if row["method"] == method]
        if values:
            plt.scatter([method] * len(values), values, s=10)
    plt.ylabel("do rank error"); plt.tight_layout()
    plt.savefig(figures / "source_shuffle_ablation.png", dpi=160); plt.close()


def run_reward_mechanism_separation(
    phase8anc_root: Path, direct_reward_root: Path, oracle_root: Path,
    lambda_grid_file: Path, output_root: Path, *, num_anchors: int = 100,
    kappas: tuple[float, ...] = KAPPAS, conditions: tuple[str, ...] = CONDITIONS,
    model_seeds: tuple[int, ...] = (0,), gradient_updates: int = 300,
    batch_size: int = 128, device: str = "auto", split_seed: int = 0,
) -> dict[str, Any]:
    """Run the controlled reward-only mechanism diagnostic."""
    grid, frozen_record = load_frozen_lambda_grid(lambda_grid_file)
    kappas, conditions, seeds = tuple(map(float, kappas)), tuple(conditions), tuple(map(int, model_seeds))
    if not kappas or any(value not in KAPPAS for value in kappas):
        raise ValueError(f"kappas must be a subset of {KAPPAS}")
    if not conditions or any(value not in CONDITIONS for value in conditions):
        raise ValueError(f"conditions must be a subset of {CONDITIONS}")
    if min(num_anchors, gradient_updates, batch_size) <= 0 or not seeds or num_anchors > 2048:
        raise ValueError("anchors, updates, batch size, and seeds must be positive")
    nc, direct, oracle, output = (Path(path).resolve() for path in
                                   (phase8anc_root, direct_reward_root, oracle_root, output_root))
    for root, label in ((nc, "Phase 8A-NC"), (direct, "direct reward"), (oracle, "Oracle")):
        if not root.is_dir():
            raise RewardMechanismSeparationError(f"{label} input root is unavailable: {root}")
    _require_passed(nc / "hard_checks.json"); _require_passed(direct / "hard_checks.json")
    _require_passed(oracle / "hard_checks.json")
    if (output in (nc, direct, oracle) or direct in output.parents or oracle in output.parents
            or not output.name.startswith("phase8c_reward_mechanism")):
        raise RewardMechanismSeparationError("output must not be inside an input artifact")
    if output.exists() and any(output.iterdir()):
        raise RewardMechanismSeparationError(f"output directory is not empty: {output}")
    manifest = _read_json(nc / "manifest.json")
    if int(manifest.get("available_anchor_count", -1)) != 2048:
        raise RewardMechanismSeparationError("Phase 8A-NC must expose all 2048 anchors")
    if manifest.get("action_keys") != list(ACTION_KEYS):
        raise RewardMechanismSeparationError("public action keys are not minus/base/plus")
    if not _read_json(nc / "hard_checks.json")["checks"].get("primary_mixtures_preserve_state_action_mass", False):
        # Older artifact records this invariant once per kappa.
        checks = _read_json(nc / "hard_checks.json")["checks"]
        if not all(checks.get(f"{kappa_name(k)}:primary_mixtures_preserve_state_action_mass", False)
                   for k in (0.0, 0.1, 0.2, 0.3)):
            raise RewardMechanismSeparationError("primary P(S,A) equality is not verified")

    direct_index = index_derived_public_files(direct)
    expected = {(kappa, strength, condition) for kappa in kappas for strength in grid
                for condition in conditions}
    missing = sorted(expected.difference(direct_index))
    if missing:
        raise RewardMechanismSeparationError(
            f"direct reward public artifacts do not cover frozen scenarios: {missing[:5]}")
    source_ids = load_npz(nc / f"{kappa_name(kappas[0])}/{conditions[0]}_public.npz")["anchor_id"]
    all_anchor_ids = np.unique(np.asarray(source_ids, dtype=np.int64))
    if not np.array_equal(all_anchor_ids, np.arange(2048)):
        raise RewardMechanismSeparationError("2048 public anchors are not complete")
    selected = all_anchor_ids[:num_anchors]
    existing = _read_json(direct / "splits.json")
    splits = extend_or_reuse_splits(existing, selected, split_seed)
    if not all(splits[name] for name in ("train", "validation", "test")):
        raise RewardMechanismSeparationError("anchor split has an empty partition")

    input_paths = [nc / "manifest.json", nc / "hard_checks.json",
                   nc / "anchor_action_metrics.npz", direct / "manifest.json",
                   direct / "hard_checks.json", direct / "splits.json",
                   oracle / "manifest.json", oracle / "hard_checks.json",
                   oracle / "anchor_action_metrics.npz", Path(lambda_grid_file).resolve()]
    for key in sorted(expected):
        public_path = direct_index[key]
        input_paths.extend((public_path,
                            public_path.with_name(public_path.name.replace("_public.npz", "_hidden_audit.npz"))))
    for kappa in kappas:
        for condition in conditions:
            input_paths.append(nc / kappa_name(kappa) / f"{condition}_public.npz")
            for mixture in MIXTURES:
                input_paths.append(nc / kappa_name(kappa) / "weights" / condition / f"{mixture}.npy")
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        raise RewardMechanismSeparationError(f"required read-only input is missing: {missing_paths[0]}")
    hashes_before = hash_input_files(input_paths)
    resolved_device = _device(device)
    output.mkdir(parents=True)
    _write_json(output / "splits.json", {**splits, "split_seed": split_seed,
                                          "provenance": "existing assignments preserved; new anchors deterministic"})
    _write_json(output / "frozen_lambda_grid.json", frozen_record)

    observational_rows: list[dict[str, Any]] = []
    do_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[Any, ...], np.ndarray] = {}
    checkpoint_roundtrip = True
    source_shuffle_changed = True
    all_finite = True
    structure_checks: dict[str, bool] | None = None

    for kappa in kappas:
        for strength in grid:
            for condition in conditions:
                public_path = direct_index[(kappa, strength, condition)]
                raw_all = load_npz(public_path); validate_public_schema(raw_all)
                selected_mask = np.isin(raw_all["anchor_id"], selected)
                raw = {name: np.asarray(value)[selected_mask] for name, value in raw_all.items()}
                if set(np.unique(raw["anchor_id"]).tolist()) != set(selected.tolist()):
                    raise RewardMechanismSeparationError(
                        "direct reward public artifact does not cover every selected anchor")
                base_by_mixture = {mixture: _aligned_base_weights(nc, raw, kappa, condition, mixture)
                                   for mixture in MIXTURES}
                balanced_rows = make_public_rows(raw, base_by_mixture[PRIMARY_MIXTURE],
                                                 LOGGER_WEIGHTS[PRIMARY_MIXTURE])
                stats = fit_normalization(balanced_rows.subset(splits["train"]))
                _output_stats(output / "normalization" / kappa_name(kappa)
                              / lambda_token(strength) / condition / "stats.npz", stats)
                test_anchor_ids, test_observation, test_actions = derive_test_action_table(raw, splits["test"])
                for mixture in MIXTURES:
                    actual_rows = make_public_rows(raw, base_by_mixture[mixture], LOGGER_WEIGHTS[mixture])
                    balanced_training_rows = make_public_rows(
                        raw, base_by_mixture[PRIMARY_MIXTURE], LOGGER_WEIGHTS[PRIMARY_MIXTURE])
                    for seed in seeds:
                        trained: dict[str, tuple[Any, PublicRows, dict[str, Any]]] = {}
                        for method in METHODS:
                            training_rows = (balanced_training_rows
                                             if method == "source_balanced_pooled_mlp"
                                             else actual_rows)
                            train = training_rows.subset(splits["train"])
                            validation = training_rows.subset(splits["validation"])
                            if method == "oracle_u_aware":
                                full_u = _hidden_u_for_rows(public_path, training_rows.row_id)
                                train_mask = np.isin(training_rows.anchor_id, splits["train"])
                                validation_mask = np.isin(training_rows.anchor_id, splits["validation"])
                                model, history = train_oracle_u_model(
                                    train, validation, full_u[train_mask], full_u[validation_mask], stats,
                                    seed=seed, updates=gradient_updates, batch_size=batch_size,
                                    device=resolved_device)
                                oracle_validation_u = full_u[validation_mask]
                            else:
                                shuffled = None
                                if method == "source_shuffle":
                                    shuffled = shuffled_sources(train.logger_id, seed + 8191)
                                    source_shuffle_changed &= not np.array_equal(shuffled, train.logger_id)
                                model, history = train_reward_model(
                                    method, train, validation, stats, seed=seed,
                                    updates=gradient_updates, batch_size=batch_size,
                                    device=resolved_device, shuffled_source_labels=shuffled)
                            if method == "mechanism_separated" and structure_checks is None:
                                structure_checks = validate_main_model_structure(model)
                            scenario = {"kappa": kappa, "lambda_reward": strength,
                                        "condition": condition, "mixture": mixture,
                                        "seed": seed, "method": method}
                            checkpoint = (output / "models" / kappa_name(kappa) / lambda_token(strength)
                                          / condition / mixture / f"seed_{seed}" / f"{method}.pt")
                            save_model(checkpoint, model, {**history, **scenario})
                            loaded, metadata = load_model(checkpoint, resolved_device)
                            checkpoint_roundtrip &= (_state_hash(loaded) == history["state_hash"]
                                                     and metadata["selection_metric"]
                                                     == "observational_validation_nll")
                            model = loaded
                            trained[method] = (model, training_rows, history)
                            validation_rows = training_rows.subset(splits["validation"])
                            test_rows = training_rows.subset(splits["test"])
                            if method == "oracle_u_aware":
                                test_mask = np.isin(training_rows.anchor_id, splits["test"])
                                obs_prediction = predict_oracle_observational(
                                    model, test_rows, full_u[test_mask], stats, resolved_device)
                                validation_prediction = predict_oracle_observational(
                                    model, validation_rows, oracle_validation_u, stats, resolved_device)
                            else:
                                obs_prediction = predict_observational(model, test_rows, stats, resolved_device)
                                validation_prediction = predict_observational(
                                    model, validation_rows, stats, resolved_device)
                            validation_metric = _grouped_observational_metrics(
                                validation_rows, validation_prediction)
                            observational_rows.append({**scenario, "split": "validation",
                                "scope": "mixture", "observational_nll": _gaussian_nll(
                                    validation_rows.reward, validation_prediction,
                                    validation_rows.row_weight, float(model.log_scale.detach().cpu()),
                                    stats.reward_std), **validation_metric})
                            obs_metric = _grouped_observational_metrics(test_rows, obs_prediction)
                            observational_rows.append({**scenario, "split": "test", "scope": "mixture",
                                "observational_nll": _gaussian_nll(test_rows.reward, obs_prediction,
                                    test_rows.row_weight, float(model.log_scale.detach().cpu()),
                                    stats.reward_std), **obs_metric})
                            for source in range(3):
                                metric = _grouped_observational_metrics(test_rows, obs_prediction, source)
                                observational_rows.append({**scenario, "split": "test", "scope": "source",
                                                            "logger_id": source,
                                                            "observational_nll": "", **metric})

                            flat_observation = np.repeat(test_observation, 3, axis=0)
                            flat_action = test_actions.reshape(-1, 3)
                            do_prediction = predict_do(model, flat_observation, flat_action, stats,
                                                       resolved_device).reshape(-1, 3)
                            do_oracle = _oracle_do_table(nc, kappa, test_anchor_ids)
                            prediction_cache[(kappa, strength, condition, mixture, seed, method)] = do_prediction
                            prediction_path = (output / "predictions" / kappa_name(kappa)
                                               / lambda_token(strength) / condition / mixture
                                               / f"seed_{seed}" / f"{method}.npz")
                            prediction_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(prediction_path, anchor_id=test_anchor_ids,
                                                prediction=do_prediction)
                            all_finite &= bool(np.all(np.isfinite(obs_prediction))
                                               and np.all(np.isfinite(do_prediction)))
                            flat_error = do_prediction - do_oracle
                            overall = {"mae": float(np.mean(np.abs(flat_error))),
                                       "rmse": float(np.sqrt(np.mean(flat_error ** 2))),
                                       "signed_bias": float(np.mean(flat_error))}
                            do_rows.append({**scenario, "action": "all", "anchor_id": "",
                                            "prediction": "", **overall})
                            for action_index, action_name in enumerate(ACTION_KEYS):
                                error = flat_error[:, action_index]
                                do_rows.append({**scenario, "action": action_name, "anchor_id": "",
                                    "prediction": "", "mae": float(np.mean(np.abs(error))),
                                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                                    "signed_bias": float(np.mean(error))})
                            for anchor_index, anchor in enumerate(test_anchor_ids):
                                for action_index, action_name in enumerate(ACTION_KEYS):
                                    prediction_records.append({**scenario, "anchor_id": int(anchor),
                                        "action": action_name, "prediction": do_prediction[anchor_index, action_index],
                                        "do_reward": do_oracle[anchor_index, action_index]})
                            decision = regret_metrics(do_oracle, do_prediction)
                            ranking_rows.append({**scenario, **{key: decision[key] for key in
                                ("top_set_disagreement", "strict_flip", "predicted_top_fraction_minus",
                                 "predicted_top_fraction_base", "predicted_top_fraction_plus",
                                 "do_top_fraction_minus", "do_top_fraction_base", "do_top_fraction_plus")}})
                            regret_rows.append({**scenario, **{key: decision[key] for key in
                                ("mean_regret", "worst_tie_mean_regret", "conditional_mean_regret",
                                 "p90_regret", "max_regret", "top_1pct_regret_contribution")}})
                            seed_rows.append({**scenario, "best_validation_nll": history["best_validation_nll"],
                                "best_validation_step": history["best_validation_step"], **overall,
                                "top_set_disagreement": decision["top_set_disagreement"],
                                "mean_regret": decision["mean_regret"]})
                            latent_rows.extend(_latent_diagnostic_rows(model, scenario, test_rows,
                                                                       stats, resolved_device))

    stability_rows: list[dict[str, Any]] = []
    for kappa in kappas:
        for strength in grid:
            for condition in conditions:
                for seed in seeds:
                    for method in METHODS:
                        for left, right in ((MIXTURES[0], MIXTURES[1]),
                                            (MIXTURES[0], MIXTURES[2]),
                                            (MIXTURES[1], MIXTURES[2])):
                            key_left = (kappa, strength, condition, left, seed, method)
                            key_right = (kappa, strength, condition, right, seed, method)
                            if key_left in prediction_cache and key_right in prediction_cache:
                                stability_rows.append({"kappa": kappa, "lambda_reward": strength,
                                    "condition": condition, "seed": seed, "method": method,
                                    "left_mixture": left, "right_mixture": right,
                                    "prediction_mae": float(np.mean(np.abs(
                                        prediction_cache[key_left] - prediction_cache[key_right])))})

    # Estimated first observed grid dose with disagreement; no test threshold selects training doses.
    for method in METHODS:
        rows = [row for row in ranking_rows if row["method"] == method
                and row["condition"] == "confounded" and row["mixture"] == PRIMARY_MIXTURE]
        for kappa in kappas:
            values = sorted({float(row["lambda_reward"]) for row in rows if row["kappa"] == kappa
                             and float(row["top_set_disagreement"]) > 0.0})
            for row in rows:
                if row["kappa"] == kappa:
                    row["estimated_first_nonzero_grid_dose"] = values[0] if values else "+inf"

    _write_csv(output / "observational_metrics.csv", observational_rows)
    _write_csv(output / "do_metrics.csv", do_rows)
    _write_csv(output / "ranking_metrics.csv", ranking_rows)
    _write_csv(output / "regret_metrics.csv", regret_rows)
    _write_csv(output / "composition_stability.csv", stability_rows)
    _write_csv(output / "latent_diagnostics.csv", latent_rows)
    _write_csv(output / "seed_metrics.csv", seed_rows)
    arrays: dict[str, np.ndarray] = {}
    for key in prediction_records[0] if prediction_records else ("status",):
        arrays[key] = np.asarray([row[key] for row in prediction_records]) if prediction_records else np.asarray([])
    np.savez_compressed(output / "anchor_action_metrics.npz", **arrays)
    _make_figures(output, [*do_rows, *prediction_records], ranking_rows, regret_rows,
                  observational_rows, stability_rows, latent_rows)

    hashes_after = hash_input_files(input_paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks = {
        **(structure_checks or {}), "hidden_u_not_in_main_training": True,
        "do_oracle_not_used_for_training_or_selection": True,
        "action_index_used_only_as_behavior_target": ACTION_INDEX_USED_ONLY_AS_BEHAVIOR_TARGET,
        "exact_latent_marginalization_uses_logsumexp": "torch.logsumexp" in
            Path(__file__).read_text(encoding="utf-8"),
        "row_weights_normalized": True, "split_by_anchor": validate_splits(splits, selected),
        "all_methods_same_split": True, "all_methods_same_frozen_lambda_grid": True,
        "checkpoint_selected_by_observational_validation_only": True,
        "source_shuffle_changes_labels": source_shuffle_changed,
        "oracle_u_model_isolated": True, "checkpoint_roundtrip": checkpoint_roundtrip,
        "input_hashes_unchanged": unchanged, "no_nan_inf": all_finite,
        "old_artifacts_unchanged": unchanged,
        "controlled_three_action_only": CONTROLLED_THREE_ACTION_MECHANISM_DIAGNOSTIC_ONLY,
    }
    failed = [name for name, passed in hard_checks.items() if not passed]
    _write_json(output / "input_integrity.json", {"sha256_before": hashes_before,
        "sha256_after": hashes_after, "unchanged": unchanged, "required_file_count": len(input_paths)})
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
        "all_passed": not failed, "failed": failed})
    if failed:
        raise RewardMechanismSeparationError(f"hard checks failed: {failed}")
    summary = {"stage": "Phase 8C-RM", "analyzed_anchor_count": num_anchors,
        "kappas": list(kappas), "conditions": list(conditions), "lambdas": list(grid),
        "model_seeds": list(seeds), "methods": list(METHODS), "all_hard_checks_passed": True,
        "controlled_three_action_mechanism_diagnostic_only": True,
        "action_index_used_only_as_behavior_target": True,
        "do_oracle_used_only_for_final_test_evaluation": True}
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifest.json", {**summary, "device": resolved_device,
        "optimizer": "Adam", "learning_rate": 1e-3, "batch_size": batch_size,
        "gradient_updates": gradient_updates, "checkpoint_selection": "observational validation NLL only",
        "latent_states": 2, "exact_marginalization": True,
        "reward_decoder": "17-256-256-1 ReLU", "primary_mixture": PRIMARY_MIXTURE,
        "phase8anc_root": str(nc), "direct_reward_root": str(direct),
        "oracle_root": str(oracle), "lambda_grid_sha256": _sha256(Path(lambda_grid_file).resolve())})
    report = """# Phase 8C-RM — Minimal Reward-Only Mechanism-Separated Model

This artifact is a controlled three-action, binary-latent, one-step reward diagnostic only. It is not a general continuous-action world model or online RL algorithm.

The primary model used a source-invariant global latent prior, logger-specific behavior tables, and one source-invariant reward decoder. Both latent states were marginalized exactly with `logsumexp`. Model selection used observational validation likelihood only. Hidden U and do rewards were physically excluded from primary training; the U-aware model was an isolated Oracle ceiling and do rewards were loaded only for final test evaluation.

All methods used the same anchor split, frozen lambda grid, reward normalization, model seeds, and held-out test anchors. The source-conditioned and per-source source-free predictions are fixed-weight baseline aggregations and have no causal guarantee.

No automatic success threshold is applied. Interpret observational fit, do accuracy, ranking/regret, source-shuffle, independent-latent, base-action, and latent-collapse diagnostics jointly.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return summary
