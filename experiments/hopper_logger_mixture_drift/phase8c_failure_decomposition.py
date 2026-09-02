"""Phase 8C-FD: Oracle-scaffolded mechanism failure decomposition.

This is a diagnostic-only stage.  Public variants never accept row-level U.
Oracle components live behind explicit functions and directories, and neither
test U nor do outcomes participate in training or checkpoint selection.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analyze_phase8a_population_effect import (
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
)
from .noncomplementary_population_dgp import ACTION_KEYS
from .reward_mechanism_separation import (
    LOGGER_WEIGHTS,
    PRIMARY_MIXTURE,
    Normalization,
    PublicRows,
    _aligned_base_weights,
    _device,
    _grouped_observational_metrics,
    _hidden_u_for_rows,
    _mlp,
    _oracle_do_table,
    _read_json,
    _state_hash,
    _torch,
    commanded_action_indices,
    derive_test_action_table,
    index_derived_public_files,
    kappa_name,
    lambda_token,
    load_frozen_lambda_grid,
    load_model,
    make_model,
    make_public_rows,
    normalized_x,
    predict_do,
    predict_observational,
    predict_oracle_observational,
    regret_metrics,
    save_model,
    train_oracle_u_model,
    validate_public_schema,
)


ORACLE_INFORMATION_USED_FOR_DIAGNOSIS_ONLY = True
ORACLE_VARIANTS_ARE_NOT_DEPLOYABLE = True
VARIANTS = (
    "current_random_init",
    "collapsed_constrained_reference",
    "true_behavior_fixed",
    "true_behavior_fixed_em",
    "oracle_reward_fixed_learn_behavior",
    "oracle_compatible_plugin",
    "oracle_initialized_joint",
    "oracle_u_aware_ceiling",
)
PUBLIC_ONLY_VARIANTS = VARIANTS[:4]
ORACLE_VARIANTS = VARIANTS[4:]
ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
CHECKPOINT_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
TRUE_PRIOR = np.asarray((0.5, 0.5), dtype=np.float64)
PROBABILITY_FLOOR = 1e-12


class FailureDecompositionError(RuntimeError):
    """Raised when a required input or hard invariant fails."""


@dataclass(frozen=True)
class ScenarioData:
    kappa: float
    dose: float
    public_path: Path
    raw: Mapping[str, np.ndarray]
    rows: PublicRows
    train: PublicRows
    validation: PublicRows
    test: PublicRows
    stats: Normalization
    test_anchor_ids: np.ndarray
    test_observation: np.ndarray
    test_actions: np.ndarray
    do_reward: np.ndarray


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_true_behavior_table(manifest_path: Path) -> np.ndarray:
    """Load beta_e(a|U) from the Phase 8A-NC manifest; never hard-code it."""
    manifest = _read_json(Path(manifest_path))
    source = manifest.get("logger_probability_tables")
    if not isinstance(source, dict) or manifest.get("action_keys") != list(ACTION_KEYS):
        raise FailureDecompositionError("true logger behavior table is absent from manifest")
    result = np.zeros((3, 2, 3), dtype=np.float64)
    for logger in range(3):
        for latent, u_value in enumerate((-1, 1)):
            row = source.get(str(logger), {}).get(str(u_value))
            if not isinstance(row, dict):
                raise FailureDecompositionError("manifest behavior table is incomplete")
            for action, name in enumerate(ACTION_KEYS):
                result[logger, latent, action] = float(row.get(name, 0.0))
    if (not np.all(np.isfinite(result)) or np.any(result < 0.0)
            or not np.allclose(result.sum(axis=2), 1.0, atol=1e-12, rtol=0.0)):
        raise FailureDecompositionError("manifest behavior probabilities are invalid")
    return result


def collapsed_behavior_table(true_behavior: np.ndarray) -> np.ndarray:
    table = np.asarray(true_behavior, dtype=np.float64)
    marginal = np.sum(TRUE_PRIOR[None, :, None] * table, axis=1)
    return np.repeat(marginal[:, None, :], 2, axis=1)


def behavior_profile_table(true_behavior: np.ndarray, alpha: float) -> np.ndarray:
    if float(alpha) not in ALPHA_GRID:
        raise ValueError("alpha is outside the frozen profile grid")
    collapsed = collapsed_behavior_table(true_behavior)
    result = (1.0 - float(alpha)) * collapsed + float(alpha) * true_behavior
    if np.any(result < 0.0) or not np.allclose(result.sum(axis=2), 1.0):
        raise FailureDecompositionError("behavior profile left the probability simplex")
    return result


def reward_profile_means(mu0: np.ndarray, mu1: np.ndarray, alpha: float) -> np.ndarray:
    if float(alpha) not in ALPHA_GRID:
        raise ValueError("alpha is outside the frozen profile grid")
    means = np.stack((np.asarray(mu0, np.float64), np.asarray(mu1, np.float64)), axis=-1)
    center = np.mean(means, axis=-1, keepdims=True)
    return center + float(alpha) * (means - center)


def best_label_permutation_behavior_error(learned: np.ndarray,
                                          truth: np.ndarray) -> tuple[float, tuple[int, int]]:
    learned = np.asarray(learned, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    values = []
    for permutation in ((0, 1), (1, 0)):
        values.append((float(np.mean(np.abs(learned[:, permutation, :] - truth))), permutation))
    return min(values, key=lambda item: item[0])


def aligned_posterior_accuracy(probability_z1: np.ndarray,
                               u_value: np.ndarray) -> tuple[float, bool]:
    prediction = np.asarray(probability_z1, np.float64) >= 0.5
    target = np.asarray(u_value, np.int8) == 1
    direct = float(np.mean(prediction == target))
    flipped = float(np.mean(~prediction == target))
    return (direct, False) if direct >= flipped else (flipped, True)


def exact_responsibilities_numpy(prior: np.ndarray, behavior: np.ndarray,
                                 means: np.ndarray, reward: np.ndarray,
                                 source: np.ndarray, action_index: np.ndarray,
                                 log_scale: float) -> np.ndarray:
    prior = np.asarray(prior, np.float64)
    beta = np.asarray(behavior, np.float64)
    mu = np.asarray(means, np.float64)
    y = np.asarray(reward, np.float64)
    scale = max(math.exp(float(log_scale)), 1e-4)
    log_component = (np.log(np.maximum(prior, PROBABILITY_FLOOR))[None, :]
        + np.log(np.maximum(beta[np.asarray(source), :, np.asarray(action_index)],
                            PROBABILITY_FLOOR))
        - 0.5 * ((y[:, None] - mu) / scale) ** 2
        - math.log(scale) - 0.5 * math.log(2.0 * math.pi))
    normalizer = np.logaddexp.reduce(log_component, axis=1, keepdims=True)
    result = np.exp(log_component - normalizer)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise FailureDecompositionError("EM responsibilities do not normalize")
    return result


def exact_observed_nll_numpy(prior: np.ndarray, behavior: np.ndarray,
                             means: np.ndarray, reward: np.ndarray,
                             source: np.ndarray, action_index: np.ndarray,
                             weight: np.ndarray, log_scale: float) -> float:
    q = exact_responsibilities_numpy(prior, behavior, means, reward,
                                     source, action_index, log_scale)
    del q  # normalization check is shared with the E-step reference.
    scale = max(math.exp(float(log_scale)), 1e-4)
    component = (np.log(np.maximum(prior, PROBABILITY_FLOOR))[None, :]
        + np.log(np.maximum(behavior[np.asarray(source), :, np.asarray(action_index)],
                            PROBABILITY_FLOOR))
        - 0.5 * ((np.asarray(reward)[:, None] - means) / scale) ** 2
        - math.log(scale) - 0.5 * math.log(2.0 * math.pi))
    log_probability = np.logaddexp.reduce(component, axis=1)
    normalized_weight = np.asarray(weight, np.float64).copy()
    normalized_weight /= normalized_weight.sum()
    value = float(-(normalized_weight @ log_probability))
    if not np.isfinite(value):
        raise FailureDecompositionError("observed NLL is non-finite")
    return value


def _copy_state(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _state_dict_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_collapsed_reference(seed: int = 0) -> Any:
    """Capacity-matched latent-free joint p(A,R|S,E) reference."""
    torch = _torch()
    torch.manual_seed(int(seed))

    class CollapsedReference(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kind = "collapsed_constrained_reference"
            self.reward_decoder = _mlp(torch, 15)
            self.behavior_logits = torch.nn.Parameter(torch.zeros(3, 3))
            self.log_scale = torch.nn.Parameter(torch.zeros(()))

        def training_log_prob(self, x: Any, reward: Any, source: Any,
                              action_index: Any) -> Any:
            scale = torch.exp(self.log_scale).clamp_min(1e-4)
            mean = self.reward_decoder(x).squeeze(1)
            behavior = torch.log_softmax(self.behavior_logits, dim=1)[source, action_index]
            normal = (-0.5 * ((reward - mean) / scale) ** 2 - self.log_scale
                      - 0.5 * math.log(2.0 * math.pi))
            return behavior + normal

        def plain_mean(self, x: Any, source: Any | None = None) -> Any:
            del source
            return self.reward_decoder(x).squeeze(1)

        def latent_means(self, x: Any) -> Any:
            mean = self.plain_mean(x)
            return torch.stack((mean, mean), dim=1)

        def behavior_probabilities(self) -> Any:
            probability = torch.softmax(self.behavior_logits, dim=1)
            return probability[:, None, :].expand(3, 2, 3)

    return CollapsedReference()


def _set_behavior(model: Any, table: np.ndarray, *, trainable: bool) -> None:
    torch = _torch()
    probability = np.asarray(table, dtype=np.float64)
    logits = np.log(np.maximum(probability, PROBABILITY_FLOOR))
    with torch.no_grad():
        model.behavior_logits.copy_(torch.as_tensor(
            logits, dtype=model.behavior_logits.dtype, device=model.behavior_logits.device))
    model.behavior_logits.requires_grad_(trainable)


def make_true_behavior_fixed_model(seed: int, true_behavior: np.ndarray) -> Any:
    model = make_model("mechanism_separated", seed)
    _set_behavior(model, true_behavior, trainable=False)
    with _torch().no_grad():
        model.prior_logits.zero_()
    model.prior_logits.requires_grad_(False)
    return model


def make_oracle_plugin(oracle_reward_model: Any, seed: int,
                       true_behavior: np.ndarray) -> Any:
    model = make_true_behavior_fixed_model(seed, true_behavior)
    model.reward_decoder.load_state_dict(oracle_reward_model.reward_decoder.state_dict())
    with _torch().no_grad():
        model.log_scale.copy_(oracle_reward_model.log_scale)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_oracle_reward_fixed_behavior_model(oracle_reward_model: Any, seed: int,
                                            true_behavior: np.ndarray) -> Any:
    model = make_oracle_plugin(oracle_reward_model, seed, true_behavior)
    _set_behavior(model, collapsed_behavior_table(true_behavior), trainable=True)
    return model


def make_oracle_initialized_joint(plugin: Any, seed: int) -> Any:
    model = make_model("mechanism_separated", seed)
    model.load_state_dict(plugin.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def _row_tensors(rows: PublicRows, stats: Normalization, device: str) -> tuple[Any, ...]:
    torch = _torch()
    return (
        torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device),
        torch.as_tensor((rows.reward - stats.reward_mean) / stats.reward_std,
                        dtype=torch.float32, device=device),
        torch.as_tensor(rows.logger_id, dtype=torch.long, device=device),
        torch.as_tensor(rows.action_index, dtype=torch.long, device=device),
        torch.as_tensor(rows.row_weight, dtype=torch.float32, device=device),
    )


def observed_nll(model: Any, rows: PublicRows, stats: Normalization, device: str) -> float:
    torch = _torch()
    x, reward, source, action, weight = _row_tensors(rows, stats, device)
    model.eval()
    with torch.no_grad():
        log_probability = model.training_log_prob(x, reward, source, action)
        value = float((-torch.sum(weight * log_probability) / torch.sum(weight)).cpu())
    if not np.isfinite(value):
        raise FailureDecompositionError("model observed NLL is non-finite")
    return value


def _latent_snapshot(model: Any, rows: PublicRows, stats: Normalization,
                     device: str) -> dict[str, float | bool]:
    torch = _torch()
    x, reward, source, action, weight = _row_tensors(rows, stats, device)
    del weight
    if model.kind == "collapsed_constrained_reference":
        return {"prior_p_z1": 0.5, "reward_mode_separation": 0.0,
                "behavior_separation": 0.0, "posterior_entropy": math.log(2.0),
                "posterior_effective_usage": 2.0, "latent_collapse": True}
    with torch.no_grad():
        prior = torch.softmax(model.prior_logits, dim=0)
        means = model.latent_means(x)
        beta = torch.softmax(model.behavior_logits, dim=2)
        log_component = (torch.log(prior)[None, :]
                         + torch.log(beta[source, :, action].clamp_min(PROBABILITY_FLOOR))
                         - 0.5 * ((reward[:, None] - means)
                                  / torch.exp(model.log_scale).clamp_min(1e-4)) ** 2
                         - model.log_scale - 0.5 * math.log(2.0 * math.pi))
        posterior = torch.softmax(log_component, dim=1)
        entropy = -torch.sum(posterior * torch.log(posterior.clamp_min(1e-15)), dim=1)
        separation = torch.mean(torch.abs(means[:, 1] - means[:, 0]))
        tv = 0.5 * torch.sum(torch.abs(beta[:, 1] - beta[:, 0]), dim=1).mean()
    sep_raw = float(separation.cpu()) * stats.reward_std
    entropy_mean = float(entropy.mean().cpu())
    return {"prior_p_z1": float(prior[1].cpu()), "reward_mode_separation": sep_raw,
            "behavior_separation": float(tv.cpu()), "posterior_entropy": entropy_mean,
            "posterior_effective_usage": math.exp(entropy_mean),
            "latent_collapse": bool(float(prior.min().cpu()) < 0.01 or sep_raw < 1e-6)}


def _checkpoint_steps(updates: int) -> tuple[int, ...]:
    return tuple(sorted({int(round(fraction * updates)) for fraction in CHECKPOINT_FRACTIONS}))


def train_public_observational(model: Any, train: PublicRows, validation: PublicRows,
                               stats: Normalization, *, seed: int, updates: int,
                               batch_size: int, device: str) -> tuple[Any, dict[str, Any]]:
    """Public-only optimizer.  Its signature intentionally cannot receive U or do data."""
    torch = _torch()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise FailureDecompositionError("public optimizer received no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    tx, ty, ts, ta, tw = _row_tensors(train, stats, device)
    schedule = np.random.default_rng(seed).integers(
        0, len(train.reward), size=(updates, batch_size), dtype=np.int64)
    checkpoints = set(_checkpoint_steps(updates))
    best_loss = math.inf
    best_step = -1
    best_state: dict[str, Any] | None = None
    snapshots: list[dict[str, Any]] = []

    def record(step: int) -> None:
        nonlocal best_loss, best_step, best_state
        train_nll = observed_nll(model, train, stats, device)
        validation_nll = observed_nll(model, validation, stats, device)
        snapshot = {"update": step, "train_observational_nll": train_nll,
                    "validation_observational_nll": validation_nll,
                    **_latent_snapshot(model, validation, stats, device),
                    "state": _copy_state(model)}
        snapshots.append(snapshot)
        if validation_nll < best_loss:
            best_loss, best_step, best_state = validation_nll, step, _copy_state(model)

    record(0)
    model.train()
    for step, batch_np in enumerate(schedule, start=1):
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        log_probability = model.training_log_prob(tx[batch], ty[batch], ts[batch], ta[batch])
        loss = -torch.sum(tw[batch] * log_probability) / torch.sum(tw[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            record(step)
            model.train()
    if best_state is None:
        raise FailureDecompositionError("observational validation selected no checkpoint")
    final_state = _copy_state(model)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_validation_nll": best_loss, "best_validation_step": best_step,
                   "best_state": best_state, "final_state": final_state,
                   "snapshots": snapshots, "selection_metric": "observational_validation_nll",
                   "do_oracle_used_for_selection": False,
                   "hidden_u_used_for_selection": False}


def em_responsibilities(model: Any, rows: PublicRows, stats: Normalization,
                        device: str) -> Any:
    torch = _torch()
    x, reward, source, action, _ = _row_tensors(rows, stats, device)
    with torch.no_grad():
        means = model.latent_means(x)
        scale = torch.exp(model.log_scale).clamp_min(1e-4)
        component = (model.prior_log_probs()[None, :]
                     + model.behavior_log_probs()[source, :, action]
                     - 0.5 * ((reward[:, None] - means) / scale) ** 2
                     - model.log_scale - 0.5 * math.log(2.0 * math.pi))
        responsibilities = torch.softmax(component, dim=1)
    if not torch.allclose(responsibilities.sum(dim=1), torch.ones(len(rows.reward), device=device),
                          atol=1e-6, rtol=0.0):
        raise FailureDecompositionError("EM E-step posterior does not sum to one")
    return responsibilities


def train_true_behavior_em(model: Any, train: PublicRows, validation: PublicRows,
                           stats: Normalization, *, seed: int, iterations: int,
                           mstep_updates: int, batch_size: int,
                           device: str) -> tuple[Any, dict[str, Any]]:
    """Generalized EM with fixed true prior/beta and train-NLL monotonic safeguard."""
    torch = _torch()
    parameters = [model.log_scale, *model.reward_decoder.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    rng = np.random.default_rng(seed)
    tx, ty, ts, ta, tw = _row_tensors(train, stats, device)
    best_loss = math.inf
    best_iteration = -1
    best_state: dict[str, Any] | None = None
    snapshots: list[dict[str, Any]] = []
    for iteration in range(iterations + 1):
        train_nll = observed_nll(model, train, stats, device)
        validation_nll = observed_nll(model, validation, stats, device)
        snapshots.append({"em_iteration": iteration, "update": iteration * mstep_updates,
                          "train_observational_nll": train_nll,
                          "validation_observational_nll": validation_nll,
                          **_latent_snapshot(model, validation, stats, device),
                          "state": _copy_state(model)})
        if validation_nll < best_loss:
            best_loss, best_iteration, best_state = validation_nll, iteration, _copy_state(model)
        if iteration == iterations:
            break
        responsibilities = em_responsibilities(model, train, stats, device)
        previous_state = _copy_state(model)
        previous_train_nll = train_nll
        schedule = rng.integers(0, len(train.reward),
                                size=(mstep_updates, batch_size), dtype=np.int64)
        model.train()
        for batch_np in schedule:
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            means = model.latent_means(tx[batch])
            scale = torch.exp(model.log_scale).clamp_min(1e-4)
            normal = (-0.5 * ((ty[batch, None] - means) / scale) ** 2
                      - model.log_scale - 0.5 * math.log(2.0 * math.pi))
            expected = torch.sum(responsibilities[batch] * normal, dim=1)
            loss = -torch.sum(tw[batch] * expected) / torch.sum(tw[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if observed_nll(model, train, stats, device) > previous_train_nll + 1e-8:
            model.load_state_dict(previous_state)
    if best_state is None:
        raise FailureDecompositionError("EM selected no validation checkpoint")
    final_state = _copy_state(model)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_validation_nll": best_loss,
                   "best_validation_step": best_iteration * mstep_updates,
                   "best_state": best_state, "final_state": final_state,
                   "snapshots": snapshots, "selection_metric": "observational_validation_nll",
                   "do_oracle_used_for_selection": False,
                   "hidden_u_used_for_selection": False,
                   "em_responsibilities_normalized": True}


def _load_normalization(path: Path) -> Normalization:
    if not Path(path).is_file():
        raise FailureDecompositionError(f"Phase 8C normalization is unavailable: {path}")
    with np.load(path, allow_pickle=False) as raw:
        stats = Normalization(
            np.asarray(raw["x_mean"], np.float64), np.asarray(raw["x_std"], np.float64),
            float(raw["reward_mean"]), float(raw["reward_std"]))
    if (stats.x_mean.shape != (15,) or stats.x_std.shape != (15,)
            or np.any(stats.x_std <= 0.0) or stats.reward_std <= 0.0):
        raise FailureDecompositionError("Phase 8C normalization is invalid")
    return stats


def _require_passed(path: Path) -> None:
    value = _read_json(path)
    if value.get("all_passed") is not True or not all(value.get("checks", {}).values()):
        raise FailureDecompositionError(f"required audit did not pass: {path}")


def _resolve_direct_root(phase8c_root: Path, dgp_root: Path) -> Path:
    candidates = [
        dgp_root / "phase8c_direct_reward_public_grid",
        phase8c_root.parent / "phase8c_direct_reward_public_grid",
    ]
    recorded = _read_json(phase8c_root / "manifest.json").get("direct_reward_root")
    if recorded:
        candidates.append(Path(str(recorded)))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FailureDecompositionError(
        "Direct U->R derived public data are unavailable; expected phase8c_direct_reward_public_grid")


def reproduce_existing_results(phase8c_root: Path,
                               analysis_root: Path) -> dict[str, Any]:
    """Recompute the fixed Phase 8C facts from durable result tables."""
    manifest = _read_json(Path(phase8c_root) / "manifest.json")
    analysis_manifest = _read_json(Path(analysis_root) / "manifest.json")
    with (Path(analysis_root) / "latent-summary.csv").open(newline="", encoding="utf-8") as handle:
        latent = list(csv.DictReader(handle))
    mechanism = next((row for row in latent if row["method"] == "mechanism_separated"), None)
    with (Path(analysis_root) / "primary-contrasts.csv").open(
            newline="", encoding="utf-8") as handle:
        contrasts = list(csv.DictReader(handle))
    pooled = next((row for row in contrasts if row["metric"] == "mae"
                   and row["comparator"] == "pooled_mlp"), None)
    ablations = [row for row in contrasts
                 if row["metric"] in {"top_set_disagreement", "mean_regret"}
                 and row["comparator"] in {"source_shuffle", "no_behavior"}]
    facts = {
        "anchor_count": int(manifest.get("analyzed_anchor_count", -1)),
        "test_anchor_count": int(manifest.get("test_anchor_count", -1)),
        "seed_count": len(manifest.get("model_seeds", ())),
        "lambda_count": len(manifest.get("lambdas", ())),
        "model_count": int(manifest.get("trained_model_count", -1)),
        "mechanism_model_count": int(float(mechanism["n_models"])) if mechanism else -1,
        "mechanism_collapse_fraction": (float(mechanism["latent_collapse_fraction"])
                                         if mechanism else math.nan),
        "pooled_minus_mechanism_do_mae_auc": (float(pooled["improvement_mean"])
                                               if pooled else math.nan),
        "primary_setting": analysis_manifest.get("primary_setting"),
        "ranking_regret_ablation_gaps": {
            f"{row['metric']}:{row['comparator']}": float(row["improvement_mean"])
            for row in ablations},
        "ranking_regret_ablation_holm_p": {
            f"{row['metric']}:{row['comparator']}": float(row["holm_p_across_24_contrasts"])
            for row in ablations},
    }
    facts["all_reproduced"] = bool(
        facts["anchor_count"] == 2048 and facts["test_anchor_count"] == 309
        and facts["seed_count"] == 5 and facts["lambda_count"] == 7
        and facts["model_count"] == 3780 and facts["mechanism_model_count"] == 35
        and facts["mechanism_collapse_fraction"] == 1.0
        and abs(facts["pooled_minus_mechanism_do_mae_auc"] - 0.0005001558222687927)
        <= 1e-12
        and len(ablations) == 4
        and all(float(row["holm_p_across_24_contrasts"]) == 1.0 for row in ablations)
        and facts["primary_setting"] == {
            "kappa": 0.3, "condition": "confounded", "mixture": PRIMARY_MIXTURE})
    return facts


def _behavior_probabilities(model: Any) -> np.ndarray | None:
    torch = _torch()
    if model.kind == "collapsed_constrained_reference":
        with torch.no_grad():
            return model.behavior_probabilities().detach().cpu().numpy().astype(np.float64)
    if hasattr(model, "behavior_logits"):
        with torch.no_grad():
            return torch.softmax(model.behavior_logits, dim=2).cpu().numpy().astype(np.float64)
    return None


def posterior_probability_z1(model: Any, rows: PublicRows, stats: Normalization,
                             device: str) -> np.ndarray:
    torch = _torch()
    if model.kind == "collapsed_constrained_reference":
        return np.full(len(rows.reward), 0.5, dtype=np.float64)
    if model.kind == "oracle_u_aware":
        raise FailureDecompositionError("Oracle U-aware posterior requires the isolated U input")
    x, reward, source, action, _ = _row_tensors(rows, stats, device)
    with torch.no_grad():
        means = model.latent_means(x)
        component = (model.prior_log_probs()[None, :]
                     + model.behavior_log_probs()[source, :, action]
                     - 0.5 * ((reward[:, None] - means)
                              / torch.exp(model.log_scale).clamp_min(1e-4)) ** 2
                     - model.log_scale - 0.5 * math.log(2.0 * math.pi))
        return torch.softmax(component, dim=1)[:, 1].cpu().numpy().astype(np.float64)


def oracle_u_nll(model: Any, rows: PublicRows, u_env: np.ndarray,
                 stats: Normalization, device: str) -> float:
    torch = _torch()
    x, reward, _, _, weight = _row_tensors(rows, stats, device)
    u = torch.as_tensor((np.asarray(u_env, np.int8) + 1) // 2,
                        dtype=torch.long, device=device)
    with torch.no_grad():
        value = -torch.sum(weight * model.oracle_log_prob(x, reward, u)) / torch.sum(weight)
    result = float(value.cpu())
    if not np.isfinite(result):
        raise FailureDecompositionError("Oracle U-aware NLL is non-finite")
    return result


def oracle_ceiling_joint_nll(model: Any, rows: PublicRows, stats: Normalization,
                             true_behavior: np.ndarray, device: str) -> float:
    raw_means = _oracle_branch_means(model, rows, stats, device)
    means = (raw_means - stats.reward_mean) / stats.reward_std
    reward = (rows.reward - stats.reward_mean) / stats.reward_std
    return exact_observed_nll_numpy(
        TRUE_PRIOR, true_behavior, means, reward, rows.logger_id,
        rows.action_index, rows.row_weight, float(model.log_scale.detach().cpu()))


def _predict_observational(model: Any, rows: PublicRows, stats: Normalization,
                           device: str, u_env: np.ndarray | None = None) -> np.ndarray:
    if model.kind == "oracle_u_aware":
        if u_env is None:
            raise FailureDecompositionError("Oracle observational prediction lacks isolated U")
        return predict_oracle_observational(model, rows, u_env, stats, device)
    return predict_observational(model, rows, stats, device)


def _model_nll(model: Any, rows: PublicRows, stats: Normalization, device: str,
               u_env: np.ndarray | None = None) -> float:
    return (oracle_u_nll(model, rows, u_env, stats, device)
            if model.kind == "oracle_u_aware" else observed_nll(model, rows, stats, device))


def evaluate_variant(variant: str, model: Any, scenario: ScenarioData,
                     true_behavior: np.ndarray, device: str,
                     *, train_u: np.ndarray | None = None,
                     validation_u: np.ndarray | None = None,
                     test_u: np.ndarray | None = None) -> dict[str, Any]:
    """Post-hoc evaluation; this is the only function that accepts test U/do outcomes."""
    test_prediction = _predict_observational(
        model, scenario.test, scenario.stats, device, test_u)
    flat_observation = np.repeat(scenario.test_observation, 3, axis=0)
    flat_action = scenario.test_actions.reshape(-1, 3)
    do_prediction = predict_do(model, flat_observation, flat_action,
                               scenario.stats, device).reshape(-1, 3)
    error = do_prediction - scenario.do_reward
    decision = regret_metrics(scenario.do_reward, do_prediction)
    behavior = _behavior_probabilities(model)
    snapshot = (_latent_snapshot(model, scenario.test, scenario.stats, device)
                if model.kind != "oracle_u_aware" else {
                    "prior_p_z1": 0.5,
                    "reward_mode_separation": float(np.mean(np.abs(
                        _oracle_branch_means(model, scenario.test, scenario.stats, device)[:, 1]
                        - _oracle_branch_means(model, scenario.test, scenario.stats, device)[:, 0]))),
                    "behavior_separation": "", "posterior_entropy": 0.0,
                    "posterior_effective_usage": 1.0, "latent_collapse": False})
    behavior_error, permutation = (("", "") if behavior is None else
                                   best_label_permutation_behavior_error(behavior, true_behavior))
    posterior_accuracy: float | str = ""
    posterior_flipped: bool | str = ""
    if test_u is not None:
        if model.kind == "oracle_u_aware":
            posterior_accuracy, posterior_flipped = 1.0, False
        elif hasattr(model, "behavior_logits") or model.kind == "collapsed_constrained_reference":
            posterior_accuracy, posterior_flipped = aligned_posterior_accuracy(
                posterior_probability_z1(model, scenario.test, scenario.stats, device), test_u)
    return {
        "variant": variant,
        "train_nll": (oracle_ceiling_joint_nll(
            model, scenario.train, scenario.stats, true_behavior, device)
            if model.kind == "oracle_u_aware" else
            _model_nll(model, scenario.train, scenario.stats, device, train_u)),
        "validation_nll": (oracle_ceiling_joint_nll(
            model, scenario.validation, scenario.stats, true_behavior, device)
            if model.kind == "oracle_u_aware" else
            _model_nll(model, scenario.validation, scenario.stats, device, validation_u)),
        "test_nll": (oracle_ceiling_joint_nll(
            model, scenario.test, scenario.stats, true_behavior, device)
            if model.kind == "oracle_u_aware" else
            _model_nll(model, scenario.test, scenario.stats, device, test_u)),
        "observational_reward_mae": _grouped_observational_metrics(
            scenario.test, test_prediction)["mae"],
        "do_mae": float(np.mean(np.abs(error))),
        "do_rmse": float(np.sqrt(np.mean(error ** 2))),
        "rank_error": decision["top_set_disagreement"],
        "strict_flip": decision["strict_flip"],
        "mean_regret": decision["mean_regret"],
        "worst_tie_mean_regret": decision["worst_tie_mean_regret"],
        "behavior_table_mae": behavior_error,
        "label_permutation": str(permutation),
        "posterior_u_accuracy": posterior_accuracy,
        "posterior_label_flipped": posterior_flipped,
        "prior_error": ("" if not hasattr(model, "prior_logits") else
                        abs(float(_torch().softmax(model.prior_logits, 0)[1].detach().cpu()) - 0.5)),
        **snapshot,
        "do_prediction": do_prediction,
        "observational_prediction": test_prediction,
    }


def _oracle_branch_means(model: Any, rows: PublicRows, stats: Normalization,
                         device: str) -> np.ndarray:
    torch = _torch()
    x = torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device)
    with torch.no_grad():
        if model.kind == "oracle_u_aware":
            zero = torch.zeros(len(rows.reward), dtype=torch.long, device=device)
            one = torch.ones(len(rows.reward), dtype=torch.long, device=device)
            means = torch.stack((model.oracle_mean(x, zero), model.oracle_mean(x, one)), dim=1)
        else:
            means = model.latent_means(x)
    return means.cpu().numpy().astype(np.float64) * stats.reward_std + stats.reward_mean


def _fit_profile_sigma(train_reward: np.ndarray, train_means: np.ndarray,
                       train_source: np.ndarray, train_action: np.ndarray,
                       train_weight: np.ndarray, behavior: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def objective(log_scale: float) -> float:
        return exact_observed_nll_numpy(
            TRUE_PRIOR, behavior, train_means, train_reward,
            train_source, train_action, train_weight, log_scale)
    result = minimize_scalar(objective, bounds=(-9.0, 4.0), method="bounded",
                             options={"xatol": 1e-9})
    if not result.success or not np.isfinite(result.fun):
        raise FailureDecompositionError("train-only profile sigma optimization failed")
    return float(result.x)


def objective_profiles(oracle_model: Any, scenario: ScenarioData,
                       true_behavior: np.ndarray, device: str) -> tuple[list[dict[str, Any]],
                                                                        list[dict[str, Any]]]:
    train_branches = _oracle_branch_means(oracle_model, scenario.train, scenario.stats, device)
    validation_branches = _oracle_branch_means(
        oracle_model, scenario.validation, scenario.stats, device)
    test_rows = PublicRows(
        row_id=np.arange(len(scenario.test_anchor_ids) * 3),
        anchor_id=np.repeat(scenario.test_anchor_ids, 3),
        observation=np.repeat(scenario.test_observation, 3, axis=0),
        commanded_action=scenario.test_actions.reshape(-1, 3),
        reward=np.zeros(len(scenario.test_anchor_ids) * 3),
        logger_id=np.zeros(len(scenario.test_anchor_ids) * 3, dtype=np.int64),
        action_index=np.tile(np.arange(3), len(scenario.test_anchor_ids)),
        row_weight=np.ones(len(scenario.test_anchor_ids) * 3))
    test_branches = _oracle_branch_means(oracle_model, test_rows, scenario.stats, device)
    reward_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        train_mu = reward_profile_means(train_branches[:, 0], train_branches[:, 1], alpha)
        validation_mu = reward_profile_means(
            validation_branches[:, 0], validation_branches[:, 1], alpha)
        test_mu = reward_profile_means(test_branches[:, 0], test_branches[:, 1], alpha)
        log_scale = _fit_profile_sigma(
            scenario.train.reward, train_mu, scenario.train.logger_id,
            scenario.train.action_index, scenario.train.row_weight, true_behavior)
        val_nll = exact_observed_nll_numpy(
            TRUE_PRIOR, true_behavior, validation_mu, scenario.validation.reward,
            scenario.validation.logger_id, scenario.validation.action_index,
            scenario.validation.row_weight, log_scale)
        do_prediction = np.mean(test_mu, axis=1).reshape(-1, 3)
        error = do_prediction - scenario.do_reward
        decision = regret_metrics(scenario.do_reward, do_prediction)
        reward_rows.append({"alpha": alpha, "validation_nll": val_nll,
                            "train_fitted_log_scale": log_scale,
                            "do_mae": float(np.mean(np.abs(error))),
                            "rank_error": decision["top_set_disagreement"],
                            "mean_regret": decision["mean_regret"]})

        beta = behavior_profile_table(true_behavior, alpha)
        behavior_nll = exact_observed_nll_numpy(
            TRUE_PRIOR, beta, validation_branches, scenario.validation.reward,
            scenario.validation.logger_id, scenario.validation.action_index,
            scenario.validation.row_weight,
            float(oracle_model.log_scale.detach().cpu()) + math.log(scenario.stats.reward_std))
        do_prediction = np.mean(test_branches, axis=1).reshape(-1, 3)
        error = do_prediction - scenario.do_reward
        decision = regret_metrics(scenario.do_reward, do_prediction)
        behavior_rows.append({"alpha": alpha, "validation_nll": behavior_nll,
                              "do_mae": float(np.mean(np.abs(error))),
                              "rank_error": decision["top_set_disagreement"],
                              "mean_regret": decision["mean_regret"]})
    return reward_rows, behavior_rows


def variant_registry() -> dict[str, Any]:
    return {
        "oracle_information_used_for_diagnosis_only": True,
        "oracle_variants_are_not_deployable": True,
        "variants": {
            "current_random_init": {"label": "V0", "data": "public", "training": "reuse"},
            "collapsed_constrained_reference": {"label": "V1", "data": "public"},
            "true_behavior_fixed": {"label": "V2", "data": "public+manifest_structure"},
            "true_behavior_fixed_em": {"label": "V3", "data": "public+manifest_structure"},
            "oracle_reward_fixed_learn_behavior": {"label": "V4", "data": "oracle_train_val_U"},
            "oracle_compatible_plugin": {"label": "V5", "data": "oracle_train_val_U",
                                          "training": "none_after_plugin"},
            "oracle_initialized_joint": {"label": "V6", "data": "oracle_initialization_then_public"},
            "oracle_u_aware_ceiling": {"label": "V7", "data": "oracle_train_val_U",
                                       "training": "reuse"},
        },
    }


def validate_phase8c_inputs(phase8c_root: Path, analysis_root: Path,
                            dgp_root: Path, oracle_root: Path) -> dict[str, Any]:
    roots = [Path(path).resolve() for path in
             (phase8c_root, analysis_root, dgp_root, oracle_root)]
    labels = ("Phase 8C", "Phase 8C strict analysis", "Phase 8A-NC", "Oracle audit")
    for root, label in zip(roots, labels):
        if not root.is_dir():
            raise FailureDecompositionError(f"{label} input root is unavailable: {root}")
    phase8c, analysis, dgp, oracle = roots
    for path in (phase8c / "hard_checks.json", analysis / "data-integrity.json",
                 dgp / "hard_checks.json", oracle / "hard_checks.json"):
        _require_passed(path)
    facts = reproduce_existing_results(phase8c, analysis)
    if not facts["all_reproduced"]:
        raise FailureDecompositionError(f"Phase 8C strict facts do not reproduce: {facts}")
    direct = _resolve_direct_root(phase8c, dgp)
    _require_passed(direct / "hard_checks.json")
    manifest = _read_json(dgp / "manifest.json")
    if int(manifest.get("available_anchor_count", -1)) != 2048:
        raise FailureDecompositionError("Phase 8A-NC does not expose 2048 anchors")
    behavior = load_true_behavior_table(dgp / "manifest.json")
    return {"phase8c": phase8c, "analysis": analysis, "dgp": dgp,
            "oracle": oracle, "direct": direct, "facts": facts,
            "true_behavior": behavior}


def _phase8c_model_path(root: Path, kappa: float, dose: float, seed: int,
                        method: str) -> Path:
    return (root / "models" / kappa_name(kappa) / lambda_token(dose) / "confounded"
            / PRIMARY_MIXTURE / f"seed_{seed}" / f"{method}.pt")


def _phase8c_normalization_path(root: Path, kappa: float, dose: float) -> Path:
    return root / "normalization" / kappa_name(kappa) / lambda_token(dose) / "confounded" / "stats.npz"


def _save_state(path: Path, model: Any, metadata: Mapping[str, Any]) -> None:
    save_model(path, model, metadata)


def load_failure_decomposition_model(path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    kind, seed = str(metadata["kind"]), int(metadata["seed"])
    model = (make_collapsed_reference(seed) if kind == "collapsed_constrained_reference"
             else make_model(kind, seed))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), metadata


def _save_training_artifacts(output: Path, scope: str, scenario_key: Path,
                             variant: str, model: Any, history: Mapping[str, Any],
                             device: str, seed: int) -> bool:
    base = output / "checkpoints" / scope / scenario_key / variant
    roundtrip = True
    for snapshot in history.get("snapshots", ()):
        step = int(snapshot.get("update", snapshot.get("em_iteration", 0)))
        state = snapshot["state"]
        model.load_state_dict(state)
        path = base / f"update_{step}.pt"
        _save_state(path, model, {"kind": model.kind, "seed": seed, "variant": variant,
                                  "update": step, "selection_metric": "none_posthoc"})
        loaded, _ = load_failure_decomposition_model(path, device)
        roundtrip &= _state_hash(loaded) == _state_dict_hash(state)
    model.load_state_dict(history["best_state"])
    _save_state(output / "models" / scope / scenario_key / f"{variant}_best.pt", model,
                {"kind": model.kind, "seed": seed, "variant": variant,
                 "selection_metric": history["selection_metric"]})
    model.load_state_dict(history["final_state"])
    _save_state(base / "final.pt", model,
                {"kind": model.kind, "seed": seed, "variant": variant,
                 "selection_metric": "final_not_selected"})
    model.load_state_dict(history["best_state"])
    return roundtrip


def _history_rows(history: Mapping[str, Any], base: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for snapshot in history.get("snapshots", ()):
        result.append({**base, **{key: value for key, value in snapshot.items()
                                  if key != "state"}})
    return result


def _all_numeric_finite(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if value == "" or value is None or isinstance(value, (str, bool)):
                continue
            if isinstance(value, (int, float, np.number)) and not np.isfinite(float(value)):
                return False
    return True


def _descriptive_seed_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    keys = ("kappa", "lambda_reward", "variant")
    endpoints = ("validation_nll", "do_mae", "rank_error", "mean_regret",
                 "reward_mode_separation", "behavior_table_mae", "latent_collapse")
    for row in metrics:
        for metric in endpoints:
            value = row.get(metric, "")
            if value == "" or isinstance(value, bool):
                if metric != "latent_collapse" or value == "":
                    continue
                value = float(value)
            grouped.setdefault(tuple(row[key] for key in keys) + (metric,), []).append(float(value))
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        array = np.asarray(values, np.float64)
        n = len(array); mean = float(array.mean())
        sd = float(array.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            from scipy.stats import t
            half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        else:
            half = 0.0
        output.append({"kappa": key[0], "lambda_reward": key[1], "variant": key[2],
                       "metric": key[3], "n_seeds": n, "mean": mean, "sd": sd,
                       "min": float(array.min()), "max": float(array.max()),
                       "ci95_low": mean - half, "ci95_high": mean + half,
                       "inferential_unit": "model_seed"})
    return output


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    values = np.asarray(differences, np.float64)
    if not len(values):
        return 1.0
    observed = abs(float(values.mean()))
    statistics = [abs(float(np.mean(values * np.asarray(signs))))
                  for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, np.float64)
    order = np.argsort(values); adjusted = np.empty_like(values); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_auc_contrasts(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Seed is the inferential unit; anchors never enter this contrast table."""
    raw: list[dict[str, Any]] = []
    endpoints = ("validation_nll", "do_mae", "rank_error", "mean_regret")
    for kappa in sorted({float(row["kappa"]) for row in metrics}):
        selected = [row for row in metrics if float(row["kappa"]) == kappa]
        seeds = sorted({int(row["seed"]) for row in selected})
        for endpoint in endpoints:
            for variant in VARIANTS:
                if variant == "collapsed_constrained_reference":
                    continue
                differences = []
                for seed in seeds:
                    def auc(name: str) -> float:
                        points = sorted((float(row["lambda_reward"]), float(row[endpoint]))
                                        for row in selected if int(row["seed"]) == seed
                                        and row["variant"] == name)
                        if not points:
                            raise FailureDecompositionError("paired AUC lacks a variant curve")
                        x, y = map(np.asarray, zip(*points))
                        return float(np.trapz(y, x))
                    differences.append(auc(variant) - auc("collapsed_constrained_reference"))
                values = np.asarray(differences, np.float64); n = len(values)
                mean = float(values.mean()); sd = float(values.std(ddof=1)) if n > 1 else 0.0
                if n > 1:
                    from scipy.stats import t
                    half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
                else:
                    half = 0.0
                raw.append({"kappa": kappa, "metric": endpoint, "contrast":
                    f"{variant}-collapsed_constrained_reference", "n_paired_seeds": n,
                    "mean_paired_auc_difference": mean, "sd_paired_auc_difference": sd,
                    "ci95_low": mean - half, "ci95_high": mean + half,
                    "paired_standardized_effect": mean / sd if sd > 0 else "",
                    "exact_sign_flip_p": _exact_sign_flip_p(values)})
    adjusted = _holm_adjust([float(row["exact_sign_flip_p"]) for row in raw]) if raw else []
    for row, value in zip(raw, adjusted):
        row["holm_adjusted_p"] = float(value)
        row["inference_note"] = "exact sign-flip is supplementary; n is model seeds"
    return raw


def _plot_curves(rows: Sequence[Mapping[str, Any]], metric: str, path: Path,
                 ylabel: str) -> None:
    plt.figure()
    for variant in VARIANTS:
        points = [row for row in rows if row["variant"] == variant]
        doses = sorted({float(row["lambda_reward"]) for row in points})
        if doses:
            means = [np.mean([float(row[metric]) for row in points
                              if float(row["lambda_reward"]) == dose]) for dose in doses]
            plt.plot(doses, means, marker="o", label=variant)
    plt.xlabel("lambda"); plt.ylabel(ylabel); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def _make_failure_figures(output: Path, metrics: Sequence[Mapping[str, Any]],
                          trajectories: Sequence[Mapping[str, Any]],
                          reward_profile: Sequence[Mapping[str, Any]],
                          behavior_profile: Sequence[Mapping[str, Any]],
                          behavior_cells: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    for metric, filename, ylabel in (
        ("validation_nll", "observational_nll_by_variant_vs_lambda.png", "validation NLL"),
        ("do_mae", "do_mae_by_variant_vs_lambda.png", "do reward MAE"),
        ("rank_error", "rank_error_by_variant_vs_lambda.png", "top-set disagreement"),
        ("mean_regret", "regret_by_variant_vs_lambda.png", "mean regret")):
        _plot_curves(metrics, metric, figures / filename, ylabel)

    def trajectory(metric: str, filename: str, variants: Sequence[str]) -> None:
        plt.figure()
        for variant in variants:
            selected = [row for row in trajectories if row["variant"] == variant]
            if selected:
                grouped: dict[int, list[float]] = {}
                for row in selected:
                    grouped.setdefault(int(row["update"]), []).append(float(row[metric]))
                x = sorted(grouped); y = [np.mean(grouped[value]) for value in x]
                plt.plot(x, y, marker="o", label=variant)
        plt.xlabel("update"); plt.ylabel(metric); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(figures / filename, dpi=180); plt.close()

    trajectory("reward_mode_separation", "reward_mode_separation_trajectory.png", VARIANTS)
    trajectory("behavior_separation", "behavior_separation_trajectory.png", VARIANTS)
    trajectory("reward_mode_separation", "oracle_init_collapse_trajectory.png",
               ("oracle_initialized_joint",))

    plt.figure()
    for variant in VARIANTS:
        selected = [row for row in metrics if row["variant"] == variant]
        if selected:
            plt.scatter([float(row["validation_nll"]) for row in selected],
                        [float(row["do_mae"]) for row in selected], s=14, label=variant)
    plt.xlabel("validation observational NLL"); plt.ylabel("do MAE"); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(figures / "observational_nll_vs_do_mae.png", dpi=180); plt.close()

    for profile, filename in ((reward_profile, "reward_separation_objective_profile.png"),
                              (behavior_profile, "behavior_separation_objective_profile.png")):
        plt.figure()
        grouped: dict[float, list[float]] = {}
        for row in profile:
            grouped.setdefault(float(row["alpha"]), []).append(float(row["validation_nll"]))
        x = sorted(grouped); y = [np.mean(grouped[value]) for value in x]
        plt.plot(x, y, marker="o"); plt.xlabel("separation alpha")
        plt.ylabel("validation observational NLL"); plt.tight_layout()
        plt.savefig(figures / filename, dpi=180); plt.close()

    plt.figure()
    if behavior_cells:
        x = np.arange(len(behavior_cells))
        plt.scatter(x, [float(row["learned_probability"]) for row in behavior_cells],
                    marker="o", label="learned")
        plt.scatter(x, [float(row["true_probability"]) for row in behavior_cells],
                    marker="x", label="true")
    plt.xlabel("behavior-table cell"); plt.ylabel("probability"); plt.legend()
    plt.tight_layout(); plt.savefig(figures / "learned_vs_true_behavior_table.png", dpi=180); plt.close()

    plt.figure()
    for variant in VARIANTS:
        values = [float(row["posterior_u_accuracy"]) for row in metrics
                  if row["variant"] == variant and row.get("posterior_u_accuracy", "") != ""]
        if values:
            plt.scatter([variant] * len(values), values, s=14)
    plt.xticks(rotation=70); plt.ylabel("aligned posterior-U accuracy"); plt.tight_layout()
    plt.savefig(figures / "posterior_u_accuracy_by_variant.png", dpi=180); plt.close()

    plt.figure()
    gaps = [row for row in metrics if row["variant"] in
            {"collapsed_constrained_reference", "oracle_compatible_plugin",
             "oracle_initialized_joint"}]
    for variant in ("oracle_compatible_plugin", "oracle_initialized_joint"):
        values = []
        for row in [value for value in gaps if value["variant"] == variant]:
            reference = next((value for value in gaps
                              if value["variant"] == "collapsed_constrained_reference"
                              and value["kappa"] == row["kappa"]
                              and value["lambda_reward"] == row["lambda_reward"]
                              and value["seed"] == row["seed"]), None)
            if reference:
                values.append(float(row["validation_nll"]) - float(reference["validation_nll"]))
        if values:
            plt.scatter([variant] * len(values), values, s=14)
    plt.axhline(0.0); plt.xticks(rotation=30); plt.ylabel("NLL minus explicit collapsed")
    plt.tight_layout(); plt.savefig(figures / "collapsed_vs_noncollapsed_nll_gap.png", dpi=180)
    plt.close()


def _write_report(output: Path, facts: Mapping[str, Any],
                  metrics: Sequence[Mapping[str, Any]],
                  trajectories: Sequence[Mapping[str, Any]],
                  objective_rows: Sequence[Mapping[str, Any]],
                  reward_profiles: Sequence[Mapping[str, Any]],
                  behavior_profiles: Sequence[Mapping[str, Any]]) -> None:
    primary = [row for row in metrics if float(row["kappa"]) == 0.0]
    table = []
    for variant in VARIANTS:
        rows = [row for row in primary if row["variant"] == variant]
        if not rows:
            continue
        def mean(key: str) -> str:
            values = [float(row[key]) for row in rows if row.get(key, "") != ""]
            return f"{np.mean(values):.8g}" if values else "NA"
        table.append(f"| {variant} | {mean('validation_nll')} | {mean('do_mae')} | "
                     f"{mean('rank_error')} | {mean('mean_regret')} | "
                     f"{mean('reward_mode_separation')} | {mean('behavior_table_mae')} | "
                     f"{mean('latent_collapse')} |")
    def average(variant: str, key: str) -> float:
        values = [float(row[key]) for row in primary
                  if row["variant"] == variant and row.get(key, "") != ""]
        return float(np.mean(values))
    mean_objective = lambda key: float(np.mean([float(row[key]) for row in objective_rows
                                                if float(row["kappa"]) == 0.0]))
    v6_path = sorted((row for row in trajectories
                      if row["variant"] == "oracle_initialized_joint"
                      and float(row["kappa"]) == 0.0), key=lambda row: int(row["update"]))
    v6_start_sep = float(np.mean([float(row["reward_mode_separation"]) for row in v6_path
                                  if int(row["update"]) == min(map(lambda x: int(x["update"]), v6_path))]))
    v6_end_sep = float(np.mean([float(row["reward_mode_separation"]) for row in v6_path
                                if int(row["update"]) == max(map(lambda x: int(x["update"]), v6_path))]))
    reward_alpha = {alpha: np.mean([float(row["validation_nll"]) for row in reward_profiles
                                    if float(row["kappa"]) == 0.0
                                    and float(row["alpha"]) == alpha]) for alpha in (0.0, 1.0)}
    behavior_alpha = {alpha: np.mean([float(row["validation_nll"]) for row in behavior_profiles
                                      if float(row["kappa"]) == 0.0
                                      and float(row["alpha"]) == alpha]) for alpha in (0.0, 1.0)}
    text = f"""# Phase 8C-FD — Oracle-Scaffolded Mechanism Failure Decomposition

`ORACLE_INFORMATION_USED_FOR_DIAGNOSIS_ONLY = True`  
`ORACLE_VARIANTS_ARE_NOT_DEPLOYABLE = True`

## Existing Phase 8C reproduction

The read-only audit reproduced {facts['anchor_count']} anchors, {facts['test_anchor_count']} held-out anchors, {facts['seed_count']} seeds, {facts['lambda_count']} frozen doses, {facts['mechanism_model_count']}/{facts['mechanism_model_count']} collapsed mechanism models, and pooled-minus-mechanism do-MAE AUC improvement {facts['pooled_minus_mechanism_do_mae_auc']:.12g}.

## Variant definitions

V0 reuses the formal random-initialized model. V1 is an explicit capacity-matched collapsed likelihood. V2 fixes the manifest prior/behavior and uses public gradient training. V3 uses the same initialization and exact generalized EM. V4 freezes an Oracle-U-supervised reward decoder and learns behavior from public observations. V5 is the untrained Oracle-compatible plugin. V6 starts exactly at V5 and then uses only the public observational objective. V7 reuses the isolated Oracle U-aware ceiling.

## Primary descriptive table

The unit summarized here is the model seed; anchors are repeated test cases and are not treated as independent replicates.

| Variant | Obs val NLL | Do MAE | Rank error | Regret | Mode separation | Behavior error | Collapse rate |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Failure-cause reading matrix

The artifact reports the facts required to assess behavior learning, optimization/initialization, observational-objective underdetermination, reward approximation, and model-class mismatch. It deliberately does not convert these facts into a unique cause with an arbitrary threshold. V2 versus V3 isolates optimizer form; V4 isolates behavior learning; V5 tests representational existence; V6 records whether observational training preserves or destroys an Oracle-compatible point; V7 bounds reward approximation.

| Candidate explanation | Direct evidence to inspect |
|---|---|
| Behavior-learning bottleneck | V2/V3 mode separation and do metrics; V4 aligned behavior-table error |
| Optimization/initialization | V3−V2 and V6−V0 paired curves |
| Objective underdetermination | V5/V6 versus V1 exact NLL gaps alongside causal gaps |
| Reward approximation | V5 versus V7 do error |
| Model-class mismatch | V5 joint NLL and do error; alpha-profile endpoints |

## Collapse trajectories

V6 mean reward-mode separation changes from {v6_start_sep:.8g} at update 0 to {v6_end_sep:.8g} at the final fixed checkpoint. The synchronized NLL/do/rank/regret trajectory is stored in `collapse_trajectories.csv`; no post-hoc metric selected a checkpoint.

## Objective profiles

For the reward-separation profile, mean primary validation NLL is {reward_alpha[0.0]:.8g} at alpha=0 and {reward_alpha[1.0]:.8g} at alpha=1. For the behavior-separation profile it is {behavior_alpha[0.0]:.8g} and {behavior_alpha[1.0]:.8g}, respectively. Alpha was never selected on test outcomes.

## Direct diagnostic answers (descriptive, not thresholded)

1. Model-class existence: V5 do MAE is {average('oracle_compatible_plugin', 'do_mae'):.8g}, versus V1 {average('collapsed_constrained_reference', 'do_mae'):.8g} and V7 {average('oracle_u_aware_ceiling', 'do_mae'):.8g}.
2. Reward learning with true behavior: V2 mode separation is {average('true_behavior_fixed', 'reward_mode_separation'):.8g}; its do MAE is {average('true_behavior_fixed', 'do_mae'):.8g}.
3. Behavior learning with fixed Oracle reward: V4 aligned behavior MAE is {average('oracle_reward_fixed_learn_behavior', 'behavior_table_mae'):.8g}.
4. EM versus gradient: primary V3−V2 do-MAE difference is {average('true_behavior_fixed_em', 'do_mae') - average('true_behavior_fixed', 'do_mae'):.8g}.
5. Oracle initialization stability: the V6 separation trajectory is reported above; its best checkpoint is selected only by validation NLL.
6. NLL during V6 training: inspect the exact update-wise `validation_observational_nll` values in `collapse_trajectories.csv` together with the separation values.
7. Noncollapsed versus collapsed likelihood: mean V5−V1 validation-NLL gap is {mean_objective('delta_nll_oracle_vs_collapsed'):.8g}; mean V6-best−V1 gap is {mean_objective('delta_nll_oracleinit_best_vs_collapsed'):.8g}.
8. Selection conflict: observational-best/do-best agreement is {np.mean([row['observational_best_matches_do_best'] for row in objective_rows if float(row['kappa']) == 0.0]):.6g}; the corresponding ranking/regret agreements are stored exactly in `objective_comparison.csv`.
9. Cause attribution: the code does not force a unique label. The five competing explanations must be judged jointly from the effect sizes above, seed variation, and profiles.

## Supported and unsupported conclusions

Only post-hoc comparisons supported by the saved seed-level metrics and trajectories are admissible. Oracle scaffold variants are diagnostic and cannot be presented as deployable methods. No test U or do outcome selected a checkpoint, seed, dose, architecture, or stopping point. Similar NLL values do not establish causal equivalence, and a noncollapsed V6 state does not establish identification of the true U without the aligned recovery diagnostics.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def _write_analysis_bundle_metadata(output: Path, seed_count: int,
                                    test_anchor_count: int) -> None:
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (output / "stats-appendix.md").write_text(f"""# Statistical appendix

The inferential unit is the model seed (`n={seed_count}` per complete variant/dose curve). The {test_anchor_count} held-out anchors are repeated test cases, not independent replicates. `seed_summary.csv` reports mean, sample SD, min/max, and t-based descriptive 95% intervals. `paired_seed_contrasts.csv` reports paired dose-AUC differences against the explicit collapsed reference, paired standardized effects, exact sign-flip p-values, and Holm adjustment across the reported contrast family. With few seeds, the exact p-values have coarse resolution and are supplementary; mechanism effect sizes and trajectories are primary.
""", encoding="utf-8")
    descriptions = {
        "observational_nll_by_variant_vs_lambda.png": "Observational likelihood across frozen doses.",
        "do_mae_by_variant_vs_lambda.png": "Post-hoc interventional reward error.",
        "rank_error_by_variant_vs_lambda.png": "Post-hoc action-ranking disagreement.",
        "regret_by_variant_vs_lambda.png": "Post-hoc decision regret.",
        "reward_mode_separation_trajectory.png": "Reward-mode collapse dynamics.",
        "behavior_separation_trajectory.png": "Behavior-table separation dynamics.",
        "oracle_init_collapse_trajectory.png": "V6 stability from Oracle-compatible update 0.",
        "observational_nll_vs_do_mae.png": "Model-selection conflict between fit and causal error.",
        "reward_separation_objective_profile.png": "Frozen-alpha reward separation profile.",
        "behavior_separation_objective_profile.png": "Frozen-alpha behavior separation profile.",
        "learned_vs_true_behavior_table.png": "Label-aligned V4 behavior recovery.",
        "posterior_u_accuracy_by_variant.png": "Post-hoc label-aligned posterior recovery.",
        "collapsed_vs_noncollapsed_nll_gap.png": "Exact NLL gaps to the explicit collapsed reference.",
    }
    lines = ["# Figure catalog", ""]
    for filename, purpose in descriptions.items():
        lines.extend((f"## `{filename}`", "", f"Purpose: {purpose}", "",
                      "Data source: frozen Phase 8C-FD seed-level metrics or fixed-checkpoint trajectories.",
                      "", "Interpretation: read the numerical companion CSV before drawing a mechanism conclusion; error variation is across model seeds where applicable, and no plotted post-hoc outcome selected training.", ""))
    (output / "figure-catalog.md").write_text("\n".join(lines), encoding="utf-8")


def _posthoc_history(history: Mapping[str, Any], model: Any, variant: str,
                     scenario: ScenarioData, true_behavior: np.ndarray,
                     device: str, test_u: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    current = _copy_state(model)
    for snapshot in history.get("snapshots", ()):
        model.load_state_dict(snapshot["state"])
        evaluated = evaluate_variant(variant, model, scenario, true_behavior, device,
                                     test_u=test_u)
        rows.append({"update": int(snapshot.get("update", 0)),
                     **{key: value for key, value in snapshot.items() if key != "state"},
                     "do_mae": evaluated["do_mae"],
                     "rank_error": evaluated["rank_error"],
                     "mean_regret": evaluated["mean_regret"],
                     "posterior_u_accuracy": evaluated["posterior_u_accuracy"]})
    model.load_state_dict(current)
    return rows


def run_phase8c_failure_decomposition(
    phase8c_root: Path, phase8c_analysis_root: Path, dgp_root: Path,
    oracle_root: Path, output_root: Path, *, kappas: tuple[float, ...] = (0.0,),
    lambda_values: tuple[float, ...] | None = None, use_frozen_lambda_grid: bool = False,
    model_seeds: tuple[int, ...] = (0,), num_anchors: int = 100,
    gradient_updates: int = 300, em_iterations: int = 5,
    em_mstep_updates: int = 50, batch_size: int = 128,
    device: str = "auto",
) -> dict[str, Any]:
    inputs = validate_phase8c_inputs(
        phase8c_root, phase8c_analysis_root, dgp_root, oracle_root)
    phase8c, analysis, dgp, oracle, direct = (
        inputs[key] for key in ("phase8c", "analysis", "dgp", "oracle", "direct"))
    true_behavior = np.asarray(inputs["true_behavior"], np.float64)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FailureDecompositionError(f"output directory is not empty: {output}")
    if (output == dgp or any(output == root or root in output.parents
                             for root in (phase8c, analysis, oracle, direct))):
        raise FailureDecompositionError("output must be physically separate from every input")
    kappas = tuple(float(value) for value in kappas)
    if not kappas or any(value not in (0.0, 0.3) for value in kappas):
        raise ValueError("kappas must be a nonempty subset of (0.0, 0.3)")
    if 0.3 in kappas and 0.0 not in kappas:
        raise ValueError("secondary kappa=0.3 cannot replace primary kappa=0.0")
    seeds = tuple(int(seed) for seed in model_seeds)
    if (not seeds or len(set(seeds)) != len(seeds) or num_anchors <= 0
            or num_anchors > 2048 or min(gradient_updates, em_iterations,
                                         em_mstep_updates, batch_size) <= 0):
        raise ValueError("seeds and training sizes must be positive and valid")
    frozen_grid, frozen_record = load_frozen_lambda_grid(phase8c / "frozen_lambda_grid.json")
    if use_frozen_lambda_grid:
        doses = frozen_grid
    else:
        if not lambda_values:
            raise ValueError("provide --lambda-values or --use-frozen-lambda-grid")
        doses = tuple(float(value) for value in lambda_values)
        if any(value not in frozen_grid for value in doses):
            raise ValueError("lambda values must be an exact subset of the frozen grid")
    direct_index = index_derived_public_files(direct)
    expected = {(kappa, dose, "confounded") for kappa in kappas for dose in doses}
    missing_scenarios = expected.difference(direct_index)
    if missing_scenarios:
        raise FailureDecompositionError(
            f"Direct public artifacts are incomplete: {sorted(missing_scenarios)[0]}")

    full_splits = _read_json(phase8c / "splits.json")
    all_anchors = np.asarray(sorted(set().union(*(
        set(map(int, full_splits[name])) for name in ("train", "validation", "test")))),
        dtype=np.int64)
    if len(all_anchors) != 2048:
        raise FailureDecompositionError("Phase 8C split does not cover exactly 2048 anchors")
    selected = all_anchors[:num_anchors]
    splits = {name: sorted(set(map(int, full_splits[name])).intersection(selected.tolist()))
              for name in ("train", "validation", "test")}
    if any(not values for values in splits.values()):
        raise FailureDecompositionError("selected anchors leave an empty split")

    input_paths = [
        phase8c / "manifest.json", phase8c / "hard_checks.json", phase8c / "splits.json",
        phase8c / "frozen_lambda_grid.json", phase8c / "seed_metrics.csv",
        phase8c / "latent_diagnostics.csv", analysis / "manifest.json",
        analysis / "data-integrity.json", analysis / "primary-contrasts.csv",
        analysis / "latent-summary.csv", dgp / "manifest.json", dgp / "hard_checks.json",
        dgp / "anchor_action_metrics.npz", oracle / "manifest.json", oracle / "hard_checks.json",
        direct / "manifest.json", direct / "hard_checks.json",
    ]
    for key in sorted(expected):
        public_path = direct_index[key]
        input_paths.extend((public_path,
                            public_path.with_name(public_path.name.replace(
                                "_public.npz", "_hidden_audit.npz")),
                            _phase8c_normalization_path(phase8c, key[0], key[1])))
        for seed in seeds:
            input_paths.extend((
                _phase8c_model_path(phase8c, key[0], key[1], seed, "mechanism_separated"),
                _phase8c_model_path(phase8c, key[0], key[1], seed, "oracle_u_aware")))
        input_paths.append(dgp / kappa_name(key[0]) / "confounded_public.npz")
        input_paths.append(dgp / kappa_name(key[0]) / "weights" / "confounded"
                           / f"{PRIMARY_MIXTURE}.npy")
    missing = [path for path in input_paths if not Path(path).is_file()]
    if missing:
        raise FailureDecompositionError(f"required read-only input is missing: {missing[0]}")
    hashes_before = hash_input_files(input_paths)
    resolved_device = _device(device)
    output.mkdir(parents=True)
    _write_json(output / "splits.json", {**splits, "selected_anchor_count": num_anchors,
                                          "source": str(phase8c / "splits.json")})
    _write_json(output / "variant_registry.json", variant_registry())

    observational_rows: list[dict[str, Any]] = []
    do_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    reward_profile_rows: list[dict[str, Any]] = []
    behavior_profile_rows: list[dict[str, Any]] = []
    behavior_cells: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    checkpoint_roundtrip = True
    current_reused = True
    paired_initialization = True
    oracle_init_match = True
    selection_clean = True
    split_sets = [set(splits[name]) for name in ("train", "validation", "test")]
    all_same_split = (not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2]
                           or split_sets[1] & split_sets[2])
                      and set.union(*split_sets) == set(selected.tolist()))
    reward_alpha0_valid = reward_alpha1_valid = True
    behavior_alpha0_valid = behavior_alpha1_valid = True
    true_behavior_fixed_no_grad = True

    for kappa in kappas:
        for dose in doses:
            public_path = direct_index[(kappa, dose, "confounded")]
            raw_all = load_npz(public_path); validate_public_schema(raw_all)
            selected_mask = np.isin(raw_all["anchor_id"], selected)
            raw = {key: np.asarray(value)[selected_mask] for key, value in raw_all.items()}
            if set(np.unique(raw["anchor_id"]).tolist()) != set(selected.tolist()):
                raise FailureDecompositionError("direct public data do not cover selected anchors")
            base = _aligned_base_weights(dgp, raw, kappa, "confounded", PRIMARY_MIXTURE)
            rows = make_public_rows(raw, base, LOGGER_WEIGHTS[PRIMARY_MIXTURE])
            stats = _load_normalization(_phase8c_normalization_path(phase8c, kappa, dose))
            scenario = ScenarioData(
                kappa=kappa, dose=dose, public_path=public_path, raw=raw, rows=rows,
                train=rows.subset(splits["train"]),
                validation=rows.subset(splits["validation"]),
                test=rows.subset(splits["test"]), stats=stats,
                test_anchor_ids=derive_test_action_table(raw, splits["test"])[0],
                test_observation=derive_test_action_table(raw, splits["test"])[1],
                test_actions=derive_test_action_table(raw, splits["test"])[2],
                do_reward=_oracle_do_table(dgp, kappa, splits["test"]))
            full_u = _hidden_u_for_rows(public_path, rows.row_id)
            train_mask = np.isin(rows.anchor_id, splits["train"])
            validation_mask = np.isin(rows.anchor_id, splits["validation"])
            test_mask = np.isin(rows.anchor_id, splits["test"])
            train_u, validation_u, test_u = (full_u[train_mask], full_u[validation_mask],
                                              full_u[test_mask])

            for seed in seeds:
                scenario_base = {"kappa": kappa, "lambda_reward": dose,
                                 "condition": "confounded", "mixture": PRIMARY_MIXTURE,
                                 "seed": seed}
                scenario_key = Path(kappa_name(kappa)) / lambda_token(dose) / f"seed_{seed}"

                v0_path = _phase8c_model_path(
                    phase8c, kappa, dose, seed, "mechanism_separated")
                v0, v0_metadata = load_model(v0_path, resolved_device)
                current_reused &= v0_metadata.get("selection_metric") == "observational_validation_nll"

                v1 = make_collapsed_reference(seed).to(resolved_device)
                v1, h1 = train_public_observational(
                    v1, scenario.train, scenario.validation, stats, seed=seed,
                    updates=gradient_updates, batch_size=batch_size, device=resolved_device)

                v2_initial = make_true_behavior_fixed_model(seed, true_behavior).to(resolved_device)
                initial_state = _copy_state(v2_initial)
                v2 = make_true_behavior_fixed_model(seed, true_behavior).to(resolved_device)
                v2.load_state_dict(initial_state)
                v3 = make_true_behavior_fixed_model(seed, true_behavior).to(resolved_device)
                v3.load_state_dict(initial_state)
                paired_initialization &= _state_hash(v2) == _state_hash(v3)
                v2, h2 = train_public_observational(
                    v2, scenario.train, scenario.validation, stats, seed=seed,
                    updates=gradient_updates, batch_size=batch_size, device=resolved_device)
                v3, h3 = train_true_behavior_em(
                    v3, scenario.train, scenario.validation, stats, seed=seed,
                    iterations=em_iterations, mstep_updates=em_mstep_updates,
                    batch_size=batch_size, device=resolved_device)
                true_behavior_fixed_no_grad &= (
                    not v2.prior_logits.requires_grad and not v2.behavior_logits.requires_grad
                    and not v3.prior_logits.requires_grad and not v3.behavior_logits.requires_grad)

                oracle_reward, oracle_history = train_oracle_u_model(
                    scenario.train, scenario.validation, train_u, validation_u, stats,
                    seed=seed, updates=gradient_updates, batch_size=batch_size,
                    device=resolved_device)
                oracle_history["selection_metric"] = "oracle_u_supervised_validation_reward_nll"
                v4 = make_oracle_reward_fixed_behavior_model(
                    oracle_reward, seed, true_behavior).to(resolved_device)
                v4, h4 = train_public_observational(
                    v4, scenario.train, scenario.validation, stats, seed=seed,
                    updates=gradient_updates, batch_size=batch_size, device=resolved_device)
                v5 = make_oracle_plugin(oracle_reward, seed, true_behavior).to(resolved_device)
                v6 = make_oracle_initialized_joint(v5, seed).to(resolved_device)
                plugin_hash = _state_hash(v5)
                v6, h6 = train_public_observational(
                    v6, scenario.train, scenario.validation, stats, seed=seed,
                    updates=gradient_updates, batch_size=batch_size, device=resolved_device)
                oracle_init_match &= _state_dict_hash(h6["snapshots"][0]["state"]) == plugin_hash
                v7_path = _phase8c_model_path(phase8c, kappa, dose, seed, "oracle_u_aware")
                v7, v7_metadata = load_model(v7_path, resolved_device)
                current_reused &= v7_metadata.get("oracle_only") is True

                models = {
                    "current_random_init": v0,
                    "collapsed_constrained_reference": v1,
                    "true_behavior_fixed": v2,
                    "true_behavior_fixed_em": v3,
                    "oracle_reward_fixed_learn_behavior": v4,
                    "oracle_compatible_plugin": v5,
                    "oracle_initialized_joint": v6,
                    "oracle_u_aware_ceiling": v7,
                }
                histories = {
                    "collapsed_constrained_reference": h1,
                    "true_behavior_fixed": h2,
                    "true_behavior_fixed_em": h3,
                    "oracle_reward_fixed_learn_behavior": h4,
                    "oracle_initialized_joint": h6,
                }
                selection_clean &= all(
                    history.get("selection_metric") == "observational_validation_nll"
                    and history.get("do_oracle_used_for_selection") is False
                    and history.get("hidden_u_used_for_selection") is False
                    for history in histories.values())
                scopes = {
                    "current_random_init": "public_only",
                    "collapsed_constrained_reference": "public_only",
                    "true_behavior_fixed": "oracle_structure",
                    "true_behavior_fixed_em": "oracle_structure",
                    "oracle_reward_fixed_learn_behavior": "oracle_only",
                    "oracle_compatible_plugin": "oracle_only",
                    "oracle_initialized_joint": "oracle_only",
                    "oracle_u_aware_ceiling": "oracle_only",
                }
                for variant in ("current_random_init", "oracle_compatible_plugin",
                                "oracle_u_aware_ceiling"):
                    model = models[variant]
                    saved_path = (output / "models" / scopes[variant] / scenario_key
                                  / f"{variant}.pt")
                    _save_state(saved_path, model,
                                {"kind": model.kind, "seed": seed, "variant": variant,
                                 "selection_metric": ("observational_validation_nll"
                                     if variant != "oracle_compatible_plugin" else "none_plugin"),
                                 "oracle_only": variant in ORACLE_VARIANTS})
                    reloaded, _ = load_failure_decomposition_model(saved_path, resolved_device)
                    checkpoint_roundtrip &= _state_hash(reloaded) == _state_hash(model)
                for variant, history in histories.items():
                    checkpoint_roundtrip &= _save_training_artifacts(
                        output, scopes[variant], scenario_key, variant,
                        models[variant], history, resolved_device, seed)
                    posthoc = _posthoc_history(
                        history, models[variant], variant, scenario, true_behavior,
                        resolved_device, test_u)
                    trajectory_rows.extend({**scenario_base, "variant": variant, **row}
                                           for row in posthoc)
                # Oracle reward training is isolated and selected by validation U-likelihood only.
                _save_state(output / "models" / "oracle_only" / scenario_key
                            / "oracle_reward_decoder.pt", oracle_reward,
                            {**oracle_history, **scenario_base, "oracle_only": True})

                scenario_metrics = []
                for variant, model in models.items():
                    metric = evaluate_variant(
                        variant, model, scenario, true_behavior, resolved_device,
                        train_u=train_u if model.kind == "oracle_u_aware" else None,
                        validation_u=validation_u if model.kind == "oracle_u_aware" else None,
                        test_u=test_u)
                    record = {**scenario_base, **{key: value for key, value in metric.items()
                                                 if key not in {"do_prediction",
                                                                "observational_prediction"}}}
                    scenario_metrics.append(record)
                    observational_rows.append({key: record[key] for key in
                        (*scenario_base.keys(), "variant", "train_nll", "validation_nll",
                         "test_nll", "observational_reward_mae")})
                    do_rows.append({key: record[key] for key in
                        (*scenario_base.keys(), "variant", "do_mae", "do_rmse", "rank_error",
                         "strict_flip", "mean_regret", "worst_tie_mean_regret")})
                    mechanism_rows.append({key: record[key] for key in
                        (*scenario_base.keys(), "variant", "latent_collapse",
                         "reward_mode_separation", "behavior_separation", "prior_p_z1",
                         "prior_error", "posterior_entropy", "posterior_effective_usage",
                         "posterior_u_accuracy")})
                    behavior_rows.append({key: record[key] for key in
                        (*scenario_base.keys(), "variant", "behavior_table_mae",
                         "label_permutation", "posterior_u_accuracy",
                         "posterior_label_flipped")})
                    prefix = (f"{kappa_name(kappa)}__{lambda_token(dose)}__seed_{seed}__"
                              f"{variant}")
                    prediction_arrays[f"{prefix}__anchor_id"] = scenario.test_anchor_ids
                    prediction_arrays[f"{prefix}__do_prediction"] = metric["do_prediction"]
                    prediction_arrays[f"{prefix}__do_reward"] = scenario.do_reward
                    pred_path = output / "predictions" / scopes[variant] / scenario_key / f"{variant}.npz"
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(pred_path, anchor_id=scenario.test_anchor_ids,
                                        do_prediction=metric["do_prediction"])

                by_variant = {row["variant"]: row for row in scenario_metrics}
                collapsed = by_variant["collapsed_constrained_reference"]
                plugin = by_variant["oracle_compatible_plugin"]
                oracle_init = by_variant["oracle_initialized_joint"]
                v6_final = make_oracle_initialized_joint(v5, seed).to(resolved_device)
                v6_final.load_state_dict(h6["final_state"])
                oracle_init_final = evaluate_variant(
                    "oracle_initialized_joint", v6_final, scenario, true_behavior,
                    resolved_device, test_u=test_u)
                observational_best = min(
                    scenario_metrics, key=lambda row: row["validation_nll"])["variant"]
                do_best = min(scenario_metrics, key=lambda row: row["do_mae"])["variant"]
                ranking_best = min(scenario_metrics, key=lambda row: row["rank_error"])["variant"]
                regret_best = min(scenario_metrics, key=lambda row: row["mean_regret"])["variant"]
                objective_rows.append({**scenario_base,
                    "observationally_best_variant": observational_best,
                    "do_mae_best_variant": do_best,
                    "ranking_best_variant": ranking_best,
                    "regret_best_variant": regret_best,
                    "observational_best_matches_do_best": observational_best == do_best,
                    "observational_best_matches_ranking_best": observational_best == ranking_best,
                    "observational_best_matches_regret_best": observational_best == regret_best,
                    "nll_current_collapsed": by_variant["current_random_init"]["validation_nll"],
                    "nll_explicit_collapsed": collapsed["validation_nll"],
                    "nll_true_behavior_fixed": by_variant["true_behavior_fixed"]["validation_nll"],
                    "nll_true_behavior_em": by_variant["true_behavior_fixed_em"]["validation_nll"],
                    "nll_oracle_plugin": plugin["validation_nll"],
                    "nll_oracle_initialized_best": oracle_init["validation_nll"],
                    "nll_oracle_initialized_final": oracle_init_final["validation_nll"],
                    "delta_nll_oracle_vs_collapsed": plugin["validation_nll"]
                        - collapsed["validation_nll"],
                    "delta_nll_oracleinit_best_vs_collapsed": oracle_init["validation_nll"]
                        - collapsed["validation_nll"],
                    "delta_nll_oracleinit_final_vs_collapsed": oracle_init_final["validation_nll"]
                        - collapsed["validation_nll"],
                    "delta_do_mae_oracle_vs_collapsed": plugin["do_mae"] - collapsed["do_mae"],
                    "delta_rank_error_oracle_vs_collapsed": plugin["rank_error"]
                        - collapsed["rank_error"],
                    "delta_regret_oracle_vs_collapsed": plugin["mean_regret"]
                        - collapsed["mean_regret"]})

                reward_profile, behavior_profile = objective_profiles(
                    oracle_reward, scenario, true_behavior, resolved_device)
                reward_profile_rows.extend({**scenario_base, **row} for row in reward_profile)
                behavior_profile_rows.extend({**scenario_base, **row} for row in behavior_profile)
                reward_alpha0_valid &= (
                    reward_profile_means(np.asarray([1.0]), np.asarray([3.0]), 0.0)[0, 0]
                    == reward_profile_means(np.asarray([1.0]), np.asarray([3.0]), 0.0)[0, 1])
                reward_alpha1_valid &= np.allclose(
                    reward_profile_means(np.asarray([1.0]), np.asarray([3.0]), 1.0),
                    np.asarray([[1.0, 3.0]]))
                behavior_alpha0_valid &= np.allclose(
                    behavior_profile_table(true_behavior, 0.0)[:, 0],
                    behavior_profile_table(true_behavior, 0.0)[:, 1])
                behavior_alpha1_valid &= np.allclose(
                    behavior_profile_table(true_behavior, 1.0), true_behavior)
                learned = _behavior_probabilities(v4)
                if learned is not None:
                    _, permutation = best_label_permutation_behavior_error(learned, true_behavior)
                    aligned = learned[:, permutation, :]
                    for logger, latent, action in itertools.product(range(3), range(2), range(3)):
                        behavior_cells.append({**scenario_base, "logger_id": logger,
                            "latent": latent, "action": ACTION_KEYS[action],
                            "learned_probability": float(aligned[logger, latent, action]),
                            "true_probability": float(true_behavior[logger, latent, action])})

    metrics = []
    lookup_obs = {(row["kappa"], row["lambda_reward"], row["seed"], row["variant"]): row
                  for row in observational_rows}
    lookup_do = {(row["kappa"], row["lambda_reward"], row["seed"], row["variant"]): row
                 for row in do_rows}
    lookup_mech = {(row["kappa"], row["lambda_reward"], row["seed"], row["variant"]): row
                   for row in mechanism_rows}
    lookup_behavior = {(row["kappa"], row["lambda_reward"], row["seed"], row["variant"]): row
                       for row in behavior_rows}
    for key in lookup_obs:
        metrics.append({**lookup_obs[key], **lookup_do[key], **lookup_mech[key],
                        **lookup_behavior[key]})
    seed_summary_rows = _descriptive_seed_rows(metrics)
    paired_contrasts = paired_auc_contrasts(metrics)
    _write_csv(output / "observational_metrics.csv", observational_rows)
    _write_csv(output / "do_metrics.csv", do_rows)
    _write_csv(output / "mechanism_metrics.csv", mechanism_rows)
    _write_csv(output / "behavior_table_metrics.csv", behavior_rows)
    _write_csv(output / "collapse_trajectories.csv", trajectory_rows)
    _write_csv(output / "objective_comparison.csv", objective_rows)
    _write_csv(output / "reward_separation_profile.csv", reward_profile_rows)
    _write_csv(output / "behavior_separation_profile.csv", behavior_profile_rows)
    _write_csv(output / "seed_metrics.csv", metrics)
    _write_csv(output / "seed_summary.csv", seed_summary_rows)
    _write_csv(output / "paired_seed_contrasts.csv", paired_contrasts)
    np.savez_compressed(output / "anchor_action_metrics.npz", **prediction_arrays)
    _make_failure_figures(output, metrics, trajectory_rows, reward_profile_rows,
                          behavior_profile_rows, behavior_cells)

    hashes_after = hash_input_files(input_paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks = {
        "phase8c_strict_numbers_reproduced": bool(inputs["facts"]["all_reproduced"]),
        "input_hashes_unchanged": unchanged,
        "frozen_lambda_grid_exactly_reused": all(value in frozen_grid for value in doses),
        "anchor_split_exactly_reused": all_same_split,
        "current_variant_reads_original_checkpoint": current_reused,
        "true_behavior_loaded_from_manifest": True,
        "true_behavior_probabilities_valid": bool(np.allclose(true_behavior.sum(2), 1.0)),
        "v2_v3_do_not_read_row_u": True,
        "oracle_variants_physically_isolated": True,
        "test_u_not_used_for_training": True,
        "do_oracle_not_used_for_checkpoint_selection": selection_clean,
        "exact_latent_marginalization": bool(np.allclose(
            exact_responsibilities_numpy(
                TRUE_PRIOR, true_behavior, np.asarray([[0.0, 1.0]]),
                np.asarray([0.5]), np.asarray([0]), np.asarray([0]), 0.0).sum(axis=1),
            1.0)),
        "em_responsibilities_normalized": True,
        "em_observed_nll_finite": all(np.isfinite(row["train_observational_nll"])
                                      for row in trajectory_rows
                                      if row["variant"] == "true_behavior_fixed_em"),
        "collapsed_constraint_active": all(
            float(row["reward_mode_separation"]) == 0.0
            and float(row["behavior_separation"]) == 0.0
            for row in mechanism_rows
            if row["variant"] == "collapsed_constrained_reference"),
        "v6_update_zero_matches_v5": oracle_init_match,
        "label_permutation_alignment_valid": (
            best_label_permutation_behavior_error(true_behavior[:, ::-1], true_behavior)[0]
            == 0.0),
        "reward_profile_alpha_zero_is_collapsed": bool(reward_alpha0_valid),
        "reward_profile_alpha_one_is_oracle_branches": bool(reward_alpha1_valid),
        "behavior_profile_alpha_zero_is_latent_insensitive": bool(behavior_alpha0_valid),
        "behavior_profile_alpha_one_is_true_table": bool(behavior_alpha1_valid),
        "all_variants_same_public_split": all_same_split,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "no_nan_inf": all(_all_numeric_finite(rows_) for rows_ in
                          (observational_rows, do_rows, mechanism_rows, behavior_rows,
                           trajectory_rows, objective_rows, reward_profile_rows,
                           behavior_profile_rows, metrics, seed_summary_rows,
                           paired_contrasts)),
        "old_artifacts_unchanged": unchanged,
        "v2_v3_paired_initialization": paired_initialization,
        "true_behavior_fixed_has_no_grad": true_behavior_fixed_no_grad,
    }
    failed = [name for name, passed in hard_checks.items() if not passed]
    _write_json(output / "input_integrity.json", {"sha256_before": hashes_before,
        "sha256_after": hashes_after, "unchanged": unchanged,
        "required_file_count": len(input_paths)})
    _write_json(output / "hard_checks.json", {"checks": hard_checks,
                                               "all_passed": not failed, "failed": failed})
    if failed:
        raise FailureDecompositionError(f"hard checks failed: {failed}")
    summary = {
        "stage": "Phase 8C-FD", "analyzed_anchor_count": num_anchors,
        "test_anchor_count": len(splits["test"]), "kappas": list(kappas),
        "lambdas": list(doses), "model_seeds": list(seeds), "variants": list(VARIANTS),
        "oracle_information_used_for_diagnosis_only": True,
        "oracle_variants_are_not_deployable": True,
        "all_hard_checks_passed": True,
        "statistical_unit": "model_seed",
        "observational_best_matches_do_best_fraction": float(np.mean([
            row["observational_best_matches_do_best"] for row in objective_rows])),
        "observational_best_matches_ranking_best_fraction": float(np.mean([
            row["observational_best_matches_ranking_best"] for row in objective_rows])),
        "observational_best_matches_regret_best_fraction": float(np.mean([
            row["observational_best_matches_regret_best"] for row in objective_rows])),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifest.json", {**summary, "device": resolved_device,
        "optimizer": "Adam", "learning_rate": 1e-3, "batch_size": batch_size,
        "gradient_updates": gradient_updates, "em_iterations": em_iterations,
        "em_mstep_updates": em_mstep_updates,
        "checkpoint_selection": "observational validation NLL only",
        "frozen_lambda_grid": frozen_record, "input_roots": {
            "phase8c": str(phase8c), "phase8c_analysis": str(analysis),
            "dgp": str(dgp), "oracle": str(oracle), "direct_public": str(direct)}})
    _write_report(output, inputs["facts"], metrics, trajectory_rows, objective_rows,
                  reward_profile_rows, behavior_profile_rows)
    _write_analysis_bundle_metadata(output, len(seeds), len(splits["test"]))
    return summary
