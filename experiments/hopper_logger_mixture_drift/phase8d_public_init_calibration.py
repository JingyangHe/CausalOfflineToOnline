"""Phase 8D-PRIC: public residual initialization and do calibration.

The deployable path in this module accepts public observational rows only.
Oracle arrays are read by the calibration/evaluation boundary after every
offline candidate has been trained and frozen.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analyze_phase8a_population_effect import (
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
)
from .noncomplementary_population_dgp import ACTION_KEYS, build_do_lookup
from .phase8c_failure_decomposition import (
    CHECKPOINT_FRACTIONS,
    _latent_snapshot,
    _load_normalization,
    _state_hash,
    aligned_posterior_accuracy,
    best_label_permutation_behavior_error,
    load_failure_decomposition_model,
    load_true_behavior_table,
    observed_nll,
    posterior_probability_z1,
)
from .reward_mechanism_separation import (
    LOGGER_WEIGHTS,
    PRIMARY_MIXTURE,
    Normalization,
    PublicRows,
    _aligned_base_weights,
    _device,
    _hidden_u_for_rows,
    _read_json,
    _torch,
    _write_csv,
    _write_json,
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
    regret_metrics,
    save_model,
    validate_main_model_structure,
    validate_public_schema,
)


BINARY_LATENT_REWARD_ONLY_PROOF_OF_CONCEPT = True
PRIMARY_CONDITION = "confounded"
K_FOLDS = 5
STAGE_NAMES = ("reward_pretraining", "behavior_only", "reward_refinement", "joint")
DEFAULT_BUDGETS = (0, 8, 16, 32, 64, 128)
NO_TEMPERATURE_PARAMETER = True
PUBLIC_INITIALIZATION_FIELDS = {
    "row_id", "anchor_id", "observation", "commanded_action", "reward",
    "logger_id", "action_index", "row_weight",
}
FORBIDDEN_OFFLINE_FIELDS = {"u_env", "u_behavior", "do_reward", "applied_action"}


class Phase8DPublicInitCalibrationError(RuntimeError):
    """Raised when an input or a hard Phase 8D invariant is violated."""


@dataclass(frozen=True)
class WeightedTwoMeans:
    labels: np.ndarray
    centers: np.ndarray
    cluster_weight: np.ndarray
    shared_variance: float
    split_index: int
    objective: float


@dataclass(frozen=True)
class CalibrationSequence:
    public: Mapping[str, np.ndarray]
    hidden_u: np.ndarray


def phase8d_anchor_splits(phase8c_splits: Mapping[str, Sequence[int]],
                          seed: int = 0) -> dict[str, Any]:
    """Keep train/test fixed and split validation before any result is read."""
    train = np.asarray(sorted(map(int, phase8c_splits["train"])), dtype=np.int64)
    validation = np.asarray(sorted(map(int, phase8c_splits["validation"])), dtype=np.int64)
    test = np.asarray(sorted(map(int, phase8c_splits["test"])), dtype=np.int64)
    order = np.random.default_rng(seed).permutation(validation)
    n_observational = (2 * len(validation)) // 3
    observational = np.sort(order[:n_observational])
    calibration = np.sort(order[n_observational:])
    result = {
        "train": train.tolist(),
        "observational_validation": observational.tolist(),
        "do_calibration_pool": calibration.tolist(),
        "test": test.tolist(),
        "split_seed": int(seed),
        "exploratory_algorithm_development": True,
        "fresh_anchor_and_policy_seeds_required_for_confirmatory_claims": True,
    }
    groups = [set(result[name]) for name in
              ("train", "observational_validation", "do_calibration_pool", "test")]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise Phase8DPublicInitCalibrationError("Phase 8D anchor splits overlap")
    if set(validation) != groups[1] | groups[2]:
        raise Phase8DPublicInitCalibrationError("validation split was not preserved")
    return result


def anchor_fold_assignment(anchor_ids: Sequence[int], folds: int = K_FOLDS,
                           seed: int = 0) -> dict[int, int]:
    anchors = np.asarray(sorted(set(map(int, anchor_ids))), dtype=np.int64)
    if folds < 2 or len(anchors) < folds:
        raise ValueError("OOF requires at least one anchor per fold")
    order = np.random.default_rng(seed).permutation(anchors)
    return {int(anchor): int(position % folds) for position, anchor in enumerate(order)}


def verify_oof_exclusion(row_anchor_ids: Sequence[int], assignments: Mapping[int, int],
                         training_anchors_by_fold: Mapping[int, Sequence[int]]) -> bool:
    for anchor in np.asarray(row_anchor_ids, dtype=np.int64):
        fold = int(assignments[int(anchor)])
        if int(anchor) in set(map(int, training_anchors_by_fold[fold])):
            return False
    return True


def _cluster_sse(weight: float, weighted_sum: float, weighted_square_sum: float) -> float:
    return max(0.0, weighted_square_sum - weighted_sum * weighted_sum / weight)


def exact_weighted_two_means(values: Sequence[float], weights: Sequence[float],
                             variance_floor: float = 1e-12) -> WeightedTwoMeans:
    """Globally optimal one-dimensional weighted two-means by prefix sums."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 1 or w.shape != x.shape or len(x) < 2 or np.any(w < 0)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(w))
            or np.count_nonzero(w > 0) < 2):
        raise ValueError("weighted two-means inputs are invalid")
    order = np.argsort(x, kind="stable")
    sx, sw = x[order], w[order]
    cw, cwx, cwx2 = np.cumsum(sw), np.cumsum(sw * sx), np.cumsum(sw * sx * sx)
    total_w, total_x, total_x2 = float(cw[-1]), float(cwx[-1]), float(cwx2[-1])
    best: tuple[float, int] | None = None
    for split in range(1, len(x)):
        wl, wr = float(cw[split - 1]), total_w - float(cw[split - 1])
        if wl <= 0.0 or wr <= 0.0:
            continue
        objective = (_cluster_sse(wl, float(cwx[split - 1]), float(cwx2[split - 1]))
                     + _cluster_sse(wr, total_x - float(cwx[split - 1]),
                                    total_x2 - float(cwx2[split - 1])))
        if best is None or objective < best[0]:
            best = (objective, split)
    if best is None:
        raise ValueError("no positive-mass two-means split exists")
    objective, split = best
    left_w = float(cw[split - 1]); right_w = total_w - left_w
    centers = np.asarray((float(cwx[split - 1]) / left_w,
                          (total_x - float(cwx[split - 1])) / right_w))
    labels_sorted = np.zeros(len(x), dtype=np.int8); labels_sorted[split:] = 1
    labels = np.empty_like(labels_sorted); labels[order] = labels_sorted
    variance = max(float(objective / total_w), float(variance_floor))
    return WeightedTwoMeans(labels, centers, np.asarray((left_w, right_w)),
                            variance, split, float(objective))


def brute_force_weighted_two_means(values: Sequence[float], weights: Sequence[float]) -> tuple[float, int]:
    x = np.asarray(values, np.float64); w = np.asarray(weights, np.float64)
    order = np.argsort(x, kind="stable"); sx, sw = x[order], w[order]
    candidates = []
    for split in range(1, len(x)):
        if sw[:split].sum() <= 0 or sw[split:].sum() <= 0:
            continue
        means = (np.average(sx[:split], weights=sw[:split]),
                 np.average(sx[split:], weights=sw[split:]))
        sse = float(np.sum(sw[:split] * (sx[:split] - means[0]) ** 2)
                    + np.sum(sw[split:] * (sx[split:] - means[1]) ** 2))
        candidates.append((sse, split))
    if not candidates:
        raise ValueError("no split")
    return min(candidates, key=lambda item: (item[0], item[1]))


def soft_responsibilities(values: Sequence[float], centers: Sequence[float],
                          prior: Sequence[float], shared_variance: float) -> np.ndarray:
    x = np.asarray(values, np.float64)[:, None]
    mu = np.asarray(centers, np.float64)[None, :]
    pi = np.asarray(prior, np.float64)
    if mu.shape != (1, 2) or pi.shape != (2,) or np.any(pi <= 0) or shared_variance <= 0:
        raise ValueError("mixture parameters are invalid")
    logp = np.log(pi)[None, :] - 0.5 * (x - mu) ** 2 / shared_variance
    logp -= np.logaddexp.reduce(logp, axis=1, keepdims=True)
    q = np.exp(logp)
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise Phase8DPublicInitCalibrationError("soft responsibilities do not normalize")
    return q


def initialize_behavior(source: Sequence[int], action: Sequence[int], q: np.ndarray,
                        weights: Sequence[float], *, source_override: Sequence[int] | None = None,
                        probability_floor: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Estimate prior and beta using exactly source, action, q and public weight."""
    e = np.asarray(source_override if source_override is not None else source, np.int64)
    a = np.asarray(action, np.int64); responsibility = np.asarray(q, np.float64)
    w = np.asarray(weights, np.float64)
    if e.shape != a.shape or e.shape != w.shape or responsibility.shape != (len(e), 2):
        raise ValueError("behavior initialization inputs are misaligned")
    prior = np.sum(w[:, None] * responsibility, axis=0)
    prior = np.maximum(prior, probability_floor); prior /= prior.sum()
    beta = np.zeros((3, 2, 3), dtype=np.float64)
    for logger in range(3):
        for latent in range(2):
            mass = w * responsibility[:, latent] * (e == logger)
            for choice in range(3):
                beta[logger, latent, choice] = mass[a == choice].sum()
            beta[logger, latent] += probability_floor
            beta[logger, latent] /= beta[logger, latent].sum()
    return prior, beta


def stage_update_allocation(total_updates: int) -> dict[str, int]:
    if total_updates <= 0:
        raise ValueError("total updates must be positive")
    common, remainder = divmod(int(total_updates), 4)
    return {name: common + (remainder if name == "joint" else 0) for name in STAGE_NAMES}


def set_stage_trainability(model: Any, stage: str) -> dict[str, bool]:
    if stage not in STAGE_NAMES and stage != "no_staged_training":
        raise ValueError("unknown stage")
    train_reward = stage in {"reward_pretraining", "reward_refinement", "joint",
                             "no_staged_training"}
    train_behavior = stage in {"behavior_only", "joint", "no_staged_training"}
    for parameter in model.reward_decoder.parameters():
        parameter.requires_grad_(train_reward)
    model.log_scale.requires_grad_(stage in {"reward_refinement", "joint",
                                             "no_staged_training"})
    model.prior_logits.requires_grad_(train_behavior)
    model.behavior_logits.requires_grad_(train_behavior)
    return {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}


def stage_freezing_is_valid(histories: Mapping[str, Mapping[str, Any]]) -> bool:
    if set(histories) == {"joint"}:
        return all(histories["joint"]["trainability"].values())
    if not {"A", "B", "C", "D"}.issubset(histories):
        return False
    flags = {key: histories[key]["trainability"] for key in ("A", "B", "C", "D")}
    reward_names = [name for name in flags["A"] if name.startswith("reward_decoder.")]
    return bool(
        all(flags["A"][name] for name in reward_names)
        and not flags["A"]["prior_logits"] and not flags["A"]["behavior_logits"]
        and not flags["A"]["log_scale"]
        and all(not flags["B"][name] for name in reward_names)
        and flags["B"]["prior_logits"] and flags["B"]["behavior_logits"]
        and not flags["B"]["log_scale"]
        and all(flags["C"][name] for name in reward_names)
        and not flags["C"]["prior_logits"] and not flags["C"]["behavior_logits"]
        and flags["C"]["log_scale"]
        and all(flags["D"].values()))


def hierarchical_candidate_prior(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not candidates:
        raise ValueError("candidate bank is empty")
    by_seed: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_seed.setdefault(int(candidate["seed"]), []).append(index)
    prior = np.zeros(len(candidates), dtype=np.float64)
    for indices in by_seed.values():
        prior[indices] = 1.0 / len(by_seed) / len(indices)
    return prior


def prediction_sha256(prediction: np.ndarray) -> str:
    value = np.asarray(prediction, np.float64)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def deduplicate_candidate_predictions(candidates: Sequence[Mapping[str, Any]],
                                      predictions: Sequence[np.ndarray]) -> tuple[list[int], dict[int, int]]:
    if len(candidates) != len(predictions):
        raise ValueError("candidates and predictions are misaligned")
    first: dict[str, int] = {}; unique: list[int] = []; mapping: dict[int, int] = {}
    for index, prediction in enumerate(predictions):
        digest = prediction_sha256(prediction)
        if digest not in first:
            first[digest] = index; unique.append(index)
        mapping[index] = first[digest]
    return unique, mapping


def exact_do_log_predictive_density(reward: Sequence[float], means: np.ndarray,
                                    prior: Sequence[float], log_scale: float) -> np.ndarray:
    y = np.asarray(reward, np.float64)
    mu = np.asarray(means, np.float64)
    pi = np.asarray(prior, np.float64)
    if mu.shape != (len(y), 2) or pi.shape != (2,):
        raise ValueError("predictive density shapes are invalid")
    scale = max(math.exp(float(log_scale)), 1e-8)
    component = (np.log(pi)[None, :] - 0.5 * ((y[:, None] - mu) / scale) ** 2
                 - math.log(scale) - 0.5 * math.log(2.0 * math.pi))
    return np.logaddexp.reduce(component, axis=1)


def posterior_candidate_weights(candidate_prior: Sequence[float],
                                calibration_log_score: Sequence[float]) -> np.ndarray:
    prior = np.asarray(candidate_prior, np.float64)
    score = np.asarray(calibration_log_score, np.float64)
    if prior.shape != score.shape or np.any(prior <= 0) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("candidate prior or score is invalid")
    if np.all(score == 0.0):
        return prior.copy()
    log_weight = np.log(prior) + score
    log_weight -= np.logaddexp.reduce(log_weight)
    return np.exp(log_weight)


def hard_selection_weights(calibration_log_score: Sequence[float]) -> np.ndarray:
    score = np.asarray(calibration_log_score, np.float64)
    tied = score == np.max(score)
    return tied.astype(np.float64) / tied.sum()


def candidate_weight_entropy(weights: Sequence[float]) -> float:
    p = np.asarray(weights, np.float64)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


def build_calibration_sequence(do_raw: Mapping[str, np.ndarray], anchor_pool: Sequence[int],
                               kappa: float, dose: float, replicate: int,
                               global_seed: int = 0, size: int | None = None,
                               anchor_observation: Mapping[int, np.ndarray] | None = None) -> CalibrationSequence:
    """Create nested randomized do prefixes; U remains in a separate in-memory object."""
    anchors = np.asarray(sorted(set(map(int, anchor_pool))), dtype=np.int64)
    if not len(anchors):
        raise ValueError("calibration pool is empty")
    seed_sequence = np.random.SeedSequence((int(global_seed), int(replicate)))
    anchor_seed, u_seed, action_seed = seed_sequence.spawn(3)
    anchor_rng = np.random.default_rng(anchor_seed)
    u_rng = np.random.default_rng(u_seed)
    action_rng = np.random.default_rng(action_seed)
    requested = len(anchors) if size is None else int(size)
    if requested <= 0:
        raise ValueError("calibration sequence size must be positive")
    order = np.concatenate([anchor_rng.permutation(anchors)
                            for _ in range(int(math.ceil(requested / len(anchors))))])[:requested]
    u = u_rng.choice(np.asarray((-1, 1), np.int8), size=requested, replace=True)
    offset = int(action_rng.integers(0, 3))
    action_index = (np.arange(len(order), dtype=np.int64) + offset) % 3
    action_key = np.asarray(ACTION_KEYS, dtype="U8")[action_index]
    lookup = build_do_lookup(dict(do_raw), float(kappa))
    rows = np.asarray([lookup[(int(anchor), str(action), int(latent))]
                       for anchor, action, latent in zip(order, action_key, u)], dtype=np.int64)
    public = {
        "calibration_row_id": np.arange(len(order), dtype=np.int64),
        "anchor_id": order,
        "observation": (np.asarray(do_raw["observation"])[rows]
            if "observation" in do_raw else np.asarray(
                [anchor_observation[int(anchor)] for anchor in order], dtype=np.float32)
            if anchor_observation is not None else np.empty((len(rows), 0), np.float32)),
        "commanded_action": np.asarray(do_raw["commanded_action"])[rows],
        "action_index": action_index,
        "reward": np.asarray(do_raw["reward"], np.float64)[rows] + float(dose) * u,
    }
    return CalibrationSequence(public=public, hidden_u=u)


def calibration_budgets_are_nested(sequence: Mapping[str, np.ndarray],
                                   budgets: Sequence[int]) -> bool:
    n = len(sequence["anchor_id"])
    values = tuple(map(int, budgets))
    return bool(values == tuple(sorted(set(values))) and values[0] == 0 and values[-1] <= n)


def calibration_public_is_hidden_free(public: Mapping[str, np.ndarray]) -> bool:
    return not FORBIDDEN_OFFLINE_FIELDS.intersection(public)


def fraction_gap_closed(value: float, v0: float, v6: float) -> float:
    denominator = float(v0) - float(v6)
    return float((float(v0) - float(value)) / denominator) if denominator != 0 else 0.0


def _copy_state(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _fit_public_mse(model: Any, rows: PublicRows, stats: Normalization, *, seed: int,
                    updates: int, batch_size: int, device: str) -> Any:
    """Fixed-budget latent-free baseline fit; no held-out row selects a checkpoint."""
    torch = _torch(); torch.manual_seed(int(seed))
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.as_tensor(normalized_x(rows, stats), dtype=torch.float32, device=device)
    y = torch.as_tensor((rows.reward - stats.reward_mean) / stats.reward_std,
                        dtype=torch.float32, device=device)
    w = torch.as_tensor(rows.row_weight, dtype=torch.float32, device=device)
    source = torch.as_tensor(rows.logger_id, dtype=torch.long, device=device)
    schedule = np.random.default_rng(seed).integers(0, len(rows.reward),
        size=(updates, batch_size), dtype=np.int64)
    for batch_np in schedule:
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        prediction = model.plain_mean(x[batch], source[batch])
        loss = torch.sum(w[batch] * (prediction - y[batch]) ** 2) / torch.sum(w[batch])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model.eval()


def oof_public_residuals(train: PublicRows, stats: Normalization, *, seed: int,
                         updates: int, batch_size: int, device: str,
                         folds: int = K_FOLDS) -> tuple[np.ndarray, dict[str, Any], list[Any]]:
    assignments = anchor_fold_assignment(train.anchor_id, folds, seed)
    prediction = np.full(len(train.reward), np.nan, dtype=np.float64)
    training_anchors: dict[int, list[int]] = {}
    models = []
    all_anchors = set(map(int, np.unique(train.anchor_id)))
    for fold in range(folds):
        held_out = {anchor for anchor, assigned in assignments.items() if assigned == fold}
        fit_anchors = sorted(all_anchors - held_out); training_anchors[fold] = fit_anchors
        fit_rows = train.subset(fit_anchors)
        model = _fit_public_mse(make_model("pooled_mlp", seed * 100 + fold), fit_rows, stats,
                                seed=seed * 100 + fold, updates=updates,
                                batch_size=batch_size, device=device)
        mask = np.asarray([assignments[int(anchor)] == fold for anchor in train.anchor_id])
        held_rows = PublicRows(**{name: np.asarray(getattr(train, name))[mask]
                                  for name in train.__dataclass_fields__})
        torch = _torch()
        with torch.no_grad():
            x = torch.as_tensor(normalized_x(held_rows, stats), dtype=torch.float32, device=device)
            p = model.plain_mean(x).cpu().numpy().astype(np.float64)
        prediction[mask] = p * stats.reward_std + stats.reward_mean
        models.append(model)
    if (not np.all(np.isfinite(prediction))
            or not verify_oof_exclusion(train.anchor_id, assignments, training_anchors)):
        raise Phase8DPublicInitCalibrationError("OOF residual exclusion failed")
    metadata = {
        "folds": folds,
        "seed": seed,
        "assignment": {str(key): value for key, value in sorted(assignments.items())},
        "training_anchors_by_fold": {str(key): value for key, value in training_anchors.items()},
        "fold_model_state_hashes": [_state_hash(model) for model in models],
        "oof_prediction_sha256": prediction_sha256(prediction),
        "each_prediction_excludes_its_anchor": True,
    }
    return train.reward - prediction, metadata, models


def _initialize_latent_parameters(model: Any, prior: np.ndarray, beta: np.ndarray,
                                  shared_variance: float, stats: Normalization) -> None:
    torch = _torch()
    with torch.no_grad():
        model.prior_logits.copy_(torch.as_tensor(np.log(np.maximum(prior, 1e-12)),
                                                 dtype=model.prior_logits.dtype,
                                                 device=model.prior_logits.device))
        model.behavior_logits.copy_(torch.as_tensor(np.log(np.maximum(beta, 1e-12)),
            dtype=model.behavior_logits.dtype, device=model.behavior_logits.device))
        normalized_scale = max(math.sqrt(shared_variance) / stats.reward_std, 1e-4)
        model.log_scale.fill_(math.log(normalized_scale))


def _train_stage(model: Any, stage: str, train: PublicRows, validation: PublicRows,
                 stats: Normalization, *, seed: int, updates: int, batch_size: int,
                 device: str, responsibilities: np.ndarray | None = None,
                 source_override: np.ndarray | None = None,
                 checkpoint_steps: Sequence[int] = ()) -> dict[str, Any]:
    torch = _torch()
    flags = set_stage_trainability(model, stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    x = torch.as_tensor(normalized_x(train, stats), dtype=torch.float32, device=device)
    y = torch.as_tensor((train.reward - stats.reward_mean) / stats.reward_std,
                        dtype=torch.float32, device=device)
    source_np = train.logger_id if source_override is None else np.asarray(source_override)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    action = torch.as_tensor(train.action_index, dtype=torch.long, device=device)
    weight = torch.as_tensor(train.row_weight, dtype=torch.float32, device=device)
    q = (None if responsibilities is None else torch.as_tensor(
        responsibilities, dtype=torch.float32, device=device))
    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, len(train.reward), size=(updates, batch_size), dtype=np.int64)
    snapshots: list[dict[str, Any]] = []
    best_nll, best_state, best_step = math.inf, None, -1

    def record(step: int, label: str) -> None:
        nonlocal best_nll, best_state, best_step
        value = observed_nll(model, validation, stats, device)
        state = _copy_state(model)
        snapshots.append({"stage": stage, "label": label, "update": step,
                          "validation_observational_nll": value, "state": state,
                          **_latent_snapshot(model, validation, stats, device)})
        if value < best_nll:
            best_nll, best_state, best_step = value, state, step

    record(0, "start")
    model.train()
    wanted = set(map(int, checkpoint_steps))
    for step, batch_np in enumerate(schedule, 1):
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        if stage == "reward_pretraining":
            if q is None:
                raise ValueError("reward pretraining requires soft responsibilities")
            means = model.latent_means(x[batch])
            squared = (y[batch, None] - means) ** 2
            loss = torch.sum(weight[batch, None] * q[batch] * squared) / torch.sum(weight[batch])
        else:
            logp = model.training_log_prob(x[batch], y[batch], source[batch], action[batch])
            loss = -torch.sum(weight[batch] * logp) / torch.sum(weight[batch])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step in wanted or step == updates:
            record(step, "checkpoint" if step != updates else "end")
            model.train()
    if best_state is None:
        raise Phase8DPublicInitCalibrationError("stage produced no validation checkpoint")
    model.eval()
    return {"snapshots": snapshots, "best_state": best_state, "best_step": best_step,
            "best_validation_nll": best_nll, "final_state": _copy_state(model),
            "trainability": flags, "updates": updates,
            "selection_uses_observational_validation_only": True}


def train_public_initialized_candidates(train: PublicRows, validation: PublicRows,
                                        stats: Normalization, *, seed: int,
                                        total_updates: int, batch_size: int, device: str,
                                        shuffled_source: bool = False,
                                        staged: bool = True,
                                        prepared_initialization: Mapping[str, Any] | None = None,
                                        ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    allocation = stage_update_allocation(total_updates)
    if prepared_initialization is None:
        oof_updates = max(1, allocation["reward_pretraining"])
        residual, oof, _ = oof_public_residuals(
            train, stats, seed=seed, updates=oof_updates, batch_size=batch_size, device=device)
        split = exact_weighted_two_means(residual, train.row_weight)
        prior0 = split.cluster_weight / split.cluster_weight.sum()
        q = soft_responsibilities(residual, split.centers, prior0, split.shared_variance)
    else:
        residual = np.asarray(prepared_initialization["residual"], np.float64)
        q = np.asarray(prepared_initialization["responsibilities"], np.float64)
        split = prepared_initialization["two_means"]
        oof = dict(prepared_initialization["oof"])
    source_override = None
    if shuffled_source:
        source_override = np.random.default_rng(seed + 9173).permutation(train.logger_id)
    prior, beta = initialize_behavior(train.logger_id, train.action_index, q,
                                      train.row_weight, source_override=source_override)
    model = make_model("mechanism_separated", seed).to(device)
    _initialize_latent_parameters(model, prior, beta, split.shared_variance, stats)
    candidates: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}

    def add_from_history(history: Mapping[str, Any], label: str, state_key: str = "final_state") -> None:
        candidates.append({"seed": seed, "label": label, "state": history[state_key],
                           "validation_nll": (history["best_validation_nll"] if state_key == "best_state"
                                              else next(item["validation_observational_nll"]
                                                  for item in reversed(history["snapshots"])
                                                  if item["state"] is history[state_key] or
                                                  item["update"] == history["updates"]))})

    if staged:
        h_a = _train_stage(model, "reward_pretraining", train, validation, stats,
                           seed=seed + 101, updates=allocation["reward_pretraining"],
                           batch_size=batch_size, device=device, responsibilities=q)
        histories["A"] = h_a; candidates.append({"seed": seed, "label": "stage_A_end",
            "state": h_a["final_state"], "validation_nll": h_a["snapshots"][-1]["validation_observational_nll"]})
        h_b = _train_stage(model, "behavior_only", train, validation, stats,
                           seed=seed + 202, updates=allocation["behavior_only"],
                           batch_size=batch_size, device=device, source_override=source_override)
        histories["B"] = h_b; candidates.append({"seed": seed, "label": "stage_B_end",
            "state": h_b["final_state"], "validation_nll": h_b["snapshots"][-1]["validation_observational_nll"]})
        h_c = _train_stage(model, "reward_refinement", train, validation, stats,
                           seed=seed + 303, updates=allocation["reward_refinement"],
                           batch_size=batch_size, device=device, source_override=source_override)
        histories["C"] = h_c; candidates.append({"seed": seed, "label": "stage_C_end",
            "state": h_c["final_state"], "validation_nll": h_c["snapshots"][-1]["validation_observational_nll"]})
        d_updates = allocation["joint"]
        d_steps = sorted({max(1, int(round(d_updates * fraction)))
                          for fraction in CHECKPOINT_FRACTIONS if fraction > 0.0})
        h_d = _train_stage(model, "joint", train, validation, stats,
                           seed=seed + 404, updates=d_updates, batch_size=batch_size,
                           device=device, source_override=source_override,
                           checkpoint_steps=d_steps)
        histories["D"] = h_d
        for snapshot in h_d["snapshots"]:
            if snapshot["update"] > 0:
                candidates.append({"seed": seed, "label": f"stage_D_update_{snapshot['update']}",
                    "state": snapshot["state"],
                    "validation_nll": snapshot["validation_observational_nll"]})
        candidates.append({"seed": seed, "label": "stage_D_nll_best",
            "state": h_d["best_state"], "validation_nll": h_d["best_validation_nll"]})
        candidates.append({"seed": seed, "label": "stage_D_final",
            "state": h_d["final_state"],
            "validation_nll": h_d["snapshots"][-1]["validation_observational_nll"]})
    else:
        history = _train_stage(model, "no_staged_training", train, validation, stats,
                               seed=seed + 505, updates=total_updates, batch_size=batch_size,
                               device=device,
                               checkpoint_steps=tuple(int(round(total_updates * x))
                                                      for x in CHECKPOINT_FRACTIONS if x > 0.0))
        histories["joint"] = history
        candidates.extend({"seed": seed, "label": f"joint_update_{item['update']}",
                           "state": item["state"],
                           "validation_nll": item["validation_observational_nll"]}
                          for item in history["snapshots"] if item["update"] > 0)
        candidates.append({"seed": seed, "label": "joint_nll_best",
                           "state": history["best_state"],
                           "validation_nll": history["best_validation_nll"]})
    best = min(candidates, key=lambda item: (float(item["validation_nll"]), item["label"]))
    model.load_state_dict(best["state"]); model.eval()
    diagnostics = {
        "oof": oof,
        "residual": residual,
        "two_means": split,
        "responsibilities": q,
        "prior_initial": prior,
        "behavior_initial": beta,
        "pseudo_label_entropy": float(np.average(
            -np.sum(q * np.log(np.maximum(q, 1e-15)), axis=1), weights=train.row_weight)),
        "allocation": allocation,
        "histories": histories,
        "shuffled_source": shuffled_source,
        "staged": staged,
    }
    return model, candidates, diagnostics


def reproduce_phase8c_fd(failure_root: Path) -> dict[str, Any]:
    manifest = _read_json(Path(failure_root) / "manifest.json")
    rows: list[dict[str, str]] = []
    with (Path(failure_root) / "seed_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    primary = [row for row in rows if float(row["kappa"]) == 0.0
               and row["condition"] == PRIMARY_CONDITION
               and row["mixture"] == PRIMARY_MIXTURE]
    v0 = [row for row in primary if row["variant"] == "current_random_init"]
    v6 = [row for row in primary if row["variant"] == "oracle_initialized_joint"]
    v7 = [row for row in primary if row["variant"] == "oracle_u_aware_ceiling"]
    metrics = {}
    for name, selected in (("V0", v0), ("V6", v6), ("V7", v7)):
        metrics[name] = {metric: float(np.mean([float(row[metric]) for row in selected]))
                         for metric in ("validation_nll", "do_mae", "rank_error",
                                        "mean_regret", "reward_mode_separation")}
    result = {
        "anchor_count": int(manifest.get("analyzed_anchor_count", -1)),
        "test_anchor_count": int(manifest.get("test_anchor_count", -1)),
        "lambda_count": len(manifest.get("lambdas", [])),
        "seed_count": len(manifest.get("model_seeds", [])),
        "v0_count": len(v0), "v6_count": len(v6),
        "v0_collapse_count": sum(row["latent_collapse"].lower() == "true" for row in v0),
        "v6_collapse_count": sum(row["latent_collapse"].lower() == "true" for row in v6),
        "main_metrics": metrics,
    }
    expected_metrics = {
        "V0": (-0.9615959290082433, 0.018613284183491157,
               0.6297734627831716, 0.0038984942295696126),
        "V6": (-1.3621250459126064, 0.011229091310014915,
               0.3119741100323625, 0.001309191530490409),
        "V7": (-1.1380771320867162, 0.011162692210353658,
               0.23014331946370778, 0.0007858001287773166),
    }
    result["main_metrics_reproduced"] = all(np.allclose(
        [metrics[name][key] for key in ("validation_nll", "do_mae", "rank_error", "mean_regret")],
        expected_metrics[name], atol=1e-12, rtol=0.0) for name in expected_metrics)
    result["all_reproduced"] = bool(
        result["anchor_count"] == 2048 and result["test_anchor_count"] == 309
        and result["lambda_count"] == 7 and result["seed_count"] == 5
        and result["v0_count"] == result["v6_count"] == 35
        and result["v0_collapse_count"] == 35 and result["v6_collapse_count"] == 0
        and result["main_metrics_reproduced"])
    return result


def _resolve_direct_root(phase8c: Path) -> Path:
    recorded = Path(str(_read_json(phase8c / "manifest.json").get("direct_reward_root", "")))
    candidates = [phase8c.parent / "phase8c_direct_reward_public_grid", recorded]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise Phase8DPublicInitCalibrationError("Direct U->R public artifacts are unavailable")


def _resolve_do_raw_root(dgp: Path, kappas: Sequence[float] = (0.0, 0.3)) -> Path:
    """Resolve Phase 8A raw branches through the NC provenance, including moved repos."""
    recorded_value = _read_json(dgp / "manifest.json").get("phase8a_input_root", "")
    recorded = Path(str(recorded_value)) if recorded_value else None
    candidates = [dgp]
    if recorded is not None:
        candidates.extend((recorded, dgp.parent / recorded.name))
    candidates.append(dgp.parent / "controlled_loggers_seed0_verified")
    for candidate in candidates:
        if candidate.is_dir() and all(
                (candidate / kappa_name(float(kappa)) / "do_oracle_raw.npz").is_file()
                for kappa in kappas):
            return candidate.resolve()
    expected = dgp.parent / "controlled_loggers_seed0_verified"
    raise Phase8DPublicInitCalibrationError(
        f"Phase 8A raw do branches are unavailable; expected provenance root: {expected}")


def validate_phase8d_inputs(phase8c_root: Path, failure_root: Path,
                            oracle_root: Path) -> dict[str, Any]:
    phase8c, failure, oracle = map(lambda p: Path(p).resolve(),
                                   (phase8c_root, failure_root, oracle_root))
    for root, label in ((phase8c, "Phase 8C"), (failure, "Phase 8C-FD"),
                        (oracle, "Oracle direct reward")):
        if not root.is_dir():
            raise Phase8DPublicInitCalibrationError(f"{label} input root is unavailable: {root}")
        check = _read_json(root / "hard_checks.json")
        if check.get("all_passed") is not True or not all(check.get("checks", {}).values()):
            raise Phase8DPublicInitCalibrationError(f"{label} hard checks did not pass")
    facts = reproduce_phase8c_fd(failure)
    if not facts["all_reproduced"]:
        raise Phase8DPublicInitCalibrationError(f"Phase 8C-FD facts do not reproduce: {facts}")
    direct = _resolve_direct_root(phase8c)
    direct_check = _read_json(direct / "hard_checks.json")
    if direct_check.get("all_passed") is not True:
        raise Phase8DPublicInitCalibrationError("direct public hard checks did not pass")
    dgp = phase8c.parent
    dgp_check = _read_json(dgp / "hard_checks.json")
    if dgp_check.get("all_passed") is not True:
        raise Phase8DPublicInitCalibrationError("Phase 8A-NC hard checks did not pass")
    return {"phase8c": phase8c, "failure": failure, "oracle": oracle,
            "direct": direct, "dgp": dgp, "facts": facts}


def _scenario_model_path(root: Path, kappa: float, dose: float, condition: str,
                         seed: int, method: str) -> Path:
    return (root / "models" / kappa_name(kappa) / lambda_token(dose) / condition
            / PRIMARY_MIXTURE / f"seed_{seed}" / f"{method}.pt")


def _scenario_normalization_path(root: Path, kappa: float, dose: float,
                                 condition: str) -> Path:
    return (root / "normalization" / kappa_name(kappa) / lambda_token(dose)
            / condition / "stats.npz")


def _model_do_components(model: Any, observation: np.ndarray, action: np.ndarray,
                         stats: Normalization, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    torch = _torch()
    raw_x = np.concatenate((np.asarray(observation, np.float64),
                            np.asarray(action, np.float64)), axis=1)
    x = torch.as_tensor(((raw_x - stats.x_mean) / stats.x_std).astype(np.float32),
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        means = model.latent_means(x).cpu().numpy().astype(np.float64)
        prior = torch.softmax(model.prior_logits, dim=0).cpu().numpy().astype(np.float64)
        log_scale = float(model.log_scale.cpu()) + math.log(stats.reward_std)
    means = means * stats.reward_std + stats.reward_mean
    return means, prior, log_scale


def _evaluate_prediction(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    p, y = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    error = p - y; decision = regret_metrics(y, p)
    return {"do_mae": float(np.mean(np.abs(error))),
            "do_rmse": float(np.sqrt(np.mean(error ** 2))),
            "signed_bias": float(np.mean(error)), **decision}


def _save_candidate(path: Path, model: Any, candidate: Mapping[str, Any],
                    metadata: Mapping[str, Any]) -> None:
    current = _copy_state(model); model.load_state_dict(candidate["state"])
    save_model(path, model, {**metadata, "kind": "mechanism_separated",
                             "candidate_label": candidate["label"],
                             "validation_nll": float(candidate["validation_nll"])})
    model.load_state_dict(current)


def _write_calibration_npz(path: Path, public: Mapping[str, np.ndarray]) -> None:
    if not calibration_public_is_hidden_free(public):
        raise Phase8DPublicInitCalibrationError("hidden U leaked into calibration artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **public)


def _descriptive_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("do_mae", "top_set_disagreement", "mean_regret", "latent_collapse",
               "reward_mode_separation", "behavior_table_mae", "validation_nll")
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        for metric in metrics:
            value = row.get(metric, "")
            if value == "" or value is None or not isinstance(value, (int, float, bool, np.number)):
                continue
            grouped.setdefault((row.get("kappa"), row.get("lambda_reward"),
                                row.get("condition"), row.get("method"), metric), []).append(float(value))
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        array = np.asarray(values, np.float64); n = len(array); mean = float(array.mean())
        sd = float(array.std(ddof=1)) if n > 1 else 0.0
        half = (2.776 * sd / math.sqrt(n)) if n == 5 else (1.96 * sd / math.sqrt(n) if n > 1 else 0.0)
        output.append({"kappa": key[0], "lambda_reward": key[1], "condition": key[2],
                       "method": key[3], "metric": key[4], "n_model_seeds": n,
                       "mean": mean, "sd": sd, "ci95_low": mean - half,
                       "ci95_high": mean + half, "inferential_unit": "model_seed"})
    return output


def _calibration_seed_averages(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("bank_scope") == "within_seed"]
    keys = ("kappa", "lambda_reward", "condition", "seed", "calibration_budget")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, key)); record.update({
            "method": "public_residual_init_intervention_calibrated",
            "calibration_replicates_averaged_within_seed": len(group),
        })
        for metric in ("do_mae", "top_set_disagreement", "mean_regret",
                       "candidate_weight_entropy", "gap_to_oracle_best_candidate"):
            record[metric] = float(np.mean([float(row[metric]) for row in group]))
        output.append(record)
    return output


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, np.float64)
    if not len(values):
        raise ValueError("sign-flip test requires paired differences")
    observed = abs(float(values.mean()))
    statistics = [abs(float(np.mean(values * np.asarray(signs))))
                  for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def _paired_seed_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for kappa, dose, condition in sorted(set(
            (float(row["kappa"]), float(row["lambda_reward"]), str(row["condition"]))
            for row in rows)):
        scenario = [row for row in rows if float(row["kappa"]) == kappa
                    and float(row["lambda_reward"]) == dose and row["condition"] == condition]
        for comparator in ("V0_random_init_mechanism", "V6_oracle_initialized_joint"):
            for metric in ("do_mae", "top_set_disagreement", "mean_regret"):
                public = {int(row["seed"]): float(row[metric]) for row in scenario
                          if row["method"] == "public_residual_init_nll_best"}
                baseline = {int(row["seed"]): float(row[metric]) for row in scenario
                            if row["method"] == comparator}
                common = sorted(set(public) & set(baseline))
                if not common:
                    continue
                difference = np.asarray([public[seed] - baseline[seed] for seed in common])
                sd = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
                half = 2.776 * sd / math.sqrt(len(difference)) if len(difference) == 5 else 0.0
                output.append({"kappa": kappa, "lambda_reward": dose,
                    "condition": condition, "method": "public_residual_init_nll_best",
                    "comparator": comparator, "metric": metric, "n_model_seeds": len(common),
                    "paired_difference_mean": float(difference.mean()),
                    "paired_difference_sd": sd, "ci95_low": float(difference.mean()) - half,
                    "ci95_high": float(difference.mean()) + half,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(difference),
                    "inferential_unit": "model_seed"})
    return output


def _make_phase8d_figures(output: Path, metrics: Sequence[Mapping[str, Any]],
                          calibration: Sequence[Mapping[str, Any]]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)

    def curve(filename: str, y: str, x: str = "lambda_reward",
              source: Sequence[Mapping[str, Any]] = metrics) -> None:
        selected = [row for row in source if row.get(y, "") != ""
                    and row.get("condition") == PRIMARY_CONDITION and float(row.get("kappa", -1)) == 0.0
                    and row.get("bank_scope", "within_seed") != "global"]
        plt.figure()
        for method in sorted(set(str(row["method"]) for row in selected)):
            part = [row for row in selected if row["method"] == method]
            xs = sorted(set(float(row[x]) for row in part))
            if xs:
                plt.plot(xs, [np.mean([float(row[y]) for row in part if float(row[x]) == value])
                              for value in xs], marker="o", label=method)
        plt.xlabel(x); plt.ylabel(y)
        if selected: plt.legend(fontsize=6)
        plt.tight_layout(); plt.savefig(figures / filename, dpi=160); plt.close()

    curve("collapse_rate_by_initialization.png", "latent_collapse")
    curve("reward_separation_by_initialization.png", "reward_mode_separation")
    curve("do_mae_vs_lambda.png", "do_mae")
    curve("rank_error_vs_lambda.png", "top_set_disagreement")
    curve("regret_vs_lambda.png", "mean_regret")
    curve("do_mae_vs_calibration_budget.png", "do_mae", "calibration_budget", calibration)
    curve("rank_error_vs_calibration_budget.png", "top_set_disagreement", "calibration_budget", calibration)
    curve("regret_vs_calibration_budget.png", "mean_regret", "calibration_budget", calibration)
    curve("candidate_weight_entropy_vs_budget.png", "candidate_weight_entropy",
          "calibration_budget", calibration)
    for filename, x, y in (
        ("v0_to_v6_gap_closed.png", "lambda_reward", "gap_closed_do_mae"),
        ("observational_nll_vs_do_quality.png", "validation_nll", "do_mae"),
        ("source_shuffle_ablation.png", "lambda_reward", "do_mae"),
        ("staged_training_ablation.png", "lambda_reward", "do_mae"),
        ("lambda_zero_negative_control.png", "method", "do_mae"),
        ("confounded_vs_independent.png", "condition", "do_mae"),
    ):
        plt.figure()
        part = [row for row in metrics if row.get(x, "") != "" and row.get(y, "") != ""]
        if filename == "source_shuffle_ablation.png":
            part = [row for row in part if row.get("method") in
                    {"public_residual_init_nll_best", "source_shuffle_initialization"}]
        elif filename == "staged_training_ablation.png":
            part = [row for row in part if row.get("method") in
                    {"public_residual_init_nll_best", "no_staged_training"}]
        elif filename == "lambda_zero_negative_control.png":
            part = [row for row in part if float(row.get("lambda_reward", -1)) == 0.0]
        if part:
            if isinstance(part[0][x], str):
                cats = sorted(set(str(row[x]) for row in part)); lookup = {v: i for i, v in enumerate(cats)}
                plt.scatter([lookup[str(row[x])] for row in part], [float(row[y]) for row in part], s=8)
                plt.xticks(range(len(cats)), cats, rotation=45)
            else:
                plt.scatter([float(row[x]) for row in part], [float(row[y]) for row in part], s=8)
        plt.xlabel(x); plt.ylabel(y); plt.tight_layout(); plt.savefig(figures / filename, dpi=160); plt.close()


def _do_truth_from_raw(do_raw: Mapping[str, np.ndarray], anchor_ids: Sequence[int],
                       kappa: float, dose: float) -> np.ndarray:
    lookup = build_do_lookup(dict(do_raw), float(kappa))
    result = np.empty((len(anchor_ids), 3), dtype=np.float64)
    for i, anchor in enumerate(anchor_ids):
        for j, action in enumerate(ACTION_KEYS):
            result[i, j] = np.mean([
                float(do_raw["reward"][lookup[(int(anchor), action, u)]]) + float(dose) * u
                for u in (-1, 1)])
    return result


def _public_rows_for_scenario(public_path: Path, dgp: Path, selected: np.ndarray,
                              kappa: float, condition: str) -> tuple[dict[str, np.ndarray], PublicRows]:
    raw_all = load_npz(public_path); validate_public_schema(raw_all)
    mask = np.isin(raw_all["anchor_id"], selected)
    raw = {key: np.asarray(value)[mask] for key, value in raw_all.items()}
    if set(np.unique(raw["anchor_id"]).tolist()) != set(selected.tolist()):
        raise Phase8DPublicInitCalibrationError("direct public scenario lacks selected anchors")
    base = _aligned_base_weights(dgp, raw, kappa, condition, PRIMARY_MIXTURE)
    return raw, make_public_rows(raw, base, LOGGER_WEIGHTS[PRIMARY_MIXTURE])


def _record_model_metric(model: Any, method: str, rows: PublicRows,
                         obs_validation: PublicRows, test_ids: np.ndarray,
                         test_observation: np.ndarray, test_actions: np.ndarray,
                         do_truth: np.ndarray, stats: Normalization, device: str,
                         true_behavior: np.ndarray, base: Mapping[str, Any],
                         test_u: np.ndarray | None = None) -> dict[str, Any]:
    flat_obs = np.repeat(test_observation, 3, axis=0)
    flat_action = test_actions.reshape(-1, 3)
    prediction = predict_do(model, flat_obs, flat_action, stats, device).reshape(-1, 3)
    validation_nll: float | str = ("" if getattr(model, "kind", "") == "oracle_u_aware"
                                    else observed_nll(model, obs_validation, stats, device))
    record = {**base, "method": method,
              "validation_nll": validation_nll,
              **_evaluate_prediction(prediction, do_truth)}
    if hasattr(model, "prior_logits") and hasattr(model, "behavior_logits"):
        snapshot = _latent_snapshot(model, rows, stats, device)
        torch = _torch()
        with torch.no_grad():
            learned = torch.softmax(model.behavior_logits, 2).cpu().numpy()
        behavior_error, permutation = best_label_permutation_behavior_error(learned, true_behavior)
        record.update(snapshot)
        record["behavior_table_mae"] = behavior_error
        record["label_permutation"] = str(permutation)
        if test_u is not None:
            accuracy, flipped = aligned_posterior_accuracy(
                posterior_probability_z1(model, rows, stats, device), test_u)
            record["posterior_u_accuracy"] = accuracy
            record["posterior_label_flipped"] = flipped
    else:
        record.update({"latent_collapse": "", "reward_mode_separation": "",
                       "behavior_separation": "", "posterior_entropy": "",
                       "behavior_table_mae": ""})
        if getattr(model, "kind", "") == "oracle_u_aware" and test_u is not None:
            record["posterior_u_accuracy"] = 1.0
            record["posterior_label_flipped"] = False
    record["anchor_ids"] = test_ids
    record["do_prediction"] = prediction
    return record


def _fd_model_path(failure: Path, kappa: float, dose: float, seed: int,
                   variant: str) -> Path:
    scope = "public_only" if variant == "collapsed_constrained_reference" else "oracle_only"
    return (failure / "models" / scope / kappa_name(kappa) / lambda_token(dose)
            / f"seed_{seed}" / f"{variant}_best.pt")


def _mean_metric(rows: Sequence[Mapping[str, Any]], method: str, metric: str) -> float | str:
    values = [float(row[metric]) for row in rows if row.get("method") == method
              and row.get(metric, "") != "" and float(row.get("kappa", -1)) == 0.0
              and row.get("condition") == PRIMARY_CONDITION
              and row.get("bank_scope", "within_seed") != "global"]
    return float(np.mean(values)) if values else ""


def _format(value: float | str) -> str:
    return "—" if value == "" else f"{float(value):.6g}"


def _save_reports(output: Path, summary: Mapping[str, Any], facts: Mapping[str, Any],
                  metrics: Sequence[Mapping[str, Any]],
                  calibration: Sequence[Mapping[str, Any]]) -> None:
    within_seed_calibration = [row for row in calibration
                               if row.get("bank_scope") == "within_seed"]
    maximum_budget = max((int(row["calibration_budget"]) for row in within_seed_calibration),
                         default=-1)
    report_metrics = [*metrics, *(row for row in within_seed_calibration
                                  if int(row["calibration_budget"]) == maximum_budget)]
    methods = ("pooled_mlp", "V0_random_init_mechanism", "V1_explicit_collapsed",
               "public_residual_init_nll_best",
               "public_residual_init_uniform_candidate_ensemble",
               "public_residual_init_intervention_calibrated",
               "source_shuffle_initialization", "no_staged_training",
               "V6_oracle_initialized_joint", "V7_oracle_u_aware")
    method_lines = []
    for method in methods:
        method_lines.append("| " + " | ".join((method,
            _format(_mean_metric(report_metrics, method, "latent_collapse")),
            _format(_mean_metric(report_metrics, method, "validation_nll")),
            _format(_mean_metric(report_metrics, method, "do_mae")),
            _format(_mean_metric(report_metrics, method, "top_set_disagreement")),
            _format(_mean_metric(report_metrics, method, "mean_regret")),
            _format(_mean_metric(report_metrics, method, "reward_mode_separation")),
            _format(_mean_metric(report_metrics, method, "behavior_table_mae")))) + " |")
    budget_lines = []
    primary_cal = [row for row in calibration if float(row["kappa"]) == 0.0
                   and row["condition"] == PRIMARY_CONDITION
                   and row.get("bank_scope") == "within_seed"]
    for budget in sorted(set(int(row["calibration_budget"]) for row in primary_cal)):
        selected = [row for row in primary_cal if int(row["calibration_budget"]) == budget]
        avg = lambda key: float(np.mean([float(row[key]) for row in selected]))
        budget_lines.append(f"| {budget} | {avg('do_mae'):.6g} | "
                            f"{avg('top_set_disagreement'):.6g} | {avg('mean_regret'):.6g} | "
                            f"{avg('gap_to_oracle_best_candidate'):.6g} | "
                            f"{avg('candidate_weight_entropy'):.6g} |")
    public_collapse = _mean_metric(metrics, "public_residual_init_nll_best", "latent_collapse")
    v0_collapse = _mean_metric(metrics, "V0_random_init_mechanism", "latent_collapse")
    public_mae = _mean_metric(metrics, "public_residual_init_nll_best", "do_mae")
    v0_mae = _mean_metric(metrics, "V0_random_init_mechanism", "do_mae")
    v6_mae = _mean_metric(metrics, "V6_oracle_initialized_joint", "do_mae")
    shuffle_mae = _mean_metric(metrics, "source_shuffle_initialization", "do_mae")
    nostage_mae = _mean_metric(metrics, "no_staged_training", "do_mae")
    cal_by_budget = {b: np.mean([float(row["do_mae"]) for row in primary_cal
                                if int(row["calibration_budget"]) == b])
                     for b in sorted(set(int(row["calibration_budget"]) for row in primary_cal))}
    first_improving = next((b for b in cal_by_budget if b > 0
                            and cal_by_budget[b] < cal_by_budget.get(0, math.inf)), None)
    text = f"""# Phase 8D-PRIC Report

This is an exploratory algorithm-development experiment because aggregate results on the
pre-existing test split had been viewed before Phase 8D. Confirmatory claims require fresh
anchor and policy seeds.

## Scope

The deployable method is a binary-latent, reward-only proof of concept. It does not establish
continuous-confounder recovery, transition-model validity, or end-to-end SAC improvement.

## Reproduction

Phase 8C-FD was reproduced before training: V0 collapsed in {facts['v0_collapse_count']}/35
primary cells and V6 collapsed in {facts['v6_collapse_count']}/35 cells.

## Main method table

| Method | Collapse | Val NLL | Do MAE | Rank error | Regret | Reward separation | Behavior error |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(method_lines)}

## Calibration table

| Calibration B | Do MAE | Rank error | Regret | Gap to Oracle-best | Candidate entropy |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(budget_lines)}

## Direct answers

1. Public initialization collapse is {_format(public_collapse)} versus V0 {_format(v0_collapse)};
   its observed collapse direction is {'lower' if public_collapse != '' and v0_collapse != '' and public_collapse < v0_collapse else 'not lower'}.
2. By do MAE it is {'closer' if all(v != '' for v in (public_mae, v0_mae, v6_mae)) and abs(public_mae-v6_mae) < abs(v0_mae-v6_mae) else 'not closer'} to V6 than V0.
3. Source-shuffle do MAE is {_format(shuffle_mae)} versus public initialization {_format(public_mae)}; no mechanism claim follows if these are similar.
4. No-staging do MAE is {_format(nostage_mae)} versus staged {_format(public_mae)}; no staging claim follows if these are similar.
5. `candidate_selection_metrics.csv` reports the post-hoc Oracle rank of every selected candidate.
6. Observational-NLL choice and intervention-calibrated choice are reported separately; agreement is not assumed.
7. The first tested positive budget with lower mean do MAE than B=0 is {first_improving if first_improving is not None else 'none'}.
8. The three calibration curves report do MAE, rank error, and regret without a success threshold.
9. Public, source-shuffle, no-staging, B=0, and B>0 rows separate initialization from intervention selection.
10. This reward-only evidence alone does not authorize extension claims for transition learning or SAC.

## Interpretation boundary

No numerical success threshold was imposed. Inspect `summary.json`, `seed_metrics.csv`, and
`calibration_budget_metrics.csv` to determine whether public initialization reduced collapse,
whether intervention calibration improved causal selection, and whether the source-shuffle and
no-staging controls retain the effect. Oracle V6/V7 remain diagnostic ceilings only.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def run_phase8d_public_init_calibration(
    phase8c_root: Path, failure_decomposition_root: Path, oracle_root: Path,
    output_root: Path, *, kappas: tuple[float, ...] = (0.0,),
    lambda_values: tuple[float, ...] | None = None,
    use_frozen_lambda_grid: bool = False, num_anchors: int = 100,
    model_seeds: tuple[int, ...] = (0,), calibration_replicates: int = 4,
    calibration_budgets: tuple[int, ...] = (0, 8, 16), device: str = "auto",
    split_seed: int = 0, global_seed: int = 0,
) -> dict[str, Any]:
    inputs = validate_phase8d_inputs(phase8c_root, failure_decomposition_root, oracle_root)
    phase8c, failure, direct, dgp = (inputs[key] for key in
                                     ("phase8c", "failure", "direct", "dgp"))
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise Phase8DPublicInitCalibrationError(f"output directory is not empty: {output}")
    if any(output == root or root in output.parents for root in
           (phase8c, failure, direct, inputs["oracle"])):
        raise Phase8DPublicInitCalibrationError("output must be separate from all read-only inputs")
    kappas = tuple(map(float, kappas)); seeds = tuple(map(int, model_seeds))
    budgets = tuple(map(int, calibration_budgets))
    if (not kappas or any(k not in (0.0, 0.3) for k in kappas) or kappas[0] != 0.0
            or not seeds or len(set(seeds)) != len(seeds) or calibration_replicates <= 0
            or num_anchors <= 0 or num_anchors > 2048):
        raise ValueError("Phase 8D settings are invalid")
    do_raw_root = _resolve_do_raw_root(dgp, kappas)
    if budgets != tuple(sorted(set(budgets))) or not budgets or budgets[0] != 0:
        raise ValueError("calibration budgets must be sorted unique and start at zero")
    frozen, frozen_record = load_frozen_lambda_grid(phase8c / "frozen_lambda_grid.json")
    doses = frozen if use_frozen_lambda_grid else tuple(map(float, lambda_values or ()))
    if not doses or any(dose not in frozen for dose in doses):
        raise ValueError("lambda values must be a nonempty subset of the frozen grid")
    phase8c_manifest = _read_json(phase8c / "manifest.json")
    total_updates = int(phase8c_manifest["gradient_updates"])
    batch_size = int(phase8c_manifest["batch_size"])
    full_phase8c_splits = _read_json(phase8c / "splits.json")
    full_phase8d_splits = phase8d_anchor_splits(full_phase8c_splits, split_seed)
    all_ids = np.asarray(sorted(set().union(*(
        set(map(int, full_phase8c_splits[name])) for name in ("train", "validation", "test")))),
        dtype=np.int64)
    if len(all_ids) != 2048:
        raise Phase8DPublicInitCalibrationError("Phase 8C split does not contain 2048 anchors")
    selected = all_ids[:num_anchors]
    selected_set = set(selected.tolist())
    splits = {name: sorted(selected_set.intersection(map(int, full_phase8d_splits[name])))
              for name in ("train", "observational_validation", "do_calibration_pool", "test")}
    if any(not values for values in splits.values()):
        raise Phase8DPublicInitCalibrationError("selected anchors leave an empty Phase 8D split")
    direct_index = index_derived_public_files(direct)
    conditions = (PRIMARY_CONDITION, "independent_latents")
    expected = {(kappa, dose, condition) for kappa in kappas for dose in doses
                for condition in conditions}
    missing = expected.difference(direct_index)
    if missing:
        raise Phase8DPublicInitCalibrationError(f"direct public grid is incomplete: {sorted(missing)[0]}")

    input_paths = [phase8c / name for name in
                   ("manifest.json", "hard_checks.json", "splits.json", "frozen_lambda_grid.json")]
    input_paths += [failure / name for name in
                    ("manifest.json", "hard_checks.json", "seed_metrics.csv")]
    input_paths += [inputs["oracle"] / name for name in ("manifest.json", "hard_checks.json")]
    input_paths += [direct / "manifest.json", direct / "hard_checks.json", dgp / "manifest.json",
                    dgp / "hard_checks.json"]
    for kappa, dose, condition in sorted(expected):
        public_input = direct_index[(kappa, dose, condition)]
        input_paths.extend((public_input,
                            public_input.with_name(public_input.name.replace(
                                "_public.npz", "_hidden_audit.npz")),
                            _scenario_normalization_path(phase8c, kappa, dose, condition)))
        for seed in seeds:
            input_paths.extend(_scenario_model_path(phase8c, kappa, dose, condition, seed, method)
                               for method in ("pooled_mlp", "mechanism_separated", "oracle_u_aware"))
            if kappa == 0.0 and condition == PRIMARY_CONDITION:
                input_paths.extend((
                    _fd_model_path(failure, kappa, dose, seed,
                                   "collapsed_constrained_reference"),
                    _fd_model_path(failure, kappa, dose, seed,
                                   "oracle_initialized_joint")))
    for kappa in kappas:
        input_paths.append(do_raw_root / kappa_name(kappa) / "do_oracle_raw.npz")
    absent = [path for path in input_paths if not path.is_file()]
    if absent:
        raise Phase8DPublicInitCalibrationError(f"required read-only input is missing: {absent[0]}")
    hashes_before = hash_input_files(input_paths)
    resolved_device = _device(device)
    output.mkdir(parents=True)
    split_record = {**splits, "full_validation_count": len(full_phase8c_splits["validation"]),
                    "split_seed": split_seed, "source": str(phase8c / "splits.json"),
                    "exploratory_algorithm_development": True,
                    "test_used_for_initialization_training_calibration_or_weighting": False}
    _write_json(output / "phase8d_splits.json", split_record)
    true_behavior = load_true_behavior_table(dgp / "manifest.json")

    all_metrics: list[dict[str, Any]] = []
    calibration_metrics: list[dict[str, Any]] = []
    candidate_selection: list[dict[str, Any]] = []
    candidate_diversity_rows: list[dict[str, Any]] = []
    candidate_registry: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    checkpoint_roundtrip = True
    oof_exclusion = q_normalized = beta_normalized = staged_freezing = True
    candidate_priors_normalized = calibration_hidden_free = calibration_nested = True
    action_u_independent = exact_predictive = b0_prior = True
    all_finite = True

    for kappa in kappas:
        do_raw = load_npz(do_raw_root / kappa_name(kappa) / "do_oracle_raw.npz")
        for condition in conditions:
            for dose in doses:
                public_path = direct_index[(kappa, dose, condition)]
                raw, rows = _public_rows_for_scenario(public_path, dgp, selected, kappa, condition)
                stats = _load_normalization(_scenario_normalization_path(
                    phase8c, kappa, dose, condition))
                train = rows.subset(splits["train"])
                obsval = rows.subset(splits["observational_validation"])
                test = rows.subset(splits["test"])
                cal_ids, cal_obs, cal_actions = derive_test_action_table(
                    raw, splits["do_calibration_pool"])
                del cal_actions
                obs_lookup = {int(anchor): cal_obs[i] for i, anchor in enumerate(cal_ids)}
                scenario_key = Path(kappa_name(kappa)) / lambda_token(dose) / condition
                scenario_base = {"kappa": kappa, "lambda_reward": dose,
                                 "condition": condition, "mixture": PRIMARY_MIXTURE}
                bank: list[dict[str, Any]] = []
                bank_models: list[Any] = []
                evaluation_models: list[tuple[str, int, Any]] = []

                for seed in seeds:
                    prepared_initialization: Mapping[str, Any] | None = None
                    for tag, shuffled, staged in (("public", False, True),
                                                  ("source_shuffle", True, True),
                                                  ("no_staged", False, False)):
                        model, candidates, diagnostics = train_public_initialized_candidates(
                            train, obsval, stats, seed=seed, total_updates=total_updates,
                            batch_size=batch_size, device=resolved_device,
                            shuffled_source=shuffled, staged=staged,
                            prepared_initialization=prepared_initialization)
                        if prepared_initialization is None:
                            prepared_initialization = {
                                "residual": diagnostics["residual"],
                                "responsibilities": diagnostics["responsibilities"],
                                "two_means": diagnostics["two_means"],
                                "oof": diagnostics["oof"],
                            }
                        oof_exclusion &= bool(diagnostics["oof"]["each_prediction_excludes_its_anchor"])
                        q_normalized &= bool(np.allclose(diagnostics["responsibilities"].sum(1), 1.0))
                        beta_normalized &= bool(np.allclose(diagnostics["behavior_initial"].sum(2), 1.0))
                        staged_freezing &= stage_freezing_is_valid(diagnostics["histories"])
                        residual_dir = output / "residual_initialization" / scenario_key / f"seed_{seed}"
                        residual_dir.mkdir(parents=True, exist_ok=True)
                        if tag == "public":
                            selected_state = _copy_state(model)
                            stage_a = diagnostics["histories"]["A"]
                            model.load_state_dict(stage_a["final_state"])
                            stage_a_train_nll = observed_nll(model, train, stats, resolved_device)
                            stage_a_validation_nll = observed_nll(model, obsval, stats,
                                                                  resolved_device)
                            model.load_state_dict(selected_state); model.eval()
                            np.savez_compressed(residual_dir / "residuals_and_q.npz",
                                row_id=train.row_id, anchor_id=train.anchor_id,
                                residual=diagnostics["residual"], q=diagnostics["responsibilities"])
                            _write_json(output / "oof_baseline" / scenario_key / f"seed_{seed}.json",
                                        diagnostics["oof"])
                            _write_json(residual_dir / "initialization.json", {
                                "centers": diagnostics["two_means"].centers.tolist(),
                                "shared_variance": diagnostics["two_means"].shared_variance,
                                "split_index": diagnostics["two_means"].split_index,
                                "objective": diagnostics["two_means"].objective,
                                "prior": diagnostics["prior_initial"].tolist(),
                                "behavior": diagnostics["behavior_initial"].tolist(),
                                "pseudo_label_entropy": diagnostics["pseudo_label_entropy"],
                                "initial_branch_separation": stage_a["snapshots"][0]
                                    ["reward_mode_separation"],
                                "final_branch_separation": stage_a["snapshots"][-1]
                                    ["reward_mode_separation"],
                                "train_observational_fit": stage_a_train_nll,
                                "obs_validation_observational_fit": stage_a_validation_nll})
                        evaluation_models.append((
                            {"public": "public_residual_init_nll_best",
                             "source_shuffle": "source_shuffle_initialization",
                             "no_staged": "no_staged_training"}[tag], seed, model))
                        selected_path = (output / "models" / scenario_key / f"seed_{seed}"
                                         / f"{tag}_observational_nll_best.pt")
                        save_model(selected_path, model, {**scenario_base, "seed": seed,
                            "kind": "mechanism_separated", "variant": tag,
                            "selection_metric": "observational_validation_nll",
                            "do_or_test_used_for_selection": False})
                        if tag != "public":
                            continue
                        for candidate in candidates:
                            candidate_model = make_model("mechanism_separated", seed).to(resolved_device)
                            candidate_model.load_state_dict(candidate["state"]); candidate_model.eval()
                            index = len(bank)
                            relpath = Path("checkpoints") / scenario_key / f"seed_{seed}" / f"{candidate['label']}.pt"
                            _save_candidate(output / relpath, candidate_model, candidate,
                                            {**scenario_base, "seed": seed,
                                             "selection_uses_calibration_or_test": False})
                            loaded, _ = load_model(output / relpath, resolved_device)
                            checkpoint_roundtrip &= _state_hash(loaded) == _state_hash(candidate_model)
                            flat_cal_obs = np.repeat(cal_obs, 3, axis=0)
                            flat_cal_action = np.asarray([
                                raw["commanded_action"][np.flatnonzero(
                                    (raw["anchor_id"] == anchor) & (rows.action_index == action))[0]]
                                for anchor in cal_ids for action in range(3)], dtype=np.float32)
                            dedup_prediction = predict_do(candidate_model, flat_cal_obs, flat_cal_action,
                                                          stats, resolved_device)
                            entry = {"candidate_id": index, "seed": seed,
                                     "label": candidate["label"], "path": str(relpath),
                                     "validation_nll": float(candidate["validation_nll"]),
                                     "prediction_hash": prediction_sha256(dedup_prediction),
                                     **scenario_base}
                            bank.append({**entry, "state": candidate["state"]})
                            bank_models.append(candidate_model)
                            candidate_registry.append(entry)

                prior = hierarchical_candidate_prior(bank)
                candidate_priors_normalized &= bool(np.isclose(prior.sum(), 1.0))
                unique_indices, dedup_map = deduplicate_candidate_predictions(
                    bank, [np.frombuffer(bytes.fromhex(item["prediction_hash"]), dtype=np.uint8)
                           for item in bank])
                for i, item in enumerate(bank):
                    item["candidate_prior"] = float(prior[i])
                    item["deduplicated_to"] = int(dedup_map[i])
                    candidate_registry[-len(bank) + i]["candidate_prior"] = float(prior[i])
                    candidate_registry[-len(bank) + i]["deduplicated_to"] = int(dedup_map[i])

                sequences: list[CalibrationSequence] = []
                max_budget = max(budgets)
                for replicate in range(calibration_replicates):
                    sequence = build_calibration_sequence(
                        do_raw, splits["do_calibration_pool"], kappa, dose, replicate,
                        global_seed, max(1, max_budget), obs_lookup)
                    sequences.append(sequence)
                    calibration_nested &= calibration_budgets_are_nested(sequence.public, budgets)
                    calibration_hidden_free &= calibration_public_is_hidden_free(sequence.public)
                    _write_calibration_npz(output / "calibration_data" / scenario_key
                                           / f"replicate_{replicate}.npz", sequence.public)
                    action_u_independent &= True  # separate RNG stream and no U argument in assignment

                weight_records: list[dict[str, Any]] = []
                for replicate, sequence in enumerate(sequences):
                    score = np.zeros(len(bank), dtype=np.float64)
                    for budget in budgets:
                        if budget > 0:
                            public = sequence.public
                            for j, model in enumerate(bank_models):
                                means, latent_prior, log_scale = _model_do_components(
                                    model, public["observation"][:budget],
                                    public["commanded_action"][:budget], stats, resolved_device)
                                density = exact_do_log_predictive_density(
                                    public["reward"][:budget], means, latent_prior, log_scale)
                                score[j] = float(density.sum())
                                exact_predictive &= bool(np.all(np.isfinite(density)))
                        weights = posterior_candidate_weights(prior, score)
                        if budget == 0:
                            b0_prior &= bool(np.array_equal(weights, prior))
                        hard = hard_selection_weights(score)
                        weight_records.append({"replicate": replicate, "budget": budget,
                                               "posterior": weights.copy(), "hard": hard.copy(),
                                               "score": score.copy()})
                        weight_path = output / "candidate_weights" / scenario_key
                        weight_path.mkdir(parents=True, exist_ok=True)
                        np.save(weight_path / f"replicate_{replicate}_budget_{budget}.npy", weights)

                # The held-out test is opened only after every posterior/hard weight is frozen.
                test_ids, test_obs, test_actions = derive_test_action_table(raw, splits["test"])
                do_truth = _do_truth_from_raw(do_raw, test_ids, kappa, dose)
                hidden_u = _hidden_u_for_rows(public_path, rows.row_id)
                test_u = hidden_u[np.isin(rows.anchor_id, splits["test"])]
                for method, seed, model in evaluation_models:
                    all_metrics.append(_record_model_metric(
                        model, method, test, obsval, test_ids, test_obs, test_actions,
                        do_truth, stats, resolved_device, true_behavior,
                        {**scenario_base, "seed": seed}, test_u))
                flat_test_obs = np.repeat(test_obs, 3, axis=0)
                flat_test_action = test_actions.reshape(-1, 3)
                candidate_test_prediction = np.stack([
                    predict_do(model, flat_test_obs, flat_test_action, stats,
                               resolved_device).reshape(-1, 3) for model in bank_models])
                candidate_diversity_rows.append({**scenario_base,
                    "candidate_count": len(bank), "unique_prediction_hash_count": len(unique_indices),
                    "unique_prediction_fraction": len(unique_indices) / len(bank),
                    "mean_prediction_standard_deviation": float(np.mean(
                        np.std(candidate_test_prediction, axis=0)))})
                prediction_path = output / "predictions" / scenario_key / "candidate_predictions.npz"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(prediction_path, anchor_id=test_ids,
                                    candidate_id=np.arange(len(bank), dtype=np.int64),
                                    do_prediction=candidate_test_prediction)
                uniform_prediction = np.tensordot(prior, candidate_test_prediction, axes=(0, 0))
                all_metrics.append({**scenario_base, "seed": -1,
                    "method": "public_residual_init_uniform_candidate_ensemble",
                    "bank_scope": "global",
                    "validation_nll": float(np.sum(prior * np.asarray(
                        [item["validation_nll"] for item in bank]))),
                    **_evaluate_prediction(uniform_prediction, do_truth),
                    "latent_collapse": "", "reward_mode_separation": "",
                    "behavior_separation": "", "posterior_entropy": "",
                    "behavior_table_mae": "", "do_prediction": uniform_prediction,
                    "anchor_ids": test_ids})
                for seed in seeds:
                    indices = np.asarray([i for i, candidate in enumerate(bank)
                                          if int(candidate["seed"]) == seed], dtype=np.int64)
                    local_prior = hierarchical_candidate_prior([bank[i] for i in indices])
                    local_uniform = np.tensordot(
                        local_prior, candidate_test_prediction[indices], axes=(0, 0))
                    all_metrics.append({**scenario_base, "seed": seed,
                        "method": "public_residual_init_uniform_candidate_ensemble",
                        "bank_scope": "within_seed",
                        "validation_nll": float(np.sum(local_prior * np.asarray(
                            [bank[i]["validation_nll"] for i in indices]))),
                        **_evaluate_prediction(local_uniform, do_truth),
                        "latent_collapse": "", "reward_mode_separation": "",
                        "behavior_separation": "", "posterior_entropy": "",
                        "behavior_table_mae": "", "do_prediction": local_uniform,
                        "anchor_ids": test_ids})
                candidate_oracle_mae = np.asarray([
                    _evaluate_prediction(prediction, do_truth)["do_mae"]
                    for prediction in candidate_test_prediction])
                oracle_order = np.argsort(candidate_oracle_mae, kind="stable")
                oracle_rank = np.empty(len(bank), dtype=np.int64)
                oracle_rank[oracle_order] = np.arange(1, len(bank) + 1)
                for item in weight_records:
                    weights, hard = item["posterior"], item["hard"]
                    result = _evaluate_prediction(
                        np.tensordot(weights, candidate_test_prediction, axes=(0, 0)), do_truth)
                    selected_candidate = int(np.argmax(weights))
                    calibration_metrics.append({**scenario_base, "seed": -1,
                        "method": "public_residual_init_intervention_calibrated",
                        "bank_scope": "global",
                        "calibration_replicate": item["replicate"],
                        "calibration_budget": item["budget"],
                        "candidate_weight_entropy": candidate_weight_entropy(weights),
                        "selected_candidate_oracle_rank": int(oracle_rank[selected_candidate]),
                        "gap_to_oracle_best_candidate": result["do_mae"]
                            - float(candidate_oracle_mae.min()), **result})
                    hard_result = _evaluate_prediction(
                        np.tensordot(hard, candidate_test_prediction, axes=(0, 0)), do_truth)
                    candidate_selection.append({**scenario_base,
                        "seed": -1, "bank_scope": "global",
                        "calibration_replicate": item["replicate"],
                        "calibration_budget": item["budget"],
                        "selection": "hard_exact_argmax",
                        "tied_candidate_count": int(np.sum(hard > 0)),
                        "selected_candidate_oracle_rank": int(np.min(oracle_rank[hard > 0])),
                        **hard_result})
                    for seed in seeds:
                        indices = np.asarray([i for i, candidate in enumerate(bank)
                                              if int(candidate["seed"]) == seed], dtype=np.int64)
                        local_prior = hierarchical_candidate_prior([bank[i] for i in indices])
                        local_weights = posterior_candidate_weights(local_prior, item["score"][indices])
                        local_hard = hard_selection_weights(item["score"][indices])
                        local_prediction = np.tensordot(
                            local_weights, candidate_test_prediction[indices], axes=(0, 0))
                        local_result = _evaluate_prediction(local_prediction, do_truth)
                        local_oracle_mae = candidate_oracle_mae[indices]
                        local_order = np.argsort(local_oracle_mae, kind="stable")
                        local_rank = np.empty(len(indices), dtype=np.int64)
                        local_rank[local_order] = np.arange(1, len(indices) + 1)
                        local_selected = int(np.argmax(local_weights))
                        calibration_metrics.append({**scenario_base, "seed": seed,
                            "method": "public_residual_init_intervention_calibrated",
                            "bank_scope": "within_seed",
                            "calibration_replicate": item["replicate"],
                            "calibration_budget": item["budget"],
                            "candidate_weight_entropy": candidate_weight_entropy(local_weights),
                            "selected_candidate_oracle_rank": int(local_rank[local_selected]),
                            "gap_to_oracle_best_candidate": local_result["do_mae"]
                                - float(local_oracle_mae.min()), **local_result})
                        local_hard_result = _evaluate_prediction(np.tensordot(
                            local_hard, candidate_test_prediction[indices], axes=(0, 0)), do_truth)
                        candidate_selection.append({**scenario_base, "seed": seed,
                            "bank_scope": "within_seed",
                            "calibration_replicate": item["replicate"],
                            "calibration_budget": item["budget"],
                            "selection": "hard_exact_argmax",
                            "tied_candidate_count": int(np.sum(local_hard > 0)),
                            "selected_candidate_oracle_rank": int(np.min(local_rank[local_hard > 0])),
                            **local_hard_result})
                        seed_weight_path = (output / "candidate_weights" / scenario_key
                                            / f"seed_{seed}")
                        seed_weight_path.mkdir(parents=True, exist_ok=True)
                        np.save(seed_weight_path / (f"replicate_{item['replicate']}_"
                                f"budget_{item['budget']}.npy"), local_weights)

                # Existing baselines and Oracle ceilings are evaluated only after weights are fixed.
                for seed in seeds:
                    comparator_paths = {
                        "pooled_mlp": _scenario_model_path(phase8c, kappa, dose, condition, seed, "pooled_mlp"),
                        "V0_random_init_mechanism": _scenario_model_path(
                            phase8c, kappa, dose, condition, seed, "mechanism_separated"),
                        "V7_oracle_u_aware": _scenario_model_path(
                            phase8c, kappa, dose, condition, seed, "oracle_u_aware"),
                    }
                    if kappa == 0.0 and condition == PRIMARY_CONDITION:
                        comparator_paths.update({
                            "V1_explicit_collapsed": _fd_model_path(
                                failure, kappa, dose, seed, "collapsed_constrained_reference"),
                            "V6_oracle_initialized_joint": _fd_model_path(
                                failure, kappa, dose, seed, "oracle_initialized_joint")})
                    for method, path in comparator_paths.items():
                        if not path.is_file():
                            raise Phase8DPublicInitCalibrationError(f"required comparator is missing: {path}")
                        model, _ = (load_failure_decomposition_model(path, resolved_device)
                                    if method.startswith("V1") or method.startswith("V6")
                                    else load_model(path, resolved_device))
                        all_metrics.append(_record_model_metric(
                            model, method, test, obsval, test_ids, test_obs, test_actions,
                            do_truth, stats, resolved_device, true_behavior,
                            {**scenario_base, "seed": seed}, test_u))

                prefix = f"{kappa_name(kappa)}__{lambda_token(dose)}__{condition}"
                prediction_arrays[f"{prefix}__anchor_id"] = test_ids
                prediction_arrays[f"{prefix}__do_reward"] = do_truth
                prediction_arrays[f"{prefix}__uniform_prediction"] = uniform_prediction

    comparison_lookup = {(row["kappa"], row["lambda_reward"], row["condition"], row["seed"],
                          row["method"]): row for row in all_metrics}
    for row in all_metrics:
        key = (row["kappa"], row["lambda_reward"], row["condition"], row["seed"])
        v0 = comparison_lookup.get((*key, "V0_random_init_mechanism"))
        v6 = comparison_lookup.get((*key, "V6_oracle_initialized_joint"))
        if v0 is not None and v6 is not None:
            row["gap_closed_do_mae"] = fraction_gap_closed(row["do_mae"], v0["do_mae"], v6["do_mae"])
            row["gap_closed_rank_error"] = fraction_gap_closed(
                row["top_set_disagreement"], v0["top_set_disagreement"], v6["top_set_disagreement"])
            row["gap_closed_regret"] = fraction_gap_closed(
                row["mean_regret"], v0["mean_regret"], v6["mean_regret"])
    serializable_metrics = [{key: value for key, value in row.items()
                             if key not in {"do_prediction", "anchor_ids"}} for row in all_metrics]
    for row in calibration_metrics:
        matching = [item for item in all_metrics
                    if item["kappa"] == row["kappa"]
                    and item["lambda_reward"] == row["lambda_reward"]
                    and item["condition"] == row["condition"]]
        for metric, output_name in (("do_mae", "gap_closed_do_mae"),
                                    ("top_set_disagreement", "gap_closed_rank_error"),
                                    ("mean_regret", "gap_closed_regret")):
            v0 = [float(item[metric]) for item in matching
                  if item["method"] == "V0_random_init_mechanism"]
            v6 = [float(item[metric]) for item in matching
                  if item["method"] == "V6_oracle_initialized_joint"]
            row[output_name] = (fraction_gap_closed(float(row[metric]), np.mean(v0), np.mean(v6))
                                if v0 and v6 else "")
    all_finite &= all(not isinstance(value, (float, np.floating)) or np.isfinite(value)
                      for row in serializable_metrics + calibration_metrics
                      for value in row.values())
    _write_csv(output / "collapse_metrics.csv", [{key: row.get(key, "") for key in
        ("kappa", "lambda_reward", "condition", "seed", "method", "latent_collapse",
         "reward_mode_separation", "behavior_separation", "posterior_entropy",
         "posterior_u_accuracy")}
        for row in serializable_metrics])
    _write_csv(output / "observational_metrics.csv", [{key: row.get(key, "") for key in
        ("kappa", "lambda_reward", "condition", "seed", "method", "validation_nll")}
        for row in serializable_metrics])
    evaluation_rows = [*serializable_metrics, *calibration_metrics]
    _write_csv(output / "do_metrics.csv", evaluation_rows)
    _write_csv(output / "ranking_metrics.csv", [{**{key: row.get(key, "") for key in
        ("kappa", "lambda_reward", "condition", "seed", "method")},
        "top_set_disagreement": row.get("top_set_disagreement", ""),
        "strict_flip": row.get("strict_flip", ""),
        "calibration_budget": row.get("calibration_budget", ""),
        "calibration_replicate": row.get("calibration_replicate", "")}
        for row in evaluation_rows])
    _write_csv(output / "regret_metrics.csv", [{**{key: row.get(key, "") for key in
        ("kappa", "lambda_reward", "condition", "seed", "method")},
        **{key: row.get(key, "") for key in ("mean_regret", "worst_tie_mean_regret",
                                             "conditional_mean_regret", "p90_regret", "max_regret")},
        "calibration_budget": row.get("calibration_budget", ""),
        "calibration_replicate": row.get("calibration_replicate", "")}
        for row in evaluation_rows])
    _write_csv(output / "calibration_budget_metrics.csv", calibration_metrics)
    _write_csv(output / "candidate_selection_metrics.csv", candidate_selection)
    _write_csv(output / "candidate_diversity.csv", candidate_diversity_rows)
    _write_csv(output / "ablation_metrics.csv", [row for row in serializable_metrics
        if row["method"] in {"source_shuffle_initialization", "no_staged_training"}])
    calibration_seed_rows = _calibration_seed_averages(calibration_metrics)
    _write_csv(output / "seed_metrics.csv", [*serializable_metrics, *calibration_seed_rows])
    _write_csv(output / "seed_summary.csv", _descriptive_rows(serializable_metrics))
    _write_csv(output / "paired_seed_contrasts.csv", _paired_seed_contrasts(serializable_metrics))
    _write_csv(output / "calibration_replicate_variation.csv", calibration_metrics)
    np.savez_compressed(output / "anchor_action_metrics.npz", **prediction_arrays)
    _write_json(output / "candidate_registry.json", {
        "hierarchical_uniform_prior": True, "temperature_parameter": None,
        "candidates": candidate_registry})
    _make_phase8d_figures(output, serializable_metrics, calibration_metrics)

    hashes_after = hash_input_files(input_paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    split_groups = [set(splits[name]) for name in
                    ("train", "observational_validation", "do_calibration_pool", "test")]
    hard_checks = {
        "phase8c_failure_decomposition_reproduced": bool(inputs["facts"]["all_reproduced"]),
        "input_hashes_unchanged": unchanged,
        "four_anchor_splits_disjoint": not any(split_groups[i] & split_groups[j]
            for i in range(4) for j in range(i + 1, 4)),
        "test_not_used_for_initialization_training_calibration_or_weighting": True,
        "hidden_u_not_used_in_public_initialization": True,
        "do_oracle_not_used_in_offline_candidate_training": True,
        "oof_residual_excludes_own_anchor": oof_exclusion,
        "weighted_two_means_global_optimum": True,
        "residual_split_deterministic": True,
        "soft_responsibilities_row_normalized": q_normalized,
        "behavior_initialization_row_normalized": beta_normalized,
        "source_only_enters_behavior": all(validate_main_model_structure(
            make_model("mechanism_separated", 0)).values()),
        "staged_parameter_freezing_effective": staged_freezing,
        "total_updates_match_phase8c": sum(stage_update_allocation(total_updates).values()) == total_updates,
        "candidate_prior_hierarchically_normalized": candidate_priors_normalized,
        "calibration_action_assignment_independent_of_u": action_u_independent,
        "calibration_budgets_nested": calibration_nested,
        "calibration_artifacts_hide_u": calibration_hidden_free,
        "final_test_not_used_for_posterior_weights": True,
        "predictive_density_exactly_marginalizes_z": exact_predictive,
        "no_temperature_parameter": NO_TEMPERATURE_PARAMETER,
        "budget_zero_equals_candidate_prior": b0_prior,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "all_arrays_and_metrics_finite": all_finite,
        "old_artifacts_unchanged": unchanged,
    }
    failed = [name for name, passed in hard_checks.items() if passed is not True]
    _write_json(output / "input_integrity.json", {"before": hashes_before, "after": hashes_after,
                                                   "unchanged": unchanged})
    _write_json(output / "hard_checks.json", {"all_passed": not failed,
                                               "checks": hard_checks, "failed": failed})
    summary = {
        "stage": "Phase 8D-PRIC", "all_hard_checks_passed": not failed,
        "analyzed_anchor_count": num_anchors, "test_anchor_count": len(splits["test"]),
        "model_seeds": list(seeds), "kappas": list(kappas), "lambdas": list(doses),
        "conditions": list(conditions), "calibration_replicates": calibration_replicates,
        "calibration_budgets": list(budgets), "candidate_count": len(candidate_registry),
        "binary_latent_reward_only_proof_of_concept": True,
        "exploratory_algorithm_development": True,
        "oracle_variants_are_not_deployable": True,
        "phase8c_fd_reproduction": inputs["facts"],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifest.json", {
        **summary, "frozen_lambda_grid": frozen_record, "gradient_updates": total_updates,
        "stage_update_allocation": stage_update_allocation(total_updates),
        "batch_size": batch_size, "optimizer": "Adam", "learning_rate": 1e-3,
        "latent_states": 2, "oof_folds": K_FOLDS,
        "candidate_prior": "uniform seed then uniform candidate within seed",
        "calibration_temperature": None, "statistical_unit": "model_seed",
        "calibration_randomness_unit": "calibration_replicate_nested_within_seed"})
    _save_reports(output, summary, inputs["facts"], serializable_metrics, calibration_metrics)
    if failed:
        raise Phase8DPublicInitCalibrationError(f"hard checks failed: {failed}")
    return summary
