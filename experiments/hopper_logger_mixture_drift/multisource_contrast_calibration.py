"""Phase 8E-MSCSC: multi-source contrast subspace calibration.

The offline path in this module consumes only public observations, commanded
actions, rewards, source labels, and public row weights. Hidden variables and
do outcomes are accepted only by DGP audit/evaluation helpers whose names make
that boundary explicit.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ACTION_KEYS = ("minus", "base", "plus")
ACTION_MARGINAL = np.asarray((0.45, 0.10, 0.45), dtype=np.float64)
Q_BASE = 0.10
SOURCE_COUNTS = (2, 3, 5, 8)
DIVERSITY_HALF_WIDTHS = (0.0, 0.05, 0.15, 0.20)
REWARD_NOISE_STDS = (0.0, 0.02)
DEFAULT_CALIBRATION_BUDGETS = (0, 8, 16, 32, 64, 128)
PUBLIC_FIELDS = {
    "anchor_id", "observation", "commanded_action", "action_index", "reward",
    "source_id", "kappa", "lambda_reward", "sigma_reward", "condition", "row_weight",
}
FORBIDDEN_PUBLIC_FIELDS = {
    "u_behavior", "u_env", "epsilon", "applied_action", "true_posterior",
    "do_reward", "true_response_branches",
}


class Phase8EMultisourceContrastError(RuntimeError):
    """Raised when a Phase 8E invariant is violated."""


@dataclass(frozen=True)
class PopulationAudit:
    singular_values: np.ndarray
    rank1_explained_variance: float
    rank1_reconstruction_error: float
    affine_do_projection_residual: float
    loading: np.ndarray
    direction: np.ndarray
    centered_norm: float
    numerical_rank: int


@dataclass(frozen=True)
class SVDInitialization:
    center_targets: np.ndarray
    contrast_targets: np.ndarray
    loadings: np.ndarray
    singular_values: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CalibrationFit:
    coefficients: np.ndarray
    prediction: np.ndarray
    residual_sum_squares: float
    rank: int


def _json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise Phase8EMultisourceContrastError(f"required read-only input is missing: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): sha256(Path(path)) for path in paths}


def require_all_passed(path: Path) -> None:
    record = _json(path)
    if record.get("all_passed") is not True or not all(record.get("checks", {}).values()):
        raise Phase8EMultisourceContrastError(f"input hard checks did not all pass: {path}")


def diversity_profile(source_count: int, diversity_half_width: float) -> np.ndarray:
    if source_count not in SOURCE_COUNTS:
        raise ValueError(f"unsupported source count: {source_count}")
    if float(diversity_half_width) not in DIVERSITY_HALF_WIDTHS:
        raise ValueError(f"unsupported diversity half-width: {diversity_half_width}")
    values = 0.75 + float(diversity_half_width) * np.linspace(-1.0, 1.0, source_count)
    if not np.all((values > 0.5) & (values < 1.0)):
        raise Phase8EMultisourceContrastError("source probabilities must lie strictly in (0.5, 1)")
    return values


def diversity_label(half_width: float) -> str:
    return {0.0: "0", 0.05: "low", 0.15: "medium", 0.20: "high"}[float(half_width)]


def multisource_behavior_probabilities(p_values: Sequence[float]) -> np.ndarray:
    """Return P(action | U, source), with U order (-1,+1)."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.all((p > 0.5) & (p < 1.0)):
        raise ValueError("p_e values must be one-dimensional and in (0.5,1)")
    table = np.empty((len(p), 2, 3), dtype=np.float64)
    table[:, 0, :] = np.stack((0.9 * p, np.full_like(p, Q_BASE), 0.9 * (1.0 - p)), axis=1)
    table[:, 1, :] = np.stack((0.9 * (1.0 - p), np.full_like(p, Q_BASE), 0.9 * p), axis=1)
    if not np.allclose(table.sum(axis=2), 1.0, atol=1e-15, rtol=0.0):
        raise Phase8EMultisourceContrastError("source action probabilities do not sum to one")
    return table


def source_action_marginals(probabilities: np.ndarray) -> np.ndarray:
    table = np.asarray(probabilities, dtype=np.float64)
    if table.ndim != 3 or table.shape[1:] != (2, 3):
        raise ValueError("probabilities must have shape [source,2,3]")
    return table.mean(axis=1)


def source_composition_state_action_mass(probabilities: np.ndarray,
                                         source_weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(source_weights, dtype=np.float64)
    marginals = source_action_marginals(probabilities)
    if weights.shape != (len(marginals),) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("source composition weights are invalid")
    return weights @ marginals


def posterior_u_plus(p_values: Sequence[float]) -> np.ndarray:
    """Return P(U=+1 | action, source) for minus/base/plus."""
    p = np.asarray(p_values, dtype=np.float64)
    return np.stack((1.0 - p, np.full_like(p, 0.5), p), axis=1)


def antithetic_reward_noise(count: int, sigma: float, seed: int) -> np.ndarray:
    if count < 0 or sigma < 0:
        raise ValueError("count and sigma must be nonnegative")
    if sigma == 0.0:
        return np.zeros(count, dtype=np.float64)
    rng = np.random.default_rng(seed)
    half = count // 2
    draws = rng.normal(0.0, sigma, size=half)
    paired = np.empty(2 * half, dtype=np.float64)
    paired[0::2] = draws
    paired[1::2] = -draws
    return np.concatenate((paired, np.zeros(count - 2 * half, dtype=np.float64)))


def validate_public_table(public: Mapping[str, np.ndarray]) -> bool:
    if set(public) != PUBLIC_FIELDS or FORBIDDEN_PUBLIC_FIELDS.intersection(public):
        return False
    n = len(np.asarray(public["anchor_id"]))
    if np.asarray(public["observation"]).shape != (n, 12):
        return False
    if np.asarray(public["commanded_action"]).shape != (n, 3):
        return False
    if any(len(np.asarray(public[name])) != n for name in PUBLIC_FIELDS):
        return False
    numeric = ("observation", "commanded_action", "reward", "kappa", "lambda_reward",
               "sigma_reward", "row_weight")
    return all(np.all(np.isfinite(np.asarray(public[name], dtype=np.float64))) for name in numeric)


def population_source_means(original_reward_branches: np.ndarray, p_values: Sequence[float],
                            lambda_reward: float, condition: str) -> np.ndarray:
    """Compute exact m_e(s,a) without exposing the branches to offline fitting."""
    branches = np.asarray(original_reward_branches, dtype=np.float64)
    if branches.ndim != 3 or branches.shape[1:] != (3, 2):
        raise ValueError("reward branches must have shape [anchor,action,u]")
    p = np.asarray(p_values, dtype=np.float64)
    if condition not in {"confounded", "independent_latents"}:
        raise ValueError("unknown latent condition")
    augmented = branches + float(lambda_reward) * np.asarray((-1.0, 1.0))[None, None, :]
    if condition == "independent_latents":
        mean = augmented.mean(axis=2)
        return np.repeat(mean[None, :, :], len(p), axis=0)
    post = posterior_u_plus(p)
    result = np.empty((len(p), branches.shape[0], 3), dtype=np.float64)
    for action in range(3):
        probability_plus = post[:, action][:, None]
        result[:, :, action] = ((1.0 - probability_plus) * augmented[None, :, action, 0]
                                + probability_plus * augmented[None, :, action, 1])
    return result


def do_reward_mean(original_reward_branches: np.ndarray, lambda_reward: float = 0.0) -> np.ndarray:
    branches = np.asarray(original_reward_branches, dtype=np.float64)
    augmented = branches + float(lambda_reward) * np.asarray((-1.0, 1.0))[None, None, :]
    return augmented.mean(axis=2)


def normalize_loadings(values: Sequence[float]) -> np.ndarray:
    loading = np.asarray(values, dtype=np.float64)
    centered = loading - loading.mean()
    scale = math.sqrt(float(np.mean(centered ** 2)))
    if scale <= np.finfo(np.float64).eps:
        return np.zeros_like(centered)
    result = centered / scale
    index = int(np.argmax(np.abs(result)))
    if result[index] < 0:
        result = -result
    return result


def deterministic_rank1_svd(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(centered, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("SVD matrix must be two-dimensional")
    if np.linalg.norm(matrix) <= np.finfo(np.float64).eps * max(1, matrix.size):
        return np.zeros(matrix.shape[0]), np.zeros(matrix.shape[1]), np.zeros(min(matrix.shape))
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    loading = normalize_loadings(left[:, 0])
    raw = left[:, 0]
    nonzero = np.abs(loading) > 1e-12
    multiplier = float(np.mean(raw[nonzero] / loading[nonzero])) if np.any(nonzero) else 0.0
    direction = singular[0] * right[0] * multiplier
    index = int(np.argmax(np.abs(loading)))
    if loading[index] < 0:
        loading, direction = -loading, -direction
    return loading, direction, singular


def audit_population_subspace(source_matrix: np.ndarray, do_response: np.ndarray) -> PopulationAudit:
    matrix = np.asarray(source_matrix, dtype=np.float64)
    target = np.asarray(do_response, dtype=np.float64)
    if matrix.ndim != 2 or target.shape != (matrix.shape[1],):
        raise ValueError("population audit dimensions are inconsistent")
    center = matrix.mean(axis=0)
    centered = matrix - center
    loading, direction, singular = deterministic_rank1_svd(centered)
    reconstructed = loading[:, None] * direction[None, :]
    norm = float(np.linalg.norm(centered))
    reconstruction = float(np.linalg.norm(centered - reconstructed))
    denominator = float(np.sum(singular ** 2))
    explained = float(singular[0] ** 2 / denominator) if denominator > 0 else 1.0
    if np.dot(direction, direction) > 0:
        coefficient = float(np.dot(target - center, direction) / np.dot(direction, direction))
        projected = center + coefficient * direction
    else:
        projected = center
    residual = float(np.linalg.norm(target - projected))
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * max(1.0, singular[0] if len(singular) else 0.0)
    rank = int(np.sum(singular > tolerance))
    return PopulationAudit(singular, explained, reconstruction, residual, loading,
                           direction, norm, rank)


def svd_initialization(source_means: np.ndarray) -> SVDInitialization:
    values = np.asarray(source_means, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("source means must have shape [source,anchor,action]")
    center = values.mean(axis=0)
    contrast = np.zeros_like(center)
    loadings = np.zeros((values.shape[0], 3), dtype=np.float64)
    all_singular: list[np.ndarray] = []
    for action in range(3):
        loading, direction, singular = deterministic_rank1_svd(
            values[:, :, action] - center[None, :, action])
        loadings[:, action] = loading
        contrast[:, action] = direction
        all_singular.append(singular)
    return SVDInitialization(center, contrast, loadings, tuple(all_singular))


def shuffle_source_within_anchor_action(anchor_id: Sequence[int], action_index: Sequence[int],
                                        source_id: Sequence[int], seed: int) -> np.ndarray:
    anchors = np.asarray(anchor_id, dtype=np.int64)
    actions = np.asarray(action_index, dtype=np.int64)
    sources = np.asarray(source_id, dtype=np.int64)
    if anchors.shape != actions.shape or anchors.shape != sources.shape:
        raise ValueError("shuffle arrays are misaligned")
    shuffled = sources.copy()
    rng = np.random.default_rng(seed)
    for anchor, action in sorted(set(zip(anchors.tolist(), actions.tolist()))):
        positions = np.flatnonzero((anchors == anchor) & (actions == action))
        shuffled[positions] = rng.permutation(sources[positions])
    return shuffled


def fixed_draw_public_table(anchor_ids: Sequence[int], observations: np.ndarray,
                            actions: np.ndarray, reward_branches: np.ndarray,
                            p_values: Sequence[float], *, kappa: float,
                            lambda_reward: float, sigma_reward: float,
                            condition: str, sample_budget: int, seed: int
                            ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Stratified fixed-budget public draw plus a separately returned hidden audit."""
    ids = np.asarray(anchor_ids, dtype=np.int64)
    obs = np.asarray(observations, dtype=np.float32)
    command = np.asarray(actions, dtype=np.float32)
    branches = np.asarray(reward_branches, dtype=np.float64)
    p = np.asarray(p_values, dtype=np.float64)
    if obs.shape != (len(ids), 12) or command.shape != (len(ids), 3, 3):
        raise ValueError("anchor public arrays are invalid")
    cells = np.asarray([(source, anchor, action) for source in range(len(p))
                        for anchor in range(len(ids)) for action in range(3)], dtype=np.int64)
    if sample_budget < len(cells):
        raise ValueError("offline sample budget must cover every source-anchor-action cell")
    rng = np.random.default_rng(seed)
    repeats, remainder = divmod(sample_budget, len(cells))
    rows = np.concatenate((np.tile(cells, (repeats, 1)),
                           cells[rng.permutation(len(cells))[:remainder]]), axis=0)
    rows = rows[rng.permutation(len(rows))]
    source, anchor_pos, action = rows.T
    posterior = posterior_u_plus(p)[source, action]
    u_behavior = np.where(rng.random(sample_budget) < posterior, 1, -1).astype(np.int8)
    if condition == "confounded":
        u_env = u_behavior.copy()
    elif condition == "independent_latents":
        u_env = np.where(rng.random(sample_budget) < 0.5, 1, -1).astype(np.int8)
    else:
        raise ValueError("unknown condition")
    epsilon = antithetic_reward_noise(sample_budget, sigma_reward, seed + 79)
    branch_index = ((u_env + 1) // 2).astype(np.int64)
    reward = branches[anchor_pos, action, branch_index] + lambda_reward * u_env + epsilon
    public = {
        "anchor_id": ids[anchor_pos], "observation": obs[anchor_pos],
        "commanded_action": command[anchor_pos, action], "action_index": action.astype(np.int8),
        "reward": reward.astype(np.float64), "source_id": source.astype(np.int16),
        "kappa": np.full(sample_budget, kappa),
        "lambda_reward": np.full(sample_budget, lambda_reward),
        "sigma_reward": np.full(sample_budget, sigma_reward),
        "condition": np.full(sample_budget, condition),
        "row_weight": np.full(sample_budget, 1.0 / sample_budget),
    }
    hidden = {
        "anchor_id": ids[anchor_pos], "action_index": action.astype(np.int8),
        "source_id": source.astype(np.int16), "u_behavior": u_behavior,
        "u_env": u_env, "epsilon": epsilon,
    }
    if not validate_public_table(public):
        raise Phase8EMultisourceContrastError("generated public table is invalid")
    return public, hidden


def empirical_source_mean_matrix(public: Mapping[str, np.ndarray], source_count: int,
                                  ordered_anchor_ids: Sequence[int]) -> np.ndarray:
    validate = validate_public_table(public)
    if not validate:
        raise ValueError("invalid public table")
    anchors = np.asarray(public["anchor_id"], dtype=np.int64)
    actions = np.asarray(public["action_index"], dtype=np.int64)
    sources = np.asarray(public["source_id"], dtype=np.int64)
    reward = np.asarray(public["reward"], dtype=np.float64)
    ids = np.asarray(ordered_anchor_ids, dtype=np.int64)
    if not np.array_equal(ids, np.sort(ids)):
        raise ValueError("ordered anchor ids must be sorted for deterministic aggregation")
    positions = np.searchsorted(ids, anchors)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], anchors):
        raise Phase8EMultisourceContrastError("public rows contain an unknown anchor")
    flat = (sources * len(ids) + positions) * 3 + actions
    size = source_count * len(ids) * 3
    count = np.bincount(flat, minlength=size)
    total = np.bincount(flat, weights=reward, minlength=size)
    if np.any(count == 0):
        raise Phase8EMultisourceContrastError("empirical SVD cell has no draw")
    return (total / count).reshape(source_count, len(ids), 3)


def calibration_features(action_index: Sequence[int], contrast_value: Sequence[float],
                         rank: int = 1) -> np.ndarray:
    action = np.asarray(action_index, dtype=np.int64)
    h = np.asarray(contrast_value, dtype=np.float64)
    if action.shape != h.shape or np.any((action < 0) | (action > 2)):
        raise ValueError("calibration feature inputs are invalid")
    one_hot = np.eye(3, dtype=np.float64)[action]
    if rank == 0:
        return one_hot
    if rank == 1:
        return np.concatenate((one_hot, one_hot * h[:, None]), axis=1)
    raise ValueError("rank must be zero or one")


def closed_form_calibration(base_prediction: Sequence[float], reward: Sequence[float],
                            features: np.ndarray) -> CalibrationFit:
    base = np.asarray(base_prediction, dtype=np.float64)
    outcome = np.asarray(reward, dtype=np.float64)
    design = np.asarray(features, dtype=np.float64)
    if design.ndim != 2 or base.shape != outcome.shape or len(base) != len(design):
        raise ValueError("calibration arrays are misaligned")
    target = outcome - base
    coefficients = np.linalg.pinv(design) @ target
    prediction = base + design @ coefficients
    residual = float(np.sum((outcome - prediction) ** 2))
    return CalibrationFit(coefficients, prediction, residual, int(np.linalg.matrix_rank(design)))


def bic_value(residual_sum_squares: float, sample_count: int, parameter_count: int) -> float:
    if sample_count <= 0 or parameter_count <= 0 or residual_sum_squares < 0:
        raise ValueError("BIC inputs are invalid")
    guard = np.finfo(np.float64).eps
    rss = max(float(residual_sum_squares), guard)
    return float(sample_count * math.log(rss / sample_count) + parameter_count * math.log(sample_count))


def bic_select_rank(rank0_rss: float, rank1_rss: float, sample_count: int) -> int:
    """Return rank 1 only for a strict BIC improvement; exact ties prefer rank 0."""
    return int(bic_value(rank1_rss, sample_count, 6) < bic_value(rank0_rss, sample_count, 3))


def gap_uncertainty_objective(design: np.ndarray, gap_features: np.ndarray) -> float:
    x = np.asarray(design, dtype=np.float64)
    c = np.asarray(gap_features, dtype=np.float64)
    dimension = c.shape[1]
    gram = x.T @ x if len(x) else np.zeros((dimension, dimension), dtype=np.float64)
    eta = np.finfo(np.float64).eps * max(1.0, float(np.trace(gram)))
    inverse = np.linalg.pinv(gram + eta * np.eye(dimension))
    return float(np.einsum("ij,jk,ik->", c, inverse, c))


def pairwise_gap_features(features: np.ndarray, anchor_ids: Sequence[int],
                          action_indices: Sequence[int]) -> np.ndarray:
    phi = np.asarray(features, dtype=np.float64)
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    actions = np.asarray(action_indices, dtype=np.int64)
    gaps: list[np.ndarray] = []
    for anchor in np.unique(anchors):
        rows = {int(actions[index]): phi[index] for index in np.flatnonzero(anchors == anchor)}
        if set(rows) != {0, 1, 2}:
            raise ValueError("each calibration anchor must expose all three actions")
        gaps.extend((rows[0] - rows[1], rows[0] - rows[2], rows[1] - rows[2]))
    return np.asarray(gaps, dtype=np.float64)


def active_query_order(features: np.ndarray, anchor_ids: Sequence[int],
                       action_indices: Sequence[int], budget: int) -> np.ndarray:
    """Outcome-blind lexicographically deterministic A-optimal query sequence."""
    phi = np.asarray(features, dtype=np.float64)
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    actions = np.asarray(action_indices, dtype=np.int64)
    if phi.ndim != 2 or len(phi) != len(anchors) or len(phi) != len(actions):
        raise ValueError("active-query inputs are misaligned")
    if budget < 0 or budget > len(phi):
        raise ValueError("active-query budget is outside the candidate pool")
    gaps = pairwise_gap_features(phi, anchors, actions)
    selected: list[int] = []
    remaining = set(range(len(phi)))
    for _ in range(budget):
        design = phi[selected] if selected else np.empty((0, phi.shape[1]))
        current_rank = int(np.linalg.matrix_rank(design)) if len(design) else 0
        current_objective = gap_uncertainty_objective(design, gaps)
        candidates: list[tuple[int, float, int, int, int]] = []
        for index in remaining:
            proposed = np.vstack((design, phi[index]))
            rank_gain = int(np.linalg.matrix_rank(proposed) - current_rank)
            reduction = current_objective - gap_uncertainty_objective(proposed, gaps)
            candidates.append((-rank_gain, -float(reduction), int(anchors[index]),
                               int(actions[index]), index))
        chosen = min(candidates)[-1]
        selected.append(chosen)
        remaining.remove(chosen)
    return np.asarray(selected, dtype=np.int64)


def random_balanced_query_order(anchor_ids: Sequence[int], action_indices: Sequence[int],
                                budget: int, seed: int) -> np.ndarray:
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    actions = np.asarray(action_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_action = [list(rng.permutation(np.flatnonzero(actions == action))) for action in range(3)]
    result: list[int] = []
    cursor = [0, 0, 0]
    while len(result) < budget:
        action = len(result) % 3
        if cursor[action] >= len(by_action[action]):
            raise ValueError("calibration pool is too small for requested balanced budget")
        result.append(int(by_action[action][cursor[action]]))
        cursor[action] += 1
    return np.asarray(result, dtype=np.int64)


def budgets_are_nested(order: Sequence[int], budgets: Sequence[int]) -> bool:
    sequence = np.asarray(order, dtype=np.int64)
    previous: set[int] = set()
    for budget in sorted(map(int, budgets)):
        current = set(sequence[:budget].tolist())
        if not previous.issubset(current):
            return False
        previous = current
    return True


def reward_prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    error = estimate - target
    return {"do_mae": float(np.mean(np.abs(error))),
            "do_rmse": float(np.sqrt(np.mean(error ** 2))),
            "signed_bias": float(np.mean(error))}


def decision_metrics(truth: np.ndarray, prediction: np.ndarray,
                     atol: float = 1e-10, rtol: float = 1e-8) -> dict[str, float]:
    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    true_top = np.isclose(target, target.max(axis=1, keepdims=True), atol=atol, rtol=rtol)
    pred_top = np.isclose(estimate, estimate.max(axis=1, keepdims=True), atol=atol, rtol=rtol)
    maximum = target.max(axis=1)
    best = maximum - np.max(np.where(pred_top, target, -np.inf), axis=1)
    worst = maximum - np.min(np.where(pred_top, target, np.inf), axis=1)
    if np.any(best < -1e-12):
        raise Phase8EMultisourceContrastError("decision regret became negative")
    positive = best > atol
    top_count = max(1, int(math.ceil(0.01 * len(best))))
    total = float(best.sum())
    return {
        "top_set_disagreement": float(np.mean(np.any(true_top != pred_top, axis=1))),
        "strict_flip": float(np.mean(~np.any(true_top & pred_top, axis=1))),
        "mean_regret": float(np.mean(best)),
        "worst_tie_mean_regret": float(np.mean(worst)),
        "conditional_mean_regret": float(np.mean(best[positive])) if np.any(positive) else 0.0,
        "p90_regret": float(np.quantile(best, 0.9)),
        "max_regret": float(np.max(best)),
        "top_1pct_regret_contribution": (float(np.sort(best)[-top_count:].sum() / total)
                                           if total > 0 else 0.0),
    }


def make_source_free_model(source_count: int, loadings: np.ndarray, seed: int = 0,
                           width: int = 64) -> Any:
    import torch
    torch.manual_seed(seed)

    class SourceFreeScalar(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(15, width), torch.nn.ReLU(),
                torch.nn.Linear(width, width), torch.nn.ReLU(),
                torch.nn.Linear(width, 1))

        def forward(self, public_x: Any) -> Any:
            return self.network(public_x).squeeze(-1)

    class MSCSC(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.g = SourceFreeScalar()
            self.h = SourceFreeScalar()
            self.raw_loadings = torch.nn.Parameter(torch.as_tensor(loadings, dtype=torch.float32))
            self.source_count = int(source_count)

        def normalized_loadings(self) -> Any:
            centered = self.raw_loadings - self.raw_loadings.mean(dim=0, keepdim=True)
            scale = torch.sqrt(torch.mean(centered.square(), dim=0, keepdim=True)).clamp_min(
                torch.finfo(centered.dtype).eps)
            return centered / scale

        def source_mean(self, public_x: Any, source: Any, action: Any) -> Any:
            loading = self.normalized_loadings()[source, action]
            return self.g(public_x) + loading * self.h(public_x)

        def rank0(self, public_x: Any) -> Any:
            return self.g(public_x)

    return MSCSC()


def validate_source_free_model(model: Any) -> bool:
    try:
        g_in = int(model.g.network[0].in_features)
        h_in = int(model.h.network[0].in_features)
        names = {name for name, _ in model.named_parameters()}
    except (AttributeError, TypeError):
        return False
    return g_in == 15 and h_in == 15 and "raw_loadings" in names and not any(
        "source" in name for name in names if name != "raw_loadings")


def fit_source_free_model(public: Mapping[str, np.ndarray], initialization: SVDInitialization,
                          ordered_anchor_ids: Sequence[int], *, seed: int, updates: int,
                          batch_size: int, device: str) -> tuple[Any, dict[str, np.ndarray], dict[str, float]]:
    """SVD-supervised initialization followed by public observational MSE fitting."""
    import torch
    if not validate_public_table(public):
        raise ValueError("invalid public training table")
    selected_device = ("cuda" if device == "auto" and torch.cuda.is_available() else
                       "cpu" if device == "auto" else device)
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise Phase8EMultisourceContrastError("CUDA was requested but is unavailable")
    model = make_source_free_model(initialization.loadings.shape[0], initialization.loadings, seed).to(selected_device)
    obs = np.asarray(public["observation"], dtype=np.float64)
    act = np.asarray(public["commanded_action"], dtype=np.float64)
    raw_x = np.concatenate((obs, act), axis=1)
    x_mean, x_std = raw_x.mean(axis=0), np.maximum(raw_x.std(axis=0), 1e-6)
    y_mean, y_std = float(np.mean(public["reward"])), max(float(np.std(public["reward"])), 1e-6)
    x = torch.as_tensor((raw_x - x_mean) / x_std, dtype=torch.float32, device=selected_device)
    reward = torch.as_tensor((np.asarray(public["reward"]) - y_mean) / y_std,
                             dtype=torch.float32, device=selected_device)
    source = torch.as_tensor(public["source_id"], dtype=torch.long, device=selected_device)
    action = torch.as_tensor(public["action_index"], dtype=torch.long, device=selected_device)

    ordered = np.asarray(ordered_anchor_ids, dtype=np.int64)
    if not np.array_equal(ordered, np.sort(ordered)):
        raise ValueError("ordered anchor ids must be sorted")
    positions = np.searchsorted(ordered, np.asarray(public["anchor_id"], dtype=np.int64))
    key = positions * 3 + np.asarray(public["action_index"], dtype=np.int64)
    unique_key, target_rows = np.unique(key, return_index=True)
    order = np.argsort(unique_key)
    target_rows = target_rows[order]
    unique_key = unique_key[order]
    center_rows = (initialization.center_targets.reshape(-1)[unique_key] - y_mean) / y_std
    contrast_rows = initialization.contrast_targets.reshape(-1)[unique_key] / y_std
    target_x = x[torch.as_tensor(target_rows, dtype=torch.long, device=selected_device)]
    center_target = torch.as_tensor(center_rows, dtype=torch.float32, device=selected_device)
    contrast_target = torch.as_tensor(contrast_rows, dtype=torch.float32, device=selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pretrain_updates = max(1, updates // 4)
    for _ in range(pretrain_updates):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model.g(target_x) - center_target).square().mean()
                + (model.h(target_x) - contrast_target).square().mean())
        loss.backward(); optimizer.step()
    generator = np.random.default_rng(seed + 901)
    final_loss = math.nan
    for _ in range(updates):
        index = generator.integers(0, len(x), size=min(batch_size, len(x)))
        batch = torch.as_tensor(index, dtype=torch.long, device=selected_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.source_mean(x[batch], source[batch], action[batch])
        loss = (prediction - reward[batch]).square().mean()
        loss.backward(); optimizer.step()
        final_loss = float(loss.detach().cpu())
    normalization = {"x_mean": x_mean, "x_std": x_std,
                     "reward_mean": np.asarray(y_mean), "reward_std": np.asarray(y_std)}
    return model, normalization, {"observational_mse_standardized": final_loss,
                                  "gradient_updates": float(updates),
                                  "batch_size": float(batch_size)}


def predict_components(model: Any, normalization: Mapping[str, np.ndarray],
                       observations: np.ndarray, actions: np.ndarray,
                       device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    selected_device = next(model.parameters()).device
    obs = np.asarray(observations, dtype=np.float64)
    command = np.asarray(actions, dtype=np.float64)
    raw = np.concatenate((np.repeat(obs[:, None, :], 3, axis=1), command), axis=2).reshape(-1, 15)
    x = torch.as_tensor((raw - normalization["x_mean"]) / normalization["x_std"],
                        dtype=torch.float32, device=selected_device)
    with torch.no_grad():
        g = model.g(x).cpu().numpy().reshape(len(obs), 3)
        h = model.h(x).cpu().numpy().reshape(len(obs), 3)
    scale, mean = float(normalization["reward_std"]), float(normalization["reward_mean"])
    return g * scale + mean, h * scale


def save_checkpoint(path: Path, model: Any, normalization: Mapping[str, np.ndarray],
                    metadata: Mapping[str, Any]) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "source_count": model.source_count,
                "loadings": model.raw_loadings.detach().cpu(),
                "normalization": {key: np.asarray(value) for key, value in normalization.items()},
                "metadata": dict(metadata)}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    import torch
    record = torch.load(path, map_location=device, weights_only=False)
    model = make_source_free_model(int(record["source_count"]),
                                   np.asarray(record["loadings"]), seed=0)
    model.load_state_dict(record["state_dict"])
    model.to(device); model.eval()
    return model, record["normalization"], record["metadata"]


def all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value))) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True
