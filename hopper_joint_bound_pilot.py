"""Local empirical Separate/Joint bound utilities for the Phase 7C Hopper pilot."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from exact_population_joint_bound import (
    compute_empirical_separate_interval,
    empirical_tuple_envelopes,
    prepare_empirical_coupling_problem,
    solve_empirical_joint_interval,
)


METHOD_NAME = "LOCAL_EMPIRICAL_JOINT_BOUND_PILOT"
FORBIDDEN_FIELDS = frozenset({"u", "hidden_u", "applied_action", "qpos", "qvel"})
REQUIRED_FIELDS = frozenset({
    "observations", "actions", "rewards", "next_observations", "terminated",
    "truncated", "collector_truncated", "source_id",
})
ACTION_LABELS = ("source_1_mean", "source_2_mean", "source_3_mean",
                 "pooled_mean", "fixed_random_1", "fixed_random_2")


def validate_public_data(data: dict[str, np.ndarray]) -> None:
    """Reject hidden simulator state and validate the public Hopper tensor schema."""
    leaked = FORBIDDEN_FIELDS.intersection(data)
    if leaked:
        raise ValueError(f"hidden fields are forbidden: {sorted(leaked)}")
    missing = REQUIRED_FIELDS.difference(data)
    if missing:
        raise ValueError(f"missing public fields: {sorted(missing)}")
    count = np.asarray(data["rewards"]).size
    shapes = {
        "observations": (count, 12), "actions": (count, 3),
        "next_observations": (count, 12),
    }
    for name, shape in shapes.items():
        if np.asarray(data[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name in REQUIRED_FIELDS - {"observations", "actions", "next_observations"}:
        if np.asarray(data[name]).shape != (count,):
            raise ValueError(f"{name} must have shape ({count},)")
    numeric = ("observations", "actions", "rewards", "next_observations")
    if any(not np.all(np.isfinite(data[name])) for name in numeric):
        raise ValueError("public data must be finite")
    if set(np.unique(data["source_id"]).tolist()) != {1, 2, 3}:
        raise ValueError("source_id must contain exactly sources 1, 2, and 3")


def load_public_data(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if "hidden" in path.name.lower():
        raise ValueError("hidden audit files must not be read")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    validate_public_data(data)
    return data


def compute_bellman_outcomes(
    data: dict[str, np.ndarray], gamma: float, reward_mean: float, reward_std: float,
    potential: Any | None,
) -> np.ndarray:
    """Compute fixed-continuation outcomes; collector truncation never closes bootstrap."""
    validate_public_data(data)
    rewards = (np.asarray(data["rewards"], dtype=np.float64) - float(reward_mean)) / (
        max(float(reward_std), 0.0) + 1e-7
    )
    done = np.asarray(data["terminated"], dtype=bool) | np.asarray(data["truncated"], dtype=bool)
    continuation = np.zeros_like(rewards)
    if potential is not None:
        continuation = np.asarray(potential(data["next_observations"]), dtype=np.float64).reshape(-1)
    outcomes = rewards + float(gamma) * (~done) * continuation
    if not np.all(np.isfinite(outcomes)):
        raise RuntimeError("Bellman outcomes contain NaN or Inf")
    return outcomes


def fit_train_normalization(
    observations: np.ndarray, outcomes: np.ndarray,
) -> dict[str, np.ndarray | float]:
    observations = np.asarray(observations, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    state_mean, state_std = observations.mean(axis=0), observations.std(axis=0)
    z_mean, z_std = float(outcomes.mean()), float(outcomes.std())
    return {
        "state_mean": state_mean, "state_std": np.maximum(state_std, 1e-8),
        "z_mean": z_mean, "z_std": max(z_std, 1e-8),
    }


def normalize_states(observations: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    return ((np.asarray(observations, dtype=np.float64) - normalization["state_mean"])
            / normalization["state_std"])


def normalize_outcomes(outcomes: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    return ((np.asarray(outcomes, dtype=np.float64) - normalization["z_mean"])
            / normalization["z_std"])


def evenly_spaced_indices(row_count: int, query_count: int) -> np.ndarray:
    if query_count < 1 or query_count > row_count:
        raise ValueError("query_count must be in [1, row_count]")
    return np.linspace(0, row_count - 1, query_count, dtype=np.int64)


class LocalEmpiricalConditioner:
    """Per-source kNN conditioning fitted exclusively on pooled train states."""

    def __init__(
        self, train_data: dict[str, np.ndarray], normalized_train_states: np.ndarray,
        normalized_train_outcomes: np.ndarray,
    ) -> None:
        validate_public_data(train_data)
        self.data = train_data
        self.states = np.asarray(normalized_train_states, dtype=np.float64)
        self.outcomes = np.asarray(normalized_train_outcomes, dtype=np.float64)
        self.source_rows = {
            source: np.flatnonzero(np.asarray(train_data["source_id"]) == source)
            for source in (1, 2, 3)
        }
        self.trees = {source: cKDTree(self.states[rows]) for source, rows in self.source_rows.items()}

    def query(
        self, normalized_state: np.ndarray, k: int, excluded_train_row: int | None = None,
    ) -> dict[str, Any]:
        atoms, radii = [], []
        for source in (1, 2, 3):
            rows = self.source_rows[source]
            extra = int(excluded_train_row is not None and excluded_train_row in rows)
            if len(rows) < k + extra:
                raise ValueError(f"source {source} has too few train rows for k={k}")
            distances, local_indices = self.trees[source].query(
                np.asarray(normalized_state), k=k + extra
            )
            distances = np.atleast_1d(distances)
            selected = rows[np.atleast_1d(local_indices)]
            if excluded_train_row is not None:
                keep = selected != excluded_train_row
                selected, distances = selected[keep], distances[keep]
            selected, distances = selected[:k], distances[:k]
            atoms.append({
                "actions": np.asarray(self.data["actions"][selected], dtype=np.float64),
                "outcomes": self.outcomes[selected],
                "probabilities": np.full(k, 1.0 / k, dtype=np.float64),
                "source_id": source, "train_rows": selected,
            })
            radii.append(float(distances[-1]))
        ordered = np.sort(radii)
        return {
            "source_atoms": atoms, "source_radii": np.asarray(radii),
            "all_source_radius": float(ordered[-1]), "best_two_radius": float(ordered[1]),
        }


def target_actions(
    source_atoms: list[dict], query_index: int, seed: int, action_count: int = 6,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create deterministic, untuned local-mean and fixed-random target actions."""
    if not 1 <= action_count <= 6:
        raise ValueError("action_count must be in [1, 6]")
    means = [np.mean(atoms["actions"], axis=0) for atoms in source_atoms]
    candidates = means + [np.mean(np.concatenate([a["actions"] for a in source_atoms]), axis=0)]
    labels = list(ACTION_LABELS[:4])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(query_index)]))
    random_number = 1
    while len(candidates) < 6:
        candidates.append(rng.uniform(-1.0, 1.0, size=3))
        labels.append(f"fixed_random_{random_number}")
        random_number += 1
    unique, unique_labels = [], []
    for action, label in zip(candidates, labels):
        if not any(np.allclose(action, known, atol=1e-12, rtol=0.0) for known in unique):
            unique.append(np.asarray(action, dtype=np.float64))
            unique_labels.append(label)
    while len(unique) < 6:
        action = rng.uniform(-1.0, 1.0, size=3)
        if not any(np.allclose(action, known, atol=1e-12, rtol=0.0) for known in unique):
            unique.append(action)
            unique_labels.append(f"fixed_random_{random_number}")
            random_number += 1
    return np.asarray(unique[:action_count]), tuple(unique_labels[:action_count])


def solve_local_problem(
    source_atoms: list[dict], target_action: np.ndarray, rho_lambda: float,
) -> dict[str, Any]:
    """Solve one exact local problem without fallback on coupling infeasibility."""
    separate = compute_empirical_separate_interval(source_atoms, target_action, rho_lambda)
    denominator = int(np.prod([len(atoms["outcomes"]) for atoms in source_atoms]))
    try:
        prepared = prepare_empirical_coupling_problem(source_atoms, rho_lambda)
    except RuntimeError as error:
        return {**separate, "feasible": False, "failure": str(error), "prepared": None,
                "feasible_tuple_fraction": 0.0}
    feasible_fraction = len(prepared["feasible_tuples"]) / denominator
    try:
        joint = solve_empirical_joint_interval(prepared, target_action)
    except RuntimeError as error:
        return {**separate, "feasible": False, "failure": str(error), "prepared": prepared,
                "feasible_tuple_fraction": float(feasible_fraction)}
    result = {**separate, **joint, "feasible": True, "failure": "", "prepared": prepared,
              "feasible_tuple_fraction": float(feasible_fraction)}
    result["upper_gain"] = result["separate_upper"] - result["joint_upper"]
    result["lower_gain"] = result["joint_lower"] - result["separate_lower"]
    result["separate_width"] = result["separate_upper"] - result["separate_lower"]
    result["joint_width"] = result["joint_upper"] - result["joint_lower"]
    result["width_gain"] = result["separate_width"] - result["joint_width"]
    result["upper_dominance_violation"] = max(0.0, result["joint_upper"] - result["separate_upper"])
    result["lower_dominance_violation"] = max(0.0, result["separate_lower"] - result["joint_lower"])
    result["interval_order_violation"] = max(0.0, result["joint_lower"] - result["joint_upper"])
    result["max_marginal_residual"] = max(
        result["max_lower_marginal_error"], result["max_upper_marginal_error"]
    )
    return result


def source3_contribution(joint12: dict[str, Any], joint123: dict[str, Any]) -> dict[str, float]:
    if not joint12["feasible"] or not joint123["feasible"]:
        raise ValueError("Source 3 contribution needs feasible 12 and 123 problems")
    upper = joint12["joint_upper"] - joint123["joint_upper"]
    lower = joint123["joint_lower"] - joint12["joint_lower"]
    return {"source3_upper_gain": upper, "source3_lower_gain": lower,
            "source3_width_gain": upper + lower,
            "source3_upper_violation": max(0.0, -upper),
            "source3_lower_violation": max(0.0, -lower)}


def dual_baselines(
    source_atoms: list[dict], target_action: np.ndarray, rho_lambda: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    separate = compute_empirical_separate_interval(source_atoms, target_action, rho_lambda)
    upper_source = int(np.argmin(separate["source_upper"]))
    lower_source = int(np.argmax(separate["source_lower"]))
    upper, lower = [], []
    target = np.asarray(target_action).reshape(1, -1)
    for source_index, atoms in enumerate(source_atoms):
        radii = rho_lambda * np.linalg.norm(atoms["actions"] - target, axis=1)
        upper.append(atoms["outcomes"] + radii if source_index == upper_source
                     else np.zeros(len(radii)))
        lower.append(atoms["outcomes"] - radii if source_index == lower_source
                     else np.zeros(len(radii)))
    return upper, lower, separate


def exact_violation_correction(
    raw_upper: float, raw_lower: float, eta_upper: float, eta_lower: float,
    separate_upper: float, separate_lower: float,
) -> dict[str, float]:
    """Apply the finite-support scalar correction and known Separate certificate."""
    corrected_upper = float(raw_upper) + max(0.0, float(eta_upper))
    corrected_lower = float(raw_lower) - max(0.0, float(eta_lower))
    return {
        "corrected_upper": corrected_upper, "corrected_lower": corrected_lower,
        "neural_upper_final": min(float(separate_upper), corrected_upper),
        "neural_lower_final": max(float(separate_lower), corrected_lower),
    }


class DualResidualNet:
    """22 -> 128 ReLU -> 128 ReLU -> 2 residual network with a zero head."""

    def __init__(self, device: str = "cpu") -> None:
        torch = importlib.import_module("torch")
        nn = torch.nn
        self.torch = torch
        self.module = nn.Sequential(nn.Linear(22, 128), nn.ReLU(), nn.Linear(128, 128),
                                    nn.ReLU(), nn.Linear(128, 2)).to(device)
        zero_initialize_output_layer(self.module[-1], nn.init.zeros_)

    def __call__(self, values: Any) -> Any:
        return self.module(values)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)


def zero_initialize_output_layer(layer: Any, zero_function: Any) -> None:
    """Zero the two-output residual head using an injectable initializer for testing."""
    zero_function(layer.weight)
    zero_function(layer.bias)


def atom_features(problem: dict[str, Any]) -> np.ndarray:
    """Create the fixed 22D input for every atom of one local problem."""
    rows = []
    for source_index, atoms in enumerate(problem["source_atoms"]):
        one_hot = np.eye(3, dtype=np.float64)[source_index]
        for action, outcome in zip(atoms["actions"], atoms["outcomes"]):
            rows.append(np.concatenate((problem["query_state"], problem["target_action"],
                                        action, [outcome], one_hot)))
    result = np.asarray(rows, dtype=np.float32)
    if result.shape[1] != 22:
        raise RuntimeError("dual feature width must be 22")
    return result


def _dual_tensors(model: DualResidualNet, problem: dict[str, Any], device: str):
    torch = model.torch
    residuals = model(torch.as_tensor(atom_features(problem), dtype=torch.float32, device=device))
    upper0, lower0, separate = dual_baselines(
        problem["source_atoms"], problem["target_action"], problem["rho_lambda"]
    )
    upper0 = torch.as_tensor(np.concatenate(upper0), dtype=torch.float32, device=device)
    lower0 = torch.as_tensor(np.concatenate(lower0), dtype=torch.float32, device=device)
    return upper0 + residuals[:, 0], lower0 + residuals[:, 1], separate


def neural_problem_values(
    model: DualResidualNet, problem: dict[str, Any], device: str,
) -> dict[str, float]:
    """Evaluate raw and exactly corrected finite-support neural dual certificates."""
    torch = model.torch
    model.eval()
    with torch.no_grad():
        upper, lower, separate = _dual_tensors(model, problem, device)
        upper_values = upper.detach().cpu().numpy().astype(np.float64)
        lower_values = lower.detach().cpu().numpy().astype(np.float64)
    counts = [len(a["outcomes"]) for a in problem["source_atoms"]]
    offsets = np.cumsum([0] + counts)
    j_upper = sum(np.mean(upper_values[offsets[e]:offsets[e + 1]]) for e in range(len(counts)))
    j_lower = sum(np.mean(lower_values[offsets[e]:offsets[e + 1]]) for e in range(len(counts)))
    tuples = problem["prepared"]["feasible_tuples"]
    upper_sum = sum(upper_values[tuples[:, e] + offsets[e]] for e in range(len(counts)))
    lower_sum = sum(lower_values[tuples[:, e] + offsets[e]] for e in range(len(counts)))
    envelope_lower, envelope_upper = empirical_tuple_envelopes(
        problem["prepared"], problem["target_action"]
    )
    eta_upper = float(np.max(np.maximum(0.0, envelope_upper - upper_sum)))
    eta_lower = float(np.max(np.maximum(0.0, lower_sum - envelope_lower)))
    corrected = exact_violation_correction(
        j_upper, j_lower, eta_upper, eta_lower,
        separate["separate_upper"], separate["separate_lower"]
    )
    values = {
        "raw_upper_objective": float(j_upper), "raw_lower_objective": float(j_lower),
        "raw_upper_violation": eta_upper, "raw_lower_violation": eta_lower,
        **corrected,
        "fallback_upper": corrected["neural_upper_final"] >= separate["separate_upper"] - 1e-8,
        "fallback_lower": corrected["neural_lower_final"] <= separate["separate_lower"] + 1e-8,
    }
    values["corrected_upper_violation"] = max(0.0, values["raw_upper_violation"]
                                                - values["raw_upper_violation"])
    values["corrected_lower_violation"] = max(0.0, values["raw_lower_violation"]
                                                - values["raw_lower_violation"])
    return values


def train_dual_residual_net(
    problems: list[dict[str, Any]], steps: int, batch_size: int, seed: int, device: str,
    audit_problems: list[dict[str, Any]] | None = None,
) -> tuple[DualResidualNet, dict[str, np.ndarray]]:
    if not problems:
        raise RuntimeError("no feasible train problems for neural training")
    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = DualResidualNet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    curve_step, train_loss, train_objective, snapshots = [], [], [], []
    log_every = max(1, steps // 20)
    for step in range(steps):
        model.train()
        chosen = rng.choice(len(problems), size=min(batch_size, len(problems)), replace=False)
        losses, objectives = [], []
        for index in chosen:
            problem = problems[int(index)]
            upper, lower, _ = _dual_tensors(model, problem, device)
            counts = [len(a["outcomes"]) for a in problem["source_atoms"]]
            offsets = np.cumsum([0] + counts)
            j_upper = sum(upper[offsets[e]:offsets[e + 1]].mean() for e in range(len(counts)))
            j_lower = sum(lower[offsets[e]:offsets[e + 1]].mean() for e in range(len(counts)))
            tuples = problem["prepared"]["feasible_tuples"]
            upper_parts, lower_parts = [], []
            for source_index in range(len(counts)):
                indices = torch.as_tensor(tuples[:, source_index] + offsets[source_index],
                                          dtype=torch.long, device=device)
                upper_parts.append(upper[indices])
                lower_parts.append(lower[indices])
            envelope_lower, envelope_upper = empirical_tuple_envelopes(
                problem["prepared"], problem["target_action"]
            )
            envelope_upper = torch.as_tensor(envelope_upper, dtype=torch.float32, device=device)
            envelope_lower = torch.as_tensor(envelope_lower, dtype=torch.float32, device=device)
            upper_violation = torch.relu(envelope_upper - torch.stack(upper_parts).sum(dim=0))
            lower_violation = torch.relu(torch.stack(lower_parts).sum(dim=0) - envelope_lower)
            losses.append(j_upper - j_lower + 100 * upper_violation.square().mean()
                          + 100 * lower_violation.square().mean()
                          + 10 * upper_violation.max() + 10 * lower_violation.max())
            objectives.append(j_upper - j_lower)
        parameter_l2 = sum(parameter.square().sum() for parameter in model.parameters())
        loss = torch.stack(losses).mean() + 1e-5 * parameter_l2
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), 10.0)
        optimizer.step()
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            curve_step.append(step + 1)
            train_loss.append(float(loss.detach().cpu()))
            train_objective.append(float(torch.stack(objectives).mean().detach().cpu()))
            snapshots.append({name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()})
            print(f"dual step {step + 1}/{steps}: loss={train_loss[-1]:.6g} ",
                  f"objective={train_objective[-1]:.6g}", flush=True)
    audit_objective = []
    sample = (audit_problems or [])[:min(32, len(audit_problems or []))]
    for state_dict in snapshots:
        model.load_state_dict(state_dict)
        values = [neural_problem_values(model, problem, device) for problem in sample]
        audit_objective.append(float(np.mean([
            value["raw_upper_objective"] - value["raw_lower_objective"] for value in values
        ])) if values else np.nan)
    model.load_state_dict(snapshots[-1])
    curves = {"step": np.asarray(curve_step), "train_loss": np.asarray(train_loss),
              "train_objective": np.asarray(train_objective),
              "audit_objective": np.asarray(audit_objective)}
    return model, curves
