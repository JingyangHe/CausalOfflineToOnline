"""Exact observable-mass AAMAS26 population/oracle analogue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from confounded_smooth_regulator import behavior_action_fn, reward_fn, transition_fn


AAMAS26_POPULATION_MASS_ANALOGUE = "AAMAS26_POPULATION_MASS_ANALOGUE"
ACTION_ATOL = 1.0e-12
ROAD_SEPARATION = 0.1
DEFAULT_HORIZON = 20
DEFAULT_GAMMA = 0.95
REWARD_UPPER_BOUND = 1.0
AAMAS26_CONTINUOUS_DENSITY_REPLACED_BY_POPULATION_MASS = True
AAMAS26_K25_MONTE_CARLO_ERROR_REMOVED = True
AAMAS26_NEURAL_APPROXIMATION_ERROR_REMOVED = True


class UnsupportedRoadSupportError(RuntimeError):
    """An observed action has no legal ROAD comparison action."""


@dataclass(frozen=True)
class ObservablePopulation:
    """Public joint law of ``(A, R, S')`` at a fixed state."""

    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    probabilities: np.ndarray
    action_atoms: np.ndarray
    action_masses: np.ndarray
    conditional_rewards: tuple[np.ndarray, ...]
    conditional_next_states: tuple[np.ndarray, ...]
    conditional_probabilities: tuple[np.ndarray, ...]
    primitive_support_size: int
    merged_action_support_size: int


def _probabilities(values: np.ndarray | list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("probabilities must be a nonempty vector")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("probabilities must be finite and positive")
    if not np.isclose(np.sum(result), 1.0, atol=ACTION_ATOL, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    return result


def _unique_actions(actions: np.ndarray) -> np.ndarray:
    atoms: list[float] = []
    for action in np.asarray(actions, dtype=np.float64):
        if not any(np.isclose(action, old, atol=ACTION_ATOL, rtol=0.0) for old in atoms):
            atoms.append(float(action))
    return np.asarray(atoms, dtype=np.float64)


def _action_indices(actions: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    result = np.empty(actions.size, dtype=np.int64)
    for index, action in enumerate(actions):
        matches = np.flatnonzero(np.isclose(atoms, action, atol=ACTION_ATOL, rtol=0.0))
        if matches.size != 1:
            raise ValueError("each outcome must match exactly one action atom")
        result[index] = matches[0]
    return result


def build_pooled_observable_population(
    state: float, kappa: float = 1.0
) -> ObservablePopulation:
    """Build the equal-pool public law, stripping all latent/source labels."""
    state = float(state)
    if not np.isfinite(state) or not -1.0 <= state <= 1.0:
        raise ValueError("state must be finite and in [-1, 1]")
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite")

    primitive: list[tuple[float, float, float, float]] = []
    for source_index in (1, 2, 3):
        for structural_sign in (-1, 1):
            for behavior_sign in (-1, 1):
                action = behavior_action_fn(
                    source_index, state, (structural_sign, behavior_sign), kappa
                )
                next_state = transition_fn(state, action, structural_sign)
                reward = reward_fn(state, action, structural_sign, next_state)
                primitive.append((action, reward, next_state, 1.0 / 12.0))

    merged: list[list[float]] = []
    for action, reward, next_state, mass in primitive:
        matching = None
        for index, old in enumerate(merged):
            if all(
                np.isclose(new, prior, atol=ACTION_ATOL, rtol=0.0)
                for new, prior in zip((action, reward, next_state), old[:3], strict=True)
            ):
                matching = index
                break
        if matching is None:
            merged.append([action, reward, next_state, mass])
        else:
            merged[matching][3] += mass
    values = np.asarray(merged, dtype=np.float64)
    atoms = _unique_actions(values[:, 0])
    indices = _action_indices(values[:, 0], atoms)
    masses = np.bincount(indices, weights=values[:, 3], minlength=atoms.size)
    masks = tuple(indices == index for index in range(atoms.size))
    return ObservablePopulation(
        actions=values[:, 0], rewards=values[:, 1], next_states=values[:, 2],
        probabilities=_probabilities(values[:, 3]),
        action_atoms=atoms, action_masses=masses,
        conditional_rewards=tuple(values[mask, 1] for mask in masks),
        conditional_next_states=tuple(values[mask, 2] for mask in masks),
        conditional_probabilities=tuple(values[mask, 3] / masses[index]
                                        for index, mask in enumerate(masks)),
        primitive_support_size=len(primitive),
        merged_action_support_size=atoms.size,
    )


def observable_action_summary(
    state: float, actions: np.ndarray, next_states: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute public action masses and ``E[S'-s | s,A=a]``."""
    actions = np.asarray(actions, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    probabilities = _probabilities(probabilities)
    if actions.ndim != 1 or next_states.shape != actions.shape:
        raise ValueError("actions and next_states must be aligned vectors")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(next_states)):
        raise ValueError("observable outcomes must be finite")
    atoms = _unique_actions(actions)
    indices = _action_indices(actions, atoms)
    masses = np.bincount(indices, weights=probabilities, minlength=atoms.size)
    increments = np.bincount(
        indices, weights=probabilities * (next_states - float(state)), minlength=atoms.size
    )
    return {
        "action_atoms": atoms, "action_masses": masses,
        "outcome_action_indices": indices, "delta_means": increments / masses,
    }


def road_candidate_mask(observed_action: float, action_atoms: np.ndarray) -> np.ndarray:
    """Apply the official one-dimensional ROAD separation rule."""
    return np.abs(np.asarray(action_atoms) - observed_action) >= ROAD_SEPARATION


def population_behavior_weights(
    action_atoms: np.ndarray, action_masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``p(a)/(p(a)+exp(E_road[log p(a')]))`` exactly."""
    action_atoms = np.asarray(action_atoms, dtype=np.float64)
    action_masses = np.asarray(action_masses, dtype=np.float64)
    if action_atoms.ndim != 1 or action_masses.shape != action_atoms.shape:
        raise ValueError("action atoms and masses must be aligned vectors")
    if np.any(action_masses <= 0.0) or not np.all(np.isfinite(action_masses)):
        raise ValueError("action masses must be finite and positive")
    weights = np.empty(action_atoms.size, dtype=np.float64)
    counts = np.empty(action_atoms.size, dtype=np.int64)
    for index, observed_action in enumerate(action_atoms):
        candidate = road_candidate_mask(observed_action, action_atoms)
        counts[index] = np.count_nonzero(candidate)
        if counts[index] == 0:
            raise UnsupportedRoadSupportError(
                f"action {observed_action:.17g} has no ROAD candidate"
            )
        candidate_mass = action_masses[candidate]
        normalized_mass = candidate_mass / np.sum(candidate_mass)
        q_road = np.exp(np.sum(normalized_mass * np.log(candidate_mass)))
        weights[index] = action_masses[index] / (action_masses[index] + q_road)
    return weights, counts


def aamas_population_state_target(
    state: float, stage: int, actions: np.ndarray, rewards: np.ndarray,
    next_states: np.ndarray, probabilities: np.ndarray, state_grid: np.ndarray,
    continuation: np.ndarray, gamma: float = DEFAULT_GAMMA,
) -> dict[str, np.ndarray | float]:
    """Apply the observable fixed-state population mass operator."""
    if not isinstance(stage, (int, np.integer)) or isinstance(stage, (bool, np.bool_)):
        raise ValueError("stage must be a positive integer")
    if stage < 1 or not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("stage/gamma configuration is invalid")
    arrays = [np.asarray(x, dtype=np.float64) for x in (actions, rewards, next_states)]
    probabilities = _probabilities(probabilities)
    if any(x.ndim != 1 for x in arrays) or len({x.size for x in arrays}) != 1:
        raise ValueError("observable outcomes must be aligned vectors")
    if arrays[0].size != probabilities.size or not all(np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("observable outcomes must be aligned and finite")
    actions, rewards, next_states = arrays
    state_grid = np.asarray(state_grid, dtype=np.float64)
    continuation = np.asarray(continuation, dtype=np.float64)
    if state_grid.ndim != 1 or continuation.shape != state_grid.shape:
        raise ValueError("state_grid and continuation must be aligned vectors")
    if state_grid.size < 2 or np.any(np.diff(state_grid) <= 0.0):
        raise ValueError("state_grid must be strictly increasing")
    if not np.all(np.isfinite(state_grid)) or not np.all(np.isfinite(continuation)):
        raise ValueError("state_grid and continuation must be finite")

    summary = observable_action_summary(state, actions, next_states, probabilities)
    atoms, masses = summary["action_atoms"], summary["action_masses"]
    indices = summary["outcome_action_indices"]
    weights, counts = population_behavior_weights(atoms, masses)
    predicted_next = float(state) + summary["delta_means"]
    predicted_values = np.interp(predicted_next, state_grid, continuation)
    road_bases = np.asarray([
        np.max(predicted_values[road_candidate_mask(action, atoms)]) for action in atoms
    ])
    observed_values = np.interp(next_states, state_grid, continuation)
    observed_returns = rewards + gamma * observed_values
    road_values = np.maximum(road_bases[indices], observed_values)
    road_returns = REWARD_UPPER_BOUND + gamma * road_values
    outcome_weights = weights[indices]
    targets = outcome_weights * observed_returns + (1.0 - outcome_weights) * road_returns
    return {
        "potential": float(np.sum(probabilities * targets)),
        "action_atoms": atoms, "action_masses": masses,
        "behavior_weights": weights, "road_candidate_counts": counts,
        "delta_means": summary["delta_means"],
    }


def solve_aamas_population_mass_analogue(
    state_grid: np.ndarray, horizon: int = DEFAULT_HORIZON,
    gamma: float = DEFAULT_GAMMA, kappa: float = 1.0,
) -> dict[str, object]:
    """Run the backward recursion with terminal ``Phi[H+1]=0``."""
    state_grid = np.asarray(state_grid, dtype=np.float64)
    grid_ok = (
        state_grid.ndim == 1 and state_grid.size >= 2
        and np.all(np.isfinite(state_grid)) and np.all(np.diff(state_grid) > 0.0)
        and state_grid[0] >= -1.0 and state_grid[-1] <= 1.0
    )
    if not grid_ok:
        raise ValueError("state_grid must be increasing and contained in [-1, 1]")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, (bool, np.bool_)):
        raise ValueError("horizon must be a positive integer")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    populations = [build_pooled_observable_population(state, kappa) for state in state_grid]
    maximum_support = max(item.merged_action_support_size for item in populations)
    shape = (state_grid.size, maximum_support)
    action_atoms, action_masses = np.full(shape, np.nan), np.full(shape, np.nan)
    behavior_weights = np.full(shape, np.nan)
    road_counts = np.full(shape, -1, dtype=np.int64)
    primitive_sizes = np.asarray([item.primitive_support_size for item in populations])
    merged_sizes = np.asarray([item.merged_action_support_size for item in populations])
    phi = np.zeros((horizon + 2, state_grid.size), dtype=np.float64)
    for stage in range(horizon, 0, -1):
        for state_index, (state, population) in enumerate(zip(state_grid, populations, strict=True)):
            result = aamas_population_state_target(
                state, stage, population.actions, population.rewards,
                population.next_states, population.probabilities,
                state_grid, phi[stage + 1], gamma,
            )
            phi[stage, state_index] = result["potential"]
            if stage == horizon:
                size = population.merged_action_support_size
                action_atoms[state_index, :size] = result["action_atoms"]
                action_masses[state_index, :size] = result["action_masses"]
                behavior_weights[state_index, :size] = result["behavior_weights"]
                road_counts[state_index, :size] = result["road_candidate_counts"]
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("population recursion produced nonfinite values")
    return {
        "method": AAMAS26_POPULATION_MASS_ANALOGUE,
        "state_grid": state_grid, "horizon": int(horizon), "gamma": float(gamma),
        "phi": phi, "primitive_support_sizes": primitive_sizes,
        "merged_action_support_sizes": merged_sizes, "action_atoms": action_atoms,
        "action_masses": action_masses, "behavior_weights": behavior_weights,
        "road_candidate_counts": road_counts,
        "empty_road_support_count": int(np.count_nonzero(road_counts == 0)),
        "uses_neural_network": False, "uses_hidden_operator_input": False,
        "uses_joint_coupling": False, "uses_tuned_road_scale": False,
        "AAMAS26_CONTINUOUS_DENSITY_REPLACED_BY_POPULATION_MASS": True,
        "AAMAS26_K25_MONTE_CARLO_ERROR_REMOVED": True,
        "AAMAS26_NEURAL_APPROXIMATION_ERROR_REMOVED": True,
    }
