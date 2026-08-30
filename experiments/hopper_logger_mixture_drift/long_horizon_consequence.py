"""True-Hopper rollout primitives for the Phase 8A-NC long-horizon audit."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from confounded_hopper import ACTUATOR_DIRECTION, ConfoundedHopperWrapper
from .anchor_pool import restore_anchor
from .fixed_public_continuation import FixedPublicContinuationPolicy
from .noncomplementary_population_dgp import (
    ACTION_KEYS,
    PRIMARY_MIXTURES,
    analytic_u_posterior,
)


ALLOWED_HORIZONS = (1, 5, 20, 50)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_KEYS)}
U_VALUES = (-1, 1)
U_INDEX = {-1: 0, 1: 1}
BRANCH_FIELDS = (
    "return_mean", "return_standard_error", "survival_probability",
    "termination_probability", "truncation_probability",
    "restricted_time_to_termination", "future_clipping_rate",
    "future_clipping_coordinate_rate",
)


class LongHorizonAuditError(RuntimeError):
    """Raised when rollout or aggregation semantics are violated."""


def exact_horizon5_sequences() -> np.ndarray:
    return np.asarray(tuple(product((-1, 1), repeat=4)), dtype=np.int8)


def generate_future_u_sequences(
    anchor_ids: np.ndarray, replicates: int, maximum_future_steps: int, seed: int,
) -> np.ndarray:
    """Generate reproducible, anchor-specific antithetic future-U sequences."""
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    if replicates <= 0 or replicates % 2:
        raise ValueError("rollout replicates must be positive and even")
    if maximum_future_steps < 0:
        raise ValueError("maximum_future_steps must be nonnegative")
    result = np.empty((len(anchors), replicates, maximum_future_steps), dtype=np.int8)
    half = replicates // 2
    for row, anchor in enumerate(anchors):
        for replicate in range(half):
            generator = np.random.default_rng(
                np.random.SeedSequence((int(seed), int(anchor), int(replicate)))
            )
            sequence = 2 * generator.integers(
                0, 2, size=maximum_future_steps, dtype=np.int8
            ) - 1
            result[row, replicate] = sequence
            result[row, replicate + half] = -sequence
    return result


def horizon_eligibility(elapsed_steps: np.ndarray, horizons: tuple[int, ...],
                        time_limit: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elapsed = np.asarray(elapsed_steps, dtype=np.int64)
    if elapsed.ndim != 1 or np.any(elapsed < 0) or np.any(elapsed >= time_limit):
        raise ValueError("elapsed_steps must be a valid TimeLimit vector")
    remaining = time_limit - elapsed
    eligible = remaining[:, None] >= np.asarray(horizons, dtype=np.int64)[None, :]
    common = remaining >= max(horizons)
    return remaining, eligible, common


def top_action_masks(values: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("action values must be a finite [N, 3] array")
    maximum = np.max(array, axis=1, keepdims=True)
    tied = np.isclose(array, maximum, atol=atol, rtol=rtol)
    return np.sum(tied.astype(np.uint8) * (1 << np.arange(3, dtype=np.uint8)), axis=1)


def decision_regret(
    do_values: np.ndarray, observational_top_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(do_values, dtype=np.float64)
    masks = np.asarray(observational_top_masks, dtype=np.uint8)
    if values.shape != (len(masks), 3):
        raise ValueError("do values and top-action masks are not aligned")
    best = np.max(values, axis=1)
    selected_best = np.empty(len(masks), dtype=np.float64)
    selected_worst = np.empty(len(masks), dtype=np.float64)
    for row, mask in enumerate(masks):
        selected = np.asarray([bool(mask & (1 << action)) for action in range(3)])
        if not np.any(selected):
            raise LongHorizonAuditError("observational top-action set is empty")
        selected_best[row] = np.max(values[row, selected])
        selected_worst[row] = np.min(values[row, selected])
    best_case = best - selected_best
    worst_case = best - selected_worst
    if np.any(best_case < -1e-12) or np.any(worst_case < -1e-12):
        raise LongHorizonAuditError("decision regret is negative")
    return np.maximum(best_case, 0.0), np.maximum(worst_case, 0.0)


def posterior_matrix() -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(PRIMARY_MIXTURES)
    matrix = np.asarray([
        [analytic_u_posterior("confounded", mixture, action) for action in ACTION_KEYS]
        for mixture in names
    ], dtype=np.float64)
    return names, matrix


def combine_initial_u_branches(
    branches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Linearly construct do, observational, and independent fixed-policy values."""
    values = np.asarray(branches, dtype=np.float64)
    if values.shape[-2:] != (3, 2) or not np.all(np.isfinite(values)):
        raise ValueError("branch array must end in [action=3, initial_u=2]")
    names, posterior = posterior_matrix()
    do = np.mean(values, axis=-1)
    observational = np.empty(values.shape[:-1] + (len(names),), dtype=np.float64)
    for mixture_index in range(len(names)):
        p_plus = posterior[mixture_index]
        observational[..., mixture_index] = (
            values[..., 1] * p_plus + values[..., 0] * (1.0 - p_plus)
        )
    independent = np.repeat(do[..., None], len(names), axis=-1)
    return do, observational, names, independent


def verify_long_horizon_identities(
    branches: np.ndarray, do: np.ndarray, observational: np.ndarray,
    mixture_names: tuple[str, ...], atol: float, rtol: float,
) -> dict[str, float]:
    values = np.asarray(branches, dtype=np.float64)
    difference = values[..., 1] - values[..., 0]
    balanced = mixture_names.index("logger12_balanced")
    logger1 = mixture_names.index("logger1_heavy")
    logger2 = mixture_names.index("logger2_heavy")
    signs = np.asarray((-1.0, 0.0, 1.0), dtype=np.float64)
    expected_balanced = 0.3 * difference * signs
    expected_heavy = (7.0 / 45.0) * difference * signs
    balanced_residual = observational[..., balanced] - do - expected_balanced
    heavy_residual = (observational[..., logger1] - observational[..., logger2]
                      - expected_heavy)
    result = {
        "balanced_maximum_absolute_residual": float(np.max(np.abs(balanced_residual))),
        "heavy_maximum_absolute_residual": float(np.max(np.abs(heavy_residual))),
    }
    if not (np.allclose(balanced_residual, 0.0, atol=atol, rtol=rtol)
            and np.allclose(heavy_residual, 0.0, atol=atol, rtol=rtol)):
        raise LongHorizonAuditError("long-horizon posterior identities failed")
    return result


def _make_environment(kappa: float) -> ConfoundedHopperWrapper:
    return ConfoundedHopperWrapper(
        gym.make("Hopper-v5"), kappa=kappa, expose_confounder=False, audit_info=True
    )


class LongHorizonRolloutEngine:
    """Restore fixed anchors and perform intervention rollouts in true Hopper."""

    def __init__(
        self, anchors: dict[str, np.ndarray], raw_by_kappa: dict[float, dict[str, np.ndarray]],
        policy: FixedPublicContinuationPolicy,
        environment_factory: Callable[[float], ConfoundedHopperWrapper] = _make_environment,
        atol: float = 1e-7, rtol: float = 1e-7,
    ) -> None:
        self.anchors = anchors
        self.raw_by_kappa = raw_by_kappa
        self.policy = policy
        self.atol, self.rtol = float(atol), float(rtol)
        self.environment = environment_factory(next(iter(raw_by_kappa)))
        # Unlock Gymnasium's OrderEnforcing wrapper once. Every audited branch
        # immediately overwrites this reset state by restoring its full anchor.
        self.environment.reset(seed=0)
        self.lookups: dict[float, dict[tuple[int, str, int], int]] = {}
        for kappa, raw in raw_by_kappa.items():
            lookup: dict[tuple[int, str, int], int] = {}
            for row in range(len(raw["anchor_id"])):
                key = (int(raw["anchor_id"][row]), str(raw["action_key"][row]),
                       int(raw["u_env"][row]))
                if key in lookup:
                    raise LongHorizonAuditError("duplicate Phase 8A do-oracle key")
                lookup[key] = row
            self.lookups[kappa] = lookup

    def close(self) -> None:
        self.environment.close()

    def _validate_first_step(
        self, anchor_id: int, action_key: str, u0: int, kappa: float,
        command: np.ndarray, observation: np.ndarray, reward: float,
        terminated: bool, truncated: bool, info: dict[str, Any],
    ) -> None:
        raw = self.raw_by_kappa[kappa]
        row = self.lookups[kappa].get((anchor_id, action_key, u0))
        if row is None:
            raise LongHorizonAuditError("Phase 8A first-step oracle key is missing")
        checks = (
            np.allclose(command, raw["commanded_action"][row], atol=self.atol, rtol=self.rtol),
            np.allclose(info["applied_action"], raw["applied_action"][row],
                        atol=self.atol, rtol=self.rtol),
            np.isclose(reward, raw["reward"][row], atol=self.atol, rtol=self.rtol),
            np.allclose(observation, raw["next_observation"][row],
                        atol=self.atol, rtol=self.rtol),
            terminated == bool(raw["terminated"][row]),
            truncated == bool(raw["truncated"][row]),
        )
        if not all(checks):
            raise LongHorizonAuditError("true Hopper first step differs from Phase 8A do oracle")

    def rollout(
        self, anchor_index: int, kappa: float, action_key: str, u0: int,
        future_u: np.ndarray, target_horizons: tuple[int, ...], gamma: float,
        verify_first_step: bool = False,
    ) -> dict[str, np.ndarray]:
        if action_key not in ACTION_INDEX or u0 not in U_INDEX:
            raise ValueError("invalid first action or initial U")
        targets = tuple(sorted(set(int(value) for value in target_horizons)))
        if not targets or targets[0] < 1 or targets[-1] > len(future_u) + 1:
            raise ValueError("future-U sequence is too short for target horizons")
        anchor_id = int(self.anchors["anchor_id"][anchor_index])
        row = self.lookups[kappa][(anchor_id, action_key, u0)]
        command = np.asarray(
            self.raw_by_kappa[kappa]["commanded_action"][row], dtype=np.float64)
        self.environment.kappa = float(kappa)
        restore_anchor(self.environment, self.anchors, anchor_index, self.atol, self.rtol)
        self.environment._hidden_u = int(u0)
        observation, reward, terminated, truncated, info = self.environment.step(command)
        public = self.environment.get_public_observation(observation)
        if verify_first_step:
            self._validate_first_step(anchor_id, action_key, u0, kappa, command, public,
                                      reward, terminated, truncated, info)

        maximum = targets[-1]
        cumulative = float(reward)
        ended = bool(terminated or truncated)
        termination_seen, truncation_seen = bool(terminated), bool(truncated)
        event_step = 1 if ended else maximum
        future_steps = future_clipped_steps = future_clipped_coordinates = 0
        snapshots: dict[int, tuple[float, float, float, float, float, float, float, float]] = {}

        def record(horizon: int) -> None:
            denominator = max(future_steps, 1)
            snapshots[horizon] = (
                cumulative, float(not ended), float(termination_seen), float(truncation_seen),
                float(event_step if ended else horizon),
                float(future_clipped_steps / denominator),
                float(future_clipped_coordinates / (3 * denominator)),
                float(future_steps),
            )

        if 1 in targets:
            record(1)
        for step in range(2, maximum + 1):
            if ended:
                if step in targets:
                    record(step)
                continue
            current_u = int(future_u[step - 2])
            commanded = self.policy.action(public)
            preclip = commanded + float(kappa) * current_u * ACTUATOR_DIRECTION
            clipped_coordinates = int(np.sum((preclip < -1.0) | (preclip > 1.0)))
            self.environment._hidden_u = current_u
            observation, next_reward, terminated, truncated, info = self.environment.step(commanded)
            public = self.environment.get_public_observation(observation)
            cumulative += (gamma ** (step - 1)) * float(next_reward)
            future_steps += 1
            future_clipped_steps += int(clipped_coordinates > 0)
            future_clipped_coordinates += clipped_coordinates
            if terminated or truncated:
                ended = True
                event_step = step
                termination_seen = bool(terminated)
                truncation_seen = bool(truncated)
            if step in targets:
                record(step)

        matrix = np.asarray([snapshots[horizon] for horizon in targets], dtype=np.float64)
        return {
            "horizons": np.asarray(targets, dtype=np.int16),
            "return": matrix[:, 0], "survival": matrix[:, 1],
            "termination": matrix[:, 2], "truncation": matrix[:, 3],
            "restricted_time": matrix[:, 4], "future_clip_rate": matrix[:, 5],
            "future_clip_coordinate_rate": matrix[:, 6], "future_steps": matrix[:, 7],
            "first_step_validated": np.asarray(verify_first_step, dtype=bool),
        }


def integrate_rollouts(
    engine: LongHorizonRolloutEngine, anchor_index: int, kappa: float,
    action_key: str, u0: int, sequences: np.ndarray, horizons: tuple[int, ...],
    gamma: float, exact: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sequence_array = np.asarray(sequences, dtype=np.int8)
    records = [engine.rollout(anchor_index, kappa, action_key, u0, sequence, horizons,
                              gamma, verify_first_step=(index == 0))
               for index, sequence in enumerate(sequence_array)]
    result: dict[str, np.ndarray] = {}
    returns = np.stack([record["return"] for record in records])
    result["return_mean"] = np.mean(returns, axis=0)
    if exact or len(returns) < 4:
        result["return_standard_error"] = np.zeros(len(horizons), dtype=np.float64)
    else:
        half = len(returns) // 2
        pair_means = 0.5 * (returns[:half] + returns[half:])
        result["return_standard_error"] = np.std(
            pair_means, axis=0, ddof=1) / np.sqrt(half)
    for source, target in (
        ("survival", "survival_probability"),
        ("termination", "termination_probability"),
        ("truncation", "truncation_probability"),
        ("restricted_time", "restricted_time_to_termination"),
        ("future_clip_rate", "future_clipping_rate"),
        ("future_clip_coordinate_rate", "future_clipping_coordinate_rate"),
    ):
        result[target] = np.mean(np.stack([record[source] for record in records]), axis=0)
    half = len(returns) // 2
    result["first_half_return_mean"] = np.mean(returns[:half], axis=0)
    result["second_half_return_mean"] = np.mean(returns[half:], axis=0)
    result["first_step_validated"] = np.asarray(True)
    return result, returns


_WORKER: dict[str, Any] = {}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _worker_initialize(config: dict[str, Any]) -> None:
    from stable_baselines3 import SAC

    anchors = _load_npz(Path(config["anchors_path"]))
    raws = {float(kappa): _load_npz(Path(path))
            for kappa, path in config["raw_paths"].items()}
    model = SAC.load(config["checkpoint_path"], device=config["device"])
    policy = FixedPublicContinuationPolicy(model)
    engine = LongHorizonRolloutEngine(
        anchors, raws, policy, atol=config["atol"], rtol=config["rtol"])
    _WORKER.clear()
    _WORKER.update(config=config, anchors=anchors, engine=engine,
                   exact_h5=np.asarray(config["exact_h5"], dtype=np.int8),
                   future_u=np.asarray(config["future_u"], dtype=np.int8))


def _rollout_anchor_worker(position: int) -> tuple[int, dict[str, np.ndarray]]:
    config, engine = _WORKER["config"], _WORKER["engine"]
    kappas, horizons = tuple(config["kappas"]), tuple(config["horizons"])
    gamma, replicates = float(config["gamma"]), int(config["replicates"])
    remaining = int(config["remaining"][position])
    shape = (len(kappas), len(horizons), 3, 2)
    output = {field: np.zeros(shape, dtype=np.float64) for field in BRANCH_FIELDS}
    output["first_step_validated"] = np.zeros((len(kappas), 3, 2), dtype=bool)
    mc_horizons = tuple(value for value in horizons if value in (20, 50))
    output["replicate_returns"] = np.zeros(
        (len(kappas), len(mc_horizons), 3, 2, replicates), dtype=np.float64)
    output["mc_h5_mean"] = np.zeros((len(kappas), 3, 2), dtype=np.float64)
    output["exact_h5_mean"] = np.zeros((len(kappas), 3, 2), dtype=np.float64)
    output["mc_h5_available"] = np.zeros((len(kappas), 3, 2), dtype=bool)
    horizon_index = {value: index for index, value in enumerate(horizons)}

    for ki, kappa in enumerate(kappas):
        for action_key in ACTION_KEYS:
            ai = ACTION_INDEX[action_key]
            for u0 in U_VALUES:
                ui = U_INDEX[u0]
                first_assigned = False
                if 5 in horizons and remaining >= 5:
                    exact, _ = integrate_rollouts(
                        engine, position, kappa, action_key, u0, _WORKER["exact_h5"],
                        (1, 5), gamma, exact=True)
                    for field in BRANCH_FIELDS:
                        output[field][ki, horizon_index[1], ai, ui] = exact[field][0]
                        output[field][ki, horizon_index[5], ai, ui] = exact[field][1]
                    output["exact_h5_mean"][ki, ai, ui] = exact["return_mean"][1]
                    output["first_step_validated"][ki, ai, ui] = True
                    first_assigned = True

                eligible_mc = tuple(value for value in mc_horizons if remaining >= value)
                if eligible_mc:
                    targets = ((5,) if remaining >= 5 else ()) + eligible_mc
                    integrated, returns = integrate_rollouts(
                        engine, position, kappa, action_key, u0,
                        _WORKER["future_u"][position], targets, gamma, exact=False)
                    target_index = {value: index for index, value in enumerate(targets)}
                    for horizon in eligible_mc:
                        hi, local = horizon_index[horizon], target_index[horizon]
                        for field in BRANCH_FIELDS:
                            output[field][ki, hi, ai, ui] = integrated[field][local]
                        mi = mc_horizons.index(horizon)
                        output["replicate_returns"][ki, mi, ai, ui] = returns[:, local]
                    if 5 in targets:
                        output["mc_h5_mean"][ki, ai, ui] = integrated["return_mean"][0]
                        output["mc_h5_available"][ki, ai, ui] = True
                    output["first_step_validated"][ki, ai, ui] = True
                    if not first_assigned and 1 in horizons:
                        one = engine.rollout(position, kappa, action_key, u0,
                                             np.empty(0, dtype=np.int8), (1,), gamma, True)
                        for field, source in (
                            ("return_mean", "return"), ("survival_probability", "survival"),
                            ("termination_probability", "termination"),
                            ("truncation_probability", "truncation"),
                            ("restricted_time_to_termination", "restricted_time"),
                            ("future_clipping_rate", "future_clip_rate"),
                            ("future_clipping_coordinate_rate", "future_clip_coordinate_rate"),
                        ):
                            output[field][ki, horizon_index[1], ai, ui] = one[source][0]
                        first_assigned = True

                if 1 in horizons and remaining >= 1 and not first_assigned:
                    one = engine.rollout(position, kappa, action_key, u0,
                                         np.empty(0, dtype=np.int8), (1,), gamma, True)
                    for field, source in (
                        ("return_mean", "return"), ("survival_probability", "survival"),
                        ("termination_probability", "termination"),
                        ("truncation_probability", "truncation"),
                        ("restricted_time_to_termination", "restricted_time"),
                        ("future_clipping_rate", "future_clip_rate"),
                        ("future_clipping_coordinate_rate", "future_clip_coordinate_rate"),
                    ):
                        output[field][ki, horizon_index[1], ai, ui] = one[source][0]
                    output["first_step_validated"][ki, ai, ui] = True
    return position, output


def execute_rollouts(
    config: dict[str, Any], num_anchors: int, num_workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Execute independent anchor tasks and assemble finite, mask-aligned arrays."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    assembled: dict[str, np.ndarray] = {}

    def insert(position: int, result: dict[str, np.ndarray]) -> None:
        for field, values in result.items():
            if field not in assembled:
                assembled[field] = np.zeros((num_anchors,) + values.shape, dtype=values.dtype)
            assembled[field][position] = values

    completed = 0
    if num_workers == 1:
        _worker_initialize(config)
        try:
            for position in range(num_anchors):
                row, result = _rollout_anchor_worker(position)
                insert(row, result)
                completed += 1
                if progress is not None:
                    progress(completed, num_anchors)
        finally:
            _WORKER["engine"].close()
            _WORKER.clear()
    else:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_initialize,
                                 initargs=(config,)) as executor:
            futures = [executor.submit(_rollout_anchor_worker, position)
                       for position in range(num_anchors)]
            for future in as_completed(futures):
                row, result = future.result()
                insert(row, result)
                completed += 1
                if progress is not None:
                    progress(completed, num_anchors)
    if completed != num_anchors or not all(
            np.all(np.isfinite(values)) for values in assembled.values()
            if np.issubdtype(values.dtype, np.number)):
        raise LongHorizonAuditError("rollout assembly is incomplete or nonfinite")
    return assembled
