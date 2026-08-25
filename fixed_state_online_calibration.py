"""Anytime-valid online calibration at one fixed stage and state."""

import math
import numpy as np

def _radius_unchecked(n, value_range, delta_online, n_actions):
    log_term = math.log(2.0 * n_actions * n * (n + 1) / delta_online)
    return value_range * math.sqrt(log_term / (2.0 * n))

def anytime_hoeffding_radius(
    n: int, value_range: float, delta_online: float, n_actions: int
) -> float:
    """Return the union-bound anytime Hoeffding radius for one action and count."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(value_range) or value_range < 0.0:
        raise ValueError("value_range must be finite and nonnegative")
    if not np.isfinite(delta_online) or not 0.0 < delta_online < 1.0:
        raise ValueError("delta_online must be in (0, 1)")
    if (
        not isinstance(n_actions, (int, np.integer))
        or isinstance(n_actions, (bool, np.bool_))
        or n_actions < 1
    ):
        raise ValueError("n_actions must be a positive integer")
    return float(_radius_unchecked(int(n), value_range, delta_online, int(n_actions)))

def update_online_interval(
    count: int,
    sample_mean: float,
    b_lower: float,
    b_upper: float,
    delta_online: float,
    n_actions: int,
) -> tuple[float, float]:
    """Return the clipped online interval, including the unsampled case."""
    b_lower, b_upper = float(b_lower), float(b_upper)
    if not np.isfinite(b_lower) or not np.isfinite(b_upper) or b_lower > b_upper:
        raise ValueError("invalid global value bounds")
    if not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)):
        raise ValueError("count must be a nonnegative integer")
    if count < 0:
        raise ValueError("count must be a nonnegative integer")
    if count == 0:
        return b_lower, b_upper
    if not np.isfinite(sample_mean):
        raise ValueError("sample_mean must be finite after sampling")
    radius = anytime_hoeffding_radius(
        int(count), b_upper - b_lower, delta_online, n_actions
    )
    return max(b_lower, float(sample_mean) - radius), min(
        b_upper, float(sample_mean) + radius
    )

def intersect_offline_online_interval(
    offline_lower: float,
    offline_upper: float,
    online_lower: float,
    online_upper: float,
    b_lower: float,
    b_upper: float,
) -> tuple[float, float, bool]:
    """Intersect clipped intervals, falling back to online on an empty intersection."""
    values = np.asarray(
        [offline_lower, offline_upper, online_lower, online_upper, b_lower, b_upper],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or b_lower > b_upper:
        raise ValueError("interval endpoints must be finite and global bounds ordered")
    off_lower = float(np.clip(offline_lower, b_lower, b_upper))
    off_upper = float(np.clip(offline_upper, b_lower, b_upper))
    on_lower = float(np.clip(online_lower, b_lower, b_upper))
    on_upper = float(np.clip(online_upper, b_lower, b_upper))
    if off_lower > off_upper or on_lower > on_upper:
        raise ValueError("input intervals must be ordered after clipping")
    lower, upper = max(off_lower, on_lower), min(off_upper, on_upper)
    if lower > upper:
        return on_lower, on_upper, True
    return lower, upper, False

def select_best_and_challenger(lower, upper):
    """Return the lower-bound best and strongest distinct upper-bound challenger."""
    best = max(range(len(lower)), key=lower.__getitem__)
    if len(lower) == 1:
        return best, None
    challenger = 0 if best != 0 else 1
    for index in range(len(lower)):
        if index != best and upper[index] > upper[challenger]:
            challenger = index
    return best, challenger

def select_next_calibration_action(lower, upper, counts, best, challenger, rng):
    """Apply the Phase 4A width, count, and random tie-breaking rule."""
    best_width = upper[best] - lower[best]
    challenger_width = upper[challenger] - lower[challenger]
    if best_width > challenger_width + 1e-15:
        return best
    if challenger_width > best_width + 1e-15:
        return challenger
    if counts[best] < counts[challenger]:
        return best
    if counts[challenger] < counts[best]:
        return challenger
    return best if int(rng.integers(2)) == 0 else challenger

def _update_calibration_action(
    chosen, outcome, counts, sums, lower, upper, offline_lower, offline_upper,
    offline_active, b_lower, b_upper, delta_online, n_actions,
):
    counts[chosen] += 1
    sums[chosen] += outcome
    mean = sums[chosen] / counts[chosen]
    radius = _radius_unchecked(counts[chosen], b_upper - b_lower, delta_online, n_actions)
    online_lower = max(b_lower, mean - radius)
    online_upper = min(b_upper, mean + radius)
    conflict = False
    if offline_active[chosen]:
        new_lower = max(offline_lower[chosen], online_lower)
        new_upper = min(offline_upper[chosen], online_upper)
        conflict = new_lower > new_upper
        if conflict:
            new_lower, new_upper = online_lower, online_upper
            offline_active[chosen] = False
    else:
        new_lower, new_upper = online_lower, online_upper
    lower[chosen], upper[chosen] = new_lower, new_upper
    violation = new_lower < b_lower or new_upper > b_upper or new_lower > new_upper
    return conflict, int(violation)

def _snapshot(interaction, counts, sums, lower, upper, offline_active):
    counts = np.asarray(counts, dtype=np.int64)
    sums = np.asarray(sums, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    means = np.divide(
        sums,
        counts,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    best = int(np.argmax(lower))
    admissible = np.flatnonzero(upper >= lower[best])
    return {
        "online_interactions": int(interaction),
        "counts": counts.copy(),
        "sample_means": means,
        "lower": lower.copy(),
        "upper": upper.copy(),
        "admissible_action_indices": admissible,
        "offline_active": np.asarray(offline_active, dtype=bool),
    }

def run_fixed_state_online_calibration(
    actions,
    initial_lower,
    initial_upper,
    sample_action_fn,
    b_lower: float,
    b_upper: float,
    delta_online: float,
    max_online_interactions: int,
    seed: int,
    record_history: bool = False,
    epsilon: float = 0.0,
) -> dict:
    """Calibrate fixed-state action intervals without access to hidden or oracle data."""
    actions = np.asarray(actions, dtype=np.float64)
    lower_array = np.asarray(initial_lower, dtype=np.float64).copy()
    upper_array = np.asarray(initial_upper, dtype=np.float64).copy()
    if actions.ndim != 1 or not len(actions) or not np.all(np.isfinite(actions)):
        raise ValueError("actions must be a nonempty finite one-dimensional array")
    if len(np.unique(actions)) != len(actions):
        raise ValueError("actions must be unique")
    if lower_array.shape != actions.shape or upper_array.shape != actions.shape:
        raise ValueError("initial bounds must match actions")
    b_lower, b_upper = float(b_lower), float(b_upper)
    if not np.isfinite(b_lower) or not np.isfinite(b_upper) or b_lower > b_upper:
        raise ValueError("invalid global value bounds")
    if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
        raise ValueError("initial bounds must be finite")
    lower_array = np.clip(lower_array, b_lower, b_upper)
    upper_array = np.clip(upper_array, b_lower, b_upper)
    if np.any(lower_array > upper_array):
        raise ValueError("initial intervals must be ordered after clipping")
    if not callable(sample_action_fn):
        raise TypeError("sample_action_fn must be callable")
    if (
        not isinstance(max_online_interactions, (int, np.integer))
        or isinstance(max_online_interactions, (bool, np.bool_))
        or max_online_interactions < 0
    ):
        raise ValueError("max_online_interactions must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    # Validate the confidence inputs even if the initial intervals certify at time zero.
    anytime_hoeffding_radius(1, b_upper - b_lower, delta_online, len(actions))

    lower, upper = lower_array.tolist(), upper_array.tolist()
    offline_lower, offline_upper = lower.copy(), upper.copy()
    offline_active = [True] * len(actions)
    counts = [0] * len(actions)
    sums = [0.0] * len(actions)
    rng = np.random.default_rng(int(seed))
    sampled_indices, conflict_actions, conflict_times = [], [], []
    history = [_snapshot(0, counts, sums, lower, upper, offline_active)] if record_history else None
    certified, certified_index = False, None
    interval_violation_count = 0

    for interaction in range(int(max_online_interactions) + 1):
        best, challenger = select_best_and_challenger(lower, upper)
        if challenger is None:
            certified, certified_index = True, best
            break
        if lower[best] + float(epsilon) >= upper[challenger]:
            certified, certified_index = True, best
            break
        if interaction == max_online_interactions:
            break

        chosen = select_next_calibration_action(
            lower, upper, counts, best, challenger, rng
        )
        outcome = float(sample_action_fn(float(actions[chosen])))
        if not np.isfinite(outcome) or not b_lower <= outcome <= b_upper:
            raise ValueError("online outcome must be finite and within global bounds")
        conflict, violation = _update_calibration_action(
            chosen, outcome, counts, sums, lower, upper, offline_lower, offline_upper,
            offline_active, b_lower, b_upper, delta_online, len(actions),
        )
        if conflict:
            conflict_actions.append(chosen)
            conflict_times.append(interaction + 1)
        interval_violation_count += violation
        sampled_indices.append(chosen)
        if record_history:
            history.append(
                _snapshot(interaction + 1, counts, sums, lower, upper, offline_active)
            )

    counts_array = np.asarray(counts, dtype=np.int64)
    sums_array = np.asarray(sums, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    means = np.divide(
        sums_array, counts_array, out=np.full(len(actions), np.nan), where=counts_array > 0
    )
    best = int(np.argmax(lower_array))
    admissible = np.flatnonzero(upper_array + float(epsilon) >= lower_array[best])
    result = {
        "certified": bool(certified),
        "certified_action_index": certified_index,
        "certified_action": None if certified_index is None else float(actions[certified_index]),
        "online_interactions": len(sampled_indices),
        "counts": counts_array,
        "sample_means": means,
        "final_lower": lower_array,
        "final_upper": upper_array,
        "final_admissible_actions": actions[admissible],
        "final_admissible_action_indices": admissible,
        "conflict_count": len(conflict_actions),
        "conflict_actions": actions[np.asarray(conflict_actions, dtype=np.int64)],
        "conflict_action_indices": np.asarray(conflict_actions, dtype=np.int64),
        "conflict_interactions": np.asarray(conflict_times, dtype=np.int64),
        "sampled_action_indices": np.asarray(sampled_indices, dtype=np.int64),
        "interval_violation_count": interval_violation_count,
    }
    if record_history:
        result["history"] = history
    return result

def run_fixed_budget_online_evaluation(
    actions, initial_lower, initial_upper, sample_action_fn, b_lower, b_upper,
    delta_online, checkpoint_budgets, seed, record_actions=False,
):
    """Run the unchanged Phase 4A policy for an exact fixed online budget."""
    actions = np.asarray(actions, dtype=np.float64)
    lower_array = np.asarray(initial_lower, dtype=np.float64)
    upper_array = np.asarray(initial_upper, dtype=np.float64)
    budgets = np.asarray(checkpoint_budgets)
    if actions.ndim != 1 or not len(actions) or len(np.unique(actions)) != len(actions):
        raise ValueError("actions must be a nonempty unique one-dimensional array")
    if lower_array.shape != actions.shape or upper_array.shape != actions.shape:
        raise ValueError("initial bounds must match actions")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
        raise ValueError("actions and bounds must be finite")
    b_lower, b_upper = float(b_lower), float(b_upper)
    lower_array = np.clip(lower_array, b_lower, b_upper)
    upper_array = np.clip(upper_array, b_lower, b_upper)
    if not np.isfinite(b_lower) or not np.isfinite(b_upper) or b_lower > b_upper or np.any(lower_array > upper_array):
        raise ValueError("invalid or unordered bounds")
    if budgets.ndim != 1 or not len(budgets) or not np.issubdtype(budgets.dtype, np.integer):
        raise ValueError("checkpoint_budgets must be nonempty integers")
    budgets = budgets.astype(np.int64)
    if budgets[0] < 0 or np.any(np.diff(budgets) <= 0):
        raise ValueError("checkpoint_budgets must be nonnegative and strictly increasing")
    if not callable(sample_action_fn):
        raise TypeError("sample_action_fn must be callable")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    anytime_hoeffding_radius(1, b_upper - b_lower, delta_online, len(actions))

    lower, upper = lower_array.tolist(), upper_array.tolist()
    offline_lower, offline_upper = lower.copy(), upper.copy()
    offline_active = [True] * len(actions)
    counts, sums = [0] * len(actions), [0.0] * len(actions)
    rng = np.random.default_rng(int(seed))
    sampled, conflict_indices = [], []
    committed, certification_time, violation_count = None, None, 0
    recommended, certified_flags, count_rows, lower_rows, upper_rows, admissible_rows = [], [], [], [], [], []

    def check_certification(time):
        nonlocal committed, certification_time
        if committed is not None:
            return
        best, challenger = select_best_and_challenger(lower, upper)
        if challenger is None or lower[best] >= upper[challenger]:
            committed, certification_time = best, time

    def record_checkpoint():
        best, _ = select_best_and_challenger(lower, upper)
        recommendation = committed if committed is not None else best
        recommended.append(recommendation)
        certified_flags.append(committed is not None)
        count_rows.append(counts.copy())
        lower_rows.append(lower.copy()); upper_rows.append(upper.copy())
        admissible_rows.append(sum(value >= lower[best] for value in upper))

    check_certification(0)
    checkpoint_index = 0
    if budgets[0] == 0:
        record_checkpoint(); checkpoint_index = 1
    for interaction in range(1, int(budgets[-1]) + 1):
        if committed is None:
            best, challenger = select_best_and_challenger(lower, upper)
            chosen = select_next_calibration_action(
                lower, upper, counts, best, challenger, rng
            )
        else:
            chosen = committed
        outcome = float(sample_action_fn(float(actions[chosen])))
        if not np.isfinite(outcome) or not b_lower <= outcome <= b_upper:
            raise ValueError("online outcome must be finite and within global bounds")
        conflict, violation = _update_calibration_action(
            chosen, outcome, counts, sums, lower, upper, offline_lower, offline_upper,
            offline_active, b_lower, b_upper, delta_online, len(actions),
        )
        if conflict:
            conflict_indices.append(chosen)
        violation_count += violation
        if record_actions:
            sampled.append(chosen)
        check_certification(interaction)
        if checkpoint_index < len(budgets) and interaction == budgets[checkpoint_index]:
            record_checkpoint(); checkpoint_index += 1
    return {
        "checkpoint_budgets": budgets,
        "recommended_action_indices": np.asarray(recommended, dtype=np.int64),
        "recommended_actions": actions[np.asarray(recommended, dtype=np.int64)],
        "certified_at_checkpoint": np.asarray(certified_flags, dtype=bool),
        "committed_action_index": committed,
        "certification_time": certification_time,
        "counts_at_checkpoints": np.asarray(count_rows, dtype=np.int64),
        "lower_at_checkpoints": np.asarray(lower_rows),
        "upper_at_checkpoints": np.asarray(upper_rows),
        "admissible_counts_at_checkpoints": np.asarray(admissible_rows, dtype=np.int64),
        "conflict_count": len(conflict_indices),
        "conflict_actions": actions[np.asarray(conflict_indices, dtype=np.int64)],
        "total_interactions": int(budgets[-1]),
        "interval_violation_count": violation_count,
        **({"sampled_action_indices": np.asarray(sampled, dtype=np.int64)} if record_actions else {}),
    }
