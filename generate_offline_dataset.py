"""Generate separated public training data and privileged audit data."""

from pathlib import Path

import numpy as np

from confounded_smooth_regulator import (
    ConfoundedSmoothRegulatorEnv,
    behavior_action_fn,
    reward_fn,
    sample_exogenous,
    transition_fn,
)


TRAIN_FIELDS = (
    "row_id",
    "source_id",
    "episode_id",
    "time_step",
    "state",
    "action",
    "reward",
    "next_state",
    "terminated",
)
AUDIT_FIELDS = (
    "row_id",
    "source_id",
    "episode_id",
    "time_step",
    "confounder_c",
    "randomizer_w",
)

FIXED_STATE_TRAIN_FIELDS = (
    "row_id",
    "query_id",
    "time_step",
    "state",
    "source_id",
    "sample_index",
    "action",
    "reward",
    "next_state",
)
FIXED_STATE_AUDIT_FIELDS = (
    "row_id",
    "query_id",
    "time_step",
    "state",
    "source_id",
    "sample_index",
    "confounder_c",
    "randomizer_w",
)


def _positive_integer(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def generate_offline_dataset(
    episodes_per_source: int,
    base_seed: int,
    horizon: int = 20,
    gamma: float = 0.95,
    kappa: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate independent trajectories for each of three historical sources."""
    episodes_per_source = _positive_integer("episodes_per_source", episodes_per_source)
    horizon = _positive_integer("horizon", horizon)
    if not isinstance(base_seed, (int, np.integer)) or isinstance(base_seed, (bool, np.bool_)):
        raise ValueError("base_seed must be a nonnegative integer")
    if base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")

    row_count = 3 * episodes_per_source * horizon
    train_data = {
        "row_id": np.arange(row_count, dtype=np.int64),
        "source_id": np.empty(row_count, dtype=np.int8),
        "episode_id": np.empty(row_count, dtype=np.int64),
        "time_step": np.empty(row_count, dtype=np.int16),
        "state": np.empty(row_count, dtype=np.float32),
        "action": np.empty(row_count, dtype=np.float64),
        "reward": np.empty(row_count, dtype=np.float64),
        "next_state": np.empty(row_count, dtype=np.float32),
        "terminated": np.empty(row_count, dtype=np.bool_),
    }
    audit_data = {
        "row_id": train_data["row_id"].copy(),
        "source_id": np.empty(row_count, dtype=np.int8),
        "episode_id": np.empty(row_count, dtype=np.int64),
        "time_step": np.empty(row_count, dtype=np.int16),
        "confounder_c": np.empty(row_count, dtype=np.int8),
        "randomizer_w": np.empty(row_count, dtype=np.int8),
    }

    row = 0
    for source_id in (1, 2, 3):
        env = ConfoundedSmoothRegulatorEnv(horizon=horizon, gamma=gamma, kappa=kappa)
        for episode_id in range(episodes_per_source):
            seed_sequence = np.random.SeedSequence([int(base_seed), source_id, episode_id])
            episode_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            observation, info = env.reset(seed=episode_seed)
            if info:
                raise RuntimeError("reset info unexpectedly contains private data")
            for time_step in range(1, horizon + 1):
                confounder, randomizer = env.get_exogenous_for_audit()
                action = env.privileged_behavior_action(source_id)
                next_observation, reward, terminated, truncated, info = env.step(action)
                if truncated or info:
                    raise RuntimeError("environment returned unexpected truncation or private info")

                train_data["source_id"][row] = source_id
                train_data["episode_id"][row] = episode_id
                train_data["time_step"][row] = time_step
                train_data["state"][row] = observation[0]
                train_data["action"][row] = action
                train_data["reward"][row] = reward
                train_data["next_state"][row] = next_observation[0]
                train_data["terminated"][row] = terminated
                audit_data["source_id"][row] = source_id
                audit_data["episode_id"][row] = episode_id
                audit_data["time_step"][row] = time_step
                audit_data["confounder_c"][row] = confounder
                audit_data["randomizer_w"][row] = randomizer
                observation = next_observation
                row += 1

    return train_data, audit_data


def generate_fixed_state_dataset(
    h_values,
    state_values,
    samples_per_state_source: int,
    base_seed: int,
    horizon: int = 20,
    gamma: float = 0.95,
    kappa: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate independent one-step samples at explicitly specified states."""
    horizon = _positive_integer("horizon", horizon)
    sample_count = _positive_integer("samples_per_state_source", samples_per_state_source)
    if not isinstance(base_seed, (int, np.integer)) or isinstance(base_seed, (bool, np.bool_)):
        raise ValueError("base_seed must be a nonnegative integer")
    if base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    h_values = tuple(h_values)
    state_values = tuple(float(state) for state in state_values)
    if not h_values or not state_values:
        raise ValueError("h_values and state_values must be nonempty")
    if any(
        not isinstance(h, (int, np.integer))
        or isinstance(h, (bool, np.bool_))
        or not 1 <= h <= horizon
        for h in h_values
    ):
        raise ValueError("every h must be an integer in [1, horizon]")
    if any(not np.isfinite(state) or not -1.0 <= state <= 1.0 for state in state_values):
        raise ValueError("every state must be finite and in [-1, 1]")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and nonnegative")

    row_count = len(h_values) * len(state_values) * 3 * sample_count
    train_data = {
        "row_id": np.arange(row_count, dtype=np.int64),
        "query_id": np.empty(row_count, dtype=np.int64),
        "time_step": np.empty(row_count, dtype=np.int16),
        "state": np.empty(row_count, dtype=np.float64),
        "source_id": np.empty(row_count, dtype=np.int8),
        "sample_index": np.empty(row_count, dtype=np.int64),
        "action": np.empty(row_count, dtype=np.float64),
        "reward": np.empty(row_count, dtype=np.float64),
        "next_state": np.empty(row_count, dtype=np.float64),
    }
    audit_data = {
        "row_id": train_data["row_id"].copy(),
        "query_id": np.empty(row_count, dtype=np.int64),
        "time_step": np.empty(row_count, dtype=np.int16),
        "state": np.empty(row_count, dtype=np.float64),
        "source_id": np.empty(row_count, dtype=np.int8),
        "sample_index": np.empty(row_count, dtype=np.int64),
        "confounder_c": np.empty(row_count, dtype=np.int8),
        "randomizer_w": np.empty(row_count, dtype=np.int8),
    }

    row = 0
    for h_index, h in enumerate(h_values):
        for state_index, state in enumerate(state_values):
            query_id = h_index * len(state_values) + state_index
            for source_id in (1, 2, 3):
                for sample_index in range(sample_count):
                    seed = np.random.SeedSequence(
                        [int(base_seed), int(h), state_index, source_id, sample_index]
                    )
                    confounder, randomizer = sample_exogenous(np.random.default_rng(seed))
                    action = behavior_action_fn(
                        source_id, state, (confounder, randomizer), kappa
                    )
                    next_state = transition_fn(state, action, confounder)
                    reward = reward_fn(state, action, confounder, next_state)
                    shared = {
                        "query_id": query_id,
                        "time_step": h,
                        "state": state,
                        "source_id": source_id,
                        "sample_index": sample_index,
                    }
                    for key, value in shared.items():
                        train_data[key][row] = value
                        audit_data[key][row] = value
                    train_data["action"][row] = action
                    train_data["reward"][row] = reward
                    train_data["next_state"][row] = next_state
                    audit_data["confounder_c"][row] = confounder
                    audit_data["randomizer_w"][row] = randomizer
                    row += 1
    return train_data, audit_data


def sample_fixed_state_online_intervention(
    h: int,
    state: float,
    action: float,
    reference: dict,
    rng: np.random.Generator,
) -> float:
    """Draw one fresh ``do(A=action)`` Bellman outcome and reveal only ``Z``."""
    horizon = int(reference["horizon"])
    if not isinstance(h, (int, np.integer)) or isinstance(h, (bool, np.bool_)):
        raise ValueError("h must be an integer in [1, horizon]")
    if not 1 <= h <= horizon:
        raise ValueError("h must be an integer in [1, horizon]")
    state, action = float(state), float(action)
    if not np.isfinite(state) or not -1.0 <= state <= 1.0:
        raise ValueError("state must be finite and in [-1, 1]")
    if not np.isfinite(action) or not -1.0 <= action <= 1.0:
        raise ValueError("action must be finite and in [-1, 1]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    confounder = -1 if int(rng.integers(2)) == 0 else 1
    next_state = transition_fn(state, action, confounder)
    reward = reward_fn(state, action, confounder, next_state)
    continuation = np.interp(
        next_state, reference["state_grid"], reference["values"][int(h) + 1]
    )
    return float(reward + float(reference["gamma"]) * continuation)


def save_offline_dataset(
    train_data: dict[str, np.ndarray],
    audit_data: dict[str, np.ndarray],
    output_dir: str | Path,
    prefix: str = "offline_smoke",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save public and privileged data separately as compressed NPZ files."""
    directory = Path(output_dir)
    train_path = directory / f"{prefix}_train.npz"
    audit_path = directory / f"{prefix}_audit.npz"
    existing = [path for path in (train_path, audit_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    if set(train_data) != set(TRAIN_FIELDS):
        raise ValueError("train_data fields do not match the public schema")
    if set(audit_data) != set(AUDIT_FIELDS):
        raise ValueError("audit_data fields do not match the privileged schema")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(train_path, **train_data)
    np.savez_compressed(audit_path, **audit_data)
    return train_path, audit_path
