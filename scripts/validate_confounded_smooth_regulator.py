"""Independent faithfulness checks for ConfoundedSmoothRegulator-v0."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confounded_smooth_regulator import (  # noqa: E402
    ConfoundedSmoothRegulatorEnv,
    behavior_action_fn,
    return_bounds,
    reward_fn,
    rho_coefficient,
    sample_exogenous,
    transition_fn,
    value_lipschitz_constants,
)


def require(condition: bool, name: str) -> None:
    if not condition:
        raise AssertionError(name)


def reproduce(seed: int) -> tuple:
    env = ConfoundedSmoothRegulatorEnv(horizon=5)
    observation, info = env.reset(seed=seed)
    trajectory = [(tuple(observation), tuple(info), env.get_exogenous_for_audit())]
    for action in (-0.8, -0.1, 0.6, 0.3, 0.0):
        current_exogenous = env.get_exogenous_for_audit()
        observation, reward, terminated, truncated, info = env.step(action)
        trajectory.append(
            (tuple(observation), reward, terminated, truncated, tuple(info), current_exogenous)
        )
    return tuple(trajectory)


def run_checks() -> list[str]:
    reports = []
    rng = np.random.default_rng(2027)

    samples = np.empty((100_000, 2), dtype=np.int8)
    for index in range(len(samples)):
        samples[index] = sample_exogenous(rng)
    probabilities = np.array(
        [[(samples[:, column] == value).mean() for value in (-1, 1)] for column in range(2)]
    )
    require(
        np.max(np.abs(probabilities - (0.5, 0.5))) <= 0.01,
        "C/W distributions",
    )
    require(abs(np.corrcoef(samples, rowvar=False)[0, 1]) <= 0.01, "C/W independence")
    reports.append("PASS fair independent C/W distribution (n=100000)")

    states = rng.uniform(-1.0, 1.0, 100_000)
    actions = rng.uniform(-1.0, 1.0, 100_000)
    confounders = rng.choice((-1, 1), 100_000)
    next_states = np.fromiter(
        (transition_fn(s, a, c) for s, a, c in zip(states, actions, confounders)),
        dtype=np.float64,
        count=len(states),
    )
    rewards = np.fromiter(
        (
            reward_fn(s, a, c, next_s)
            for s, a, c, next_s in zip(states, actions, confounders, next_states)
        ),
        dtype=np.float64,
        count=len(states),
    )
    require(np.all((-1.0 <= states) & (states <= 1.0)), "sampled state range")
    require(np.all((-1.0 <= actions) & (actions <= 1.0)), "sampled action range")
    require(np.all((-1.0 < next_states) & (next_states < 1.0)), "next-state range")
    require(np.all((-1e-12 <= rewards) & (rewards <= 1.0 + 1e-12)), "reward range")
    reports.append("PASS transition, action, state, and reward ranges (n=100000)")

    count = 20_000
    states = rng.uniform(-1.0, 1.0, count)
    other_states = rng.uniform(-1.0, 1.0, count)
    actions = rng.uniform(-1.0, 1.0, count)
    other_actions = rng.uniform(-1.0, 1.0, count)
    confounders = rng.choice((-1, 1), count)
    f_sa = np.fromiter(
        (transition_fn(s, a, c) for s, a, c in zip(states, actions, confounders)),
        dtype=np.float64,
        count=count,
    )
    f_other_a = np.fromiter(
        (transition_fn(s, a, c) for s, a, c in zip(states, other_actions, confounders)),
        dtype=np.float64,
        count=count,
    )
    f_other_s = np.fromiter(
        (transition_fn(s, a, c) for s, a, c in zip(other_states, actions, confounders)),
        dtype=np.float64,
        count=count,
    )
    r_sa = np.fromiter(
        (reward_fn(s, a, c) for s, a, c in zip(states, actions, confounders)),
        dtype=np.float64,
        count=count,
    )
    r_other_a = np.fromiter(
        (reward_fn(s, a, c) for s, a, c in zip(states, other_actions, confounders)),
        dtype=np.float64,
        count=count,
    )
    r_other_s = np.fromiter(
        (reward_fn(s, a, c) for s, a, c in zip(other_states, actions, confounders)),
        dtype=np.float64,
        count=count,
    )
    ratios = (
        np.max(np.abs(f_sa - f_other_a) / np.abs(actions - other_actions)),
        np.max(np.abs(f_sa - f_other_s) / np.abs(states - other_states)),
        np.max(np.abs(r_sa - r_other_a) / np.abs(actions - other_actions)),
        np.max(np.abs(r_sa - r_other_s) / np.abs(states - other_states)),
    )
    bounds = (0.40, 0.65, 1.06, 0.585)
    require(all(ratio <= bound + 1e-10 for ratio, bound in zip(ratios, bounds)), "smoothness bounds")
    reports.append("PASS smoothness bounds (0.40, 0.65, 1.06, 0.585)")

    horizon, gamma = 20, 0.95
    constants = value_lipschitz_constants(horizon, gamma)
    require(np.all(np.isfinite(constants[1:])), "value Lipschitz constants")
    for h in range(1, horizon + 1):
        require(np.all(np.isfinite(return_bounds(h, horizon, gamma))), f"return bounds h={h}")
        require(np.isfinite(rho_coefficient(h, horizon, gamma)), f"rho coefficient h={h}")
    reports.append("PASS finite return, value-Lipschitz, and rho bounds")

    for source_id in (1, 2, 3):
        for w in (-1, 1):
            require(
                behavior_action_fn(source_id, 0.2, (-1, w), 0.0)
                == behavior_action_fn(source_id, 0.2, (1, w), 0.0),
                f"kappa=0 source {source_id}",
            )
    reports.append("PASS kappa=0 removes behavioral dependence on C")

    exogenous_values = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    audited_actions = tuple(
        behavior_action_fn(1, 0.0, exogenous, 1.0) for exogenous in exogenous_values
    )
    expected_actions = (0.0, np.tanh(1.10), np.tanh(-1.10), 0.0)
    require(np.allclose(audited_actions, expected_actions, atol=1e-15), "four-atom action audit")
    require(audited_actions[:2] != audited_actions[2:], "C changes the action distribution")
    require(
        abs(audited_actions[0] - audited_actions[3]) <= 1e-15,
        "action must not uniquely reveal C",
    )
    reports.append("PASS four-atom hidden-confounding and non-revelation audit")

    associated_action = behavior_action_fn(1, 0.0, (-1, 1), 1.0)
    observational_reward = reward_fn(0.0, associated_action, -1)
    interventional_reward = 0.5 * sum(
        reward_fn(0.0, associated_action, confounder) for confounder in (-1, 1)
    )
    require(
        abs(observational_reward - interventional_reward) > 1e-6,
        "observational/interventional confounding gap",
    )
    reports.append("PASS observational/interventional reward gap")

    env = ConfoundedSmoothRegulatorEnv(horizon=2)
    observation, info = env.reset(seed=4)
    require(observation.shape == (2,) and info == {}, "reset hidden-variable leakage")
    observation, _, _, _, info = env.step(0.0)
    require(observation.shape == (2,) and info == {}, "step hidden-variable leakage")
    require(reproduce(991) == reproduce(991), "seed reproducibility")
    reports.append("PASS public-data non-leakage and exact seeded reproduction")
    return reports


def main() -> int:
    try:
        reports = run_checks()
    except Exception as exc:
        print(f"FAIL {exc}")
        print("ENVIRONMENT_MISMATCH")
        return 1
    for report in reports:
        print(report)
    print("ENVIRONMENT_FAITHFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
