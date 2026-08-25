"""End-to-end audit for the Phase 1B oracle reference and offline dataset."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from confounded_smooth_regulator import (  # noqa: E402
    behavior_action_fn,
    reward_fn,
    transition_fn,
)
from generate_offline_dataset import (  # noqa: E402
    AUDIT_FIELDS,
    TRAIN_FIELDS,
    generate_offline_dataset,
    save_offline_dataset,
)
from oracle_ground_truth import save_oracle_reference, solve_oracle  # noqa: E402


def require(condition: bool, name: str) -> None:
    if not condition:
        raise AssertionError(name)


def audit_oracles(output_dir: Path) -> None:
    coarse = solve_oracle(n_state=501, n_action=501)
    fine = solve_oracle(n_state=1001, n_action=1001)
    require(
        np.array_equal(fine["state_grid"][::2], coarse["state_grid"]),
        "nested state grids",
    )
    for h in range(1, 21):
        difference = np.max(np.abs(coarse["values"][h] - fine["values"][h, ::2]))
        certificate = (
            coarse["numerical_error_bound"][h]
            + fine["numerical_error_bound"][h]
            + 1e-12
        )
        require(difference <= certificate, f"oracle refinement certificate h={h}")

    path = save_oracle_reference(fine, output_dir / "oracle_reference.npz")
    print("h V_min V_max rho_coefficient numerical_error_bound")
    for h in (1, 5, 10, 15, 20):
        print(
            f"{h:2d} {fine['values'][h].min():.8f} {fine['values'][h].max():.8f} "
            f"{fine['rho_coefficients'][h]:.8f} {fine['numerical_error_bound'][h]:.8f}"
        )
    print(f"oracle_artifact={path}")


def audit_rows(train_data: dict, audit_data: dict) -> None:
    require(set(train_data) == set(TRAIN_FIELDS), "public training schema")
    require(set(audit_data) == set(AUDIT_FIELDS), "privileged audit schema")
    require(len(train_data["row_id"]) == 6000, "smoke dataset row count")
    require(np.array_equal(train_data["row_id"], np.arange(6000)), "continuous row_id")
    for key in ("row_id", "source_id", "episode_id", "time_step"):
        require(np.array_equal(train_data[key], audit_data[key]), f"aligned {key}")

    states = train_data["state"].astype(np.float64)
    actions = train_data["action"]
    confounders = audit_data["confounder_c"]
    next_states = transition_fn(states, actions, confounders)
    rewards = reward_fn(states, actions, confounders, next_states)
    require(
        np.allclose(train_data["next_state"], next_states, rtol=1e-7, atol=5e-8),
        "all-row transition replay",
    )
    require(
        np.allclose(train_data["reward"], rewards, rtol=1e-7, atol=5e-8),
        "all-row reward replay",
    )
    expected_actions = np.fromiter(
        (
            behavior_action_fn(source, state, (confounder, randomizer), 1.0)
            for source, state, confounder, randomizer in zip(
                train_data["source_id"],
                states,
                confounders,
                audit_data["randomizer_w"],
            )
        ),
        dtype=np.float64,
        count=len(states),
    )
    require(
        np.allclose(actions, expected_actions, rtol=1e-7, atol=5e-8),
        "all-row historical behavior replay",
    )


def audit_sources(train_data: dict, audit_data: dict) -> None:
    source_actions = []
    for source_id in (1, 2, 3):
        selected = train_data["source_id"] == source_id
        count = int(np.count_nonzero(selected))
        require(count == 2000, f"source {source_id} row count")
        confounders = audit_data["confounder_c"][selected]
        randomizers = audit_data["randomizer_w"][selected]
        states = train_data["state"][selected]
        probability_tolerance = 6.0 * np.sqrt(0.25 / count)
        for name, samples in (("C", confounders), ("W", randomizers)):
            for value in (-1, 1):
                probability = np.mean(samples == value)
                require(
                    abs(probability - 0.5) <= probability_tolerance,
                    f"source {source_id} P({name}={value})",
                )
        correlation_limit = 6.0 / np.sqrt(count)
        cw_correlation = float(np.corrcoef(confounders, randomizers)[0, 1])
        state_c_correlation = float(np.corrcoef(states, confounders)[0, 1])
        state_w_correlation = float(np.corrcoef(states, randomizers)[0, 1])
        require(abs(cw_correlation) <= correlation_limit, f"source {source_id} corr(C,W)")
        require(
            abs(state_c_correlation) <= correlation_limit,
            f"source {source_id} corr(state,C)",
        )
        require(
            abs(state_w_correlation) <= correlation_limit,
            f"source {source_id} corr(state,W)",
        )
        actions = train_data["action"][selected]
        rewards = train_data["reward"][selected]
        source_actions.append(actions)
        print(
            f"source={source_id} corr_CW={cw_correlation:.5f} "
            f"corr_state_C={state_c_correlation:.5f} corr_state_W={state_w_correlation:.5f} "
            f"mean_action={actions.mean():.5f} std_action={actions.std():.5f} "
            f"mean_abs_action={np.abs(actions).mean():.5f} mean_reward={rewards.mean():.5f}"
        )
    require(
        all(
            not np.array_equal(source_actions[left], source_actions[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ),
        "historical source action arrays differ",
    )


def audit_reproducibility() -> None:
    first = generate_offline_dataset(2, 2027, horizon=3)
    second = generate_offline_dataset(2, 2027, horizon=3)
    for first_part, second_part in zip(first, second):
        for key in first_part:
            require(np.array_equal(first_part[key], second_part[key]), f"reproducible {key}")


def run_checks() -> None:
    output_dir = ROOT / "artifacts" / "phase1b"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_oracles(output_dir)
    train_data, audit_data = generate_offline_dataset(100, 2027)
    audit_rows(train_data, audit_data)
    audit_sources(train_data, audit_data)
    train_path, audit_path = save_offline_dataset(
        train_data, audit_data, output_dir, overwrite=True
    )
    print(f"smoke_rows={len(train_data['row_id'])}")
    print(f"train_artifact={train_path}")
    print(f"audit_artifact={audit_path}")
    audit_reproducibility()


def main() -> int:
    try:
        run_checks()
    except Exception as exc:
        print(f"FAIL {exc}")
        print("PHASE1B_MISMATCH")
        return 1
    print("PHASE1B_FAITHFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
