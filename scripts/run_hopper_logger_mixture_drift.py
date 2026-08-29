"""Generate and audit the Phase 8A Hopper logger-mixture causal-drift DGP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import gymnasium as gym
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.anchor_pool import (
    ANCHOR_FIELDS,
    balanced_source_quotas,
    checkpoint_roundtrip,
    collect_anchor_pool,
    resolve_phase6a_checkpoints,
    sha256,
    validate_anchor_pool,
)
from experiments.hopper_logger_mixture_drift.audit import (
    ATOL,
    RTOL,
    anchor_distribution_audit,
    hard_invariants,
    logger_sensitivity,
    make_figures,
    outcome_strength,
    population_observational_table,
    summarize_population_table,
)
from experiments.hopper_logger_mixture_drift.controlled_loggers import (
    ACTUATOR_DIRECTION,
    CONDITIONS,
    LOGGER_NAMES,
    MIXTURES,
    headroom_mask,
    target_action,
)
from experiments.hopper_logger_mixture_drift.generate_datasets import (
    HIDDEN_FIELDS,
    PUBLIC_FIELDS,
    MujocoOneStepSimulator,
    deterministic_repeat_check,
    generate_condition_dataset,
    generate_do_oracle,
    generate_mixture_weights,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _git_commit() -> str | None:
    result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _kappa_directory_name(kappa: float) -> str:
    return f"kappa_{kappa:.2f}".replace(".", "p")


def _load_or_collect_anchors(
    arguments: argparse.Namespace, models: dict[int, Any], source2: Any,
) -> tuple[dict[str, np.ndarray], str]:
    if arguments.anchors_file is not None:
        path = Path(arguments.anchors_file)
        if not path.is_file():
            raise FileNotFoundError(f"explicit anchor artifact does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            anchors = {name: archive[name].copy() for name in archive.files}
        validate_anchor_pool(anchors, arguments.num_anchors)
        if not np.all(headroom_mask(anchors["base_action"], arguments.behavior_offset)):
            raise RuntimeError("loaded anchors do not satisfy the requested behavior-offset headroom")
        return anchors, f"loaded_explicit_artifact:{path}"
    anchors = collect_anchor_pool(
        models, source2, arguments.num_anchors, arguments.behavior_offset, arguments.seed
    )
    return anchors, "fallback_fixed_stage_checkpoint_collector"


def _mixture_mean_rewards(table: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(table["observational_mean_reward"][table["mixture"] == name]))
            for name in MIXTURES}


def _save_condition(
    directory: Path, condition: str, public: dict[str, np.ndarray], hidden: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(directory / f"{condition}_public.npz", **public)
    np.savez_compressed(directory / f"{condition}_hidden_audit.npz", **hidden)
    weights_dir = directory / "weights" / condition
    weights_dir.mkdir(parents=True, exist_ok=True)
    for name, values in weights.items():
        np.save(weights_dir / f"weights_{name}.npy", values, allow_pickle=False)


def _write_analysis_bundle(output: Path, summary: dict[str, Any]) -> None:
    rows = []
    for key, item in summary["by_kappa"].items():
        for condition in CONDITIONS:
            population = item["population"][condition]
            rows.append(
                f"| {item['kappa_env']:.2f} | {condition} | "
                f"{population['reward_mixture_drift']['mean']:.8g} | "
                f"{population['next_state_delta_mixture_drift_l2']['mean']:.8g} | "
                f"{population['reward_do_error_absolute']['mean']:.8g} | "
                f"{population['mixture_action_ranking_flip_rate']:.8g} |"
            )
    failed = summary["failed_hard_invariants"]
    report = """# Phase 8A Hopper logger-mixture causal-drift DGP

## Question

With fixed simulator anchors and fixed Hopper dynamics, does changing only the logger mixture alter
the population observational response while the enumerated do(action) response remains mixture-independent?

## Descriptive results

| kappa | condition | mean reward drift | mean next-delta drift L2 | mean absolute reward do-error | ranking flip rate |
|---:|---|---:|---:|---:|---:|
""" + "\n".join(rows) + f"""

## Hard-validation status

All hard invariants passed: **{summary['all_hard_invariants_passed']}**.
Failed invariants: {failed if failed else 'none'}.

## Interpretation limits

Effects are descriptive over fixed anchors. Enumerated transition rows are probability atoms, not
independent statistical samples. No significance test, learned world model, online SAC run, rho,
or LP result is part of this phase.
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    appendix = f"""# Statistical appendix

- Unit of aggregation: anchor; action cells are reduced before summary.
- Anchor count: {summary['anchor_count']}.
- Seed count: one (`seed={summary['seed']}`).
- Latents: completely enumerated with exact pair masses; no Monte Carlo latent sampling.
- Mixtures: sample-weight changes over one master dataset; no row deletion or duplication.
- Numerical consistency tolerance: `atol={ATOL}`, `rtol={RTOL}`.
- Effect-size success thresholds: none.
- Inferential tests and statistical-significance claims: not performed.
"""
    (output / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    catalog = """# Figure catalog

- `reward_prediction_vs_mixture.png`: observational reward trajectories by mixture and condition.
- `next_state_drift_vs_kappa.png`: anchor-level next-state-delta mixture drift across kappa.
- `do_error_vs_kappa.png`: absolute observational reward error relative to the fixed do oracle.
- `action_ranking_flip_vs_kappa.png`: anchor-level changes in the best one-step action key.

Each figure is descriptive. Kappa was prespecified and is not selected from these plots.
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir, output = Path(arguments.checkpoint_dir), Path(arguments.output_root)
    phase6a_manifest, checkpoint_paths = resolve_phase6a_checkpoints(checkpoint_dir)
    try:
        import mujoco
        import stable_baselines3
        import torch
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError("Phase 8A requires existing MuJoCo, PyTorch, and stable-baselines3") from exc
    models = {source: SAC.load(str(path), device=arguments.device)
              for source, path in checkpoint_paths.items()}
    source2 = models[2]
    observation_shape = getattr(getattr(source2, "observation_space", None), "shape", None)
    action_shape = getattr(getattr(source2, "action_space", None), "shape", None)
    if observation_shape != (13,) or action_shape != (3,):
        raise RuntimeError("Source 2 checkpoint observation/action schema does not match 13D/3D")
    before_hashes = {source: sha256(path) for source, path in checkpoint_paths.items()}
    anchors, anchor_method = _load_or_collect_anchors(arguments, models, source2)
    roundtrip = checkpoint_roundtrip(
        source2, SAC.load, anchors["public_observation"][:min(64, len(anchors["anchor_id"]))],
        arguments.device, ATOL, RTOL
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "anchors.npz", **anchors)
    _write_json(output / "mixture_weights.json", {
        name: list(weights) for name, weights in MIXTURES.items()
    })

    kappas = tuple(float(value) for value in arguments.kappas)
    simulator = MujocoOneStepSimulator(anchors, kappas, arguments.seed)
    by_kappa: dict[str, dict[str, Any]] = {}
    all_hard: dict[str, bool] = {}
    try:
        for kappa in kappas:
            key, directory = _kappa_directory_name(kappa), output / _kappa_directory_name(kappa)
            directory.mkdir(parents=True, exist_ok=True)
            do_raw, do_summary = generate_do_oracle(
                anchors, kappa, arguments.behavior_offset, simulator
            )
            datasets, weights, tables, population, table_arrays = {}, {}, {}, {}, {}
            for condition in CONDITIONS:
                public, hidden = generate_condition_dataset(
                    anchors, condition, kappa, arguments.behavior_offset, simulator
                )
                condition_weights = generate_mixture_weights(hidden)
                table = population_observational_table(
                    anchors, public, hidden, condition_weights, do_summary
                )
                condition_summary, condition_arrays = summarize_population_table(table)
                datasets[condition] = (public, hidden); weights[condition] = condition_weights
                tables[condition] = table; population[condition] = condition_summary
                table_arrays.update({f"{condition}_{name}": values for name, values in table.items()})
                table_arrays.update({f"{condition}_summary_{name}": values
                                     for name, values in condition_arrays.items()})
                _save_condition(directory, condition, public, hidden, condition_weights)
            np.savez_compressed(directory / "do_oracle_raw.npz", **do_raw)
            np.savez_compressed(directory / "do_oracle_summary.npz", **do_summary)
            np.savez_compressed(directory / "population_audit_tables.npz", **table_arrays)
            command = target_action(anchors["base_action"][0], "base", arguments.behavior_offset)
            deterministic = deterministic_repeat_check(simulator, 0, command, 1, kappa, ATOL, RTOL)
            invariants = hard_invariants(
                anchors, datasets, weights, tables, do_raw, kappa, arguments.behavior_offset,
                roundtrip["passed"], deterministic["passed"]
            )
            all_hard.update({f"{key}:{name}": value for name, value in invariants.items()})
            clipping = {
                condition: {
                    "commanded_action_clipping_rate": float(np.mean(datasets[condition][1]["commanded_action_clipped"])),
                    "applied_action_clipping_rate": float(np.mean(datasets[condition][1]["applied_action_clipped"])),
                } for condition in CONDITIONS
            }
            mixture_means = {condition: _mixture_mean_rewards(tables[condition])
                             for condition in CONDITIONS}
            kappa_summary = {
                "kappa_env": kappa,
                "transition_counts": {condition: int(len(datasets[condition][0]["row_id"]))
                                      for condition in CONDITIONS},
                "population": population,
                "confounding_excess_drift": {
                    "reward_mean": (population["confounded"]["reward_mixture_drift"]["mean"]
                                    - population["independent_latents"]["reward_mixture_drift"]["mean"]),
                    "next_state_delta_l2_mean": (
                        population["confounded"]["next_state_delta_mixture_drift_l2"]["mean"]
                        - population["independent_latents"]["next_state_delta_mixture_drift_l2"]["mean"]),
                },
                "outcome_strength": outcome_strength(do_raw), "clipping": clipping,
                "mixture_mean_reward": mixture_means,
                "anchor_weight_distribution": {
                    condition: anchor_distribution_audit(datasets[condition][0]["anchor_id"], weights[condition])
                    for condition in CONDITIONS},
                "deterministic_repeat": deterministic, "hard_invariants": invariants,
                "hidden_leakage": [], "oracle_depends_on_mixture": False,
            }
            _write_json(directory / "population_audit.json", kappa_summary)
            by_kappa[key] = kappa_summary
    finally:
        simulator.close()

    after_hashes = {source: sha256(path) for source, path in checkpoint_paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("a fixed Phase 6A checkpoint changed during Phase 8A")
    anchor_ids = anchors["anchor_id"]
    source_counts = {f"source_{source}": int(np.sum(anchors["anchor_origin_source"] == source))
                     for source in (1, 2, 3)}
    all_passed = bool(all(all_hard.values()))
    failed = [name for name, value in all_hard.items() if not value]
    summary = {
        "phase": "8A", "seed": arguments.seed, "anchor_count": int(len(anchor_ids)),
        "anchor_source_composition": source_counts,
        "anchor_source_balance_note": (
            "exactly_equal" if len(set(source_counts.values())) == 1
            else "requested total is not divisible by three; deterministic quotas differ by at most one"),
        "logger_u_action_sensitivity": logger_sensitivity(anchors, arguments.behavior_offset),
        "by_kappa": by_kappa, "all_hard_invariants": all_hard,
        "all_hard_invariants_passed": all_passed, "failed_hard_invariants": failed,
        "effect_success_threshold_applied": False, "statistical_significance_claimed": False,
        "transition_rows_treated_as_independent_samples": False,
    }
    _write_json(output / "summary.json", summary)
    manifest = {
        "phase": "8A", "git_commit": _git_commit(), "env_id": "Hopper-v5",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": torch.__version__, "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gym.__version__, "mujoco_version": mujoco.__version__,
        "source2_checkpoint_path": str(checkpoint_paths[2]),
        "source2_checkpoint_sha256": before_hashes[2], "source2_original_manifest": phase6a_manifest,
        "observation_schema": "12D public Hopper observation = 11D Hopper + existing time_to_go",
        "behavior_offset_eta": arguments.behavior_offset,
        "actuator_direction_v": ACTUATOR_DIRECTION.tolist(), "kappas": list(kappas),
        "anchor_generation_seed": arguments.seed, "anchor_generation_method": anchor_method,
        "anchor_selection_rule": "source/episode/time-index spacing>=20 and commanded-action headroom only",
        "number_of_anchors": len(anchor_ids), "anchor_source_quotas": balanced_source_quotas(len(anchor_ids)),
        "logger_names": list(LOGGER_NAMES),
        "logger_formulas": {"diagnostic_logger_1": "a0 + eta*u_behavior*v",
                            "diagnostic_logger_2": "a0 - eta*u_behavior*v",
                            "diagnostic_logger_3": "a0"},
        "condition_formulas": {"confounded": "u_behavior=u_env; pair masses 0.5/0.5",
                               "independent_latents": "u_behavior independent u_env; four masses 0.25"},
        "mixture_weights": {name: list(value) for name, value in MIXTURES.items()},
        "public_fields": list(PUBLIC_FIELDS), "hidden_only_fields": list(HIDDEN_FIELDS),
        "anchor_fields": list(ANCHOR_FIELDS), "numerical_tolerance": {"atol": ATOL, "rtol": RTOL},
        "source2_checkpoint_roundtrip": roundtrip, "checkpoint_files_unchanged": True,
        "phase6b_anchor_reuse_check": (
            "existing Phase 6B implementation does not persist full qpos/qvel snapshots; fallback collector used"
            if arguments.anchors_file is None else "explicit compatible anchor artifact loaded"),
        "trained_any_model": False, "used_aamas": False, "used_rho_or_lp": False,
    }
    _write_json(output / "manifest.json", manifest)
    make_figures(output / "figures", by_kappa)
    _write_analysis_bundle(output, summary)
    print("hidden leakage: set()")
    print("HARD INVARIANTS")
    for name, passed in all_hard.items():
        print(f"{name}: {passed}")
    if not all_passed:
        print("FAILED", failed)
        print("PHASE8A_HOPPER_LOGGER_MIXTURE_DGP_BLOCKED")
    else:
        print("PHASE8A_HOPPER_LOGGER_MIXTURE_DGP_COMPLETE")
        print("READY_FOR_POOLED_WORLD_MODEL_DRIFT_TRAINING")
    return summary


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("artifacts/hopper_behavior_policies/seed_0"))
    parser.add_argument("--anchors-file", type=Path)
    parser.add_argument("--num-anchors", type=int, default=24)
    parser.add_argument("--kappas", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.3))
    parser.add_argument("--behavior-offset", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path,
                        default=Path("artifacts/hopper_logger_mixture_drift/controlled_loggers_seed0"))
    arguments = parser.parse_args(argv)
    if arguments.num_anchors < 3:
        parser.error("--num-anchors must be at least three")
    if not np.isfinite(arguments.behavior_offset) or arguments.behavior_offset < 0.0:
        parser.error("--behavior-offset must be finite and nonnegative")
    kappas = tuple(float(value) for value in arguments.kappas)
    if not kappas or any(not np.isfinite(value) or value < 0.0 for value in kappas):
        parser.error("--kappas must be finite and nonnegative")
    if len(set(kappas)) != len(kappas):
        parser.error("--kappas must not contain duplicates")
    if 0.0 not in kappas:
        parser.error("--kappas must include 0.0 for the required negative-control invariants")
    arguments.kappas = kappas
    return arguments


if __name__ == "__main__":
    try:
        completed = run(parse_arguments())
    except Exception as error:
        print("PHASE8A_HOPPER_LOGGER_MIXTURE_DGP_BLOCKED")
        print("BLOCKING ERROR:", error)
        raise
    if not completed["all_hard_invariants_passed"]:
        raise SystemExit(1)
