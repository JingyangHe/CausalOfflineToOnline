"""Run one explicit stage of Phase 8H-ON-Q."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8h_compute_matched_online_quick import (  # noqa: E402
    Phase8HComputeMatchedOnlineError,
    run_online,
    run_preflight,
    run_stage_a,
    run_stage_b,
)


DEFAULT_PHASE8H = Path(
    "artifacts/hopper_logger_mixture_drift/phase8h_quick_multipolicy_aamas")
DEFAULT_SCALING = Path(
    "artifacts/hopper_logger_mixture_drift/phase8h_data_scaling")
DEFAULT_OUTPUT = Path(
    "artifacts/hopper_logger_mixture_drift/phase8h_compute_matched_online_quick")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=("preflight", "stage-a", "stage-b", "online-smoke", "stage-c"))
    parser.add_argument("--phase8h-root", type=Path, default=DEFAULT_PHASE8H)
    parser.add_argument("--scaling-root", type=Path, default=DEFAULT_SCALING)
    parser.add_argument("--missing-n32-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--component-updates", type=int, default=4000)
    parser.add_argument("--offline-data-n", type=int, default=32)
    parser.add_argument("--run-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--potential-config", default="official-frozen")
    parser.add_argument("--online-steps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        if args.phase == "preflight":
            result = run_preflight(
                args.phase8h_root, args.scaling_root, args.output_root,
                missing_n32_seeds=args.missing_n32_seeds,
                run_ids=args.run_ids, online_steps=args.online_steps)
            for key in ("missing_component_models", "missing_required_existing_component_models",
                        "potential_count", "online_run_count",
                        "online_training_environment_steps", "estimated_evaluation_episodes",
                        "estimated_file_count", "estimated_storage_bytes"):
                print(f"{key}: {result[key]}")
            print("PHASE8H_ON_Q_PREFLIGHT_COMPLETE")
            return 0
        if args.phase == "stage-a":
            result = run_stage_a(
                args.phase8h_root, args.scaling_root, args.output_root,
                missing_n32_seeds=args.missing_n32_seeds,
                component_updates=args.component_updates, device=args.device,
                external_repo=args.external_repo)
            print(f"trained missing components: {result['trained_component_count']}")
            print("PHASE8H_COMPUTE_MATCHED_CHECK_COMPLETE")
            return 0
        if args.phase == "stage-b":
            result = run_stage_b(
                args.phase8h_root, args.scaling_root, args.output_root,
                offline_data_n=args.offline_data_n, run_ids=args.run_ids,
                potential_config=args.potential_config, device=args.device,
                external_repo=args.external_repo)
            print(f"potential checkpoints: {result['potential_count']}")
            print("PHASE8H_FULL_POTENTIAL_TRAINING_COMPLETE")
            return 0
        smoke = args.phase == "online-smoke"
        eval_every = min(args.eval_every, args.online_steps) if smoke else args.eval_every
        result = run_online(
            args.output_root, run_ids=args.run_ids, online_steps=args.online_steps,
            eval_every=eval_every, eval_episodes=args.eval_episodes,
            device=args.device, external_repo=args.external_repo, smoke=smoke)
        print(f"online runs: {result['online_run_count']}")
        print("PHASE8H_ONLINE_SMOKE_COMPLETE" if smoke
              else "PHASE8H_SHORT_ONLINE_PILOT_COMPLETE")
        return 0
    except (Phase8HComputeMatchedOnlineError, FileNotFoundError, KeyError,
            ValueError, RuntimeError) as error:
        print("PHASE8H_ON_Q_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
