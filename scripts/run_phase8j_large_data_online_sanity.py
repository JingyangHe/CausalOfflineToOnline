"""Run one explicit stage of Phase 8J-Q."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8j_large_data_online_sanity import (  # noqa: E402
    FORMAL_EVAL_STEPS,
    Phase8JLargeDataOnlineSanityError,
    run_online,
    run_potentials,
)


DEFAULT_SCALING = Path("artifacts/hopper_logger_mixture_drift/phase8h_data_scaling")
DEFAULT_OUTPUT = Path("artifacts/hopper_logger_mixture_drift/phase8j_large_data_online_sanity")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("potentials", "online-smoke", "online"))
    parser.add_argument("--phase8h-scaling-root", type=Path, default=DEFAULT_SCALING)
    parser.add_argument("--samples-per-anchor-source", type=int, default=128)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--component-updates", type=int, default=4000)
    parser.add_argument("--online-seed", type=int, default=0)
    parser.add_argument("--online-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--online-steps", type=int)
    parser.add_argument("--eval-steps", nargs="+", type=int)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        if args.phase == "potentials":
            result = run_potentials(
                args.phase8h_scaling_root, args.output_root,
                samples_per_anchor_source=args.samples_per_anchor_source,
                model_seeds=args.model_seeds, component_updates=args.component_updates,
                device=args.device, external_repo=args.external_repo)
            print(f"potential checkpoints: {result['potential_count']}")
            print("PHASE8J_POTENTIALS_COMPLETE")
            return 0
        smoke = args.phase == "online-smoke"
        default_steps = 5_000 if smoke else 50_000
        online_steps = default_steps if args.online_steps is None else args.online_steps
        eval_steps = tuple(args.eval_steps) if args.eval_steps is not None else (
            (0, 5_000) if smoke else FORMAL_EVAL_STEPS)
        result = run_online(
            args.output_root,
            online_seeds=(args.online_seed,) if smoke else args.online_seeds,
            online_steps=online_steps, eval_steps=eval_steps,
            eval_episodes=args.eval_episodes, device=args.device,
            external_repo=args.external_repo, smoke=smoke)
        print(f"online runs: {result['online_run_count']}")
        if smoke:
            print("PHASE8J_ONLINE_SMOKE_COMPLETE")
        else:
            print("PHASE8J_LARGE_DATA_ONLINE_SANITY_COMPLETE")
            print("READY_FOR_MULTI_SOURCE_PBRS_REVIEW")
        return 0
    except (Phase8JLargeDataOnlineSanityError, FileNotFoundError, KeyError,
            ValueError, RuntimeError) as error:
        print("PHASE8J_LARGE_DATA_ONLINE_SANITY_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

