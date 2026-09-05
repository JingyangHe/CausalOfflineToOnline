"""Run one explicit stage of Phase 8J-FIX-Q."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8j_potential_clamp_fix_quick import (  # noqa: E402
    DEFAULT_LEGACY_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCALING_ROOT,
    FORMAL_EVAL_STEPS,
    MODEL_SEEDS,
    Phase8JClampFixError,
    SMOKE_STEPS,
    run_analyze,
    run_online,
    run_potentials,
    run_preflight_and_tests,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=(
        "preflight-and-tests", "potentials", "online-smoke", "online", "analyze"))
    parser.add_argument("--phase8h-scaling-root", type=Path, default=DEFAULT_SCALING_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        common = {
            "output_root": args.output_root,
            "external_repo": args.external_repo,
            "device": args.device,
        }
        if args.phase == "preflight-and-tests":
            run_preflight_and_tests(
                args.phase8h_scaling_root, args.legacy_root, **common)
            print("POTENTIAL_CLAMP_REPAIR_VALIDATED")
            return 0
        if args.phase == "potentials":
            result = run_potentials(
                args.phase8h_scaling_root, args.legacy_root, **common)
            print(f"repaired potential checkpoints: {result['potential_count']}")
            print("REPAIRED_POTENTIAL_TRAINING_COMPLETE")
            print("MANUAL_REVIEW_REQUIRED_BEFORE_ONLINE_SMOKE")
            return 0
        if args.phase == "online-smoke":
            result = run_online(
                **common, smoke=True, online_seeds=(0,), online_steps=2_000,
                eval_steps=SMOKE_STEPS, eval_episodes=5)
            print(f"smoke runs: {result['online_run_count']}")
            print("REPAIRED_PBRS_SMOKE_COMPLETE")
            return 0
        if args.phase == "online":
            result = run_online(
                **common, smoke=False, online_seeds=MODEL_SEEDS, online_steps=50_000,
                eval_steps=FORMAL_EVAL_STEPS, eval_episodes=5)
            print(f"online runs: {result['online_run_count']}")
            print("REPAIRED_PBRS_SAC_PILOT_COMPLETE")
            return 0
        run_analyze(args.output_root)
        print("PHASE8J_FIX_ANALYSIS_COMPLETE")
        return 0
    except (Phase8JClampFixError, FileNotFoundError, KeyError, ValueError,
            RuntimeError) as error:
        print("PHASE8J_POTENTIAL_CLAMP_FIX_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
