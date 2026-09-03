from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8g_quick_exact_shared_world import (  # noqa: E402
    Phase8GQuickExactSharedWorldError,
    run_phase8g_quick_exact_shared_world,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8G-Q exact joint shared-world bound gate")
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--phase8anc-root", type=Path, required=True)
    parser.add_argument("--phase8f-root", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, required=True)
    parser.add_argument("--kappas", nargs="+", type=float, required=True)
    parser.add_argument("--lambda-values", nargs="+", type=float, required=True)
    parser.add_argument("--source-settings", nargs="+", required=True)
    parser.add_argument("--include-lambda-zero-control", action="store_true")
    parser.add_argument("--include-independent-control", action="store_true")
    parser.add_argument("--include-source-shuffle", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = run_phase8g_quick_exact_shared_world(
            args.phase8a_root,
            args.phase8anc_root,
            args.phase8f_root,
            args.output_root,
            num_anchors=args.num_anchors,
            kappas=args.kappas,
            lambda_values=args.lambda_values,
            source_settings=args.source_settings,
            include_lambda_zero_control=args.include_lambda_zero_control,
            include_independent_control=args.include_independent_control,
            include_source_shuffle=args.include_source_shuffle,
        )
    except (Phase8GQuickExactSharedWorldError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8G_QUICK_EXACT_SHARED_WORLD_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['anchor_count']}")
    print(f"scenarios: {summary['scenario_count']}")
    print("PHASE8G_QUICK_EXACT_SHARED_WORLD_COMPLETE")
    print("READY_FOR_SHARED_WORLD_GATE_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
