from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8f_quick_bound_intersection import (  # noqa: E402
    Phase8FQuickBoundError,
    run_phase8f_quick_bound_intersection,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8F-Q exact multi-source causal-bound intersection gate"
    )
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--phase8anc-root", type=Path, required=True)
    parser.add_argument("--phase8eq-root", type=Path, required=True)
    parser.add_argument("--kappas", nargs="+", type=float, required=True)
    parser.add_argument("--lambda-values", nargs="+", type=float, required=True)
    parser.add_argument("--source-settings", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        summary = run_phase8f_quick_bound_intersection(
            arguments.phase8a_root,
            arguments.phase8anc_root,
            arguments.phase8eq_root,
            arguments.output_root,
            kappas=arguments.kappas,
            lambda_values=arguments.lambda_values,
            source_settings=arguments.source_settings,
        )
    except (Phase8FQuickBoundError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8F_QUICK_BOUND_INTERSECTION_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['anchor_count']}")
    print(f"scenarios: {summary['scenario_count']}")
    print("PHASE8F_QUICK_BOUND_INTERSECTION_COMPLETE")
    print("READY_FOR_BOUND_UTILITY_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

