"""CLI for the Phase 8A-NC-LH true-Hopper consequence audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.hopper_logger_mixture_drift.analyze_long_horizon_consequence import (
    run_long_horizon_audit,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fixed-policy finite-horizon Hopper intervention values")
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--phase8anc-root", type=Path, required=True)
    parser.add_argument("--phase8ac-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, default=2048)
    parser.add_argument("--kappas", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.3))
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 5, 20, 50))
    parser.add_argument("--rollout-reps", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        summary = run_long_horizon_audit(
            arguments.phase8a_root, arguments.phase8anc_root, arguments.phase8ac_root,
            arguments.output_root, num_anchors=arguments.num_anchors,
            kappas=tuple(arguments.kappas), horizons=tuple(arguments.horizons),
            rollout_reps=arguments.rollout_reps,
            bootstrap_reps=arguments.bootstrap_reps, seed=arguments.seed,
            num_workers=arguments.num_workers, gamma=arguments.gamma,
            device=arguments.device)
    except Exception as error:
        print("PHASE8A_NC_LONG_HORIZON_AUDIT_BLOCKED", flush=True)
        print(f"BLOCKING ERROR: {error}", flush=True)
        raise
    print(f"output: {arguments.output_root}")
    print(f"common eligible anchors: {summary['eligibility']['common_horizon_eligible']}")
    print("PHASE8A_NC_LONG_HORIZON_AUDIT_COMPLETE")
    print("READY_FOR_LONG_HORIZON_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
