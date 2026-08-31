"""Run Phase 8B-RS-O exact oracle reward-confounding audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.oracle_direct_reward_audit import (  # noqa: E402
    run_oracle_direct_reward_audit,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8anc-root", type=Path, required=True)
    parser.add_argument("--reward-signal-root", type=Path)
    parser.add_argument("--kappas", type=float, nargs="+", default=(0.0, 0.3))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        run_oracle_direct_reward_audit(
            args.phase8anc_root, args.output_root,
            reward_signal_root=args.reward_signal_root,
            kappas=tuple(args.kappas), bootstrap_reps=args.bootstrap_reps,
            seed=args.seed,
        )
    except Exception as exc:
        print("ORACLE_DIRECT_REWARD_CONFOUNDING_AUDIT_BLOCKED")
        print(f"BLOCKING ERROR: {exc}")
        raise
    print("ORACLE_DIRECT_REWARD_CONFOUNDING_AUDIT_COMPLETE")
    print("READY_FOR_NEURAL_REWARD_SIGNAL_TEST")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

