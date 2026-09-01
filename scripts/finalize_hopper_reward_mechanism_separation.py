"""Finalize a complete Phase 8C run after a reporting-only failure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.reward_mechanism_finalization import (  # noqa: E402
    finalize_reward_mechanism_separation,
)


DEFAULT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8anc-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--direct-reward-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path,
                        default=DEFAULT_ROOT / "oracle_direct_reward_confounding_audit")
    parser.add_argument("--lambda-grid-file", type=Path,
                        default=Path("analysis/phase8b_rs_low_dose_threshold_audit/frozen_lambda_grid.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = finalize_reward_mechanism_separation(
            args.phase8anc_root, args.direct_reward_root, args.oracle_root,
            args.lambda_grid_file, args.output_root)
    except Exception as exc:
        print("PHASE8C_REWARD_MECHANISM_FINALIZATION_BLOCKED")
        print(f"BLOCKING ERROR: {exc}")
        raise
    print(f"validated models: {summary['trained_model_count']}")
    print("PHASE8C_REWARD_MECHANISM_FINALIZATION_COMPLETE")
    print("READY_FOR_MECHANISM_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

