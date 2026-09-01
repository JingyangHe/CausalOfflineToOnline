from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.low_dose_threshold_audit import (  # noqa: E402
    LowDoseThresholdAuditError,
    run_audit,
)


DEFAULT_PHASE8A = Path(
    "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact Phase 8B-RS low-dose threshold audit")
    parser.add_argument("--oracle-root", type=Path,
                        default=DEFAULT_PHASE8A / "oracle_direct_reward_confounding_audit")
    parser.add_argument("--ranking-audit-root", type=Path,
                        default=Path("analysis/phase8b_rs_ranking_calibration_regret_audit"))
    parser.add_argument("--phase8a-root", type=Path, default=DEFAULT_PHASE8A)
    parser.add_argument("--output-root", type=Path,
                        default=Path("analysis/phase8b_rs_low_dose_threshold_audit"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_audit(args.oracle_root, args.ranking_audit_root,
                            args.phase8a_root, args.output_root, args.seed)
    except (LowDoseThresholdAuditError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8B_RS_LOW_DOSE_THRESHOLD_AUDIT_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['analyzed_anchor_count']}")
    print("PHASE8B_RS_LOW_DOSE_THRESHOLD_AUDIT_COMPLETE")
    print("READY_FOR_MANUAL_DOSE_GRID_FREEZE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
