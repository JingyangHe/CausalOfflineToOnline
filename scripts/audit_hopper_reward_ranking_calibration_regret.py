from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.ranking_calibration_regret_audit import (  # noqa: E402
    RankingCalibrationRegretAuditError,
    run_audit,
)


DEFAULT_PARENT = Path(
    "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Phase 8B-RS ranking/calibration/regret audit")
    parser.add_argument("--neural-root", type=Path,
                        default=DEFAULT_PARENT / "phase8b_reward_signal_calibration")
    parser.add_argument("--oracle-root", type=Path,
                        default=DEFAULT_PARENT / "oracle_direct_reward_confounding_audit")
    parser.add_argument("--output-root", type=Path,
                        default=Path("analysis/phase8b_rs_ranking_calibration_regret_audit"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_audit(args.neural_root, args.oracle_root, args.output_root)
    except (RankingCalibrationRegretAuditError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8B_RS_RANKING_CALIBRATION_REGRET_AUDIT_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"test anchors: {summary['test_anchor_count']}")
    print("PHASE8B_RS_RANKING_CALIBRATION_REGRET_AUDIT_COMPLETE")
    print("READY_FOR_RANKING_CALIBRATION_REGRET_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

