from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8d_public_init_calibration import (  # noqa: E402
    Phase8DPublicInitCalibrationError,
    run_phase8d_public_init_calibration,
)


DEFAULT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8D public initialization and do calibration")
    parser.add_argument("--phase8c-root", type=Path, required=True)
    parser.add_argument("--failure-decomposition-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--kappas", nargs="+", type=float, default=[0.0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lambda-values", nargs="+", type=float)
    group.add_argument("--use-frozen-lambda-grid", action="store_true")
    parser.add_argument("--num-anchors", type=int, default=100)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--calibration-replicates", type=int, default=4)
    parser.add_argument("--calibration-budgets", nargs="+", type=int, default=[0, 8, 16])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_ROOT / "phase8d_public_init_calibration_smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = run_phase8d_public_init_calibration(
            args.phase8c_root, args.failure_decomposition_root, args.oracle_root,
            args.output_root, kappas=tuple(args.kappas),
            lambda_values=(tuple(args.lambda_values) if args.lambda_values else None),
            use_frozen_lambda_grid=args.use_frozen_lambda_grid,
            num_anchors=args.num_anchors, model_seeds=tuple(args.model_seeds),
            calibration_replicates=args.calibration_replicates,
            calibration_budgets=tuple(args.calibration_budgets), device=args.device)
    except (Phase8DPublicInitCalibrationError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8D_PUBLIC_INITIALIZATION_CALIBRATION_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['analyzed_anchor_count']}")
    print("PHASE8D_PUBLIC_INITIALIZATION_CALIBRATION_COMPLETE")
    print("READY_FOR_PHASE8D_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
