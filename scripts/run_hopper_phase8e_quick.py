from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8e_quick_go_nogo import (  # noqa: E402
    Phase8EMultisourceContrastError,
    run_phase8e_quick,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8E-Q quick multi-source go/no-go")
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, required=True)
    parser.add_argument("--source-settings", nargs="+", required=True)
    parser.add_argument("--lambda-values", nargs="+", type=float, required=True)
    parser.add_argument("--reward-noise-std", type=float, required=True)
    parser.add_argument("--offline-sample-budget", type=int, required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gradient-updates", type=int, required=True)
    parser.add_argument("--calibration-budgets", nargs="+", type=int, required=True)
    parser.add_argument("--calibration-replicates", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = run_phase8e_quick(
            args.phase8a_root, args.output_root, num_anchors=args.num_anchors,
            source_settings=args.source_settings, lambda_values=args.lambda_values,
            reward_noise_std=args.reward_noise_std,
            offline_sample_budget=args.offline_sample_budget,
            model_seeds=args.model_seeds, gradient_updates=args.gradient_updates,
            calibration_budgets=args.calibration_budgets,
            calibration_replicates=args.calibration_replicates,
            device=args.device, data_seed=args.data_seed)
    except (Phase8EMultisourceContrastError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8E_QUICK_GO_NO_GO_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"scenarios: {summary['scenario_count']}")
    print("PHASE8E_QUICK_GO_NO_GO_COMPLETE")
    print("READY_FOR_PHASE8E_QUICK_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
