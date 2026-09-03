from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8e_multisource_contrast import (  # noqa: E402
    Phase8EMultisourceContrastError,
    run_phase8e_multisource_contrast,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8E multi-source contrast calibration")
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--direct-reward-root", type=Path, required=True)
    parser.add_argument("--lambda-grid-file", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, required=True)
    parser.add_argument("--source-counts", nargs="+", type=int, required=True)
    parser.add_argument("--diversity-half-widths", nargs="+", type=float, required=True)
    parser.add_argument("--reward-noise-stds", nargs="+", type=float, required=True)
    parser.add_argument("--kappas", nargs="+", type=float, required=True)
    parser.add_argument("--conditions", nargs="+", required=True,
                        choices=("confounded", "independent_latents"))
    parser.add_argument("--offline-sample-budget", type=int, required=True)
    parser.add_argument("--calibration-budgets", nargs="+", type=int, required=True)
    parser.add_argument("--calibration-replicates", type=int, required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = run_phase8e_multisource_contrast(
            args.phase8a_root, args.direct_reward_root, args.lambda_grid_file,
            args.output_root, num_anchors=args.num_anchors,
            source_counts=tuple(args.source_counts),
            diversity_half_widths=tuple(args.diversity_half_widths),
            reward_noise_stds=tuple(args.reward_noise_stds), kappas=tuple(args.kappas),
            conditions=tuple(args.conditions), offline_sample_budget=args.offline_sample_budget,
            calibration_budgets=tuple(args.calibration_budgets),
            calibration_replicates=args.calibration_replicates,
            model_seeds=tuple(args.model_seeds), device=args.device)
    except (Phase8EMultisourceContrastError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8E_MULTISOURCE_CONTRAST_CALIBRATION_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['analyzed_anchor_count']}")
    print("PHASE8E_MULTISOURCE_CONTRAST_CALIBRATION_COMPLETE")
    print("READY_FOR_MULTISOURCE_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
