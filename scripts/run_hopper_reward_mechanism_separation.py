from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.reward_mechanism_separation import (  # noqa: E402
    LambdaGridNotFrozenError,
    RewardMechanismSeparationError,
    run_reward_mechanism_separation,
)


DEFAULT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8C-RM reward-only mechanism separation")
    parser.add_argument("--phase8anc-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--direct-reward-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path,
                        default=DEFAULT_ROOT / "oracle_direct_reward_confounding_audit")
    parser.add_argument("--lambda-grid-file", type=Path,
                        default=Path("analysis/phase8b_rs_low_dose_threshold_audit/frozen_lambda_grid.json"))
    parser.add_argument("--num-anchors", type=int, default=100)
    parser.add_argument("--kappas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--conditions", nargs="+", default=["confounded", "independent_latents"])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--gradient-updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_ROOT / "phase8c_reward_mechanism_smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = run_reward_mechanism_separation(
            args.phase8anc_root, args.direct_reward_root, args.oracle_root,
            args.lambda_grid_file, args.output_root, num_anchors=args.num_anchors,
            kappas=tuple(args.kappas), conditions=tuple(args.conditions),
            model_seeds=tuple(args.model_seeds), gradient_updates=args.gradient_updates,
            batch_size=args.batch_size, device=args.device, split_seed=args.split_seed)
    except LambdaGridNotFrozenError as exc:
        print("LAMBDA_GRID_NOT_FROZEN", file=sys.stderr)
        print(f"PHASE8C_REWARD_MECHANISM_SEPARATION_BLOCKED: {exc}", file=sys.stderr)
        return 2
    except (RewardMechanismSeparationError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8C_REWARD_MECHANISM_SEPARATION_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['analyzed_anchor_count']}")
    print("PHASE8C_REWARD_MECHANISM_SEPARATION_COMPLETE")
    print("READY_FOR_MECHANISM_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
