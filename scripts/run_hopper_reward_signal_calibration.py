"""Run Phase 8B-RS direct U-to-reward calibration and neural audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.reward_signal_calibration import (  # noqa: E402
    CALIBRATION_KAPPAS,
    CONDITIONS,
    PRIMARY_MIXTURE_NAMES,
    REWARD_STRENGTHS,
    run_reward_signal_calibration,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8anc-root", type=Path, required=True)
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--phase8b-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--kappas", type=float, nargs="+", default=CALIBRATION_KAPPAS)
    parser.add_argument("--reward-strengths", type=float, nargs="+", default=REWARD_STRENGTHS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--mixtures", nargs="+", default=PRIMARY_MIXTURE_NAMES)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--gradient-updates", "--updates", dest="updates", type=int,
                        default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        run_reward_signal_calibration(
            args.phase8anc_root, args.phase8a_root, args.output_root,
            phase8b_root=args.phase8b_root, num_anchors=args.num_anchors,
            kappas=tuple(args.kappas), reward_strengths=tuple(args.reward_strengths),
            conditions=tuple(args.conditions), mixtures=tuple(args.mixtures),
            model_seeds=tuple(args.model_seeds), updates=args.updates,
            batch_size=args.batch_size, device=args.device, split_seed=args.split_seed)
    except Exception as exc:
        print("PHASE8B_REWARD_SIGNAL_CALIBRATION_BLOCKED")
        print(f"BLOCKING ERROR: {exc}")
        raise
    print("PHASE8B_REWARD_SIGNAL_CALIBRATION_COMPLETE")
    print("READY_FOR_REWARD_SIGNAL_LEARNABILITY_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

