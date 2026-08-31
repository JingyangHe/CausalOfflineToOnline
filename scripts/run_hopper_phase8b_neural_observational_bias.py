"""Command-line entry point for Phase 8B-NC."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.neural_observational_bias import (  # noqa: E402
    CONDITIONS,
    EXPECTED_KAPPAS,
    NeuralObservationalBiasError,
    PRIMARY_MIXTURE_NAMES,
    run_neural_observational_bias,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8anc-root", type=Path, required=True)
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--phase8ac-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, default=2048)
    parser.add_argument("--kappas", type=float, nargs="+", default=EXPECTED_KAPPAS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--mixtures", nargs="+", default=PRIMARY_MIXTURE_NAMES)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--gradient-updates", "--updates", dest="updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        run_neural_observational_bias(
            args.phase8anc_root, args.phase8a_root, args.phase8ac_root, args.output_root,
            num_anchors=args.num_anchors, kappas=tuple(args.kappas),
            conditions=tuple(args.conditions), mixtures=tuple(args.mixtures),
            model_seeds=tuple(args.model_seeds), updates=args.updates,
            batch_size=args.batch_size, device=args.device, split_seed=args.split_seed)
    except Exception as exc:
        print("PHASE8B_NC_NEURAL_OBSERVATIONAL_BIAS_BLOCKED")
        print(f"BLOCKING ERROR: {exc}")
        raise
    print("PHASE8B_NC_NEURAL_OBSERVATIONAL_BIAS_COMPLETE")
    print("READY_FOR_NEURAL_BIAS_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
