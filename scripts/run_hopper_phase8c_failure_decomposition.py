from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8c_failure_decomposition import (  # noqa: E402
    FailureDecompositionError,
    run_phase8c_failure_decomposition,
)


DEFAULT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/noncomplementary_loggers_seed0_verified")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8C-FD Oracle-scaffolded mechanism failure decomposition")
    parser.add_argument("--phase8c-root", type=Path, required=True)
    parser.add_argument("--phase8c-analysis-root", type=Path, required=True)
    parser.add_argument("--dgp-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--kappas", nargs="+", type=float, default=[0.0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lambda-values", nargs="+", type=float)
    group.add_argument("--use-frozen-lambda-grid", action="store_true")
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num-anchors", type=int, default=100)
    parser.add_argument("--gradient-updates", type=int, default=300)
    parser.add_argument("--em-iterations", type=int, default=5)
    parser.add_argument("--em-mstep-updates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_ROOT / "phase8c_failure_decomposition_smoke")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        summary = run_phase8c_failure_decomposition(
            arguments.phase8c_root, arguments.phase8c_analysis_root,
            arguments.dgp_root, arguments.oracle_root, arguments.output_root,
            kappas=tuple(arguments.kappas),
            lambda_values=(tuple(arguments.lambda_values)
                           if arguments.lambda_values is not None else None),
            use_frozen_lambda_grid=arguments.use_frozen_lambda_grid,
            model_seeds=tuple(arguments.model_seeds),
            num_anchors=arguments.num_anchors,
            gradient_updates=arguments.gradient_updates,
            em_iterations=arguments.em_iterations,
            em_mstep_updates=arguments.em_mstep_updates,
            batch_size=arguments.batch_size, device=arguments.device)
    except (FailureDecompositionError, FileNotFoundError, KeyError, ValueError) as exc:
        print("PHASE8C_FAILURE_DECOMPOSITION_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['analyzed_anchor_count']}")
    print("PHASE8C_FAILURE_DECOMPOSITION_COMPLETE")
    print("READY_FOR_FAILURE_CAUSE_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
