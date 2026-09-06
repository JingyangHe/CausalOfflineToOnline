"""Run one explicit stage of Phase 8J-BI-Q."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8j_core_bootstrap_isolation import (  # noqa: E402
    COMPONENT_UPDATES,
    DEFAULT_FIX_ROOT,
    DEFAULT_FVI_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GAMMA,
    INNER_EPOCHS,
    MODEL_SEED,
    OUTER_ITERATIONS,
    SAMPLES_PER_ANCHOR_SOURCE,
    TOTAL_EPOCHS,
    VARIANTS,
    Phase8JBootstrapIsolationError,
    run_analyze,
    run_preflight,
    run_train,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "train", "analyze"))
    parser.add_argument("--phase8j-fix-root", type=Path, default=DEFAULT_FIX_ROOT)
    parser.add_argument("--phase8j-fvi-root", type=Path, default=DEFAULT_FVI_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--samples-per-anchor-source", type=int,
                        default=SAMPLES_PER_ANCHOR_SOURCE)
    parser.add_argument("--component-updates", type=int, default=COMPONENT_UPDATES)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--outer-iterations", type=int, default=OUTER_ITERATIONS)
    parser.add_argument("--inner-epochs", type=int, default=INNER_EPOCHS)
    parser.add_argument("--total-epochs", type=int, default=TOTAL_EPOCHS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    common = {
        "fix_root": args.phase8j_fix_root,
        "fvi_root": args.phase8j_fvi_root,
        "output_root": args.output_root,
        "external_repo": args.external_repo,
        "device": args.device,
    }
    try:
        if args.phase == "preflight":
            run_preflight(
                **common,
                model_seed=args.model_seed,
                samples_per_anchor_source=args.samples_per_anchor_source,
                component_updates=args.component_updates,
                gamma=args.gamma,
            )
            print("PHASE8J_CORE_BOOTSTRAP_ISOLATION_PREFLIGHT_COMPLETE")
            return 0
        if args.phase == "train":
            run_train(
                **common,
                variants=args.variants,
                model_seed=args.model_seed,
                samples_per_anchor_source=args.samples_per_anchor_source,
                component_updates=args.component_updates,
                gamma=args.gamma,
                outer_iterations=args.outer_iterations,
                inner_epochs=args.inner_epochs,
                total_epochs=args.total_epochs,
            )
            print("PHASE8J_CORE_BOOTSTRAP_ISOLATION_TRAINING_COMPLETE")
            return 0
        run_analyze(args.output_root)
        print("PHASE8J_CORE_BOOTSTRAP_ISOLATION_COMPLETE")
        return 0
    except (Phase8JBootstrapIsolationError, FileNotFoundError, KeyError,
            ValueError, RuntimeError) as error:
        print("PHASE8J_CORE_BOOTSTRAP_ISOLATION_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
