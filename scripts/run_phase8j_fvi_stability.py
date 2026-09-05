"""Run one explicit stage of Phase 8J-FVI-Q."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8j_fvi_stability import (  # noqa: E402
    COMPONENT_UPDATES,
    DEFAULT_FIX_ROOT,
    DEFAULT_OUTPUT_ROOT,
    METHOD,
    MODEL_SEED,
    SAMPLES_PER_ANCHOR_SOURCE,
    VARIANTS,
    Phase8JFVIError,
    run_analyze,
    run_preflight_and_tests,
    run_train,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=("preflight-and-tests", "train", "analyze"))
    parser.add_argument("--phase8j-fix-root", type=Path, default=DEFAULT_FIX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--samples-per-anchor-source", type=int,
                        default=SAMPLES_PER_ANCHOR_SOURCE)
    parser.add_argument("--component-updates", type=int, default=COMPONENT_UPDATES)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--outer-iterations", type=int, default=40)
    parser.add_argument("--inner-epochs", type=int, default=5)
    parser.add_argument("--total-epochs", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        common = {"fix_root": args.phase8j_fix_root, "output_root": args.output_root,
                  "external_repo": args.external_repo, "device": args.device}
        if args.phase == "preflight-and-tests":
            run_preflight_and_tests(
                **common, method=args.method, model_seed=args.model_seed,
                samples_per_anchor_source=args.samples_per_anchor_source,
                component_updates=args.component_updates)
            print("PHASE8J_FVI_PREFLIGHT_COMPLETE")
            return 0
        if args.phase == "train":
            run_train(
                **common, variants=args.variants, method=args.method,
                model_seed=args.model_seed, outer_iterations=args.outer_iterations,
                inner_epochs=args.inner_epochs, total_epochs=args.total_epochs)
            print("PHASE8J_FVI_PAIRED_TRAINING_COMPLETE")
            return 0
        run_analyze(args.output_root)
        print("PHASE8J_FVI_STABILITY_AUDIT_COMPLETE")
        return 0
    except (Phase8JFVIError, FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print("PHASE8J_FVI_STABILITY_DIAGNOSTIC_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
