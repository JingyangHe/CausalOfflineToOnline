"""CLI for the read-only Phase 8A-R population-effect review."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.hopper_logger_mixture_drift.analyze_phase8a_population_effect import (
    PopulationEffectAuditError,
    run_review,
)


DEFAULT_ROOT = Path(
    "artifacts/hopper_logger_mixture_drift/controlled_loggers_seed0_verified"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the Phase 8A logger-mixture population effects from raw artifacts."
    )
    parser.add_argument("--phase8a-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-anchors", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output = arguments.output_root or arguments.phase8a_root / "population_effect_review"
    try:
        result = run_review(
            phase8a_root=arguments.phase8a_root,
            output_root=output,
            bootstrap_reps=arguments.bootstrap_reps,
            seed=arguments.seed,
            max_anchors=arguments.max_anchors,
        )
    except (PopulationEffectAuditError, ValueError, OSError) as error:
        print("PHASE8A_POPULATION_EFFECT_AUDIT_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1
    if not result["all_hard_checks_passed"]:
        print("PHASE8A_POPULATION_EFFECT_AUDIT_BLOCKED", file=sys.stderr)
        return 1
    print("PHASE8A_POPULATION_EFFECT_REVIEW_COMPLETE")
    print("READY_FOR_SCIENTIFIC_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
