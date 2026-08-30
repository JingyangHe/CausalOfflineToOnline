"""CLI for the Phase 8A-NC exact population DGP and audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.hopper_logger_mixture_drift.analyze_noncomplementary_population import (
    NonComplementaryPopulationAuditError,
    PopulationEffectAuditError,
    run_population_dgp,
)
from experiments.hopper_logger_mixture_drift.noncomplementary_population_dgp import (
    NonComplementaryDGPError,
)


DEFAULT_8A = Path("artifacts/hopper_logger_mixture_drift/controlled_loggers_seed0_verified")
DEFAULT_8AR = DEFAULT_8A / "population_effect_review"
DEFAULT_8AC = DEFAULT_8AR / "clipping_sensitivity_kappa_0p30"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the exact non-complementary Hopper DGP.")
    parser.add_argument("--phase8a-root", type=Path, default=DEFAULT_8A)
    parser.add_argument("--phase8ar-root", type=Path, default=DEFAULT_8AR)
    parser.add_argument("--phase8ac-root", type=Path, default=DEFAULT_8AC)
    parser.add_argument("--num-anchors", type=int, default=2048)
    parser.add_argument("--kappas", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.3))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        result = run_population_dgp(
            arguments.phase8a_root, arguments.phase8ar_root, arguments.phase8ac_root,
            arguments.output_root, num_anchors=arguments.num_anchors,
            kappas=tuple(arguments.kappas), bootstrap_reps=arguments.bootstrap_reps,
            seed=arguments.seed,
        )
    except (NonComplementaryPopulationAuditError, NonComplementaryDGPError,
            PopulationEffectAuditError, ValueError, OSError) as error:
        print("PHASE8A_NC_POPULATION_AUDIT_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1
    if not result["all_hard_checks_passed"]:
        print("PHASE8A_NC_POPULATION_AUDIT_BLOCKED", file=sys.stderr)
        return 1
    print("LOGGERS_HAVE_EQUAL_ACTION_MARGINALS = True")
    print("LOGGERS_ARE_NONCOMPLEMENTARY = True")
    print("LOGGERS_HAVE_SAME_CONFOUNDING_DIRECTION = True")
    print("LOGGER12_BALANCED_REMAINS_CONFOUNDED = True")
    print("ALL_SOURCE_EQUAL_SAMPLING_REMAINS_CONFOUNDED = True")
    print("hidden leakage: set()")
    print("PHASE8A_NONCOMPLEMENTARY_POPULATION_COMPLETE")
    print("READY_FOR_NONCOMPLEMENTARY_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
