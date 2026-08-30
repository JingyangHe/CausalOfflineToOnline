"""CLI for the Phase 8A-C applied-action clipping audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.hopper_logger_mixture_drift.analyze_phase8a_clipping_sensitivity import (
    ClippingSensitivityAuditError,
    run_audit,
)


DEFAULT_ROOT = Path("artifacts/hopper_logger_mixture_drift/controlled_loggers_seed0_verified")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit kappa=0.3 applied-action clipping sensitivity.")
    parser.add_argument("--phase8a-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase8ar-root", type=Path,
                        default=DEFAULT_ROOT / "population_effect_review")
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_ROOT / "population_effect_review" /
                        "clipping_sensitivity_kappa_0p30")
    parser.add_argument("--kappa", type=float, default=0.3)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-anchors", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        result = run_audit(
            arguments.phase8a_root, arguments.phase8ar_root, arguments.output_root,
            kappa=arguments.kappa, bootstrap_reps=arguments.bootstrap_reps,
            seed=arguments.seed, max_anchors=arguments.max_anchors,
        )
    except (ClippingSensitivityAuditError, ValueError, OSError) as error:
        print("PHASE8A_CLIPPING_SENSITIVITY_AUDIT_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1
    if not result["all_hard_checks_passed"]:
        print("PHASE8A_CLIPPING_SENSITIVITY_AUDIT_BLOCKED", file=sys.stderr)
        return 1
    print("PHASE8A_CLIPPING_SENSITIVITY_AUDIT_COMPLETE")
    print("READY_FOR_CLIPPING_EFFECT_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
