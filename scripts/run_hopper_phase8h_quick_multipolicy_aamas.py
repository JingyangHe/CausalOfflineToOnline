"""Run the Phase 8H-Q action-wise multi-policy AAMAS quick gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8h_quick_multipolicy_aamas import (  # noqa: E402
    Phase8HQuickMultipolicyAAMASError,
    run_phase8h_quick_multipolicy_aamas,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8a-root", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, required=True)
    parser.add_argument("--samples-per-anchor-source", type=int, required=True)
    parser.add_argument("--candidate-actions-per-source", type=int, required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gradient-updates", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-independent-control", action="store_true")
    parser.add_argument("--reference-sac-checkpoint", type=Path)
    parser.add_argument("--external-repo", type=Path,
                        default=Path("external/li_aamas2026"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        summary = run_phase8h_quick_multipolicy_aamas(
            arguments.phase8a_root, arguments.output_root,
            num_anchors=arguments.num_anchors,
            samples_per_anchor_source=arguments.samples_per_anchor_source,
            candidate_actions_per_source=arguments.candidate_actions_per_source,
            model_seeds=arguments.model_seeds,
            gradient_updates=arguments.gradient_updates,
            device=arguments.device,
            include_independent_control=arguments.include_independent_control,
            reference_sac_checkpoint=arguments.reference_sac_checkpoint,
            external_repo=arguments.external_repo,
        )
    except (Phase8HQuickMultipolicyAAMASError, FileNotFoundError, KeyError,
            ValueError, RuntimeError) as error:
        message = str(error)
        if "external AAMAS" in message or "official AAMAS" in message:
            print("PHASE8H_AAMAS_BASELINE_MISSING_OR_MISMATCHED", file=sys.stderr)
        print("PHASE8H_QUICK_MULTIPOLICY_AAMAS_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1
    print(f"anchors: {summary['anchor_count']}")
    print(f"models: {summary['model_count']}")
    print("PHASE8H_QUICK_MULTIPOLICY_AAMAS_COMPLETE")
    print("READY_FOR_ACTION_LEVEL_ENVELOPE_REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
