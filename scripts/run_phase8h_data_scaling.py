"""Run Phase 8H-DS source-wise data scaling diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8h_data_scaling import (  # noqa: E402
    Phase8HDataScalingError,
    run_phase8h_data_scaling,
)
from experiments.hopper_logger_mixture_drift.phase8h_quick_multipolicy_aamas import (  # noqa: E402
    Phase8HQuickMultipolicyAAMASError,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8h-root", type=Path, required=True)
    parser.add_argument("--samples-per-anchor-source", nargs="+", type=int, required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--include-n32-extra-compute-control", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        summary = run_phase8h_data_scaling(
            arguments.phase8h_root, arguments.output_root,
            samples_per_anchor_source=arguments.samples_per_anchor_source,
            model_seeds=arguments.model_seeds,
            include_n32_extra_compute_control=arguments.include_n32_extra_compute_control,
            device=arguments.device, external_repo=arguments.external_repo)
    except (Phase8HDataScalingError, Phase8HQuickMultipolicyAAMASError,
            FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print("PHASE8H_DATA_SCALING_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1
    print(f"models: {summary['model_count']}")
    print(f"scenarios: {summary['scenario_count']}")
    print("PHASE8H_DATA_SCALING_COMPLETE")
    print("READY_FOR_SAMPLE_SIZE_DIAGNOSIS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
