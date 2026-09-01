"""Materialize the frozen 2048-anchor direct-reward input grid for Phase 8C."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.direct_reward_materialization import (  # noqa: E402
    materialize_frozen_direct_reward_grid,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8anc-root", type=Path, required=True)
    parser.add_argument("--legacy-direct-root", type=Path, required=True)
    parser.add_argument("--lambda-grid-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        manifest = materialize_frozen_direct_reward_grid(
            args.phase8anc_root, args.legacy_direct_root, args.lambda_grid_file,
            args.output_root, split_seed=args.split_seed)
    except Exception as exc:
        print("PHASE8C_DIRECT_REWARD_MATERIALIZATION_BLOCKED")
        print(f"BLOCKING ERROR: {exc}")
        raise
    print(f"materialized scenarios: {manifest['scenario_count']}")
    print(f"anchors per scenario: {manifest['analyzed_anchor_count']}")
    print("PHASE8C_DIRECT_REWARD_MATERIALIZATION_COMPLETE")
    print("PHASE8C_FORMAL_INPUTS_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

