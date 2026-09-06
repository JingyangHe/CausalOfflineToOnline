"""Run the read-only Phase 8J-BF-Q Bellman backup forensic analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8j_bellman_backup_forensics import (  # noqa: E402
    DEFAULT_FVI_ROOT,
    DEFAULT_OUTPUT_ROOT,
    METHOD,
    MODEL_SEED,
    Phase8JBellmanForensicsError,
    REQUIRED_CHECKPOINTS,
    run_analyze,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("analyze",))
    parser.add_argument("--fvi-root", type=Path, default=DEFAULT_FVI_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(REQUIRED_CHECKPOINTS))
    parser.add_argument("--simulator-anchor-count", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        result = run_analyze(
            fvi_root=args.fvi_root, output_root=args.output_root,
            external_repo=args.external_repo, device=args.device,
            method=args.method, model_seed=args.model_seed,
            checkpoints=args.checkpoints,
            simulator_anchor_count=args.simulator_anchor_count)
        print("PHASE8J_BELLMAN_BACKUP_FORENSICS_COMPLETE")
        for label in result["mechanism_labels"]:
            print(label)
        return 0
    except (Phase8JBellmanForensicsError, FileNotFoundError, KeyError, ValueError,
            RuntimeError) as error:
        print("PHASE8J_BELLMAN_BACKUP_FORENSICS_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
