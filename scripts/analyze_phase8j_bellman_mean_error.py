"""Run the read-only Phase 8J-BME-Q Bellman mean-error audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hopper_logger_mixture_drift.phase8j_bellman_mean_error import (  # noqa: E402
    DEFAULT_FIX_ROOT,
    DEFAULT_FVI_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    REQUESTED_EPOCHS,
    BellmanMeanErrorAuditError,
    run_audit,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8j-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase8j-fix-root", type=Path, default=DEFAULT_FIX_ROOT)
    parser.add_argument("--phase8j-fvi-root", type=Path, default=DEFAULT_FVI_ROOT)
    parser.add_argument("--external-repo", type=Path, default=Path("external/li_aamas2026"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", nargs="+", type=int, default=list(REQUESTED_EPOCHS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        result = run_audit(
            phase8j_root=args.phase8j_root,
            output_root=args.output_root,
            model_seed=args.model_seed,
            requested_epochs=args.epochs,
            fix_root=args.phase8j_fix_root,
            fvi_root=args.phase8j_fvi_root,
            external_repo=args.external_repo,
            device=args.device,
        )
        print("PHASE8J_BELLMAN_MEAN_ERROR_AUDIT_COMPLETE")
        print(result["decision"])
        return 0
    except (BellmanMeanErrorAuditError, FileNotFoundError, KeyError, ValueError,
            RuntimeError) as error:
        print("PHASE8J_BELLMAN_MEAN_ERROR_AUDIT_BLOCKED", file=sys.stderr)
        print(f"BLOCKING ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
