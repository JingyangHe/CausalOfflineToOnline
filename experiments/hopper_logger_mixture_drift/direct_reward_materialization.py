"""Materialize the frozen Phase 8C direct-reward public input grid.

This stage performs no model fitting and no grid selection.  It copies the
verified Phase 8A-NC rows and applies the already specified positive-control
reward channel ``reward + lambda_reward * u_env``.  Hidden variables remain in
separate audit files and are never placed in the public training artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np

from .analyze_phase8a_population_effect import (
    hash_input_files,
    input_hashes_unchanged,
    load_npz,
)
from .reward_mechanism_separation import (
    extend_or_reuse_splits,
    kappa_name,
    lambda_token,
    load_frozen_lambda_grid,
    validate_public_schema,
)
from .reward_signal_calibration import (
    CONDITIONS,
    FORBIDDEN_DERIVED_PUBLIC_FIELDS,
    make_derived_artifacts,
    validate_derived_artifacts,
)


FORMAL_KAPPAS = (0.0, 0.3)
FORMAL_ANCHOR_COUNT = 2048


class DirectRewardMaterializationError(RuntimeError):
    """Raised when frozen-grid materialization cannot be certified."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DirectRewardMaterializationError(f"required JSON is unavailable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")


def _require_all_passed(path: Path) -> None:
    value = _read_json(path)
    if value.get("all_passed") is not True or not all(value.get("checks", {}).values()):
        raise DirectRewardMaterializationError(f"input hard checks did not all pass: {path}")


def validate_source_pair(public: Mapping[str, np.ndarray], hidden: Mapping[str, np.ndarray],
                         expected_kappa: float, expected_condition: str) -> np.ndarray:
    """Validate one complete Phase 8A-NC public/hidden source pair."""
    required_public = {
        "row_id", "anchor_id", "observation", "commanded_action", "reward",
        "next_observation", "terminated", "truncated", "logger_id", "condition",
        "kappa_env",
    }
    required_hidden = {"row_id", "u_env", "u_behavior", "action_key", "logger_id"}
    if not required_public.issubset(public) or not required_hidden.issubset(hidden):
        raise DirectRewardMaterializationError("Phase 8A-NC source schema is incomplete")
    if FORBIDDEN_DERIVED_PUBLIC_FIELDS.intersection(public):
        raise DirectRewardMaterializationError("Phase 8A-NC public source exposes hidden data")
    if not np.array_equal(public["row_id"], hidden["row_id"]):
        raise DirectRewardMaterializationError("public and hidden source rows are not aligned")
    if not np.array_equal(public["logger_id"], hidden["logger_id"]):
        raise DirectRewardMaterializationError("public and hidden logger IDs are not aligned")
    if not np.isin(hidden["u_env"], (-1, 1)).all():
        raise DirectRewardMaterializationError("u_env is not binary")
    if not np.allclose(np.asarray(public["kappa_env"], dtype=np.float64),
                       expected_kappa, atol=0.0, rtol=0.0):
        raise DirectRewardMaterializationError("source kappa does not match its scenario")
    if set(np.asarray(public["condition"]).astype(str).tolist()) != {expected_condition}:
        raise DirectRewardMaterializationError("source condition does not match its scenario")
    anchors = np.unique(np.asarray(public["anchor_id"], dtype=np.int64))
    if not np.array_equal(anchors, np.arange(FORMAL_ANCHOR_COUNT)):
        raise DirectRewardMaterializationError("source does not contain exactly 2048 anchors")
    return anchors


def materialize_scenario(public: Mapping[str, np.ndarray], hidden: Mapping[str, np.ndarray],
                         lambda_reward: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply the frozen reward dose and revalidate public/audit separation."""
    derived_public, hidden_audit = make_derived_artifacts(public, hidden, lambda_reward)
    leakage = validate_derived_artifacts(derived_public, hidden_audit)
    validate_public_schema(derived_public)
    if leakage:
        raise DirectRewardMaterializationError(f"hidden leakage: {sorted(leakage)}")
    expected = (np.asarray(hidden_audit["original_reward"], dtype=np.float64)
                + float(lambda_reward) * np.asarray(hidden_audit["u_env"], dtype=np.float64))
    if not np.array_equal(np.asarray(derived_public["reward"], dtype=np.float64), expected):
        raise DirectRewardMaterializationError("direct reward formula is not exact")
    if not np.all(np.isfinite(expected)):
        raise DirectRewardMaterializationError("materialized reward is non-finite")
    return derived_public, hidden_audit


def materialize_frozen_direct_reward_grid(
    phase8anc_root: Path,
    legacy_direct_root: Path,
    lambda_grid_file: Path,
    output_root: Path,
    *,
    kappas: Sequence[float] = FORMAL_KAPPAS,
    conditions: Sequence[str] = CONDITIONS,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Create all 2048-anchor public scenarios needed by Phase 8C."""
    nc = Path(phase8anc_root).resolve()
    legacy = Path(legacy_direct_root).resolve()
    grid_path = Path(lambda_grid_file).resolve()
    output = Path(output_root).resolve()
    grid, frozen_record = load_frozen_lambda_grid(grid_path)
    kappas = tuple(map(float, kappas))
    conditions = tuple(conditions)
    if kappas != FORMAL_KAPPAS or conditions != CONDITIONS:
        raise DirectRewardMaterializationError(
            f"formal materialization requires kappas={FORMAL_KAPPAS} and conditions={CONDITIONS}")
    if not nc.is_dir() or not legacy.is_dir():
        raise DirectRewardMaterializationError("Phase 8A-NC or legacy direct-reward root is missing")
    _require_all_passed(nc / "hard_checks.json")
    _require_all_passed(legacy / "hard_checks.json")
    if output in (nc, legacy) or legacy in output.parents:
        raise DirectRewardMaterializationError(
            "output must be a new artifact outside the legacy direct-reward artifact")
    if not output.name.startswith("phase8c_direct_reward_public_grid"):
        raise DirectRewardMaterializationError(
            "output directory name must start with phase8c_direct_reward_public_grid")
    if output.exists() and any(output.iterdir()):
        raise DirectRewardMaterializationError(f"output directory is not empty: {output}")

    input_paths = [nc / "manifest.json", nc / "hard_checks.json",
                   legacy / "manifest.json", legacy / "hard_checks.json",
                   legacy / "splits.json", grid_path]
    scenario_sources: dict[tuple[float, str], tuple[Path, Path]] = {}
    for kappa in kappas:
        for condition in conditions:
            public_path = nc / kappa_name(kappa) / f"{condition}_public.npz"
            hidden_path = nc / kappa_name(kappa) / f"{condition}_hidden_audit.npz"
            scenario_sources[(kappa, condition)] = (public_path, hidden_path)
            input_paths.extend((public_path, hidden_path))
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise DirectRewardMaterializationError(f"required read-only input is missing: {missing[0]}")
    hashes_before = hash_input_files(input_paths)

    all_anchor_ids: np.ndarray | None = None
    row_count_by_scenario: dict[str, int] = {}
    generated_files: list[str] = []
    output.mkdir(parents=True)
    for kappa in kappas:
        for condition in conditions:
            public_path, hidden_path = scenario_sources[(kappa, condition)]
            public, hidden = load_npz(public_path), load_npz(hidden_path)
            anchors = validate_source_pair(public, hidden, kappa, condition)
            if all_anchor_ids is None:
                all_anchor_ids = anchors
            elif not np.array_equal(all_anchor_ids, anchors):
                raise DirectRewardMaterializationError("anchor sets differ across scenarios")
            for lambda_reward in grid:
                derived_public, hidden_audit = materialize_scenario(
                    public, hidden, lambda_reward)
                directory = output / "derived_data" / kappa_name(kappa) / lambda_token(lambda_reward)
                directory.mkdir(parents=True, exist_ok=True)
                public_output = directory / f"{condition}_public.npz"
                hidden_output = directory / f"{condition}_hidden_audit.npz"
                np.savez_compressed(public_output, **derived_public)
                np.savez_compressed(hidden_output, **hidden_audit)
                generated_files.extend((str(public_output.resolve()), str(hidden_output.resolve())))
                row_count_by_scenario[
                    f"kappa={kappa}:lambda={lambda_reward}:condition={condition}"
                ] = len(derived_public["row_id"])

    assert all_anchor_ids is not None
    existing_splits = _read_json(legacy / "splits.json")
    splits = extend_or_reuse_splits(existing_splits, all_anchor_ids, split_seed)
    _write_json(output / "splits.json", {
        **splits,
        "split_seed": split_seed,
        "provenance": "legacy 512 assignments preserved; remaining anchors deterministically assigned",
    })
    _write_json(output / "frozen_lambda_grid.json", frozen_record)

    expected_scenario_count = len(kappas) * len(grid) * len(conditions)
    scenario_complete = len(row_count_by_scenario) == expected_scenario_count
    all_rows_cover_2048 = all(count > 0 for count in row_count_by_scenario.values())
    hashes_after = hash_input_files(input_paths)
    unchanged = input_hashes_unchanged(hashes_before, hashes_after)
    hard_checks = {
        "phase8anc_hard_checks_passed": True,
        "legacy_direct_hard_checks_passed": True,
        "lambda_grid_manually_frozen": frozen_record.get("manually_frozen") is True,
        "held_out_test_not_used_to_select_grid": frozen_record.get("held_out_test_used") is False,
        "all_2048_anchors_available": np.array_equal(all_anchor_ids, np.arange(2048)),
        "all_frozen_scenarios_materialized": scenario_complete,
        "every_scenario_has_rows": all_rows_cover_2048,
        "reward_formula_exact": True,
        "public_hidden_leakage_empty": True,
        "public_and_hidden_rows_aligned": True,
        "legacy_split_assignments_preserved": all(
            set(map(int, existing_splits.get(name, ()))).issubset(set(splits[name]))
            for name in ("train", "validation", "test")
        ),
        "input_hashes_unchanged": unchanged,
        "old_artifacts_unchanged": unchanged,
    }
    failed = [name for name, passed in hard_checks.items() if not passed]
    _write_json(output / "input_integrity.json", {
        "sha256_before": hashes_before,
        "sha256_after": hashes_after,
        "unchanged": unchanged,
        "required_file_count": len(input_paths),
    })
    _write_json(output / "hard_checks.json", {
        "checks": hard_checks, "all_passed": not failed, "failed": failed,
    })
    manifest = {
        "stage": "Phase 8C-RM-INPUT",
        "phase8anc_root": str(nc),
        "legacy_direct_root": str(legacy),
        "available_anchor_count": FORMAL_ANCHOR_COUNT,
        "analyzed_anchor_count": FORMAL_ANCHOR_COUNT,
        "kappas": list(kappas),
        "lambda_grid": list(grid),
        "conditions": list(conditions),
        "scenario_count": expected_scenario_count,
        "generated_file_count": len(generated_files),
        "row_count_by_scenario": row_count_by_scenario,
        "reward_definition": "original_reward + lambda_reward * u_env",
        "model_training_performed": False,
        "automatic_grid_selection": False,
        "hidden_u_location": "isolated *_hidden_audit.npz only",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    _write_json(output / "manifest.json", manifest)
    if failed:
        raise DirectRewardMaterializationError(f"hard checks failed: {failed}")
    return manifest
