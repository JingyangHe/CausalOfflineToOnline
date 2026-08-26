"""Train and audit the Phase 7C local empirical Hopper Joint-bound pilot."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aamas_hopper_adapter import file_sha256, load_aamas_potential
from hopper_joint_bound_pilot import (
    ACTION_LABELS,
    METHOD_NAME,
    LocalEmpiricalConditioner,
    compute_bellman_outcomes,
    evenly_spaced_indices,
    fit_train_normalization,
    load_public_data,
    neural_problem_values,
    normalize_outcomes,
    normalize_states,
    solve_local_problem,
    source3_contribution,
    target_actions,
    train_dual_residual_net,
)


FORMAL_RHO_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
PRIMARY_RHO = 1.0
GAMMA = 0.99


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _git_commit() -> str | None:
    completed = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT,
                               capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"count": int(array.size), "finite_count": 0}
    return {
        "count": int(array.size), "finite_count": int(finite.size),
        "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)), "p90": float(np.percentile(finite, 90)),
        "minimum": float(np.min(finite)), "maximum": float(np.max(finite)),
        "strict_positive_fraction": float(np.mean(finite > 1e-8)),
        "zero_fraction": float(np.mean(np.abs(finite) <= 1e-8)),
    }


def _records_to_arrays(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    arrays = {}
    for field in fields:
        if not records:
            arrays[field] = np.asarray([], dtype=str if field in {"action_label", "failure"} else float)
            continue
        values = [record[field] for record in records]
        if isinstance(values[0], str):
            arrays[field] = np.asarray(values, dtype=str)
        else:
            arrays[field] = np.asarray(values)
    return arrays


def _configure_torch(seed: int, requested_device: str):
    torch = importlib.import_module("torch")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    warnings = []
    if str(device).startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8", ":16:8",
    }:
        warnings.append("CUBLAS_WORKSPACE_CONFIG is unset; CUDA may not be bitwise reproducible")
    return torch, device, warnings


def _reward_statistics(train: dict[str, np.ndarray], continuation_dir: Path | None) -> tuple[float, float]:
    if continuation_dir is None:
        rewards = np.asarray(train["rewards"], dtype=np.float64)
        return float(np.mean(rewards)), float(np.std(rewards, ddof=1))
    with np.load(continuation_dir / "normalization.npz", allow_pickle=False) as normalization:
        return float(normalization["reward_mean"]), float(normalization["reward_std"])


def _query_local_bank(
    data: dict[str, np.ndarray], query_indices: np.ndarray, query_set: str,
    conditioner: LocalEmpiricalConditioner, normalized_states: np.ndarray,
    k: int, action_count: int, seed: int,
) -> list[dict[str, Any]]:
    bank = []
    for query_index, row in enumerate(query_indices):
        excluded = int(row) if query_set == "train" else None
        local = conditioner.query(normalized_states[row], k, excluded_train_row=excluded)
        actions, labels = target_actions(local["source_atoms"], query_index, seed, action_count)
        bank.append({
            "query_set": query_set, "query_index": query_index, "data_row": int(row),
            "query_state": normalized_states[row], "source_atoms": local["source_atoms"],
            "source_radii": local["source_radii"],
            "all_source_radius": local["all_source_radius"],
            "best_two_radius": local["best_two_radius"],
            "target_actions": actions, "action_labels": labels,
        })
    return bank


def _solve_bank(bank: list[dict[str, Any]], rho_lambda: float) -> tuple[list[dict], list[dict]]:
    records, neural_problems = [], []
    for query in bank:
        for action_index, (action, label) in enumerate(zip(
            query["target_actions"], query["action_labels"]
        )):
            solved123 = solve_local_problem(query["source_atoms"], action, rho_lambda)
            solved12 = solve_local_problem(query["source_atoms"][:2], action, rho_lambda)
            base = {
                "query_index": query["query_index"], "data_row": query["data_row"],
                "action_index": action_index, "action_label": label,
                "target_action": action, "source_radii": query["source_radii"],
                "all_source_radius": query["all_source_radius"],
                "best_two_radius": query["best_two_radius"], "rho_lambda": rho_lambda,
                "feasible": solved123["feasible"], "joint12_feasible": solved12["feasible"],
                "failure": solved123.get("failure", ""),
                "feasible_tuple_fraction": solved123["feasible_tuple_fraction"],
                "lower_solver_status": solved123.get("lower_solver_status", -1),
                "upper_solver_status": solved123.get("upper_solver_status", -1),
                "separate_lower": solved123["separate_lower"],
                "separate_upper": solved123["separate_upper"],
            }
            numeric = ("joint_lower", "joint_upper", "separate_width", "joint_width",
                       "upper_gain", "lower_gain", "width_gain", "max_marginal_residual",
                       "upper_dominance_violation", "lower_dominance_violation",
                       "interval_order_violation")
            for field in numeric:
                base[field] = solved123.get(field, np.nan)
            contribution = {
                "source3_upper_gain": np.nan, "source3_lower_gain": np.nan,
                "source3_width_gain": np.nan, "source3_upper_violation": np.nan,
                "source3_lower_violation": np.nan,
            }
            if solved123["feasible"] and solved12["feasible"]:
                contribution = source3_contribution(solved12, solved123)
            base.update(contribution)
            records.append(base)
            if solved123["feasible"]:
                neural_problems.append({
                    "query_state": query["query_state"], "target_action": action,
                    "source_atoms": query["source_atoms"], "rho_lambda": rho_lambda,
                    "prepared": solved123["prepared"], "exact": solved123,
                    "record_index": len(records) - 1,
                })
    return records, neural_problems


def _exact_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [record for record in records if record["feasible"]]
    result = {
        "problem_count": len(records), "feasible_count": len(feasible),
        "feasible_rate": len(feasible) / max(len(records), 1),
        "coupling_lp_failure_count": len(records) - len(feasible),
        "upper_gain": _summary([r["upper_gain"] for r in feasible]),
        "lower_gain": _summary([r["lower_gain"] for r in feasible]),
        "width_gain": _summary([r["width_gain"] for r in feasible]),
        "separate_width": _summary([r["separate_width"] for r in feasible]),
        "joint_width": _summary([r["joint_width"] for r in feasible]),
        "feasible_tuple_fraction": _summary([r["feasible_tuple_fraction"] for r in records]),
        "upper_dominance_violation_count": sum(r["upper_dominance_violation"] > 1e-8 for r in feasible),
        "lower_dominance_violation_count": sum(r["lower_dominance_violation"] > 1e-8 for r in feasible),
        "interval_order_violation_count": sum(r["interval_order_violation"] > 1e-8 for r in feasible),
        "maximum_marginal_residual": max((r["max_marginal_residual"] for r in feasible), default=np.nan),
    }
    for label in sorted(set(record["action_label"] for record in records)):
        selected = [r for r in feasible if r["action_label"] == label]
        result.setdefault("by_action_type", {})[label] = {
            "count": len(selected), "width_gain": _summary([r["width_gain"] for r in selected])
        }
    result["by_query_state"] = {
        str(index): _summary([r["width_gain"] for r in feasible if r["query_index"] == index])
        for index in sorted(set(r["query_index"] for r in records))
    }
    return result


def _source3_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [r for r in records if r["feasible"] and r["joint12_feasible"]]
    radii = np.asarray([r["source_radii"][2] for r in feasible])
    threshold = float(np.median(radii)) if len(radii) else np.nan
    small = [r for r in feasible if r["source_radii"][2] <= threshold]
    return {
        "comparable_count": len(feasible),
        "source3_infeasible_problem_count": sum(
            r["joint12_feasible"] and not r["feasible"] for r in records
        ),
        "upper_gain": _summary([r["source3_upper_gain"] for r in feasible]),
        "lower_gain": _summary([r["source3_lower_gain"] for r in feasible]),
        "width_gain": _summary([r["source3_width_gain"] for r in feasible]),
        "strict_contribution_fraction": float(np.mean([
            r["source3_width_gain"] > 1e-8 for r in feasible
        ])) if feasible else np.nan,
        "upper_dominance_violation_count": sum(r["source3_upper_violation"] > 1e-8 for r in feasible),
        "lower_dominance_violation_count": sum(r["source3_lower_violation"] > 1e-8 for r in feasible),
        "small_source3_radius_threshold": threshold,
        "small_source3_radius_width_gain": _summary([r["source3_width_gain"] for r in small]),
    }


def _neural_records(
    model: Any, problems: list[dict[str, Any]], exact_records: list[dict[str, Any]], device: str,
) -> list[dict[str, Any]]:
    results = []
    for problem in problems:
        values = neural_problem_values(model, problem, device)
        exact = problem["exact"]
        source = exact_records[problem["record_index"]]
        values.update({
            "query_index": source["query_index"], "action_index": source["action_index"],
            "action_label": source["action_label"],
            "upper_duality_gap": values["neural_upper_final"] - exact["joint_upper"],
            "lower_duality_gap": exact["joint_lower"] - values["neural_lower_final"],
            "upper_exact_violation": max(0.0, exact["joint_upper"] - values["neural_upper_final"]),
            "lower_exact_violation": max(0.0, values["neural_lower_final"] - exact["joint_lower"]),
            "upper_separate_violation": max(0.0, values["neural_upper_final"] - exact["separate_upper"]),
            "lower_separate_violation": max(0.0, exact["separate_lower"] - values["neural_lower_final"]),
            "interval_order_violation": max(0.0, values["neural_lower_final"] - values["neural_upper_final"]),
            "strict_improvement": (
                values["neural_upper_final"] < exact["separate_upper"] - 1e-8
                or values["neural_lower_final"] > exact["separate_lower"] + 1e-8
            ),
        })
        results.append(values)
    return results


def _neural_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_values = [value for record in records for value in record.values()
                      if isinstance(value, (float, np.floating))]
    return {
        "evaluated_count": len(records),
        "raw_upper_violation": _summary([r["raw_upper_violation"] for r in records]),
        "raw_lower_violation": _summary([r["raw_lower_violation"] for r in records]),
        "corrected_upper_violation": _summary([r["corrected_upper_violation"] for r in records]),
        "corrected_lower_violation": _summary([r["corrected_lower_violation"] for r in records]),
        "upper_duality_gap": _summary([r["upper_duality_gap"] for r in records]),
        "lower_duality_gap": _summary([r["lower_duality_gap"] for r in records]),
        "fallback_to_separate_upper_rate": float(np.mean([r["fallback_upper"] for r in records])),
        "fallback_to_separate_lower_rate": float(np.mean([r["fallback_lower"] for r in records])),
        "strict_improvement_rate": float(np.mean([r["strict_improvement"] for r in records])),
        "nan_count": int(np.isnan(numeric_values).sum()), "inf_count": int(np.isinf(numeric_values).sum()),
        "exact_certificate_violation_count": sum(
            max(r["upper_exact_violation"], r["lower_exact_violation"]) > 1e-8 for r in records
        ),
        "separate_dominance_violation_count": sum(
            max(r["upper_separate_violation"], r["lower_separate_violation"]) > 1e-8 for r in records
        ),
        "interval_order_violation_count": sum(r["interval_order_violation"] > 1e-8 for r in records),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.reward_only and not arguments.smoke:
        raise RuntimeError("formal Phase 7C must use the frozen AAMAS continuation")
    torch, device, determinism_warnings = _configure_torch(arguments.seed, arguments.device)
    data_dir, output_dir = Path(arguments.data_dir), Path(arguments.output_dir)
    train_path, audit_path = data_dir / "train_public.npz", data_dir / "audit_public.npz"
    manifest_path = data_dir / "manifest.json"
    train, audit = load_public_data(train_path), load_public_data(audit_path)
    if not arguments.smoke and (len(train["rewards"]) != 48_000 or len(audit["rewards"]) != 12_000):
        raise RuntimeError("formal Phase 7C requires 48000 train and 12000 audit rows")

    continuation_dir = None if arguments.reward_only else Path(arguments.aamas_continuation_dir)
    reward_mean, reward_std = _reward_statistics(train, continuation_dir)
    potential = None if arguments.reward_only else load_aamas_potential(continuation_dir, device)
    train_z = compute_bellman_outcomes(train, GAMMA, reward_mean, reward_std, potential)
    audit_z = compute_bellman_outcomes(audit, GAMMA, reward_mean, reward_std, potential)
    normalization = fit_train_normalization(train["observations"], train_z)
    normalized_train_states = normalize_states(train["observations"], normalization)
    normalized_audit_states = normalize_states(audit["observations"], normalization)
    normalized_train_z = normalize_outcomes(train_z, normalization)
    normalized_audit_z = normalize_outcomes(audit_z, normalization)
    if not np.all(np.isfinite(normalized_audit_z)):
        raise RuntimeError("audit Z normalization produced nonfinite values")
    conditioner = LocalEmpiricalConditioner(train, normalized_train_states, normalized_train_z)

    train_queries, audit_queries = ((16, 8) if arguments.smoke else (128, 64))
    k, action_count, neural_steps = ((4, 3, 50) if arguments.smoke else (8, 6, 3000))
    train_indices = evenly_spaced_indices(len(train["rewards"]), train_queries)
    audit_indices = evenly_spaced_indices(len(audit["rewards"]), audit_queries)
    train_bank = _query_local_bank(train, train_indices, "train", conditioner,
                                   normalized_train_states, k, action_count, arguments.seed)
    audit_bank = _query_local_bank(audit, audit_indices, "audit", conditioner,
                                   normalized_audit_states, k, action_count, arguments.seed)
    train_exact, train_problems = _solve_bank(train_bank, PRIMARY_RHO)
    exact_records, audit_problems = _solve_bank(audit_bank, PRIMARY_RHO)

    rho_grid = (PRIMARY_RHO,) if arguments.smoke else FORMAL_RHO_GRID
    rho_records = []
    for rho_lambda in rho_grid:
        sensitivity, _ = _solve_bank(audit_bank[:min(16, len(audit_bank))], rho_lambda)
        rho_records.extend(sensitivity)
    rho_summary = {}
    for rho_lambda in rho_grid:
        selected = [record for record in rho_records if record["rho_lambda"] == rho_lambda]
        exact = _exact_summary(selected)
        rho_summary[str(rho_lambda)] = {
            "feasible_local_problem_rate": exact["feasible_rate"],
            "feasible_tuple_fraction": exact["feasible_tuple_fraction"],
            "separate_mean_width": exact["separate_width"].get("mean", np.nan),
            "joint_mean_width": exact["joint_width"].get("mean", np.nan),
            "mean_width_gain": exact["width_gain"].get("mean", np.nan),
            "strict_gain_fraction": exact["width_gain"].get("strict_positive_fraction", np.nan),
            "source3": _source3_summary(selected),
            "neighbor_radius": _summary([record["all_source_radius"] for record in selected]),
            "coupling_lp_failure_count": exact["coupling_lp_failure_count"],
            "interval_order_violation_count": exact["interval_order_violation_count"],
        }

    model, curves = train_dual_residual_net(
        train_problems, neural_steps, 8, arguments.seed, device, audit_problems
    )
    neural_records = _neural_records(model, audit_problems, exact_records, device)
    exact_summary = _exact_summary(exact_records)
    summary = {
        "method_name": METHOD_NAME, "exact_primary_rho": exact_summary,
        "source3_contribution": _source3_summary(exact_records),
        "neural_dual": _neural_summary(neural_records), "rho_sensitivity": rho_summary,
        "RHO_SENSITIVITY_ONLY_NO_SELECTION": True,
        "scientific_scope": "local empirical finite-query fixed-continuation pilot",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "normalization.npz",
                        state_mean=normalization["state_mean"], state_std=normalization["state_std"],
                        z_mean=normalization["z_mean"], z_std=normalization["z_std"],
                        reward_mean=reward_mean, reward_std=reward_std)
    exact_fields = (
        "query_index", "data_row", "action_index", "action_label", "target_action",
        "source_radii", "all_source_radius", "best_two_radius", "rho_lambda", "feasible",
        "joint12_feasible", "failure", "feasible_tuple_fraction", "separate_lower",
        "lower_solver_status", "upper_solver_status", "separate_upper", "joint_lower",
        "joint_upper", "separate_width", "joint_width",
        "upper_gain", "lower_gain", "width_gain", "max_marginal_residual",
        "upper_dominance_violation", "lower_dominance_violation", "interval_order_violation",
        "source3_upper_gain", "source3_lower_gain", "source3_width_gain",
        "source3_upper_violation", "source3_lower_violation",
    )
    np.savez_compressed(output_dir / "exact_audit_results.npz",
                        **_records_to_arrays(exact_records, exact_fields))
    neural_fields = (
        "query_index", "action_index", "action_label", "raw_upper_objective",
        "raw_lower_objective", "raw_upper_violation", "raw_lower_violation",
        "corrected_upper", "corrected_lower", "neural_upper_final", "neural_lower_final",
        "fallback_upper", "fallback_lower", "corrected_upper_violation",
        "corrected_lower_violation", "upper_duality_gap", "lower_duality_gap",
        "upper_exact_violation", "lower_exact_violation", "upper_separate_violation",
        "lower_separate_violation", "interval_order_violation", "strict_improvement",
    )
    np.savez_compressed(output_dir / "neural_audit_results.npz",
                        **_records_to_arrays(neural_records, neural_fields), **curves)
    np.savez_compressed(output_dir / "rho_sensitivity.npz",
                        **_records_to_arrays(rho_records, exact_fields))
    torch.save({"model_state_dict": model.state_dict(), "architecture": [22, 128, 128, 2],
                "seed": arguments.seed, "primary_rho_lambda": PRIMARY_RHO},
               output_dir / "joint_dual_checkpoint.pt")
    _json_write(output_dir / "summary.json", summary)

    continuation_hashes = None
    if continuation_dir is not None:
        continuation_hashes = {
            "aamas_checkpoint": file_sha256(continuation_dir / "aamas_checkpoint.pt"),
            "normalization": file_sha256(continuation_dir / "normalization.npz"),
        }
    phase7a_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "phase": "7C", "method_name": METHOD_NAME, "our_git_commit": _git_commit(),
        "data_paths": {"train_public": str(train_path), "audit_public": str(audit_path),
                       "phase7a_manifest": str(manifest_path)},
        "data_sha256": {"train_public": file_sha256(train_path),
                        "audit_public": file_sha256(audit_path),
                        "phase7a_manifest": file_sha256(manifest_path)},
        "aamas_continuation_path": None if continuation_dir is None else str(continuation_dir),
        "aamas_continuation_sha256": continuation_hashes,
        "continuation_type": "reward_only_smoke" if arguments.reward_only
                             else "frozen_aamas_state_potential",
        "continuation_is_final_joint_recursion": False, "gamma": GAMMA,
        "z_normalization": {"mean": normalization["z_mean"], "std": normalization["z_std"],
                            "statistics_source": "train_public_only"},
        "state_normalization": {"mean": normalization["state_mean"].tolist(),
                                "std": normalization["state_std"].tolist(),
                                "statistics_source": "train_public_only"},
        "k_neighbors": k, "train_query_count": train_queries,
        "audit_query_count": audit_queries, "actions_per_query": action_count,
        "rho_grid": list(rho_grid), "primary_rho_lambda": PRIMARY_RHO,
        "exact_lp_solver": "scipy.optimize.linprog(method='highs')",
        "neural_architecture": [22, 128, 128, 2],
        "optimizer": {"name": "Adam", "learning_rate": 1e-3, "steps": neural_steps,
                      "problem_batch_size": 8, "gradient_clipping_norm": 10.0},
        "penalties": {"mean_squared_violation": 100.0, "maximum_violation": 10.0,
                      "parameter_l2": 1e-5},
        "source_mapping": phase7a_manifest.get("source_mapping"),
        "reward_normalization_source": "AAMAS train-only checkpoint" if continuation_dir
                                       else "train-only reward-only smoke",
        "device": str(device), "seed": arguments.seed, "python_version": platform.python_version(),
        "determinism_warnings": determinism_warnings,
        "hidden_data_accessed": False, "audit_used_for_training": False,
        "audit_used_for_checkpoint_selection": False,
        "continuous_state_conditioning_is_approximate": True,
        "rho_pathwise_certificate_available": False,
        "full_hopper_causal_coverage_claimed": False, "online_ready_potential": False,
        "global_response_range_clipping_used": False, "online_sac_run": False,
        "RHO_SENSITIVITY_ONLY_NO_SELECTION": True,
    }
    _json_write(output_dir / "manifest.json", manifest)
    return summary


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("artifacts/hopper_method_pilot/stage_seed0"))
    parser.add_argument("--aamas-continuation-dir", type=Path,
                        default=Path("artifacts/aamas_hopper_pilot/stage_seed0/seed_0"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/hopper_joint_bound_pilot/stage_seed0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reward-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    completed = run(parse_arguments())
    print("method:", completed["method_name"])
    print("LOCAL_EMPIRICAL_JOINT_BOUND_PILOT")
    print("PHASE7C_HOPPER_JOINT_BOUND_PILOT_COMPLETE")
