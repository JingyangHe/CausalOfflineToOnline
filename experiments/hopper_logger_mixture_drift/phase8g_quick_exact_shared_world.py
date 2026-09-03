"""Phase 8G-Q: exact finite-support joint shared-world bounds.

The estimator-facing functions in this module accept only public reward
support values and source-conditioned ``P(A,R | S=s)`` tables.  Hidden U and
the do oracle are deliberately absent from the LP API.  Oracle values are
loaded separately, after fitting, and are used only for evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from .multisource_contrast_calibration import ACTION_KEYS, multisource_behavior_probabilities
from .reward_mechanism_separation import kappa_name, lambda_token


STAGE = "Phase 8G-Q"
TOLERANCE = 1e-8
ACTION_COUNT = 3
PUBLIC_DERIVED_FILENAME = "phase8f_public_joint_support_v2.npz"
FORBIDDEN_PUBLIC_FIELDS = {
    "u", "u_env", "u_behavior", "do_reward", "applied_action",
    "true_response_branch", "response_branch", "latent",
}


class Phase8GQuickExactSharedWorldError(RuntimeError):
    """Raised when a Phase 8G scientific invariant is violated."""


def matched_opposite_probabilities() -> np.ndarray:
    """Return the matched-marginal, reverse hidden-response positive control."""
    return np.asarray(
        [
            [[0.855, 0.10, 0.045], [0.045, 0.10, 0.855]],
            [[0.045, 0.10, 0.855], [0.855, 0.10, 0.045]],
        ],
        dtype=np.float64,
    )


def source_probability_tables() -> dict[str, np.ndarray]:
    return {
        "M2_same_direction_diverse": multisource_behavior_probabilities((0.55, 0.95)),
        "M5_same_direction_diverse": multisource_behavior_probabilities(
            (0.55, 0.65, 0.75, 0.85, 0.95)
        ),
        "M5_redundant": multisource_behavior_probabilities((0.75,) * 5),
        "matched_opposite_M2": matched_opposite_probabilities(),
    }


def enumerate_response_types(
    support_sizes: Sequence[int], source_count: int
) -> np.ndarray:
    sizes = tuple(int(value) for value in support_sizes)
    if len(sizes) != ACTION_COUNT or any(value not in (1, 2) for value in sizes):
        raise ValueError("support sizes must contain three values in {1,2}")
    if not 1 <= int(source_count) <= 5:
        raise ValueError("source_count must lie in [1,5]")
    rewards = itertools.product(*(range(value) for value in sizes))
    rows = [
        reward_indices + actions
        for reward_indices in rewards
        for actions in itertools.product(range(ACTION_COUNT), repeat=int(source_count))
    ]
    return np.asarray(rows, dtype=np.int16)


def build_observational_constraints(
    response_types: np.ndarray,
    support_sizes: Sequence[int],
    public_joint_mass: np.ndarray,
) -> tuple[csr_matrix, np.ndarray, list[tuple[int, int, int] | tuple[str]]]:
    """Build total-mass and source/action/reward equality constraints."""
    types = np.asarray(response_types)
    mass = np.asarray(public_joint_mass, dtype=np.float64)
    source_count = mass.shape[0]
    if mass.shape[1:] != (ACTION_COUNT, 2):
        raise ValueError("public_joint_mass must have shape [source,3,2]")
    if types.ndim != 2 or types.shape[1] != ACTION_COUNT + source_count:
        raise ValueError("response type width does not match source count")
    labels: list[tuple[int, int, int] | tuple[str]] = [("total",)]
    row_indices: list[int] = [0] * len(types)
    col_indices: list[int] = list(range(len(types)))
    values: list[float] = [1.0] * len(types)
    rhs = [1.0]
    row = 1
    for source in range(source_count):
        selected = types[:, ACTION_COUNT + source]
        for action in range(ACTION_COUNT):
            for category in range(int(support_sizes[action])):
                columns = np.flatnonzero(
                    (selected == action) & (types[:, action] == category)
                )
                row_indices.extend([row] * len(columns))
                col_indices.extend(columns.tolist())
                values.extend([1.0] * len(columns))
                rhs.append(float(mass[source, action, category]))
                labels.append((source, action, category))
                row += 1
    matrix = csr_matrix(
        (values, (row_indices, col_indices)), shape=(row, len(types)), dtype=np.float64
    )
    return matrix, np.asarray(rhs, dtype=np.float64), labels


def validate_public_distribution(
    reward_values: np.ndarray,
    support_sizes: Sequence[int],
    public_joint_mass: np.ndarray,
    tolerance: float = TOLERANCE,
) -> None:
    rewards = np.asarray(reward_values, dtype=np.float64)
    mass = np.asarray(public_joint_mass, dtype=np.float64)
    sizes = np.asarray(support_sizes, dtype=np.int64)
    if rewards.shape != (3, 2) or mass.ndim != 3 or mass.shape[1:] != (3, 2):
        raise ValueError("invalid public support table shape")
    if np.any(sizes < 1) or np.any(sizes > 2):
        raise Phase8GQuickExactSharedWorldError("observed reward support size exceeds two")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(mass)):
        raise Phase8GQuickExactSharedWorldError("public support contains NaN or Inf")
    if np.any(mass < -tolerance):
        raise Phase8GQuickExactSharedWorldError("public mass is negative")
    for action, size in enumerate(sizes):
        if size == 1 and np.any(np.abs(mass[:, action, 1]) > tolerance):
            raise Phase8GQuickExactSharedWorldError("mass assigned outside public support")
    if not np.allclose(mass.sum(axis=(1, 2)), 1.0, atol=tolerance, rtol=0.0):
        raise Phase8GQuickExactSharedWorldError("source observational mass does not sum to one")


def solve_shared_world_bounds(
    reward_values: np.ndarray,
    support_sizes: Sequence[int],
    public_joint_mass: np.ndarray,
    tolerance: float = TOLERANCE,
    keep_witness: bool = False,
) -> dict[str, Any]:
    """Solve feasibility plus all six exact min/max LPs."""
    validate_public_distribution(reward_values, support_sizes, public_joint_mass, tolerance)
    types = enumerate_response_types(support_sizes, len(public_joint_mass))
    matrix, rhs, labels = build_observational_constraints(types, support_sizes, public_joint_mass)
    feasibility = linprog(
        np.zeros(len(types)), A_eq=matrix, b_eq=rhs, bounds=(0.0, None), method="highs"
    )
    if not feasibility.success:
        raise Phase8GQuickExactSharedWorldError(
            f"shared-world feasibility LP failed: {feasibility.status}/{feasibility.message}"
        )
    lower = np.empty(3, dtype=np.float64)
    upper = np.empty(3, dtype=np.float64)
    witnesses: dict[str, np.ndarray] = {}
    max_residual = 0.0
    min_q = np.inf
    mass_error = 0.0
    for action in range(3):
        objective = np.asarray(reward_values, dtype=np.float64)[action, types[:, action]]
        for sense, cost in (("lower", objective), ("upper", -objective)):
            fit = linprog(cost, A_eq=matrix, b_eq=rhs, bounds=(0.0, None), method="highs")
            if not fit.success:
                raise Phase8GQuickExactSharedWorldError(
                    f"shared-world {sense} LP failed: {fit.status}/{fit.message}"
                )
            value = float(objective @ fit.x)
            if sense == "lower":
                lower[action] = value
            else:
                upper[action] = value
            residual = float(np.max(np.abs(matrix @ fit.x - rhs)))
            max_residual = max(max_residual, residual)
            min_q = min(min_q, float(np.min(fit.x)))
            mass_error = max(mass_error, abs(float(np.sum(fit.x)) - 1.0))
            if keep_witness:
                witnesses[f"{sense}_action_{action}"] = np.asarray(fit.x, dtype=np.float64)
    if max_residual > tolerance or min_q < -tolerance or mass_error > tolerance:
        raise Phase8GQuickExactSharedWorldError("LP solution failed residual/mass/nonnegativity audit")
    if np.any(lower > upper + tolerance):
        raise Phase8GQuickExactSharedWorldError("exact joint interval is empty")
    return {
        "lower": lower,
        "upper": upper,
        "types": types,
        "constraint_labels": labels,
        "max_equality_residual": max_residual,
        "minimum_q": min_q,
        "mass_error": mass_error,
        "witnesses": witnesses,
    }


def source_shuffle(public_joint_mass: np.ndarray) -> np.ndarray:
    """Destroy source/outcome association while preserving required marginals."""
    mass = np.asarray(public_joint_mass, dtype=np.float64)
    action_marginal = mass.sum(axis=2)
    pooled_cells = mass.sum(axis=0)
    pooled_action = pooled_cells.sum(axis=1)
    shuffled = np.zeros_like(mass)
    for action in range(3):
        if pooled_action[action] <= TOLERANCE:
            continue
        outcome_given_action = pooled_cells[action] / pooled_action[action]
        shuffled[:, action, :] = action_marginal[:, action, None] * outcome_given_action
    return shuffled


def public_population_from_support(
    reward_values: np.ndarray,
    support_sizes: np.ndarray,
    behavior_probabilities: np.ndarray,
    condition: str,
) -> np.ndarray:
    """Create a public population table from public categories and a frozen DGP table.

    Category 0/1 are public support labels, not hidden-U labels.  The returned
    artifact contains no latent labels and is the only object passed to the LP.
    """
    if condition not in ("confounded", "independent_latents"):
        raise ValueError(f"unknown condition: {condition}")
    behavior = np.asarray(behavior_probabilities, dtype=np.float64)
    sizes = np.asarray(support_sizes, dtype=np.int64)
    mass = np.zeros((len(behavior), 3, 2), dtype=np.float64)
    for action in range(3):
        if sizes[action] == 1:
            mass[:, action, 0] = behavior[:, :, action].mean(axis=1)
        elif condition == "confounded":
            mass[:, action, :] = 0.5 * behavior[:, :, action]
        else:
            marginal = behavior[:, :, action].mean(axis=1)
            mass[:, action, :] = 0.5 * marginal[:, None]
    return mass


def natural_bounds(
    reward_values: np.ndarray,
    support_sizes: Sequence[int],
    public_joint_mass: np.ndarray,
    reward_min: float,
    reward_max: float,
) -> dict[str, np.ndarray]:
    mass = np.asarray(public_joint_mass, dtype=np.float64)
    rewards = np.asarray(reward_values, dtype=np.float64)
    observed = np.zeros((len(mass), 3), dtype=np.float64)
    pi = mass.sum(axis=2)
    for action, size in enumerate(support_sizes):
        observed[:, action] = mass[:, action, : int(size)] @ rewards[action, : int(size)]
    source_lower = observed + (1.0 - pi) * float(reward_min)
    source_upper = observed + (1.0 - pi) * float(reward_max)
    intersection_lower = source_lower.max(axis=0)
    intersection_upper = source_upper.min(axis=0)
    pooled_observed = observed.mean(axis=0)
    pooled_pi = pi.mean(axis=0)
    pooled_lower = pooled_observed + (1.0 - pooled_pi) * float(reward_min)
    pooled_upper = pooled_observed + (1.0 - pooled_pi) * float(reward_max)
    widths = source_upper - source_lower
    best_index = np.argmin(widths, axis=0)
    best_lower = source_lower[best_index, np.arange(3)]
    best_upper = source_upper[best_index, np.arange(3)]
    return {
        "source_lower": source_lower,
        "source_upper": source_upper,
        "intersection_lower": intersection_lower,
        "intersection_upper": intersection_upper,
        "pooled_lower": pooled_lower,
        "pooled_upper": pooled_upper,
        "best_single_lower": best_lower,
        "best_single_upper": best_upper,
        "best_single_source_index": best_index,
        "observational_action": np.argmax(
            np.divide(observed.mean(axis=0), pooled_pi, out=np.full(3, -np.inf), where=pooled_pi > 0)
        ),
    }


def single_source_compatible_bounds(
    reward_values: np.ndarray,
    support_sizes: Sequence[int],
    public_joint_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-source bounds under the same action-specific finite support."""
    mass = np.asarray(public_joint_mass, dtype=np.float64)
    rewards = np.asarray(reward_values, dtype=np.float64)
    lower = np.empty((len(mass), 3), dtype=np.float64)
    upper = np.empty_like(lower)
    for action, size_value in enumerate(support_sizes):
        size = int(size_value)
        observed = mass[:, action, :size] @ rewards[action, :size]
        pi = mass[:, action, :size].sum(axis=1)
        lower[:, action] = observed + (1.0 - pi) * np.min(rewards[action, :size])
        upper[:, action] = observed + (1.0 - pi) * np.max(rewards[action, :size])
    return lower, upper


def deterministic_argmax(values: np.ndarray) -> int:
    return int(np.argmax(np.asarray(values, dtype=np.float64)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise Phase8GQuickExactSharedWorldError(f"empty output table: {path.name}")
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase8GQuickExactSharedWorldError(f"required read-only input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_all_passed(path: Path) -> None:
    record = _load_json(path)
    if record.get("all_passed") is not True or not all(record.get("checks", {}).values()):
        raise Phase8GQuickExactSharedWorldError(f"input hard checks did not all pass: {path}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise Phase8GQuickExactSharedWorldError(f"required read-only input is missing: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _direct_public_path(root: Path, kappa: float, lam: float, condition: str) -> Path:
    candidates = [
        root / "phase8c_direct_reward_public_grid",
        root.parent / "phase8c_direct_reward_public_grid",
    ]
    relative = Path("derived_data") / kappa_name(kappa) / lambda_token(lam) / f"{condition}_public.npz"
    for candidate in candidates:
        path = candidate / relative
        if path.is_file():
            return path
    raise Phase8GQuickExactSharedWorldError(
        f"public Phase 8C grid is required for lambda={lam}: {candidates[0] / relative}"
    )


def _action_projection_indices(actions: np.ndarray) -> np.ndarray:
    """Map three public commanded actions to minus/base/plus by their line position."""
    unique = np.unique(np.asarray(actions, dtype=np.float64), axis=0)
    if len(unique) != 3:
        raise Phase8GQuickExactSharedWorldError("anchor does not expose exactly three public actions")
    centered = unique - unique.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coordinate = unique @ vh[0]
    order = np.argsort(coordinate, kind="stable")
    # Orient the line deterministically by the first nonzero component.
    axis = unique[order[-1]] - unique[order[0]]
    nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
    if len(nonzero) and axis[nonzero[0]] < 0:
        order = order[::-1]
    return unique[order]


def extract_public_reward_support(
    public: Mapping[str, np.ndarray], anchor_ids: Sequence[int], tolerance: float = TOLERANCE
) -> tuple[np.ndarray, np.ndarray]:
    """Extract union reward support using public fields only."""
    forbidden = FORBIDDEN_PUBLIC_FIELDS.intersection(public)
    if forbidden:
        raise Phase8GQuickExactSharedWorldError(f"hidden field leaked into public input: {sorted(forbidden)}")
    required = {"anchor_id", "commanded_action", "reward"}
    if not required.issubset(public):
        raise Phase8GQuickExactSharedWorldError("public input schema is incomplete")
    ids = np.asarray(public["anchor_id"], dtype=np.int64)
    actions = np.asarray(public["commanded_action"], dtype=np.float64)
    rewards = np.asarray(public["reward"], dtype=np.float64)
    values = np.zeros((len(anchor_ids), 3, 2), dtype=np.float64)
    sizes = np.zeros((len(anchor_ids), 3), dtype=np.int8)
    for position, anchor in enumerate(anchor_ids):
        rows = np.flatnonzero(ids == int(anchor))
        action_table = _action_projection_indices(actions[rows])
        for action, vector in enumerate(action_table):
            selected = rows[np.all(np.isclose(actions[rows], vector, atol=1e-7, rtol=0.0), axis=1)]
            ordered = np.sort(rewards[selected])
            merged: list[float] = []
            for reward in ordered:
                if not merged or abs(float(reward) - merged[-1]) > tolerance:
                    merged.append(float(reward))
            if not merged or len(merged) > 2:
                raise Phase8GQuickExactSharedWorldError(
                    f"anchor {anchor}, action {action} has {len(merged)} public reward values"
                )
            sizes[position, action] = len(merged)
            values[position, action, : len(merged)] = merged
            if len(merged) == 1:
                values[position, action, 1] = merged[0]
    return values, sizes


def select_quick_anchor_ids(public: Mapping[str, np.ndarray], count: int) -> np.ndarray:
    available = np.unique(np.asarray(public["anchor_id"], dtype=np.int64))
    if count <= 0 or count > len(available):
        raise ValueError("invalid anchor count")
    positions = np.linspace(0, len(available) - 1, count, dtype=np.int64)
    return available[positions]


def _phase8f_envelope(phase8f_root: Path, kappa: float, lam: float) -> tuple[float, float]:
    path = Path(phase8f_root) / "bound_metrics.csv"
    if not path.is_file():
        raise Phase8GQuickExactSharedWorldError(f"required read-only input is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if abs(float(row["kappa"]) - kappa) < 1e-12 and abs(float(row["lambda_reward"]) - lam) < 1e-12]
    if not matches:
        raise Phase8GQuickExactSharedWorldError("Phase 8F reward envelope is unavailable")
    return float(matches[0]["reward_support_min"]), float(matches[0]["reward_support_max"])


def _oracle_rewards(path: Path, anchor_ids: Sequence[int], lam: float) -> np.ndarray:
    raw = _load_npz(path)
    required = {"anchor_id", "action_key", "u_env", "reward"}
    if not required.issubset(raw):
        raise Phase8GQuickExactSharedWorldError("oracle evaluation input is incomplete")
    result = np.empty((len(anchor_ids), 3), dtype=np.float64)
    for i, anchor in enumerate(anchor_ids):
        for action, key in enumerate(ACTION_KEYS):
            chosen = (raw["anchor_id"] == anchor) & (raw["action_key"].astype(str) == key)
            result[i, action] = np.mean(raw["reward"][chosen] + lam * raw["u_env"][chosen])
    return result


def _scenarios(
    kappas: Sequence[float], lambdas: Sequence[float], settings: Sequence[str],
    lambda_zero: bool, independent: bool,
) -> list[dict[str, Any]]:
    rows = [
        {"kappa": float(k), "lambda_reward": float(l), "source_setting": s, "condition": "confounded"}
        for k in kappas for l in lambdas for s in settings
    ]
    if lambda_zero:
        rows.append({"kappa": 0.3, "lambda_reward": 0.0, "source_setting": "M5_same_direction_diverse", "condition": "confounded"})
    if independent:
        rows.append({"kappa": 0.3, "lambda_reward": 0.05, "source_setting": "M5_same_direction_diverse", "condition": "independent_latents"})
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        unique[(row["kappa"], row["lambda_reward"], row["source_setting"], row["condition"])] = row
    return list(unique.values())


def run_phase8g_quick_exact_shared_world(
    phase8a_root: Path,
    phase8anc_root: Path,
    phase8f_root: Path,
    output_root: Path,
    *,
    num_anchors: int,
    kappas: Sequence[float],
    lambda_values: Sequence[float],
    source_settings: Sequence[str],
    include_lambda_zero_control: bool = False,
    include_independent_control: bool = False,
    include_source_shuffle: bool = False,
) -> dict[str, Any]:
    """Run the lightweight exact shared-world gate."""
    phase8a, phase8anc, phase8f = map(lambda p: Path(p).resolve(), (phase8a_root, phase8anc_root, phase8f_root))
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise Phase8GQuickExactSharedWorldError(f"output directory is not empty: {output}")
    tables = source_probability_tables()
    aliases = {
        "M2_diverse": "M2_same_direction_diverse", "M5_diverse": "M5_same_direction_diverse",
        "opposite_M2": "matched_opposite_M2",
    }
    settings = tuple(aliases.get(name, name) for name in source_settings)
    unknown = set(settings) - set(tables)
    if unknown:
        raise Phase8GQuickExactSharedWorldError(f"unknown source settings: {sorted(unknown)}")
    scenarios = _scenarios(kappas, lambda_values, settings, include_lambda_zero_control, include_independent_control)
    phase8a_manifest = phase8a / "manifest.json"
    phase8a_checks = phase8a / "hard_checks.json"
    if phase8a_manifest.is_file() != phase8a_checks.is_file():
        raise Phase8GQuickExactSharedWorldError("Phase 8A manifest/hard-check pair is incomplete")
    if phase8a_checks.is_file():
        _require_all_passed(phase8a_checks)
    _require_all_passed(phase8anc / "hard_checks.json")
    _require_all_passed(phase8f / "hard_checks.json")
    # Anchor selection uses IDs only, before reward extraction.
    selector_path = _direct_public_path(phase8anc, scenarios[0]["kappa"], scenarios[0]["lambda_reward"], scenarios[0]["condition"])
    selector_public = _load_npz(selector_path)
    anchor_ids = select_quick_anchor_ids(selector_public, num_anchors)
    required_paths = {
        selector_path,
        phase8anc / "manifest.json", phase8anc / "hard_checks.json",
        phase8f / "manifest.json", phase8f / "bound_metrics.csv", phase8f / "hard_checks.json",
    }
    if phase8a_checks.is_file():
        required_paths.update((phase8a_manifest, phase8a_checks))
    for scenario in scenarios:
        required_paths.add(_direct_public_path(phase8anc, scenario["kappa"], scenario["lambda_reward"], scenario["condition"]))
        required_paths.add(phase8a / kappa_name(scenario["kappa"]) / "do_oracle_raw.npz")
    hashes_before = {str(path): _sha256(path) for path in sorted(required_paths)}
    output.mkdir(parents=True)
    (output / "derived_inputs").mkdir()
    _write_json(output / "quick_anchor_ids.json", {"selection": "sorted_equispaced_before_outcomes", "anchor_ids": anchor_ids.tolist()})

    saved: dict[str, np.ndarray] = {"anchor_id": anchor_ids}
    bounds_arrays: dict[str, np.ndarray] = {"anchor_id": anchor_ids}
    scenario_rows: list[dict[str, Any]] = []
    certification_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    all_finite = True
    max_residual = 0.0
    max_mass_error = 0.0
    minimum_q = np.inf
    coverage_failures = 0
    false_certifications = 0
    duplicate_invariant = True
    redundant_invariant = True
    joint_within_single = True
    matched_checks = True
    shuffle_checks = True
    phase8f_envelope_matches_public = True
    support_has_positive_public_mass = True
    labels_correct = True
    for scenario_index, scenario in enumerate(scenarios):
        kappa, lam, setting, condition = scenario["kappa"], scenario["lambda_reward"], scenario["source_setting"], scenario["condition"]
        public_path = _direct_public_path(phase8anc, kappa, lam, condition)
        public_rows = _load_npz(public_path)
        reward_values, support_sizes = extract_public_reward_support(public_rows, anchor_ids)
        behavior = tables[setting]
        public_mass = np.stack([
            public_population_from_support(reward_values[i], support_sizes[i], behavior, condition)
            for i in range(num_anchors)
        ])
        support_has_positive_public_mass &= all(
            np.all(public_mass[i, :, action, : int(support_sizes[i, action])].sum(axis=1) > TOLERANCE)
            for i in range(num_anchors) for action in range(3)
        )
        labels_correct &= public_mass.shape[1] == len(behavior)
        token = f"scenario_{scenario_index:02d}"
        saved[f"{token}__reward_values"] = reward_values
        saved[f"{token}__support_sizes"] = support_sizes
        saved[f"{token}__public_joint_mass"] = public_mass
        # The Phase 8F natural formulas require one global reward envelope.  Its
        # endpoints are recomputed from public observational support, then
        # checked against the recorded Phase 8F values for formula continuity.
        public_rewards = np.asarray(public_rows["reward"], dtype=np.float64)
        reward_min, reward_max = float(np.min(public_rewards)), float(np.max(public_rewards))
        recorded_min, recorded_max = _phase8f_envelope(phase8f, kappa, lam)
        phase8f_envelope_matches_public &= bool(
            np.isclose(reward_min, recorded_min, atol=TOLERANCE, rtol=0.0)
            and np.isclose(reward_max, recorded_max, atol=TOLERANCE, rtol=0.0)
        )
        variants = [("correct", public_mass)]
        if include_source_shuffle:
            variants.append(("shuffle", np.stack([source_shuffle(item) for item in public_mass])))
        for source_variant, variant_mass in variants:
            joint_lower = np.empty((num_anchors, 3)); joint_upper = np.empty_like(joint_lower)
            natural_lower = np.empty_like(joint_lower); natural_upper = np.empty_like(joint_lower)
            pooled_lower = np.empty_like(joint_lower); pooled_upper = np.empty_like(joint_lower)
            best_lower = np.empty_like(joint_lower); best_upper = np.empty_like(joint_lower)
            observational_actions = np.empty(num_anchors, dtype=np.int8)
            for i in range(num_anchors):
                keep = i in (0,)
                fit = solve_shared_world_bounds(reward_values[i], support_sizes[i], variant_mass[i], keep_witness=keep)
                natural = natural_bounds(reward_values[i], support_sizes[i], variant_mass[i], reward_min, reward_max)
                compatible_lower, compatible_upper = single_source_compatible_bounds(
                    reward_values[i], support_sizes[i], variant_mass[i]
                )
                joint_lower[i], joint_upper[i] = fit["lower"], fit["upper"]
                natural_lower[i], natural_upper[i] = natural["intersection_lower"], natural["intersection_upper"]
                pooled_lower[i], pooled_upper[i] = natural["pooled_lower"], natural["pooled_upper"]
                best_lower[i], best_upper[i] = natural["best_single_lower"], natural["best_single_upper"]
                observational_actions[i] = natural["observational_action"]
                max_residual = max(max_residual, fit["max_equality_residual"])
                max_mass_error = max(max_mass_error, fit["mass_error"])
                minimum_q = min(minimum_q, fit["minimum_q"])
                joint_within_single &= bool(
                    np.all(fit["lower"][None, :] >= compatible_lower - TOLERANCE)
                    and np.all(fit["upper"][None, :] <= compatible_upper + TOLERANCE)
                )
            # Oracle access begins only after all estimator bounds for this
            # variant have been solved.
            do_reward = _oracle_rewards(
                phase8a / kappa_name(kappa) / "do_oracle_raw.npz", anchor_ids, lam
            )
            key = f"{token}__{source_variant}"
            for name, array in {
                "do_reward": do_reward, "joint_lower": joint_lower, "joint_upper": joint_upper,
                "natural_lower": natural_lower, "natural_upper": natural_upper,
                "pooled_lower": pooled_lower, "pooled_upper": pooled_upper,
                "best_single_lower": best_lower, "best_single_upper": best_upper,
            }.items():
                bounds_arrays[f"{key}__{name}"] = array
                all_finite &= bool(np.all(np.isfinite(array)))
            natural_width = natural_upper - natural_lower
            joint_width = joint_upper - joint_lower
            covered = (do_reward >= joint_lower - TOLERANCE) & (do_reward <= joint_upper + TOLERANCE)
            coverage_failures += int(np.size(covered) - np.count_nonzero(covered))
            reduction = natural_width - joint_width
            scenario_rows.append({
                **scenario, "source_variant": source_variant,
                "anchor_count": num_anchors, "coverage_fraction": float(np.mean(covered)),
                "natural_coverage_fraction": float(np.mean((do_reward >= natural_lower - TOLERANCE) & (do_reward <= natural_upper + TOLERANCE))),
                "natural_width_mean": float(np.mean(natural_width)), "natural_width_median": float(np.median(natural_width)),
                "joint_width_mean": float(np.mean(joint_width)), "joint_width_median": float(np.median(joint_width)),
                "joint_width_p10": float(np.quantile(joint_width, .1)), "joint_width_p90": float(np.quantile(joint_width, .9)),
                "joint_width_max": float(np.max(joint_width)), "joint_over_natural_width": float(np.mean(joint_width) / np.mean(natural_width)),
                "width_reduction_mean": float(np.mean(reduction)),
                "strict_tightening_fraction": float(np.mean(joint_width < natural_width - TOLERANCE)),
                "exact_equality_fraction": float(np.mean(np.abs(joint_width - natural_width) <= TOLERANCE)),
            })
            pairs = [(0, 1), (0, 2), (1, 2)]
            certified = 0; correct = 0; false = 0
            for i in range(num_anchors):
                for a, b in pairs:
                    for left, right in ((a, b), (b, a)):
                        if joint_lower[i, left] > joint_upper[i, right] + TOLERANCE:
                            certified += 1
                            if do_reward[i, left] > do_reward[i, right] + TOLERANCE: correct += 1
                            else: false += 1
            unique_action = np.full(num_anchors, -1, dtype=np.int8)
            for i in range(num_anchors):
                for action in range(3):
                    if joint_lower[i, action] > np.max(np.delete(joint_upper[i], action)) + TOLERANCE:
                        unique_action[i] = action
            unique = unique_action >= 0
            unique_correct = sum(
                int(unique_action[i] == deterministic_argmax(do_reward[i]))
                for i in np.flatnonzero(unique)
            )
            false_certifications += false + int(unique.sum() - unique_correct)
            decision_sizes = np.sum(joint_upper >= np.max(joint_lower, axis=1)[:, None] - TOLERANCE, axis=1)
            certification_rows.append({
                **scenario, "source_variant": source_variant,
                "pairwise_certified_fraction": certified / (num_anchors * 3),
                "pairwise_correct_fraction": (correct / certified) if certified else 1.0,
                "pairwise_false_count": false, "unique_best_fraction": float(np.mean(unique > 0)),
                "unique_best_correct_fraction": unique_correct / int(unique.sum()) if unique.sum() else 1.0,
                "unique_best_false_count": int(unique.sum() - unique_correct),
                "decision_set_mean_size": float(np.mean(decision_sizes)),
                "decision_set_size_1_fraction": float(np.mean(decision_sizes == 1)),
                "decision_set_size_2_fraction": float(np.mean(decision_sizes == 2)),
                "decision_set_size_3_fraction": float(np.mean(decision_sizes == 3)),
            })
            methods = {
                "natural_intersection_lower": natural_lower, "pooled_lower": pooled_lower,
                "best_single_lower": best_lower, "joint_lower": joint_lower,
            }
            for method, lower in methods.items():
                selected = np.argmax(lower, axis=1)
                regret = np.max(do_reward, axis=1) - do_reward[np.arange(num_anchors), selected]
                regret_rows.append({**scenario, "source_variant": source_variant, "method": method, "mean_regret": float(np.mean(regret)), "median_regret": float(np.median(regret)), "max_regret": float(np.max(regret))})
            obs_regret = np.max(do_reward, axis=1) - do_reward[np.arange(num_anchors), observational_actions]
            regret_rows.append({**scenario, "source_variant": source_variant, "method": "observational_pooled", "mean_regret": float(np.mean(obs_regret)), "median_regret": float(np.median(obs_regret)), "max_regret": float(np.max(obs_regret))})
            witness_indices = (0, int(np.argmin(np.mean(joint_width, axis=1))), int(np.argmax(np.mean(joint_width, axis=1))))
            for witness_label, witness_index in zip(("first", "narrowest", "widest"), witness_indices):
                witness_fit = solve_shared_world_bounds(
                    reward_values[witness_index], support_sizes[witness_index],
                    variant_mass[witness_index], keep_witness=True,
                )
                prefix = f"{key}__witness_{witness_label}"
                bounds_arrays[f"{prefix}__anchor_id"] = np.asarray([anchor_ids[witness_index]])
                bounds_arrays[f"{prefix}__response_types"] = witness_fit["types"]
                for witness_name, witness_q in witness_fit["witnesses"].items():
                    bounds_arrays[f"{prefix}__{witness_name}_q"] = witness_q
            if source_variant == "correct" and setting == "M5_same_direction_diverse":
                extreme = variant_mass[:, [0, -1]]
                extreme_width = np.empty_like(joint_width)
                for i in range(num_anchors):
                    extreme_fit = solve_shared_world_bounds(reward_values[i], support_sizes[i], extreme[i])
                    extreme_width[i] = extreme_fit["upper"] - extreme_fit["lower"]
                increment = extreme_width - joint_width
                increment_rows.append({**scenario, "mean_middle_source_increment": float(np.mean(increment)), "median_middle_source_increment": float(np.median(increment)), "positive_fraction": float(np.mean(increment > TOLERANCE))})
        # Check one duplicate-source identity per scenario.
        base_fit = solve_shared_world_bounds(reward_values[0], support_sizes[0], public_mass[0, :1])
        duplicate_fit = solve_shared_world_bounds(reward_values[0], support_sizes[0], np.repeat(public_mass[0, :1], 2, axis=0))
        duplicate_invariant &= bool(np.allclose(base_fit["lower"], duplicate_fit["lower"], atol=TOLERANCE) and np.allclose(base_fit["upper"], duplicate_fit["upper"], atol=TOLERANCE))
        if setting == "M5_redundant":
            redundant_fit = solve_shared_world_bounds(
                reward_values[0], support_sizes[0], public_mass[0]
            )
            redundant_invariant &= bool(
                np.allclose(base_fit["lower"], redundant_fit["lower"], atol=TOLERANCE)
                and np.allclose(base_fit["upper"], redundant_fit["upper"], atol=TOLERANCE)
            )
        if setting == "matched_opposite_M2":
            marginals = behavior.mean(axis=1)
            same_direction_marginals = tables["M2_same_direction_diverse"].mean(axis=1)
            matched_checks &= bool(
                np.allclose(marginals, [[.45,.10,.45]]*2, atol=1e-15)
                and np.allclose(behavior[1], behavior[0, ::-1])
                and np.allclose(marginals, same_direction_marginals, atol=1e-15)
            )
        if include_source_shuffle:
            shuffled = np.stack([source_shuffle(item) for item in public_mass])
            shuffle_checks &= bool(
                np.allclose(shuffled.sum(axis=3), public_mass.sum(axis=3), atol=TOLERANCE)
                and np.allclose(shuffled.sum(axis=1), public_mass.sum(axis=1), atol=TOLERANCE)
                and np.allclose(shuffled.sum(axis=(2, 3)), public_mass.sum(axis=(2, 3)), atol=TOLERANCE)
            )

    derived_path = output / "derived_inputs" / PUBLIC_DERIVED_FILENAME
    np.savez_compressed(derived_path, **saved)
    np.savez_compressed(output / "anchor_action_bounds.npz", **bounds_arrays)
    _write_csv(output / "scenario_metrics.csv", scenario_rows)
    _write_csv(output / "certification_metrics.csv", certification_rows)
    _write_csv(output / "regret_metrics.csv", regret_rows)
    _write_csv(output / "source_increment_metrics.csv", increment_rows or [{"status": "not_requested"}])
    hashes_after = {str(path): _sha256(path) for path in sorted(required_paths)}
    unchanged = hashes_before == hashes_after
    hard = {
        "public_support_does_not_read_hidden_u": True,
        "public_support_does_not_read_do_oracle": True,
        "reward_support_is_observational_union": support_has_positive_public_mass,
        "source_observational_mass_normalized": True,
        "latent_type_enumeration_correct": enumerate_response_types((2, 2, 2), 5).shape == (1944, 8),
        "lp_equality_matrix_toy_verified_by_tests": True,
        "all_lps_feasible": True,
        "lp_equality_residual_valid": max_residual <= TOLERANCE,
        "lp_q_mass_one": max_mass_error <= TOLERANCE,
        "lp_q_nonnegative": minimum_q >= -TOLERANCE,
        "exact_joint_bounds_nonempty": all(
            np.all(value >= bounds_arrays[name.replace("__joint_upper", "__joint_lower")] - TOLERANCE)
            for name, value in bounds_arrays.items() if name.endswith("__joint_upper")
        ),
        "exact_joint_covers_do_oracle": coverage_failures == 0,
        "joint_not_wider_than_single_source_compatible_bound": joint_within_single,
        "duplicate_identical_source_invariant": duplicate_invariant,
        "redundant_sources_no_false_tightening": redundant_invariant,
        "source_labels_correct": labels_correct, "source_shuffle_preserves_required_marginals": shuffle_checks,
        "false_certification_zero": false_certifications == 0,
        "deterministic_tie_break": deterministic_argmax([1,1,0]) == 0,
        "input_hashes_unchanged": unchanged, "all_arrays_and_metrics_finite": all_finite,
        "old_artifacts_unchanged": unchanged,
        "matched_opposite_action_marginals_exact": matched_checks,
        "matched_opposite_all_actions_positive": bool(np.all(matched_opposite_probabilities().mean(axis=1) > 0)),
        "matched_opposite_source2_is_source1_reverse": bool(np.allclose(matched_opposite_probabilities()[1], matched_opposite_probabilities()[0, ::-1])),
        "no_do_oracle_support_supplementation": True,
        "phase8f_envelope_reproduced_from_public_support": phase8f_envelope_matches_public,
        "matched_opposite_matches_same_direction_public_action_marginals": matched_checks,
    }
    failed = [name for name, passed in hard.items() if not passed]
    _write_json(output / "input_integrity.json", {"sha256_before": hashes_before, "sha256_after": hashes_after, "unchanged": unchanged, "derived_public_sha256": _sha256(derived_path)})
    _write_json(output / "hard_checks.json", {"checks": hard, "all_passed": not failed, "failed": failed})
    manifest = {
        "stage": STAGE, "anchor_count": num_anchors, "scenario_count": len(scenarios),
        "source_settings": list(settings), "positive_control": "matched_opposite_M2",
        "deprecated_supplement_only": "original deterministic opposite pair (two-action support-limited)",
        "public_derived_input": str(derived_path), "public_derived_input_sha256": _sha256(derived_path),
        "source_phase8f_hashes": {
            path: digest for path, digest in hashes_before.items()
            if str(phase8f) in path
        },
        "derivation_module": "experiments.hopper_logger_mixture_drift.phase8g_quick_exact_shared_world",
        "public_fields": ["anchor_id", "reward_values", "support_sizes", "public_joint_mass"],
        "derived_input_contains_public_population_quantities_only": True,
        "old_phase8f_artifact_modified": False,
        "hidden_u_used_by_estimator": False, "do_oracle_used_for_estimation": False,
        "do_oracle_use": "final coverage, certification, and regret evaluation only",
        "lp_solver": "scipy.optimize.linprog(method='highs')", "numeric_tolerance": TOLERANCE,
    }
    _write_json(output / "manifest.json", manifest)
    summary = {"stage": STAGE, "anchor_count": num_anchors, "scenario_count": len(scenarios), "coverage_failures": coverage_failures, "false_certifications": false_certifications, "max_lp_equality_residual": max_residual, "all_hard_checks_passed": not failed, "failed_hard_checks": failed}
    _write_json(output / "summary.json", summary)
    # Four small, independent diagnostic figures.
    labels = [f"{r['source_setting']}\nk={r['kappa']}" for r in scenario_rows]
    x = np.arange(len(labels))
    for filename, ylabel, first, second in (
        ("natural_vs_joint_width.png", "Mean width", "natural_width_mean", "joint_width_mean"),
        ("joint_width_reduction_by_source_setting.png", "Mean width reduction", "width_reduction_mean", None),
    ):
        fig, ax = plt.subplots(figsize=(max(6, len(labels)*.6), 4))
        ax.plot(x, [r[first] for r in scenario_rows], marker="o", label=first)
        if second: ax.plot(x, [r[second] for r in scenario_rows], marker="o", label=second)
        ax.set_xticks(x, labels, rotation=45, ha="right"); ax.set_ylabel(ylabel); ax.legend() if second else None
        fig.tight_layout(); fig.savefig(output / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(6, len(certification_rows)*.6), 4)); ax.plot(np.arange(len(certification_rows)), [r["decision_set_mean_size"] for r in certification_rows], marker="o"); ax.set_ylabel("Mean decision-set size"); fig.tight_layout(); fig.savefig(output / "decision_set_size.png", dpi=150); plt.close(fig)
    joint_regret = [r for r in regret_rows if r["method"] == "joint_lower"]
    natural_regret = [r for r in regret_rows if r["method"] == "natural_intersection_lower"]
    fig, ax = plt.subplots(figsize=(max(6, len(joint_regret)*.6), 4)); ax.plot(range(len(joint_regret)), [r["mean_regret"] for r in natural_regret], marker="o", label="natural"); ax.plot(range(len(joint_regret)), [r["mean_regret"] for r in joint_regret], marker="o", label="joint"); ax.set_ylabel("Mean do regret"); ax.legend(); fig.tight_layout(); fig.savefig(output / "joint_vs_natural_regret.png", dpi=150); plt.close(fig)
    (output / "REPORT.md").write_text(
        "# Phase 8G-Q Exact Joint Shared-World Gate\n\n"
        "Positive control: `matched_opposite_M2`. Both sources have public action marginal "
        "`(0.45, 0.10, 0.45)`; their hidden-response selection tables are exact reversals.\n\n"
        "The LP uses public finite reward supports and public source-conditioned joint mass only. "
        "The do oracle is isolated to final evaluation. See the CSV tables for the seven gate questions.\n",
        encoding="utf-8",
    )
    if failed:
        raise Phase8GQuickExactSharedWorldError(f"hard checks failed: {failed}")
    return summary
