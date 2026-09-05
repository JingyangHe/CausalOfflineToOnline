from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from aamas_hopper_adapter import _import_official_module
from experiments.hopper_logger_mixture_drift.phase8h_compute_matched_online_quick import (
    GAMMA,
    discounted_shaping_sum,
)
from experiments.hopper_logger_mixture_drift.phase8j_potential_clamp_fix_quick import (
    BACKUP_METHODS,
    FORMAL_EVAL_STEPS,
    MODEL_SEEDS,
    ONLINE_METHODS,
    POTENTIAL_METHODS,
    SMOKE_STEPS,
    _health_classification,
    _load_repaired_potential,
    _train_one_repaired_potential,
    _write_csv,
    make_repaired_potential_network,
    run_online,
)


EXTERNAL = Path("external/li_aamas2026")


def test_frozen_repair_contract() -> None:
    assert MODEL_SEEDS == (0, 1, 2)
    assert POTENTIAL_METHODS == (
        "pooled_native", "pooled_union", "state_min", "action_min")
    assert len(ONLINE_METHODS) == 5
    assert SMOKE_STEPS == (0, 2_000)
    assert FORMAL_EVAL_STEPS == (0, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000)
    assert BACKUP_METHODS == {
        "pooled_native": "pooled_aamas_native_full",
        "pooled_union": "pooled_aamas_union_full",
        "state_min": "state_min_full",
        "action_min": "action_min_full",
    }


def test_old_clamp_reproduces_zero_prediction_gradient() -> None:
    torch = pytest.importorskip("torch")
    official = _import_official_module(EXTERNAL)
    old = official.Critic(12, 3, 1000, 2.0, 0.88, GAMMA)
    inputs = torch.zeros(8, 12)
    targets = torch.linspace(80.0, 96.0, 8)
    output = old(inputs).reshape(-1)
    loss = torch.nn.functional.mse_loss(output, targets)
    loss.backward()
    assert torch.allclose(output, torch.full_like(output, 88.0))
    assert old.network[-1].weight.grad.norm().item() == 0.0
    assert old.network[-1].bias.grad.norm().item() == 0.0


def test_repaired_forward_has_no_final_clamp_and_effective_gradient() -> None:
    torch = pytest.importorskip("torch")
    official = _import_official_module(EXTERNAL)
    repaired = make_repaired_potential_network(official, 0.88, 2.0, "cpu")
    torch.manual_seed(7)
    inputs = torch.randn(8, 12)
    targets = torch.linspace(80.0, 96.0, 8)
    assert torch.equal(repaired(inputs), repaired.network(inputs))
    output = repaired(inputs).reshape(-1)
    loss = torch.nn.functional.mse_loss(output, targets)
    loss.backward()
    assert torch.isfinite(output).all()
    assert repaired.network[-1].weight.grad.norm().item() > 0.0
    assert repaired.network[-1].bias.grad.norm().item() > 0.0


def test_fixed_batch_repaired_network_can_fit() -> None:
    torch = pytest.importorskip("torch")
    official = _import_official_module(EXTERNAL)
    torch.manual_seed(0)
    model = make_repaired_potential_network(official, 0.88, 2.0, "cpu")
    inputs = torch.randn(32, 12)
    targets = 2.0 * inputs[:, 0] - inputs[:, 1] + 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(inputs).reshape(-1), targets).item()
    for _ in range(100):
        loss = torch.nn.functional.mse_loss(model(inputs).reshape(-1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.mse_loss(model(inputs).reshape(-1), targets).item()
    assert final < initial * 0.2
    assert float(model(inputs).detach().std()) > 0.0


def test_health_logic_distinguishes_blocked_and_target_degeneracy() -> None:
    row = {
        "backup_target_min": 0.0, "backup_target_max": 2.0,
        "backup_target_mean": 1.0, "backup_target_std": 0.5,
        "potential_min": 0.0, "potential_max": 0.0, "potential_std": 0.0,
        "validation_residual_mae": 1.0,
        "probe_max_abs_change_from_initial": 0.0,
        "prediction_gradient_norm_final_epoch": 0.0,
    }
    status, _ = _health_classification(row)
    assert status == "TRAINING_BLOCKED"
    row["backup_target_std"] = 0.0
    status, _ = _health_classification(row)
    assert status == "TARGET_DEGENERACY_REQUIRES_MANUAL_REVIEW"


def test_current_target_and_online_loader_use_repaired_factory() -> None:
    trainer = inspect.getsource(_train_one_repaired_potential)
    loader = inspect.getsource(_load_repaired_potential)
    assert trainer.count("make_repaired_potential_network") == 2
    assert "target.eval().requires_grad_(False)" in trainer
    assert "make_repaired_potential_network" in loader
    assert "network.eval()" in loader
    assert "network.requires_grad_(False)" in loader


def test_training_keeps_own_target_and_common_source_continuation() -> None:
    source = inspect.getsource(_train_one_repaired_potential)
    assert "_TorchPotentialValue(target" in source
    assert "source_models, pooled_model" in source
    assert '"fixed_reference_value_used": False' in source


def test_online_contract_declares_isolation_and_repaired_loading() -> None:
    source = inspect.getsource(run_online)
    for text in (
        "_online_potential", "potential.network.eval().requires_grad_(False)",
        "replay_initially_empty", "replay_matches_commanded_actions",
        "returned_info_private_leak", "raw_reward_formula_match",
        "evaluation_rng_unchanged", "potential_file_blake2b_128",
    ):
        assert text in source


def test_pbrs_terminal_truncation_and_telescoping() -> None:
    terminal = discounted_shaping_sum([2.0, 3.0, 5.0], [False, True])
    truncated = discounted_shaping_sum([2.0, 3.0, 5.0], [False, False])
    assert terminal == pytest.approx(-2.0)
    assert truncated == pytest.approx(-2.0 + GAMMA ** 2 * 5.0)


def test_no_external_source_edit_is_required() -> None:
    source = inspect.getsource(make_repaired_potential_network)
    assert "class RepairedPotentialCritic(official.Critic)" in source
    assert "return self.network(state)" in source
    forward_line = next(line.strip() for line in source.splitlines()
                        if line.strip().startswith("return self.network"))
    assert forward_line == "return self.network(state)"


def test_csv_supports_final_only_audit_fields(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.csv"
    _write_csv(destination, [{"step": 0}, {"step": 2_000, "run_complete": True}])
    text = destination.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "step,run_complete"
    assert text.splitlines()[-1] == "2000,True"
