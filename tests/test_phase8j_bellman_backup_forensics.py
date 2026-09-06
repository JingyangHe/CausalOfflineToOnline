from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from aamas_hopper_adapter import compute_official_continuous_action_backup
from experiments.hopper_logger_mixture_drift import phase8j_bellman_backup_forensics as bf
from scripts.run_phase8j_bellman_backup_forensics import parse_arguments


def test_frozen_scope_and_cli_defaults() -> None:
    assert bf.PHASE == "Phase 8J-BF-Q"
    assert bf.METHOD == "pooled_union"
    assert bf.MODEL_SEED == 0
    assert bf.REQUIRED_CHECKPOINTS == (0, 20, 40, 60, 80, 100, 120)
    assert bf.COARSE_CHECKPOINTS == (0, 50, 100, 150, 200)
    assert bf.OPTIONAL_LATE_CHECKPOINTS == (140, 160, 180, 200)
    assert bf.SIMULATOR_LATENT_REPLICATES == 2
    args = parse_arguments(["--phase", "analyze"])
    assert tuple(args.checkpoints) == bf.REQUIRED_CHECKPOINTS
    assert args.simulator_anchor_count == 128


def test_checkpoint_protocol_accepts_only_prespecified_fine_or_coarse_grids() -> None:
    assert bf._checkpoint_protocol(bf.REQUIRED_CHECKPOINTS) == ("fine_prespecified", 20)
    assert bf._checkpoint_protocol(bf.COARSE_CHECKPOINTS) \
        == ("coarse_existing_milestones", 50)
    with pytest.raises(bf.Phase8JBellmanForensicsError):
        bf._checkpoint_protocol((0, 100, 200))


def _dummy_components(torch):
    class Behavior(torch.nn.Module):
        def forward(self, states):
            return torch.distributions.Normal(
                torch.zeros((len(states), 3), device=states.device),
                torch.full((len(states), 3), 0.4, device=states.device))

    class Zeros(torch.nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.width = width

        def forward(self, values):
            return torch.zeros((len(values), self.width), device=values.device)

    return SimpleNamespace(
        device=torch.device("cpu"), behavior_model=Behavior(),
        state_difference_model=Zeros(12), reward_model=Zeros(1),
        reward_std=1.0, reward_mean=0.0, reward_upper=1.0,
        gamma=0.99, action_separation=0.1, not_action_samples=25,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="local PyTorch DLL stack is unavailable")
def test_exact_decomposition_reproduces_actual_adapter_backup() -> None:
    torch = pytest.importorskip("torch")
    model = _dummy_components(torch)
    rng = np.random.default_rng(7)
    states = rng.normal(size=(2, 12)).astype(np.float32)
    candidates = np.tanh(rng.normal(size=(2, 3, 3))).astype(np.float32)
    noise = rng.normal(size=(6, 25, 3)).astype(np.float32)

    class Potential(torch.nn.Module):
        def forward(self, x):
            return x[:, :1] + 0.25 * x[:, 1:2]

    potential = Potential()
    mean = np.zeros((1, 12), dtype=np.float32)
    std = np.ones((1, 12), dtype=np.float32)
    context = {"torch": torch, "device": torch.device("cpu")}
    probe = {"states": states, "candidates": candidates,
             "terminated": np.asarray([False, False]), "common_noise": noise}
    observed = bf._decompose_backup(probe, model, potential, mean, std, context)
    expected = compute_official_continuous_action_backup(
        model, states, candidates,
        lambda x: x[:, 0] + 0.25 * x[:, 1], common_noise=noise)
    assert observed["final_backup"] == pytest.approx(expected, abs=1e-7)
    assert observed["observed_weight"] + observed["road_weight"] \
        == pytest.approx(np.ones((2, 3)), abs=1e-12)


def test_simulator_audit_uses_expectation_of_value_not_value_of_mean(monkeypatch) -> None:
    monkeypatch.setattr(
        bf, "_value",
        lambda network, states, mean, std, context: np.square(np.asarray(states)[:, 0]))
    latent_next = np.zeros((1, 1, 2, 12), dtype=np.float64)
    latent_next[0, 0, 0, 0] = -1.0
    latent_next[0, 0, 1, 0] = 3.0
    truth = {
        "reward": np.zeros((1, 1)),
        "next_state": latent_next.mean(axis=2),
        "next_state_by_latent": latent_next,
        "terminated_by_latent": np.zeros((1, 1, 2), dtype=bool),
    }
    decomposition = {
        "final_backup": np.asarray([[5.0 * bf.GAMMA]]),
        "model_next": np.zeros((1, 1, 12)),
        "model_reward": np.zeros((1, 1)),
        "model_next_value": np.zeros((1, 1)),
        "observed_weight": np.ones((1, 1)),
    }
    probe = {"anchor_ids": np.asarray([11])}
    rows, _ = bf._simulator_rows(
        "matched_moving_target", 0, probe, decomposition, truth,
        object(), np.zeros((1, 12), dtype=np.float32),
        np.ones((1, 12), dtype=np.float32),
        {})
    # E[V(S')] = ((-1)^2 + 3^2) / 2 = 5; V(E[S']) would incorrectly be 1.
    assert rows[0]["true_do_next_value"] == pytest.approx(5.0)


def test_inventory_reports_missing_required_checkpoints_without_fabrication(
        tmp_path: Path) -> None:
    import json

    class FakeTorch:
        @staticmethod
        def load(path, **kwargs):
            return json.loads(Path(path).read_text(encoding="utf-8"))

    for variant, directory in zip(
            bf.VARIANTS, ("moving_target", "frozen_outer_target")):
        root = tmp_path / directory
        root.mkdir(parents=True)
        for epoch, filename in ((0, "initial.pt"), (20, "epoch_20.pt")):
            (root / filename).write_text(json.dumps({"metadata": {
                "stage": bf.FVI_PHASE, "variant": variant, "epoch": epoch,
                "eligible_for_sac": False,
            }}), encoding="utf-8")
    inventory, paths, missing = bf._inventory_checkpoints(
        tmp_path, FakeTorch(), (0, 20, 40))
    assert len(inventory) == 4
    assert all(set(mapping) == {0, 20} for mapping in paths.values())
    assert missing == [(variant, 40) for variant in bf.VARIANTS]


def test_mechanism_decision_uses_growth_order_without_magnitude_cutoff() -> None:
    rows = []
    values = {
        0: (1.0, 1.0, 1.0, 1.0),
        20: (1.0, 2.0, 3.0, 5.0),
    }
    names = ("D0_observed_only", "D1_model_next_no_max",
             "D2_model_next_candidate_mean", "D3_original_candidate_max")
    for epoch, standard_deviations in values.items():
        for name, std in zip(names, standard_deviations):
            rows.append({"variant": bf.VARIANTS[0], "epoch": epoch,
                         "diagnostic_backup": name, "std": std})
    labels, evidence = bf._mechanism_labels(rows, [], [])
    assert "MODEL_NEXT_STATE_VALUE_EXTRAPOLATION_PRIMARY" in labels
    assert "MODEL_EXTRAPOLATION_AMPLIFIED_BY_MAX" in labels
    assert "CANDIDATE_MAXIMIZATION_PRIMARY" not in labels
    assert bf.VARIANTS[0] in evidence


def test_candidate_table_is_real_parquet_when_dependency_is_available(
        tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "rows.parquet"
    bf._write_parquet(path, [{"epoch": 0, "value": 1.25}])
    assert path.read_bytes()[:4] == b"PAR1"


def test_exactly_ten_required_figures_are_renderable(tmp_path: Path) -> None:
    probe, branch, maximum, ood, diagnostic, twin, amplification = [], [], [], [], [], [], []
    for variant in bf.VARIANTS:
        for epoch in (0, 20):
            for name in ("training_anchor_states", "real_logged_next_states",
                         "model_generated_next_states"):
                probe.append({"variant": variant, "epoch": epoch,
                              "probe_set": name, "p99": 1.0 + epoch})
            for name in ("observed_return", "road_return", "reward_contribution",
                         "observed_continuation_contribution",
                         "road_continuation_contribution",
                         "max_selection_increment"):
                branch.append({"variant": variant, "epoch": epoch,
                               "quantity": name, "p99": 2.0, "std": 1.0})
            for name in ("candidate_mean", "candidate_p95", "candidate_max"):
                maximum.append({"variant": variant, "epoch": epoch,
                                "candidate_statistic": name,
                                "selected_diagnostic": "not_applicable",
                                "error_scope": "not_applicable", "mean": 1.0,
                                "bellman_error_mean": 0.0})
            for name in ("negative_behavior_log_density",
                         "standardized_nearest_train_distance"):
                maximum.append({"variant": variant, "epoch": epoch,
                                "candidate_statistic": "not_applicable",
                                "selected_diagnostic": name,
                                "error_scope": "not_applicable", "mean": 1.0,
                                "bellman_error_mean": 0.0})
            for scope in ("all_candidates", "max_selected"):
                maximum.append({"variant": variant, "epoch": epoch,
                                "candidate_statistic": "not_applicable",
                                "selected_diagnostic": "not_applicable",
                                "error_scope": scope, "mean": 0.0,
                                "bellman_error_mean": 1.0})
            for subset in ("all", "top_1pct_growth", "top_5pct_growth"):
                ood.append({"variant": variant, "epoch": epoch, "subset": subset,
                            "mean_abs_model_vs_simulator_value_error": 1.0})
            for name in ("D0_observed_only", "D1_model_next_no_max",
                         "D2_model_next_candidate_mean", "D3_original_candidate_max"):
                diagnostic.append({"variant": variant, "epoch": epoch,
                                   "diagnostic_backup": name, "p99": 1.0})
            twin.append({"variant": variant, "epoch": epoch,
                         "readout": "single_critic_not_applicable",
                         "disagreement_mean": 0.0})
            if epoch:
                for ratio in ("infinity_norm", "rmse"):
                    amplification.append({"variant": variant, "epoch": epoch,
                                          "ratio_type": ratio, "ratio": 1.0})
    paths = bf._figures(
        tmp_path, probe, branch, ood, maximum, diagnostic, twin, amplification)
    assert len(paths) == 10
    assert all(Path(path).is_file() for path in paths)
