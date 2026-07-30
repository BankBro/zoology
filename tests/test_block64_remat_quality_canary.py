from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260730-03-a1-block64-remat-quality-canary"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    monkeypatch.setenv("MQAR_BLOCK64_REMAT_RUN_TAG", "pytest")
    common = load_module("block64_remat_common_test", EXPERIMENT / "common.py")
    monkeypatch.setitem(sys.modules, "common", common)
    return load_module("block64_remat_experiment_test", EXPERIMENT / "experiment.py")


def test_variants_only_change_remat(monkeypatch):
    experiment = load_experiment(monkeypatch)
    a0 = experiment.build_config("a0-block64", "screen")
    a1 = experiment.build_config("a1-block64", "screen")
    differences = experiment.config_differences(a0, a1)
    assert len(differences) == 1
    assert differences[0].endswith("fox_gd_residual_remat_mode")
    kwargs = experiment.BASE._find_flash_kwargs(a1.model)
    assert kwargs["block_len"] == 64
    assert kwargs["fox_gd_residual_write_topk"] == 4
    assert kwargs["fox_remote_read_topk"] == 16
    assert kwargs["fox_gd_residual_selected_read_backward_backend"] == "triton_deterministic"


def test_trajectory_gate_requires_replay_for_hash_divergence(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    monkeypatch.setenv("MQAR_BLOCK64_REMAT_RUN_TAG", "pytest")
    common = load_module("block64_remat_common_eval_test", EXPERIMENT / "common.py")
    monkeypatch.setitem(sys.modules, "common", common)
    experiment = load_module("experiment", EXPERIMENT / "experiment.py")
    monkeypatch.setitem(sys.modules, "experiment", experiment)
    evaluate = load_module("block64_remat_evaluate_test", EXPERIMENT / "evaluate.py")
    config = {"best_config": {"kwargs": {"BT": 64}, "num_warps": 4}}
    base = {
        "telemetry": {"records": [{"step": 1, "loss": 1.0}]},
        "resume": {"model_state_sha256": "a", "optimizer_state_sha256": "b"},
        "gate_autotune": {"layer_norm_gated_bwd_kernel": config},
    }
    candidate = {
        **base,
        "resume": {"model_state_sha256": "different", "optimizer_state_sha256": "b"},
    }
    gate = evaluate.trajectory_gate(
        {"a0-block64": base, "a1-block64": candidate}
    )
    assert gate["requires_fla_replay"] is True
    assert gate["passed"] is False


def test_longest_eval_cases_fit_2080ti(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    monkeypatch.setenv("MQAR_BLOCK64_REMAT_RUN_TAG", "pytest")
    common = load_module("block64_remat_common_batch_test", EXPERIMENT / "common.py")
    longest = [case for case in common.LONGER_CASES if case[0] == 8190]
    assert len(longest) == 2
    assert {case[2] for case in longest} == {4}
