from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / (
    "zoology/experiments/flash_vqg/scripts/"
    "20260731-01-selected-read-warp-mqar-screen"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_common(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    monkeypatch.setenv("MQAR_SELECTED_WARP_RUN_TAG", "pytest")
    common = load_module("selected_warp_common_test", EXPERIMENT / "common.py")
    monkeypatch.setitem(sys.modules, "common", common)
    return common


def test_matrix_is_seed123_three_point(monkeypatch):
    common = load_common(monkeypatch)
    assert common.SEEDS == (123,)
    assert common.FORMAL_ORDER == (
        ("s1-head8192", 123),
        ("r1a-owner-w2", 123),
        ("r1b-preproject-w2", 123),
    )


def test_configs_only_change_selected_backward(monkeypatch):
    common = load_common(monkeypatch)
    experiment = load_module("selected_warp_experiment_test", EXPERIMENT / "experiment.py")
    baseline = experiment.build_config(common.BASELINE, 123, "screen")
    for variant in common.VARIANTS[1:]:
        candidate = experiment.build_config(variant, 123, "screen")
        differences = experiment.config_differences(baseline, candidate)
        assert len(differences) == 1
        assert differences[0].endswith(
            "fox_gd_residual_selected_read_backward_backend"
        )
        kwargs = experiment.BASE.BASE._find_flash_kwargs(candidate.model)
        assert kwargs["fox_gd_residual_selected_read_backward_backend"] == common.BACKENDS[variant]
        assert kwargs["fox_gd_residual_selected_read_chunk_size"] == 8192
        assert kwargs["fox_gd_residual_builder"] == "grouped_chunk_torch_ref"


def test_screen_uses_block64_bf16_one_epoch(monkeypatch):
    common = load_common(monkeypatch)
    experiment = load_module("selected_warp_screen_test", EXPERIMENT / "experiment.py")
    for variant in common.VARIANTS:
        config = experiment.build_config(variant, 123, "screen")
        kwargs = experiment.BASE.BASE._find_flash_kwargs(config.model)
        assert config.precision == "amp_bfloat16"
        assert config.max_epochs == 1
        assert config.max_train_steps is None
        assert kwargs["block_len"] == 64
        assert kwargs["fox_gd_residual_remat_mode"] == "post_phase1"


def test_runtime_audit_rejects_fallback_or_persistent(monkeypatch):
    load_common(monkeypatch)
    experiment = load_module("selected_warp_audit_test", EXPERIMENT / "experiment.py")
    states = {
        "layer": {
            "fox_gd_residual_triton_runtime_audit": {
                "selected_calls": 4,
                "selected_recompute_calls": 4,
                "persistent_calls": 0,
                "actual_core_dtype": "float32",
            }
        }
    }
    assert experiment.runtime_audit(states)["passed"] is True
    states["layer"]["fox_gd_residual_triton_runtime_audit"]["selected_fallbacks"] = 1
    assert experiment.runtime_audit(states)["passed"] is False
