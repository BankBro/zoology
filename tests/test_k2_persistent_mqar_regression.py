from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "zoology/experiments/flash_vqg/scripts/20260730-04-k2-persistent-scan-mqar-regression"


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
    monkeypatch.setenv("MQAR_K2_PERSISTENT_RUN_TAG", "pytest")
    common = load_module("k2_mqar_common_test", EXPERIMENT / "common.py")
    monkeypatch.setitem(sys.modules, "common", common)
    return common


def test_matrix_is_three_seed_paired(monkeypatch):
    common = load_common(monkeypatch)
    assert common.SEEDS == (123, 124, 125)
    assert common.FORMAL_ORDER == (
        ("p0-a1-block64", 123),
        ("k2-persistent-p8", 123),
        ("p0-a1-block64", 124),
        ("k2-persistent-p8", 124),
        ("p0-a1-block64", 125),
        ("k2-persistent-p8", 125),
    )


def test_configs_only_change_builder(monkeypatch):
    common = load_common(monkeypatch)
    experiment = load_module("k2_mqar_experiment_test", EXPERIMENT / "experiment.py")
    p0 = experiment.build_config("p0-a1-block64", 123, "formal")
    k2 = experiment.build_config("k2-persistent-p8", 123, "formal")
    differences = experiment.BASE.config_differences(p0, k2)
    assert len(differences) == 1
    assert differences[0].endswith("fox_gd_residual_builder")
    for variant, config in (("p0-a1-block64", p0), ("k2-persistent-p8", k2)):
        kwargs = experiment.BASE.BASE._find_flash_kwargs(config.model)
        assert config.precision == "amp_bfloat16"
        assert config.max_epochs == 4
        assert kwargs["block_len"] == 64
        assert kwargs["fox_gd_residual_remat_mode"] == "post_phase1"
        assert kwargs["fox_gd_residual_builder"] == common.BUILDERS[variant]
        assert kwargs["fox_gd_residual_persistent_tile_blocks"] == 8
        assert kwargs["fox_gd_residual_selected_read_backward_backend"] == "triton_deterministic"


def test_screen_is_one_epoch_from_canonical_init(monkeypatch):
    load_common(monkeypatch)
    experiment = load_module("k2_mqar_experiment_screen_test", EXPERIMENT / "experiment.py")
    config = experiment.build_config("k2-persistent-p8", 123, "screen")
    assert config.max_epochs == 1
    assert config.max_train_steps is None
    assert config.init_checkpoint_strict is True


def test_runtime_audit_distinguishes_persistent_path(monkeypatch):
    load_common(monkeypatch)
    experiment = load_module("k2_mqar_experiment_audit_test", EXPERIMENT / "experiment.py")

    def state(persistent: int):
        return {
            "layer": {
                "fox_gd_residual_triton_runtime_audit": {
                    "selected_calls": 4,
                    "selected_recompute_calls": 4,
                    "persistent_calls": persistent,
                    "actual_core_dtype": "float32",
                }
            }
        }

    assert experiment.runtime_audit(state(0), "p0-a1-block64")["passed"] is True
    assert experiment.runtime_audit(state(4), "k2-persistent-p8")["passed"] is True
    assert experiment.runtime_audit(state(0), "k2-persistent-p8")["passed"] is False


def test_quality_gate_requires_every_seed_and_matching_fla(monkeypatch):
    load_common(monkeypatch)
    analyze = load_module("k2_mqar_analyze_test", EXPERIMENT / "analyze.py")
    base = {
        "standard_delta": 0.0,
        "extrapolation_macro_delta": 0.0,
        "fla_config_match": True,
    }
    rows = [{**base, "seed": seed} for seed in (123, 124, 125)]
    assert analyze.quality_decision(rows, True)["status"] == "passed"
    rows[1]["standard_delta"] = -0.011
    assert analyze.quality_decision(rows, True)["status"] == "quality_rejected"
    rows[1]["standard_delta"] = 0.0
    rows[2]["fla_config_match"] = False
    assert analyze.quality_decision(rows, True)["status"] == "requires_fla_replay"

