from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260801-01-fastest-flash-vs-gdn-mqar"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_experiment():
    os.environ["MQAR_FASTEST_GDN_RUN_TAG"] = "pytest"
    common = load_module("common", SCRIPT_DIR / "common.py")
    experiment = load_module("fastest_gdn_mqar_experiment_test", SCRIPT_DIR / "experiment.py")
    return common, experiment


def test_formal_matrix_and_evaluation_contract():
    common, _experiment = load_experiment()
    assert len(common.FORMAL_ORDER) == 9
    assert set(common.FORMAL_ORDER) == {
        (arm, seed) for arm in common.ARMS for seed in common.SEEDS
    }
    assert len(common.STANDARD_CASES) == 8
    assert len(common.LONGER_CASES) == 5
    assert len(common.EVAL_CASES) == 13
    assert common.STANDARD_CASES[-1][:3] == (1024, 256, 1000)
    assert common.LONGER_CASES[0][:3] == (1024, 256, 500)


def test_flash_arms_share_model_contract_and_change_registered_backends():
    common, experiment = load_experiment()
    fastest = experiment.build_config(common.FASTEST, 123, "formal")
    canonical = experiment.build_config(common.CANONICAL, 123, "formal")
    fastest_audit = experiment.model_audit(fastest, common.FASTEST)
    canonical_audit = experiment.model_audit(canonical, common.CANONICAL)
    assert fastest_audit["state_sha256"] == canonical_audit["state_sha256"]
    assert fastest_audit["trainable_parameters"] == 1_160_390
    assert canonical_audit["trainable_parameters"] == 1_160_390
    assert fastest_audit["flash_kwargs"]["fox_gd_residual_builder"] == "persistent_scan_triton"
    assert fastest_audit["flash_kwargs"]["fox_gd_residual_persistent_backward_backend"] == "fixed_slot_vjp"
    assert fastest_audit["flash_kwargs"]["fox_gd_residual_geometry_backend"] == "head_grouped"
    assert fastest_audit["flash_kwargs"]["fox_gd_residual_selected_read_forward_backend"] == "hoisted_w2"
    assert canonical_audit["flash_kwargs"]["fox_gd_residual_builder"] == "grouped_chunk_torch_ref"
    assert canonical_audit["flash_kwargs"]["fox_gd_residual_selected_read_backward_backend"] == "triton_deterministic_s1_head"


def test_gdn_and_flash_training_exposure_match():
    common, experiment = load_experiment()
    configs = {
        arm: experiment.build_config(arm, 125, "formal") for arm in common.ARMS
    }
    for config in configs.values():
        assert config.precision == "amp_bfloat16"
        assert tuple(config.data.batch_size) == (64, 16)
        assert config.gradient_accumulation_steps == 4
        assert config.max_epochs == 4
        assert config.validations_per_epoch == 4
        assert config.early_stopping_metric is None
    gdn_audit = experiment.model_audit(configs[common.GDN], common.GDN)
    assert gdn_audit["trainable_parameters"] == 1_335_942
    assert gdn_audit["active_state_capacity"] == 131_072
