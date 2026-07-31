from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / (
    "zoology/experiments/flash_vqg/scripts/"
    "20260731-02-residual-value-bottleneck-mqar"
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
    monkeypatch.setenv("MQAR_RESIDUAL_VALUE_RUN_TAG", "pytest")
    common = load_module("residual_value_mqar_common_test", EXPERIMENT / "common.py")
    monkeypatch.setitem(sys.modules, "common", common)
    return common


def test_q0_matrix_is_three_residual_value_widths(monkeypatch):
    common = load_common(monkeypatch)
    assert common.VARIANT_DIMS == {
        "u64-a1-s1": 64,
        "u32-a1-s1": 32,
        "u16-a1-s1": 16,
    }
    assert common.SEEDS == (123, 124, 125)


def test_configs_only_change_residual_value_dim(monkeypatch):
    common = load_common(monkeypatch)
    experiment = load_module("residual_value_mqar_experiment_test", EXPERIMENT / "experiment.py")
    baseline = experiment.build_config(common.BASELINE, 123, "screen")
    for variant in common.VARIANTS[1:]:
        candidate = experiment.build_config(variant, 123, "screen")
        differences = experiment._value_differences(baseline, candidate)
        assert len(differences) == 1
        assert differences[0].endswith("fox_gd_residual_value_dim")
        kwargs = experiment.BASE.BASE._find_flash_kwargs(candidate.model)
        assert int(kwargs["fox_gd_residual_value_dim"]) == common.VARIANT_DIMS[variant]


def test_screen_uses_a1_s1_block64_bf16(monkeypatch):
    common = load_common(monkeypatch)
    experiment = load_module("residual_value_mqar_screen_test", EXPERIMENT / "experiment.py")
    for variant in common.VARIANTS:
        config = experiment.build_config(variant, 123, "screen")
        kwargs = experiment.BASE.BASE._find_flash_kwargs(config.model)
        assert config.precision == "amp_bfloat16"
        assert config.max_epochs == 1
        assert kwargs["block_len"] == 64
        assert kwargs["fox_gd_residual_remat_mode"] == "post_phase1"
        assert kwargs["fox_gd_residual_builder"] == "grouped_chunk_torch_ref"
        assert kwargs["fox_gd_residual_selected_read_backward_backend"] == (
            "triton_deterministic_s1_head"
        )
        assert all(isinstance(value, str) for value in config.resume_identity.values())


def test_derived_init_preserves_common_state(monkeypatch):
    common = load_common(monkeypatch)
    experiment = load_module("residual_value_mqar_init_test", EXPERIMENT / "experiment.py")
    canonical = torch.load(experiment.canonical_init_path(), map_location="cpu", weights_only=False)
    for variant in common.VARIANTS[1:]:
        config = experiment.build_config(variant, 123, "screen")
        derived = torch.load(experiment.init_path(variant), map_location="cpu", weights_only=False)
        for key, value in canonical["model_state_dict"].items():
            assert torch.equal(derived["model_state_dict"][key], value)
        projection_keys = experiment._projection_keys(derived["model_state_dict"])
        assert len(projection_keys) == 1
        audit = experiment.model_audit(config, variant)
        assert audit["projection_orthogonal_max_abs"] <= 1e-5
        assert audit["state_sha256"] == common.EXPECTED_STATE_HASHES[variant]


def test_evaluator_replaces_base_sources_and_event_payload(monkeypatch):
    load_common(monkeypatch)
    experiment = load_module("residual_value_mqar_eval_experiment_test", EXPERIMENT / "experiment.py")
    monkeypatch.setitem(sys.modules, "experiment", experiment)
    evaluate = load_module("residual_value_mqar_evaluate_test", EXPERIMENT / "evaluate.py")
    assert evaluate.BASE.BASE.sources is evaluate.sources
    assert evaluate.BASE.BASE.event_payload is evaluate.event_payload


def test_evaluation_shadow_repairs_legacy_identity(monkeypatch, tmp_path):
    load_common(monkeypatch)
    experiment = load_module("residual_value_mqar_shadow_experiment_test", EXPERIMENT / "experiment.py")
    monkeypatch.setitem(sys.modules, "experiment", experiment)
    evaluate = load_module("residual_value_mqar_shadow_evaluate_test", EXPERIMENT / "evaluate.py")
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    (source / "train_config.json").write_text(
        json.dumps({"resume_identity": {"seed": 123, "residual_value_dim": 32}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate, "run_root", lambda: tmp_path / "run")
    result = {
        "variant": "u32-a1-s1",
        "last_checkpoint": {"path": str(checkpoint)},
    }
    shadow = evaluate.evaluation_checkpoint(result)
    payload = json.loads((shadow.parent / "train_config.json").read_text(encoding="utf-8"))
    assert payload["resume_identity"] == {"seed": "123", "residual_value_dim": "32"}
    assert shadow.stat().st_ino == checkpoint.stat().st_ino


def test_training_patch_uses_saved_upstream_functions(monkeypatch):
    load_common(monkeypatch)
    experiment = load_module("residual_value_mqar_training_patch_test", EXPERIMENT / "experiment.py")

    def fake_training(variant, seed, phase):
        config = experiment.UPSTREAM.build_config(variant, seed, phase)
        row = experiment.UPSTREAM.descriptor(variant, seed)
        assert row["residual_value_dim"] == 64
        assert config.precision == "amp_bfloat16"
        return 17

    monkeypatch.setattr(experiment.UPSTREAM, "run_training", fake_training)
    assert experiment.run_training("u64-a1-s1", 123, "smoke") == 17
