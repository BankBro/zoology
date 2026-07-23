from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("test_efficiency_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_module()


@pytest.fixture(scope="module")
def configs(bench):
    formal = bench._build_flash_config("formal")
    core = bench._build_flash_config("core")
    gdn = bench._build_gdn_config(formal.data)
    return formal, core, gdn


def test_flash_config_is_current_r16_joint(configs, bench):
    formal, _, _ = configs
    kwargs = bench._find_flash_kwargs(formal.model)
    assert formal.model.d_model == 128
    assert formal.model.n_layers == 2
    assert formal.model.embed_dropout == 0.1
    assert formal.model.resid_dropout == 0.0
    assert formal.model.drop_path == 0.0
    assert formal.data.batch_size == (64, 16)
    assert formal.gradient_accumulation_steps == 4
    assert formal.validations_per_epoch == 4
    assert kwargs["num_codebook_vectors"] == 64
    assert kwargs["fox_remote_read_topk"] == 16
    assert kwargs["fox_gd_residual_rank"] == 16
    assert kwargs["fox_gd_residual_write_topk"] == 4
    assert kwargs["fox_gd_residual_update_norm_softcap"] == 0.5
    assert kwargs["fox_gd_residual_update_norm_softcap_mode"] == "smooth_p4"
    assert kwargs["fox_gd_residual_injection_warmup_start_train_steps"] == 0
    assert kwargs["fox_gd_residual_injection_warmup_end_train_steps"] == 2048


def test_core_only_disables_metrics(configs, bench):
    formal, core, _ = configs
    formal_kwargs = bench._find_flash_kwargs(formal.model)
    core_kwargs = bench._find_flash_kwargs(core.model)
    assert formal_kwargs["enable_layer_metrics"] is True
    assert formal_kwargs["fox_phase2_metrics_mode"] == "lite"
    assert core_kwargs["enable_layer_metrics"] is False
    assert core_kwargs["fox_phase2_metrics_mode"] == "off"
    assert core_kwargs["fox_gd_residual_grouped_chunk_backend"] == "torch"
    assert core_kwargs["fox_gd_residual_selected_read_backend"] == "materialized"

    optimized = bench._build_flash_config(
        "core",
        grouped_chunk_backend="triton",
        selected_read_backend="triton_remat",
    )
    optimized_kwargs = bench._find_flash_kwargs(optimized.model)
    assert optimized_kwargs["fox_gd_residual_grouped_chunk_backend"] == "triton"
    assert optimized_kwargs["fox_gd_residual_selected_read_backend"] == "triton_remat"


def test_same_scale_model_accounting(configs, bench):
    formal, _, gdn = configs
    flash_model = bench.LanguageModel(formal.model)
    gdn_model = bench.LanguageModel(gdn.model)
    flash_params = sum(parameter.numel() for parameter in flash_model.parameters())
    gdn_params = sum(parameter.numel() for parameter in gdn_model.parameters())
    assert flash_params == 1_160_390
    assert gdn_params == 1_335_942
    assert 2 * 64 * 64 * 16 == 2 * 256 * 256 == 131_072
    assert gdn.model.name == "gated_delta_net_expanded_k"
    assert gdn.data.batch_size == formal.data.batch_size


def test_canonical_init_hash(bench):
    payload = bench.torch.load(bench.CANONICAL_INIT, map_location="cpu")
    actual = bench._state_dict_hash(payload["model_state_dict"])
    assert actual == bench.EXPECTED_INIT_HASH
    assert payload["model_state_sha256"] == actual


def test_cache_manifest_and_percentile(configs, bench):
    formal, _, _ = configs
    items = bench._cache_items(formal.data)
    assert len(items) == 13
    assert {item["role"] for item in items} == {"train", "test"}
    assert bench._percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert bench._percentile([1.0, 2.0, 3.0, 4.0], 90) == pytest.approx(3.7)
    assert bench.os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_parser_supports_out_of_band_scalar_metric_capture(bench, tmp_path):
    args = bench._parser().parse_args(
        [
            "run",
            "--model",
            "flash",
            "--phase",
            "eval",
            "--run-kind",
            "timing",
            "--output-dir",
            str(tmp_path),
            "--capture-scalar-metrics",
            "--flash-grouped-chunk-backend",
            "triton",
            "--flash-selected-read-backend",
            "triton_remat",
        ]
    )
    assert args.capture_scalar_metrics is True
    assert args.flash_grouped_chunk_backend == "triton"
    assert args.flash_selected_read_backend == "triton_remat"

    op_profile = bench._parser().parse_args(
        [
            "run",
            "--model",
            "flash",
            "--phase",
            "eval",
            "--run-kind",
            "op-profile",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert op_profile.run_kind == "op-profile"
