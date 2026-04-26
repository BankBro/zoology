from __future__ import annotations

from pathlib import Path

import torch

from zoology.analysis.flash_vqg.flash_vqg_analysis_suite import _metric_specs_from_config
from zoology.config import DataConfig, DataSegmentConfig, ModelConfig, TrainConfig
from zoology.experiments.flash_vqg.metrics_white_list import derive_flash_metric_controls
from zoology.logger import _model_summary_metrics


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))

    def state_size(self, sequence_length: int = 2048):
        return sequence_length + 1


def test_derive_flash_metric_controls_respects_white_list_scope():
    no_model_metrics = derive_flash_metric_controls(["valid/accuracy", "valid/mqar_case/*"])
    lite_remote_metrics = derive_flash_metric_controls(["attn/o_remote_energy_ratio", "valid/attn/clr_alpha_norm_mean"])
    full_remote_metrics = derive_flash_metric_controls(["valid/attn/remote_topk_den_capture_ratio"])
    vq_metrics_only = derive_flash_metric_controls(["vq/c_entropy", "valid/vq/relative_err_mean"])
    gd_residual_metrics = derive_flash_metric_controls(
        ["attn/gd_residual_lambda_mean", "valid/attn/gd_residual_mu_valid_ratio"]
    )
    gd_residual_debug_metrics = derive_flash_metric_controls(
        [
            "attn/gd_residual_debug_event_count",
            "valid/attn/gd_residual_debug_l_state_max",
            "valid/attn/gd_residual_debug_l_state_p95",
            "layer_*/attn/gd_residual_debug_avg_events_per_group",
        ]
    )

    assert no_model_metrics == {
        "enable_layer_metrics": False,
        "fox_phase2_metrics_mode": "off",
    }
    assert lite_remote_metrics == {
        "enable_layer_metrics": True,
        "fox_phase2_metrics_mode": "lite",
    }
    assert full_remote_metrics == {
        "enable_layer_metrics": True,
        "fox_phase2_metrics_mode": "full",
    }
    assert vq_metrics_only == {
        "enable_layer_metrics": True,
        "fox_phase2_metrics_mode": "off",
    }
    assert gd_residual_metrics == {
        "enable_layer_metrics": True,
        "fox_phase2_metrics_mode": "lite",
    }
    assert gd_residual_debug_metrics == {
        "enable_layer_metrics": True,
        "fox_phase2_metrics_mode": "lite",
    }


def test_metric_specs_from_config_respects_metrics_white_list():
    config_dict = {
        "metrics_white_list": [
            "valid/accuracy",
            "valid/mqar_case/*",
            "attn/clr_alpha_norm_mean",
        ],
        "data": {
            "train_configs": [{"input_seq_len": 64, "num_kv_pairs": 4}],
            "test_configs": [{"input_seq_len": 64, "num_kv_pairs": 4}],
        },
        "model": {
            "n_layers": 2,
        },
    }

    metric_specs = _metric_specs_from_config(config_dict)

    assert sorted(metric_specs) == [
        "attn/clr_alpha_norm_mean",
        "valid/accuracy",
        "valid/mqar_case/accuracy-64x4",
    ]


def test_model_summary_metrics_respects_metrics_white_list():
    base_config = TrainConfig(
        model=ModelConfig(name="toy"),
        data=DataConfig(
            train_configs=[DataSegmentConfig(input_seq_len=16, num_examples=4)],
            test_configs=[DataSegmentConfig(input_seq_len=16, num_examples=4)],
        ),
        metrics_white_list=["state_size"],
    )

    metrics = _model_summary_metrics(_TinyModel(), base_config)

    assert metrics == {"state_size": 17}


def test_doc_aligned_metrics_white_list_templates_are_trimmed():
    template_dir = (
        Path(__file__).resolve().parents[1]
        / "zoology"
        / "experiments"
        / "flash_vqg"
        / "metrics_white_lists"
    )

    def _read_template(name: str) -> list[str]:
        return [
            line.strip()[2:]
            for line in (template_dir / name).read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("- ")
        ]

    e0 = _read_template("e0.yaml")
    e1 = _read_template("e1.yaml")
    e3 = _read_template("e3.yaml")
    e4 = _read_template("e4.yaml")
    e5 = _read_template("e5.yaml")
    e6 = _read_template("e6.yaml")
    e8 = _read_template("e8.yaml")

    assert "train/loss" in e0
    assert "train/loss" in e1
    assert "train/loss" in e5
    assert "train/loss" in e6
    assert "train/loss" in e8
    assert "valid/loss" in e8
    assert "valid/mqar_case/*" not in e8
    assert "valid/input_seq_len/*" in e8
    assert "valid/num_kv_pairs/*" in e8
    assert "attn/o_remote_local_cos" not in e1
    assert "attn/remote_dominance_rate" not in e1
    assert "attn/clr_alpha_norm_mean" in e1
    assert "attn/clr_h_norm_mean" in e1
    assert "attn/remote_routing_entropy" not in e1
    assert "attn/remote_win_rate" not in e3
    assert "attn/remote_win_rate" not in e4
    assert "attn/o_remote_energy_ratio" in e3
    assert "attn/clr_den_neg_ratio" in e3
    assert "attn/o_remote_local_cos" not in e6
    assert "attn/remote_dominance_rate" not in e6


def test_derive_flash_metric_controls_treats_new_routing_metrics_as_full_only():
    controls = derive_flash_metric_controls(
        [
            "attn/remote_routing_entropy",
            "valid/attn/remote_top1_top2_margin",
            "layer_0/attn/remote_topk_den_capture_ratio",
        ]
    )

    assert controls == {
        "enable_layer_metrics": True,
        "fox_phase2_metrics_mode": "full",
    }
