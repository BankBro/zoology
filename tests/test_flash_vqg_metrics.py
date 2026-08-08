import torch

from zoology.mixers.flash_vqg import FlashVQGMixer
import pandas as pd

from zoology.analysis.flash_vqg.flash_vqg_analysis_suite import (
    _candidate_metrics_from_config,
    _filter_model_metrics,
    _metric_specs_from_config,
)


def test_flash_vqg_mixer_extracts_scalar_metrics_only():
    mixer = object.__new__(FlashVQGMixer)
    mixer._last_aux = {
        "metrics": {
            "attn/remote_win_rate": torch.tensor(0.25),
            "attn/o_remote_energy_ratio": 0.5,
            "attn/skip_me": torch.tensor([1.0, 2.0]),
            "attn/not_finite": float("inf"),
        }
    }

    assert mixer.get_scalar_metrics() == {
        "attn/remote_win_rate": 0.25,
        "attn/o_remote_energy_ratio": 0.5,
    }


def test_flash_vqg_mixer_batches_scalar_tensor_transfer(monkeypatch):
    mixer = object.__new__(FlashVQGMixer)
    mixer._last_aux = {
        "metrics": {
            "attn/a": torch.tensor(0.25),
            "attn/b": torch.tensor(0.5),
            "attn/c": 0.75,
        }
    }
    original_stack = torch.stack
    calls = []

    def counted_stack(tensors, *args, **kwargs):
        calls.append(len(tensors))
        return original_stack(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", counted_stack)

    assert mixer.get_scalar_metrics() == {
        "attn/a": 0.25,
        "attn/b": 0.5,
        "attn/c": 0.75,
    }
    assert calls == [2]


def test_flash_vqg_mixer_includes_registered_balance_auxiliary_loss():
    mixer = object.__new__(FlashVQGMixer)
    mixer.codebook_beta = 0.25
    mixer.attn = type(
        "AttentionStub",
        (),
        {
            "config": type(
                "ConfigStub", (), {"vq_balance_loss_weight": 0.01}
            )(),
            "res_proj": type(
                "ProjectionStub", (), {"weight": torch.nn.Parameter(torch.ones(()))}
            )(),
        },
    )()
    mixer._last_aux = {
        "l_commit": torch.tensor(2.0),
        "l_balance": torch.tensor(3.0),
    }

    assert torch.equal(mixer.get_auxiliary_loss(), torch.tensor(0.53))


def test_candidate_metrics_include_attn_and_valid_variants():
    config_dict = {
        "model": {
            "n_layers": 2,
        },
    }

    metric_specs = _metric_specs_from_config(config_dict)
    metrics = _candidate_metrics_from_config(config_dict)

    assert "attn/remote_win_rate" in metrics
    assert "valid/attn/remote_win_rate" in metrics
    assert "layer_1/attn/remote_win_rate" in metrics
    assert "valid/layer_1/attn/remote_win_rate" in metrics
    assert "attn/gd_residual_write_q_top1_mean" in metrics
    assert "valid/attn/gd_residual_write_q_entropy_mean" in metrics
    assert "layer_1/attn/gd_residual_write_q_raw_top1_mean" in metrics
    assert "valid/layer_1/attn/gd_residual_write_q_smoothing_active" in metrics
    assert "attn/clr_alpha_norm_mean" in metrics
    assert "valid/attn/clr_h_norm_mean" in metrics
    assert "attn/remote_routing_entropy" in metrics
    assert "valid/attn/remote_top1_top2_margin" in metrics
    assert "layer_1/attn/remote_topk_den_capture_ratio" in metrics
    assert "vq/write_entropy_mean" in metrics
    assert "valid/vq/write_entropy_mean" in metrics
    assert "vq/write_top1_mass_mean" in metrics
    assert "valid/vq/write_top1_mass_mean" in metrics
    assert "__swanlab__.cpu.pct" not in metrics
    assert "valid/vq/c_entropy" in metrics
    assert metric_specs["num_parameters"].chart_type == "bar"
    assert metric_specs["state_size"].chart_type == "bar"
    assert metric_specs["valid/attn/remote_win_rate"].chart_type == "line"


def test_e7_candidate_metrics_keep_default_metric_names():
    config_dict = {
        "model": {
            "n_layers": 1,
        },
        "metrics_white_list": [
            "valid/accuracy",
            "valid/input_seq_len/*",
        ],
    }

    metrics = _candidate_metrics_from_config(config_dict, eval_task="e7")
    metric_specs = _metric_specs_from_config(config_dict, eval_task="e7")

    assert "valid/accuracy" in metrics
    assert "valid/input_seq_len/accuracy-64" in metrics
    assert all(not metric.startswith("e7/") for metric in metrics)
    assert metric_specs["valid/accuracy"].chart_type == "line"


def test_filter_model_metrics_respects_metric_specs():
    history = pd.DataFrame(
        [
            {"metric": "train/loss", "step": 0, "epoch": 1, "timestamp": None, "value": 1.0},
            {"metric": "__swanlab__.cpu.pct", "step": 0, "epoch": 1, "timestamp": None, "value": 50.0},
            {"metric": "valid/accuracy", "step": 1, "epoch": 1, "timestamp": None, "value": 0.5},
            {"metric": "custom/debug", "step": 1, "epoch": 1, "timestamp": None, "value": 0.2},
        ]
    )
    metric_specs = {
        "train/loss": _metric_specs_from_config({}).get("train/loss"),
        "valid/accuracy": _metric_specs_from_config({}).get("valid/accuracy"),
    }

    filtered = _filter_model_metrics(history, metric_specs)

    assert filtered["metric"].tolist() == ["train/loss", "valid/accuracy"]
