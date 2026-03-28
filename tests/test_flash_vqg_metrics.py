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
