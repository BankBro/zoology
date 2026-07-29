from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260729-03-mqar-seed124-remat-causal-diagnosis"
)


def load_module(name: str, filename: str):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_configs_change_only_remat_mode(monkeypatch, tmp_path):
    common = load_module("seed124_diag_common", "common.py")
    monkeypatch.setenv("MQAR_SEED124_DIAG_RUN_TAG", "unit")
    monkeypatch.setattr(common, "run_root", lambda: tmp_path)
    a0 = common.build_config("a0-fixed-off", label="initial", max_train_steps=16)
    a1 = common.build_config(
        "a1-fixed-post-phase1",
        label="initial",
        max_train_steps=16,
    )
    assert a0.seed == a1.seed == 124
    assert a0.gradient_accumulation_steps == a1.gradient_accumulation_steps == 4
    assert a0.validations_per_epoch == a1.validations_per_epoch == 4
    assert a0.precision == a1.precision == "amp_bfloat16"
    assert common.config_differences(a0, a1) == [
        "model.sequence_mixer.kwargs.configs[1].kwargs.fox_gd_residual_remat_mode"
    ]


def test_compare_traces_finds_first_forward_difference(tmp_path):
    compare = load_module("seed124_diag_compare", "compare_traces.py")
    common_rows = [
        {
            "event": "forward",
            "window": 9,
            "optimizer_step": 8,
            "micro_step": 3,
            "input": {"sha256": "same"},
            "target": {"sha256": "same"},
            "loss": {"sha256": "loss-9"},
            "rng_before_sha256": "rng",
            "rng_after_sha256": "rng2",
        },
        {
            "event": "forward",
            "window": 10,
            "optimizer_step": 9,
            "micro_step": 2,
            "input": {"sha256": "same-input"},
            "target": {"sha256": "same-target"},
            "loss": {"sha256": "left-loss"},
            "rng_before_sha256": "rng3",
            "rng_after_sha256": "rng4",
        },
    ]
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left.write_text("\n".join(json.dumps(row) for row in common_rows) + "\n")
    right_rows = [dict(row) for row in common_rows]
    right_rows[1] = json.loads(json.dumps(right_rows[1]))
    right_rows[1]["loss"]["sha256"] = "right-loss"
    right.write_text("\n".join(json.dumps(row) for row in right_rows) + "\n")
    result = compare.compare(left, right)
    assert result["first_mismatch_window"] == 10
    assert result["first_mismatch"]["micro_step"] == 2
    assert result["first_mismatch"]["classification"] == "forward_loss"


def test_tensor_hash_supports_scalar_tensor():
    probe = load_module("seed124_diag_probe", "probe.py")
    scalar = torch.tensor(1.25)
    assert probe.tensor_hash(scalar) == probe.tensor_hash(scalar.clone())


def test_gradient_analysis_groups_runs_and_names_tensors(tmp_path):
    analyze = load_module("seed124_diag_analyze", "analyze_gradients.py")
    paths = []
    for index, grad_hash in enumerate(("group-a", "group-a", "group-b")):
        path = tmp_path / f"trace-{index}.jsonl"
        row = {
            "event": "after_backward",
            "window": 1,
            "micro_step": 0,
            "grad_sha256": grad_hash,
            "grad_tensors": {
                "same": {"sha256": "same"},
                "changed": {"sha256": grad_hash},
            },
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths.append(path)
    result = analyze.analyze(paths, window=1, micro_step=0)
    assert result["gradient_group_count"] == 2
    comparison = result["comparisons"][0]
    assert comparison["different_tensor_count"] == 1
    assert comparison["different_tensors"][0]["name"] == "changed"
