from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "zoology/experiments/flash_vqg/scripts/20260730-01-a1-acceleration-mqar-probe"
os.environ.setdefault("MQAR_A1_ACCEL_RUN_TAG", "pytest-a1-accel-mqar")


def _load(name: str, filename: str):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_previous_common = sys.modules.get("common")
EXPERIMENT = _load("a1_accel_mqar_experiment", "experiment.py")
if _previous_common is None:
    sys.modules.pop("common", None)
else:
    sys.modules["common"] = _previous_common
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))


def test_candidate_changes_only_registered_capacity_controls():
    reference = EXPERIMENT.model_audit(EXPERIMENT.build_config("a1-reference", "screen"))
    candidate = EXPERIMENT.model_audit(EXPERIMENT.build_config("a1-block256-k2r8", "screen"))
    assert reference["parameters"] == candidate["parameters"] == 1_160_390
    assert reference["state_sha256"] == candidate["state_sha256"]
    assert (reference["block_len"], reference["write_topk"], reference["read_topk"]) == (32, 4, 16)
    assert (candidate["block_len"], candidate["write_topk"], candidate["read_topk"]) == (256, 2, 8)
    assert reference["remat_mode"] == candidate["remat_mode"] == "post_phase1"
    assert reference["selected_backward"] == candidate["selected_backward"] == "triton_deterministic"


def test_probe_is_single_seed_one_epoch_screen():
    for variant in EXPERIMENT.VARIANTS:
        config = EXPERIMENT.build_config(variant, "screen")
        assert config.seed == 123
        assert config.data.seed == 123
        assert config.max_epochs == 1
        assert config.max_train_steps is None
        assert config.precision == "float32"
        assert tuple(config.data.batch_size) == (64, 16)
        assert config.gradient_accumulation_steps == 4
        assert config.resume_identity["seed"] == "123"
