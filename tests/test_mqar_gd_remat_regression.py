import importlib.util
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260729-01-mqar-gd-remat-regression"
)
os.environ.setdefault("MQAR_GD_REMAT_RUN_TAG", "pytest-mqar-gd-remat")


def _load(name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load("common", "mqar_gd_remat_common")
_previous_common = sys.modules.get("common")
_previous_experiment = sys.modules.get("experiment")
try:
    sys.modules["common"] = COMMON
    EXPERIMENT = _load("experiment", "mqar_gd_remat_experiment")
    sys.modules["experiment"] = EXPERIMENT
    EVALUATE = _load("evaluate", "mqar_gd_remat_evaluate")
    COLLECTOR = _load("collect_artifacts", "mqar_gd_remat_collector")
    QUEUE = _load("run_queue", "mqar_gd_remat_queue")
finally:
    if _previous_common is None:
        sys.modules.pop("common", None)
    else:
        sys.modules["common"] = _previous_common
    if _previous_experiment is None:
        sys.modules.pop("experiment", None)
    else:
        sys.modules["experiment"] = _previous_experiment


def test_matrix_is_three_seed_pair_and_alternates_order():
    rows = COMMON.training_descriptors()
    assert len(rows) == 6
    assert {(row["variant"], row["seed"]) for row in rows} == {
        (variant, seed) for variant in COMMON.VARIANTS for seed in COMMON.SEEDS
    }
    assert [(row["variant"], row["seed"]) for row in rows] == list(
        COMMON.FORMAL_ORDER
    )


def test_configs_differ_only_in_registered_remat_mode(monkeypatch):
    monkeypatch.setenv("MQAR_GD_REMAT_RUN_TAG", "pytest-config")
    a0 = EXPERIMENT.build_config("a0-off", 124, "formal")
    a1 = EXPERIMENT.build_config("a1-post-phase1", 124, "formal")
    differences = EXPERIMENT.config_differences(a0, a1)
    assert len(differences) == 1
    assert differences[0].endswith(".kwargs.fox_gd_residual_remat_mode")
    a0_kwargs = EXPERIMENT.BASE._find_flash_kwargs(a0.model)
    a1_kwargs = EXPERIMENT.BASE._find_flash_kwargs(a1.model)
    assert a0_kwargs["block_len"] == a1_kwargs["block_len"] == 32
    assert a0_kwargs["fox_gd_residual_remat_mode"] == "off"
    assert a1_kwargs["fox_gd_residual_remat_mode"] == "post_phase1"
    assert a0.precision == a1.precision == "amp_bfloat16"


def test_event_payload_binds_variant_checkpoint_and_fixed_dataset(monkeypatch):
    monkeypatch.setenv("MQAR_GD_REMAT_RUN_TAG", "pytest-event")
    source = {
        "source_id": "3090-a1-post-phase1-s124-bf16-last",
        "machine": "3090",
        "model": "flash",
        "variant": "a1-post-phase1",
        "remat_mode": "post_phase1",
        "seed": 124,
        "train_precision": "bf16",
        "checkpoint_role": "last",
        "checkpoint_path": "/tmp/last.pt",
        "checkpoint_file_sha256": "a" * 64,
        "checkpoint_model_state_sha256": "b" * 64,
    }
    event = EVALUATE.event_payload(source, COMMON.LONGER_CASES[-1], "formal")
    assert event["eval_precision"] == "bf16"
    assert event["eval_batch_size"] == 16
    assert event["num_examples"] == 500
    assert event["expected_dataset_hash"] == COMMON.LONGER_CASES[-1][-1]
    assert event["checkpoint_model_state_sha256"] == "b" * 64
    assert event["variant"] == "a1-post-phase1"


def _training_result(variant: str, seed: int, endpoint_accuracy: float):
    checkpoint = {
        "path": f"/{variant}-{seed}.pt",
        "file_sha256": f"{seed:064x}",
        "epoch": 4,
        "metrics": {
            "valid/loss": 0.1,
            "valid/accuracy": 0.99,
            "valid/mqar_case/accuracy-1024x256": endpoint_accuracy,
        },
    }
    return {
        **COMMON.descriptor(variant, seed),
        "last_checkpoint": checkpoint,
        "best_checkpoint": checkpoint,
        "wall_clock_sec": 10.0,
        "telemetry": {
            "optimizer_step_wall_sec_p50": 0.1,
            "peak_allocated_mib": 100.0,
            "peak_reserved_mib": 120.0,
        },
        "resolved_config_path": "/config.json",
        "resolved_config_sha256": "c" * 64,
    }


def _evaluation(delta: float):
    rows = []
    for seed in COMMON.SEEDS:
        for shape in ("1024x256", *COMMON.EXTRAPOLATION_SHAPES):
            for variant, accuracy in (
                ("a0-off", 0.8),
                ("a1-post-phase1", 0.8 + delta),
            ):
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "shape": shape,
                        "checkpoint_role": "last",
                        "accuracy": accuracy,
                    }
                )
    return rows


def test_quality_gate_uses_registered_noninferiority_margins():
    passing_training = []
    failing_training = []
    for seed in COMMON.SEEDS:
        passing_training.extend(
            [
                _training_result("a0-off", seed, 0.95),
                _training_result("a1-post-phase1", seed, 0.941),
            ]
        )
        failing_training.extend(
            [
                _training_result("a0-off", seed, 0.95),
                _training_result("a1-post-phase1", seed, 0.939),
            ]
        )
    passing, _ = COLLECTOR.quality_summary(passing_training, _evaluation(-0.019))
    failing, _ = COLLECTOR.quality_summary(failing_training, _evaluation(-0.021))
    assert passing["passed"] is True
    assert failing["passed"] is False


def test_queue_stops_after_hard_gate_failure(monkeypatch):
    monkeypatch.setenv("MQAR_GD_REMAT_RUN_TAG", "pytest-queue")
    queue = QUEUE.Queue()
    calls = []
    monkeypatch.setattr(queue, "preflight", lambda: calls.append("preflight"))

    def fail_trajectory():
        calls.append("trajectory")
        raise RuntimeError("failed")

    monkeypatch.setattr(queue, "trajectory", fail_trajectory)
    monkeypatch.setattr(queue, "smoke", lambda: calls.append("smoke"))
    with pytest.raises(RuntimeError, match="failed"):
        queue.run()
    assert calls == ["preflight", "trajectory"]
