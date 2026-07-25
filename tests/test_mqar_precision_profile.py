import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260726-01-mqar-precision-profile"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _load(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"precision_profile_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load("common")
EXPERIMENT = _load("experiment")
EVAL_QUEUE = _load("eval_queue")
RUN_QUEUE = _load("run_queue")
COORDINATOR = _load("coordinator")
COLLECTOR = _load("collect_results")


def test_training_matrix_has_exact_machine_and_global_counts(monkeypatch):
    assert len(COMMON.training_descriptors("2080ti")) == 12
    assert len(COMMON.training_descriptors("3090")) == 18
    assert COMMON.expected_training_count() == 30
    first = COMMON.training_descriptors("2080ti")[:4]
    assert [row["descriptor_id"] for row in first] == [
        "2080ti-gdn-s123-fp32",
        "2080ti-flash-s123-fp32",
        "2080ti-gdn-s124-fp32",
        "2080ti-flash-s124-fp32",
    ]


def test_smoke_config_uses_stratified_real_cache_batches(monkeypatch):
    monkeypatch.setenv("MQAR_PRECISION_MACHINE", "2080ti")
    config = EXPERIMENT.build_config("flash", 123, "fp16", "smoke")
    assert config.precision == "amp_float16"
    assert config.max_train_steps == 3
    assert config.gradient_accumulation_steps == 4
    assert config.data.train_batch_segment_order == list(
        EXPERIMENT.SMOKE_TRAIN_SEGMENT_ORDER
    )
    assert config.data.test_batch_segment_order == list(
        EXPERIMENT.SMOKE_VALID_SEGMENT_ORDER
    )
    assert config.resume_stop_after_optimizer_step == 1
    kwargs = EXPERIMENT.BASE._find_flash_kwargs(config.model)
    assert kwargs["fox_gd_residual_triton_input_policy"] == "fp32_boundary"


def test_stress_and_formal_configs_preserve_separate_semantics(monkeypatch):
    monkeypatch.setenv("MQAR_PRECISION_MACHINE", "3090")
    stress = EXPERIMENT.build_config("flash", 125, "bf16", "stress")
    formal = EXPERIMENT.build_config("gdn", 125, "bf16", "formal")
    assert stress.training_runtime_initial_state == {
        EXPERIMENT.FLASH_RUNTIME_MODULE: {
            "fox_gd_residual_train_forward_count": 2048
        }
    }
    assert stress.resume_stop_after_optimizer_step is None
    assert formal.precision == "amp_bfloat16"
    assert formal.max_epochs == 4
    assert formal.max_train_steps is None
    assert formal.data.train_batch_segment_order is None
    assert formal.data.test_batch_segment_order is None


def test_eval_event_identity_binds_precision_shape_data_and_checkpoint(monkeypatch):
    monkeypatch.setenv("MQAR_PRECISION_MACHINE", "2080ti")
    source = {
        "source_id": "2080ti-flash-s123-fp16-last",
        "model": "flash",
        "seed": 123,
        "train_precision": "fp16",
        "checkpoint_role": "last",
        "checkpoint_path": "/tmp/checkpoint.pt",
        "checkpoint_file_sha256": "a" * 64,
        "checkpoint_model_state_sha256": "b" * 64,
    }
    event = EVAL_QUEUE.event_payload(
        source=source,
        eval_precision="fp16",
        shape=(8190, 2047),
        batch_size=2,
        mode="smoke",
        max_batches=3,
        controlled_interrupt=True,
    )
    assert event["num_examples"] == 500
    assert event["expected_dataset_hash"] == COMMON.LONGER_DATASET_HASHES[
        "8190x2047"
    ]
    assert event["controlled_interrupt_after_batches"] == 1
    assert event["max_batches"] == 3
    assert "evalfp16" in event["event_id"]
    assert event["checkpoint_model_state_sha256"] == "b" * 64


def test_batch_invariance_requires_exact_predictions_and_accuracy():
    selected = {
        "prediction_sample_sha256": ["a", "b"],
        "sample_accuracy_values": [1.0, 0.5],
        "sample_loss_values": [0.1, 0.2],
    }
    smaller = {
        "prediction_sample_sha256": ["a", "b"],
        "sample_accuracy_values": [1.0, 0.5],
        "sample_loss_values": [0.1001, 0.1999],
    }
    assert EVAL_QUEUE._compare_invariance(selected, smaller, "fp16")["passed"]
    smaller["prediction_sample_sha256"][1] = "changed"
    assert not EVAL_QUEUE._compare_invariance(selected, smaller, "fp16")["passed"]


def test_queue_file_lock_rejects_duplicate_worker(tmp_path):
    first = RUN_QUEUE.acquire_lock(tmp_path / "queue.lock")
    try:
        with pytest.raises(RuntimeError, match="already held"):
            RUN_QUEUE.acquire_lock(tmp_path / "queue.lock")
    finally:
        first.close()


def test_global_gate_binds_commits_and_cache():
    local_gate = {
        "status": "passed",
        "machine": "2080ti",
        "binding_sha256": "local",
    }
    remote_gate = {
        "status": "passed",
        "machine": "3090",
        "binding_sha256": "remote",
    }
    preflight = {
        "status": "passed",
        "environment": {
            "zoology_commit": "zoology",
            "flash_commit": "flash",
        },
        "cache": {"combined_content_sha256": "cache"},
        "jobs": [
            {
                "model": "gdn",
                "seed": 123,
                "train_precision": "fp32",
                "phase": "formal",
                "normalized_config_sha256": "config",
            }
        ],
    }
    gate = COORDINATOR.build_global_gate(
        local_gate,
        remote_gate,
        preflight,
        preflight,
    )
    assert gate["status"] == "passed"
    assert gate["checks"] == {
        "zoology_commit": True,
        "flash_commit": True,
        "cache_content": True,
        "machines": True,
        "shared_config_hashes": True,
    }


def test_aggregation_keeps_standard_and_longer_protocols_separate():
    rows = []
    for num_examples in (1000, 500):
        for seed, accuracy in zip((123, 124, 125), (0.2, 0.4, 0.6)):
            rows.append(
                {
                    "source_machine": "2080ti",
                    "model": "flash",
                    "checkpoint_role": "last",
                    "train_precision": "fp32",
                    "eval_precision": "fp32",
                    "shape": "1024x256",
                    "num_examples": num_examples,
                    "seed": seed,
                    "accuracy": accuracy,
                }
            )
    summary = COLLECTOR.aggregate(rows)
    assert len(summary) == 2
    assert {row["num_examples"] for row in summary} == {500, 1000}
    assert all(row["n_seeds"] == 3 for row in summary)
    assert all(row["accuracy_mean"] == pytest.approx(0.4) for row in summary)
