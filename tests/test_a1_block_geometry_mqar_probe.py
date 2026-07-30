from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "zoology/experiments/flash_vqg/scripts/20260730-02-a1-block-geometry-mqar-probe/experiment.py"
os.environ.setdefault("MQAR_A1_GEOMETRY_RUN_TAG", "pytest-a1-block-geometry")


def _load_experiment():
    spec = importlib.util.spec_from_file_location("a1_block_geometry_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPERIMENT = _load_experiment()


def test_scaled_curriculum_preserves_tokens_batches_and_block_counts():
    reference = EXPERIMENT.build_config("a1-reference", "screen")
    for variant in ("a1-block128", "a1-block128-k2r8"):
        candidate = EXPERIMENT.build_config(variant, "screen")
        assert EXPERIMENT.train_tokens(candidate) == EXPERIMENT.train_tokens(reference)
        assert EXPERIMENT.train_batches(candidate) == EXPERIMENT.train_batches(reference)
        assert EXPERIMENT.geometry_rows(candidate) == EXPERIMENT.geometry_rows(reference)


def test_scaled_candidate_changes_only_registered_geometry_and_sparsity():
    block = EXPERIMENT.build_config("a1-block128", "screen")
    sparse = EXPERIMENT.build_config("a1-block128-k2r8", "screen")
    block_audit = EXPERIMENT.BASE.model_audit(block)
    sparse_audit = EXPERIMENT.BASE.model_audit(sparse)
    assert block_audit["state_sha256"] == sparse_audit["state_sha256"]
    assert (block_audit["block_len"], block_audit["write_topk"], block_audit["read_topk"]) == (128, 4, 16)
    assert (sparse_audit["block_len"], sparse_audit["write_topk"], sparse_audit["read_topk"]) == (128, 2, 8)


def test_standard_accuracy_reads_registered_metric(tmp_path, monkeypatch):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "last_checkpoint": {
                    "metrics": {
                        "valid/mqar_case/accuracy-1024x256": 0.75,
                        "valid/mqar_case/accuracy-4096x1024": 0.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(EXPERIMENT, "result_path", lambda variant, phase: result)

    assert EXPERIMENT.standard_accuracy("a1-reference") == 0.75
    assert EXPERIMENT.standard_accuracy("a1-block128") == 0.5
