from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260725-01-current-baselines-longer-mqar"
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_training_matrix_and_resolved_configs():
    experiment = _load("test_current_longer_experiment", "experiment.py")
    assert experiment.JOB_ORDER == (
        ("flash", 123),
        ("gdn", 123),
        ("flash", 124),
        ("gdn", 124),
        ("flash", 125),
        ("gdn", 125),
    )
    for model, seed in experiment.JOB_ORDER:
        smoke = experiment.build_config(model, seed, "smoke")
        formal = experiment.build_config(model, seed, "formal")
        assert smoke.seed == formal.seed == seed
        assert smoke.data.seed == formal.data.seed == 123
        assert smoke.max_epochs == 1
        assert smoke.max_train_steps == 4
        assert smoke.max_validation_batches == 2
        assert formal.max_epochs == 4
        assert formal.max_train_steps is None
        assert formal.max_validation_batches is None
        assert formal.data.batch_size == (64, 16)
        assert formal.gradient_accumulation_steps == 4
        assert formal.validations_per_epoch == 4
        assert formal.early_stopping_metric is None
        assert formal.checkpoint.best_metric == "valid/accuracy"
        assert formal.checkpoint.save_last is True
        assert formal.checkpoint.save_best is True
        assert Path(formal.init_checkpoint_path).resolve() == experiment.init_path(model)
        for run_type, config in (("smoke", smoke), ("formal", formal)):
            expected = experiment.EXPECTED_NORMALIZED_CONFIG_SHA256[(model, seed, run_type)]
            assert experiment.normalized_config_sha256(config) == expected


def test_machine_isolation_preserves_normalized_configs(monkeypatch):
    monkeypatch.setenv("LONGER_MQAR_MACHINE", "2080ti")
    old = _load("test_current_longer_machine_2080ti", "experiment.py")
    monkeypatch.setenv("LONGER_MQAR_MACHINE", "3090")
    new = _load("test_current_longer_machine_3090", "experiment.py")

    assert old.OUTPUT_ROOT == SCRIPT_DIR / "outputs"
    assert new.OUTPUT_ROOT == SCRIPT_DIR / "outputs/machines/3090"
    assert old.MACHINE_SPEC["cuda_visible_device"] == "1"
    assert new.MACHINE_SPEC["cuda_visible_device"] == "0"
    assert old.launch_id("formal").endswith("-2080ti-formal")
    assert new.launch_id("formal").endswith("-3090-formal")

    for model, seed in old.JOB_ORDER:
        for run_type in ("smoke", "formal"):
            old_config = old.build_config(model, seed, run_type)
            new_config = new.build_config(model, seed, run_type)
            assert old.normalized_config_payload(old_config) == new.normalized_config_payload(new_config)
            assert old.normalized_config_sha256(old_config) == new.normalized_config_sha256(new_config)
            assert old_config.checkpoint.root_dir != new_config.checkpoint.root_dir
            if run_type == "formal":
                assert "-3090-formal" in new_config.run_id


def test_model_specific_invariants():
    experiment = _load("test_current_longer_invariants", "experiment.py")
    flash = experiment.build_config("flash", 123, "formal")
    flash_kwargs = experiment.BASE._find_flash_kwargs(flash.model)
    assert flash_kwargs["num_codebook_vectors"] == 64
    assert flash_kwargs["fox_gd_residual_rank"] == 16
    assert flash_kwargs["fox_remote_read_topk"] == 16
    assert flash_kwargs["fox_gd_residual_write_topk"] == 4
    assert flash_kwargs["fox_gd_residual_update_norm_softcap"] == 0.5
    assert flash_kwargs["fox_gd_residual_update_norm_softcap_mode"] == "smooth_p4"
    assert flash_kwargs["fox_gd_residual_injection_warmup_end_train_steps"] == 2048
    assert flash_kwargs["fox_gd_residual_grouped_chunk_backend"] == "triton"
    assert flash_kwargs["fox_gd_residual_selected_read_backend"] == "triton_remat"

    gdn = experiment.build_config("gdn", 123, "formal")
    kwargs = experiment._find_nested_kwargs(gdn.model.model_dump(mode="json"), lambda item: item.get("expand_k") is not None)
    assert gdn.model.name == "gated_delta_net_expanded_k"
    assert kwargs["num_heads"] == 2
    assert kwargs["expand_k"] == 4
    assert kwargs["expand_v"] == 4
    assert kwargs["use_gate"] is False
    assert 2 * 64 * 64 * 16 == 2 * 256 * 256 == 131_072


def test_manifest_deduplicates_last_best_by_model_state(tmp_path):
    runner = _load("test_current_longer_runner", "longer_mqar_runner.py")
    rows = []
    for model in ("flash", "gdn"):
        for seed in (123, 124, 125):
            checkpoint = tmp_path / f"{model}-{seed}.pt"
            checkpoint.write_bytes(f"{model}-{seed}".encode())
            file_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state_hash = hashlib.sha256(f"state-{model}-{seed}".encode()).hexdigest()
            for role in ("last", "best"):
                rows.append(
                    {
                        "source_id": f"{model}-s{seed}-{role}",
                        "model": model,
                        "seed": seed,
                        "checkpoint_role": role,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_file_sha256": file_hash,
                        "checkpoint_model_state_sha256": state_hash,
                    }
                )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    logical, unique = runner.load_manifest(manifest)
    assert len(logical) == 12
    assert len(unique) == 6
    assert all(set(source["checkpoint_roles"]) == {"last", "best"} for source in unique)
    assert runner.BATCH_CANDIDATES == (32, 16, 8, 4, 2, 1)
    assert tuple(runner.EXPECTED_DATASET_HASHES) == ("1024x256", "2048x512", "4096x1024", "8190x512", "8190x2047")


def test_collector_summary_and_paired_classification():
    collector = _load("test_current_longer_collector", "collect_artifacts.py")
    rows = []
    for role in ("last", "best"):
        for model in ("flash", "gdn"):
            for seed in (123, 124, 125):
                for index, slc in enumerate(collector.SLICES):
                    base = 0.95 if model == "flash" else 0.90
                    rows.append(
                        {
                            "checkpoint_role": role,
                            "model": model,
                            "seed": seed,
                            "slice": slc,
                            "accuracy": base - 0.1 * index + 0.001 * (seed - 123),
                        }
                    )
    summary = collector.summarize(rows)
    deltas = collector.paired_deltas(rows)
    role_comparison = collector.checkpoint_role_comparison(rows)
    assert len(summary) == 20
    assert len(deltas) == 10
    assert len(role_comparison) == 30
    assert all(row["classification"] == "稳健领先" for row in deltas)
    assert all(row["positive_seed_count"] == 3 for row in deltas)


def test_cross_machine_delta_matrix():
    collector = _load("test_current_longer_cross_collector", "collect_cross_machine_artifacts.py")
    rows = []
    for machine, machine_delta in (("2080ti", 0.0), ("3090", 0.01)):
        for role in collector.ROLES:
            for model in collector.MODELS:
                for seed in collector.SEEDS:
                    for index, slc in enumerate(collector.SLICES):
                        rows.append({
                            "machine": machine,
                            "model": model,
                            "seed": str(seed),
                            "checkpoint_role": role,
                            "slice": slc,
                            "accuracy": str(0.9 - index * 0.1 + machine_delta),
                            "dataset_hash": f"hash-{slc}",
                        })
    deltas = collector.cross_machine_deltas(rows)
    assert len(deltas) == 60
    assert all(abs(row["accuracy_delta_3090_minus_2080ti"] - 0.01) < 1e-12 for row in deltas)
    assert all(row["dataset_hash_match"] for row in deltas)


def test_plan_and_queue_have_required_gates():
    queue_text = (SCRIPT_DIR / "run_queue.py").read_text(encoding="utf-8")
    plan_text = (ROOT / "docs/plans/20260725-01-current-baselines-longer-mqar-plan.md").read_text(encoding="utf-8")
    assert "TRAINING_SMOKE_PASSED.json" in (SCRIPT_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "SMOKE_DONE.json" in queue_text
    assert "EVAL_SMOKE_PASSED.json" in (SCRIPT_DIR / "longer_mqar_runner.py").read_text(encoding="utf-8")
    assert "FAILED.json" in queue_text
    assert "DONE.json" in queue_text
    assert "LONGER_MQAR_MACHINE" in queue_text
    assert "outputs/machines" in queue_text
    assert "Plan -> 实验 -> Report" in plan_text
