#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
os.environ.setdefault("GDN_KERNEL_DTYPE", "float32")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TORCH_DETERMINISTIC", "0")

import torch


EXPERIMENT_ID = "20260725-01-current-baselines-longer-mqar"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
FLASH_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
GENERATED_ROOT = REPO_ROOT / "zoology/experiments/flash_vqg/generated"
FORMAL_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / f"fvqg-{EXPERIMENT_ID}-2080ti"
SMOKE_CHECKPOINT_ROOT = OUTPUT_ROOT / "smoke-checkpoints"
GATE_DIR = OUTPUT_ROOT / "gates"
EXPECTED_PYTHON = Path("/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python")
BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py"
)
SEEDS = (123, 124, 125)
MODELS = ("flash", "gdn")
JOB_ORDER = tuple((model, seed) for seed in SEEDS for model in MODELS)
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_BATCH_ORDER_HASHES = {
    0: "fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320",
    1: "b9d52c40883bf347d481b8d0b79141885643f4f554bbe7016acd1b1e3d69b7c4",
    2: "5d31531aafcb4a4383a2ac711fbc9c0b2727e95c48b12d1902f0bb22cc3b6f20",
    3: "6ae4c4584b2b365741cb9973e714825e75c138c4c8af40406333f7e612f42839",
}
EXPECTED_INIT = {
    "flash": {
        "file_sha256": "26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878",
        "state_sha256": "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0",
        "params": 1_160_390,
    },
    "gdn": {
        "file_sha256": "a4e76e7776bdc83a582c2613cd7d9782100a9148aa119763ecaaeeb8273f7b71",
        "state_sha256": "bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6",
        "params": 1_335_942,
    },
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module(BASE_SCRIPT, "current_baseline_longer_mqar_base")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    tmp.replace(path)


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    return BASE._state_dict_hash(state_dict)


def init_path(model: str) -> Path:
    return Path(BASE.CANONICAL_INIT if model == "flash" else BASE.GDN_CANONICAL_INIT).resolve()


def run_id(model: str, seed: int, run_type: str) -> str:
    return f"{model}-s{seed}-fixedinit-s124-d123-b64ga4-{run_type}"


def launch_id(run_type: str) -> str:
    return f"{EXPERIMENT_ID}-2080ti-{run_type}"


def result_path(model: str, seed: int, run_type: str) -> Path:
    return OUTPUT_ROOT / run_type / run_id(model, seed, run_type) / "result.json"


def checkpoint_root(run_type: str) -> Path:
    return FORMAL_CHECKPOINT_ROOT if run_type == "formal" else SMOKE_CHECKPOINT_ROOT


def checkpoint_run_dir(config: Any) -> Path:
    return Path(config.checkpoint.root_dir) / str(config.launch_id) / str(config.run_id)


def _find_nested_kwargs(payload: Any, predicate) -> dict[str, Any]:
    if isinstance(payload, dict):
        if predicate(payload):
            return payload
        for value in payload.values():
            found = _find_nested_kwargs(value, predicate)
            if found:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _find_nested_kwargs(value, predicate)
            if found:
                return found
    return {}


def build_config(model: str, seed: int, run_type: str):
    if model not in MODELS:
        raise ValueError(f"未知模型: {model}")
    if seed not in SEEDS:
        raise ValueError(f"未知seed: {seed}")
    if run_type not in {"smoke", "formal"}:
        raise ValueError(f"未知run_type: {run_type}")

    flash = BASE._build_flash_config("core", "triton", "triton_remat")
    gdn = BASE._build_gdn_config(flash.data)
    config = flash if model == "flash" else gdn
    config.seed = int(seed)
    config.data.seed = 123
    config.max_epochs = 4 if run_type == "formal" else 1
    config.max_train_steps = None if run_type == "formal" else 4
    config.max_validation_batches = None if run_type == "formal" else 2
    config.validations_per_epoch = 4
    config.early_stopping_metric = None
    config.early_stopping_threshold = None
    config.metrics_white_list = []
    config.logger.backend = "none"
    config.checkpoint.enabled = True
    config.checkpoint.save_best = True
    config.checkpoint.save_last = True
    config.checkpoint.best_metric = "valid/accuracy"
    config.checkpoint.best_mode = "max"
    config.checkpoint.root_dir = str(checkpoint_root(run_type))
    config.init_checkpoint_path = str(init_path(model))
    config.init_checkpoint_strict = True
    config.launch_id = launch_id(run_type)
    config.sweep_id = EXPERIMENT_ID
    config.run_id = run_id(model, seed, run_type)
    return config


def configure_numerics() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise RuntimeError("TRITON_F32_DEFAULT必须为ieee.")
    if os.environ.get("GDN_KERNEL_DTYPE") != "float32":
        raise RuntimeError("GDN_KERNEL_DTYPE必须为float32.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE必须为0.")


def environment_metadata() -> dict[str, Any]:
    import fla
    import triton

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    nvidia_smi = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True)
    return {
        "sys_executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "fla": importlib.metadata.version("flash-linear-attention"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": visible,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "nvidia_smi_ok": nvidia_smi.returncode == 0,
        "nvidia_smi": nvidia_smi.stdout.strip(),
        "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT"),
        "gdn_kernel_dtype": os.environ.get("GDN_KERNEL_DTYPE"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "zoology_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "zoology_status": git_value(REPO_ROOT, "status", "--short"),
        "flash_branch": git_value(FLASH_ROOT, "branch", "--show-current"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "flash_status": git_value(FLASH_ROOT, "status", "--short"),
        "recorded_at_utc": utc_now(),
    }


def batch_order_hashes(config: Any) -> dict[int, dict[str, Any]]:
    train_loader, _ = BASE.prepare_data(config.data)
    sampler = train_loader.sampler
    rows: dict[int, dict[str, Any]] = {}
    for epoch in range(4):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        order = list(iter(sampler))
        digest = hashlib.sha256()
        for value in order:
            digest.update(int(value).to_bytes(8, "little", signed=True))
        actual = digest.hexdigest()
        rows[epoch] = {
            "num_batches": len(order),
            "sha256": actual,
            "expected": EXPECTED_BATCH_ORDER_HASHES[epoch],
            "match": actual == EXPECTED_BATCH_ORDER_HASHES[epoch],
            "first_16": order[:16],
        }
    return rows


def validate_config(config: Any, model_name: str, seed: int, run_type: str) -> dict[str, Any]:
    model = BASE.LanguageModel(config.model)
    payload = torch.load(init_path(model_name), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    state_hash = state_dict_hash(model.state_dict())
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_payload = config.model.model_dump(mode="json")
    if model_name == "flash":
        kwargs = BASE._find_flash_kwargs(config.model)
        invariants = {
            "model_name": config.model.name == "flash_vqg",
            "num_codebook_vectors": kwargs.get("num_codebook_vectors") == 64,
            "rank": kwargs.get("fox_gd_residual_rank") == 16,
            "read_topk": kwargs.get("fox_remote_read_topk") == 16,
            "write_topk": kwargs.get("fox_gd_residual_write_topk") == 4,
            "softcap": kwargs.get("fox_gd_residual_update_norm_softcap") == 0.5,
            "softcap_mode": kwargs.get("fox_gd_residual_update_norm_softcap_mode") == "smooth_p4",
            "warmup_start": kwargs.get("fox_gd_residual_injection_warmup_start_train_steps") == 0,
            "warmup_end": kwargs.get("fox_gd_residual_injection_warmup_end_train_steps") == 2048,
            "grouped_backend": kwargs.get("fox_gd_residual_grouped_chunk_backend") == "triton",
            "read_backend": kwargs.get("fox_gd_residual_selected_read_backend") == "triton_remat",
        }
    else:
        kwargs = _find_nested_kwargs(model_payload, lambda item: item.get("expand_k") is not None)
        invariants = {
            "model_name": config.model.name == "gated_delta_net_expanded_k",
            "num_heads": kwargs.get("num_heads") == 2,
            "expand_k": kwargs.get("expand_k") == 4,
            "expand_v": kwargs.get("expand_v") == 4,
            "use_gate": kwargs.get("use_gate") is False,
            "use_short_conv": kwargs.get("use_short_conv") is True,
            "conv_size": kwargs.get("conv_size") == 4,
        }
    common = {
        "seed": config.seed == seed,
        "data_seed": config.data.seed == 123,
        "batch_size": tuple(config.data.batch_size) == (64, 16),
        "gradient_accumulation_steps": config.gradient_accumulation_steps == 4,
        "max_epochs": config.max_epochs == (4 if run_type == "formal" else 1),
        "validations_per_epoch": config.validations_per_epoch == 4,
        "early_stopping_disabled": config.early_stopping_metric is None and config.early_stopping_threshold is None,
        "init_path": Path(config.init_checkpoint_path).resolve() == init_path(model_name),
        "init_file_hash": sha256_file(init_path(model_name)) == EXPECTED_INIT[model_name]["file_sha256"],
        "init_state_hash": state_hash == EXPECTED_INIT[model_name]["state_sha256"],
        "trainable_params": params == EXPECTED_INIT[model_name]["params"],
        "capacity": 131_072 == (2 * 64 * 64 * 16 if model_name == "flash" else 2 * 256 * 256),
        "strict_load": not incompatible.missing_keys and not incompatible.unexpected_keys,
    }
    checks = {**common, **invariants}
    return {
        "model": model_name,
        "seed": seed,
        "run_type": run_type,
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "checkpoint_run_dir": str(checkpoint_run_dir(config)),
        "init_path": str(init_path(model_name)),
        "init_file_sha256": sha256_file(init_path(model_name)),
        "init_model_state_sha256": state_hash,
        "trainable_params": params,
        "active_state_capacity": 131_072,
        "checks": checks,
        "passed": all(checks.values()),
    }


def generated_config_path(config: Any) -> Path:
    return GENERATED_ROOT / str(config.launch_id) / f"{config.run_id}.json"


def write_resolved_config(config: Any) -> Path:
    from zoology.checkpoints import serialize_train_config

    path = generated_config_path(config)
    write_json(path, serialize_train_config(config))
    return path


def preflight(output: Path) -> dict[str, Any]:
    configure_numerics()
    env = environment_metadata()
    env_checks = {
        "python_path": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "python_version": env["python"] == "3.12.11",
        "torch": env["torch"] == "2.6.0+cu118",
        "cuda": env["torch_cuda"] == "11.8",
        "triton": env["triton"] == "3.2.0",
        "fla": env["fla"] == "0.4.2",
        "cuda_available": bool(env["cuda_available"]),
        "gpu": env["gpu_name"] == "NVIDIA GeForce RTX 2080 Ti",
        "visible_gpu": env["cuda_visible_devices"] == "1",
        "nvidia_smi": bool(env["nvidia_smi_ok"]),
        "tf32_disabled": not env["matmul_tf32"] and not env["cudnn_tf32"],
        "zoology_branch": env["zoology_branch"] == EXPERIMENT_ID,
        "zoology_clean": env["zoology_status"] == "",
        "flash_commit": env["flash_commit"] == "ec770f33676036432c6514acd1ac05bd2d01f3e8",
        "flash_clean": env["flash_status"] == "",
    }
    base_config = build_config("flash", 123, "formal")
    cache = BASE._cache_content_hash(base_config.data)
    orders = batch_order_hashes(base_config)
    jobs: list[dict[str, Any]] = []
    for model, seed in JOB_ORDER:
        for run_type in ("smoke", "formal"):
            config = build_config(model, seed, run_type)
            resolved = write_resolved_config(config)
            row = validate_config(config, model, seed, run_type)
            row["resolved_config_path"] = str(resolved)
            row["resolved_config_sha256"] = sha256_file(resolved)
            jobs.append(row)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "passed" if all(env_checks.values()) and cache.get("match") and all(row["match"] for row in orders.values()) and all(job["passed"] for job in jobs) else "failed",
        "recorded_at_utc": utc_now(),
        "environment": env,
        "environment_checks": env_checks,
        "cache": cache,
        "batch_orders": orders,
        "jobs": jobs,
    }
    write_json(output, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"硬预检失败, 见 {output}")
    return payload


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    from zoology.checkpoints import load_checkpoint

    payload = torch.load(path, map_location="cpu", weights_only=False)
    bundle = load_checkpoint(path, device="cpu", strict=True)
    metrics = payload.get("metrics") or {}
    finite_metrics = all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (float, int)))
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "model_state_sha256": state_dict_hash(payload["model_state_dict"]),
        "epoch_index_zero_based": int(payload.get("epoch", -1)),
        "epoch": int(payload.get("epoch", -1)) + 1,
        "metrics": metrics,
        "finite_metrics": finite_metrics,
        "strict_load": not bundle["missing_keys"] and not bundle["unexpected_keys"],
    }


def train_one(model: str, seed: int, run_type: str, resume: bool) -> dict[str, Any]:
    configure_numerics()
    preflight_path = OUTPUT_ROOT / "preflight.json"
    if not preflight_path.exists() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "passed":
        raise RuntimeError("缺少通过的preflight.json.")
    if run_type == "formal":
        for gate in ("TRAINING_SMOKE_PASSED.json", "SMOKE_DONE.json"):
            gate_path = GATE_DIR / gate
            if not gate_path.exists() or json.loads(gate_path.read_text(encoding="utf-8")).get("status") != "passed":
                raise RuntimeError(f"正式训练缺少门槛: {gate_path}")

    config = build_config(model, seed, run_type)
    resolved_path = write_resolved_config(config)
    resolved_sha256 = sha256_file(resolved_path)
    out = result_path(model, seed, run_type)
    if resume and out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        expected_epoch = 4 if run_type == "formal" else 1
        identity_matches = all((
            previous.get("status") == "completed",
            previous.get("model") == model,
            previous.get("seed") == seed,
            previous.get("data_seed") == 123,
            previous.get("run_type") == run_type,
            previous.get("run_id") == config.run_id,
            previous.get("launch_id") == config.launch_id,
            previous.get("configured_max_epochs") == config.max_epochs,
            previous.get("resolved_config_sha256") == resolved_sha256,
        ))
        if identity_matches:
            reusable = True
            for role in ("last", "best"):
                meta = previous.get(f"{role}_checkpoint") or {}
                path = Path(meta.get("path") or "")
                if not path.exists() or sha256_file(path) != meta.get("file_sha256"):
                    reusable = False
                    break
                current = _checkpoint_metadata(path)
                if (
                    current["model_state_sha256"] != meta.get("model_state_sha256")
                    or not current["strict_load"]
                    or not current["finite_metrics"]
                    or current["epoch"] > expected_epoch
                    or (role == "last" and current["epoch"] != expected_epoch)
                ):
                    reusable = False
                    break
            if reusable:
                print(json.dumps({"status": "skipped_completed", "result": str(out)}, ensure_ascii=False))
                return previous

    run_dir = checkpoint_run_dir(config)
    started_at = utc_now()
    started = time.perf_counter()
    status = "completed"
    error = None
    tb = None
    try:
        from zoology.train import train

        train(config)
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
    ended_at = utc_now()
    wall = time.perf_counter() - started
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "model": model,
        "seed": seed,
        "data_seed": 123,
        "run_type": run_type,
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "status": status,
        "error": error,
        "traceback": tb,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_clock_sec": wall,
        "configured_max_epochs": config.max_epochs,
        "expected_final_epoch": 4 if run_type == "formal" else 1,
        "resolved_config_path": str(resolved_path),
        "resolved_config_sha256": resolved_sha256,
        "checkpoint_run_dir": str(run_dir),
        "environment": environment_metadata(),
    }
    if status == "completed":
        try:
            result["last_checkpoint"] = _checkpoint_metadata(last_path)
            result["best_checkpoint"] = _checkpoint_metadata(best_path)
            expected_epoch = 4 if run_type == "formal" else 1
            if result["last_checkpoint"]["epoch"] != expected_epoch:
                raise RuntimeError(f"last checkpoint epoch应为{expected_epoch}, 实际{result['last_checkpoint']['epoch']}")
            if not result["last_checkpoint"]["finite_metrics"] or not result["best_checkpoint"]["finite_metrics"]:
                raise RuntimeError("checkpoint metrics包含非有限值.")
        except BaseException as exc:
            result["status"] = "failed"
            result["error"] = f"checkpoint_audit:{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
    write_json(out, result)
    print(json.dumps({"status": result["status"], "result": str(out)}, ensure_ascii=False))
    if result["status"] != "completed":
        raise RuntimeError(result.get("error") or "训练失败")
    return result


def _first_batches_by_length(loader: Any, lengths: set[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    found: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for inputs, targets, _slices in loader:
        seq_len = int(inputs.shape[1])
        if seq_len in lengths and seq_len not in found:
            found[seq_len] = (inputs.contiguous(), targets.contiguous())
        if set(found) == lengths:
            break
    if set(found) != lengths:
        raise RuntimeError(f"缺少shape batch: expected={sorted(lengths)}, actual={sorted(found)}")
    return found


def shape_smoke(model_name: str, output: Path) -> dict[str, Any]:
    configure_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用.")
    config = build_config(model_name, 123, "smoke")
    train_loader, test_loader = BASE.prepare_data(config.data)
    train_batches = _first_batches_by_length(train_loader, {64, 128, 256})
    eval_batches = _first_batches_by_length(test_loader, {64, 128, 256, 512, 1024})
    model = BASE.LanguageModel(config.model)
    init_payload = torch.load(init_path(model_name), map_location="cpu", weights_only=False)
    model.load_state_dict(init_payload["model_state_dict"], strict=True)
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rows: list[dict[str, Any]] = []
    started_at = utc_now()
    torch.cuda.reset_peak_memory_stats()
    for seq_len in (64, 128, 256):
        inputs, targets = train_batches[seq_len]
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        BASE._set_dense_teacher(model, targets)
        try:
            hidden = model.backbone(inputs)
            logits = model.lm_head(hidden)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.flatten())
            loss = loss + BASE._auxiliary_loss(model, inputs.device)
        finally:
            BASE._clear_dense_teacher(model)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        value = float(loss.detach().cpu().item())
        if not math.isfinite(value):
            raise RuntimeError(f"训练shape {seq_len} loss非有限: {value}")
        rows.append({"phase": "train", "input_seq_len": seq_len, "batch_size": int(inputs.shape[0]), "loss": value, "status": "completed"})
        del inputs, targets, hidden, logits, loss
    model.eval()
    with torch.no_grad():
        for seq_len in (64, 128, 256, 512, 1024):
            inputs, targets = eval_batches[seq_len]
            inputs = inputs.to("cuda")
            targets = targets.to("cuda")
            BASE._set_dense_teacher(model, targets)
            try:
                hidden = model.backbone(inputs)
                logits = model.lm_head(hidden)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.flatten())
            finally:
                BASE._clear_dense_teacher(model)
            torch.cuda.synchronize()
            value = float(loss.detach().cpu().item())
            if not math.isfinite(value):
                raise RuntimeError(f"eval shape {seq_len} loss非有限: {value}")
            rows.append({"phase": "eval", "input_seq_len": seq_len, "batch_size": int(inputs.shape[0]), "loss": value, "status": "completed"})
            del inputs, targets, hidden, logits, loss
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "status": "completed",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "rows": rows,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "environment": environment_metadata(),
    }
    write_json(output, payload)
    return payload


def build_source_manifest(run_type: str, output: Path) -> list[dict[str, Any]]:
    expected_epoch = 4 if run_type == "formal" else 1
    rows: list[dict[str, Any]] = []
    for model, seed in JOB_ORDER:
        rp = result_path(model, seed, run_type)
        if not rp.exists():
            raise RuntimeError(f"缺少训练result: {rp}")
        result = json.loads(rp.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            raise RuntimeError(f"训练未完成: {rp}")
        config_path = Path(result["resolved_config_path"])
        for role in ("last", "best"):
            meta = result[f"{role}_checkpoint"]
            path = Path(meta["path"])
            current = _checkpoint_metadata(path)
            if current["file_sha256"] != meta["file_sha256"] or current["model_state_sha256"] != meta["model_state_sha256"]:
                raise RuntimeError(f"checkpoint hash漂移: {path}")
            if current["epoch"] > expected_epoch or (role == "last" and current["epoch"] != expected_epoch):
                raise RuntimeError(f"checkpoint epoch异常: {path}")
            rows.append({
                "source_id": f"{model}-s{seed}-{role}",
                "model": model,
                "config_family": "cb64-r16-joint" if model == "flash" else "gdnxk-h2-ek4-ev4-usegate0",
                "seed": seed,
                "data_seed": 123,
                "checkpoint_role": role,
                "checkpoint_path": str(path.resolve()),
                "checkpoint_file_sha256": current["file_sha256"],
                "checkpoint_model_state_sha256": current["model_state_sha256"],
                "checkpoint_epoch": current["epoch"],
                "checkpoint_metrics": current["metrics"],
                "train_config_path": str((Path(result["checkpoint_run_dir"]) / "train_config.json").resolve()),
                "resolved_config_path": str(config_path.resolve()),
                "resolved_config_sha256": result["resolved_config_sha256"],
                "source_result_path": str(rp.resolve()),
                "source_run_type": run_type,
                "active_state_capacity": 131_072,
                "trainable_params": EXPECTED_INIT[model]["params"],
            })
    write_json(output, rows)
    write_csv(output.with_suffix(".csv"), rows)
    if run_type == "smoke":
        write_json(GATE_DIR / "TRAINING_SMOKE_PASSED.json", {
            "status": "passed",
            "experiment_id": EXPERIMENT_ID,
            "completed_runs": 6,
            "logical_checkpoint_roles": len(rows),
            "manifest": str(output),
            "recorded_at_utc": utc_now(),
        })
    return rows


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="当前Flash/GDN基线的Longer-MQAR训练入口.")
    sub = root.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--output", type=Path, default=OUTPUT_ROOT / "preflight.json")
    train = sub.add_parser("train")
    train.add_argument("--model", choices=MODELS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--run-type", choices=("smoke", "formal"), required=True)
    train.add_argument("--resume", action="store_true")
    manifest = sub.add_parser("build-manifest")
    manifest.add_argument("--run-type", choices=("smoke", "formal"), required=True)
    manifest.add_argument("--output", type=Path, required=True)
    shapes = sub.add_parser("shape-smoke")
    shapes.add_argument("--model", choices=MODELS, required=True)
    shapes.add_argument("--output", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        payload = preflight(args.output)
        print(json.dumps({"status": payload["status"], "output": str(args.output)}, ensure_ascii=False))
        return 0
    if args.command == "train":
        train_one(args.model, args.seed, args.run_type, args.resume)
        return 0
    if args.command == "build-manifest":
        rows = build_source_manifest(args.run_type, args.output)
        print(json.dumps({"status": "completed", "sources": len(rows), "output": str(args.output)}, ensure_ascii=False))
        return 0
    if args.command == "shape-smoke":
        payload = shape_smoke(args.model, args.output)
        print(json.dumps({"status": payload["status"], "output": str(args.output)}, ensure_ascii=False))
        return 0
    raise RuntimeError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
