from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_ID = "20260729-03-mqar-seed124-remat-causal-diagnosis"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("ZOOLOGY_REPO_ROOT", SCRIPT_DIR.parents[4])
).resolve()
FLASH_ROOT = Path(
    os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")
).resolve()
PYTHON = Path("/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python")
EXPECTED_FLASH_COMMIT = "d7dbb1282d20ad860634ee4b8f0a74b948fe6c61"
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_INIT_FILE_HASH = "26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878"
EXPECTED_INIT_STATE_HASH = "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0"
EXPECTED_PARAMETERS = 1_160_390
SEED = 124
DATA_SEED = 123
VARIANTS = {
    "a0-fixed-off": "off",
    "a1-fixed-post-phase1": "post_phase1",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FLASH_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_ROOT / "src"))

BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts"
    / "20260724-01-flash-vqg-gd-residual-efficiency"
    / "efficiency_benchmark.py"
)


def _load_base():
    spec = importlib.util.spec_from_file_location("seed124_diag_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def run_tag() -> str:
    value = os.environ.get("MQAR_SEED124_DIAG_RUN_TAG", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError("MQAR_SEED124_DIAG_RUN_TAG must be a non-empty safe name.")
    return value


def run_root() -> Path:
    return SCRIPT_DIR / "outputs" / "3090" / run_tag()


def generated_root() -> Path:
    return (
        REPO_ROOT
        / "zoology/experiments/flash_vqg/generated"
        / f"{EXPERIMENT_ID}-{run_tag()}"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        )
        handle.flush()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def init_path() -> Path:
    return Path(BASE.CANONICAL_INIT).resolve()


def configure_numerics() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise RuntimeError("TRITON_F32_DEFAULT must be ieee.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be 0.")


def source_identity(variant: str, label: str) -> dict[str, str]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_tag": run_tag(),
        "label": label,
        "variant": variant,
        "remat_mode": VARIANTS[variant],
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "cache_content_sha256": EXPECTED_CACHE_HASH,
        "init_file_sha256": sha256_file(init_path()),
    }


def build_config(
    variant: str,
    *,
    label: str,
    max_train_steps: int,
):
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}.")
    if max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive.")
    config = BASE._build_flash_config("core", "triton", "triton_remat")
    config.seed = SEED
    config.data.seed = DATA_SEED
    config.data.batch_size = (64, 16)
    config.data.train_batch_segment_order = None
    config.data.test_batch_segment_order = None
    config.gradient_accumulation_steps = 4
    config.validations_per_epoch = 4
    config.max_epochs = 4
    config.max_train_steps = int(max_train_steps)
    config.max_validation_batches = None
    config.early_stopping_metric = None
    config.early_stopping_threshold = None
    config.precision = "amp_bfloat16"
    config.resume_enabled = True
    config.resume_stop_after_optimizer_step = None
    config.max_grad_scaler_skips = 0
    config.max_consecutive_grad_scaler_skips = 0
    config.resume_identity = source_identity(variant, label)
    config.training_runtime_initial_state = {}
    config.metrics_white_list = []
    config.logger.backend = "none"
    config.checkpoint.enabled = True
    config.checkpoint.save_best = True
    config.checkpoint.save_last = True
    config.checkpoint.best_metric = "valid/accuracy"
    config.checkpoint.best_mode = "max"
    config.checkpoint.root_dir = str(run_root() / "checkpoints" / label)
    config.init_checkpoint_path = str(init_path())
    config.init_checkpoint_strict = True
    config.launch_id = f"{EXPERIMENT_ID}-{run_tag()}-{label}"
    config.sweep_id = EXPERIMENT_ID
    config.run_id = f"{variant}-s{SEED}-bf16-b64ga4-{label}"
    config.training_telemetry_path = str(
        run_root() / "probes" / label / variant / "telemetry.jsonl"
    )
    kwargs = BASE._find_flash_kwargs(config.model)
    kwargs["fox_gd_residual_triton_input_policy"] = "fp32_boundary"
    kwargs["fox_gd_residual_remat_mode"] = VARIANTS[variant]
    return config


def serialize_config(config: Any) -> dict[str, Any]:
    from zoology.checkpoints import serialize_train_config

    return serialize_train_config(config)


def normalized_config(config: Any) -> dict[str, Any]:
    payload = serialize_config(config)
    payload["run_id"] = "<run>"
    payload["launch_id"] = "<launch>"
    payload["max_train_steps"] = "<steps>"
    payload["checkpoint"]["root_dir"] = "<checkpoint>"
    payload["training_telemetry_path"] = "<telemetry>"
    payload["resume_identity"]["label"] = "<label>"
    payload["resume_identity"]["variant"] = "<variant>"
    payload["resume_identity"]["remat_mode"] = "<remat>"
    return payload


def flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value, child))
        return result
    if isinstance(payload, list):
        result = {}
        for index, value in enumerate(payload):
            result.update(flatten(value, f"{prefix}[{index}]"))
        return result
    return {prefix: payload}


def config_differences(left: Any, right: Any) -> list[str]:
    left_flat = flatten(normalized_config(left))
    right_flat = flatten(normalized_config(right))
    keys = sorted(set(left_flat) | set(right_flat))
    return [key for key in keys if left_flat.get(key) != right_flat.get(key)]
