from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
os.environ.setdefault("GDN_KERNEL_DTYPE", "float32")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import torch


EXPERIMENT_ID = "20260724-02-gdn-ek4-fla-compatibility"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")
).resolve()
BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/"
    / "20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module(BASE_SCRIPT, "gdn_fla_compatibility_efficiency_base")
BASE.EXPERIMENT_ID = EXPERIMENT_ID
BASE.SCRIPT_DIR = SCRIPT_DIR
GDN_CANONICAL_INIT = BASE.GDN_CANONICAL_INIT
FLASH_CANONICAL_INIT = BASE.CANONICAL_INIT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(root: str | None, *args: str) -> str | None:
    if not root:
        return None


def _cuda_device_attribute(attribute: int) -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        runtime = ctypes.CDLL("libcudart.so")
        value = ctypes.c_int()
        result = runtime.cudaDeviceGetAttribute(
            ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_int(0)
        )
        return value.value if result == 0 else None
    except OSError:
        return None
    path = Path(root)
    if not path.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict[str, Any]:
    import fla
    import triton

    package_root = Path(fla.__file__).resolve().parent
    state_kernel = package_root / "ops/common/chunk_delta_h.py"
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    source_root = os.environ.get("FLA_SOURCE_ROOT")
    return {
        "fla_variant": os.environ.get("FLA_VARIANT", "unknown"),
        "fla_version": _distribution_version("flash-linear-attention")
        or getattr(fla, "__version__", None),
        "fla_module": str(Path(fla.__file__).resolve()),
        "fla_state_kernel": str(state_kernel),
        "fla_state_kernel_sha256": sha256_file(state_kernel) if state_kernel.exists() else None,
        "fla_source_root": source_root,
        "fla_source_commit": _git_value(source_root, "rev-parse", "HEAD"),
        "fla_source_status": _git_value(source_root, "status", "--short"),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gdn_kernel_dtype": os.environ.get("GDN_KERNEL_DTYPE"),
        "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "gpu_name": props.name if props is not None else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if props is not None else None,
        "gpu_shared_memory_per_block": getattr(
            props, "shared_memory_per_block", None
        )
        or _cuda_device_attribute(8),
        "gpu_shared_memory_per_block_optin": getattr(
            props, "shared_memory_per_block_optin", None
        )
        or _cuda_device_attribute(97),
    }


_BASE_ENVIRONMENT = BASE._environment


def _augmented_environment() -> dict[str, Any]:
    payload = dict(_BASE_ENVIRONMENT())
    payload.update(environment_metadata())
    return payload


BASE._environment = _augmented_environment


def configure_numerics() -> None:
    BASE._configure_numerics()
    if os.environ.get("GDN_KERNEL_DTYPE") != "float32":
        raise RuntimeError("正式兼容性口径要求 GDN_KERNEL_DTYPE=float32.")
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise RuntimeError("正式兼容性口径要求 TRITON_F32_DEFAULT=ieee.")


def write_json(path: Path, payload: Any) -> None:
    BASE._write_json(path, payload)


def build_model_config(model_name: str):
    flash = BASE._build_flash_config("core", "triton", "triton_remat")
    gdn = BASE._build_gdn_config(flash.data)
    if model_name == "flash":
        return flash, flash
    if model_name == "gdn":
        return gdn, flash
    raise ValueError(f"未知模型: {model_name}")
