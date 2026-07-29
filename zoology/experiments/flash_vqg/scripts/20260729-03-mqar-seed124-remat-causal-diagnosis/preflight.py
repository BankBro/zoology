#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from common import (
    BASE,
    EXPECTED_CACHE_HASH,
    EXPECTED_FLASH_COMMIT,
    EXPECTED_INIT_FILE_HASH,
    EXPECTED_INIT_STATE_HASH,
    EXPECTED_PARAMETERS,
    FLASH_ROOT,
    PYTHON,
    REPO_ROOT,
    atomic_write_json,
    build_config,
    config_differences,
    configure_numerics,
    git_value,
    init_path,
    run_root,
    sha256_file,
    utc_now,
)


def environment_metadata() -> dict[str, Any]:
    import fla
    import triton

    cuda_available = torch.cuda.is_available()
    return {
        "sys_executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "fla": fla.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used_bytes": (
            int(torch.cuda.device_memory_used()) if cuda_available else None
        ),
        "zoology_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "flash_commit": git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        "zoology_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "flash_branch": git_value(FLASH_ROOT, "branch", "--show-current"),
        "zoology_status": git_value(REPO_ROOT, "status", "--short"),
        "flash_status": git_value(FLASH_ROOT, "status", "--short"),
    }


def run_test(command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    output = "\n".join(
        value.strip() for value in (result.stdout, result.stderr) if value.strip()
    )
    return {
        "command": command,
        "return_code": int(result.returncode),
        "passed": result.returncode == 0,
        "output_tail": "\n".join(output.splitlines()[-30:]),
    }


def model_audit(config: Any) -> dict[str, Any]:
    model = BASE.LanguageModel(config.model)
    payload = torch.load(init_path(), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    kwargs = BASE._find_flash_kwargs(config.model)
    return {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_sha256": BASE._state_dict_hash(model.state_dict()),
        "strict": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "block_len": kwargs.get("block_len"),
        "rank": kwargs.get("fox_gd_residual_rank"),
        "read_topk": kwargs.get("fox_remote_read_topk"),
        "write_topk": kwargs.get("fox_gd_residual_write_topk"),
        "grouped_backend": kwargs.get("fox_gd_residual_grouped_chunk_backend"),
        "selected_backend": kwargs.get("fox_gd_residual_selected_read_backend"),
        "input_policy": kwargs.get("fox_gd_residual_triton_input_policy"),
        "remat_mode": kwargs.get("fox_gd_residual_remat_mode"),
    }


def main() -> int:
    configure_numerics()
    output = run_root() / "preflight.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {output}")
    env = environment_metadata()
    a0 = build_config("a0-fixed-off", label="preflight", max_train_steps=16)
    a1 = build_config(
        "a1-fixed-post-phase1",
        label="preflight",
        max_train_steps=16,
    )
    cache = BASE._cache_content_hash(a0.data)
    differences = config_differences(a0, a1)
    a0_audit = model_audit(a0)
    a1_audit = model_audit(a1)
    zoology_test = run_test(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            "tests/test_mqar_seed124_remat_causal_diagnosis.py",
        ],
        REPO_ROOT,
    )
    flash_test = run_test(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            "tests/test_fox_gd_residual_v1.py",
            "-k",
            "selected_read",
        ],
        FLASH_ROOT,
    )
    common_audit = {
        key: value
        for key, value in a0_audit.items()
        if key != "remat_mode"
    }
    checks = {
        "python_path": Path(sys.executable).resolve() == PYTHON.resolve(),
        "python_version": env["python"] == "3.12.11",
        "torch": env["torch"] == "2.6.0+cu118",
        "cuda": env["torch_cuda"] == "11.8",
        "triton": env["triton"] == "3.2.0",
        "fla": env["fla"] == "0.4.2",
        "cuda_available": env["cuda_available"],
        "gpu": env["gpu_name"] == "NVIDIA GeForce RTX 3090",
        "visible_gpu": env["cuda_visible_devices"] == "0",
        "gpu_free": env["gpu_used_bytes"] is not None
        and env["gpu_used_bytes"] < 1024**3,
        "zoology_clean": not env["zoology_status"],
        "flash_clean": not env["flash_status"],
        "zoology_branch": bool(
            re.fullmatch(
                r"\d{8}-\d{6}-mqar-seed124-remat-causal-diagnosis",
                env["zoology_branch"],
            )
        ),
        "flash_commit": env["flash_commit"] == EXPECTED_FLASH_COMMIT,
        "cache": cache.get("combined_content_sha256") == EXPECTED_CACHE_HASH,
        "init_file": sha256_file(init_path()) == EXPECTED_INIT_FILE_HASH,
        "single_variable": len(differences) == 1
        and differences[0].endswith("fox_gd_residual_remat_mode"),
        "model_common": common_audit
        == {key: value for key, value in a1_audit.items() if key != "remat_mode"},
        "model_parameters": a0_audit["parameters"] == EXPECTED_PARAMETERS,
        "model_state": a0_audit["state_sha256"] == EXPECTED_INIT_STATE_HASH,
        "model_shape": a0_audit["block_len"] == 32
        and a0_audit["rank"] == 16
        and a0_audit["read_topk"] == 16
        and a0_audit["write_topk"] == 4,
        "model_backends": a0_audit["grouped_backend"] == "triton"
        and a0_audit["selected_backend"] == "triton_remat"
        and a0_audit["input_policy"] == "fp32_boundary",
        "remat_modes": a0_audit["remat_mode"] == "off"
        and a1_audit["remat_mode"] == "post_phase1",
        "zoology_tests": zoology_test["passed"],
        "flash_tests": flash_test["passed"],
    }
    payload = {
        "experiment_id": "20260729-03-mqar-seed124-remat-causal-diagnosis",
        "recorded_at_utc": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "environment": env,
        "cache": cache,
        "config_differences": differences,
        "a0_model_audit": a0_audit,
        "a1_model_audit": a1_audit,
        "zoology_test": zoology_test,
        "flash_test": flash_test,
        "checks": checks,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if payload["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
