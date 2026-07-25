from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260726-01-mqar-precision-profile"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")
).resolve()
FLASH_ROOT = Path(
    os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FLASH_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FLASH_ROOT / "src"))
PYTHON = Path("/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python")
MACHINES = {
    "2080ti": {
        "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
        "visible_gpu": "1",
        "train_precisions": ("fp32", "fp16"),
        "eval_precisions": ("fp32", "fp16"),
    },
    "3090": {
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "visible_gpu": "0",
        "train_precisions": ("fp32", "fp16", "bf16"),
        "eval_precisions": ("fp32", "fp16", "bf16"),
    },
}
MODELS = ("gdn", "flash")
SEEDS = (123, 124, 125)
STANDARD_SHAPES = (
    (64, 4),
    (64, 8),
    (64, 16),
    (128, 32),
    (256, 64),
    (512, 64),
    (512, 128),
    (1024, 256),
)
LONGER_SHAPES = (
    (1024, 256),
    (2048, 512),
    (4096, 1024),
    (8190, 512),
    (8190, 2047),
)
ALL_SHAPES = STANDARD_SHAPES + LONGER_SHAPES
ALL_EVAL_CASES = tuple(
    (sequence_length, num_kv_pairs, 1000)
    for sequence_length, num_kv_pairs in STANDARD_SHAPES
) + tuple(
    (sequence_length, num_kv_pairs, 500)
    for sequence_length, num_kv_pairs in LONGER_SHAPES
)
BATCH_CANDIDATES = (128, 64, 32, 16, 8, 4, 2, 1)
LONGER_DATASET_HASHES = {
    "1024x256": "f30f1e09e1300deb2e0f430d7c50c47a77f58f43657aa4d6f425f938a91a9acb",
    "2048x512": "e446efc9f4dc774377c56c403a468c747870beb18ff0320f15df794dcb69f015",
    "4096x1024": "0981292ce8ebea1402a4c3a7a6655dfc1de42688c329bd5ea921541a0f1d16ed",
    "8190x512": "37a8533a985fabd3db03daf9dfd3a8e0936513ab724039c63b9a22bf352d002d",
    "8190x2047": "8c16a91ea127bb7bf2130238ecd56b49fb1fd73bb3e0c2f4c586ddba8e9b97a9",
}
PRECISION_CONFIG = {
    "fp32": "float32",
    "fp16": "amp_float16",
    "bf16": "amp_bfloat16",
}
GDN_KERNEL_DTYPE = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def machine_name() -> str:
    value = os.environ.get("MQAR_PRECISION_MACHINE", "2080ti").strip().lower()
    if value not in MACHINES:
        raise RuntimeError(f"Unsupported machine: {value}")
    return value


def output_root(machine: str | None = None) -> Path:
    machine = machine or machine_name()
    return SCRIPT_DIR / "outputs" / "machines" / machine


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shape_name(sequence_length: int, num_kv_pairs: int) -> str:
    return f"{sequence_length}x{num_kv_pairs}"


def examples_for_shape(shape: tuple[int, int]) -> int:
    return 1000 if shape in STANDARD_SHAPES else 500


def training_descriptors(machine: str) -> list[dict[str, Any]]:
    rows = []
    for precision in MACHINES[machine]["train_precisions"]:
        for seed in SEEDS:
            for model in MODELS:
                rows.append(
                    {
                        "machine": machine,
                        "model": model,
                        "seed": seed,
                        "train_precision": precision,
                        "descriptor_id": (
                            f"{machine}-{model}-s{seed}-{precision}"
                        ),
                    }
                )
    return rows


def expected_training_count(machine: str | None = None) -> int:
    if machine is not None:
        return len(training_descriptors(machine))
    return sum(len(training_descriptors(value)) for value in MACHINES)
