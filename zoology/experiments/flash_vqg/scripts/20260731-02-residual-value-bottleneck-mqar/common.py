from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260731-02-residual-value-bottleneck-mqar"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
FLASH_ROOT = Path(os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")).resolve()
for root in (REPO_ROOT, FLASH_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

PYTHON = Path("/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python")
EXPECTED_FLASH_COMMIT = "cc3f92b8a972f1c51c3deabeafd0d9f180bc2b16"
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_INIT_FILE_HASH = "26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878"
EXPECTED_INIT_STATE_HASH = "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0"
BASELINE = "u64-a1-s1"
VARIANT_DIMS = {BASELINE: 64, "u32-a1-s1": 32, "u16-a1-s1": 16}
VARIANTS = tuple(VARIANT_DIMS)
EXPECTED_PARAMETERS = {
    BASELINE: 1_160_390,
    "u32-a1-s1": 1_164_486,
    "u16-a1-s1": 1_162_438,
}
EXPECTED_STATE_HASHES = {
    BASELINE: EXPECTED_INIT_STATE_HASH,
    "u32-a1-s1": "e02703fbbd202cffdbad83b3768f994f0db4500ba470f49235ab0a5ede1714b4",
    "u16-a1-s1": "4705b6d8f27badbf1322ff22bcb672fcc873f0d7f1c4ca8d06ec107b9101f373",
}
SEEDS = (123, 124, 125)
FORMAL_ORDER = tuple((variant, seed) for seed in SEEDS for variant in VARIANTS)
LONGER_CASES = (
    (1024, 256, 128, "f30f1e09e1300deb2e0f430d7c50c47a77f58f43657aa4d6f425f938a91a9acb"),
    (2048, 512, 64, "e446efc9f4dc774377c56c403a468c747870beb18ff0320f15df794dcb69f015"),
    (4096, 1024, 32, "0981292ce8ebea1402a4c3a7a6655dfc1de42688c329bd5ea921541a0f1d16ed"),
    (8190, 512, 16, "37a8533a985fabd3db03daf9dfd3a8e0936513ab724039c63b9a22bf352d002d"),
    (8190, 2047, 16, "8c16a91ea127bb7bf2130238ecd56b49fb1fd73bb3e0c2f4c586ddba8e9b97a9"),
)
EXTRAPOLATION_SHAPES = ("2048x512", "4096x1024", "8190x512", "8190x2047")
Q0_RELATIVE_DELTA_MIN = -0.10


def run_tag() -> str:
    value = os.environ.get("MQAR_RESIDUAL_VALUE_RUN_TAG", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError("MQAR_RESIDUAL_VALUE_RUN_TAG must be a non-empty safe name.")
    return value


def run_root() -> Path:
    return SCRIPT_DIR / "outputs" / "3090" / run_tag()


def generated_root() -> Path:
    return REPO_ROOT / "zoology/experiments/flash_vqg/generated" / f"{EXPERIMENT_ID}-{run_tag()}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def descriptor(variant: str, seed: int) -> dict[str, Any]:
    if variant not in VARIANT_DIMS or seed not in SEEDS:
        raise ValueError(f"Unsupported descriptor: {variant}, {seed}.")
    return {
        "descriptor_id": f"3090-{variant}-s{seed}-bf16",
        "machine": "3090",
        "model": "flash",
        "variant": variant,
        "residual_value_dim": VARIANT_DIMS[variant],
        "seed": seed,
        "data_seed": 123,
        "train_precision": "bf16",
    }


def training_descriptors() -> list[dict[str, Any]]:
    return [descriptor(variant, seed) for variant, seed in FORMAL_ORDER]
