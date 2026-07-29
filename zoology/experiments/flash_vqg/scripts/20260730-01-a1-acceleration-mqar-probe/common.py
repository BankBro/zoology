from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260730-01-a1-acceleration-mqar-probe"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/lyj/mnt/project/zoology")
FLASH_ROOT = Path("/home/lyj/mnt/project/Flash-VQG")
PYTHON = Path("/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python")
EXPECTED_FLASH_COMMIT = "114eadbd1d2e3c9a43b927e54f6ad9a2692c40e8"
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_INIT_FILE_HASH = "26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878"
EXPECTED_INIT_STATE_HASH = "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0"
EXPECTED_PARAMETERS = 1_160_390
SEED = 123
VARIANTS = {
    "a1-reference": {"block_len": 32, "write_topk": 4, "read_topk": 16},
    "a1-block256-k2r8": {"block_len": 256, "write_topk": 2, "read_topk": 8},
}
LONGER_CASES = (
    (1024, 256, 32, "f30f1e09e1300deb2e0f430d7c50c47a77f58f43657aa4d6f425f938a91a9acb"),
    (2048, 512, 32, "e446efc9f4dc774377c56c403a468c747870beb18ff0320f15df794dcb69f015"),
    (4096, 1024, 16, "0981292ce8ebea1402a4c3a7a6655dfc1de42688c329bd5ea921541a0f1d16ed"),
    (8190, 512, 16, "37a8533a985fabd3db03daf9dfd3a8e0936513ab724039c63b9a22bf352d002d"),
    (8190, 2047, 16, "8c16a91ea127bb7bf2130238ecd56b49fb1fd73bb3e0c2f4c586ddba8e9b97a9"),
)


def run_tag() -> str:
    value = os.environ.get("MQAR_A1_ACCEL_RUN_TAG", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError("MQAR_A1_ACCEL_RUN_TAG must be a non-empty safe name.")
    return value


def run_root() -> Path:
    return SCRIPT_DIR / "outputs" / "2080ti" / run_tag()


def generated_root() -> Path:
    return REPO_ROOT / "zoology/experiments/flash_vqg/generated" / f"{EXPERIMENT_ID}-{run_tag()}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
