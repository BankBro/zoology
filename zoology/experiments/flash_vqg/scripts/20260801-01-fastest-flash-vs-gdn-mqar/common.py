from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260801-01-fastest-flash-vs-gdn-mqar"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")
).resolve()
FLASH_ROOT = Path(
    os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")
).resolve()
for root in (REPO_ROOT, FLASH_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

PYTHON = Path("/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python")
EXPECTED_FLASH_COMMIT = "396ae65b89b53aad316fbbf7daf55a92a551d684"
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_INIT = {
    "flash": {
        "file": "26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878",
        "state": "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0",
        "params": 1_160_390,
    },
    "gdn": {
        "file": "a4e76e7776bdc83a582c2613cd7d9782100a9148aa119763ecaaeeb8273f7b71",
        "state": "bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6",
        "params": 1_335_942,
    },
}

FASTEST = "flash-fastest"
CANONICAL = "flash-canonical"
GDN = "gdn"
ARMS = (FASTEST, CANONICAL, GDN)
SEEDS = (123, 124, 125)
FORMAL_ORDER = (
    (FASTEST, 123),
    (CANONICAL, 123),
    (GDN, 123),
    (GDN, 124),
    (FASTEST, 124),
    (CANONICAL, 124),
    (CANONICAL, 125),
    (GDN, 125),
    (FASTEST, 125),
)

STANDARD_CASES = (
    (64, 4, 1000, None),
    (64, 8, 1000, None),
    (64, 16, 1000, None),
    (128, 32, 1000, None),
    (256, 64, 1000, None),
    (512, 64, 1000, None),
    (512, 128, 1000, None),
    (1024, 256, 1000, None),
)
LONGER_CASES = (
    (1024, 256, 500, "f30f1e09e1300deb2e0f430d7c50c47a77f58f43657aa4d6f425f938a91a9acb"),
    (2048, 512, 500, "e446efc9f4dc774377c56c403a468c747870beb18ff0320f15df794dcb69f015"),
    (4096, 1024, 500, "0981292ce8ebea1402a4c3a7a6655dfc1de42688c329bd5ea921541a0f1d16ed"),
    (8190, 512, 500, "37a8533a985fabd3db03daf9dfd3a8e0936513ab724039c63b9a22bf352d002d"),
    (8190, 2047, 500, "8c16a91ea127bb7bf2130238ecd56b49fb1fd73bb3e0c2f4c586ddba8e9b97a9"),
)
EVAL_CASES = STANDARD_CASES + LONGER_CASES
EXTRAPOLATION_SHAPES = ("2048x512", "4096x1024", "8190x512", "8190x2047")
BATCH_CANDIDATES = (128, 64, 32, 16, 8, 4, 2, 1)


def run_tag() -> str:
    value = os.environ.get("MQAR_FASTEST_GDN_RUN_TAG", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError("MQAR_FASTEST_GDN_RUN_TAG must be a non-empty safe name.")
    return value


def run_root() -> Path:
    return SCRIPT_DIR / "outputs" / "3090" / run_tag()


def generated_root() -> Path:
    name = f"{EXPERIMENT_ID}-{run_tag()}"
    return REPO_ROOT / "zoology/experiments/flash_vqg/generated" / name


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def arm_model(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm: {arm}.")
    return "gdn" if arm == GDN else "flash"


def descriptor(arm: str, seed: int) -> dict[str, Any]:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError(f"Unsupported descriptor: {arm}, {seed}.")
    return {
        "descriptor_id": f"3090-{arm}-s{seed}-bf16",
        "machine": "3090",
        "arm": arm,
        "model": arm_model(arm),
        "seed": seed,
        "data_seed": 123,
        "train_precision": "bf16",
    }


def training_descriptors(phase: str) -> list[dict[str, Any]]:
    pairs = FORMAL_ORDER if phase == "formal" else tuple((arm, 123) for arm in ARMS)
    return [descriptor(arm, seed) for arm, seed in pairs]


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
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_id(case: tuple[int, int, int, str | None]) -> str:
    sequence_length, num_kv_pairs, num_examples, _hash = case
    return f"{sequence_length}x{num_kv_pairs}-n{num_examples}"
