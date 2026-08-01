from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260801-02-flash-late-degradation-causal-diagnosis"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")
).resolve()
FLASH_ROOT = Path(
    os.environ.get("FLASH_VQG_ROOT", "/home/lyj/mnt/project/Flash-VQG")
).resolve()
PYTHON = Path("/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python")

ZOOLOGY_BASE_COMMIT = "829ab6a960f118c65da93d62ea86463d74a7ef19"
EXPECTED_FLASH_COMMIT = "182180fd7a0770caf72b2dec6e6d27616dfd31a3"
EXPECTED_CACHE_HASH = "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8"
EXPECTED_INIT_FILE_HASH = "26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878"
EXPECTED_INIT_STATE_HASH = "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0"
EXPECTED_PARAMETERS = 1_160_390

SEEDS = (123, 124, 125)
GATE_MODES = ("fixed", "default")
PHASES = ("smoke", "screen", "formal")

TORCH_BACKWARD = "torch_chunked"
S1_BACKWARD = "triton_deterministic_s1_head"
W2_BACKWARD = "triton_state_owner_r1a_s1_w2"


def _canonical(
    block_len: int,
    local_num_blocks: int,
    backward: str,
    chunk: int,
) -> dict[str, Any]:
    return {
        "family": "canonical",
        "block_len": block_len,
        "local_num_blocks": local_num_blocks,
        "selected_backward": backward,
        "selected_chunk": chunk,
    }


def _fastest(
    block_len: int,
    local_num_blocks: int,
    backward: str = W2_BACKWARD,
    chunk: int = 8192,
) -> dict[str, Any]:
    return {
        "family": "fastest",
        "block_len": block_len,
        "local_num_blocks": local_num_blocks,
        "selected_backward": backward,
        "selected_chunk": chunk,
    }


ARM_SPECS: dict[str, dict[str, Any]] = {
    "ctrl-current": _canonical(64, 2, S1_BACKWARD, 8192),
    "ctrl-bridge": _canonical(32, 2, TORCH_BACKWARD, 2048),
    "factor-block": _canonical(64, 2, TORCH_BACKWARD, 2048),
    "factor-backend": _canonical(32, 2, S1_BACKWARD, 2048),
    "factor-chunk": _canonical(32, 2, TORCH_BACKWARD, 8192),
    "interaction-block-backend": _canonical(64, 2, S1_BACKWARD, 2048),
    "interaction-block-chunk": _canonical(64, 2, TORCH_BACKWARD, 8192),
    "interaction-backend-chunk": _canonical(32, 2, S1_BACKWARD, 8192),
    "mechanism-window128": _canonical(32, 4, TORCH_BACKWARD, 2048),
    "mechanism-boundary64": _canonical(64, 1, TORCH_BACKWARD, 2048),
    "fastest-current": _fastest(64, 2),
    "fastest-block32-local2": _fastest(32, 2),
    "fastest-block32-local4": _fastest(32, 4),
    "fastest-block64-local1": _fastest(64, 1),
    "fastest-selected-torch": _fastest(64, 2, TORCH_BACKWARD, 2048),
    "fastest-chunk2048": _fastest(64, 2, W2_BACKWARD, 2048),
    "fastest-bridge": _fastest(32, 2, TORCH_BACKWARD, 2048),
}
ARMS = tuple(ARM_SPECS)

SINGLE_FACTOR_ARMS = (
    "factor-block",
    "factor-backend",
    "factor-chunk",
)
INTERACTION_ARMS = (
    "interaction-block-backend",
    "interaction-block-chunk",
    "interaction-backend-chunk",
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


def run_tag() -> str:
    value = os.environ.get("MQAR_LATE_DEGRADATION_RUN_TAG", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError("MQAR_LATE_DEGRADATION_RUN_TAG must be a non-empty safe name.")
    return value


def run_root() -> Path:
    return SCRIPT_DIR / "outputs" / "3090" / run_tag()


def generated_root() -> Path:
    return (
        REPO_ROOT
        / "zoology/experiments/flash_vqg/generated"
        / f"{EXPERIMENT_ID}-{run_tag()}"
    )


def arm_spec(arm: str) -> dict[str, Any]:
    if arm not in ARM_SPECS:
        raise ValueError(f"Unsupported arm: {arm}.")
    return dict(ARM_SPECS[arm])


def descriptor(arm: str, seed: int, gate_mode: str) -> dict[str, Any]:
    spec = arm_spec(arm)
    if seed not in SEEDS or gate_mode not in GATE_MODES:
        raise ValueError(f"Unsupported descriptor: {arm}, {seed}, {gate_mode}.")
    return {
        "descriptor_id": f"3090-{arm}-s{seed}-bf16-{gate_mode}",
        "machine": "3090",
        "model": "flash",
        "arm": arm,
        "family": spec["family"],
        "seed": seed,
        "data_seed": 123,
        "train_precision": "bf16",
        "gate_mode": gate_mode,
    }


def case_id(case: tuple[int, int, int, str | None]) -> str:
    sequence_length, num_kv_pairs, num_examples, _ = case
    return f"{sequence_length}x{num_kv_pairs}-n{num_examples}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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
