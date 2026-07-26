#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOCAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(LOCAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPT_DIR))

from common import (  # noqa: E402
    EXPERIMENT_ID,
    atomic_write_json,
    load_json,
    output_root,
    stable_json_sha256,
    utc_now,
    PYTHON,
)


REMOTE_HOST = "lyj@192.168.2.114"
REMOTE_CONTAINER = "Flash-VQG-tun"
REMOTE_PROJECT = Path("/home/lyj/mnt/project/zoology")
RELATIVE_OUTPUT = Path(
    "zoology/experiments/flash_vqg/scripts/"
    "20260726-01-mqar-precision-profile/outputs/machines/3090"
)


def remote_read(relative_path: Path) -> dict[str, Any] | None:
    path = REMOTE_PROJECT / RELATIVE_OUTPUT / relative_path
    remote_command = shlex.join(
        [
            "docker",
            "exec",
            "-u",
            "lyj",
            REMOTE_CONTAINER,
            "cat",
            str(path),
        ]
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        REMOTE_HOST,
        remote_command,
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def remote_write(relative_path: Path, payload: dict[str, Any]) -> None:
    path = REMOTE_PROJECT / RELATIVE_OUTPUT / relative_path
    encoded = base64.b64encode(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ).decode("ascii")
    code = (
        "import base64,pathlib;"
        f"p=pathlib.Path({str(path)!r});"
        "p.parent.mkdir(parents=True,exist_ok=True);"
        "t=p.with_suffix(p.suffix+'.tmp');"
        f"t.write_bytes(base64.b64decode({encoded!r}));"
        "t.replace(p)"
    )
    remote_command = shlex.join(
        [
            "docker",
            "exec",
            "-u",
            "lyj",
            REMOTE_CONTAINER,
            "/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python",
            "-c",
            code,
        ]
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        REMOTE_HOST,
        remote_command,
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Could not write remote gate: {result.stderr}")


def build_global_gate(
    local_gate: dict[str, Any],
    remote_gate: dict[str, Any],
    local_preflight: dict[str, Any],
    remote_preflight: dict[str, Any],
) -> dict[str, Any]:
    for payload in (local_gate, remote_gate, local_preflight, remote_preflight):
        if payload.get("status") != "passed":
            raise RuntimeError("A source gate has not passed.")
    local_env = local_preflight["environment"]
    remote_env = remote_preflight["environment"]
    def config_hashes(payload: dict[str, Any]) -> dict[tuple[Any, ...], str]:
        return {
            (
                row["model"],
                int(row["seed"]),
                row["train_precision"],
                row["phase"],
            ): row["normalized_config_sha256"]
            for row in payload["jobs"]
        }

    checks = {
        "zoology_commit": local_env["zoology_commit"] == remote_env["zoology_commit"],
        "flash_commit": local_env["flash_commit"] == remote_env["flash_commit"],
        "cache_content": (
            local_preflight["cache"]["combined_content_sha256"]
            == remote_preflight["cache"]["combined_content_sha256"]
        ),
        "machines": {local_gate["machine"], remote_gate["machine"]}
        == {"2080ti", "3090"},
        "shared_config_hashes": all(
            remote_hash == config_hashes(local_preflight).get(key)
            for key, remote_hash in config_hashes(remote_preflight).items()
            if key[2] in {"fp32", "fp16"}
        ),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "zoology_commit": local_env["zoology_commit"],
        "flash_commit": local_env["flash_commit"],
        "cache_content_sha256": local_preflight["cache"][
            "combined_content_sha256"
        ],
        "local_smoke_binding": local_gate["binding_sha256"],
        "remote_smoke_binding": remote_gate["binding_sha256"],
        "binding_sha256": stable_json_sha256(
            {
                "local": local_gate,
                "remote": remote_gate,
                "local_preflight": local_preflight,
                "remote_preflight": remote_preflight,
            }
        ),
        "recorded_at_utc": utc_now(),
    }
    if payload["status"] != "passed":
        raise RuntimeError(f"Global gate checks failed: {checks}")
    return payload


def wait_and_release(poll_seconds: int) -> dict[str, Any]:
    local_root = output_root("2080ti")
    status_path = local_root / "coordinator-status.json"
    while True:
        local_gate_path = local_root / "gates" / "LOCAL_SMOKE_PASSED.json"
        local_preflight_path = local_root / "preflight.json"
        local_gate = load_json(local_gate_path) if local_gate_path.exists() else None
        local_preflight = (
            load_json(local_preflight_path) if local_preflight_path.exists() else None
        )
        remote_gate = remote_read(Path("gates/LOCAL_SMOKE_PASSED.json"))
        remote_preflight = remote_read(Path("preflight.json"))
        atomic_write_json(
            status_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "waiting",
                "local_ready": bool(local_gate),
                "remote_ready": bool(remote_gate),
                "updated_at_utc": utc_now(),
            },
        )
        if all((local_gate, local_preflight, remote_gate, remote_preflight)):
            payload = build_global_gate(
                local_gate,
                remote_gate,
                local_preflight,
                remote_preflight,
            )
            relative = Path("gates/GLOBAL_FORMAL_GATE.json")
            atomic_write_json(local_root / relative, payload)
            remote_write(relative, payload)
            atomic_write_json(
                status_path,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "released",
                    "global_binding_sha256": payload["binding_sha256"],
                    "updated_at_utc": utc_now(),
                },
            )
            return payload
        time.sleep(poll_seconds)


def wait_for_completion(poll_seconds: int) -> None:
    local_root = output_root("2080ti")
    status_path = local_root / "coordinator-status.json"
    while True:
        local_queue_path = local_root / "status.json"
        local_status = (
            load_json(local_queue_path) if local_queue_path.exists() else None
        )
        remote_status = remote_read(Path("status.json"))
        statuses = {
            "2080ti": None if local_status is None else local_status.get("status"),
            "3090": None if remote_status is None else remote_status.get("status"),
        }
        atomic_write_json(
            status_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "waiting_formal_completion",
                "machine_statuses": statuses,
                "updated_at_utc": utc_now(),
            },
        )
        if "failed" in statuses.values():
            raise RuntimeError(f"A machine queue failed: {statuses}")
        if set(statuses.values()) == {"completed"}:
            return
        time.sleep(poll_seconds)


def collect_results() -> None:
    script = LOCAL_SCRIPT_DIR / "collect_results.py"
    log_path = output_root("2080ti") / "logs" / "collect-results.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            [str(PYTHON), str(script)],
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Result collection failed: {log_path}")
    atomic_write_json(
        output_root("2080ti") / "coordinator-status.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "completed",
            "report": str(
                Path("/home/lyj/mnt/project/zoology/docs")
                / f"{EXPERIMENT_ID}-report.md"
            ),
            "updated_at_utc": utc_now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--no-collect", action="store_true")
    args = parser.parse_args()
    wait_and_release(args.poll_seconds)
    if not args.no_collect:
        wait_for_completion(max(args.poll_seconds, 30))
        collect_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
