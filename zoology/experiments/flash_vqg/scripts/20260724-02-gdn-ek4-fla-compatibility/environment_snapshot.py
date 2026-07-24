#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import EXPERIMENT_ID, environment_metadata, write_json


def command(
    args: list[str], *, environment: dict[str, str] | None = None
) -> dict[str, object]:
    completed = subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="保存候选环境的可审计依赖快照.")
    parser.add_argument("--machine", choices=("2080ti", "3090"), required=True)
    parser.add_argument("--fla-variant", choices=("current040", "v042", "v050"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["FLA_VARIANT"] = args.fla_variant
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "machine": args.machine,
        "fla_variant": args.fla_variant,
        "environment": environment_metadata(),
        "pip_freeze": command([sys.executable, "-m", "pip", "freeze", "--all"]),
        "pip_check": command([sys.executable, "-m", "pip", "check"]),
        "pip_check_isolated": command(
            [sys.executable, "-m", "pip", "check"],
            environment=os.environ | {"PYTHONNOUSERSITE": "1"},
        ),
        "conda_list": command(
            ["/home/lyj/miniconda3/bin/conda", "list", "--json", "-p", sys.prefix]
        ),
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    write_json(args.output, payload)
    print(args.output)
    return 0 if payload["pip_check_isolated"]["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
