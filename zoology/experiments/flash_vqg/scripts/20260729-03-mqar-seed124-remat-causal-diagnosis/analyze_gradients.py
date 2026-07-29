#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import atomic_write_json


def load_event(path: Path, *, window: int, micro_step: int) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row.get("event") == "after_backward"
            and int(row.get("window", -1)) == window
            and int(row.get("micro_step", -1)) == micro_step
        ):
            if "grad_tensors" not in row:
                raise KeyError(f"Detailed gradients are missing from {path}.")
            return row
    raise KeyError(f"Backward event window={window}, micro={micro_step} not found: {path}")


def tensor_differences(
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    left_tensors = left["grad_tensors"]
    right_tensors = right["grad_tensors"]
    for name in sorted(set(left_tensors) | set(right_tensors)):
        left_value = left_tensors.get(name)
        right_value = right_tensors.get(name)
        left_hash = None if left_value is None else left_value.get("sha256")
        right_hash = None if right_value is None else right_value.get("sha256")
        if left_hash != right_hash:
            result.append(
                {
                    "name": name,
                    "left": left_value,
                    "right": right_value,
                }
            )
    return result


def analyze(paths: list[Path], *, window: int, micro_step: int) -> dict[str, Any]:
    events = {
        str(path.resolve()): load_event(path, window=window, micro_step=micro_step)
        for path in paths
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for path, event in events.items():
        groups[event["grad_sha256"]].append(path)
    representatives = [values[0] for _, values in sorted(groups.items())]
    comparisons = []
    for left_index, left_path in enumerate(representatives):
        for right_path in representatives[left_index + 1 :]:
            differences = tensor_differences(events[left_path], events[right_path])
            comparisons.append(
                {
                    "left": left_path,
                    "right": right_path,
                    "different_tensor_count": len(differences),
                    "different_tensors": differences,
                }
            )
    return {
        "window": window,
        "micro_step": micro_step,
        "trace_count": len(paths),
        "gradient_group_count": len(groups),
        "gradient_groups": dict(sorted(groups.items())),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--micro-step", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        args.traces,
        window=args.window,
        micro_step=args.micro_step,
    )
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
