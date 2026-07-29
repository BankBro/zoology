#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import atomic_write_json


TRAIN_EVENTS = {
    "module_forward",
    "forward",
    "after_backward",
    "before_optimizer",
    "after_optimizer",
    "after_zero_grad",
}
IGNORED_FIELDS = {"recorded_at_utc"}


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def event_key(row: dict[str, Any]) -> tuple[Any, ...]:
    event = row["event"]
    micro_step = int(row.get("micro_step", -1))
    if event == "module_forward":
        order = micro_step * 10
    elif event == "forward":
        order = micro_step * 10 + 1
    elif event == "after_backward":
        order = micro_step * 10 + 2
    else:
        order = {
            "before_optimizer": 100,
            "after_optimizer": 101,
            "after_zero_grad": 102,
        }[event]
    key: list[Any] = [int(row.get("window", -1)), order, event]
    if event in {"forward", "after_backward", "module_forward"}:
        key.append(micro_step)
    if event == "module_forward":
        key.extend([row.get("module"), int(row.get("call_index", -1))])
    return tuple(key)


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in IGNORED_FIELDS}


def first_leaf_difference(
    left: Any,
    right: Any,
    prefix: str = "",
) -> tuple[str, Any, Any] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                return child, left.get(key), right.get(key)
            difference = first_leaf_difference(left[key], right[key], child)
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{prefix}.length", len(left), len(right)
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = first_leaf_difference(
                left_value,
                right_value,
                f"{prefix}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return prefix, left, right
    return None


def classify(event: str, field: str) -> str:
    if field.startswith("input") or field.startswith("target"):
        return "data_or_sampler"
    if "rng" in field:
        return "rng_lifecycle"
    if event == "module_forward":
        return "module_forward"
    if event == "forward":
        return "forward_loss"
    if event == "after_backward" and "grad" in field:
        return "backward_gradient"
    if event == "after_optimizer" and "model" in field:
        return "optimizer_update"
    if event == "before_optimizer" and "model" in field:
        return "previous_update"
    return event


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_rows = [row for row in load_trace(left_path) if row["event"] in TRAIN_EVENTS]
    right_rows = [row for row in load_trace(right_path) if row["event"] in TRAIN_EVENTS]
    left = {event_key(row): row for row in left_rows}
    right = {event_key(row): row for row in right_rows}
    common_keys = sorted(set(left) & set(right))
    mismatches = []
    exact_events = 0
    for key in common_keys:
        difference = first_leaf_difference(normalize(left[key]), normalize(right[key]))
        if difference is None:
            exact_events += 1
            continue
        field, left_value, right_value = difference
        row = left[key]
        mismatches.append(
            {
                "key": list(key),
                "event": row["event"],
                "window": int(row["window"]),
                "micro_step": row.get("micro_step"),
                "module": row.get("module"),
                "field": field,
                "left": left_value,
                "right": right_value,
                "classification": classify(row["event"], field),
            }
        )
    missing_left = [list(key) for key in sorted(set(right) - set(left))]
    missing_right = [list(key) for key in sorted(set(left) - set(right))]
    first = mismatches[0] if mismatches else None
    return {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "left_events": len(left),
        "right_events": len(right),
        "common_events": len(common_keys),
        "exact_events": exact_events,
        "mismatch_count": len(mismatches),
        "first_mismatch": first,
        "first_mismatch_window": None if first is None else int(first["window"]),
        "missing_from_left": missing_left[:20],
        "missing_from_right": missing_right[:20],
        "mismatches": mismatches[:100],
        "exact_on_common_events": not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.left, args.right)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
