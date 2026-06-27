#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HASH_FIELDS = [
    "model_params_sha256",
    "grad_sha256",
    "optimizer_state_sha256",
    "inputs_sha256",
    "targets_sha256",
    "logits_sha256",
    "preds_sha256",
]


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_path"] = str(path)
    return payload


def _record_key(record: dict) -> tuple:
    return (
        record.get("stage"),
        record.get("optimizer_step"),
        record.get("micro_step"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payloads = [_load(Path(path)) for path in args.json_files]
    rows = []
    all_keys = sorted({
        _record_key(record)
        for payload in payloads
        for record in payload.get("records", [])
    })

    for key in all_keys:
        stage, optimizer_step, micro_step = key
        matching = []
        for payload in payloads:
            records = {
                _record_key(record): record
                for record in payload.get("records", [])
            }
            record = records.get(key, {})
            matching.append((payload, record))

        for field in HASH_FIELDS:
            values = [
                record.get(field)
                for _, record in matching
                if record.get(field) is not None
            ]
            if not values:
                continue
            rows.append({
                "stage": stage,
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "field": field,
                "all_match": len(set(values)) == 1,
                "distinct_count": len(set(values)),
                **{
                    payload["probe"]["probe_name"] if "probe_name" in payload.get("probe", {}) else payload["probe"]["name"]: record.get(field)
                    for payload, record in matching
                },
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
