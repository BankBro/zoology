#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _smoke_ok(root: Path, machine: str, variant: str) -> bool:
    marker = root / "smoke" / machine / variant / "success.json"
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("success")) and payload.get("variant") == variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write-json", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    machines = ("2080ti", "3090")
    if all(_smoke_ok(root, machine, "fixed-r64") for machine in machines):
        selected = "fixed-r64"
        reason = "both_machines_fixed_r64_smoke_passed"
    elif all(_smoke_ok(root, machine, "fixed-r32") for machine in machines):
        selected = "fixed-r32"
        reason = "fixed_r64_smoke_failed_or_missing_and_both_machines_fixed_r32_smoke_passed"
    else:
        selected = "sched16to2-linear512"
        reason = "fixed_r64_and_fixed_r32_not_available_on_both_machines"

    payload = {
        "selected_rmax": selected,
        "reason": reason,
        "final_variants": [
            "baseline-r2",
            "baseline-r4",
            "fixed-r8",
            "fixed-r16",
            selected,
            "write-mass-r2",
            "write-mass-injwarm512-r2",
        ],
    }
    args.write_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
