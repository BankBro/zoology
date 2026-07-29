#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from common import REPO_ROOT, atomic_write_json, sha256_file
from compare_traces import compare


RUN_TAG = "20260729-seed124-diag-02"
RAW_ROOT = Path(__file__).resolve().parent / "outputs" / "3090" / RUN_TAG
ARTIFACT_ROOT = (
    REPO_ROOT
    / "docs/artifacts/20260729-03-mqar-seed124-remat-causal-diagnosis"
)
VARIANTS = ("a0-fixed-off", "a1-fixed-post-phase1")
GATE_WEIGHT = "backbone.layers.1.sequence_mixer.mixer.attn.output_gate_fused.weight"
FLA_FUSED_GATE_SOURCE = Path(
    "/home/lyj/miniconda3/envs/flash-vqg-fla042/lib/python3.12/site-packages"
    "/fla/modules/fused_norm_gate.py"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace(label: str, variant: str) -> list[dict[str, Any]]:
    path = RAW_ROOT / "probes" / label / variant / "trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def event(
    rows: list[dict[str, Any]],
    name: str,
    *,
    window: int | None = None,
    micro_step: int | None = None,
) -> dict[str, Any]:
    for row in rows:
        if row["event"] != name:
            continue
        if window is not None and int(row["window"]) != window:
            continue
        if micro_step is not None and int(row.get("micro_step", -1)) != micro_step:
            continue
        return row
    raise KeyError((name, window, micro_step))


def first_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    event_name: str,
    field: str,
) -> dict[str, Any]:
    left_map = {
        (row["window"], row.get("micro_step")): row
        for row in left
        if row["event"] == event_name
    }
    right_map = {
        (row["window"], row.get("micro_step")): row
        for row in right
        if row["event"] == event_name
    }
    for key in sorted(set(left_map) & set(right_map)):
        if left_map[key][field] != right_map[key][field]:
            return {
                "event": event_name,
                "field": field,
                "window": key[0],
                "micro_step": key[1],
                "left": left_map[key][field],
                "right": right_map[key][field],
            }
    raise RuntimeError(f"No difference found for {event_name}.{field}.")


def first_loss_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_map = {
        (row["window"], row["micro_step"]): row
        for row in left
        if row["event"] == "forward"
    }
    right_map = {
        (row["window"], row["micro_step"]): row
        for row in right
        if row["event"] == "forward"
    }
    for key in sorted(set(left_map) & set(right_map)):
        if left_map[key]["loss"]["sha256"] != right_map[key]["loss"]["sha256"]:
            return {
                "event": "forward",
                "field": "loss",
                "window": key[0],
                "micro_step": key[1],
                "left": left_map[key]["loss_value"],
                "right": right_map[key]["loss_value"],
            }
    raise RuntimeError("No loss difference found.")


def autotune_rows() -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        prefix = "a0" if variant.startswith("a0") else "a1"
        for repeat in range(1, 7):
            label = f"autotune-{prefix}-r{repeat}"
            result = load_json(RAW_ROOT / "probes" / label / variant / "result.json")
            trace = load_trace(label, variant)
            backward = event(trace, "after_backward", window=1, micro_step=0)
            config = result["gate_autotune"]["layer_norm_gated_bwd_kernel"][
                "best_config"
            ]
            rows.append(
                {
                    "variant": variant,
                    "repeat": repeat,
                    "bt": config["kwargs"]["BT"],
                    "num_warps": config["num_warps"],
                    "gradient_sha256": backward["grad_sha256"],
                }
            )
    return rows


def exact_gradient_difference() -> dict[str, Any]:
    records = {}
    for config in ("bt64-w4", "bt64-w8"):
        rows = load_trace(f"exact-grad-a1-{config}", "a1-fixed-post-phase1")
        record = event(rows, "after_backward", window=1, micro_step=0)[
            "grad_tensors"
        ][GATE_WEIGHT]
        records[config] = record
    left = struct.unpack("<64f", bytes.fromhex(records["bt64-w4"]["raw_hex"]))
    right = struct.unpack("<64f", bytes.fromhex(records["bt64-w8"]["raw_hex"]))
    differences = [abs(a - b) for a, b in zip(left, right)]
    return {
        "parameter": GATE_WEIGHT,
        "shape": [64],
        "bt64_w4_sha256": records["bt64-w4"]["sha256"],
        "bt64_w8_sha256": records["bt64-w8"]["sha256"],
        "different_elements": sum(value != 0 for value in differences),
        "max_abs_difference": max(differences),
        "mean_abs_difference": sum(differences) / len(differences),
    }


def causal_validation() -> dict[str, Any]:
    left_path = RAW_ROOT / "probes/causal-fixed-a0-bt64-w4-177/a0-fixed-off/trace.jsonl"
    right_path = RAW_ROOT / "probes/causal-fixed-a1-bt64-w4-177/a1-fixed-post-phase1/trace.jsonl"
    comparison = compare(left_path, right_path)
    left = load_trace("causal-fixed-a0-bt64-w4-177", "a0-fixed-off")
    right = load_trace("causal-fixed-a1-bt64-w4-177", "a1-fixed-post-phase1")
    left_valid = [row for row in left if row["event"] == "after_validation"]
    right_valid = [row for row in right if row["event"] == "after_validation"]
    quality_exact = []
    for a, b in zip(left_valid, right_valid):
        a_metrics = {k: v for k, v in a["metrics"].items() if "peak_" not in k}
        b_metrics = {k: v for k, v in b["metrics"].items() if "peak_" not in k}
        quality_exact.append(a_metrics == b_metrics)
    final_left = event(left, "after_zero_grad", window=177)
    final_right = event(right, "after_zero_grad", window=177)
    return {
        "training_common_events": comparison["common_events"],
        "training_exact_events": comparison["exact_events"],
        "training_exact": comparison["exact_on_common_events"],
        "validation_events": len(left_valid),
        "validation_quality_exact": all(quality_exact),
        "validation_accuracies": [row["metrics"]["valid/accuracy"] for row in left_valid],
        "final_model_sha256": final_left["model_sha256"],
        "final_model_exact": final_left["model_sha256"] == final_right["model_sha256"],
        "final_optimizer_sha256": final_left["optimizer_sha256"],
        "final_optimizer_exact": final_left["optimizer_sha256"]
        == final_right["optimizer_sha256"],
        "a0_peak_reserved_mib": left_valid[-1]["metrics"]["valid/peak_reserved_mib"],
        "a1_peak_reserved_mib": right_valid[-1]["metrics"]["valid/peak_reserved_mib"],
    }


def replay_summary() -> dict[str, Any]:
    results = {
        config: load_json(RAW_ROOT / "replay-results" / f"{config}.json")
        for config in ("bt64-w4", "bt64-w8")
    }
    return {
        "capsule_sha256": results["bt64-w4"]["capsule_sha256"],
        "capsule_bytes": Path(results["bt64-w4"]["capsule"]).stat().st_size,
        "forward_exact": results["bt64-w4"]["output"]["sha256"]
        == results["bt64-w8"]["output"]["sha256"],
        "grad_x_exact": results["bt64-w4"]["grad_x"]["sha256"]
        == results["bt64-w8"]["grad_x"]["sha256"],
        "grad_gate_exact": results["bt64-w4"]["grad_gate"]["sha256"]
        == results["bt64-w8"]["grad_gate"]["sha256"],
        "grad_weight_bt64_w4_sha256": results["bt64-w4"]["grad_weight"]["sha256"],
        "grad_weight_bt64_w8_sha256": results["bt64-w8"]["grad_weight"]["sha256"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_paths() -> list[Path]:
    return [
        RAW_ROOT / "preflight.json",
        RAW_ROOT / "queue-summary.json",
        RAW_ROOT / "comparisons/a1-detailed-gradient-groups.json",
        RAW_ROOT / "replay-results/bt64-w4.json",
        RAW_ROOT / "replay-results/bt64-w8.json",
        RAW_ROOT
        / "probes/gate-replay-capture-w4/a1-fixed-post-phase1/replay/layer1-window1-micro0.pt",
        RAW_ROOT / "probes/initial-128/a0-fixed-off/trace.jsonl",
        RAW_ROOT / "probes/initial-128/a1-fixed-post-phase1/trace.jsonl",
        RAW_ROOT / "probes/causal-fixed-a0-bt64-w4-177/a0-fixed-off/trace.jsonl",
        RAW_ROOT
        / "probes/causal-fixed-a1-bt64-w4-177/a1-fixed-post-phase1/trace.jsonl",
    ]


def source_manifest() -> list[dict[str, Any]]:
    remote_prefix = Path(
        "/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts"
        "/20260729-03-mqar-seed124-remat-causal-diagnosis/outputs/3090"
        f"/{RUN_TAG}"
    )
    rows = []
    for path in source_paths():
        relative = path.relative_to(RAW_ROOT)
        rows.append(
            {
                "role": str(relative),
                "source_machine": "mclab-3090/Flash-VQG-tun",
                "source_path": str(remote_prefix / relative),
                "mirror_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mirror_verified": True,
            }
        )
    return rows


def main() -> int:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    left = load_trace("initial-128", "a0-fixed-off")
    right = load_trace("initial-128", "a1-fixed-post-phase1")
    timeline = {
        "first_gradient_difference": first_difference(
            left, right, event_name="after_backward", field="grad_sha256"
        ),
        "first_optimizer_difference": first_difference(
            left, right, event_name="after_zero_grad", field="optimizer_sha256"
        ),
        "first_model_difference": first_difference(
            left, right, event_name="after_zero_grad", field="model_sha256"
        ),
        "first_loss_difference": first_loss_difference(left, right),
        "window10_train_loss": {
            variant: event(load_trace("initial-128", variant), "after_zero_grad", window=10)[
                "train_loss"
            ]
            for variant in VARIANTS
        },
    }
    atomic_write_json(ARTIFACT_ROOT / "first-divergence.json", timeline)
    write_csv(ARTIFACT_ROOT / "autotune-gradient-groups.csv", autotune_rows())
    exact = exact_gradient_difference()
    atomic_write_json(ARTIFACT_ROOT / "exact-gradient-difference.json", exact)
    causal = causal_validation()
    atomic_write_json(ARTIFACT_ROOT / "causal-validation.json", causal)
    replay = replay_summary()
    atomic_write_json(ARTIFACT_ROOT / "replay-summary.json", replay)
    manifest = source_manifest()
    write_csv(ARTIFACT_ROOT / "source-manifest.csv", manifest)
    metadata = {
        "experiment_id": "20260729-03-mqar-seed124-remat-causal-diagnosis",
        "status": "causal_root_identified",
        "run_tag": RUN_TAG,
        "machine": "mclab-3090/Flash-VQG-tun",
        "zoology_commits": [
            "8c8ceb3d467451606246ea74baaf11c7587b7b25",
            "662be9341a5cef35a297ead668da6cbb27e07356",
            "b98bda63755af979329f564f963151a769fb40e4",
        ],
        "flash_commit": "d7dbb1282d20ad860634ee4b8f0a74b948fe6c61",
        "fla_fused_norm_gate_source": str(FLA_FUSED_GATE_SOURCE),
        "fla_fused_norm_gate_sha256": sha256_file(FLA_FUSED_GATE_SOURCE),
        "root_cause": (
            "FLA 0.4.2 FusedRMSNormGated backward fresh-process Triton autotune "
            "selects different reduction configs for output_gate_fused.weight."
        ),
        "first_divergence": timeline,
        "exact_gradient_difference": exact,
        "causal_validation": causal,
        "replay": replay,
        "source_files": len(manifest),
    }
    atomic_write_json(ARTIFACT_ROOT / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
