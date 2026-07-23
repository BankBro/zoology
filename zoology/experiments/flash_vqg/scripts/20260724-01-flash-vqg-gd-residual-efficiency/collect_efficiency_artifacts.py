#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "20260724-01-flash-vqg-gd-residual-efficiency"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
FLASH_ROOT = REPO_ROOT.parent / "Flash-VQG"
OUTPUTS = SCRIPT_DIR / "outputs"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _metric(summary: dict[str, Any], name: str, field: str = "p50") -> float | None:
    for row in summary.get("summaries", []):
        if row.get("metric") == name:
            value = row.get(field)
            return None if value is None else float(value)
    return None


def _timing_row(path: Path, *, machine: str, stage: str, reused: bool = False) -> dict[str, Any]:
    summary = _read_json(path)
    model = summary["model"]
    return {
        "machine": machine,
        "stage": stage,
        "model": model["name"],
        "phase": summary["phase"],
        "repeat_id": summary.get("repeat_id"),
        "metrics_mode": summary.get("metrics_mode"),
        "warmup": summary.get("warmup"),
        "active": summary.get("active"),
        "wall_p50_ms": _metric(summary, "wall_ms", "p50"),
        "wall_p90_ms": _metric(summary, "wall_ms", "p90"),
        "cuda_p50_ms": _metric(summary, "cuda_total_ms", "p50"),
        "backbone_p50_ms": _metric(summary, "backbone", "p50"),
        "backward_p50_ms": _metric(summary, "backward", "p50"),
        "optimizer_step_p50_ms": _metric(summary, "optimizer_step", "p50"),
        "lm_head_p50_ms": _metric(summary, "lm_head", "p50"),
        "cross_entropy_p50_ms": _metric(summary, "cross_entropy", "p50"),
        "metrics_wall_p50_ms": _metric(summary, "metrics_wall_ms", "p50"),
        "loss_sync_wall_p50_ms": _metric(summary, "loss_sync_wall_ms", "p50"),
        "metric_count": _metric(summary, "metric_count", "p50"),
        "peak_allocated_bytes": summary["memory"]["peak_allocated_bytes"],
        "peak_reserved_bytes": summary["memory"]["peak_reserved_bytes"],
        "trainable_parameters": model["trainable_parameters"],
        "active_state_capacity": model.get("active_state_capacity"),
        "grouped_chunk_backend": summary.get("flash_implementation", {}).get(
            "grouped_chunk_backend"
        ),
        "selected_read_backend": summary.get("flash_implementation", {}).get(
            "selected_read_backend"
        ),
        "frozen_gdn_reused": reused,
        "source": str(path.relative_to(REPO_ROOT)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    result = {
        "machine": first["machine"],
        "stage": "final-3-repeat-median",
        "model": first["model"],
        "phase": first["phase"],
        "repeat_count": len(rows),
        "repeat_ids": ";".join(str(row["repeat_id"]) for row in rows),
        "sources": ";".join(row["source"] for row in rows),
    }
    for key in (
        "wall_p50_ms",
        "wall_p90_ms",
        "cuda_p50_ms",
        "backbone_p50_ms",
        "backward_p50_ms",
        "optimizer_step_p50_ms",
        "lm_head_p50_ms",
        "cross_entropy_p50_ms",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = statistics.median(values) if values else None
    return result


def _baseline_timing() -> list[dict[str, Any]]:
    rows = []
    for machine in ("2080ti", "3090"):
        for path in sorted((OUTPUTS / machine / "baseline").glob("*/summary.json")):
            rows.append(_timing_row(path, machine=machine, stage="reference-baseline"))
    return rows


def _final_timing() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for machine in ("2080ti", "3090"):
        final_dir = OUTPUTS / machine / "final-current"
        if not final_dir.exists():
            final_dir = OUTPUTS / machine / "final"
        for model in ("flash", "gdn"):
            for phase in ("eval", "train"):
                paths = sorted(final_dir.glob(f"{model}-{phase}-core-r*/summary.json"))
                if machine == "2080ti" and model == "gdn":
                    baseline_r1 = OUTPUTS / machine / "baseline" / f"gdn-{phase}-core-r1/summary.json"
                    if baseline_r1.exists() and baseline_r1 not in paths:
                        paths.insert(0, baseline_r1)
                rows = [
                    _timing_row(
                        path,
                        machine=machine,
                        stage="final",
                        reused=model == "gdn" and "baseline" in path.parts,
                    )
                    for path in paths
                ]
                run_rows.extend(rows)
                if len(rows) >= 3:
                    aggregates.append(_aggregate(rows[:3]))
    return run_rows, aggregates


def _ratio_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["machine"], row["model"], row["phase"]): row for row in aggregates}
    rows = []
    for machine in ("2080ti", "3090"):
        for phase in ("eval", "train"):
            flash = by_key.get((machine, "flash", phase))
            gdn = by_key.get((machine, "gdn", phase))
            if flash is None or gdn is None:
                rows.append(
                    {
                        "machine": machine,
                        "phase": phase,
                        "metric": "wall_p50_ms",
                        "flash": flash.get("wall_p50_ms") if flash else None,
                        "gdn": gdn.get("wall_p50_ms") if gdn else None,
                        "flash_over_gdn": None,
                        "threshold": 2.0,
                        "pass": False,
                        "status": "missing frozen GDN measurement",
                    }
                )
                continue
            ratio = float(flash["wall_p50_ms"]) / float(gdn["wall_p50_ms"])
            rows.append(
                {
                    "machine": machine,
                    "phase": phase,
                    "metric": "wall_p50_ms",
                    "flash": flash["wall_p50_ms"],
                    "gdn": gdn["wall_p50_ms"],
                    "flash_over_gdn": ratio,
                    "threshold": 2.0,
                    "pass": ratio <= 2.0,
                    "status": "measured",
                }
            )
    return rows


def _memory_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for machine in ("2080ti", "3090"):
        for path in sorted((OUTPUTS / machine / "final-memory").glob("*/summary.json")):
            summary = _read_json(path)
            rows.append(
                {
                    "machine": machine,
                    "model": summary["model"]["name"],
                    "phase": summary["phase"],
                    "peak_allocated_bytes": summary["memory"]["peak_allocated_bytes"],
                    "peak_allocated_gib": summary["memory"]["peak_allocated_bytes"] / 2**30,
                    "peak_reserved_bytes": summary["memory"]["peak_reserved_bytes"],
                    "peak_reserved_gib": summary["memory"]["peak_reserved_bytes"] / 2**30,
                    "snapshot": summary["memory"].get("snapshot"),
                    "source": str(path.relative_to(REPO_ROOT)),
                }
            )
    by_key = {(row["machine"], row["model"], row["phase"]): row for row in rows}
    ratios = []
    for machine in ("2080ti", "3090"):
        for phase in ("eval", "train"):
            flash = by_key.get((machine, "flash", phase))
            gdn = by_key.get((machine, "gdn", phase))
            ratio = (
                flash["peak_allocated_bytes"] / gdn["peak_allocated_bytes"]
                if flash and gdn
                else None
            )
            ratios.append(
                {
                    "machine": machine,
                    "phase": phase,
                    "metric": "max_memory_allocated",
                    "flash_bytes": flash["peak_allocated_bytes"] if flash else None,
                    "gdn_bytes": gdn["peak_allocated_bytes"] if gdn else None,
                    "flash_over_gdn": ratio,
                    "threshold": 2.0,
                    "pass": ratio is not None and ratio <= 2.0,
                    "status": "measured" if ratio is not None else "missing frozen GDN measurement",
                }
            )
    return rows, ratios


def _baseline_memory_rows(baseline_timing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "machine": row["machine"],
            "model": row["model"],
            "phase": row["phase"],
            "peak_allocated_bytes": row["peak_allocated_bytes"],
            "peak_allocated_gib": row["peak_allocated_bytes"] / 2**30,
            "peak_reserved_bytes": row["peak_reserved_bytes"],
            "peak_reserved_gib": row["peak_reserved_bytes"] / 2**30,
            "source": row["source"],
        }
        for row in baseline_timing
    ]


def _waterfall_rows() -> list[dict[str, Any]]:
    candidates = [
        ("eval", "reference", "baseline/flash-eval-core-r1", "accepted baseline"),
        ("eval", "fused grouped recurrence", "smoke/flash-eval-core-triton-grouped-v1", "accepted"),
        ("eval", "+ fused selected read", "smoke/flash-eval-core-triton-grouped-read-v1", "accepted"),
        ("eval", "+ core pack pruning", "smoke/flash-eval-core-pack-metrics-off-v1", "accepted"),
        ("eval", "+ Triton stable event order", "smoke/flash-eval-core-all-v1", "rejected: regression"),
        ("train", "reference", "baseline/flash-train-core-r1", "accepted baseline"),
        ("train", "fused grouped recurrence", "smoke/flash-train-core-triton-grouped-v1", "accepted"),
        ("train", "+ fused selected read", "smoke/flash-train-core-triton-grouped-read-v1", "accepted"),
        ("train", "+ deterministic selected backward", "smoke/flash-train-core-deterministic-v1", "accepted"),
    ]
    rows = []
    for phase, candidate, rel, decision in candidates:
        path = OUTPUTS / "2080ti" / rel / "summary.json"
        summary = _read_json(path)
        rows.append(
            {
                "machine": "2080ti",
                "phase": phase,
                "candidate": candidate,
                "decision": decision,
                "warmup": summary["warmup"],
                "active": summary["active"],
                "wall_p50_ms": _metric(summary, "wall_ms", "p50"),
                "peak_allocated_gib": summary["memory"]["peak_allocated_bytes"] / 2**30,
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )
    rows.extend(
        [
            {
                "machine": "2080ti",
                "phase": "train",
                "candidate": "fused LM projection + cross entropy",
                "decision": "not prioritized: measured LM head + CE is not the leading hotspot",
            },
            {
                "machine": "2080ti",
                "phase": "train/eval",
                "candidate": "gradient checkpointing or reduced hyperparameters",
                "decision": "rejected by semantic/performance constraints",
            },
        ]
    )
    return rows


def _tensor_lifetime_rows() -> list[dict[str, Any]]:
    specs = [
        ("M_state", [64, 2, 8, 64, 64, 16], "retained", "retained recurrent boundary state"),
        ("M_remote", [64, 2, 8, 64, 64, 16], "eliminated", "direct indexed reads from M_state"),
        ("M_sel", [64, 2, 8, 32, 16, 64, 16], "eliminated", "never written to global HBM"),
        ("C_sel", [64, 2, 8, 32, 16, 64], "eliminated", "loaded inside selected-read kernel"),
        ("z", [64, 2, 8, 32, 16, 16], "eliminated", "computed in kernel/rematerialized in backward"),
        ("d_read", [64, 2, 8, 32, 16, 16], "eliminated", "computed in kernel/rematerialized in backward"),
        ("proposal", [64, 2, 8, 32, 16, 64], "eliminated", "reduced before global write"),
        ("logits", [64, 256, 8192], "retained", "LM head was not the measured primary bottleneck"),
    ]
    rows = []
    for name, shape, final_status, note in specs:
        numel = 1
        for dim in shape:
            numel *= dim
        rows.append(
            {
                "tensor": name,
                "reference_shape": "x".join(map(str, shape)),
                "dtype": "float32",
                "reference_bytes": numel * 4,
                "reference_mib": numel * 4 / 2**20,
                "active_gd_layers": 1,
                "final_global_materialization": final_status,
                "note": note,
            }
        )
    return rows


def _metrics_rows() -> list[dict[str, Any]]:
    rows = []
    for machine in ("2080ti", "3090"):
        for phase in ("eval", "train"):
            for mode in ("core", "formal"):
                path = OUTPUTS / machine / "final" / f"flash-{phase}-{mode}-r1/summary.json"
                if not path.exists():
                    continue
                row = _timing_row(path, machine=machine, stage=f"final-{mode}")
                rows.append(row)
    for policy in ("legacy", "optimized"):
        path = OUTPUTS / "2080ti" / "p0" / f"flash-train-formal-{policy}-r1/summary.json"
        if path.exists():
            rows.append(_timing_row(path, machine="2080ti", stage=f"p0-{policy}"))
    return rows


def _equivalence_rows() -> list[dict[str, Any]]:
    rows = []
    for machine in ("2080ti", "3090"):
        path = OUTPUTS / machine / "equivalence/reference-vs-triton-deterministic-v2.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        comparisons = payload["comparisons"]
        rows.append(
            {
                "machine": machine,
                "passed": payload["passed"],
                "eval_hidden_max_abs": comparisons["eval_hidden"]["max_abs"],
                "eval_hidden_relative_l2": comparisons["eval_hidden"]["relative_l2"],
                "eval_loss_abs": comparisons["eval_loss_abs"],
                "train_hidden_max_abs": comparisons["train_hidden"]["max_abs"],
                "train_loss_max_abs": comparisons["train_losses"]["max_abs"],
                "gradient_max_abs": comparisons["gradients"]["max_abs"],
                "gradient_max_relative_l2": comparisons["gradients"]["max_relative_l2"],
                "parameter_step_max_abs": comparisons["parameters_after_step"]["max_abs"],
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest() -> list[dict[str, Any]]:
    paths = []
    for machine in ("2080ti", "3090"):
        for group in ("baseline", "final", "final-memory", "equivalence", "final-profile"):
            root = OUTPUTS / machine / group
            if not root.exists():
                continue
            paths.extend(root.rglob("summary.json"))
            paths.extend(root.rglob("*.csv"))
            paths.extend(root.rglob("*.json.gz"))
            paths.extend(root.rglob("memory-snapshot.pickle"))
        preflight = OUTPUTS / machine / "preflight.json"
        if preflight.exists():
            paths.append(preflight)
    rows = []
    for path in sorted(set(paths)):
        rows.append(
            {
                "machine": "3090" if "3090" in path.parts else "2080ti",
                "source_machine_path": str(path),
                "main_workspace_mirror": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "mirrored": True,
                "committed": False,
            }
        )
    return rows


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_timing()
    final_runs, aggregates = _final_timing()
    timing_ratios = _ratio_rows(aggregates)
    memory, memory_ratios = _memory_rows()
    _write_csv(ARTIFACT_DIR / "baseline-timing.csv", baseline)
    _write_csv(ARTIFACT_DIR / "baseline-memory.csv", _baseline_memory_rows(baseline))
    _write_csv(ARTIFACT_DIR / "final-timing.csv", [*final_runs, *aggregates])
    _write_csv(ARTIFACT_DIR / "performance-ratios.csv", timing_ratios)
    _write_csv(ARTIFACT_DIR / "final-memory.csv", memory)
    _write_csv(ARTIFACT_DIR / "memory-ratios.csv", memory_ratios)
    _write_csv(ARTIFACT_DIR / "candidate-waterfall.csv", _waterfall_rows())
    _write_csv(ARTIFACT_DIR / "tensor-lifetime.csv", _tensor_lifetime_rows())
    _write_csv(ARTIFACT_DIR / "metrics-on-off-comparison.csv", _metrics_rows())
    _write_csv(ARTIFACT_DIR / "equivalence-summary.csv", _equivalence_rows())
    _write_csv(ARTIFACT_DIR / "source-manifest.csv", _source_manifest())

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "zoology": {
            "branch": _git_value(REPO_ROOT, "branch", "--show-current"),
            "commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        },
        "flash_vqg": {
            "branch": _git_value(FLASH_ROOT, "branch", "--show-current"),
            "commit": _git_value(FLASH_ROOT, "rev-parse", "HEAD"),
        },
        "canonical_hashes": {
            "cache_content": "d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8",
            "flash_init": "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0",
            "batch_order": "fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320",
            "gdn_init": "bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6",
        },
        "active_gd_residual_layers": 1,
        "numerics": "FP32, PyTorch TF32 off, cuDNN TF32 off for benchmark runner, TRITON_F32_DEFAULT=ieee",
        "hard_timing_ratios": timing_ratios,
        "hard_memory_ratios": memory_ratios,
        "known_blockers": [
            "The frozen FP32 GDN chunk kernel does not compile on RTX 3090 sm86: required shared memory exceeds the 101376-byte hardware limit.",
        ],
    }
    _write_json(ARTIFACT_DIR / "metadata.json", metadata)
    (ARTIFACT_DIR / "README.md").write_text(
        f"# {EXPERIMENT_ID}\n\n"
        "Flash-VQG gd_residual_v1 baseline-r16-joint GPU memory and runtime audit. "
        "All hard timing rows use fixed canonical inputs, FP32/IEEE matmul policy, "
        "warmup >= 5, active >= 10, and fresh-process repeats. Formal quality runs "
        "and the 32/128/512-step trajectory are added after completion.\n\n"
        "The actual two-layer model contains one active Flash-VQG GD-residual layer "
        "and one BaseConv layer; tensor-lifetime estimates therefore use one GD layer.\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact_dir": str(ARTIFACT_DIR)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
