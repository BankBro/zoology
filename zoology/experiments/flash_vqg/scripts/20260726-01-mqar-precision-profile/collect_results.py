#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LOCAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(LOCAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPT_DIR))

from common import (  # noqa: E402
    EXPERIMENT_ID,
    LONGER_SHAPES,
    REPO_ROOT,
    atomic_write_json,
    load_json,
    output_root,
    utc_now,
)
from coordinator import remote_read  # noqa: E402


ARTIFACT_DIR = REPO_ROOT / "docs" / "artifacts" / EXPERIMENT_ID
REPORT_PATH = REPO_ROOT / "docs" / f"{EXPERIMENT_ID}-report.md"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)


def load_machine_details() -> dict[str, list[dict[str, Any]]]:
    local_path = output_root("2080ti") / "formal-detail.json"
    local = load_json(local_path)
    remote = remote_read(Path("formal-detail.json"))
    if remote is None:
        raise RuntimeError("Remote formal detail is unavailable.")
    return {"2080ti": local, "3090": remote}


def flatten(details: dict[str, list[dict[str, Any]]]):
    training = []
    evaluation = []
    for machine, rows in details.items():
        for row in rows:
            result = row["training_result"]
            training.append(
                {
                    "machine": machine,
                    "descriptor_id": row["descriptor_id"],
                    "model": result["model"],
                    "seed": result["seed"],
                    "train_precision": result["train_precision"],
                    "status": result["status"],
                    "wall_clock_sec": result["wall_clock_sec"],
                    "started_at_utc": result["started_at_utc"],
                    "ended_at_utc": result["ended_at_utc"],
                    "gdn_kernel_dtype": result["gdn_kernel_dtype"],
                    "grad_scaler_skips": result["resume_audit"][
                        "grad_scaler_skips"
                    ],
                    "model_state_dtypes": ";".join(
                        result["resume_audit"]["model_state_dtypes"]
                    ),
                    "optimizer_state_dtypes": ";".join(
                        result["resume_audit"]["optimizer_state_dtypes"]
                    ),
                    "optimizer_step_wall_sec_p50": result["telemetry"].get(
                        "optimizer_step_wall_sec_p50"
                    ),
                    "optimizer_step_wall_sec_p90": result["telemetry"].get(
                        "optimizer_step_wall_sec_p90"
                    ),
                    "peak_allocated_mib": result["telemetry"].get(
                        "peak_allocated_mib"
                    ),
                    "peak_reserved_mib": result["telemetry"].get(
                        "peak_reserved_mib"
                    ),
                    "last_checkpoint_path": result["last_checkpoint"]["path"],
                    "last_checkpoint_sha256": result["last_checkpoint"][
                        "file_sha256"
                    ],
                    "best_checkpoint_path": result["best_checkpoint"]["path"],
                    "best_checkpoint_sha256": result["best_checkpoint"][
                        "file_sha256"
                    ],
                }
            )
            for event in row["evaluation"]:
                evaluation.append({"source_machine": machine, **event})
    return training, evaluation


def validate_counts(training: list[dict[str, Any]], evaluation: list[dict[str, Any]]):
    if len(training) != 30:
        raise RuntimeError(f"Expected 30 training rows, got {len(training)}.")
    expected = {"2080ti": 12 * 2 * 2 * 13, "3090": 18 * 2 * 3 * 13}
    observed = defaultdict(int)
    for row in evaluation:
        observed[row["source_machine"]] += 1
    if dict(observed) != expected:
        raise RuntimeError(
            f"Unexpected logical eval counts: expected={expected}, observed={dict(observed)}"
        )


def aggregate(evaluation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in evaluation:
        key = (
            row["source_machine"],
            row["model"],
            row["checkpoint_role"],
            row["train_precision"],
            row["eval_precision"],
            row["shape"],
            int(row["num_examples"]),
        )
        buckets[key].append(float(row["accuracy"]))
    rows = []
    for key, values in sorted(buckets.items()):
        (
            machine,
            model,
            role,
            train_precision,
            eval_precision,
            shape,
            num_examples,
        ) = key
        rows.append(
            {
                "machine": machine,
                "model": model,
                "checkpoint_role": role,
                "train_precision": train_precision,
                "eval_precision": eval_precision,
                "shape": shape,
                "num_examples": num_examples,
                "n_seeds": len(values),
                "accuracy_mean": statistics.mean(values),
                "accuracy_population_std": statistics.pstdev(values),
            }
        )
    return rows


def make_figure(summary: list[dict[str, Any]], role: str) -> None:
    import matplotlib.pyplot as plt

    longer_order = [f"{seq}x{kv}" for seq, kv in LONGER_SHAPES]
    colors = {"flash": "#0072B2", "gdn": "#D55E00"}
    line_styles = {"fp32": "-", "fp16": "--", "bf16": ":"}
    markers = {"fp32": "o", "fp16": "s", "bf16": "^"}
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for panel, (axis, machine) in enumerate(zip(axes, ("2080ti", "3090"))):
        rows = [
            row
            for row in summary
            if row["machine"] == machine
            and row["checkpoint_role"] == role
            and row["train_precision"] == row["eval_precision"]
            and int(row["num_examples"]) == 500
        ]
        groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            groups[(row["model"], row["train_precision"])][row["shape"]] = row
        for (model, precision), by_shape in sorted(groups.items()):
            means = [float(by_shape[shape]["accuracy_mean"]) for shape in longer_order]
            stds = [
                float(by_shape[shape]["accuracy_population_std"])
                for shape in longer_order
            ]
            x = list(range(len(longer_order)))
            axis.plot(
                x,
                means,
                color=colors[model],
                linestyle=line_styles[precision],
                marker=markers[precision],
                markersize=3.5,
                linewidth=1.4,
                label=f"{model.upper()} {precision.upper()}",
            )
            axis.fill_between(
                x,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=colors[model],
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(f"{chr(65 + panel)}  {machine}", loc="left", fontweight="bold")
        axis.set_xticks(range(len(longer_order)), longer_order, rotation=25, ha="right")
        axis.set_xlabel("Sequence length x key-value pairs")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Per-example MQAR accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(
        f"Matching train/eval precision, {role} checkpoint (mean ± population SD, n=3)",
        y=1.02,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    figure_dir = ARTIFACT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"matching-precision-{role}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def source_manifest(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in training:
        for role in ("last", "best"):
            rows.append(
                {
                    "machine": row["machine"],
                    "descriptor_id": row["descriptor_id"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "train_precision": row["train_precision"],
                    "checkpoint_role": role,
                    "source_path": row[f"{role}_checkpoint_path"],
                    "sha256": row[f"{role}_checkpoint_sha256"],
                    "mirror_status": "large_raw_retained_on_source_machine",
                }
            )
    return rows


def write_report(training: list[dict[str, Any]], evaluation: list[dict[str, Any]]):
    completed = sum(row["status"] == "completed" for row in training)
    report = f"""# MQAR 低精度与长度泛化实验报告

## 1. 结果概览

`{EXPERIMENT_ID}` 已完成 {completed}/30 个正式训练 run 和 {len(evaluation)} 个逻辑 checkpoint-eval 事件. 所有结果均通过双机 smoke, controlled resume, 全量 batch capacity, batch invariance 和 global commit/cache gate 后生成.

## 2. 实验口径

RTX 2080 Ti 比较 FP32 与 AMP-FP16, RTX 3090 比较 FP32, AMP-FP16 与 AMP-BF16. 每个模型和 dtype 使用 seeds `123,124,125`, 固定 B64, GA4 和 4 epochs. Flash-VQG 仅在 grouped update 与 selected-read Triton core 外建立 FP32 boundary; GDN 使用与实验 dtype 匹配的 FLA kernel dtype.

主结果使用 matching train/eval dtype. Off-diagonal 网格只用于机制分析. 两张 GPU 分别计算 3 seeds 的 mean 与 population std, 不合并为 `n=6`.

## 3. 产物

- Last 图: [matching-precision-last.pdf](artifacts/{EXPERIMENT_ID}/figures/matching-precision-last.pdf).
- Best 图: [matching-precision-best.pdf](artifacts/{EXPERIMENT_ID}/figures/matching-precision-best.pdf).
- 正式明细: [final.csv](artifacts/{EXPERIMENT_ID}/final.csv).
- 汇总: [precision-grid-summary.csv](artifacts/{EXPERIMENT_ID}/combined/precision-grid-summary.csv).
- Source manifest: [source-manifest.csv](artifacts/{EXPERIMENT_ID}/source-manifest.csv).

## 4. 结论

数值结论需结合生成的 matching dtype 图表与 seed-paired 明细复核后定稿. 本自动报告不把 off-diagonal 结果解释为 official 模型质量差异.
"""
    temporary = REPORT_PATH.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(REPORT_PATH)


def collect() -> dict[str, Any]:
    details = load_machine_details()
    training, evaluation = flatten(details)
    validate_counts(training, evaluation)
    summary = aggregate(evaluation)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ARTIFACT_DIR / "training.csv", training)
    write_csv(ARTIFACT_DIR / "final.csv", evaluation)
    write_csv(ARTIFACT_DIR / "source-manifest.csv", source_manifest(training))
    write_csv(
        ARTIFACT_DIR / "combined" / "precision-grid-summary.csv",
        summary,
    )
    for machine, rows in details.items():
        atomic_write_json(
            ARTIFACT_DIR / "machines" / machine / "formal-detail.json",
            rows,
        )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "training_rows": len(training),
        "logical_evaluation_rows": len(evaluation),
        "statistics": "mean and population standard deviation over three seeds per machine",
        "gpu_pooling": "disabled",
        "collected_at_utc": utc_now(),
    }
    atomic_write_json(ARTIFACT_DIR / "metadata.json", metadata)
    make_figure(summary, "last")
    make_figure(summary, "best")
    write_report(training, evaluation)
    return metadata


if __name__ == "__main__":
    print(json.dumps(collect(), ensure_ascii=False, indent=2))
