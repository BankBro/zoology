#!/usr/bin/env python3
"""生成2080 Ti/3090的best与last Longer-MQAR正式曲线."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/lyj/mnt/project/zoology")
ARTIFACT_ROOT = ROOT / "docs/artifacts/20260725-01-current-baselines-longer-mqar"
DETAIL = ARTIFACT_ROOT / "combined/longer-mqar-detail.csv"
SUMMARY = ARTIFACT_ROOT / "combined/longer-mqar-summary.csv"
OUT_DIR = ARTIFACT_ROOT / "figures"

SLICES = ["1024x256", "2048x512", "4096x1024", "8190x512", "8190x2047"]
SEEDS = [123, 124, 125]
MACHINES = ["2080ti", "3090"]
MODELS = {
    "flash": {
        "label": "Flash baseline-r16-joint",
        "color": "#D55E00",
        "marker": "s",
    },
    "gdn": {
        "label": "GDN h2-ek4-ev4",
        "color": "#0072B2",
        "marker": "^",
    },
}
GPU_STYLES = {
    "2080ti": {
        "label": "2080 Ti",
        "linestyle": "-",
        "markerfacecolor": "auto",
        "offset": -0.035,
    },
    "3090": {
        "label": "3090",
        "linestyle": "--",
        "markerfacecolor": "white",
        "offset": 0.035,
    },
}


def aggregate(role: str) -> pd.DataFrame:
    detail = pd.read_csv(DETAIL)
    selected = detail[
        (detail["checkpoint_role"] == role)
        & detail["machine"].isin(MACHINES)
        & detail["model"].isin(MODELS)
        & detail["seed"].isin(SEEDS)
        & detail["slice"].isin(SLICES)
        & (detail["status"] == "completed")
    ].copy()

    expected = len(MACHINES) * len(MODELS) * len(SEEDS) * len(SLICES)
    if len(selected) != expected:
        raise RuntimeError(f"{role} Longer-MQAR应为{expected}行, 实际{len(selected)}行.")
    counts = selected.groupby(["machine", "model", "seed", "slice"]).size()
    if len(counts) != expected or not (counts == 1).all():
        raise RuntimeError("machine × model × seed × slice不是唯一完整矩阵.")

    rows = []
    for machine in MACHINES:
        for model, style in MODELS.items():
            for slc in SLICES:
                values = (
                    selected[
                        (selected["machine"] == machine)
                        & (selected["model"] == model)
                        & (selected["slice"] == slc)
                    ]
                    .sort_values("seed")["accuracy"]
                    .astype(float)
                    .to_numpy()
                )
                if len(values) != len(SEEDS) or not np.isfinite(values).all():
                    raise RuntimeError(f"{machine} {model} {slc}缺少有限的三seed结果.")
                rows.append({
                    "machine": machine,
                    "gpu_label": GPU_STYLES[machine]["label"],
                    "model": model,
                    "config": style["label"],
                    "checkpoint_role": role,
                    "slice": slc,
                    "mean": float(np.mean(values)),
                    "population_std": float(np.std(values, ddof=0)),
                    "n": len(values),
                    "seeds": ",".join(str(seed) for seed in SEEDS),
                    "accuracy_seed123": float(values[0]),
                    "accuracy_seed124": float(values[1]),
                    "accuracy_seed125": float(values[2]),
                    "active_state_capacity": 131_072,
                    "num_examples": 500,
                    "dtype_policy": "float32_ieee_tf32_off",
                })
    aggregate_data = pd.DataFrame(rows)
    verify_against_summary(aggregate_data, role)
    return aggregate_data


def verify_against_summary(aggregate_data: pd.DataFrame, role: str) -> None:
    summary = pd.read_csv(SUMMARY)
    summary = summary[summary["checkpoint_role"] == role].copy()
    expected = len(MACHINES) * len(MODELS) * len(SLICES)
    if len(summary) != expected:
        raise RuntimeError(f"combined summary中{role}应为{expected}行, 实际{len(summary)}行.")
    merged = aggregate_data.merge(
        summary,
        on=["machine", "model", "checkpoint_role", "slice"],
        suffixes=("_figure", "_summary"),
        validate="one_to_one",
    )
    mean_delta = np.abs(merged["mean"] - merged["accuracy_mean"])
    std_delta = np.abs(merged["population_std"] - merged["accuracy_population_std"])
    if float(mean_delta.max()) > 1e-12 or float(std_delta.max()) > 1e-12:
        raise RuntimeError(
            f"绘图统计与combined summary不一致: mean={mean_delta.max()}, std={std_delta.max()}"
        )


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 13.5,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def make_plot(aggregate_data: pd.DataFrame, role: str) -> tuple[Path, Path, Path]:
    configure_style()
    x = np.arange(len(SLICES), dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.17)

    for model, model_style in MODELS.items():
        for machine in MACHINES:
            gpu_style = GPU_STYLES[machine]
            series = (
                aggregate_data[
                    (aggregate_data["model"] == model)
                    & (aggregate_data["machine"] == machine)
                ]
                .set_index("slice")
                .loc[SLICES]
                .reset_index()
            )
            facecolor = (
                model_style["color"]
                if gpu_style["markerfacecolor"] == "auto"
                else gpu_style["markerfacecolor"]
            )
            ax.errorbar(
                x + gpu_style["offset"],
                series["mean"].to_numpy(dtype=float),
                yerr=series["population_std"].to_numpy(dtype=float),
                color=model_style["color"],
                marker=model_style["marker"],
                markerfacecolor=facecolor,
                markeredgecolor=model_style["color"],
                linestyle=gpu_style["linestyle"],
                linewidth=2.25,
                markersize=6.3,
                markeredgewidth=1.1,
                capsize=3.0,
                capthick=1.1,
                elinewidth=1.15,
                label=f"{model_style['label']} · {gpu_style['label']}",
                zorder=3,
            )

    ax.axvspan(3.5, 4.5, color="#f0f0f0", zorder=-2)
    ax.axvline(0.5, color="#777777", linewidth=0.9, linestyle=":", alpha=0.75, zorder=-1)
    ax.text(0.54, 0.965, "length extrapolation →", ha="left", va="top", fontsize=7.8, color="#555555")
    ax.text(4.0, 0.965, "hardest\n8190x2047", ha="center", va="top", fontsize=7.7, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(SLICES)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.18, len(SLICES) - 0.82)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Longer-MQAR eval slice (input length × KV pairs)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.75, alpha=0.75)

    fig.suptitle(
        f"Current baselines across GPUs: Longer-MQAR accuracy ({role}.pt)",
        fontsize=13.5,
        y=0.975,
    )
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.3,
        borderaxespad=0.0,
    )
    legend._legend_box.align = "left"

    fig.text(
        0.105,
        0.027,
        "Mean ± population SD over training seeds 123/124/125 (n=3 per GPU); "
        "500 examples per slice; FP32, TF32 off. Training and eval both ran on the labeled GPU.",
        ha="left",
        va="bottom",
        fontsize=7.35,
        color="#4d4d4d",
    )

    stem = OUT_DIR / f"longer-mqar-accuracy-{role}"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    return png_path, svg_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成2080 Ti/3090的Longer-MQAR正式曲线.")
    parser.add_argument("--role", choices=("last", "best", "all"), default="all")
    args = parser.parse_args()
    roles = ("last", "best") if args.role == "all" else (args.role,)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for role in roles:
        aggregate_data = aggregate(role)
        data_path = OUT_DIR / f"longer-mqar-accuracy-{role}-data.csv"
        aggregate_data.to_csv(data_path, index=False, lineterminator="\n")
        outputs = make_plot(aggregate_data, role)
        print(data_path)
        for path in outputs:
            print(path)


if __name__ == "__main__":
    main()
