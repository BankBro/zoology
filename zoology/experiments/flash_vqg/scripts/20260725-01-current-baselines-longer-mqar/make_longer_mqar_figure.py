#!/usr/bin/env python3
"""Generate the current Flash/GDN baseline Longer-MQAR accuracy curve."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/lyj/mnt/project/zoology")
DETAIL = (
    ROOT
    / "docs/artifacts/20260725-01-current-baselines-longer-mqar/longer-mqar-detail.csv"
)
OUT_DIR = ROOT / "docs/artifacts/20260725-01-current-baselines-longer-mqar/figures"

SLICES = ["1024x256", "2048x512", "4096x1024", "8190x512", "8190x2047"]
SEEDS = [123, 124, 125]
MODELS = {
    "flash": {
        "label": "Flash baseline-r16-joint",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2.8,
    },
    "gdn": {
        "label": "GDN h2-ek4-ev4",
        "color": "#0072B2",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 2.5,
    },
}


def aggregate() -> pd.DataFrame:
    detail = pd.read_csv(DETAIL)
    selected = detail[
        (detail["checkpoint_role"] == "last")
        & detail["model"].isin(MODELS)
        & detail["seed"].isin(SEEDS)
        & detail["slice"].isin(SLICES)
        & (detail["status"] == "completed")
    ].copy()

    expected = len(MODELS) * len(SEEDS) * len(SLICES)
    if len(selected) != expected:
        raise RuntimeError(f"Longer-MQAR主结果应为{expected}行, 实际{len(selected)}行.")
    counts = selected.groupby(["model", "seed", "slice"]).size()
    if len(counts) != expected or not (counts == 1).all():
        raise RuntimeError("model × seed × slice不是唯一完整矩阵.")

    rows = []
    for model, style in MODELS.items():
        for slc in SLICES:
            values = (
                selected[(selected["model"] == model) & (selected["slice"] == slc)]
                .sort_values("seed")["accuracy"]
                .astype(float)
                .to_numpy()
            )
            if len(values) != len(SEEDS) or not np.isfinite(values).all():
                raise RuntimeError(f"{model} {slc}缺少有限的三seed结果.")
            rows.append(
                {
                    "model": model,
                    "config": style["label"],
                    "checkpoint_role": "last",
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
                }
            )
    return pd.DataFrame(rows)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_plot(aggregate_data: pd.DataFrame) -> tuple[Path, Path, Path]:
    configure_style()
    x = np.arange(len(SLICES))
    fig, ax = plt.subplots(figsize=(7.3, 6.3), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.15)

    for model, style in MODELS.items():
        series = (
            aggregate_data[aggregate_data["model"] == model]
            .set_index("slice")
            .loc[SLICES]
            .reset_index()
        )
        ax.errorbar(
            x,
            series["mean"].to_numpy(dtype=float),
            yerr=series["population_std"].to_numpy(dtype=float),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=6.8,
            markeredgewidth=0.8,
            capsize=3.5,
            capthick=1.25,
            elinewidth=1.25,
            label=style["label"],
            zorder=3,
        )

    ax.axvspan(3.5, 4.5, color="#f2f2f2", zorder=-2)
    ax.text(
        4.0,
        0.965,
        "hardest\n8190x2047",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )
    ax.axvline(0.5, color="#888888", linewidth=0.9, linestyle=":", alpha=0.75, zorder=-1)
    ax.text(
        0.54,
        0.965,
        "length extrapolation →",
        ha="left",
        va="top",
        fontsize=8,
        color="#666666",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(SLICES)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.15, len(SLICES) - 0.85)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Longer-MQAR eval slice (input length x KV pairs)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.75)

    fig.suptitle("Current baselines: Longer-MQAR accuracy extrapolation", fontsize=14, y=0.982)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    legend._legend_box.align = "left"

    fig.text(
        0.12,
        0.026,
        "Mean ± population std over training seeds 123/124/125 (n=3); "
        "last.pt; 500 examples per slice; active capacity = 131k.",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#555555",
    )

    stem = OUT_DIR / "longer-mqar-accuracy-curve"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )
    return png_path, svg_path, pdf_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_data = aggregate()
    data_path = OUT_DIR / "longer-mqar-accuracy-curve-data.csv"
    aggregate_data.to_csv(data_path, index=False, lineterminator="\n")
    png_path, svg_path, pdf_path = make_plot(aggregate_data)
    print(data_path)
    print(png_path)
    print(svg_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
