from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from zoology.analysis.utils import fetch_wandb_runs


DEFAULT_PROJECT = "flash_vqg_vs_gdn"
DEFAULT_ENTITY = "scu-mclab"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_LABELS = {
    "flash_vqg": "Flash-VQG",
    "gated_delta_net": "Gated DeltaNet",
}
MODEL_ORDER = ["Flash-VQG", "Gated DeltaNet"]
KV_COLUMNS = [
    "valid/num_kv_pairs/accuracy-4",
    "valid/num_kv_pairs/accuracy-8",
    "valid/num_kv_pairs/accuracy-16",
    "valid/num_kv_pairs/accuracy-32",
    "valid/num_kv_pairs/accuracy-64",
    "valid/num_kv_pairs/accuracy-128",
    "valid/num_kv_pairs/accuracy-256",
]
COMBO_PREFIX = "valid/mqar_case/accuracy-"


def resolve_project_path(project: str = DEFAULT_PROJECT, entity: str = DEFAULT_ENTITY) -> str:
    if "/" in project:
        return project
    return f"{entity}/{project}"


def default_output_dir(mode: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_dir = DEFAULT_RESULTS_DIR / f"{timestamp}-{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fetch_runs(
    *,
    project: str = DEFAULT_PROJECT,
    entity: str = DEFAULT_ENTITY,
    launch_ids: list[str] | None = None,
    sweep_ids: list[str] | None = None,
) -> pd.DataFrame:
    kwargs = {}
    if launch_ids:
        kwargs["launch_id"] = launch_ids
    if sweep_ids:
        kwargs["sweep_id"] = sweep_ids
    df = fetch_wandb_runs(project_name=resolve_project_path(project, entity), **kwargs)
    if df.empty:
        raise ValueError("没有查到任何 run. 请检查 launch_id / sweep_id / project / entity.")
    return df


def _ensure_output_dir(output_dir: str | Path | None, mode: str) -> Path:
    if output_dir is None:
        return default_output_dir(mode)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _format_learning_rate(summary: pd.DataFrame) -> pd.DataFrame:
    if "learning_rate" in summary.columns:
        summary["learning_rate"] = summary["learning_rate"].map(lambda x: f"{x:.2e}")
    return summary


def filter_d128_runs(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["model.d_model"] == 128]
    filtered = filtered[filtered["model.name"].isin(MODEL_LABELS.keys())]
    if filtered.empty:
        raise ValueError("过滤 d_model=128 和目标模型后没有剩余 run.")
    filtered["Model"] = filtered["model.name"].map(MODEL_LABELS)
    filtered["Model"] = pd.Categorical(filtered["Model"], categories=MODEL_ORDER, ordered=True)
    return filtered


def pick_best_runs_by_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid_df = df.dropna(subset=[metric]).copy()
    if valid_df.empty:
        raise ValueError(f"所有 run 的 {metric} 都为空.")
    idx = valid_df.groupby("model.name")[metric].idxmax(skipna=True).dropna()
    best_df = valid_df.loc[idx].copy()
    best_df["Model"] = best_df["model.name"].map(MODEL_LABELS)
    best_df["Model"] = pd.Categorical(best_df["Model"], categories=MODEL_ORDER, ordered=True)
    return best_df.sort_values("Model").reset_index(drop=True)


def make_d128_summary(best_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    columns = [
        "Model",
        "learning_rate",
        metric,
        "valid/loss",
        "state_size",
        "num_parameters",
        "run_id",
        "launch_id",
        "sweep_id",
    ]
    summary = best_df[[c for c in columns if c in best_df.columns]].copy()
    return _format_learning_rate(summary)


def plot_best_metric(best_df: pd.DataFrame, metric: str, output_path: Path):
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=best_df,
        x="Model",
        y=metric,
        hue="Model",
        order=MODEL_ORDER,
        hue_order=MODEL_ORDER,
        palette=["#4E79A7", "#9C755F"],
        legend=False,
    )
    plt.ylim(0, 1.0)
    plt.xlabel("")
    plt.ylabel(metric)
    plt.title("Best LR at d_model=128")
    for i, row in best_df.reset_index(drop=True).iterrows():
        lr_text = f"lr={row['learning_rate']:.1e}" if "learning_rate" in row else ""
        plt.text(i, row[metric] + 0.01, lr_text, ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_kv_slices(best_df: pd.DataFrame, output_path: Path) -> pd.DataFrame | None:
    available = [c for c in KV_COLUMNS if c in best_df.columns]
    if not available:
        return None
    rows = []
    for _, row in best_df.iterrows():
        for col in available:
            rows.append(
                {
                    "Model": row["Model"],
                    "num_kv_pairs": int(col.split("-")[-1]),
                    "accuracy": row[col],
                }
            )
    plot_df = pd.DataFrame(rows).dropna()
    if plot_df.empty:
        return None
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=plot_df,
        x="num_kv_pairs",
        y="accuracy",
        hue="Model",
        hue_order=MODEL_ORDER,
        marker="o",
        palette=["#4E79A7", "#9C755F"],
    )
    plt.xscale("log", base=2)
    plt.ylim(0, 1.0)
    plt.xlabel("num_kv_pairs")
    plt.ylabel("accuracy")
    plt.title("Best LR slice accuracy by num_kv_pairs")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_df


def run_d128_analysis(
    *,
    project: str = DEFAULT_PROJECT,
    entity: str = DEFAULT_ENTITY,
    launch_ids: list[str] | None = None,
    sweep_ids: list[str] | None = None,
    metric: str = "valid/accuracy",
    output_dir: str | Path | None = None,
    expected_runs: int = 8,
) -> dict:
    output_path = _ensure_output_dir(output_dir, mode="d128")
    df = fetch_runs(project=project, entity=entity, launch_ids=launch_ids, sweep_ids=sweep_ids)
    df = filter_d128_runs(df)
    if expected_runs and len(df) != expected_runs:
        print(f"[warn] 期望 {expected_runs} 个 run, 实际拿到 {len(df)} 个.")

    best_df = pick_best_runs_by_model(df, metric=metric)
    summary = make_d128_summary(best_df, metric=metric)

    full_csv = output_path / "full_runs.csv"
    best_csv = output_path / "best_runs.csv"
    plot_best_path = output_path / "best_valid_accuracy.png"
    plot_kv_path = output_path / "best_num_kv_slices.png"

    df.sort_values(["model.name", "learning_rate"]).to_csv(full_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    plot_best_metric(best_df, metric=metric, output_path=plot_best_path)
    kv_plot_df = plot_kv_slices(best_df, output_path=plot_kv_path)

    print("\n=== 最佳 lr 摘要 ===")
    print(summary.to_string(index=False))
    print(f"\n已保存: {full_csv}")
    print(f"已保存: {best_csv}")
    print(f"已保存: {plot_best_path}")
    if kv_plot_df is not None:
        print(f"已保存: {plot_kv_path}")

    return {
        "full_runs": df,
        "best_runs": best_df,
        "summary": summary,
        "output_dir": output_path,
    }


def filter_dmodel_runs(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["model.name"].isin(MODEL_LABELS.keys())]
    filtered = filtered[filtered["model.d_model"].isin([64, 128, 256])]
    if filtered.empty:
        raise ValueError("过滤目标模型和 d_model 后没有剩余 run.")
    filtered["Model"] = filtered["model.name"].map(MODEL_LABELS)
    filtered["Model"] = pd.Categorical(filtered["Model"], categories=MODEL_ORDER, ordered=True)
    return filtered


def pick_best_runs_by_model_and_dmodel(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid_df = df.dropna(subset=[metric]).copy()
    if valid_df.empty:
        raise ValueError(f"所有 run 的 {metric} 都为空.")
    idx = valid_df.groupby(["model.name", "model.d_model"])[metric].idxmax(skipna=True).dropna()
    best_df = valid_df.loc[idx].copy()
    best_df["Model"] = best_df["model.name"].map(MODEL_LABELS)
    best_df["Model"] = pd.Categorical(best_df["Model"], categories=MODEL_ORDER, ordered=True)
    return best_df.sort_values(["Model", "model.d_model"]).reset_index(drop=True)


def parse_case(case_name: str) -> tuple[int, int]:
    seq_len, num_kv_pairs = case_name.split("x")
    return int(seq_len), int(num_kv_pairs)


def combo_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(COMBO_PREFIX)]
    return sorted(cols, key=lambda c: parse_case(c.removeprefix(COMBO_PREFIX)))


def combos_to_long(best_df: pd.DataFrame) -> pd.DataFrame:
    cols = combo_columns(best_df)
    rows = []
    for _, row in best_df.iterrows():
        for col in cols:
            case_name = col.removeprefix(COMBO_PREFIX)
            seq_len, num_kv_pairs = parse_case(case_name)
            rows.append(
                {
                    "Model": row["Model"],
                    "model.d_model": row["model.d_model"],
                    "mqar_case": case_name,
                    "input_seq_len": seq_len,
                    "num_kv_pairs": num_kv_pairs,
                    "accuracy": row[col],
                }
            )
    combo_df = pd.DataFrame(rows).dropna(subset=["accuracy"])
    if combo_df.empty:
        raise ValueError("没有任何可用的 `valid/mqar_case/accuracy-*` 指标.")
    combo_df["Model"] = pd.Categorical(combo_df["Model"], categories=MODEL_ORDER, ordered=True)
    return combo_df.sort_values(["input_seq_len", "num_kv_pairs", "Model", "model.d_model"]).reset_index(drop=True)


def make_dmodel_summary(best_df: pd.DataFrame, selection_metric: str, target_case: str) -> pd.DataFrame:
    target_col = f"{COMBO_PREFIX}{target_case}"
    columns = [
        "Model",
        "model.d_model",
        "learning_rate",
        selection_metric,
        target_col,
        "valid/loss",
        "state_size",
        "num_parameters",
        "run_id",
        "launch_id",
        "sweep_id",
    ]
    summary = best_df[[c for c in columns if c in best_df.columns]].copy()
    summary = _format_learning_rate(summary)
    if target_col in summary.columns:
        summary = summary.rename(columns={target_col: f"accuracy@{target_case}"})
    return summary


def plot_target_case(combo_df: pd.DataFrame, target_case: str, output_path: Path):
    target_df = combo_df[combo_df["mqar_case"] == target_case].copy()
    if target_df.empty:
        raise ValueError(f"没有找到目标组合 `{target_case}` 的指标.")
    plt.figure(figsize=(6.5, 4))
    sns.lineplot(
        data=target_df,
        x="model.d_model",
        y="accuracy",
        hue="Model",
        hue_order=MODEL_ORDER,
        marker="o",
        palette=["#4E79A7", "#9C755F"],
    )
    plt.ylim(0, 1.0)
    plt.xlabel("d_model")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy @ {target_case}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_cases(combo_df: pd.DataFrame, output_path: Path):
    g = sns.relplot(
        data=combo_df,
        x="model.d_model",
        y="accuracy",
        hue="Model",
        hue_order=MODEL_ORDER,
        col="mqar_case",
        col_wrap=3,
        kind="line",
        marker="o",
        palette=["#4E79A7", "#9C755F"],
        facet_kws={"sharey": True, "sharex": True},
        height=3.2,
        aspect=1.1,
    )
    g.set_axis_labels("d_model", "accuracy")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.set_ylim(0, 1.0)
    g.figure.suptitle("Best-LR MQAR accuracy by test case", y=1.02)
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def run_dmodel_analysis(
    *,
    project: str = DEFAULT_PROJECT,
    entity: str = DEFAULT_ENTITY,
    launch_ids: list[str] | None = None,
    sweep_ids: list[str] | None = None,
    selection_metric: str = "valid/accuracy",
    target_case: str = "512x64",
    output_dir: str | Path | None = None,
    expected_runs: int = 24,
) -> dict:
    output_path = _ensure_output_dir(output_dir, mode="dmodel")
    df = fetch_runs(project=project, entity=entity, launch_ids=launch_ids, sweep_ids=sweep_ids)
    df = filter_dmodel_runs(df)
    if expected_runs and len(df) != expected_runs:
        print(f"[warn] 期望 {expected_runs} 个 run, 实际拿到 {len(df)} 个.")

    best_df = pick_best_runs_by_model_and_dmodel(df, metric=selection_metric)
    combo_df = combos_to_long(best_df)
    summary = make_dmodel_summary(best_df, selection_metric=selection_metric, target_case=target_case)

    full_csv = output_path / "full_runs.csv"
    best_csv = output_path / "best_runs.csv"
    combo_csv = output_path / "best_runs_combo_accuracy.csv"
    target_plot = output_path / f"accuracy_{target_case}_vs_dmodel.png"
    combo_plot = output_path / "combo_accuracy_vs_dmodel.png"

    df.sort_values(["model.name", "model.d_model", "learning_rate"]).to_csv(full_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    combo_df.to_csv(combo_csv, index=False)
    plot_target_case(combo_df, target_case=target_case, output_path=target_plot)
    plot_all_cases(combo_df, output_path=combo_plot)

    print("\n=== 最佳 lr 摘要 ===")
    print(summary.to_string(index=False))
    print(f"\n已保存: {full_csv}")
    print(f"已保存: {best_csv}")
    print(f"已保存: {combo_csv}")
    print(f"已保存: {target_plot}")
    print(f"已保存: {combo_plot}")

    return {
        "full_runs": df,
        "best_runs": best_df,
        "combo_runs": combo_df,
        "summary": summary,
        "output_dir": output_path,
    }
