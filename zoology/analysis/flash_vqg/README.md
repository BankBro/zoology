# Flash-VQG Analysis

这个目录现在只负责 Flash-VQG 的 SwanLab analysis.

## 语义

- analysis 是唯一入口
- 执行 analysis 时会先抓取数据, 再落盘, 再画图
- 不再单独区分 export
- 只支持 `swanlab`
- 支持两种 source:
  - `remote`
  - `local`

## 输入

- `--launch-id`
- `--source {remote,local}`

默认:

- `source=remote`

## run 发现来源

- analysis 只依赖 `generated/<launch_id>/manifest.json`
- `remote` 通过 manifest 里的 `experiment_id` 拉远端数据
- `local` 通过 manifest 里的 `run_dir` / `backup_file` 读本地数据
- 不依赖 `results/autocollect`

## 输出目录

默认输出根目录:

- `zoology/analysis/flash_vqg/results/`

目录结构:

```text
results/
  <launch_id>/
    <run_id>/
      data/
        summary.json
        history.csv
        metadata.json
        metrics_index.json
      pics/
        <metric>.png
        ...
    <run_id>/
      ...
    launch_analysis/
      <metric>.png
      ...
      run_summary.csv
      metrics_index.json
```

## 用法

远端分析:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis \
  --launch-id <launch_id> \
  --source remote
```

本地分析:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis \
  --launch-id <launch_id> \
  --source local
```

## 行为边界

- 数据源不可用时直接报错
- 不做 `remote -> local` 或 `local -> remote` 自动回退
- 单个 metric 缺失不报错, 仅跳过
- `launch_analysis/run_summary.csv` 会额外展开关键结构字段, 如 `block_len`, `local_num_blocks`, `if_remote_enabled`, `num_codebook_vectors`
- 当前只处理 scalar metric
