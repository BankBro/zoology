# 20260625-01 Flash-VQG Early-Window Trace Artifact

本目录存放 `20260625-01-flash-vqg-early-window-trace` 的轻量 summary, metadata 和 source manifest。本轮是 diagnostic / exploratory, 不写 official ledger。

预期文件:

- `metadata.json`
- `machine-summary.csv`
- `run-summary.csv`
- `early-window-metrics.csv`
- `early-window-step-summary.csv`
- `read-trace-summary.csv`
- `source-manifest.csv`
- `invalid-runs.csv`, 如有失败, OOM 或中断

raw trace 如体积小可压缩归档, 否则保留在 `zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/outputs/` 或 source machine 原位, 并在 `source-manifest.csv` 记录。
