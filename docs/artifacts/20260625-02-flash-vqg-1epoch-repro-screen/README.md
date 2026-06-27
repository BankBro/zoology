# 20260625-02 Flash-VQG 1-Epoch Repro Screen Artifact

本目录汇总 `20260625-02-flash-vqg-1epoch-repro-screen` 的 diagnostic / exploratory 结果。本轮不写 official ledger。

## 摘要

`1 epoch` screen 已完成 2080ti GPU0 和 3090 GPU0 上 `s123` / `s124` 的 `r1-r4` repeat。

主要结果:

- `2080ti s123` 稳定且高: mean `0.941750`, gap `0.015000`.
- `2080ti s124` 不稳定: mean `0.805250`, gap `0.155000`.
- `3090 s123` 不稳定: mean `0.477000`, gap `0.345000`.
- `3090 s124` 因 `r4` collapse 而不稳定: mean `0.688325`, gap `0.905700`.

关键审计发现:

- 本轮实际加载的 13 个 cache 文件在 2080ti 和 3090 上不一致。
- 文件级 sha256: `0/13` match.
- `torch.load` 后的 tensor 内容级 hash: `0/13` match.
- 因此, 当前跨机器差异被训练数据 cache mismatch 污染, 不能直接作为 machine/runtime 结论。

## 文件说明

核心结果表:

- `run-summary.csv`: 每条 run 的状态和 final validation metrics.
- `repeat-summary.csv`: 按 machine 和 seed 聚合的 `r1-r4` 结果.
- `invalid-runs.csv`: 保留失败尝试, 用于审计。

Cache 审计:

- `cache-hash-summary.csv`: 从日志解析出的 cache 名称在 2080ti 本地文件系统上的 hash.
- `cache-cross-machine-summary.csv`: 2080ti 与 3090 源机器文件级 sha256 对照.
- `cache-content-cross-machine-summary.csv`: 2080ti 与 3090 源机器内容级 tensor hash 对照.

Trace 与元数据:

- `early-window-metrics.csv`
- `step-window-summary.csv`
- `read-trace-summary.csv`
- `preflight-summary.csv`
- `machine-summary.csv`
- `source-manifest.csv`
- `metadata.json`

## 判读边界

本 artifact 可以支持的结论是: `1 epoch` 是有效的 diagnostic screen, 能快速暴露 high/low 分叉和 repeat instability。

本 artifact 不支持干净的跨机器结论, 因为 2080ti 与 3090 的训练数据 cache 内容不一致。

下一步应先统一 cache 内容, 确认 `cache-content-cross-machine-summary.csv` 全部为 `content_match=true`, 再重跑 `1 epoch` screen。不要直接进入 `4 epoch` confirm。
