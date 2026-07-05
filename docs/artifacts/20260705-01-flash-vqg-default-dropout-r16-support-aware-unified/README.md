# 20260705-01-flash-vqg-default-dropout-r16-support-aware-unified

统一收尾 default-dropout fixed-r16 复现, read support 邻域, P2 trace, 以及 read-confidence / softmargin 机制 screen.

## 结果口径

- 正式训练结果只使用 `formal-20260705T020000Z` 视图收集, 不混入 smoke runs.
- 3090 formal queue 完成 15/15, failed 0.
- 2080ti formal queue 完成 12/12, failed 0. `fixed-r24`, `fixed-r32`, `sched32to16-linear512` 在 smoke 阶段 OOM, 所以没有在 2080ti 上正式启动.
- `trace-*` variants 只训练 256 optimizer steps, 用于 early-window read/write/injection 诊断, 不用于判断完整 1ep 准确率.

## 核心文件

- `run-summary.csv`: per-run final metrics.
- `cross-machine-comparison.csv`: paired final hard gap.
- `variant-summary.csv`: per-variant best/final paired summary.
- `formal-early-window-summary.csv`: trace variants 的 step 0/16/64/128/256 关键标量.
- `formal-early-window-cross-machine.csv`: trace variants 的跨机器关键标量对比.
- `formal-early-window-step256-summary.csv`: report 使用的 step 256 摘要表.
- `cache-init-preflight-summary.csv`: canonical init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence manifest.
- `metadata.json`: collection metadata and variant config snapshot.

自动收集器生成的 `early-window-summary.csv`, `read-trace-summary.csv`, `read-trace-cross-machine.csv`, `read-trace-cross-machine-summary.csv`, `first-mismatch-summary.csv`, `hash-probe-comparison-summary.csv`, `preflight-effective-summary.csv` 在本轮为空占位. 本轮 trace 文件位于 queue-local `traces/` 下, 因此另外整理为 `formal-early-window-*` 三个 CSV.
