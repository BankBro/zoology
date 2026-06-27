# 20260625-01 Flash-VQG Early-Window Trace Artifact

本目录存放 `20260625-01-flash-vqg-early-window-trace` 的轻量 summary, metadata 和 source manifest。本轮是 diagnostic / exploratory, 不写 official ledger。

## 1. 状态

- P0 smoke 已完成, 2080ti 和 3090 均通过 train-step read trace 基本检查.
- Wave 1 已完成, 5 条 run 均为 `completed`, `invalid-runs.csv` 为空.
- 3090 轻量 evidence 已镜像回 2080ti 主工作区相同相对路径, `source-manifest.csv` 记录 source path, mirror path, sha256, file size 和 mirror status.
- raw trace 体积较小, 保留在 `zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/outputs/traces/`.

## 2. 文件

- `metadata.json`: experiment_id, branch, commit, dtype policy, row counts, official ledger status.
- `stage3-run-summary.csv`: Wave 1 run 状态和最终 validation 指标.
- `stage3-key-metrics.csv`: Wave 1 train-step scalar metrics 与 read-side summary 合并表.
- `stage3-step-window-summary.csv`: Wave 1 early-window scalar metrics.
- `stage3-read-trace-summary.csv`: Wave 1 read trace 聚合 summary.
- `source-manifest.csv`: queue logs, run logs, trace JSONL, generated config/manifest 的审计清单.
- `invalid-runs.csv`: 失败, OOM 或中断 run 清单, 本轮为空.
- `machine-summary.csv`, `run-summary.csv`, `early-window-metrics.csv`, `early-window-step-summary.csv`, `read-trace-summary.csv`: 包含 smoke 与 Wave 1 的完整收集表.

## 3. 核心结果

Wave 1 最终 `valid/mqar_case/accuracy-1024x256`:

| machine | target | final 1024x256 accuracy | status |
|---|---:|---:|---|
| 2080ti | `default-s123` | 0.973 | high |
| 2080ti | `default-s124` | 0.813 | low |
| 3090 | `default-s123` | 0.804 | low |
| 3090 | `default-s124` | 0.962 | high |
| 3090 | `hard04-s123` | 0.858 | partial / low |

本轮没有复现稳定的 `s123 low, s124 high` seed 规律。结果更像 machine/GPU/concurrency/nondeterminism 混淆: 同一 seed 在不同机器上落入不同 basin, 同一机器上两个 seed 也出现反向结果。

## 4. 诊断信号

- 3090 `default-s123` 在 step 130 已出现 read-side 收敛倾向: `read_entropy_mean=1.151`, `read_selected_mass_mean=0.752`, `read_margin_top1_top2_mean=3.135`.
- 3090 `default-s123` 在 step 203 强锁定: `read_entropy_mean=0.176`, `read_selected_mass_mean=0.971`, `read_margin_top1_top2_mean=6.728`, `read_unique_top1_ids=2`, 同时 `update_norm_p95=0.474`, `m_norm_max=3.884`.
- 2080ti `default-s123` 也在 step 203 出现更强 read-side 锁定, 但最终是 high basin, 因此 read lock 本身不能单独解释 low basin.
- 3090 `hard04-s123` 明显压低 write/update pressure: step 203 `update_norm_p95=0.046`, `update_norm_max=0.203`, 低于 3090 `default-s123` 的 `0.474/1.065`, 但最终 1024x256 accuracy 仍只有 0.858.
- `m_norm` 与 `update_norm` 都有信号, 但当前结果受 machine/concurrency 混淆, 不足以提炼 guard 条件.

## 5. 结论

本轮成功完成 trace hook, smoke, Wave 1 运行, 轻量 evidence 镜像和 artifact 收集。但实验结论是负向的: 不能直接进入 guarded cap release。下一步应先做单机单 run deterministic / concurrency 排查, 确认 seed 与 basin 的关系是否可复现。
