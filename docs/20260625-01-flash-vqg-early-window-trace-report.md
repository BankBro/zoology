# 20260625-01 Flash-VQG Early-Window Trace Report

## 1. 状态

本轮实验 `20260625-01-flash-vqg-early-window-trace` 已完成。定位是 diagnostic / exploratory, 不写 official ledger。

- zoology: `flash-vqg@75eb000`.
- Flash-VQG: `20260428-gd-residual-v1-sync@eed5778`.
- dtype policy: launch config 未显式启用 AMP, bf16 或 fp16 override, 使用默认 torch/zoology runtime dtype; GD residual builder 为 `grouped_chunk_torch_ref`, pack mode 为 `semivec_ref`.
- P0 smoke: 2080ti 与 3090 均通过, `read_trace_train_steps` 能按 train step 写出 `read_trace.jsonl`.
- Wave 1: 5 条 run 均完成, `invalid-runs.csv` 为空.
- 3090 轻量 evidence 已镜像回 2080ti 主工作区, `source-manifest.csv` 记录 source path, mirror path, sha256 和 mirror status.

## 2. 产物

主要 artifact 位于 `docs/artifacts/20260625-01-flash-vqg-early-window-trace/`:

- `stage3-run-summary.csv`: 5 条 Wave 1 run 的最终结果.
- `stage3-key-metrics.csv`: 35 行, 5 条 run x 7 个 train-step windows.
- `stage3-step-window-summary.csv`: 35 行 scalar summary.
- `stage3-read-trace-summary.csv`: 35 行 read trace summary.
- `source-manifest.csv`: 112 条轻量 evidence 审计记录.
- `metadata.json`: branch, commit, dtype policy, official ledger status 和 row counts.

raw trace 保留在 `zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/outputs/traces/`, 每条 Wave 1 run 都覆盖 step `0,64,130,203,352,448,705`, 每个 step 64 条 trace record。

## 3. 最终结果

Wave 1 最终 `valid/mqar_case/accuracy-1024x256`:

| machine | target | final 1024x256 accuracy | final valid accuracy | final valid loss | basin |
|---|---|---:|---:|---:|---|
| 2080ti | `default-s123` | 0.973 | 0.995 | 0.066 | high |
| 2080ti | `default-s124` | 0.813 | 0.972 | 0.232 | low |
| 3090 | `default-s123` | 0.804 | 0.967 | 0.222 | low |
| 3090 | `default-s124` | 0.962 | 0.994 | 0.0712 | high |
| 3090 | `hard04-s123` | 0.858 | 0.974 | 0.182 | partial / low |

关键结论: 本轮没有复现稳定的 `s123 low, s124 high` seed 规律。`s123` 在 2080ti 是 high, 在 3090 是 low; `s124` 在 2080ti 是 low, 在 3090 是 high。因此 P2 cross-machine repeat 没有消除混淆, 反而证明当前不能把现象简单归因于 seed。

## 4. Early-Window 信号

3090 `default-s123` 是本轮最清晰的 low-basin 例子:

| step | loss | update_norm_p95 | update_norm_max | uncapped_sum_zeta_p95 | m_norm_max | read_entropy_mean | selected_mass_mean | read_margin_mean | unique_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8.677 | 0.113 | 0.356 | 0.268 | 1.030 | 3.685 | 0.164 | 0.362 | 29 |
| 130 | 8.454 | 0.084 | 0.487 | 0.251 | 0.804 | 1.151 | 0.752 | 3.135 | 2 |
| 203 | 8.404 | 0.474 | 1.065 | 0.644 | 3.884 | 0.176 | 0.971 | 6.728 | 2 |

这里最早可见的 read-side 分叉在 step 130: entropy 明显下降, selected mass 和 top1 margin 上升, unique top1 候选塌缩到 2 个。step 203 时 write/update pressure 与 state health 同时恶化, read-side 已强锁定。

但这个信号不能直接推广为 guard 条件。2080ti `default-s123` 在 step 203 也出现更强 read-side 锁定, 例如 `read_entropy_mean=0.0018`, `selected_mass_mean=0.987`, `read_margin_mean=10.098`, 但最终是 high basin。说明 “read 很早变自信” 不是充分条件。

## 5. hard04 对照

3090 `hard04-s123` 显著压低 write/update pressure:

| step | default update_p95/max | hard04 update_p95/max | default final acc | hard04 final acc |
|---:|---:|---:|---:|---:|
| 203 | 0.474 / 1.065 | 0.046 / 0.203 | 0.804 | 0.858 |
| 448 | 0.154 / 0.978 | 0.063 / 0.209 | 0.804 | 0.858 |
| 705 | 0.341 / 2.695 | 0.095 / 0.235 | 0.804 | 0.858 |

hard04 的确降低了 pressure, 也让 read-side 不那么早进入极端锁定。例子: step 203 时 hard04 `read_entropy_mean=1.622`, `selected_mass_mean=0.654`, 而 default 为 `0.176` 和 `0.971`。但最终 accuracy 只从 0.804 提到 0.858, 没有救回 high basin。因此 hard04 只能作为 pressure-control 证据, 不能作为最终机制。

## 6. 问题回答

1. `s123` 跨机器是否仍然低: 否。3090 为 0.804, 2080ti 为 0.973.
2. `s124` 跨机器是否仍然高: 否。3090 为 0.962, 2080ti 为 0.813.
3. low/high basin 最早在哪个 step 分叉: 当前不能给全局结论。对 3090 `default-s123`, read-side 在 step 130 已明显塌缩, step 203 pressure 和 state 指标同步恶化.
4. 分叉最早体现在什么指标: 对 3090 `default-s123`, read-side 先于最强 pressure spike 出现; 但 2080ti `default-s123` 反例说明 read-side 自信锁定不是充分条件.
5. `update_norm_p95/max` 是否比 `m_norm` 更早区分坏 seed: 未能证明。当前低/高结果被 machine/GPU/concurrency 混淆, 且多个 run 中 update_norm 与 m_norm 在关键 step 同步变化.
6. read trace 是否显示候选早期跳变或自信读错: 显示候选塌缩和高 selected mass, 尤其是 3090 `default-s123` step 130/203. 但是否“读错”还需要和 per-sample target/correctness 对齐, 当前 summary 只证明 read-side lock-in.
7. hard04 是否压低 early update pressure: 是, 在 step 203/448/705 都显著压低.
8. hard04 是否只是降低 pressure, 但没有完全救回 read-side lock-in: 是。hard04 减轻了 read-side 极端锁定, 但最终仍是 partial / low.
9. 是否已有足够证据进入 guarded cap release: 没有。应暂停 guard 方向, 先排查 reproducibility 与 machine/concurrency.

## 7. 下一步

下一轮建议先做最小 deterministic / concurrency 排查, 不做 guard:

- 同一机器单 GPU 单 run 串行跑 `default-s123` 和 `default-s124`, 避免同 GPU 多进程并发.
- 3090 与 2080ti 都跑相同顺序, 至少各自重复一次, 记录 GPU id, CUDA/cuDNN deterministic flags, torch version, driver, launch order.
- 保持 `read_trace_train_steps=0,64,130,203,352,448,705`, 但可以先只跑 default, 不跑 hard04.
- 若单 run 串行后 seed/basin 关系稳定, 再回到 pressure-aware guard 设计.
- 若仍随机器或 launch order 翻转, 先查 nondeterminism, dtype/runtime 差异, data loader order, GPU 并发和 kernel 实现差异.

当前结论是: trace infrastructure 有效, hard04 压力控制有效, 但 seed instability 的主因尚未定位。不要基于本轮结果写 cap_hit_ratio, m_norm 或 update_norm guard。
