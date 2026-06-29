# 20260630-01 Flash-VQG 训练期 read_topk 渐进收缩实验计划

status: planned
ledger: not written

## 目标

本轮验证训练期 read candidate 数量是否应该从较大 `topk` 逐步收缩到 `topk=4`。

背景结论:

- `20260629-03` 显示 dense-read 4ep 能把 2080ti/3090 成对 final gap 收进 4pp, 说明 read top-k candidate flip 是重要不稳定放大器。
- `20260629-04` 显示评估期 `topk=4` 在已有 checkpoint 上整体最好, 但这不等价于训练期固定 `topk=4` 一定最好。

本轮只做 1 epoch screen, 不改 Flash-VQG 核心机制, 不加回 dropout, 不写 official MQAR ledger。

## 固定条件

- MQAR canonical cache: content hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- Canonical init: model state hash `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.
- `seed=123`, `data_seed=123`.
- `cb64-r16`, `fox_gd_residual_write_topk=4`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `max_epochs=1`, `validations_per_epoch=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- 机器: 2080ti x1 + 3090 x1.

## Variants

| variant | 训练期 read_topk 策略 | 目的 |
|---|---|---|
| `fixed-r2-baseline` | 固定 `2` | 当前默认口径对照 |
| `fixed-r4` | 固定 `4` | 检查 eval 最优 topk 是否可直接用于训练 |
| `sched16to4` | `16 -> 4`, `linear_int`, train forward step `0..448` | 低成本宽读 warmup |
| `sched64to4` | `64 -> 4`, `linear_int`, train forward step `0..448` | dense-like 早期稳定器, 后期收紧 |

`sched*to4` 是过渡性逐步减小, 不是一次性跳变。`step <= 0` 使用 initial, `0 < step < 448` 线性插值并四舍五入, `step >= 448` 固定为 4。

## 执行与监控

启动前硬门槛:

- 目标容器内 `nvidia-smi` 可用。
- 目标容器内 `torch.cuda.is_available()` 为 true。
- cache content hash 与 canonical hash match。
- init state hash 与 canonical hash match。
- 每个 variant 的 preflight 确认 `max_epochs=1`, no-dropout, cb64-r16, read_topk/schedule 配置符合本计划。

运行队列:

- 2080ti GPU0: `fixed-r2-baseline -> sched64to4`.
- 2080ti GPU1: `fixed-r4 -> sched16to4`.
- 3090 GPU0: `fixed-r2-baseline -> fixed-r4 -> sched16to4 -> sched64to4`.

训练 trace step:

```text
0,64,130,176,203,352,353,448,528,704
```

长任务进入稳定训练后, 每次显式 `sleep 20m` 再轮询日志和 GPU 状态。

## Artifact 和报告

Artifact:

```text
docs/artifacts/20260630-01-flash-vqg-train-read-topk-screen/
```

报告:

```text
docs/20260630-01-flash-vqg-train-read-topk-screen-report.md
```

至少包含:

- `cache-init-preflight-summary.csv`
- `run-summary.csv`
- `variant-summary.csv`
- `cross-machine-comparison.csv`
- `queue-summary.csv`
- `invalid-runs.csv`
- `source-manifest.csv`
- `metadata.json`
- `README.md`

## 判定

优先看 hard slice `valid/mqar_case/accuracy-1024x256`。

- 如果 `fixed-r4` 明显优于 `fixed-r2-baseline`, 下一步做 `fixed-r4` 4ep confirm。
- 如果 `sched16to4` 或 `sched64to4` 优于固定 topk, 下一步只确认最好的 schedule。
- 如果所有 variant 都不稳定, 不扩大 read_topk 网格, 回到 gate/logf 或 M_state/code-aware decay 机制分析。
- 4pp 是用户当前可接受跨机器误差线, 但 1ep screen 只用于筛选, 不是最终结论。
