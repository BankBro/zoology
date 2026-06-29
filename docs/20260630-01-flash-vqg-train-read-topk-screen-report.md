# 20260630-01 Flash-VQG 训练期 read_topk 渐进收缩实验报告

status: running
ledger: not written

## 目标

本轮验证训练期 read candidate 数量是否应该从较大 `topk` 逐步收缩到 `topk=4`。

这不是最终机制改造, 而是一个低成本 1 epoch screen:

```text
dense-read 证明去掉 read top-k candidate flip 有帮助;
eval topk sweep 显示 topk=4 在已有 ckpt 上最好;
本轮检查训练期固定 topk=4 或渐进收缩到 4 是否能继承这两个信号。
```

本轮是 diagnostic screen, 不写 official MQAR ledger。

## 执行口径

共同配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`.
- `fox_gd_residual_write_topk=4`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `max_epochs=1`, `validations_per_epoch=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- 使用同一份 canonical MQAR cache 和 canonical init checkpoint。

Variants:

| variant | 训练期 read_topk 策略 | 目的 |
|---|---|---|
| `fixed-r2-baseline` | 固定 `2` | 当前默认口径对照 |
| `fixed-r4` | 固定 `4` | 检查 eval 最优 topk 是否可直接用于训练 |
| `sched16to4` | `16 -> 4`, `linear_int`, train forward step `0..448` | 低成本宽读 warmup |
| `sched64to4` | `64 -> 4`, `linear_int`, train forward step `0..448` | dense-like 早期稳定器, 后期收紧 |

`sched*to4` 是过渡性逐步减小, 不是一次性跳变。

前置硬门槛:

- 目标容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 均需通过。
- MQAR cache content hash 需为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- init model state hash 需为 `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.
- preflight 需确认 no-dropout, cb64-r16, 以及每个 variant 的 read_topk/schedule 配置正确。

## 结果

待训练完成后从 artifact 回填。

主指标是 `valid/mqar_case/accuracy-1024x256`。

| variant | strategy | 2080ti final | 3090 final | gap | within 4pp | 2080ti best | 3090 best | best gap |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `fixed-r2-baseline` | fixed 2 | pending | pending | pending | pending | pending | pending | pending |
| `fixed-r4` | fixed 4 | pending | pending | pending | pending | pending | pending | pending |
| `sched16to4` | 16 -> 4 | pending | pending | pending | pending | pending | pending | pending |
| `sched64to4` | 64 -> 4 | pending | pending | pending | pending | pending | pending | pending |

## 初步判读

待结果完成后更新。

判定口径:

- 如果 `fixed-r4` 明显优于 `fixed-r2-baseline`, 下一步做 `fixed-r4` 4ep confirm。
- 如果 `sched16to4` 或 `sched64to4` 优于固定 topk, 下一步只确认最好的 schedule。
- 如果所有 variant 都不稳定, 不扩大 read_topk 网格, 回到 gate/logf 或 M_state/code-aware decay 机制分析。

## 产物

Artifact:

```text
docs/artifacts/20260630-01-flash-vqg-train-read-topk-screen/
```

核心文件:

- `run-summary.csv`: per-run final metrics.
- `variant-summary.csv`: 每个 variant 的 2080ti/3090 成对结果。
- `cross-machine-comparison.csv`: 每个 variant 的 cross-machine gap。
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight。
- `queue-summary.csv`: queue 状态。
- `invalid-runs.csv`: failed/interrupted/pending run。
- `source-manifest.csv`: raw evidence 路径和 sha256。
- `metadata.json`: 收尾元数据。
