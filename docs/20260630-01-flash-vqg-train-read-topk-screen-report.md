# 20260630-01 Flash-VQG 训练期 read_topk 渐进收缩实验报告

status: completed
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

主指标是 `valid/mqar_case/accuracy-1024x256`。

| variant | strategy | 2080ti final | 3090 final | gap | within 4pp | 2080ti best | 3090 best | best gap |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `fixed-r2-baseline` | fixed 2 | 0.592 | 0.582 | 1.0pp | yes | 0.592 | 0.582 | 1.0pp |
| `fixed-r4` | fixed 4 | 0.928 | 0.923 | 0.5pp | yes | 0.928 | 0.923 | 0.5pp |
| `sched16to4` | 16 -> 4 | 0.923 | 0.895 | 2.8pp | yes | 0.923 | 0.895 | 2.8pp |
| `sched64to4` | 64 -> 4 | failed | 0.907 | n/a | n/a | failed | 0.907 | n/a |

`sched64to4` 在 2080ti 上失败, 记录为 `failed:1`. 错误是 CUDA OOM:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.88 GiB.
```

3090 上 `sched64to4` 可以跑完, 但它没有超过 `fixed-r4`, 且同一配置在 2080ti 不可用。因此本轮不能把 `sched64to4` 当成实用方案。

## 判读

本轮最直接的结论是: 训练期固定 `read_topk=4` 是当前最干净的候选配置。

相对默认 `read_topk=2`, 固定 `read_topk=4` 在两台机器上都显著提高 hard case:

| machine | fixed-r2 | fixed-r4 | delta |
|---|---:|---:|---:|
| 2080ti | 0.592 | 0.928 | +33.6pp |
| 3090 | 0.582 | 0.923 | +34.1pp |

跨机器稳定性也更好看。`fixed-r4` 的 2080ti/3090 final gap 是 0.5pp, 在 4pp 容忍线内; `sched16to4` gap 是 2.8pp, 也在容忍线内, 但绝对准确率低于固定 r4。

本轮没有看到“先大 topk, 后收缩到 4”比“直接固定 4”更好:

- `sched16to4`: 两台都跑完, final 为 0.923/0.895, 不如 `fixed-r4` 的 0.928/0.923。
- `sched64to4`: 3090 final 为 0.907, 不如 `fixed-r4`; 2080ti 直接 OOM。

这说明本轮最值得继续确认的不是更大的 read_topk warmup, 而是更简单的 `fixed-r4`。它既提升效果, 又没有引入 `sched64to4` 那种显存风险。

## 限制

本轮只是 1 epoch screen, 不是最终正式结果。它能回答“下一步优先确认哪个方向”, 不能单独证明 `fixed-r4` 在 4 epoch 或更多 seed 上一定最优。

另外, 本轮仍然是 no-dropout 条件。它延续前面为了定位跨机器不稳定而采用的诊断口径, 尚未回答“dropout 加回来后是否仍然稳定”。

## 下一步

建议下一步做 `fixed-r4` 4 epoch confirm:

- 机器: 2080ti x1 + 3090 x1。
- 配置: canonical cache/init, no-dropout, seed=123, cb64-r16, write_topk=4, train-time read_topk=4。
- 判定: 重点看 1024x256 final/best 是否保持高位, 以及两机 gap 是否仍在 4pp 内。

如果 4 epoch confirm 通过, 再考虑加回 dropout 做独立验证。不要现在继续扩大 `read_topk` schedule 网格; 本轮证据已经显示大 topk warmup 没有比固定 4 更有价值, 且 `64 -> 4` 有明显显存代价。

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
