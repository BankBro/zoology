# 20260630-04 Flash-VQG default-dropout fixed-r4 1ep screen 报告

status: completed_diagnostic
ledger: not written

## 目标

本轮先做 1 epoch screen, 不直接跑 4 epoch。核心问题是:

```text
在加回 default dropout 后, fixed-r4 是否仍保持 no-dropout 下观察到的跨机器稳定性?
```

同时利用 2080ti 的第二张 GPU 补一条 `fixed-r2-baseline`, 用来判断 default dropout 下 `read_topk=2` 是否仍是必要对照。这个 supplemental run 不是跨机器结论, 只用于下一步决策。

## 执行口径

共同配置:

- `seed=124`, `data_seed=123`.
- `cb64-r16`.
- `vq_weight_mode=dense_softmax`.
- `fox_gd_residual_write_topk=4`.
- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- `max_epochs=1`, 每 epoch `704` optimizer steps.
- 使用同一份 canonical MQAR cache.
- 使用同一份 seed124 canonical init checkpoint.

Variants:

| variant | train-time read_topk | machines | 用途 |
|---|---:|---|---|
| `fixed-r4` | 4 | 2080ti + 3090 | 主跨机器 screen |
| `fixed-r2-baseline` | 2 | 2080ti only | supplemental 同机 baseline |

注意:

```text
G/L coarse memory 使用 dense softmax 权重.
M_state residual GD 写入仍是 top-k write, write_topk=4.
本轮主要变量是把 embed_dropout 加回 0.1.
```

代码版本:

- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `bc391c0`.
- `fixed-r4` queues 在 zoology commit `3d36bd5` 后启动。
- 后续 commit `6b5a40d` 只补充 `fixed-r2-baseline` 队列支持和文档说明; 2080ti `fixed-r4` 的 result JSON 中 env snapshot 显示 `6b5a40d`, 是因为训练过程中仓库提交了 supplemental 代码, 不代表 fixed-r4 训练语义发生变化。

前置硬门槛:

- 2080ti 和 3090 容器内 `nvidia-smi` 与 `torch.cuda.is_available()` 均通过。
- MQAR cache content hash 为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`, 3 条 run 均 match。
- seed124 init model state hash 为 `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`, 3 条 run 均 match。
- `invalid-runs.csv` 为空。

本轮是 diagnostic / exploratory screen, 不写 official MQAR ledger。

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`。

| machine | variant | read_topk | final valid acc | final 1024x256 | best 1024x256 | queue duration |
|---|---|---:|---:|---:|---:|---:|
| 2080ti | `fixed-r4` | 4 | 0.781 | 0.284 | 0.284 | 45.0 min |
| 3090 | `fixed-r4` | 4 | 0.710 | 0.135 | 0.135 | 45.0 min |
| 2080ti | `fixed-r2-baseline` | 2 | 0.976 | 0.877 | 0.877 | 85.0 min |

`fixed-r2-baseline` 的 queue duration 包含 queue monitor 训练结束后的 `sleep 1200` 状态刷新滞后, 因此不能直接拿来和 `fixed-r4` 比较速度。该 run 的训练结果已经正常写出, GPU 已释放。

跨机器主对比:

| variant | 2080ti final | 3090 final | final gap | within 4pp | 结论 |
|---|---:|---:|---:|---|---|
| `fixed-r4` | 0.284 | 0.135 | 14.9pp | no | 未通过 |

同机 supplemental 对比:

| machine | fixed-r2 | fixed-r4 | delta |
|---|---:|---:|---:|
| 2080ti | 0.877 | 0.284 | +59.3pp |

## 判读

本轮最直接的结论是:

```text
default dropout 下, fixed-r4 不能直接进入 4 epoch confirm.
```

原因很简单: 主跨机器 `fixed-r4` 不但 gap 是 `14.9pp`, 明显超过 4pp 容忍线, 两边绝对分数也都很低。这个结果和 no-dropout 下的 `fixed-r4` 形成强烈反差:

| 实验 | dropout | seed | epoch | variant | 2080ti final | 3090 final | gap |
|---|---|---:|---:|---|---:|---:|---:|
| `20260630-02` | off | 124 | 1 | `fixed-r4` | 0.900 | 0.897 | 0.3pp |
| `20260630-03` | off | 124 | 4 | `fixed-r4` | 0.944 | 0.953 | 0.9pp |
| `20260630-04` | default | 124 | 1 | `fixed-r4` | 0.284 | 0.135 | 14.9pp |

所以问题不是 `fixed-r4` 在所有条件下都不行。更准确地说:

```text
fixed-r4 在 no-dropout/canonical cache/init 下是强候选,
但加回 embed_dropout=0.1 后, 当前 1 epoch 表现明显崩掉.
```

2080ti 上的 `fixed-r2-baseline` 很关键, 因为它说明 default dropout 本身不一定导致 1 epoch 学不起来。同一台 2080ti, 同一 cache/init/seed/dropout 口径下:

```text
read_topk=2: 1024x256 = 0.877
read_topk=4: 1024x256 = 0.284
```

这提示风险更像是:

```text
dropout 扰动 + read_topk=4 / residual read-write-state 耦合
```

而不是简单的:

```text
dropout 一开, Flash-VQG 就整体不工作.
```

但这里必须保守: `fixed-r2-baseline` 目前只有 2080ti 一条, 还不能证明 `fixed-r2` default dropout 跨机器稳定。它只能说明下一步应该先补 3090 的同口径 `fixed-r2` 1 epoch, 而不是继续把 `fixed-r4` 拉长到 4 epoch。

## 限制

本轮只覆盖:

- `seed=124`.
- `data_seed=123`.
- `cb64-r16`.
- `write_topk=4`.
- default dropout 中显式加回 `embed_dropout=0.1`.
- `fixed-r4` 两机各一条, `fixed-r2` 只有 2080ti 一条。

因此不能直接外推到所有 seed, 也不能断言 `fixed-r2` 是最终 default dropout 方案。

本轮也没有定位 `fixed-r4 + dropout` 低分的具体机制。当前合理怀疑包括:

- dropout 增大 hidden state / score 的早期扰动。
- `read_topk=4` 在 noisy early state 下读入更多低质量 candidate。
- residual `M_state` 的 top-k write 和 read support 与 dropout 扰动耦合后, 早期 memory 轨迹被带偏。

这些只是解释方向, 不是本轮直接证明的因果结论。

## 下一步

不要马上跑 `20260630-04 default-dropout fixed-r4 4ep confirm`。

建议最小下一步是:

```text
补跑 3090 fixed-r2-baseline 1 epoch.
```

具体口径保持:

- same canonical MQAR cache.
- same seed124 canonical init.
- `seed=124`, `data_seed=123`.
- `cb64-r16`.
- `write_topk=4`.
- `read_topk=2`.
- `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- `max_epochs=1`.

判定:

| 3090 fixed-r2 结果 | 说明 | 后续 |
|---|---|---|
| 接近 2080ti `0.877`, gap <= 4pp | default dropout 下 `fixed-r2` 可能稳定, `fixed-r4` 是当前 dropout 风险点 | 再考虑 `fixed-r2` 4ep confirm 或专门拆 `r4 + dropout` 机制 |
| 明显低于 2080ti, gap > 4pp | default dropout 下即使 r2 也有跨机器风险 | 回到 dropout/RNG 与 write/state 路径拆解 |
| 绝对分数也很低 | 2080ti fixed-r2 可能是单次好结果或机器/轨迹差异 | 需要补 repeat, 不推进 4ep |

如果 `fixed-r2` 跨机器 1ep 通过, 再决定是:

1. 做 default-dropout `fixed-r2` 4ep confirm.
2. 或专门做 `fixed-r4 + dropout` 的 short probe, 看 dropout 后 read support, write support, `M_state` norm/update 是否异常。

## 产物

Artifact:

```text
docs/artifacts/20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen/
```

核心文件:

- `run-summary.csv`: 3 条 completed run 的 final/best 指标和配置摘要。
- `variant-summary.csv`: `fixed-r4` 两机成对结果, 以及 `fixed-r2-baseline` 单机结果。
- `cross-machine-comparison.csv`: `fixed-r4` 的 1024x256 cross-machine gap。
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash evidence。
- `queue-summary.csv`: queue 状态。
- `invalid-runs.csv`: 本轮为空。
- `source-manifest.csv`: mirrored raw evidence 路径和 sha256。
- `metadata.json`: 收尾元数据。
