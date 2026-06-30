# 20260630-02 Flash-VQG s124 readk4 gate 实验计划

status: planned
ledger: not written

## 目标

本轮低成本复查历史 `cb64-r16 s124 fixed readk4` 反例在当前严格控制口径下是否仍然成立。

背景:

- `20260622-03` 中 `cb64-r16 fixed readk4` 的历史 `s124` 两条结果为 `0.831/0.849`, 而 readk2 replacement `s124` 为 `0.959`, 因此 fixed readk4 不能作为全局默认。
- `20260630-01` 中 `cb64-r16 s123`, no-dropout, canonical cache/init, 1 epoch 下, `fixed-r4` 为 `0.928/0.923`, 明显优于 `fixed-r2` 的 `0.592/0.582`。

本轮只回答一个问题:

```text
如果把当前 canonical/no-dropout/cross-machine 口径切到 seed=124,
fixed-r4 是继续高分, 还是复现历史 s124 风险?
```

本轮是 diagnostic gate, 不跑 4 epoch, 不写 official MQAR ledger。

## 固定条件

- MQAR canonical cache: content hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- `seed=124`, `data_seed=123`.
- `cb64-r16`, `fox_gd_residual_write_topk=4`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `max_epochs=1`, `validations_per_epoch=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- 机器: 2080ti x1 + 3090 x1。

本轮必须生成新的 `seed=124` canonical init checkpoint:

- 在 2080ti 容器内生成 `cb64r16-s124-init.pt`.
- 复制到 3090 容器相同实验 outputs 路径.
- 两边用 state_dict tensor hash 验证完全一致。

不得复用 `seed=123` canonical init 来代表 `s124`。

## Variants

| variant | 训练期 read_topk 策略 | 目的 |
|---|---|---|
| `fixed-r2-baseline` | 固定 `2` | 同 seed 同口径 baseline, 判断 s124 本身是否低 |
| `fixed-r4` | 固定 `4` | 主实验, 复查历史 s124 readk4 风险是否仍存在 |

共 4 条 1 epoch run:

| machine | fixed-r2-baseline | fixed-r4 |
|---|---:|---:|
| 2080ti | 1ep | 1ep |
| 3090 | 1ep | 1ep |

## 执行与监控

启动前硬门槛:

- 目标容器内 `nvidia-smi` 可用。
- 目标容器内 `torch.cuda.is_available()` 为 true。
- cache content hash 与 canonical hash match。
- `seed=124` init state hash 两机 match。
- 每个 variant 的 preflight 确认 `seed=124`, `data_seed=123`, `max_epochs=1`, no-dropout, cb64-r16, read_topk 配置符合本计划。

运行队列:

- 2080ti: `fixed-r2-baseline`, `fixed-r4`.
- 3090: `fixed-r2-baseline`, `fixed-r4`.

训练 trace step:

```text
0,64,130,176,203,352,353,448,528,704
```

长任务进入稳定训练后, 每次显式 `sleep 20m` 再轮询日志和 GPU 状态。

## Artifact 和报告

Artifact:

```text
docs/artifacts/20260630-02-flash-vqg-s124-readk4-gate/
```

报告:

```text
docs/20260630-02-flash-vqg-s124-readk4-gate-report.md
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

主指标是 `valid/mqar_case/accuracy-1024x256`。

- 如果 `fixed-r4` 两机都高, 且 2080ti/3090 final gap <= 4pp, 且不低于同机 `fixed-r2`, 则通过 gate, 下一步进入 `seed=123 fixed-r4 4ep confirm`。
- 如果 `fixed-r4` 明显低于同机 `fixed-r2`, 则 fixed-r4 不能作为当前主线。
- 如果 `fixed-r4` 一边高一边低, 则跨机器稳定性仍未解决。
- 如果 `fixed-r2` 和 `fixed-r4` 都低, 则标记为 `s124` 当前口径本身不稳, 后续不直接归因于 read_topk。

本轮不能单独证明 `fixed-r4` 是最终训练配置, 也不能回答 dropout 加回后的稳定性。
