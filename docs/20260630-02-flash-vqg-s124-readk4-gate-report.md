# 20260630-02 Flash-VQG s124 readk4 gate 报告

status: completed_diagnostic
ledger: not written

## 目标

本轮复查历史 `cb64-r16 s124 fixed readk4` 反例在当前严格控制口径下是否仍然成立。

背景对照:

- `20260622-03` 中 `cb64-r16 fixed readk4` 的历史 `s124` 两条结果为 `0.831/0.849`, 而 readk2 replacement `s124` 为 `0.959`, 因此 fixed readk4 不能作为全局默认。
- `20260630-01` 中 `cb64-r16 s123`, no-dropout, canonical cache/init, 1 epoch 下, `fixed-r4` 为 `0.928/0.923`, 明显优于 `fixed-r2` 的 `0.592/0.582`。

本轮只做 `s124` 1 epoch diagnostic gate, 不写 official MQAR ledger。

## 执行口径

代码版本:

- zoology: `flash-vqg`, commit `df6cd6a`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `bc391c0`.

共同配置:

- `seed=124`, `data_seed=123`.
- `cb64-r16`.
- `fox_gd_residual_write_topk=4`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `max_epochs=1`, `validations_per_epoch=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- 使用同一份 canonical MQAR cache.
- 使用本轮新生成的 `seed=124` canonical init checkpoint.

前置硬门槛:

- 2080ti 和 3090 容器内 `nvidia-smi` 与 `torch.cuda.is_available()` 均通过。
- MQAR cache content hash 为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- `seed=124` init model state hash 为 `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- 两机 init state hash 完全一致。
- preflight 确认 `seed=124`, `data_seed=123`, `max_epochs=1`, no-dropout, cb64-r16, `write_topk=4`, read_topk 配置正确。

Variants:

| variant | 训练期 read_topk | 目的 |
|---|---:|---|
| `fixed-r2-baseline` | 2 | 同 seed baseline, 判断 `s124` 本身是否低 |
| `fixed-r4` | 4 | 主实验, 复查历史 `s124 readk4` 风险 |

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`。

| variant | 2080ti final | 3090 final | gap | within 4pp | 2080ti valid acc | 3090 valid acc |
|---|---:|---:|---:|---|---:|---:|
| `fixed-r2-baseline` | 0.775 | 0.840 | 6.5pp | no | 0.958 | 0.969 |
| `fixed-r4` | 0.900 | 0.897 | 0.3pp | yes | 0.981 | 0.980 |

同机对比:

| machine | fixed-r2 | fixed-r4 | delta |
|---|---:|---:|---:|
| 2080ti | 0.775 | 0.900 | +12.5pp |
| 3090 | 0.840 | 0.897 | +5.7pp |

耗时:

| machine | variant | duration |
|---|---|---:|
| 2080ti | `fixed-r2-baseline` | 76.8 min |
| 2080ti | `fixed-r4` | 54.8 min |
| 3090 | `fixed-r2-baseline` | 51.4 min |
| 3090 | `fixed-r4` | 41.7 min |

本轮 4 条 run 全部完成, 无 failed/interrupted run。

## 判读

本轮没有复现历史 `s124 fixed readk4` 风险。

在当前严格口径下:

```text
same canonical MQAR cache
same seed=124 canonical init
no-dropout
cb64-r16
1 epoch
2080ti + 3090
```

`fixed-r4` 两机 final 分别为 `0.900/0.897`, gap 只有 `0.3pp`, 明显在用户可接受的 4pp 内。同时 `fixed-r4` 在两台机器上都高于同机 `fixed-r2`:

- 2080ti: `+12.5pp`.
- 3090: `+5.7pp`.

更重要的是, `fixed-r2` 本身跨机器 gap 为 `6.5pp`, 没有进入 4pp, 而 `fixed-r4` gap 只有 `0.3pp`。因此 `metadata.summary.all_1024x256_within_4pp=false` 不能解读为本轮失败; 它是因为 baseline `fixed-r2` 不稳定, 不是因为主实验 `fixed-r4` 不稳定。

这说明旧的 `cb64-r16 s124 readk4` 低分反例至少不是在当前 canonical cache/init/no-dropout 口径下必然复现。更准确的定位是:

```text
fixed-r4 仍不能直接升级为全局默认,
但 s124 不再构成阻止当前 fixed-r4 主线继续确认的硬反例。
```

## 限制

本轮只是 1 epoch diagnostic gate, 不能替代 4 epoch confirm。

本轮仍然是 no-dropout 口径, 不能回答 dropout 加回后是否稳定。

本轮只覆盖 `seed=124`, `data_seed=123`, `cb64-r16`, 不能外推到所有 seed, capacity 或 longer-MQAR。

历史反例与本轮不同的因素包括历史代码路径, dropout/RNG 口径, cache/init 是否锁定, 以及是否 strict same-wave cross-machine。当前结果只能说明在当前更严格口径下, 旧 `s124` 风险没有复现。

## 下一步

建议进入 `fixed-r4` 4 epoch confirm:

- `seed=123`, `data_seed=123`.
- canonical MQAR cache.
- canonical init.
- no-dropout.
- `cb64-r16`.
- `fox_gd_residual_write_topk=4`.
- train-time `fox_remote_read_topk=4`.
- 2080ti x1 + 3090 x1.

判定仍看 `valid/mqar_case/accuracy-1024x256`:

- final/best 是否保持高位。
- 2080ti/3090 final gap 是否 <= 4pp。
- best-final drop 是否可控。

如果 4 epoch confirm 通过, 再单独设计 dropout 加回实验。不要现在继续扩大 `sched*to4` 网格, 也不要把 `fixed-r4` 直接写成最终默认。

## 产物

Artifact:

```text
docs/artifacts/20260630-02-flash-vqg-s124-readk4-gate/
```

核心文件:

- `run-summary.csv`: 4 条有效 run 的 final/best 指标和配置摘要。
- `variant-summary.csv`: `fixed-r2` / `fixed-r4` 的两机成对结果。
- `cross-machine-comparison.csv`: 两个 variant 的 1024x256 cross-machine gap。
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash evidence。
- `queue-summary.csv`: queue 状态。
- `invalid-runs.csv`: 本轮为空。
- `source-manifest.csv`: mirrored raw evidence 路径和 sha256。
- `metadata.json`: 收尾元数据。
