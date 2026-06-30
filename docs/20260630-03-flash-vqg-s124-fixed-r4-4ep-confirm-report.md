# 20260630-03 Flash-VQG s124 fixed-r4 4ep confirm 报告

status: completed_diagnostic
ledger: not written

## 目标

本轮验证 `seed=124` 下 `fixed-r4` 的 1 epoch 正信号能否延续到 4 epoch。

上一轮 `20260630-02` 显示, 在 canonical cache/init/no-dropout 口径下, `fixed-r4` 的 1 epoch 两机结果为 `0.900/0.897`, gap `0.3pp`。本轮只做最小确认, 不继续扩大 readk/schedule/dropout 网格。

## 执行口径

代码版本:

- zoology: `flash-vqg`, commit `2dcd362`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `bc391c0`.

共同配置:

- `seed=124`, `data_seed=123`.
- `cb64-r16`.
- `fox_gd_residual_write_topk=4`.
- train-time `fox_remote_read_topk=4`.
- `max_epochs=4`, `validations_per_epoch=4`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- 使用同一份 canonical MQAR cache.
- 使用同一份 seed124 canonical init checkpoint.

前置硬门槛:

- 2080ti 和 3090 容器内 `nvidia-smi` 与 `torch.cuda.is_available()` 均通过。
- MQAR cache content hash 为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`, 两边 match。
- seed124 init model state hash 为 `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`, 两边 match。
- preflight 确认 `max_epochs=4`, 每 epoch `704` optimizer steps, total `2816` optimizer steps, no-dropout, `read_topk=4`, `write_topk=4`.

本轮是 diagnostic / exploratory confirm, 不写 official MQAR ledger。

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`。

| machine | final 1024x256 | best 1024x256 | best-final drop | final valid acc | duration |
|---|---:|---:|---:|---:|---:|
| 2080ti | 0.944 | 0.949 | 0.5pp | 0.990 | 219.1 min |
| 3090 | 0.953 | 0.955 | 0.2pp | 0.991 | 165.0 min |

跨机器 gap:

| variant | 2080ti final | 3090 final | final gap | within 4pp | best gap |
|---|---:|---:|---:|---|---:|
| `fixed-r4` | 0.944 | 0.953 | 0.9pp | yes | 0.6pp |

本轮 2 条 run 全部有效完成, `invalid-runs.csv` 为空。

## 判读

本轮 `s124 fixed-r4 4ep confirm` 通过当前 no-dropout 跨机器容忍线。

具体来说:

- `final` gap 是 `0.9pp`, 明显小于用户可接受的 `4pp`。
- 两边 final 都处在高位: 2080ti `0.944`, 3090 `0.953`。
- best-final drop 很小: 2080ti `0.5pp`, 3090 `0.2pp`。
- 这说明 `fixed-r4` 的 `s124` 正信号不是只在 1 epoch 短跑中成立, 至少在当前 4 epoch/no-dropout/canonical cache/init 口径下仍然成立。

结合前两轮:

| 实验 | seed | epoch | variant | 2080ti final | 3090 final | gap | 结论 |
|---|---:|---:|---|---:|---:|---:|---|
| `20260630-01` | 123 | 1 | `fixed-r4` | 0.928 | 0.923 | 0.5pp | 稳定 |
| `20260630-02` | 124 | 1 | `fixed-r4` | 0.900 | 0.897 | 0.3pp | 稳定 |
| `20260630-03` | 124 | 4 | `fixed-r4` | 0.944 | 0.953 | 0.9pp | 稳定 |

这比 `fixed-r2` 更有说服力。上一轮同 seed `fixed-r2` 的 1 epoch gap 是 `6.5pp`, 且两边分数低于 `fixed-r4`。因此目前最合理的判断是:

```text
在 no-dropout, canonical cache/init, cb64-r16, write_topk=4 口径下,
train-time read_topk=4 是比 read_topk=2 更稳也更强的候选。
```

## 限制

本轮仍然是 no-dropout, 所以不能回答 dropout 加回后的稳定性。

本轮只覆盖:

- `seed=124`.
- `data_seed=123`.
- `cb64-r16`.
- `write_topk=4`.
- train-time `read_topk=4`.
- 2080ti x1 + 3090 x1.

因此不能直接外推到所有 seed, capacity, dropout, longer-MQAR 或其他 read/write 配置。

另一个执行细节: 2080ti 的训练 Python 已正常退出并写出 `result_json`, 日志无 `Traceback` / OOM / NaN / Inf, GPU 也释放; 但外层 queue monitor 在训练完成后仍停留在 `sleep 1200`, 没有自动追加 `completed` 行。收尾时基于完整 `result_json`, 完整日志和空闲 GPU 手动补齐了 2080ti 的 `completed` queue 行, 以免 collect 把成功 run 误判为 invalid。这个问题只影响 queue wrapper 状态记录, 不影响训练结果本身。

## 下一步

建议下一步不要再继续扩同一批 no-dropout fixed-r4 重复实验。当前证据已经足够支持:

```text
fixed-r4 可以作为 no-dropout 稳定候选继续推进。
```

下一步应回到真正还没回答的问题:

```text
dropout 加回来以后, read_topk=4 是否还能稳定?
```

建议最小下一轮:

- canonical MQAR cache.
- canonical init.
- `seed=124`, `data_seed=123`.
- `cb64-r16`.
- `write_topk=4`.
- train-time `read_topk=4`.
- `max_epochs=4`.
- 加回当前默认 dropout 口径, 至少 `embed_dropout=0.1`, 其他 dropout 按当前默认配置明确记录。
- 2080ti x1 + 3090 x1。

判定仍看 `valid/mqar_case/accuracy-1024x256` final/best, best-final drop, 和两机 final gap 是否 `<= 4pp`。

如果 default dropout 下 `fixed-r4` 仍稳定, 才能把 `read_topk=4` 推向更接近实际训练默认的候选。如果 dropout 下再次不稳, 下一步再考虑 read candidate 稳定化, residual/write/state 限幅, 或 dropout/RNG 路径的单独拆解。

## 产物

Artifact:

```text
docs/artifacts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/
```

核心文件:

- `run-summary.csv`: 两条有效 run 的 final/best 指标和配置摘要。
- `variant-summary.csv`: `fixed-r4` 的两机成对结果。
- `cross-machine-comparison.csv`: 1024x256 cross-machine gap。
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash evidence。
- `queue-summary.csv`: queue 状态。
- `invalid-runs.csv`: 本轮为空。
- `source-manifest.csv`: mirrored raw evidence 路径和 sha256。
- `metadata.json`: 收尾元数据。
