# 20260629-03 Flash-VQG dense-read 4ep confirm 报告

status: completed_diagnostic
ledger: not written

## 目标

本轮验证 `20260629-02` 中 `dense-read` 的 1 epoch 正信号能否延续到完整 4 epoch。

核心问题是:

```text
如果 cb64 下把 read_topk 从 2 改为 64, 即读取全部 code,
去掉 read top-k candidate flip 这个离散放大点,
跨机器 1024x256 hard slice 是否能稳定在 4pp 容忍线内?
```

本轮是 diagnostic / confirm screen, 不写 official MQAR ledger。

## 执行口径

代码版本:

- zoology: `flash-vqg`, commit `6b172ab`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `bc391c0`.

共同配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`.
- `fox_remote_read_topk=64`.
- `num_codebook_vectors=64`.
- `fox_gd_residual_dense_read_chunked=True`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `max_epochs=4`, `validations_per_epoch=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- 使用同一份 canonical init checkpoint。

前置硬门槛:

- 目标容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 均通过。
- MQAR cache content hash 为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- init model state hash 为 `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.
- preflight 确认 `read_topk=64`, `num_codebook_vectors=64`, `dense_read_chunked=True`.

## 有效结果

主指标是 `valid/mqar_case/accuracy-1024x256`。

| machine | target | duration | final valid acc | final 1024x256 | best 1024x256 | best-final drop |
|---|---|---:|---:|---:|---:|---:|
| 2080ti | `dense-read-4ep-s123-r1` | 300.1 min | 0.986 | 0.916 | 0.947 | 3.1pp |
| 2080ti | `dense-read-4ep-s123-r2` | 298.6 min | 0.989 | 0.937 | 0.948 | 1.1pp |
| 3090 | `dense-read-4ep-s123-r1` | 213.5 min | 0.983 | 0.896 | 0.941 | 4.5pp |
| 3090 | `dense-read-4ep-s123-r2` | 214.2 min | 0.984 | 0.901 | 0.953 | 5.2pp |

成对跨机器对比:

| pair | 2080ti final | 3090 final | final gap | within 4pp | 2080ti best | 3090 best | best gap |
|---|---:|---:|---:|---|---:|---:|---:|
| r1 | 0.916 | 0.896 | 2.0pp | true | 0.947 | 0.941 | 0.6pp |
| r2 | 0.937 | 0.901 | 3.6pp | true | 0.948 | 0.953 | 0.5pp |

同机器 repeat:

| machine | final range | final spread | best range | best spread |
|---|---|---:|---|---:|
| 2080ti | 0.916 / 0.937 | 2.1pp | 0.947 / 0.948 | 0.1pp |
| 3090 | 0.896 / 0.901 | 0.5pp | 0.941 / 0.953 | 1.2pp |

全局口径:

- 4 条有效 run 的 final 1024x256 min/max 为 `0.896 / 0.937`, spread 为 `4.1pp`.
- 4 条有效 run 的 best 1024x256 min/max 为 `0.941 / 0.953`, spread 为 `1.2pp`.

因此如果按成对跨机器比较, r1 和 r2 都在 4pp 内。若按四条 final 的全局 min-max 看, 刚好略高于 4pp, 主要来自 `2080ti r2=0.937` 与 `3090 r1=0.896`。这说明 dense-read 明显改善稳定性, 但 hard slice final 仍有末期波动。

## 无效启动

| queue | machine | status | 原因 | 是否计入结果 |
|---|---|---|---|---|
| `2080ti-r1` | 2080ti | `failed:stale-start` | 本地 `nohup` 启动没有进入训练 | no |
| `2080ti-r1-retry1` | 2080ti | `failed:stale-start` | 本地 `nohup` 启动没有稳定接管训练 | no |
| `3090-r2` | 3090 | `failed:1` | 手写 launch 脚本使用了不存在的 init checkpoint 路径 | no |

上述无效 run 均未产生有效训练结果, 已记录到 `invalid-runs.csv`, 不参与质量结论。

后补启动的 `2080ti-r2` 和 `3090-r2-retry1` 训练完成后, 原始 `queue-status.tsv` 因手写 launch wrapper 少了标准表头和独立 `finished_at` 字段, 收集前已补成标准状态记录。该修正只影响 queue metadata, 不改训练日志, config 或 result JSON。

## 判读

本轮支持 dense-read 方向:

```text
在 same cache, same init, no-dropout, seed=123 条件下,
把 read_topk=2 改成 read_topk=64 后,
两组 2080ti vs 3090 成对 4ep final gap 均在 4pp 内。
```

和 `20260628-02` no-dropout/read_topk=2 相比, 这是明显进展:

| setting | 2080ti final | 3090 final | cross-machine gap |
|---|---:|---:|---:|
| no-dropout, read_topk=2, r1 | 0.840 | 0.790 | 5.0pp |
| no-dropout, read_topk=2, 3090 r2 vs 2080ti r1 | 0.840 | 0.762 | 7.8pp |
| dense-read, read_topk=64, r1 | 0.916 | 0.896 | 2.0pp |
| dense-read, read_topk=64, r2 | 0.937 | 0.901 | 3.6pp |

这和 17-step probe / 1ep screen 的判断一致:

```text
read top-k candidate flip 是 Flash-VQG 跨机器不稳定的重要放大器。
```

但 dense-read 还不能直接当最终训练方案:

- 它改变了 read support, cb64 下等价于读全部 code.
- 计算和显存成本更高。
- final hard slice 仍有 late validation volatility, 尤其 3090 best-to-final drop 为 4.5pp / 5.2pp.

更稳妥的结论是:

```text
dense-read 是成功的 diagnostic confirm:
去掉 read top-k 离散候选翻转后, 4ep 跨机器 final gap 明显收敛。
后续应把它转化为更低成本的 read-candidate 稳定化方案,
而不是直接把 dense-read 当默认最终方法。
```

## 后续建议

优先设计低成本 read 稳定方案:

1. `read_topk` warmup: 训练早期使用较大 read top-k, 后期逐步收紧。
2. top-k margin 监控: 当候选分数太接近时, 记录或避免硬翻转。
3. soft/dense screen 后收紧: 用 dense/soft read 作为早期稳定器, 再回到稀疏 read。
4. 保留 gate/logf 稳定化研究, 但不要把 `round1e-5` 当方案。

下一轮若进入机制改进, 推荐先做低成本 `read_topk schedule` 1ep screen, 不直接启动多条 4ep。

## 产物

Artifact:

```text
docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/
```

核心文件:

- `run-summary.csv`: 4 条有效 4ep run 指标。
- `cross-machine-comparison.csv`: 以 2080ti r1 为参考的 final gap 对比。
- `pairwise-repeat-comparison.csv`: r1/r2 成对跨机器 final/best gap。
- `invalid-runs.csv`: 3 条无效启动记录。
- `cache-init-preflight-summary.csv`: cache/init/preflight 证据。
- `queue-summary.csv`: 标准化 queue 状态。
- `source-manifest.csv`: 轻量 raw evidence manifest.
- `metadata.json`: collection metadata.
