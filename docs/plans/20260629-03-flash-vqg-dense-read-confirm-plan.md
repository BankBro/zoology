# 20260629-03 Flash-VQG dense-read confirm plan

## 目的

上一轮 `20260629-02` 的最强信号是 `dense-read`:

- 17-step probe 中, `logf/M_state` 仍然 mismatch, 但 `phase2_read/top_idx` 和 `forward/loss` 在 step16 回到 match.
- 1 epoch screen 中, 2080ti `1024x256=0.892`, 3090 `1024x256=0.894`, 跨机器 gap 只有 `0.002`.

本轮只验证这个信号是否能撑到 4 epoch. 核心问题不是继续调参, 而是:

```text
cb64 下去掉 read top-k 硬候选选择后, Flash-VQG/gd_residual_v1 的跨机器效果是否能稳定在可接受范围内, 且没有明显性能税?
```

## 实验条件

共同条件:

- same MQAR canonical cache.
- same canonical init checkpoint.
- same batch order.
- no-dropout.
- `seed=123`.
- `data_seed=123`.
- `cb64-r16`.
- `write_topk=4`.
- `max_epochs=4`.
- `gradient_accumulation_steps=4`.
- `train_batch_size=64`.
- `validations_per_epoch=4`.
- 2080ti x1 + 3090 x1, 3090 若明显提前完成可追加 r2.

关键配置:

| item | value |
|---|---|
| `num_codebook_vectors` | `64` |
| `fox_remote_read_topk` | `64` |
| `fox_gd_residual_dense_read_chunked` | `True` |
| `fox_gd_residual_rank` | `16` |
| `fox_gd_residual_write_topk` | `4` |

解释:

- 当前 codebook size 是 64, 所以 `fox_remote_read_topk=64` 等价 full-code read.
- `fox_gd_residual_dense_read_chunked=True` 只是让 full-code GD residual read 分 chunk 计算, 避免展开全码张量造成显存风险. 它不改变本轮要验证的 dense-read 语义.
- 这轮仍是 diagnostic / confirm screen, 不写 official MQAR ledger.

## 前置硬门槛

启动训练前必须在两边容器内确认:

| item | expected |
|---|---|
| cache content hash | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| canonical init state hash | `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf` |
| CUDA/NVML | available |
| `torch.cuda.is_available()` | `True` |
| `train_batches` | `2815` |
| optimizer steps per epoch | `704` |
| total optimizer steps | `2816` |
| dropout | all zero |
| `fox_remote_read_topk` | `64` |
| `fox_gd_residual_dense_read_chunked` | `True` |

若任一项失败, 停止启动训练并记录原因.

## 执行流程

1. 新增 runner:
   - `zoology/experiments/flash_vqg/scripts/20260629-03-flash-vqg-dense-read-confirm/dense_read_4ep.py`
2. 本地验证:
   - `py_compile`.
   - `config-summary`.
   - GPU/CUDA 检查.
   - `cache-hash`, `verify-init`, `preflight`.
   - 1-step smoke.
3. commit/push zoology.
4. 3090 容器内 pull 到相同 commit, Flash-VQG 保持 `bc391c0`.
5. 双机 preflight.
6. 启动:
   - 2080ti: `dense-read-4ep-s123-r1`.
   - 3090: `dense-read-4ep-s123-r1`.
   - 若 3090 明显提前完成且 2080ti 仍在稳定训练, 可追加 `dense-read-4ep-s123-r2`.
7. 进入长训练稳定期后显式 `sleep 1200` 轮询.
8. 回收 3090 轻量 raw evidence 到主工作区.
9. collect artifact 并生成 report.
10. commit/push report 和 artifact.

## 判定口径

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

主要判定:

| 结果 | 解释 | 下一步 |
|---|---|---|
| 4ep `1024x256` gap <= 4pp, 且两边没有明显掉分 | dense-read 的 1ep 信号成立 | 继续做低成本 read-candidate 稳定化, 如 read_topk warmup 或 schedule |
| gap <= 4pp, 但性能明显低于 baseline | dense-read 稳定但有性能税 | 查 full-code read 是否改变有效 inductive bias, 再设计 warmup/schedule |
| gap > 4pp | 1ep 正信号不足以解释 4ep | 回到 gate/logf, state build, write support, M_state decay 的机制拆解 |
| 训练失败或显存不可接受 | full-code read 不是直接候选路径 | 优先做 read_topk larger/warmup, 不继续 dense full-code 长跑 |

用户当前接受误差:

```text
1024x256 accuracy gap within 4pp is acceptable.
```

## 产物

- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/run-summary.csv`
- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/cross-machine-comparison.csv`
- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/cache-init-preflight-summary.csv`
- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/queue-summary.csv`
- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/invalid-runs.csv`
- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/source-manifest.csv`
- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/metadata.json`
- `docs/artifacts/20260629-03-flash-vqg-dense-read-confirm/README.md`
- `docs/20260629-03-flash-vqg-dense-read-confirm-report.md`
