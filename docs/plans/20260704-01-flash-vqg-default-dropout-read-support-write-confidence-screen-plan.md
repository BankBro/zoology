# 20260704-01 Flash-VQG default-dropout read-support / write-confidence screen plan

## 目的

本轮在 default dropout 训练协议下, 用 7 个 variant x 2 台机器 = 14 个 paired 1ep run, 判断两件事:

1. 扩大 `read_topk` 是否能缓解 `gd_residual_v1` 的跨机器不稳定.
2. `topk_mass_scaled` 写入强度是否能缓解低置信 top-k write 对 `M_state` 的污染风险.

本轮是 diagnostic screen, 不是 official MQAR 正式实验, 不写 official ledger.

## 固定口径

- `seed=124`.
- `data_seed=123`.
- canonical MQAR cache.
- canonical seed124 init.
- same batch order.
- `max_epochs=1`.
- `max_train_steps=704`.
- `embed_dropout=0.1`.
- `resid_dropout=0.0`.
- `drop_path=0.0`.
- `cb64-r16`.
- `write_topk=4`.
- `vq_weight_mode=dense_softmax`.
- `read_trace_enabled=false`.
- `read_trace_train_steps=[]`.
- `train_inline_event_trace_enabled=false`.
- shadow dense read disabled.

## Variant

最终正式队列固定为 7 个 variant:

| variant | read_topk | write strength | injection warmup | 目的 |
|---|---:|---|---|---|
| `baseline-r2` | 2 | `renorm_topk` | none | default dropout r2 baseline |
| `baseline-r4` | 4 | `renorm_topk` | none | r4 high-risk baseline |
| `fixed-r8` | 8 | `renorm_topk` | none | read support ladder |
| `fixed-r16` | 16 | `renorm_topk` | none | read support ladder |
| `fixed-rmax` | selected | `renorm_topk` | none | highest safe read support |
| `write-mass-r2` | 2 | `topk_mass_scaled` | none | confidence-aware write probe |
| `write-mass-injwarm512-r2` | 2 | `topk_mass_scaled` | linear 0 -> 512 | write scaling + injection warmup |

`fixed-rmax` 由 smoke 决定:

1. 两机 `fixed-r64` smoke 都通过, 则 `fixed-rmax=fixed-r64`.
2. 否则两机 `fixed-r32` smoke 都通过, 则 `fixed-rmax=fixed-r32`.
3. 否则 `fixed-rmax=sched16to2-linear512`.

`fixed-r64` 必须设置:

```text
fox_remote_read_topk=64
fox_gd_residual_dense_read_chunked=true
```

`sched16to2-linear512` 设置:

```text
fox_remote_read_topk_initial=16
fox_remote_read_topk_final=2
fox_remote_read_topk_release_start_train_steps=0
fox_remote_read_topk_release_end_train_steps=512
fox_remote_read_topk_schedule=linear_int
fox_remote_read_topk_eval_policy=final
```

## 执行流程

1. 在 2080ti 主工作区新增脚本和 plan, commit/push.
2. 3090 在 `Flash-VQG-tun` 容器内 git pull 到同一 commit.
3. 两机执行 preflight:
   - `nvidia-smi` / NVML available.
   - `torch.cuda.is_available() == true`.
   - MQAR cache content hash match.
   - canonical init tensor hash match.
   - batch order hash match.
4. 两机先 smoke `fixed-r64`, 若失败则 smoke `fixed-r32`.
5. 根据两机 smoke 结果确定 `fixed-rmax`.
6. 对最终 7 个 variant 在两机都跑 `max_train_steps=8` smoke.
7. 只有 14 个 smoke 全部通过, 才启动正式 14 run 队列.
8. 两台机器各启动一个顺序队列, 自动接续 7 个正式 1ep run.
9. 启动后监控到稳定训练阶段再退出会话.

## 退出会话前必须确认

- `fixed-rmax` 已确定.
- 14 个最终 smoke 全部通过.
- 两机正式队列均已启动.
- 两机第一个正式 run 已进入 train loop 并持续推进.
- GPU/NVML/CUDA 正常.
- 队列 PID, output root, queue log, per-run log 路径已记录.
- 队列 manifest 明确列出剩余自动接续 run.

## 收尾计划

用户确认实验结束后, 再执行:

- 回收两机轻量 raw evidence.
- 生成 `docs/artifacts/20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen/`.
- 写 `docs/20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen-report.md`.
- commit/push report 和 artifact.

报告至少回答:

1. `fixed-rmax` 最终选择了 r64, r32, 还是 fallback.
2. `read_topk=2/4/8/16/max` 的分数和 gap 如何变化.
3. default dropout 下 near/full read support 是否有帮助.
4. `topk_mass_scaled` 是否改善 r2.
5. `topk_mass_scaled + injection warmup` 是否比单独 write scaling 更好.
6. 哪些 variant 值得进入下一轮 4ep confirm 或机制实现.
