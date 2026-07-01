# 20260702-01 Flash-VQG update_norm event trace probe plan

## 背景

`20260701-04` 显示, default dropout, fixed read_topk=2 下, `fox_gd_residual_update_norm_cap=0.5` 可以把 2080ti/3090 的 1ep hard slice gap 从 `43.7pp` 降到 `2.8pp`, 但 read support 仍然不匹配。这说明 cap 更像是在降低 divergent residual write 的后果, 而不是让两台机器走回同一条离散路径。

本轮不把 hard cap 当作最终方案。目标是补事件级证据, 判断是否存在少数偏大的 `M_state` residual update, 以及这些事件是否集中出现在早期训练阶段并与后续 gap 收敛相关。

## 核心问题

1. baseline-r2 中, 哪些 step/layer/code/token 的 uncapped update norm 最大?
2. 这些 top update 如果套用 hypothetical cap=0.5, 会有多少事件被截断, 截断比例多大?
3. cap0.5-r2 中, 实际 cap 命中事件的分布和 baseline 的 hypothetical cap 命中分布是否相近?
4. cap0.5 稳定 gap 的同时, 是否只是减少少数 top update 的幅度, 而不是修复 read support mismatch?

## 实验配置

共同条件:

- `seed=124`
- `data_seed=123`
- canonical MQAR cache
- canonical seed124 init checkpoint
- `cb64-r16`
- `write_topk=4`
- `read_topk=2`
- `embed_dropout=0.1`
- `resid_dropout=0.0`
- `drop_path=0.0`
- `max_epochs=1`
- `max_train_steps=704`
- machines: `2080ti` + `3090`

variants:

| variant | update_norm_cap | 用途 |
|---|---:|---|
| `baseline-r2` | none | 观察未限制的 top update event |
| `ucap0p5-r2` | 0.5 | 复核 cap0.5 的稳定化效果和实际 cap hit |

trace steps:

```text
0,1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,704
```

注意: 当前 trace 是在指定 optimizer step 上对固定 validation batch 做 eval forward, 用于观察同一训练进度下的 state/read/write 机制快照。它不是直接记录实际 training minibatch 的每一次写入。

## 新增诊断开关

Flash-VQG 新增默认关闭配置:

```text
fox_gd_residual_update_event_trace_enabled = false
fox_gd_residual_update_event_trace_topk = 64
fox_gd_residual_update_event_trace_hypothetical_cap = null
```

实验中开启:

```text
fox_gd_residual_update_event_trace_enabled = true
fox_gd_residual_update_event_trace_topk = 64
fox_gd_residual_update_event_trace_hypothetical_cap = 0.5
```

写入文件:

```text
update_event_trace.jsonl
```

每条记录包含:

- `global_step`, `layer_idx`, `block_idx`, `sample_idx`, `head_idx`, `code_idx`, `token_pos`
- `update_norm_uncapped`, `err_norm`, `zeta_before_update_cap`, `zeta_after_update_cap`
- `actual_cap_hit`, `actual_cap_scale`
- `hypothetical_cap_hit`, `hypothetical_cap_scale`
- write routing context: `raw_topk_mass`, `write_top1_mass`, `write_q_top1`, `write_q_entropy`

## 判定口径

如果 baseline 在早期 step 出现明显更大的 top update, 且 hypothetical cap=0.5 会命中这些事件, 同时 cap0.5 的 final gap 继续保持在 4pp 内, 则支持:

```text
少数偏大的 residual update 是扰动放大器之一.
```

如果 baseline top update 并不大, 或 cap0.5 仍然出现类似大 update 但 final gap 仍稳定, 则说明 cap 的收益可能来自其他训练动态, 需要继续拆 beta/lambda 或 read/write support。

## 产物

实验完成后新增:

```text
docs/20260702-01-flash-vqg-update-norm-event-trace-probe-report.md
docs/artifacts/20260702-01-flash-vqg-update-norm-event-trace-probe/
```

artifact 至少包含:

- `run-summary.csv`
- `variant-gap-summary.csv`
- `cap-metrics-summary.csv`
- `early-window-summary.csv`
- `update-event-trace-summary.csv`
- `update-event-step-summary.csv`
- `update-event-cross-machine-summary.csv`
- `update-event-top.csv`
- `read-trace-cross-machine-summary.csv`
- `hash-probe-comparison-summary.csv`
- `cache-init-preflight-summary.csv`
- `metadata.json`
