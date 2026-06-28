# 20260629-01 Flash-VQG gate/logf stability plan

status: implementation_ready
experiment_id: `20260629-01-flash-vqg-gate-logf-stability`

## 目标

本轮不先跑长训练, 先验证一个窄问题:

```text
在相同 cache, 相同 canonical init, 相同 batch order, no-dropout 条件下,
fox_gate_logf 的跨 GPU 极小数值差异是否是后续 G/L/M state, read top-k, preds/loss 分叉的触发点?
```

本轮 P0/P1 是 diagnostic / exploratory, 不写 official MQAR ledger.

## 已排除项

- cache 不一致: 已用 canonical cache content hash 排除.
- init 不一致: 已用 canonical init state hash 排除.
- dropout 作为充分方案: no-dropout 1 epoch 有帮助, 但 4 epoch confirm 失败.
- fixed `read_topk=4` 主线: 只在 cb256-like layout 是局部正例, 不是全局默认.
- 单纯 `m_norm` guard: 历史 write-control 审计已说明不足.

## P0 gate/logf origin probe

共同配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`, `read_topk=2`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- canonical cache hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.

矩阵:

| machine | run |
|---|---|
| 2080ti | `gate-logf-probe-s123-r1` |
| 3090 | `gate-logf-probe-s123-r1` |

trace step:

```text
0,1,4,16
```

新增 trace:

- `fox_gate/input_x`
- `fox_gate/logits_cuda`
- `fox_gate/logf_cuda`
- `fox_gate/logits_ref_fp64_cpu`
- `fox_gate/logf_ref_fp64_cpu`
- 原有 `state_build/logf_all`, `G_state`, `L_state`, `M_state`, `phase2_read/top_idx`, `forward/preds`, `forward/loss`.

判读:

- `logits_cuda` 已 mismatch: 优先查 `fox_gate_proj(x)` linear/GPU 数值路径.
- `logits_cuda` match 但 `logf_cuda` mismatch: 优先查 `F.logsigmoid` 或后处理.
- CPU fp64 shadow match 但 CUDA path mismatch: GPU 数值路径是直接来源.
- 若 P0 不能定位来源, 暂停 P1, 不开长训.

## P1 1 epoch screen

只在 P0 有明确结果后启动.

已实现的最小干预开关:

- `fox_gate_logf_compute_mode=fp32_linear`: 显式用 fp32 `F.linear` 计算 gate logits.
- `fox_gate_logf_round_quantum=1e-6` 或 `1e-5`: 对最终 `logf` 做轻量 rounding.
- `fox_gate_logit_normalizer=32` 或 `64`: 降低 gate/logf 动态幅度.

P1 先跑每个候选的 `2080ti x1 + 3090 x1`; 有希望再补 `3090 r2`.

通过条件:

- 主指标 `valid/mqar_case/accuracy-1024x256`.
- 3090 相对 2080ti gap <= 4pp.
- 3090 repeat gap <= 3pp.
- final 分数不能明显低于 no-dropout 1 epoch 同口径.
- 无 NaN/OOM/Traceback, cache/init/batch order hash 全 match.

## P2 4 epoch confirm

只有 P1 通过才启动.

矩阵:

| machine | repeat |
|---|---:|
| 2080ti | 1 |
| 3090 | 2 |

仍只跑 `s123`, 不直接扩三 seed.

通过条件:

- final `1024x256` gap <= 4pp.
- 3090 repeat 稳定.
- best-final gap 小.
- final 不明显低于历史可用区间.

失败即停止该方向, 不补同类 4 epoch repeat.

## 产物

脚本:

```text
zoology/experiments/flash_vqg/scripts/20260629-01-flash-vqg-gate-logf-stability/gate_logf_stability_probe.py
```

Artifact:

```text
docs/artifacts/20260629-01-flash-vqg-gate-logf-stability/
```

核心表:

- `trace-summary.csv`
- `cross-machine-trace-comparison.csv`
- `gate-comparison-summary.csv`
- `preflight-summary.csv`
- `source-manifest.csv`
- `metadata.json`

报告:

```text
docs/20260629-01-flash-vqg-gate-logf-stability-report.md
```
