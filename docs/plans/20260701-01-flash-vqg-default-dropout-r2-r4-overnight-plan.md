# 20260701-01 Flash-VQG default-dropout r2/r4 overnight plan

status: planned
ledger: not written for diagnostic/probe runs

## 目标

本轮利用约 8 小时 GPU 窗口, 先回答一个硬问题:

```text
default dropout 下, fixed-r2 在 3090 上能否复现 2080ti 的 0.877 signal?
```

如果通过, 再启动 `fixed-r2` 4 epoch paired confirm. 如果不通过, 不继续盲跑 4 epoch, 转入 bounded diagnostic queue, 用短窗口 probe 拆清 `fixed-r2` 和 `fixed-r4` 在 read support, residual injection, M_state/write pressure 上如何分开。

## 共同条件

- branch: `flash-vqg`.
- seed: `124`.
- data seed: `123`.
- canonical MQAR cache: 内容 hash 必须 match.
- canonical seed124 init checkpoint: state_dict tensor hash 必须 match.
- model: `cb64-r16`.
- `vq_weight_mode=dense_softmax`.
- `fox_gd_residual_write_topk=4`.
- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- 主指标: `valid/mqar_case/accuracy-1024x256`.
- 4pp 以内视为可接受 cross-machine gap.

## 实验调度

### P0: 3090 fixed-r2 1ep gate

配置:

```text
machine = 3090
read_topk = 2
write_topk = 4
max_epochs = 1
```

判定:

| P0 结果 | 处理 |
|---|---|
| `final_1024x256 >= 0.837` | 通过. 启动 `fixed-r2` 4ep paired confirm. |
| `0.800 <= final_1024x256 < 0.837` | 边界. 不跑 4ep, 重复 fixed-r2 1ep, 同时跑 B2 probe. |
| `final_1024x256 < 0.800` | 失败. 不跑 4ep, 转入 B1/B2/B3 diagnostic queue. |

### P0 通过分支

启动:

```text
2080ti GPU0: fixed-r2 4ep
3090 GPU0: fixed-r2 4ep
2080ti GPU1: B2 r2/r4 early probe, 如果空闲
```

### P0 不通过或边界分支

按剩余时间顺序启动:

1. B1: `fixed-r2` 1ep repeat, 2080ti + 3090 paired.
2. B2: `fixed-r2` / `fixed-r4` early read-support probe, `max_train_steps=128`.
3. B3: `fixed-r4-residual-zero` 1ep, 使用 `fox_gd_residual_residual_norm_mode=zero`.
4. B4: 如果仍有时间, `fixed-r4-dropout005` 1ep.

## B2 probe 指标

B2 不是正式质量实验, 只用于定位 r2/r4 早期分叉。固定:

```text
trace steps = 0,16,64,128
valid batch = 441
read_trace_query_only = true
max_train_steps = 128
```

必须记录:

- quality: `train/loss`, `early_window/loss`, final `valid/loss`, final `1024x256 hard acc`.
- read support churn: `read_candidate_retention_mean`, `churn_mean`, `top1_flip_rate`, `probe_count`, `has_prev`.
- read support confidence: `read_margin_top1_top2_mean/p05`, `read_entropy_mean`, `read_selected_mass_mean/p05`.
- residual injection: `gd_residual_lambda_mean`, `gd_residual_inject_ratio`.
- M/write pressure: `m_norm_mean/max`, `update_norm_mean/p95/max`, `write_strength_mean/p95/max`, `sum_zeta_mean/p95/max`.
- write routing: `raw_topk_mass_mean/p05`, `write_top1_mass_mean`, `write_q_entropy_mean`, `write_q_top1_mean`.
- VQ routing: `vq/relative_err_mean`, `vq/c_entropy`, `vq/write_entropy_mean`, `vq/write_top1_mass_mean`.
- layer view: 至少保留 `layer_0`, `layer_1`, `layer_all`.
- raw trace: `read_trace.jsonl`, 包含 sample hash, token/head/layer, top-k ids/scores/probs, margin, entropy, selected mass.

判读时必须分开:

- `read_churn`: 同一 run 内固定 valid batch 随训练步变化的 candidate churn.
- `cross-machine support mismatch`: 用 `read_trace.jsonl` 对齐同一 sample/token/head/layer 后比较 2080ti 与 3090 的 top-k support.

## 收尾

生成:

- `docs/artifacts/20260701-01-flash-vqg-default-dropout-r2-r4-overnight/`
- `docs/20260701-01-flash-vqg-default-dropout-r2-r4-overnight-report.md`

报告必须包含:

- preflight cache/init hash.
- P0 结果和分支决策.
- 1ep/4ep run summary.
- B2 early-window scalar summary.
- B2 read trace summary 和 cross-machine support comparison.
- 是否进入 4ep, 如果没有, 明确原因.
- 下一步实验建议.
