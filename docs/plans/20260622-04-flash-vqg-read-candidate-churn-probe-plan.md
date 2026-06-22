# 20260622-04 Flash-VQG read candidate churn probe plan

## 目的

这轮实验只回答一个具体问题: fixed `read_topk` 正负效果是否能被 read candidate 的稳定性解释。这里的 candidate churn 指同一验证 batch 位置上的 query token, 在不同 validation 时刻选出的 remote read topk code 集合变化有多大。

## 背景

前面的 readk boundary audit 已经收紧到两个事实:

- `cb256-r4/r8 + fixed read_topk=4` 多数历史 evidence 是正向的.
- `cb128-r8 + fixed read_topk=4` 有 repeat collapse 反例, 不能作为全局默认.

因此现在需要补 read-side telemetry, 不再只看 final hard accuracy。候选频繁换的直觉解释是: 如果同一验证样本位置在 early validation 之间 topk code 集合变化很大, residual read 更可能处在不稳定边界; 如果 topk 扩大后 churn 降低或 selected mass 更健康, 它可能解释 cb256 系列为什么被救回。

## 实现范围

代码侧新增:

- Flash-VQG `gd_residual_v1` phase2 remote read 记录 topk candidate set.
- Zoology validation runtime 只在指定 valid batch 上打开 probe, 默认 `valid_batch=0`, `max_samples=16`, 只看 target query token.
- CLI 新增 `--read-churn-probe-*` 参数, 并写入 generated config, TrainConfig 和 manifest summary.

新增指标:

- `attn/gd_residual_read_candidate_probe_count`
- `attn/gd_residual_read_candidate_has_prev`
- `attn/gd_residual_read_candidate_retention_mean`
- `attn/gd_residual_read_candidate_churn_mean`
- `attn/gd_residual_read_candidate_top1_flip_rate`

已存在并需要一起记录的 read-side 指标:

- `attn/gd_residual_read_margin_top1_top2_mean`
- `attn/gd_residual_read_margin_top1_top2_p05`
- `attn/gd_residual_read_entropy_mean`
- `attn/gd_residual_read_selected_mass_mean`
- `attn/gd_residual_read_selected_mass_p05`
- `attn/gd_residual_remote_read_topk_effective`

## 实验矩阵

本轮只跑三条 3090 长训, 2080ti 只做 smoke:

| run | layout | seed | read_topk | 作用 |
|---|---:|---:|---:|---|
| `cb256r8-readk4-s123-churn` | `cb256-r8` | `123` | `4` | cb256 正向候选, 看 churn 是否低 |
| `cb256r8-readk2-s123-churn` | `cb256-r8` | `123` | `2` | matched baseline, 对照 readk4 |
| `cb128r8-readk4-s125-churn` | `cb128-r8` | `125` | `4` | failure boundary, 看 churn/flip 是否异常 |

公共配置:

- `data_seed=123`
- `d_model=128`
- `block_len=32`, `local_num_blocks=2`
- `max_epochs=4`
- `validations_per_epoch=4`
- `train_batch_size=64`, `eval_batch_size=16`, `gradient_accumulation_steps=4`
- `fox_remote_formula=gd_residual_v1`
- `fox_gd_residual_write_topk=4`
- `vq_score_mode=codebook_dot`, `vq_weight_mode=dense_softmax`, `vq_update_mode=grad`, `vq_softmax_tau=0.25`
- early stopping disabled

## 判断口径

这轮不要求直接证明根因, 只做边界分类:

- 如果 `cb256-r8 readk4` 的 churn/flip 明显低于 readk2, 且 selected mass/margin 更健康, 则支持 "read_topk=4 通过稳定候选覆盖救 cb256".
- 如果 `cb128-r8 readk4` 的 churn/flip 高或 margin/selected mass 异常, 则支持 "cb128 是 read-side boundary/failure case".
- 如果三条 churn 都相近, 则 readk4 正负效果更可能来自 write/state 或 layout capacity, 下一步转向 write-side guarded cap + per-code usage.

## 收尾产物

长训完成后收尾:

- 3090 generated config, manifest, logs 轻量同步回 2080ti.
- 生成 `docs/artifacts/20260622-04-flash-vqg-read-candidate-churn-probe/`.
- 写 `docs/20260622-04-flash-vqg-read-candidate-churn-probe-report.md`.
- report 同时列 final 和 best hard accuracy, 并报告每条 run 的 churn/retention/top1 flip/read margin/entropy/selected mass.

