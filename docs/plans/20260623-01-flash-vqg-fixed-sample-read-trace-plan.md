# 20260623-01 Flash-VQG fixed-sample read trace plan

## 目的

上一轮 read candidate churn probe 只记录了聚合 overlap, 没有把固定 validation 样本, 固定 query token 的 candidate 明细落盘. 本实验补齐这个证据缺口, 用严格 trace 判断 readk4 是否通过稳定候选或扩大候选覆盖救回 `cb256-r8 s125`, 以及 `cb128-r8 s125` 的边界风险是否能在 read trace 中看出来.

## 机器与配置

正式训练在 `3090` 的 `Flash-VQG-tun` 容器内运行. 2080ti 只做代码开发, smoke, 同步和收尾.

首批 3 条 run:

| target | 目的 |
|---|---|
| `cb256r8-readk2-s125-trace` | 历史 weak baseline |
| `cb256r8-readk4-s125-trace` | 历史 readk4 rescue 对照 |
| `cb128r8-readk4-s125-trace` | 边界 layout 对照 |

公共训练设置:

```text
data_seed=123
max_epochs=4
validations_per_epoch=4
gradient_accumulation_steps=4
train_batch_size=64
eval_batch_size=16
gd_rank=8
write_topk=4
vq_topk=4
vq_tau=0.25
```

## Trace 内容

只记录 validation batch `441`, 前 `4` 个样本, 每样本前 `8` 个 query position, 所有 layer/head. 在当前 eval batch size 16 下, batch `441` 对应 `1024x256` hard slice 的第一批, 避免短 slice 上 remote read 被 mask 掉导致 `selected_mass=0`. 每条 JSONL record 包含:

```text
run_id, epoch, global_step, valid_batch_idx, sample_idx
input_hash, target_hash
layer_idx, head_idx, block_idx, token_idx, query_pos
read_topk
topk_candidate_ids, topk_scores, topk_probs
margin_top1_top2, entropy, selected_mass
```

## 收尾与判断

收尾时归档:

```text
docs/artifacts/20260623-01-flash-vqg-fixed-sample-read-trace/
```

至少包含 `final.csv`, `final_best_metrics.csv`, `source_manifest.csv`, `metadata.json`, `trace_summary.csv`, `README.md`, 以及压缩后的 `traces/*.jsonl.gz`.

核心判断:

- `cb256-r8 s125 readk4` 是否比 readk2 有更低 top1 flip, 更高 retention, 更好的 margin 或 selected mass.
- `cb128-r8 s125 readk4` 是否表现出高 `m_norm` 同时伴随 margin/entropy/candidate 异常.
- 如果 fixed-sample trace 能解释 readk4 rescue, 下一步再实现 margin-aware read gate. 如果不能解释, 优先转向 write cap guard.
