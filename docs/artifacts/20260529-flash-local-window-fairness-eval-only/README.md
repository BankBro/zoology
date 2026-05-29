# Flash local window fairness eval-only 补强

本 artifact 只保存 20260529 Flash local window fairness 的 eval-only 补强结果. 不包含新训练, 不写入正式训练 ledger.

## 子目录

- `stage3-longer-mqar-bucket/`: 对阶段 3 已训练的 `local-only`, `local1`, `local4` checkpoint 跑 longer-MQAR distance bucket eval. 每个 variant 覆盖 `last.pt` 和 `best.pt`, slices 为 `1024x256`, `2048x512`, `4096x512`, `4096x1024`.
- `near-distance-enriched/`: 使用同一批 stage3 checkpoint 跑 near-distance enriched eval, 重点补足 `<=32`, `33-64`, `65-128` 三个距离桶.

## 关键文件

- `source_checkpoints.csv`: 由 `stage3_train_summary.csv` 自动生成, 不手填 checkpoint 路径. 包含 checkpoint 绝对路径, 相对路径, sha256, `best/last` 类型, 训练 summary 来源和 git commit.
- `slice_summary.csv`: 每个 checkpoint x slice 的整体 accuracy, stderr, 95% CI, batch size, wall time, run status.
- `distance_bucket.csv`: 每个 checkpoint x slice x distance bucket 的 n, correct, accuracy, stderr, 95% CI.
- `eval_runs.csv`: 每个 eval run 的简要状态.
- `metadata.json`: 命令, seed, slice, dataset mode, batch fallback, GPU, git commit 和运行状态.
- `status.md`: 生成状态日志.

## 距离定义

Distance bucket 使用 MQAR sample-level position metadata 计算:

```text
distance = query_pos - value_pos
```

runner 显式构造并保存 `query_pos`, `key_pos`, `value_pos` 对应 metadata. 禁止通过 token value 在 `input_ids` 中反查 target 位置, 因为 MQAR token 可能重复.

## near-distance enriched 生成策略

`near-distance-enriched` 使用 `dataset_mode=near_enriched`, `num_examples=500`, `near_pairs_per_bucket=16`.

每个样本只强化一个近距离桶, 样本之间轮换三个目标桶:

- `<=32`: 固定代表距离 31, 以便每个强化样本放入 16 个互不冲突 query gap.
- `33-64`: 在 odd distances 33-63 内均匀采样.
- `65-128`: 在 odd distances 65-127 内均匀采样.

其他 query gap 由剩余可用位置随机无放回填充, 因此非目标桶也可能出现自然样本. 所有 bucket 仍按实际 `query_pos - value_pos` metadata 统计.

## 运行状态

- 机器: `mclab-3090` 的 `Flash-VQG-tun` 容器.
- 环境: `/home/lyj/miniconda3/envs/flash-vqg/bin/python`.
- dtype: `torch-fp32; GDN_KERNEL_DTYPE=float32`.
- batch candidates: `8,4,2,1`; 两段正式 eval 实际 batch size 都为 8, OOM fallback 为 0.
- sanity: stage3 checkpoint 没有 official longer-MQAR ref, 因此 `sanity_status=no_ref`; 这不是 invalid.

## 报告

人读总结见 `docs/20260529-flash-local-window-fairness-eval-only-report.md`.
