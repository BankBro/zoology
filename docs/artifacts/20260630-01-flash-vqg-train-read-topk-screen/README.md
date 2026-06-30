# 20260630-01-flash-vqg-train-read-topk-screen

本 artifact 收尾训练期 read_topk 稳定性 1 epoch screen. 本轮是 diagnostic screen, 不写 official MQAR ledger.

共同配置: `seed=123`, `data_seed=123`, `cb64-r16`, `write_topk=4`, no-dropout, canonical cache/init, 1 epoch. 变量只有 train-time read_topk 策略.

Variants: `fixed-r2-baseline`, `fixed-r4`, `sched16to4`, `sched64to4`. `sched*to4` 使用 `linear_int` 在 train forward step `0..448` 逐步收缩到 4.

## 文件

- `run-summary.csv`: per-run final metrics.
- `variant-summary.csv`: 每个 variant 的 2080ti/3090 成对结果.
- `cross-machine-comparison.csv`: 每个 variant 的 1024x256 cross-machine gap.
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.
- `queue-summary.csv`: queue 状态.
- `invalid-runs.csv`: failed/interrupted/pending run.
- `source-manifest.csv`: raw evidence 路径和 sha256.
- `metadata.json`: 收尾元数据.

注: `n_validation_summaries` 对 tqdm 相邻重复 summary 做了去重; `n_validation_summary_lines` 保留原始日志匹配行数. `best_*` 是日志观测到的 best validation metric, 不是 saved-best checkpoint 复评.
