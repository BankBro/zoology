# 20260630-02-flash-vqg-s124-readk4-gate

本 artifact 收尾 `seed=124` fixed read_topk gate 1 epoch screen. 本轮是 diagnostic screen, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, no-dropout, canonical MQAR cache, seed124 canonical init, 1 epoch. 变量只有 train-time read_topk: fixed 2 vs fixed 4.

## 文件

- `run-summary.csv`: per-run final metrics.
- `variant-summary.csv`: 每个 variant 的 2080ti/3090 成对结果.
- `cross-machine-comparison.csv`: 每个 variant 的 1024x256 cross-machine gap.
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.
- `queue-summary.csv`: queue 状态.
- `invalid-runs.csv`: failed/interrupted/pending run.
- `source-manifest.csv`: raw evidence 路径和 sha256.
- `metadata.json`: 收尾元数据.

注: `best_*` 是日志观测到的 best validation metric, 不是 saved-best checkpoint 复评.
