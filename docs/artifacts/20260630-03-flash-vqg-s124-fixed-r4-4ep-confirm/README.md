# 20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm

本 artifact 收尾 `seed=124` fixed-r4 4 epoch confirm. 本轮是 diagnostic / exploratory confirm, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, train-time `read_topk=4`, no-dropout, canonical MQAR cache, seed124 canonical init, `max_epochs=4`.

## 文件

- `run-summary.csv`: per-run final/best metrics.
- `variant-summary.csv`: fixed-r4 的 2080ti/3090 成对结果.
- `cross-machine-comparison.csv`: fixed-r4 的 1024x256 cross-machine gap.
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.
- `queue-summary.csv`: queue 状态.
- `invalid-runs.csv`: failed/interrupted/pending run.
- `source-manifest.csv`: mirrored raw evidence 路径和 sha256.
- `metadata.json`: 收尾元数据.

注: `best_*` 是日志观测到的 best validation metric, 不是 saved-best checkpoint 复评. 本轮仍是 no-dropout confirm, 不能回答 dropout 加回后的稳定性。
