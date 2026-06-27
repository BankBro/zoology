# 20260628-01-flash-vqg-stability-ablation

本 artifact 收尾 no-embed-dropout 1 epoch screen. 本轮是 diagnostic / exploratory, 不写 official MQAR ledger.

## 文件

- `run-summary.csv`: per-run final metrics.
- `cross-machine-comparison.csv`: 以 2080ti run 为参考的 1024x256 gap.
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.
- `queue-summary.csv`: queue 状态.
- `invalid-runs.csv`: failed/interrupted/pending run.
- `source-manifest.csv`: raw evidence 路径和 sha256.
- `metadata.json`: 收尾元数据.
