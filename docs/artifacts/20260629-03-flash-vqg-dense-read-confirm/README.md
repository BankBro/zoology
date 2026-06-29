# 20260629-03-flash-vqg-dense-read-confirm

本 artifact 收尾 dense-read 4 epoch confirm. 本轮是 diagnostic / confirm screen, 不写 official MQAR ledger.

关键配置: `fox_remote_read_topk=64`, `num_codebook_vectors=64`, `fox_gd_residual_dense_read_chunked=True`, no-dropout, canonical cache/init.

## 文件

- `run-summary.csv`: per-run final metrics.
- `cross-machine-comparison.csv`: 以 2080ti run 为参考的 1024x256 gap.
- `pairwise-repeat-comparison.csv`: r1/r2 成对跨机器 final/best gap.
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.
- `queue-summary.csv`: queue 状态.
- `invalid-runs.csv`: failed/interrupted/pending run.
- `source-manifest.csv`: raw evidence 路径和 sha256.
- `metadata.json`: 收尾元数据.

注: `n_validation_summaries` 对 tqdm 相邻重复 summary 做了去重; `n_validation_summary_lines` 保留原始日志匹配行数. `best_*` 是日志观测到的 best validation metric, 不是 saved-best checkpoint 复评.
