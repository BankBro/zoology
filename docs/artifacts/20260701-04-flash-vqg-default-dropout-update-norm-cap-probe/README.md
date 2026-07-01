# 20260701-04-flash-vqg-default-dropout-update-norm-cap-probe

本 artifact 收尾 default-dropout update_norm_cap diagnostic probe. 本轮只测试现有 hard update cap 是否能缓解 default-r2 跨机器 1ep 分叉, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `read_topk=2`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, `max_train_steps=704`.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `variant-gap-summary.csv`: 2080ti/3090 final hard gap by variant.
- `cap-metrics-summary.csv`: step704 cap hit, update norm, M norm, lambda/inject metrics.
- `early-window-summary.csv`: train-step eval read/write scalar metrics.
- `read-trace-cross-machine-summary.csv`: 2080ti/3090 read support match summary.
- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.
- `first-mismatch-summary.csv`: first cross-machine mismatch by target.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `queue-summary.csv`: queue status.
- `source-manifest.csv`: mirrored lightweight raw evidence.

注意: `update_norm_cap` 使用 detached scale, 是 diagnostic hard cap, 不是最终机制方案.
