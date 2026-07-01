# 20260702-01-flash-vqg-update-norm-event-trace-probe

本 artifact 收尾 default-dropout update_norm event trace diagnostic. 本轮只比较 `baseline-r2` 与 `ucap0p5-r2`, 用事件级 trace 观察 top residual update, 不把 hard cap 当作最终机制方案, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `read_topk=2`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, `max_train_steps=704`.

注意: `update_event_trace.jsonl` 来自指定 optimizer step 上的 fixed validation batch eval forward, 用于观察同一训练进度下的机制快照, 不是直接记录实际 training minibatch.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `variant-gap-summary.csv`: 2080ti/3090 final hard gap by variant.
- `cap-metrics-summary.csv`: step704 cap hit, update norm, M norm, lambda/inject metrics.
- `early-window-summary.csv`: train-step eval read/write scalar metrics.
- `update-event-step-summary.csv`: per-step/layer top update event summary.
- `update-event-trace-summary.csv`: per-machine/variant/layer event aggregate.
- `update-event-cross-machine-summary.csv`: paired event aggregate comparison.
- `update-event-top.csv`: global top 512 event rows.
- `read-trace-cross-machine-summary.csv`: 2080ti/3090 read support match summary.
- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence.
