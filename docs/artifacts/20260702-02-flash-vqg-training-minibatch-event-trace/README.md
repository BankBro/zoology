# 20260702-02-flash-vqg-training-minibatch-event-trace

本 artifact 收尾 default-dropout training-minibatch residual event trace diagnostic. 本轮只比较 `baseline-r2` 与 `ucap0p5-r2`, 在真实训练 forward 中观察 top residual update, 不把 hard cap 当作最终机制方案, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `read_topk=2`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, `max_train_steps=704`.

注意: `train-inline-*` 文件来自真实 training minibatch forward. `early-window-*` 和 `read-trace-*` 仍是指定训练进度上的 fixed validation batch eval snapshot, 两类证据不能混用.

训练 forward 在 optimizer step 递增前发生, 因此 inline `train_step=703` 表示产生第 704 次 optimizer update 的训练窗口.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `variant-gap-summary.csv`: 2080ti/3090 final hard gap by variant.
- `cap-metrics-summary.csv`: step704 cap hit, update norm, M norm, lambda/inject metrics.
- `early-window-summary.csv`: train-step eval read/write scalar metrics.
- `train-inline-event-step-summary.csv`: real training minibatch per-step/layer top update event summary.
- `train-inline-event-micro-summary.csv`: real training minibatch per-step/micro/layer event summary.
- `train-inline-event-trace-summary.csv`: real training minibatch per-machine/variant/layer aggregate.
- `train-inline-event-cross-machine-summary.csv`: paired inline event aggregate comparison.
- `cap-hit-timeline.csv`: cap hit and scale timeline from real training minibatches.
- `code-head-hotspot-summary.csv`: top event concentration by layer/head/code.
- `train-inline-event-top.csv`: global top 512 inline event rows.
- `read-trace-cross-machine-summary.csv`: 2080ti/3090 read support match summary.
- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence.
