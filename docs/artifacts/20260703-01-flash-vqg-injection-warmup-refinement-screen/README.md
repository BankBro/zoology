# 20260703-01-flash-vqg-injection-warmup-refinement-screen

本 artifact 收尾 default-dropout residual injection warmup refinement 1ep screen. 本轮只控制 residual correction 注入到 `O_base` 的强度, 不改变 `M_state` build/write/read, 不改变 dropout 协议, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `read_topk=2`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, `max_train_steps=704` optimizer steps.

Warmup step 说明: Flash-VQG 内部使用 train-forward counter. 本轮 `gradient_accumulation_steps=4`, 所以 optimizer step 704 对应 train-forward step 2816, optimizer step 1024 对应 train-forward step 4096, optimizer step 32 对应 train-forward step 128.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `variant-gap-summary.csv`: 2080ti/3090 final hard gap by variant.
- `injection-warmup-summary.csv`: warmup factor, inject ratio, lambda and loss by train step.
- `early-window-summary.csv`: train-step eval read/write scalar metrics.
- `read-trace-cross-machine-summary.csv`: 2080ti/3090 read support match summary.
- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence.
