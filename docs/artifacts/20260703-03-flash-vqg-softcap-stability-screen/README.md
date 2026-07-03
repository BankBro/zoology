# 20260703-03-flash-vqg-softcap-stability-screen

本 artifact 收尾 default-dropout smooth cap 1ep stability screen. 本轮保持 `read_topk=2`, `write_topk=4`, `embed_dropout=0.1`, canonical cache/init, 并明确关闭 read trace. Smooth cap 使用 `scale=(1+(x/cap)^4)^(-1/4)`, cap=0.5.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.
- `softcap-metrics-summary.csv`: step704 softcap hit/scale and residual metrics.
- `early-window-summary.csv`: train-step scalar metrics.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence.
