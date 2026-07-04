# 20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen

本 artifact 收尾 default-dropout safe limiter read_topk=2/4 1ep screen. 本轮保持 `write_topk=4`, `embed_dropout=0.1`, canonical cache/init, 并明确关闭 read trace, train inline event trace 和 shadow dense read. 测试项包括 baseline, safe residual injection limit ratio=1.0/2.0, 以及 M_state update_norm hard cap 从 0.5 线性释放到 0.8/1.0 over 512 optimizer steps.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.
- `limiter-metrics-summary.csv`: step704/final limiter hit/scale and residual metrics.
- `early-window-summary.csv`: train-step scalar metrics.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence.
