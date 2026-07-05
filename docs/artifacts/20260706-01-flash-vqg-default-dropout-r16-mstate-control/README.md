# 20260706-01-flash-vqg-default-dropout-r16-mstate-control

本 artifact 收尾 default-dropout fixed-r16 M_state control paired 1ep screen. 本轮固定 `read_topk=16`, `write_topk=4`, `embed_dropout=0.1`, canonical cache/init/batch order, 并关闭 read trace, train inline event trace 和 shadow dense read. 测试项包括 baseline, smooth update softcap=0.5, hard M_state norm cap=6.0, 以及 update softcap + residual injection warmup 0->512 optimizer steps.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.
- `mechanism-metrics-summary.csv`: final validation residual memory/read/write metrics parsed from logs.
- `early-window-summary.csv`: train-step scalar metrics if available.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence.

Collection notes:

- `run-summary.csv` and `cross-machine-comparison.csv` include only the two formal queues `mstate-2080ti-gpu0-20260705T074648Z` and `mstate-3090-gpu0-20260705T074648Z`. Earlier smoke outputs were excluded.
- `results/*.json` has `train_result=null`; final/best metrics were parsed from formal logs. tqdm may redraw the same validation summary multiple times in raw logs, so collectors deduplicate validation summaries before reporting final/best values.
