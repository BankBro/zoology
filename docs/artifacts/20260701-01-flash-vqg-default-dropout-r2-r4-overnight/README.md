# 20260701-01-flash-vqg-default-dropout-r2-r4-overnight

本 artifact 收尾 default-dropout fixed-r2/fixed-r4 overnight diagnostic. probe/失败/中断 run 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `resid_dropout=0`, `drop_path=0`.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `decision-summary.csv`: hand-curated decision table used by the report; use this instead of raw variant grouping for P0 decisions.
- `variant-summary.csv`: per-variant cross-machine summary.
- `early-window-summary.csv`: B2 train-step read trace scalar metrics.
- `read-trace-summary.csv`: fixed sample read trace aggregate.
- `read-trace-cross-machine-summary.csv`: 2080ti/3090 trace support match summary.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `queue-summary.csv`: queue status.
- `source-manifest.csv`: mirrored lightweight raw evidence.
