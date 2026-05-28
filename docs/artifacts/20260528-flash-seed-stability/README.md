# 20260528 Flash seed stability artifacts

This directory records the strict official Flash seed stability补跑 for the 131k capacity decomposition follow-up.

Scope:

- `data_seed=123`.
- `b64_ga4`: train batch 64, eval batch 16, gradient accumulation 4.
- fp32 official/default, 4 epochs, early stopping disabled.
- New completed targets: `cb256-r4-s124`, `cb256-r4-s125`, `cb64-r16-s124`, `cb64-r16-s125`.

Files:

- `flash-seed-stability-final.csv`: final run-level metrics used as the formal source artifact for the canonical ledger.
- `flash-seed-stability-source-manifest.csv`: raw manifest, analysis, checkpoint, SwanLab, and log paths for each run.

Canonical ledger rows are appended to `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv` with `source_artifact=docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv`.
