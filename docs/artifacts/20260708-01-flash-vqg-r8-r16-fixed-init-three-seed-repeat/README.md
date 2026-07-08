# 20260708-01 Flash-VQG R8/R16 fixed-init three-seed repeat

This artifact contains formal-only results for the fixed canonical init repeat screen.

Scope:

- Machines: 2080ti GPU1 and 3090 GPU0.
- Formal launch timestamp: 20260707T172212Z.
- Completed formal runs: 24/24.
- Training seeds: 123, 124, 125.
- Repeats: 1, 2.
- Read top-k: 8, 16.
- Fixed model initialization: canonical seed124 checkpoint.
- Data seed and MQAR cache: canonical data_seed=123 cache, content hash verified.
- Heavy read trace, hash probe, train inline event trace, and D-geometry trace were disabled for formal training.

Main files:

- `run-summary.csv`: per-run final and best metrics, formal only.
- `cross-machine-comparison.csv`: paired 2080ti vs 3090 final 1024x256 comparison for each seed/read_topk/repeat.
- `variant-seed-repeat-summary.csv`: four-run summary for each training seed and read_topk.
- `within-machine-repeat-summary.csv`: same-machine repeat spread.
- `variant-summary.csv`: paired result plus residual read/write/state scalar metrics.
- `mechanism-metrics-summary.csv`: per-run residual read/write/state metrics parsed from final validation logs.
- `cache-init-preflight-summary.csv`: MQAR cache and canonical init verification.
- `batch-order-summary.csv`: batch order hashes, confirming matched order for paired runs.
- `formal-ledger.csv`: formal MQAR ledger entries.
- `source-manifest.csv`: mirrored logs/config/result/hash evidence with sha256.
- `metadata.json`: experiment metadata and summary.

Screen rule:

A pair passes only when both machines have final 1024x256 accuracy >= 0.85 and the paired gap is <= 4 percentage points.

Key outcome:

- read_topk=8: 0/6 paired runs pass.
- read_topk=16: 2/6 paired runs pass.
- No seed/read_topk group passes the stricter 2 machines x 2 repeats stability rule.
