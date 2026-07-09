# 20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen

Default-dropout cb64-r16 support-confidence screen. Both machines used the same canonical MQAR cache, seed124 init checkpoint, and batch order. Formal training used one epoch per run with heavy read/hash/D-geometry traces disabled.

Core outputs:

- `run-summary.csv`: per-machine per-run final and best metrics parsed from logs.
- `cross-machine-comparison.csv`: paired 2080ti vs 3090 comparison by seed and variant.
- `variant-summary.csv`: two-seed pass/fail summary by support-confidence variant.
- `mechanism-metrics-summary.csv`: final residual read/write/state scalar metrics.
- `cache-init-preflight-summary.csv`: cache and init hash checks.
- `batch-order-summary.csv`: batch-order checks.
- `source-manifest.csv`: mirrored raw evidence paths and hashes/counts.

Screen rule: both machines final 1024x256 accuracy >= 0.85 and paired gap <= 4 percentage points.
