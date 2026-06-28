# 20260629-02-flash-vqg-gate-topk-mstate-causal-probe artifact

本 artifact 包含 Flash-VQG `gd_residual_v1` gate/top-k/M-state causal probe 的 17-step trace summary 和追加 1ep effect screen summary.

- `trace-summary.csv`: per-machine trace hashes and summaries.
- `cross-machine-trace-comparison.csv`: 2080ti vs 3090 joined trace comparison.
- `gate-comparison-summary.csv`: focused gate/logf/state/read/pred comparison rows.
- `variant-summary.csv`: first mismatch and step-16 match status by variant.
- `preflight-summary.csv`: cache/init/batch-order/code preflight evidence.
- `source-manifest.csv`: raw JSON evidence hashes.
- `effect-screen-summary.csv`: `constant-logf-f0.95` and `dense-read` 1ep effect screen metrics.
- `effect-screen-source-manifest.csv`: mirrored 1ep `config.json`, `result.json`, and `stdout.log` hashes.
- `metadata.json`: first mismatch metadata and 1ep collection metadata.

核心 1ep 结论:

- `constant-logf-f0.95`: 1024x256 hard slice almost zero on both machines, diagnostic only.
- `dense-read`: 1024x256 is `0.892` on 2080ti and `0.894` on 3090, gap `0.002`, within the 4pp tolerance.
