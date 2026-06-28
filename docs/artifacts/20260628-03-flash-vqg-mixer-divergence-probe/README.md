# 20260628-03-flash-vqg-mixer-divergence-probe artifact

This artifact contains the layer-1 Flash-VQG mixer divergence probe summaries.

First mismatch: optimizer step `0`, micro step `0`, layer `1`, `state_build/logf_all`. Step 0 phase1 q/k/v and VQ routing matched, so this probe points to the FoX gate/logf state-build path before read top-k diverges.

- `trace-summary.csv`: per-machine trace hashes and summaries.
- `cross-machine-trace-comparison.csv`: 2080ti vs 3090 joined trace comparison.
- `preflight-summary.csv`: cache/init/batch-order/code preflight evidence.
- `source-manifest.csv`: raw JSON evidence hashes.
- `metadata.json`: first mismatch metadata.
