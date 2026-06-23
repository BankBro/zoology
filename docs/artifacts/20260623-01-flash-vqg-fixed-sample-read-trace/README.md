# 20260623-01 Flash-VQG fixed-sample read trace artifact

This artifact contains lightweight CSV/JSON summaries and compressed JSONL read traces. Checkpoint `.pt` files are not copied into git artifacts.

Only the effective wave with launch suffix `2026-06-23-07-19-43` is included in the formal summaries. Earlier batch-0 waves were diagnostic only: they traced `64x4` short-slice validation batches where remote read was mostly masked out and `selected_mass=0`, so they are excluded from the formal artifact.

Files:

- `final.csv`: final checkpoint metrics.
- `final_best_metrics.csv`: final and best checkpoint metrics.
- `trace_summary.csv`: fixed-sample candidate summary across 16 validation steps.
- `trace_step_summary.csv`: per-step fixed-sample summary.
- `source_manifest.csv`: source file inventory and hashes.
- `metadata.json`: collection metadata.
- `raw_summary.json`: manifest, checkpoint metric, and log summaries.
- `trace_archives.csv`: compressed trace archive hashes.
- `traces/*.jsonl.gz`: compressed fixed-sample trace records.
