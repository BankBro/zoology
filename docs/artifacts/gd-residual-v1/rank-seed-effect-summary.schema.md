# gd_residual_v1 rank-seed effect summary schema

`rank-seed-effect-summary.csv` is the canonical append-only run-level ledger for gd_residual_v1 rank/seed experiments.

Rules:

- One complete training run per row. Repeats, same-GPU repeats, and cross-GPU repeats are appended as new rows.
- Do not overwrite an existing row unless correcting a documented extraction error.
- Use `replicate_id` to distinguish repeated runs: `orig`, `repeat1`, `repeat2`, `crossgpu1`, etc.
- Use `run_type` to distinguish run intent: `capacity_sweep`, `anchor`, `original`, `same_gpu_repeat`, `cross_gpu_repeat`.
- Use `configured_max_epochs`, `final_epoch`, `final_validation_index`, `final_validation_phase`, and `checkpoint_label` to separate epoch4, epoch32, or future training-length comparisons.
- Use `num_codebook_vectors` to record codebook size. The current 2026-05-18 seed/rank rows are all `256`.
- Keep `source_artifact`, `source_run_set`, `run_id`, and `swanlab_url` populated so every row can be traced back to its source.
- If a source artifact is rounded, mark `source_precision=reported_rounded`; otherwise use `full_precision`.

Current scope:

- `summary_scope=epoch4_final_only`
- `comparison_scope=gd_residual_v1_cb256_noearly4ep`
- `configured_max_epochs=4`
- `final_epoch=4`
- `final_validation_index=8`
- `final_validation_phase=epoch_end`
- `checkpoint_label=epoch4_noearly`
- `early_stopping_disabled=true`
