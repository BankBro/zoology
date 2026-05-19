# gd_residual_v1 rank-seed effect summary schema

`rank-seed-effect-summary.csv` is the canonical append-only run-level ledger for gd_residual_v1 rank/seed experiments.

Rules:

- One complete training run per row. Repeats, same-GPU repeats, and cross-GPU repeats are appended as new rows.
- Do not overwrite an existing row unless correcting a documented extraction error.
- Use `replicate_id` to distinguish repeated runs: `orig`, `repeat1`, `repeat2`, `crossgpu1`, etc.
- Use `run_type` to distinguish run intent: `capacity_sweep`, `anchor`, `original`, `same_gpu_repeat`, `cross_gpu_repeat`.
- Use `configured_max_epochs`, `final_epoch`, `final_validation_index`, `final_validation_phase`, and `checkpoint_label` to separate epoch4, epoch32, or future training-length comparisons.
- Use `num_codebook_vectors` to record codebook size. The current 2026-05-18 seed/rank rows are all `256`.
- Use `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `effective_train_batch_size`, and `batch_accum_profile` to record the batch/accumulation training profile. The current epoch4 official rows use `64`, `16`, `4`, `256`, and `64x4`.
- Keep `source_artifact`, `source_run_set`, `run_id`, and `swanlab_url` populated so every row can be traced back to its source.
- If a source artifact is rounded, mark `source_precision=reported_rounded`; otherwise use `full_precision`.

Current scope:

- `summary_scope=epoch4_final_only`
- `comparison_scope=gd_residual_v1_cb256_noearly4ep`
- `train_batch_size=64`
- `eval_batch_size=16`
- `gradient_accumulation_steps=4`
- `effective_train_batch_size=256`
- `batch_accum_profile=64x4`
- `configured_max_epochs=4`
- `final_epoch=4`
- `final_validation_index=8`
- `final_validation_phase=epoch_end`
- `checkpoint_label=epoch4_noearly`
- `early_stopping_disabled=true`
