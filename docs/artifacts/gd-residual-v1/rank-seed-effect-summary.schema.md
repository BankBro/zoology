# gd_residual_v1 rank-seed effect summary schema

`rank-seed-effect-summary.csv` is the canonical append-only run-level ledger for gd_residual_v1 rank/seed experiments.

Rules:

- One complete training run per row. Repeats, same-GPU repeats, and cross-GPU repeats are appended as new rows.
- Do not overwrite an existing row unless correcting a documented extraction error.
- Use `replicate_id` to distinguish repeated runs: `orig`, `repeat1`, `repeat2`, `crossgpu1`, etc.
- Use `run_type` to distinguish run intent: `capacity_sweep`, `anchor`, `original`, `same_gpu_repeat`, `cross_gpu_repeat`.
- Use `configured_max_epochs`, `final_epoch`, `final_validation_index`, `final_validation_phase`, and `checkpoint_label` to separate epoch4, epoch32, or future training-length comparisons.
- Use `num_codebook_vectors` to record codebook size. The current 2026-05-18 seed/rank rows are all `256`.
- Use `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `effective_train_batch_size`, and `batch_accum_profile` to record the batch/accumulation training profile. The current epoch4 official rows use `64`, `16`, `4`, `256`, and `b64_ga4`.
- Use the normalized metadata columns `dtype_policy`, `outer_model_dtype`, `hidden_states_dtype`, `kernel_input_dtype`, `actual_kernel_dtype`, `dtype_comparison_scope`, `official_scope`, `metadata_verification_level`, `train_config_path`, and `metadata_backfill_status` for dtype/scope filtering. Do not encode verification strength in the dtype value itself.
- Keep `source_artifact`, `source_run_set`, `run_id`, and `swanlab_url` populated so every row can be traced back to its source.
- If a source artifact is rounded, mark `source_precision=reported_rounded`; otherwise use `full_precision`.

Normalized values:

- `batch_accum_profile`: use `b64_ga4`, `b128_ga2`, or `b256_ga1`. Do not use legacy aliases like `64x4` or `128x2`.
- `dtype_policy`: use `float32`, `float16`, `bfloat16`, `auto`, `input`, or `unknown`. Do not use aliases like `fp32`, `default_float32_on_2080ti`, or `float32_inferred`; put evidence strength in `metadata_verification_level`.
- `outer_model_dtype`, `hidden_states_dtype`, and `kernel_input_dtype`: use `float32`, `float16`, `bfloat16`, `not_applicable`, `not_recorded`, or `unknown`.
- `actual_kernel_dtype`: use `float32`, `float16`, `bfloat16`, `not_applicable`, `not_recorded`, or `unknown`. For historical Flash-VQG rows without runtime dtype telemetry, use `not_recorded`.
- `dtype_comparison_scope`: use `float32_only`, `auto_or_mixed_dtype_probe`, or `unknown_dtype_legacy`.
- `official_scope`: use `b64_ga4_fp32_official`, `b64_ga4_fp32_historical_inferred`, `batch_accum_probe`, `rank_search_b128_ga2`, `dtype_probe`, `failed_or_incomplete`, or `unknown`.
- `metadata_verification_level`: use `verified_runtime_artifact`, `verified_artifact`, `verified_train_config`, `inferred_default_no_amp`, `reported_only`, or `unknown`.
- `metadata_backfill_status`: use `native_artifact_metadata_normalized`, `normalized_from_existing_artifact_no_actual_kernel_dtype`, `backfilled_from_train_config_and_code_inference`, `not_backfilled`, or `unknown`.

Interpretation:

- `comparison_scope` is the historical experiment grouping label. Use `official_scope` for normalized official/probe/search filtering.
- `dynamic_capacity` is the active Flash-VQG `gd_residual_v1` residual-memory capacity for the FlashVQG mixer layer, computed from `num_codebook_vectors * rank * d_model`. It is not a sum over `model.n_layers` and does not include the BaseConv layer in Hybrid configs.
- `dtype_policy=float32` means the run is treated as float32 policy. If `metadata_verification_level=inferred_default_no_amp`, this was inferred from train config plus the no-AMP training path rather than measured runtime dtype telemetry.
- `actual_kernel_dtype=not_recorded` must not be silently interpreted as a measured `float32` kernel. It only means no separate runtime kernel dtype telemetry was recorded for that row.
- `b64_ga4_fp32_historical_inferred` rows can be useful for historical analysis, but stricter direct official comparisons should prefer `official_scope=b64_ga4_fp32_official` unless the report explicitly accepts inferred historical metadata.

Current scope:

- `summary_scope=epoch4_final_only`
- `comparison_scope=gd_residual_v1_cb256_noearly4ep`
- `train_batch_size=64`
- `eval_batch_size=16`
- `gradient_accumulation_steps=4`
- `effective_train_batch_size=256`
- `batch_accum_profile=b64_ga4`
- `configured_max_epochs=4`
- `final_epoch=4`
- `final_validation_index=8`
- `final_validation_phase=epoch_end`
- `checkpoint_label=epoch4_noearly`
- `early_stopping_disabled=true`
