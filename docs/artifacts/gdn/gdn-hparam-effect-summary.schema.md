# GDN Hparam Effect Summary Schema

Canonical ledger:

- `docs/artifacts/gdn/gdn-hparam-effect-summary.csv`

Scope:

- Records completed GDN MQAR runs that reached the intended final epoch/checkpoint.
- Smoke, debug, failed, interrupted, or partial runs should stay out of this table unless a report explicitly promotes them.
- Direct quality comparisons must use rows with the same `comparison_scope`, `dtype_comparison_scope`, `batch_accum_profile`, `configured_max_epochs`, `early_stopping_disabled`, seed/data policy, and relevant model hparams.

Required identity fields:

- `summary_scope`: e.g. `epoch4_final_only`.
- `comparison_scope`: the official comparison group, including dtype and batch profile when relevant.
- `model_family`, `config_family`, `config`.
- `num_heads`, `expand_v`, `use_gate`, `use_short_conv`, `conv_size`.
- `seed`, `data_seed`.
- `run_id`, `launch_id`, `swanlab_url`, `swanlab_run_dir`.
- `zoology_branch`, `zoology_commit`.

Required training profile fields:

- `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `effective_train_batch_size`, `batch_accum_profile`.
- `configured_max_epochs`, `max_epochs_source`, `final_epoch`.
- `validations_per_epoch`, `final_validation_index`, `final_validation_phase`.
- `checkpoint_label`, `early_stopping_disabled`.
- `replicate_id`, `run_type`, `baseline_role`.

Required dtype fields:

- `dtype_policy`: user-facing dtype policy for the row.
- `outer_model_dtype`: dtype for the outer model path.
- `hidden_states_dtype`: dtype observed on hidden states around the GDN path when available.
- `kernel_input_dtype`: dtype entering the attention/mixer/kernel core path.
- `actual_kernel_dtype`: actual runtime GDN kernel dtype when available.
- `gdn_kernel_dtype_policy`: explicit `GDN_KERNEL_DTYPE` policy normalized as a dtype value.
- `dtype_comparison_scope`: comparison-safe dtype group, e.g. `float32_only` or `bf16_only`.
- `official_scope`: normalized official/probe/search grouping for safe filtering.
- `metadata_verification_level`: evidence strength for dtype/scope metadata.
- `train_config_path`: local `train_config.json` path used to verify run configuration.
- `metadata_backfill_status`: whether metadata is native artifact metadata or normalized/backfilled from existing artifacts.
- `gpu`, `gpu_name`, `gpu_compute_capability`.

Normalized values:

- `batch_accum_profile`: use `b64_ga4`, `b128_ga2`, or `b256_ga1`. Do not use legacy aliases like `64x4` or `128x2`.
- `dtype_policy`: use `float32`, `float16`, `bfloat16`, `auto`, `input`, or `unknown`. Do not use strings like `GDN_KERNEL_DTYPE=float32`; put the GDN kernel policy in `gdn_kernel_dtype_policy`.
- `outer_model_dtype`, `hidden_states_dtype`, `kernel_input_dtype`, `actual_kernel_dtype`, and `gdn_kernel_dtype_policy`: use `float32`, `float16`, `bfloat16`, `not_applicable`, `not_recorded`, or `unknown`.
- `dtype_comparison_scope`: use `float32_only`, `auto_or_mixed_dtype_probe`, or `unknown_dtype_legacy`.
- `comparison_scope`: use normalized comparison groups such as `b64_ga4_fp32_official`, `b128_ga2_fp32_capacity_search`, `b64_ga4_fp32_historical_probe`, or `unknown`.
- `official_scope`: use `b64_ga4_fp32_official`, `capacity_search_b128_ga2`, `dtype_probe`, `failed_or_incomplete`, or `unknown`.
- `metadata_verification_level`: use `verified_runtime_artifact`, `verified_artifact`, `verified_train_config`, `inferred_default_no_amp`, `reported_only`, or `unknown`.
- `metadata_backfill_status`: use `native_artifact_metadata_normalized`, `normalized_from_existing_artifact_or_report`, `normalized_from_existing_ledger`, `not_backfilled`, or `unknown`.

Interpretation:

- `comparison_scope` identifies comparable GDN experiment groups. `official_scope` is the normalized high-level filter for official/probe/search rows.
- `dynamic_state_capacity` is the active GDN recurrent-state capacity for the GDN mixer layer only, computed as `num_heads * head_k_dim * head_v_dim`. It is not multiplied by `model.n_layers` and does not include the BaseConv layer in Hybrid configs.
- `dtype_policy=float32` means the row is treated as float32 policy. `gdn_kernel_dtype_policy=float32` means the GDN kernel was explicitly configured or promoted as fp32 policy.
- `actual_kernel_dtype=float32` means the GDN runtime/kernel dtype is recorded as float32 in the source artifact/report path for that row.
- `b128_ga2_fp32_capacity_search` rows should not be mixed into `b64_ga4_fp32_official` quality comparisons.

Required result fields:

- `status`, `trainable_params`, `dynamic_state_capacity`.
- `elapsed_sec`, `wall_clock`, `oom`, `observed_peak_memory_used_mib`, `peak_memory_total_mib`.
- `valid_loss`, `valid_accuracy`.
- `valid_mqar_case_accuracy_1024x256`, `valid_mqar_case_accuracy_512x128`.
- `valid_input_seq_len_accuracy_512`, `valid_input_seq_len_accuracy_1024`.
- `valid_num_kv_pairs_accuracy_128`, `valid_num_kv_pairs_accuracy_256`.

Source fields:

- `source_artifact`: artifact CSV/JSON used to populate the row.
- `source_precision`: `full_precision` when copied from raw analysis/artifact values; otherwise describe rounding.
- `source_run_set`: short experiment set id.
- `note`: concise caveats, promotion notes, or direct comparison constraints.
