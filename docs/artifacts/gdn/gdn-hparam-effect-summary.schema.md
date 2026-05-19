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
- `kernel_input_dtype`: dtype entering the attention/mixer/kernel core path.
- `dtype_comparison_scope`: comparison-safe dtype group, e.g. `float32_only` or `bf16_only`.
- `gpu`, `gpu_name`, `gpu_compute_capability`.

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
