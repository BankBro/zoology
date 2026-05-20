# Flash-VQG historical b64_ga4 fp32 metadata audit
## Scope
- Audited canonical ledger rows from `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`.
- Included completed Flash-VQG gd_residual_v1 rows with `TRAIN_BATCH_SIZE=64`, `EVAL_BATCH_SIZE=16`, `GRADIENT_ACCUMULATION_STEPS=4`, excluding the later explicit r10 b64_ga4 fp32 official rows.
- This audit creates a sidecar metadata audit table. A follow-up normalization pass also extends the canonical ledger with standardized dtype/scope metadata columns.
- In historical launch names like `b64-ga4-eb16`, `eb16` means `eval_batch_size=16`, not fp16.

## Result
- Audited rows: `16`.
- `train_config.json` found and matching ledger values: `16/16`.
- Full precision final metrics already recoverable from committed artifacts: `11/16`.
- Report-precision-only rows: `5/16`. These are the 20260516 capacity sweep rows.
- Runtime dtype telemetry directly verified: `0/16`.
- Dtype policy can be normalized to `float32`, but runtime-measured actual dtype was not recorded for historical rows.
- Peak memory recovered from committed/run manifest evidence: `1/16`.

## Rows By Source
- `20260516_capacity_sweep`: `5` rows.
- `20260517_r4_r8_crossgpu`: `4` rows.
- `20260517_rank_stability_r16_robustness`: `4` rows.
- `20260517_same_seed_repeat`: `2` rows.
- `20260519_rank_gdn_capacity_up`: `1` rows.

## What Can Be Backfilled
- Training profile: `b64_ga4`, `TRAIN_BATCH_SIZE=64`, `EVAL_BATCH_SIZE=16`, `GRADIENT_ACCUMULATION_STEPS=4`, `effective_train_batch_size=256`.
- Training length and validation cadence: `MAX_EPOCHS=4`, `validations_per_epoch=2`, final epoch-end validation.
- Early stopping: disabled, verified by `early_stopping_metric=null` and `early_stopping_threshold=null` in `train_config.json`.
- Data/model basics: `DATA_SEED=123`, run `seed`, `DMODEL=128`, `n_layers=2`, `num_heads=2`, `num_codebook_vectors=256`.
- Flash-VQG GD/VQ hyperparams: `FOX_REMOTE_FORMULA=gd_residual_v1`, `FOX_REMOTE_READ_TOPK=2`, `FOX_GD_RESIDUAL_WRITE_TOPK=4`, `FOX_GD_RESIDUAL_BUILDER=grouped_chunk_torch_ref`, `FOX_GD_RESIDUAL_PACK_MODE=semivec_ref`, `FOX_GD_RESIDUAL_CHUNK_SIZE=64`, `FOX_GD_RESIDUAL_MU_MIN_COUNT=0.1`, `VQ_SCORE_MODE=codebook_dot`, `VQ_WEIGHT_MODE=dense_softmax`, `VQ_UPDATE_MODE=grad`, `VQ_SOFTMAX_TAU=0.25`, `VQ_TOPK=4`.
- Source artifact and train_config path for every audited row.

## What Cannot Be Strictly Backfilled
- Runtime-measured actual dtype was not logged for these historical Flash-VQG rows. The safe normalized values are `dtype_policy=float32`, `actual_kernel_dtype=not_recorded`, and `metadata_verification_level=inferred_default_no_amp`.
- Most historical peak memory values were not committed. Only the 20260519 r16 fallback row has peak memory in its run manifest.
- The 20260516 capacity sweep final metrics are stored at report precision in the canonical/source CSV, so final metrics cannot be losslessly upgraded to full precision from committed artifacts alone.

## Dtype Evidence
- Audited `train_config.json` files have no dtype or precision field.
- `zoology/train.py` train/test path calls `compute_loss` directly and does not wrap the Flash-VQG path in AMP/autocast or GradScaler.
- `zoology/model.py` sets token embedding default dtype to `torch.float32`; without `.half()`, `.bfloat16()`, autocast, or an explicit dtype override, the model follows PyTorch float32 defaults.
- Therefore these rows are compatible with a historical `b64_ga4 + inferred fp32` comparison scope, but should not be labeled as `actual_runtime_dtype_verified=float32`.

## Recommendation
- Do not overwrite existing metric values in canonical ledger rows. Metadata normalization should use explicit columns keyed by `run_id` and `source_run_set`.
- If we want to promote these rows into a broader official historical table, use normalized values like `dtype_policy=float32`, `actual_kernel_dtype=not_recorded`, `metadata_verification_level=inferred_default_no_amp`, and `official_scope=b64_ga4_fp32_historical_inferred`.
- Keep the current strict 20260519 r10 rows as the only Flash-VQG rows with explicit `b64_ga4 + fp32 official` reporting, unless we accept the inferred-dtype historical scope as a separate tier.

## Ledger Normalization
- `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv` was extended with normalized dtype/scope metadata columns after this audit.
- Standardized values now use `dtype_policy=float32`, not `float32_inferred`; evidence strength is represented by `metadata_verification_level`.
- Legacy `batch_accum_profile` values such as `64x4` and `128x2` were normalized to `b64_ga4` and `b128_ga2`.
- Historical Flash-VQG rows without runtime dtype telemetry use `actual_kernel_dtype=not_recorded` and `metadata_verification_level=inferred_default_no_amp`.

## Artifacts
- `docs/artifacts/20260520-flash-vqg-historical-b64-fp32-audit/flash-historical-b64-fp32-metadata-audit.csv`
- `docs/artifacts/20260520-flash-vqg-historical-b64-fp32-audit/flash-historical-b64-fp32-compatible-summary.csv`
- `docs/artifacts/20260520-flash-vqg-historical-b64-fp32-audit/flash-historical-b64-fp32-audit-manifest.json`
