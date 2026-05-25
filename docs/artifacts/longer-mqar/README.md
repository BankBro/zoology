# Longer-MQAR eval-only canonical ledger

This directory stores the canonical ledger for checkpoint-only MQAR length extrapolation. It is intentionally separate from Flash-VQG and GDN training canonical ledgers.

## Files

- `longer-mqar-eval-summary.csv`: one row per `checkpoint x MQAR slice x eval protocol x hardware/software environment x eval attempt`.
- Runner 路径: `zoology/experiments/flash_vqg/scripts/20260521-longer-mqar-canonical/longer_mqar_eval_runner.py`. 本 artifact 目录只保存生成后的 ledger 和 manifest, 不保存长期维护的执行脚本.

## Row Semantics

- `run_type` must be `longer_mqar_eval_only` for normal rows.
- `eval_event_id` is the stable row identity for one eval event.
- `eval_batch_id` groups events produced by the same eval batch, for example `20260520-longer-mqar-eval`.
- `source_*` fields describe the checkpoint source training run and checkpoint identity.
- `source_ckpt_sha256` is the content identity of the checkpoint. Paths may change across machines; hashes should not.
- `source_train_config_sha256` records the training config content used to produce the checkpoint.
- `source_dynamic_capacity_total` is normalized to the active dynamic-memory capacity used for Flash/GDN comparison, not a multi-layer cumulative model-state total. In current Hybrid configs there is one active FlashVQG or GDN mixer layer plus one BaseConv layer, so GDN `source_dynamic_capacity_total` equals `source_dynamic_capacity_per_layer`.
- If a future run uses multiple active FlashVQG/GDN mixer layers, add an explicit active-layer count field before deriving any cumulative capacity. Do not silently multiply by `model.n_layers`, because Hybrid may include non-Flash/GDN layers.
- `eval_protocol_id` identifies the eval dataset/protocol/slice/sample count.
- `eval_hardware_profile_id` identifies the hardware/software profile under which the eval event ran.
- `eval_batch_size` is an execution/throughput setting, not a task definition. Accuracy comparisons should use matching `source_ckpt_sha256`, `eval_protocol_id`, and completed status.
- Wall-clock and peak memory should only be compared within the same or comparable `eval_hardware_profile_id`.
- Completed rows have `eval_status=completed`. Failed, OOM, or missing-checkpoint rows may be appended with `eval_status!=completed`, but should not be included in completed-only aggregate comparisons.

## Batch Search

Adaptive batch search is hardware dependent. Future adaptive runs should record:

- `batch_search_status`
- `batch_search_slice`
- `batch_search_candidates`
- `batch_search_best_eval_batch_size`
- `batch_search_peak_memory_mb`
- `batch_search_hardware_profile_id`
- `batch_search_reusable_scope=same_gpu_same_dtype_same_runner_only`

The initial `20260520` backfill did not perform recorded adaptive batch search, so those rows use `batch_search_status=not_recorded`.

## Current Backfill

The initial rows were backfilled from `docs/artifacts/20260520-longer-mqar-eval/` artifacts. Existing formal/smoke results used `vocab_size=8192`, `num_passes=1`, `random_non_queries=true`, and `power_a=0.01`. Hardware profile fields were backfilled from the current host's `nvidia-smi` output and are marked with `eval_hardware_backfill_status=current_host_inferred`.

## Proposal Defense Slide 10 Aggregate

The proposal defense slide 10 table is derived from `longer-mqar-eval-summary.csv` with the following aggregation rule:

- Use completed formal eval rows only.
- Use `source_scope=b64_ga4_fp32_official`.
- De-duplicate repeated eval attempts by averaging rows with the same `source_ckpt_sha256 x input_seq_len x num_kv_pairs`.
- Compute the displayed mean and population std across de-duplicated checkpoints.
- Single-checkpoint rows report `(-)` for std.

| config | role | active cap | params | n | seeds | 1024x256 | 2048x512 | 4096x1024 | 8190x2047 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| Flash cb256-r10 | practical strong | 327,680 | 1.184M | 4 | 123,124,125,126 | 0.9137 ± 0.0830 | 0.7081 ± 0.2362 | 0.4590 ± 0.2716 | 0.2327 ± 0.1654 |
| Flash cb64-r16 | 131k Flash best | 131,072 | 1.160M | 1 | 123 | 0.9691 (-) | 0.8230 (-) | 0.4689 (-) | 0.1622 (-) |
| GDN h2-ev10 | GDN best candidate | 81,920 | 1.435M | 5 | 123,124,125,126,127 | 0.8360 ± 0.0154 | 0.3478 ± 0.0175 | 0.0772 ± 0.0147 | 0.0063 ± 0.0026 |
| GDN h2-ev8 | stable GDN baseline | 65,536 | 1.368M | 5 | 123,124,125,126,127 | 0.8291 ± 0.0128 | 0.3616 ± 0.0114 | 0.0972 ± 0.0132 | 0.0108 ± 0.0038 |
| GDN h2-ev16 | 131k GDN capacity-matched | 131,072 | 1.635M | 3 | 123,124,125 | 0.7974 ± 0.0676 | 0.3554 ± 0.0484 | 0.1067 ± 0.0161 | 0.0154 ± 0.0025 |
