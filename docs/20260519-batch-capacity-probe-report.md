# Batch Capacity Probe Report

## Scope

This is a micro-batch capacity probe, not an official quality run. The goal was to check whether the upcoming runs can use `train_batch_size=128`, `gradient_accumulation_steps=2`, and a larger validation micro-batch on 2080 Ti GPUs.

Validation does not use gradient accumulation. The validation-side capacity knob is `EVAL_BATCH_SIZE`.

## Results

| model | probe | train batch | eval batch | ga | seq len | status | peak allocated | peak reserved |
|---|---|---:|---:|---:|---:|---|---:|---:|
| gd_residual_v1 r16 s126 | train fwd/bwd | 128 | 16 | 2 | 256 | OOM |  |  |
| gd_residual_v1 r16 s126 | train fwd/bwd | 64 | 16 | 4 | 256 | ok | 6746 MiB | 8624 MiB |
| gd_residual_v1 r16 s126 | eval fwd | 64 | 32 | 4 | 1024 | ok | 5675 MiB | 6968 MiB |
| gd_residual_v1 r16 s126 | eval fwd | 64 | 64 | 4 | 1024 | OOM |  |  |
| GDN dmodel128 s126 | train fwd/bwd | 128 | 32 | 2 | 256 | ok | 4428 MiB | 4820 MiB |
| GDN dmodel128 s126 | eval fwd | 128 | 64 | 2 | 1024 | ok | 4109 MiB | 4858 MiB |
| GDN dmodel128 s126 | eval fwd | 128 | 128 | 2 | 1024 | ok | 8206 MiB | 9566 MiB |
| GDN h2-ev8 dmodel128 s123 | train fwd/bwd | 256 | 64 | 1 | 256 | ok | 9420 MiB | 9876 MiB |
| GDN h2-ev8 dmodel128 s123 | eval fwd | 256 | 128 | 1 | 1024 | ok | 8207 MiB | 10462 MiB |

## Decision

- `gd_residual_v1` rank 16 should not use `128x2` on this 2080 Ti setup. For `r16-s126`, use `train_batch_size=64`, `gradient_accumulation_steps=4`, effective train batch `256`.
- `gd_residual_v1` rank 16 should use `EVAL_BATCH_SIZE=32` rather than 64. Eval batch 64 OOMs on the 1024-token hard slice.
- Lower-rank Flash-VQG runs can continue using `128x2` if they have already passed capacity/quality checks, but the run ledger must record train/eval batch and gradient accumulation.
- GDN can use `train_batch_size=128`, `gradient_accumulation_steps=2`. The largest planned `h2-ev8` GDN variant also passes `train_batch_size=256`, `gradient_accumulation_steps=1`, but train reserved memory reaches about 9876 MiB.
- GDN `h2-ev8` validation batch 128 passes, but it reserves about 10462 MiB, so use `EVAL_BATCH_SIZE=128` only if the GPU is otherwise idle. Use 64 for a wider memory margin.

Recommended immediate policy:

- Flash-VQG gd_residual_v1 rank 16: `TRAIN_BATCH_SIZE=64`, `GRADIENT_ACCUMULATION_STEPS=4`, `EVAL_BATCH_SIZE=32`.
- Flash-VQG gd_residual_v1 rank 8 and lower: `TRAIN_BATCH_SIZE=128`, `GRADIENT_ACCUMULATION_STEPS=2`, `EVAL_BATCH_SIZE=32`, pending per-config capacity checks if rank or model capacity increases.
- GDN: `TRAIN_BATCH_SIZE=256`, `GRADIENT_ACCUMULATION_STEPS=1` is feasible for the planned h2-ev8 worst case, but memory margin is narrow. Use `EVAL_BATCH_SIZE=64` for conservative runs, or `128` when validation throughput is the priority and the GPU is clean.
