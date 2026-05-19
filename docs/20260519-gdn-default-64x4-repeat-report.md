# GDN default 64x4 same-seed repeat report

Date: 2026-05-19

## Question

This run checks whether the current GDN default baseline script reproduces the 4 epoch `64x4` no-early-stop result under the same seed and data seed.

Tested profile:

- model: `gated_delta_net`, `use_gate=false`
- run_type: `same_seed_repeat`
- seed: `123`
- data_seed: `123`
- max_epochs: `4`
- early stopping: disabled
- train_batch_size: `64`
- gradient_accumulation_steps: `4`
- effective_train_batch_size: `256`
- eval_batch_size: `16`
- GPU: `0`

Run:

- launch_id: `flash-vqg-20260519-gdn-default-64x4-repeat-2026-05-19-03-29-43`
- run_id: `gated_delta_net-default-s123-d123-noearly4ep-b64-ga4-repeat`

## Result

The run completed without OOM. It closely reproduces the earlier `use_gate=false` noearly4ep GDN row from 20260515.

| metric | 20260515 usegate0 noearly4ep | 64x4 repeat | delta |
|---|---:|---:|---:|
| valid/loss | 0.345000 | 0.343976 | -0.001024 |
| valid/accuracy | 0.962000 | 0.962484 | +0.000484 |
| 1024x256 | 0.711000 | 0.712516 | +0.001516 |

This indicates no meaningful same-seed same-profile drift for the current `use_gate=false` GDN noearly4ep setup.

## Epoch Metrics

| epoch | valid/loss | valid/accuracy | 1024x256 | 512x128 |
|---:|---:|---:|---:|---:|
| 1 | 0.699542 | 0.930323 | 0.512254 | 0.934594 |
| 2 | 0.430306 | 0.953796 | 0.653520 | 0.978133 |
| 3 | 0.366047 | 0.960243 | 0.696582 | 0.986344 |
| 4 | 0.343976 | 0.962484 | 0.712516 | 0.988281 |

Additional final slice metrics:

| metric | value |
|---|---:|
| valid/input_seq_len/accuracy-512 | 0.993789 |
| valid/input_seq_len/accuracy-1024 | 0.712516 |
| valid/num_kv_pairs/accuracy-128 | 0.988281 |
| valid/num_kv_pairs/accuracy-256 | 0.712516 |

## Runtime

| item | value |
|---|---:|
| status | completed |
| wall_clock | 00:05:11 |
| OOM | false |
| peak memory used | 2869 MiB / 11264 MiB |

## Reference Boundary

There are two different GDN references in the existing artifacts:

| reference | valid/loss | valid/accuracy | 1024x256 | interpretation |
|---|---:|---:|---:|---|
| `gated_delta_net-usegate0-s123-d123-noearly4ep` | 0.345 | 0.962 | 0.711 | Comparable to this repeat. Current script uses `use_gate=false`. |
| `gated_delta_net-default-s123-d123` epoch4 | 0.268798 | 0.972832 | 0.788387 | Legacy old baseline row, not reproduced by the current noearly GDN script. |

Against the legacy old default epoch4 row, this repeat is weaker:

| metric | legacy default epoch4 | 64x4 repeat | delta |
|---|---:|---:|---:|
| valid/loss | 0.268798 | 0.343976 | +0.075178 |
| valid/accuracy | 0.972832 | 0.962484 | -0.010347 |
| 1024x256 | 0.788387 | 0.712516 | -0.075871 |

This should not be interpreted as same-seed drift. The legacy run used an older code commit and a different training-control profile (`max_epochs=32`, early stopping enabled), while the current noearly4ep GDN baseline is represented by `gated_delta_net-usegate0-s123-d123-noearly4ep` and this `64x4` repeat. The direct same-profile comparison is stable.

## Batch Profile Comparison

Compared with the completed `256x1` GDN probe, `64x4` is substantially better:

| metric | 256x1 probe | 64x4 repeat | delta |
|---|---:|---:|---:|
| valid/loss | 0.418216 | 0.343976 | -0.074240 |
| valid/accuracy | 0.946996 | 0.962484 | +0.015488 |
| 1024x256 | 0.610906 | 0.712516 | +0.101609 |

This strengthens the earlier conclusion: GDN should not switch to `256x1` based on capacity alone. For current GDN experiments, keep `64x4` as the official profile unless a separate `256x1` repeat series recovers quality.

## Artifacts

- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-final.csv`
- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-epoch-end-valid.csv`
- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-validation-history.csv`
- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-slice-level.csv`
- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-runtime-summary.csv`
- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-comparison.csv`
- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-memory.csv`
- `docs/artifacts/20260519-gdn-default-64x4-repeat/gdn-64x4-repeat-run-manifest.json`
