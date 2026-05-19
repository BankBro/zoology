# GDN default baseline 256x1 probe report

Date: 2026-05-19

## Question

This probe tests whether the original `gated_delta_net-default-s123-d123` baseline keeps comparable 4 epoch quality when the effective train batch stays at 256 but the training micro-batch changes to `256x1`.

The tested profile is:

- model: `gated_delta_net_default`
- run_type: `batch_accum_probe`
- seed: `123`
- data_seed: `123`
- max_epochs: `4`
- early stopping: disabled
- train_batch_size: `256`
- gradient_accumulation_steps: `1`
- effective_train_batch_size: `256`
- eval_batch_size: `16`

Reference is the old GDN default baseline row from `docs/artifacts/20260515-gd-gdn-noearly4ep-comparison/final-comparison.csv`:

- run_id: `gated_delta_net-default-s123-d123`
- checkpoint: `epoch4`
- valid/loss: `0.268798`
- valid/accuracy: `0.97283154296875`
- valid/mqar_case/accuracy-1024x256: `0.788387`

## Result

The `256x1` run completed without OOM, but quality dropped materially relative to the old GDN default epoch4 reference.

| metric | old default epoch4 | 256x1 probe epoch4 | delta |
|---|---:|---:|---:|
| valid/loss | 0.268798 | 0.418216 | +0.149418 |
| valid/accuracy | 0.972832 | 0.946996 | -0.025835 |
| 1024x256 | 0.788387 | 0.610906 | -0.177481 |

This is a large hard-slice regression. The result indicates that `256x1` changes the optimization/runtime behavior enough that it should not be used as the default GDN training profile without more repeat diagnostics.

## Epoch Metrics

| epoch | valid/loss | valid/accuracy | 1024x256 | 512x128 |
|---:|---:|---:|---:|---:|
| 1 | 0.955106 | 0.888670 | 0.331363 | 0.806359 |
| 2 | 0.520121 | 0.935173 | 0.541320 | 0.944688 |
| 3 | 0.453518 | 0.942378 | 0.581863 | 0.959914 |
| 4 | 0.418216 | 0.946996 | 0.610906 | 0.967625 |

Additional final slice metrics:

| metric | value |
|---|---:|
| valid/input_seq_len/accuracy-512 | 0.982859 |
| valid/input_seq_len/accuracy-1024 | 0.610906 |
| valid/num_kv_pairs/accuracy-128 | 0.967625 |
| valid/num_kv_pairs/accuracy-256 | 0.610906 |

## Runtime

| item | value |
|---|---:|
| status | completed |
| wall_clock | 00:06:35 |
| OOM | false |
| GPU | 0 |
| peak memory used | 8599 MiB / 11264 MiB |

The runtime is short for this baseline, but the quality loss is too large for a pure throughput-driven profile switch.

## Interpretation

`256x1` and `64x4` share the same effective train batch size of 256, but they are not bitwise or optimization-trajectory equivalent. This run suggests the original GDN default baseline is sensitive to the micro-batch split.

Recommended policy for near-term GDN experiments:

- Do not switch the GDN baseline family to `256x1` based on capacity alone.
- Keep official GDN baseline and hparam comparisons on the existing `64x4` profile unless a same-seed repeat shows `256x1` can recover quality.
- If `256x1` remains interesting for throughput, run at least a same-seed repeat and one additional seed before treating it as a viable official profile.

## Artifacts

- `docs/artifacts/20260519-gdn-default-b256-ga1-probe/gdn-b256-ga1-probe-final.csv`
- `docs/artifacts/20260519-gdn-default-b256-ga1-probe/gdn-b256-ga1-probe-epoch-end-valid.csv`
- `docs/artifacts/20260519-gdn-default-b256-ga1-probe/gdn-b256-ga1-probe-validation-history.csv`
- `docs/artifacts/20260519-gdn-default-b256-ga1-probe/gdn-b256-ga1-probe-slice-level.csv`
- `docs/artifacts/20260519-gdn-default-b256-ga1-probe/gdn-b256-ga1-probe-runtime-summary.csv`
- `docs/artifacts/20260519-gdn-default-b256-ga1-probe/gdn-b256-ga1-probe-vs-original-baseline.csv`
- `docs/artifacts/20260519-gdn-default-b256-ga1-probe/gdn-b256-ga1-probe-run-manifest.json`
