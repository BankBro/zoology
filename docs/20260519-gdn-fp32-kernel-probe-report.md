# GDN float32 kernel dtype probe report

Date: 2026-05-19

## Question

This run tests whether the current comparable GDN baseline improves when the FLA gated-delta kernel inputs are forced to `float32`.

Tested profile:

- model: `gated_delta_net`, `use_gate=false`
- run_type: `gdn_kernel_dtype_probe`
- `GDN_KERNEL_DTYPE=float32`
- seed: `123`
- data_seed: `123`
- max_epochs: `4`
- early stopping: disabled
- train_batch_size: `64`
- gradient_accumulation_steps: `4`
- effective_train_batch_size: `256`
- eval_batch_size: `16`
- GPU: `0`, RTX 2080 Ti, sm75

Run:

- launch_id: `flash-vqg-20260519-gdn-default-fp32-kernel-2026-05-19-05-00-58`
- run_id: `gated_delta_net-default-s123-d123-noearly4ep-b64-ga4-gdnkernel-fp32`

## Result

The run completed without OOM. Forcing the GDN FLA kernel path to `float32` did not improve the current 4 epoch noearly baseline. It landed essentially on the same quality as the current `64x4` repeat, with slightly worse final loss and negligible accuracy/hard-slice differences.

| metric | 64x4 auto/fp16-kernel repeat | fp32 kernel probe | delta |
|---|---:|---:|---:|
| valid/loss | 0.343976 | 0.347751 | +0.003775 |
| valid/accuracy | 0.962484 | 0.962247 | -0.000237 |
| valid/mqar_case/accuracy-1024x256 | 0.712516 | 0.711828 | -0.000687 |
| valid/mqar_case/accuracy-512x128 | 0.988281 | 0.987602 | -0.000680 |
| valid/input_seq_len/accuracy-512 | 0.993789 | 0.993324 | -0.000465 |
| valid/input_seq_len/accuracy-1024 | 0.712516 | 0.711828 | -0.000687 |
| valid/num_kv_pairs/accuracy-128 | 0.988281 | 0.987602 | -0.000680 |
| valid/num_kv_pairs/accuracy-256 | 0.712516 | 0.711828 | -0.000687 |

Interpretation: no evidence that the current GDN weakness is caused by the sm75 default fp16 kernel cast. The final `1024x256` delta is only `-0.000687`, and overall accuracy delta is `-0.000237`.

## Epoch Metrics

| epoch | valid/loss | valid/accuracy | 1024x256 | 512x128 |
|---:|---:|---:|---:|---:|
| 1 | 0.698940 | 0.930562 | 0.514297 | 0.935125 |
| 2 | 0.435673 | 0.953241 | 0.651016 | 0.976742 |
| 3 | 0.369895 | 0.959833 | 0.694918 | 0.985414 |
| 4 | 0.347751 | 0.962247 | 0.711828 | 0.987602 |

Additional final slice metrics:

| metric | value |
|---|---:|
| valid/input_seq_len/accuracy-512 | 0.993324 |
| valid/input_seq_len/accuracy-1024 | 0.711828 |
| valid/num_kv_pairs/accuracy-128 | 0.987602 |
| valid/num_kv_pairs/accuracy-256 | 0.711828 |

## Runtime

| item | value |
|---|---:|
| status | completed |
| wall_clock | 00:07:30 |
| OOM | false |
| observed peak memory used | 3483 MiB / 11264 MiB |

The fp32 run took `450.526` seconds by manifest timestamps, compared with `310.550` seconds for the current `64x4` repeat. The first epoch included visible Triton/FLA cold compile overhead for fp32 kernels; later epochs were close to normal GDN step speed. The runtime is therefore not better than the current auto/fp16-kernel path.

## Reference Boundary

The directly comparable baseline is the current noearly4ep `use_gate=false` profile, especially `gated_delta_net-default-s123-d123-noearly4ep-b64-ga4-repeat` and the earlier `gated_delta_net-usegate0-s123-d123-noearly4ep` row.

Do not treat `gated_delta_net-default-s123-d123` as the current noearly4ep baseline. That legacy run used an older code commit and different training-control profile, so it is not the correct comparator for this dtype probe.

## Conclusion

Forcing `float32` does not recover the legacy stronger GDN result and costs extra cold compile/runtime. However, because the project dtype policy now requires RTX 2080 Ti/sm75 official MQAR runs to prefer float32, this completed run is promoted to the current GDN 2080 Ti float32 baseline despite not improving over the auto/fp16-kernel repeat. It is recorded in `docs/artifacts/gdn/gdn-hparam-effect-summary.csv` with `baseline_role=current_float32_2080ti_baseline`, and should only be directly compared with rows in the same `float32_only` dtype scope.

## Artifacts

- `docs/artifacts/20260519-gdn-fp32-kernel-probe/gdn-fp32-kernel-final.csv`
- `docs/artifacts/20260519-gdn-fp32-kernel-probe/gdn-fp32-kernel-epoch-end-valid.csv`
- `docs/artifacts/20260519-gdn-fp32-kernel-probe/gdn-fp32-kernel-validation-history.csv`
- `docs/artifacts/20260519-gdn-fp32-kernel-probe/gdn-fp32-kernel-slice-level.csv`
- `docs/artifacts/20260519-gdn-fp32-kernel-probe/gdn-fp32-kernel-runtime-summary.csv`
- `docs/artifacts/20260519-gdn-fp32-kernel-probe/gdn-fp32-kernel-comparison.csv`
- `docs/artifacts/20260519-gdn-fp32-kernel-probe/gdn-fp32-kernel-run-manifest.json`
