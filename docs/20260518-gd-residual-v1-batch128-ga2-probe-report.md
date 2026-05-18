# gd_residual_v1 Batch128 GA2 Probe Report

## Summary

- Run status: completed. No OOM detected.
- Run type: `batch_accum_probe`.
- Training口径: `train_batch_size=128`, `gradient_accumulation_steps=2`, effective batch size = `256`.
- 对照口径: 64x4 的 effective batch size 也是 `256`, 但 128x2 和 64x4 不是 bitwise equivalent, 不能混入 64x4 official results 当同一训练口径.
- 结论: final quality 按阈值可视为基本等价, runtime 明显更快, peak memory 为 9687 MiB / 11264 MiB (86.0%). 可以考虑作为后续 Flash-VQG 训练口径, 但必须在后续所有实验中显式记录 batch/accum 口径.

## Artifacts

- `docs/artifacts/20260518-gd-batch128-ga2-probe/batch128-ga2-probe-final.csv`
- `docs/artifacts/20260518-gd-batch128-ga2-probe/batch128-ga2-probe-epoch-end-valid.csv`
- `docs/artifacts/20260518-gd-batch128-ga2-probe/batch128-ga2-probe-validation-history.csv`
- `docs/artifacts/20260518-gd-batch128-ga2-probe/batch128-ga2-probe-slice-level.csv`
- `docs/artifacts/20260518-gd-batch128-ga2-probe/batch128-ga2-probe-run-manifest.json`
- `docs/artifacts/20260518-gd-batch128-ga2-probe/batch128-ga2-probe-runtime-summary.csv`
- `docs/artifacts/20260518-gd-batch128-ga2-probe/batch128-ga2-probe-final-deltas.csv`

## Runtime

| item | value |
|---|---:|
| status | completed |
| OOM | false |
| GPU | 0 |
| wall-clock | 02:40:21 |
| elapsed_sec | 9621 |
| peak memory | 9687 MiB / 11264 MiB |
| avg epoch wall time | 00:40:05 |
| avg effective optimizer-step wall time, incl validation | 3.412 sec |
| 64x4 original wall-clock | 04:15:27 |
| 64x4 repeat wall-clock | 03:57:31 |
| speedup vs original | 37.23% |
| speedup vs repeat | 32.49% |

Runtime note: `avg effective optimizer-step wall time` is normalized by full wall-clock and includes validation/analysis overhead. The tqdm train-loop timings are noisy because mid-epoch validation pauses are folded into the epoch progress display.

## Epoch-End Quality

| epoch | valid/loss | valid/accuracy | 1024x256 |
|---:|---:|---:|---:|
| 1 | 0.314975 | 0.971202 | 0.869605 |
| 2 | 0.116435 | 0.987819 | 0.942879 |
| 3 | 0.089558 | 0.990451 | 0.955043 |
| 4 | 0.065413 | 0.993677 | 0.967539 |

## Final Quality Comparison

| metric | 64x4 original | 64x4 repeat | 128x2 probe | delta vs original | delta vs repeat |
|---|---:|---:|---:|---:|---:|
| `valid/loss` | 0.046001 | 0.047221 | 0.065413 | +0.019412 | +0.018193 |
| `valid/accuracy` | 0.996053 | 0.995818 | 0.993677 | -0.002376 | -0.002141 |
| `valid/mqar_case/accuracy-1024x256` | 0.982844 | 0.981223 | 0.967539 | -0.015305 | -0.013684 |
| `valid/mqar_case/accuracy-512x128` | 0.996141 | 0.995086 | 0.993719 | -0.002422 | -0.001367 |
| `valid/input_seq_len/accuracy-512` | 0.995102 | 0.994832 | 0.993375 | -0.001727 | -0.001457 |
| `valid/input_seq_len/accuracy-1024` | 0.982844 | 0.981223 | 0.967539 | -0.015305 | -0.013684 |
| `valid/num_kv_pairs/accuracy-128` | 0.996141 | 0.995086 | 0.993719 | -0.002422 | -0.001367 |
| `valid/num_kv_pairs/accuracy-256` | 0.982844 | 0.981223 | 0.967539 | -0.015305 | -0.013684 |

Quality verdict: compared with the 64x4 same-GPU repeat, `valid/accuracy` delta is -0.002141 and `1024x256` delta is -0.013684. Both are within the requested equivalence thresholds: `<0.005` accuracy and `<0.03` hard slice.

## GD Residual Metrics

| metric | 64x4 repeat | 128x2 probe | delta |
|---|---:|---:|---:|
| `valid/attn/gd_residual_write_strength_mean` | 0.058737 | 0.016175 | -0.042563 |
| `valid/attn/gd_residual_m_norm_mean` | 0.013626 | 0.019351 | +0.005724 |
| `valid/attn/gd_residual_m_norm_max` | 7.293782 | 10.674713 | +3.380931 |
| `valid/attn/gd_residual_mu_valid_ratio` | 0.426165 | 0.414202 | -0.011963 |
| `valid/attn/gd_residual_lambda_mean` | 0.166054 | 0.231876 | +0.065822 |
| `valid/attn/gd_residual_inject_ratio` | 0.060541 | 0.118813 | +0.058273 |

## VQ Metrics

| metric | 64x4 repeat | 128x2 probe | delta |
|---|---:|---:|---:|
| `valid/vq/relative_err_mean` | 0.071677 | 0.069796 | -0.001880 |
| `valid/vq/c_entropy` | 3.451194 | 3.393678 | -0.057515 |
| `valid/vq/c_usage_mean` | 20.337302 | 20.337301 | -0.000000 |
| `valid/vq/write_entropy_mean` | 3.177669 | 3.115465 | -0.062205 |
| `valid/vq/write_top1_mass_mean` | 0.373247 | 0.341783 | -0.031464 |

## Decision

128x2 is not numerically identical to 64x4, but this single diagnostic run suggests it is quality-equivalent under the requested threshold and materially faster in wall-clock. Because peak memory reached about 86.0% of GPU0 total memory, the margin is acceptable on this 2080 Ti for this config but should be monitored if rank, dmodel, sequence mix, or model capacity increases.

Recommendation: 可以考虑作为后续 Flash-VQG 训练口径, 但必须在后续所有实验中显式记录 batch/accum 口径.
