# 2026-05-19 gd_residual_v1 rank + GDN capacity-up report

## Scope

- Flash-VQG runs use `gd_residual_v1`, `cb=256`, `MAX_EPOCHS=4`, early stopping disabled.
- GDN runs use `use_gate=false`, `use_short_conv=true`, `conv_size=4`, `GDN_KERNEL_DTYPE=float32`.
- `128x2` and `64x4` both keep effective train batch size at 256, but they are not bitwise equivalent and are not mixed into one official comparison scope.
- Failed fallback attempts are recorded in `run-manifest.json`; final quality tables use completed runs only.

## Artifacts

- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/flash-final.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/flash-epoch-end-valid.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/flash-slice-level.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/gdn-final.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/gdn-epoch-end-valid.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/gdn-slice-level.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/model-capacity-and-params.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/combined-comparison.csv`
- `docs/artifacts/20260519-gd-rank-gdn-capacity-up/run-manifest.json`

## Run Status

| run | status | batch/GA | eval batch | dtype | wall-clock | OOM/fallback |
| --- | --- | --- | --- | --- | --- | --- |
| `gd-r10-wk4-mu01-t025-cb256-s125-d123-noearly4ep` | completed | 128x2 | 32 | default_float32_on_2080ti | 01:57:02 | none |
| `gd-r6-wk4-mu01-t025-cb256-s125-d123-noearly4ep` | completed | 128x2 | 32 | default_float32_on_2080ti | 02:30:53 | none |
| `gd-r8-wk4-mu01-t025-cb256-s126-d123-noearly4ep` | completed | 128x2 | 32 | default_float32_on_2080ti | 02:50:42 | none |
| `gd-r16-wk4-mu01-t025-cb256-s126-d123-noearly4ep` | completed | 64x4 | 16 | default_float32_on_2080ti | 05:28:28 | b128-ga2-eb32:oom;b128-ga2-eb16:oom |
| `gdn-usegate0-h2-ev2-s123-d123-noearly4ep` | completed | 128x2 | 32 | GDN_KERNEL_DTYPE=float32 | 00:07:52 | none |
| `gdn-usegate0-h2-ev2-s125-d123-noearly4ep` | completed | 128x2 | 32 | GDN_KERNEL_DTYPE=float32 | 00:07:45 | none |
| `gdn-usegate0-h2-ev4-s123-d123-noearly4ep` | completed | 128x2 | 32 | GDN_KERNEL_DTYPE=float32 | 00:13:11 | none |
| `gdn-usegate0-h2-ev8-s123-d123-noearly4ep` | completed | 128x2 | 32 | GDN_KERNEL_DTYPE=float32 | 00:15:46 | none |
| `gdn-usegate0-h2-ev2-s124-d123-noearly4ep` | completed | 128x2 | 32 | GDN_KERNEL_DTYPE=float32 | 00:05:47 | none |
| `gdn-usegate0-h1-ev2-s123-d123-noearly4ep` | completed | 128x2 | 32 | GDN_KERNEL_DTYPE=float32 | 00:12:46 | none |
| `gdn-usegate0-h1-ev4-s123-d123-noearly4ep` | completed | 128x2 | 32 | GDN_KERNEL_DTYPE=float32 | 00:12:16 | none |

## Flash Results

| config | batch/GA | loss | acc | 1024x256 | 512x128 | peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cb256-r6-s125-d123` | 128x2 | 0.145781 | 0.979705 | 0.894250 | 0.974313 | 10359 |
| `cb256-r8-s126-d123` | 128x2 | 0.233719 | 0.970270 | 0.800082 | 0.976172 | 10455 |
| `cb256-r10-s125-d123` | 128x2 | 0.046766 | 0.996467 | 0.985121 | 0.997648 | 10809 |
| `cb256-r16-s126-d123` | 64x4 | 0.105231 | 0.990707 | 0.942695 | 0.995320 | 8883 |

Flash interpretation:

- In the completed `128x2` Flash scope, `r10-s125` is the strongest row: loss `0.046766`, accuracy `0.996467`, and `1024x256=0.985121`.
- `r6-s125` is much weaker on the hard slice (`1024x256=0.894250`), so rank 6 is not a good immediate anchor.
- `r8-s126` is a weak seed/profile row (`1024x256=0.800082`), even though earlier `r8-s125` rows were strong. This reinforces the existing rank/seed sensitivity caveat.
- `r16-s126` completed only after fallback to `64x4/eval16`; it is a strong quality row (`1024x256=0.942695`) but is not in the same batch accumulation scope as the `128x2` Flash rows.
- Mainline judgement: continue treating `r10-s125` and the prior strong `r8-s125`/`r16` rows as the useful anchors. Do not infer that `r8` is generally weak from `r8-s126` alone.

## GDN Results

| config | batch/GA | dtype | params | dyn cap | loss | acc | 1024x256 | peak MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdn-h1-ev2-usegate0-s123-d123` | 128x2 | float32 | 1167747 | 32768 | 0.434304 | 0.943428 | 0.590215 | 5487 |
| `gdn-h2-ev2-usegate0-s123-d123` | 128x2 | float32 | 1167878 | 16384 | 0.388440 | 0.953590 | 0.651508 | 5043 |
| `gdn-h2-ev2-usegate0-s124-d123` | 128x2 | float32 | 1167878 | 16384 | 0.366552 | 0.958182 | 0.680297 | 5029 |
| `gdn-h2-ev2-usegate0-s125-d123` | 128x2 | float32 | 1167878 | 16384 | 0.748819 | 0.883348 | 0.321414 | 5043 |
| `gdn-h1-ev4-usegate0-s123-d123` | 128x2 | float32 | 1234563 | 65536 | 0.369440 | 0.957702 | 0.687984 | 5493 |
| `gdn-h2-ev4-usegate0-s123-d123` | 128x2 | float32 | 1234566 | 32768 | 0.402248 | 0.955918 | 0.668121 | 5047 |
| `gdn-h2-ev8-usegate0-s123-d123` | 128x2 | float32 | 1367942 | 65536 | 0.284167 | 0.978405 | 0.829688 | 5309 |

GDN interpretation:

- All GDN capacity-up rows completed at `128x2/eval32` with `GDN_KERNEL_DTYPE=float32`; no GDN fallback was needed.
- Within the GDN `128x2` float32 scope, `h2-ev8-s123` is clearly best: loss `0.284167`, accuracy `0.978405`, and `1024x256=0.829688`.
- Capacity helps, but not monotonically across every shape. `h2-ev8` improves over `h2-ev2-s123` by `+0.024815` accuracy and `+0.178180` on `1024x256`, while `h2-ev4-s123` is only slightly better than `h2-ev2-s123`.
- Head merge does not provide a clean win. `h1-ev2-s123` is worse than `h2-ev2-s123`; `h1-ev4-s123` is modestly better than `h2-ev4-s123` on `1024x256`, but still far behind `h2-ev8-s123`.
- The `h2-ev2` seed sweep is unstable: `s124` reaches `1024x256=0.680297`, `s123` reaches `0.651508`, and `s125` collapses to `0.321414`. GDN seed variance remains a real caveat.
- GDN has not caught Flash-VQG in this 4 epoch MQAR setting. The best GDN row, `h2-ev8-s123`, trails Flash `r10-s125` by `0.018062` overall accuracy and `0.155434` on `1024x256`.

## Decision

- Flash: keep `r10-s125` as the strongest new `128x2` rank follow-up result. Treat `r8-s126` as a weak seed/profile row, not as a full rejection of rank 8.
- GDN: `h2-ev8` is the best capacity-up direction in this batch, but it still does not approach Flash hard-slice quality enough to replace Flash as the mainline.
- Comparison scope: Flash `r6/r8/r10` are `128x2`; Flash `r16-s126` is `64x4` fallback; GDN rows are `128x2` and `float32_only`. These scopes must stay separated in official conclusions.

## Caveats

- `batch_accum_profile` differences are experimental conditions, not just logging details.
- `dtype_comparison_scope=float32_only` is required for current GDN fairness on 2080 Ti.
- SwanLab local directories, checkpoints, tmp logs, and generated analysis images are not part of the committed artifact set.
