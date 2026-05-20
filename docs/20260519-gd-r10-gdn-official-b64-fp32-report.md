# 2026-05-19 Flash r10 / GDN b64_ga4 fp32 official report

## Scope

- 本轮新跑 completed 结果统一为 `MAX_EPOCHS=4`, early stopping disabled, `TRAIN_BATCH_SIZE=64`, `GRADIENT_ACCUMULATION_STEPS=4`, `EVAL_BATCH_SIZE=16`, `DATA_SEED=123`, `dtype_policy=float32`.
- GDN 新跑结果均显式设置 `GDN_KERNEL_DTYPE=float32`, 并记录 `actual_kernel_dtype=float32`.
- 本报告的 official comparison scope 是 `b64_ga4 + fp32`. 不把 `128x2`, rank/search 结果, dtype probe, auto-fp16 结果混入 completed official 结论.
- `gdn-h2-ev2-s123` 按要求不重跑, 仅以历史 `64x4 + fp32 kernel` probe 作为 reference 行引用, 不计为本轮新 completed run.

## Artifacts

- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/flash-r10-final.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/flash-r10-epoch-end-valid.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/flash-r10-slice-level.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/gdn-official-final.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/gdn-official-epoch-end-valid.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/gdn-official-slice-level.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/model-capacity-and-params.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/combined-official-comparison.csv`
- `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32/run-manifest.json`

## Run Status

| run | status | wall-clock | loss | acc | 1024x256 | peak MiB | OOM/fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gd-r10-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 03:31:23 | 0.149564 | 0.981746 | 0.892020 | 6389 | none |
| `gd-r10-s124-d123-b64-ga4-fp32-noearly4ep` | completed | 04:31:27 | 0.044934 | 0.996684 | 0.985242 | 6403 | none |
| `gd-r10-s125-d123-b64-ga4-fp32-noearly4ep` | completed | 03:46:26 | 0.039301 | 0.997251 | 0.990730 | 6361 | none |
| `gd-r10-s126-d123-b64-ga4-fp32-noearly4ep` | completed | 05:24:42 | 0.240044 | 0.967531 | 0.787195 | 6385 | none |
| `gdn-h2-ev2-s124-d123-b64-ga4-fp32-noearly4ep` | completed | 00:07:48 | 0.331054 | 0.964425 | 0.725734 | 3483 | none |
| `gdn-h2-ev8-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 00:11:36 | 0.289848 | 0.978623 | 0.831684 | 3877 | none |
| `gdn-h2-ev8-s125-d123-b64-ga4-fp32-noearly4ep` | completed | 00:10:41 | 0.283829 | 0.980528 | 0.847398 | 3877 | none |
| `gdn-h1-ev2-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 00:09:34 | 0.400872 | 0.948956 | 0.632508 | 3941 | none |
| `gdn-h2-ev2-s125-d123-b64-ga4-fp32-noearly4ep` | completed | 00:05:58 | 0.384506 | 0.951634 | 0.644246 | 3469 | none |
| `gdn-h2-ev8-s124-d123-b64-ga4-fp32-noearly4ep` | completed | 00:08:02 | 0.292128 | 0.977516 | 0.823707 | 3863 | none |
| `gdn-h2-ev4-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 00:07:17 | 0.470866 | 0.943018 | 0.587766 | 3453 | none |
| `gdn-h1-ev4-s123-d123-b64-ga4-fp32-noearly4ep` | failed | 00:00:10 |  |  |  | n/a | infrastructure failure: CUDA/NVML/Triton driver unavailable; no fallback attempted |
| `gdn-h2-ev16-s123-d123-b64-ga4-fp32-noearly4ep` | failed | 00:00:10 |  |  |  | n/a | infrastructure failure: CUDA/NVML/Triton driver unavailable; no fallback attempted |
| `gdn-h2-ev2-s123-d123-b64-ga4-fp32-reference` | reference | 00:07:31 | 0.347751 | 0.962247 | 0.711828 | 3483 | none |

## Flash r10 Stability

- Flash r10 1024x256: n=4, mean=0.913797, min=0.787195, max=0.990730, range=0.203535, stdev=0.082954.
- Flash r10 overall accuracy: n=4, mean=0.985803, min=0.967531, max=0.997251, range=0.029720, stdev=0.012245.
- r10-s125 的旧 `128x2` hard slice 是 0.985121, 本轮 `64x4/fp32` 是 0.990730, delta=+0.005609.
- r10-s125 的旧 `128x2` overall acc 是 0.996467, 本轮 `64x4/fp32` 是 0.997251, delta=+0.000784.

## GDN Stability And Capacity

- GDN h2-ev8 seeds 123/124/125 1024x256: n=3, mean=0.834263, min=0.823707, max=0.847398, range=0.023691, stdev=0.009842.
- GDN h2-ev8 seeds 123/124/125 overall accuracy: n=3, mean=0.978889, min=0.977516, max=0.980528, range=0.003012, stdev=0.001244.

| config | status | params | dynamic capacity | loss | acc | 1024x256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gdn-h1-ev2-usegate0-s123-d123` | completed | 1167747 | 32768 | 0.400872 | 0.948956 | 0.632508 |
| `gdn-h2-ev2-usegate0-s123-d123` | reference | 1167878 | 16384 | 0.347751 | 0.962247 | 0.711828 |
| `gdn-h2-ev2-usegate0-s124-d123` | completed | 1167878 | 16384 | 0.331054 | 0.964425 | 0.725734 |
| `gdn-h2-ev2-usegate0-s125-d123` | completed | 1167878 | 16384 | 0.384506 | 0.951634 | 0.644246 |
| `gdn-h1-ev4-usegate0-s123-d123` | failed | 1234563 | 65536 |  |  |  |
| `gdn-h2-ev4-usegate0-s123-d123` | completed | 1234566 | 32768 | 0.470866 | 0.943018 | 0.587766 |
| `gdn-h2-ev8-usegate0-s123-d123` | completed | 1367942 | 65536 | 0.289848 | 0.978623 | 0.831684 |
| `gdn-h2-ev8-usegate0-s124-d123` | completed | 1367942 | 65536 | 0.292128 | 0.977516 | 0.823707 |
| `gdn-h2-ev8-usegate0-s125-d123` | completed | 1367942 | 65536 | 0.283829 | 0.980528 | 0.847398 |
| `gdn-h2-ev16-usegate0-s123-d123` | failed | 1634694 | 131072 |  |  |  |

## Flash vs GDN

- 最强 Flash r10 是 `cb256-r10-s125-d123`, 1024x256=0.990730, acc=0.997251.
- 最强 GDN h2-ev8 是 `gdn-h2-ev8-usegate0-s125-d123`, 1024x256=0.847398, acc=0.980528.
- h2-ev8 距离 Flash r10 best hard slice 仍差 0.143332, overall acc 差 0.016723.

## Answers

1. Flash r10 是否稳定: 不完全稳定. seeds 124/125 很强, seed 123 中等, seed 126 明显掉点. 1024x256 hard slice range=0.203535, overall acc range=0.029720, 说明 r10 在该口径下仍有 seed sensitivity.
2. r10-s125 128x2 强结果是否复现: 在同 seed 下复现. r10-s125 旧 `128x2` hard slice=0.985121, 本轮 `64x4/fp32` hard slice=0.990730, delta=+0.005609. 旧 `128x2` overall acc=0.996467, 本轮=0.997251, delta=+0.000784. 这些旧值只用于复现性解释, 不混入 official comparison table.
3. GDN h2-ev8 是否稳定: 相对稳定. seeds 123/124/125 的 1024x256 hard slice range=0.023691, overall acc range=0.003012, 最强 seed 125 为 0.847398, 最低 seed 124 为 0.823707.
4. GDN h2-ev8 是否追近 Flash r10: 没有追近到同一质量水平. 最强 GDN h2-ev8 hard slice=0.847398, 最强 Flash r10 hard slice=0.990730, gap=0.143332. overall acc gap=0.016723. 1024x256 hard slice 是主要短板.
5. GDN capacity-up 趋势: h2-ev8 明显好于 h2-ev2 低容量组, 但 h2-ev4-s123 低于 h2-ev2-s123 reference 和 h2-ev2-s124, 说明扩 `expand_v` 不是单调保证. h1-ev2-s123 也低于 h2-ev2 reference. h1-ev4 和 h2-ev16 未完成, 失败原因是 CUDA/NVML/Triton driver 不可用, 不是 OOM, 因此不能据此判断 ev4/h1 或 ev16 的质量趋势.
6. 近参数量, 近动态容量, 双约束公平比较下一步: 当前 Flash r10 params=1,184,198, dynamic_capacity=327,680. GDN h2-ev8 params=1,367,942, dynamic_capacity=65,536. h2-ev16 dynamic_capacity=131,072 仍低于 Flash r10, 但 params 更高且本轮未完成. 当前没有一个 GDN row 同时满足近参数量和近动态容量. 下一步应等 CUDA/NVML 恢复后补跑 h1-ev4/h2-ev16, 再设计两个独立轴: 一组固定 params 接近 Flash, 一组固定 dynamic capacity 接近 Flash, 最后只在双约束交集内做 direct official comparison.
7. Scope 声明: 本轮新 completed 结果只属于 `b64_ga4 + fp32` official comparison scope. 本报告不把 `128x2`, search, probe, auto-fp16 结果混入 official 质量结论. h2-ev2-s123 只是按要求引用的同 batch/dtype reference.

## Notes

- New completed Flash rows: 4.
- New completed GDN rows: 7. Reference rows: 1.
- Failed/OOM rows保留在 `gdn-official-final.csv` 和 `run-manifest.json`, 但不追加到 canonical ledger.
