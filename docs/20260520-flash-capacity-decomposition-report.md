# 2026-05-20 Flash capacity decomposition report

## Scope

- 本轮只跑 Flash-VQG `gd_residual_v1`, 不跑 GDN, 不重跑 dense baseline, 不做 event_pack 优化, 不做真实任务, 不改核心模型源码.
- 新 completed 结果统一为 `MAX_EPOCHS=4`, early stopping disabled, `TRAIN_BATCH_SIZE=64`, `GRADIENT_ACCUMULATION_STEPS=4`, `EVAL_BATCH_SIZE=16`, `DATA_SEED=123`, `SEED_VALUES=123`, `dtype_policy=float32`.
- 本轮 completed Flash 结果属于 `b64_ga4 + fp32` official scope, 但每个 decomposition 配置目前只有 seed123, 因此跨 cb/rank 的强弱结论只作为 single-seed trend.
- GDN h2-ev8/h2-ev10 只从既有 report/artifacts 读取为 reference, 不新增 GDN run.
- 不混入 historical inferred rows, b128_ga2 rows, auto-fp16 rows.

## Artifacts

- `docs/artifacts/20260520-flash-capacity-decomposition/flash-capacity-decomp-final.csv`
- `docs/artifacts/20260520-flash-capacity-decomposition/flash-capacity-decomp-epoch-end-valid.csv`
- `docs/artifacts/20260520-flash-capacity-decomposition/flash-capacity-decomp-slice-level.csv`
- `docs/artifacts/20260520-flash-capacity-decomposition/flash-capacity-decomp-model-capacity-and-params.csv`
- `docs/artifacts/20260520-flash-capacity-decomposition/flash-capacity-decomp-combined-with-gdn.csv`
- `docs/artifacts/20260520-flash-capacity-decomposition/flash-capacity-decomp-run-manifest.json`

## Run Status

| run | status | wall-clock | loss | acc | 1024x256 | 512x128 | peak MiB | dynamic capacity | params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gd-cb128-r8-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 04:53:20 | 0.257383 | 0.962073 | 0.772105 | 0.958914 | 4849 | 131072 | 1167558 |
| `gd-cb256-r4-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 05:07:14 | 0.149548 | 0.980348 | 0.895023 | 0.974586 | 4343 | 131072 | 1183430 |
| `gd-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 04:09:51 | 0.070768 | 0.993879 | 0.968711 | 0.995086 | 6319 | 131072 | 1160390 |
| `gd-cb128-r10-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 05:15:11 | 0.312174 | 0.956491 | 0.704160 | 0.968547 | 5477 | 163840 | 1167814 |

## 131k Capacity Decomposition

| config | cb | rank | loss | acc | 1024x256 | 512x128 | VQ usage mean | write top1 mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cb128-r8-s123-d123` | 128 | 8 | 0.257383 | 0.962073 | 0.772105 | 0.958914 | 40.674603 | 0.502074 |
| `cb256-r4-s123-d123` | 256 | 4 | 0.149548 | 0.980348 | 0.895023 | 0.974586 | 20.337302 | 0.246565 |
| `cb64-r16-s123-d123` | 64 | 16 | 0.070768 | 0.993879 | 0.968711 | 0.995086 | 81.349206 | 0.531780 |

- 131k 容量档 seed123 下最强的是 `cb64-r16-s123-d123`: 1024x256=0.968711, overall acc=0.993879.

## GDN References

- GDN h2-ev8 5-seed 1024x256: n=5, mean=0.828155, std=0.013317, range=0.040980, min=0.806418, max=0.847398.
- GDN h2-ev8 5-seed overall acc: n=5, mean=0.978032, std=0.001772, range=0.005448, min=0.975081, max=0.980528.
- GDN h2-ev10 5-seed 1024x256: n=5, mean=0.834573, std=0.015274, range=0.046125, min=0.812797, max=0.858922.
- GDN h2-ev10 5-seed overall acc: n=5, mean=0.978884, std=0.001958, range=0.005891, min=0.976080, max=0.981971.

## Answers

1. 131k 容量档谁最强: `cb64-r16-s123-d123` 是本轮 seed123 best-of-family, hard slice 比 h2-ev8 5-seed mean 高 +0.140556, overall acc 比 h2-ev8 mean 高 +0.015847.
2. cb128-r10 是否强于 GDN h2-ev10: 否. cb128-r10 seed123 hard=0.704160, h2-ev10 5-seed mean hard=0.834573, delta=-0.130413; overall acc delta=-0.022393.
3. cb/rank 分解影响明显. 在相同 131k dynamic capacity 下, 本轮 seed123 呈现 `cb64-r16 > cb256-r4 > cb128-r8` 的 hard-slice 排序. 这说明更多 codebook slots 不一定更好, 更高 rank 在这个 seed 下更有利, 但这是 single-seed trend, 不能直接升级为稳定结论.
4. Official scope: 四个新 Flash rows 都是 completed `b64_ga4 + fp32` official training runs, 可进入 canonical ledger. 但 decomposition 比较当前只有 seed123, 只能作为 single-seed trend; 若要 claim 稳定 best-of-family, 需要对最强分解补 s124/s125.
5. 下一步: 优先给 `cb64-r16` 补 seeds 124/125, 同时可给 `cb256-r4` 做 paired seed 复核, 因为它在历史和本轮都不是完全稳定的低容量强点.

## Caveats

- 本轮没有训练新的 GDN reference, GDN 对照来自既有 `20260520-gdn-capacity-layout-followup` 和 `20260520-gdn-h2-ev10-multiseed-followup` artifacts.
- Flash decomposition 目前每个配置只有一个 seed, 不能和 GDN 5-seed baseline 在稳定性上对称.
- `cb128-r10` 是 163k capacity 上界候选, 但本轮 seed123 明显弱, 不建议作为后续 capacity-fair Flash baseline.
