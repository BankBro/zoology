# GDN Capacity/Layout Follow-up MQAR Report

- Generated: 2026-05-20T09:02:28.550294+00:00
- Scope: `b64_ga4_fp32_official_gdn` only, `TRAIN_BATCH_SIZE=64`, `GRADIENT_ACCUMULATION_STEPS=4`, `EVAL_BATCH_SIZE=16`, `MAX_EPOCHS=4`, early stopping disabled, `DATA_SEED=123`, `DMODEL=128`, `LR=1e-3`, `GDN_KERNEL_DTYPE=float32`.
- New artifacts: `docs/artifacts/20260520-gdn-capacity-layout-followup/`.
- Prior GDN references are from `docs/20260519-gd-r10-gdn-official-b64-fp32-report.md` and `docs/artifacts/20260519-gd-r10-gdn-official-b64-fp32`.
- Canonical GDN ledger appended rows: 10.

## New Completed Runs
| run | status | wall-clock | loss | acc | 1024x256 | 512x128 | peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gdn-h2-ev8-s126-d123-b64-ga4-fp32-noearly4ep | completed | 00:16:28 | 0.286705 | 0.978413 | 0.831566 | 0.996141 | 3877 |
| gdn-h2-ev8-s127-d123-b64-ga4-fp32-noearly4ep | completed | 00:07:55 | 0.301022 | 0.975081 | 0.806418 | 0.996258 | 3863 |
| gdn-h2-ev16-s124-d123-b64-ga4-fp32-noearly4ep | completed | 00:22:17 | 0.279802 | 0.980092 | 0.844723 | 0.996422 | 3873 |
| gdn-h2-ev16-s125-d123-b64-ga4-fp32-noearly4ep | completed | 00:10:25 | 0.273887 | 0.980463 | 0.846758 | 0.997148 | 3859 |
| gdn-h1-ev8-s123-d123-b64-ga4-fp32-noearly4ep | completed | 00:23:56 | 0.293875 | 0.973618 | 0.802078 | 0.995227 | 4071 |
| gdn-h1-ev8-s124-d123-b64-ga4-fp32-noearly4ep | completed | 00:15:34 | 0.268032 | 0.981024 | 0.854195 | 0.997938 | 4057 |
| gdn-h1-ev8-s125-d123-b64-ga4-fp32-noearly4ep | completed | 00:19:42 | 0.305587 | 0.970588 | 0.780473 | 0.992719 | 4071 |
| gdn-h2-ev6-s123-d123-b64-ga4-fp32-noearly4ep | completed | 00:13:08 | 0.296228 | 0.975709 | 0.811367 | 0.994789 | 3513 |
| gdn-h2-ev10-s123-d123-b64-ga4-fp32-noearly4ep | completed | 00:12:02 | 0.275982 | 0.981971 | 0.858922 | 0.998203 | 3599 |
| gdn-h2-ev12-s123-d123-b64-ga4-fp32-noearly4ep | completed | 00:12:42 | 0.291562 | 0.980175 | 0.844816 | 0.997586 | 3857 |

## h2-ev8 Stability
| seed | loss | acc | 1024x256 | 512x128 | source |
| --- | --- | --- | --- | --- | --- |
| 123 | 0.289848 | 0.978623 | 0.831684 | 0.997656 | 20260519_gd_r10_gdn_official_b64_fp32 |
| 124 | 0.292128 | 0.977516 | 0.823707 | 0.996797 | 20260519_gd_r10_gdn_official_b64_fp32 |
| 125 | 0.283829 | 0.980528 | 0.847398 | 0.998047 | 20260519_gd_r10_gdn_official_b64_fp32 |
| 126 | 0.286705 | 0.978413 | 0.831566 | 0.996141 | 20260520_gdn_capacity_layout_followup |
| 127 | 0.301022 | 0.975081 | 0.806418 | 0.996258 | 20260520_gdn_capacity_layout_followup |

- 1024x256 hard slice over seeds 123/124/125/126/127: n=5, mean=0.828155, std(pop)=0.013317, range=0.040980, min=0.806418, max=0.847398.
- valid/accuracy over the same seeds: mean=0.978032, std(pop)=0.001772, range=0.005448.
- valid/loss over the same seeds: mean=0.290706, std(pop)=0.005874, range=0.017194.
- Conclusion: h2-ev8 remains a stable multi-seed GDN baseline, but it is no longer the strongest single-run point after adding h2-ev10-s123.

## 131k Capacity Layout
| config | seed | loss | acc | 1024x256 | 512x128 |
| --- | --- | --- | --- | --- | --- |
| h2-ev16 | 123 | 0.408903 | 0.960377 | 0.704930 | 0.979117 |
| h2-ev16 | 124 | 0.279802 | 0.980092 | 0.844723 | 0.996422 |
| h2-ev16 | 125 | 0.273887 | 0.980463 | 0.846758 | 0.997148 |
| h1-ev8 | 123 | 0.293875 | 0.973618 | 0.802078 | 0.995227 |
| h1-ev8 | 124 | 0.268032 | 0.981024 | 0.854195 | 0.997938 |
| h1-ev8 | 125 | 0.305587 | 0.970588 | 0.780473 | 0.992719 |

- h2-ev16 hard-slice mean over seeds 123/124/125: 0.798803.
- h1-ev8 hard-slice mean over seeds 123/124/125: 0.812249.
- h2-ev8 hard-slice mean over seeds 123/124/125/126/127: 0.828155.
- h2-ev16 stable weak vs h2-ev8: false. s124/s125 recover to h2-ev8-level, while s123 is the weak outlier.
- h1-ev8 stronger than h2-ev16 in the 131k capacity tier by 3-seed mean: true, but not consistently per seed.

## h2 Capacity Curve
| h2 expand_v | capacity/layer | capacity total | params | loss | acc | 1024x256 | 512x128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 16384 | 32768 | 1167878 | 0.347751 | 0.962247 | 0.711828 | 0.987602 |
| 6 | 49152 | 98304 | 1301254 | 0.296228 | 0.975709 | 0.811367 | 0.994789 |
| 8 | 65536 | 131072 | 1367942 | 0.289848 | 0.978623 | 0.831684 | 0.997656 |
| 10 | 81920 | 163840 | 1434630 | 0.275982 | 0.981971 | 0.858922 | 0.998203 |
| 12 | 98304 | 196608 | 1501318 | 0.291562 | 0.980175 | 0.844816 | 0.997586 |
| 16 | 131072 | 262144 | 1634694 | 0.408903 | 0.960377 | 0.704930 | 0.979117 |

- h2 curve monotonic on seed 123: false.
- h2 curve best seed-123 point: h2-ev10 with 1024x256=0.858922.
- Conclusion: expand_v 越大不一定越好; capacity-up is non-monotonic when a larger expand_v lowers the hard-slice score.

## Best Of Family
| rank | config | seed | 1024x256 | acc | loss | source |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | h2-ev10 | 123 | 0.858922 | 0.981971 | 0.275982 | 20260520_gdn_capacity_layout_followup |
| 2 | h1-ev8 | 124 | 0.854195 | 0.981024 | 0.268032 | 20260520_gdn_capacity_layout_followup |
| 3 | h2-ev8 | 125 | 0.847398 | 0.980528 | 0.283829 | 20260519_gd_r10_gdn_official_b64_fp32 |
| 4 | h2-ev16 | 125 | 0.846758 | 0.980463 | 0.273887 | 20260520_gdn_capacity_layout_followup |
| 5 | h2-ev12 | 123 | 0.844816 | 0.980175 | 0.291562 | 20260520_gdn_capacity_layout_followup |
| 6 | h2-ev16 | 124 | 0.844723 | 0.980092 | 0.279802 | 20260520_gdn_capacity_layout_followup |
| 7 | h2-ev8 | 123 | 0.831684 | 0.978623 | 0.289848 | 20260519_gd_r10_gdn_official_b64_fp32 |
| 8 | h2-ev8 | 126 | 0.831566 | 0.978413 | 0.286705 | 20260520_gdn_capacity_layout_followup |

- Current best GDN single run by 1024x256: h2-ev10, seed 123, 1024x256=0.858922, acc=0.981971.
- Recommended best GDN baseline for future Flash-VQG capacity-fair comparisons: use h2-ev8 as the current multi-seed baseline, and treat h2-ev10 as the new best single-run candidate that needs s124/s125 follow-up before promotion.

## Answers
1. h2-ev8 新增 s126/s127 后仍稳定. 5-seed 1024x256 mean=0.828155, std(pop)=0.013317, range=0.040980.
2. h2-ev16 的 s124/s125 没有继续弱于 h2-ev8. h2-ev16 3-seed hard mean=0.798803, 低于 h2-ev8 5-seed hard mean=0.828155, 但这是因为 s123=0.704930 明显偏低; s124/s125 分别为 0.844723/0.846758, 已恢复到 h2-ev8 附近. 因此 ev16 变弱不稳定.
3. h1-ev8 按 3-seed mean 强于 h2-ev16, 0.812249 vs 0.798803, 但不是逐 seed 稳定强于. 这说明 131k 容量档里 head layout 是关键变量, 但还不能简单判定 h1-ev8 全面优于 h2-ev16.
4. h2-ev6/ev10/ev12 与 h2-ev8 的 seed-123 曲线显示当前 seed-123 sweet spot 是 h2-ev10, 不是 h2-ev8. h2-ev10 需要补 s124/s125 才能作为稳定 baseline.
5. GDN capacity-up 不呈单调趋势: true. expand_v 增大后可能变弱.
6. 当前最强 GDN best-of-family single run 是 h2-ev10-s123, hard=0.858922. 当前最强 multi-seed baseline 仍是 h2-ev8.
7. 后续和 Flash-VQG 做容量公平比较时, GDN 应先选 h2-ev8 作为已固化 multi-seed baseline, 同时补 h2-ev10 的 s124/s125 来判断是否升级 best GDN baseline.
8. 本报告所有 completed 结果属于 b64_ga4 + fp32 official GDN scope, 不和 128x2, auto-fp16, probe 结果混表.

## Artifact Files
- `gdn-followup-final.csv`
- `gdn-followup-epoch-end-valid.csv`
- `gdn-followup-slice-level.csv`
- `gdn-followup-model-capacity-and-params.csv`
- `gdn-followup-combined-with-prior.csv`
- `gdn-followup-run-manifest.json`
