# GDN h2-ev10 Multi-seed Follow-up MQAR Report

- Generated: 2026-05-20T12:39:46.984816+00:00
- Scope: `b64_ga4_fp32_official` only, `model=gated_delta_net`, `use_gate=false`, `use_short_conv=true`, `conv_size=4`, `num_heads=2`, `expand_v=10`, `TRAIN_BATCH_SIZE=64`, `GRADIENT_ACCUMULATION_STEPS=4`, `EVAL_BATCH_SIZE=16`, `MAX_EPOCHS=4`, early stopping disabled, `DATA_SEED=123`, `DMODEL=128`, `LR=1e-3`, `GDN_KERNEL_DTYPE=float32`.
- New artifacts: `docs/artifacts/20260520-gdn-h2-ev10-multiseed-followup/`.
- References: `docs/20260520-gdn-capacity-layout-followup-report.md` and `docs/artifacts/20260520-gdn-capacity-layout-followup` for h2-ev8 seeds 123/124/125/126/127 and h2-ev10 seed123.
- Canonical GDN ledger 已包含本轮 4 条 completed final 行; 后续重建 artifact 时按 `run_id` 去重, 不会重复追加.
- 可选 `gdn-h2-ev10-s123-repeat` 未运行; 五 seed 统计直接引用已有 official h2-ev10 seed123 结果.

## New Completed Runs
| run | status | wall-clock | loss | acc | 1024x256 | 512x128 | peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gdn-h2-ev10-s124-d123-b64-ga4-fp32-noearly4ep | completed | 00:16:41 | 0.281057 | 0.979341 | 0.837105 | 0.997812 | 3613 |
| gdn-h2-ev10-s125-d123-b64-ga4-fp32-noearly4ep | completed | 00:08:22 | 0.303185 | 0.977708 | 0.825699 | 0.996914 | 3599 |
| gdn-h2-ev10-s126-d123-b64-ga4-fp32-noearly4ep | completed | 00:10:50 | 0.305995 | 0.976080 | 0.812797 | 0.996281 | 3613 |
| gdn-h2-ev10-s127-d123-b64-ga4-fp32-noearly4ep | completed | 00:08:25 | 0.294408 | 0.979319 | 0.838344 | 0.997539 | 3599 |

## h2-ev10 Five-seed Statistics
| seed | loss | acc | 1024x256 | 512x128 | source |
| --- | --- | --- | --- | --- | --- |
| 123 | 0.275982 | 0.981971 | 0.858922 | 0.998203 | 20260520_gdn_capacity_layout_followup |
| 124 | 0.281057 | 0.979341 | 0.837105 | 0.997812 | 20260520_gdn_h2_ev10_multiseed_followup |
| 125 | 0.303185 | 0.977708 | 0.825699 | 0.996914 | 20260520_gdn_h2_ev10_multiseed_followup |
| 126 | 0.305995 | 0.976080 | 0.812797 | 0.996281 | 20260520_gdn_h2_ev10_multiseed_followup |
| 127 | 0.294408 | 0.979319 | 0.838344 | 0.997539 | 20260520_gdn_h2_ev10_multiseed_followup |

- valid/loss: n=5, mean=0.292125, std(pop)=0.011857, range=0.030013, min=0.275982, max=0.305995.
- valid/accuracy: n=5, mean=0.978884, std(pop)=0.001958, range=0.005891, min=0.976080, max=0.981971.
- valid/mqar_case/accuracy-1024x256: n=5, mean=0.834573, std(pop)=0.015274, range=0.046125, min=0.812797, max=0.858922.

## h2-ev10 vs h2-ev8
| config | n | loss mean | loss std | acc mean | acc std | 1024x256 mean | 1024x256 std | 1024x256 range | best 1024x256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2-ev8 | 5 | 0.290706 | 0.005874 | 0.978032 | 0.001772 | 0.828155 | 0.013317 | 0.040980 | 0.847398 |
| h2-ev10 | 5 | 0.292125 | 0.011857 | 0.978884 | 0.001958 | 0.834573 | 0.015274 | 0.046125 | 0.858922 |

## Paired Seed Hard-slice Comparison
| seed | h2-ev8 1024x256 | h2-ev10 1024x256 | ev10 - ev8 |
| --- | --- | --- | --- |
| 123 | 0.831684 | 0.858922 | 0.027238 |
| 124 | 0.823707 | 0.837105 | 0.013398 |
| 125 | 0.847398 | 0.825699 | -0.021699 |
| 126 | 0.831566 | 0.812797 | -0.018770 |
| 127 | 0.806418 | 0.838344 | 0.031926 |

- 1024x256 上 h2-ev10 相对 h2-ev8 的同 seed wins/losses: wins=3, losses=2, ties=0.
- h2-ev10 是否稳定强于 h2-ev8: false.
- h2-ev10 最强单 run: seed 123, 1024x256=0.858922.
- h2-ev8 最强单 run: seed 125, 1024x256=0.847398.

## Answers
1. h2-ev10 seeds 123/124/125/126/127 的 1024x256 mean=0.834573, std(pop)=0.015274, range=0.046125; valid/accuracy mean=0.978884, std(pop)=0.001958, range=0.005891; valid/loss mean=0.292125, std(pop)=0.011857, range=0.030013.
2. h2-ev10 是否稳定强于 h2-ev8: false. h2-ev10 1024x256 mean=0.834573 高于 h2-ev8 mean=0.828155, 但 paired seeds 中 wins=3, losses=2, 且 h2-ev10 std/range 更大, 所以不能判定稳定强于.
3. h2-ev10 是否可以升级为 best GDN baseline: false. 当前建议 best GDN baseline=h2-ev8 作为主稳定 baseline; h2-ev10 作为更高均值/上界候选.
4. h2-ev10 不是只有 s123 强: s124/s127 也强于 h2-ev8 对应 seed, 但 s125/s126 低于 h2-ev8. 因此继续保留 h2-ev8 作为稳定 baseline, 同时把 h2-ev10 作为更高均值/上界候选.
5. 后续 Flash-VQG 容量公平实验建议对齐: 主对齐 h2-ev8; 资源允许时同时报告 h2-ev10 作为 GDN 上界候选.
6. 本报告所有 completed 结果都属于 b64_ga4 + fp32 official GDN scope, 不和 128x2, auto-fp16, probe 结果混表.

## Artifact Files
- `gdn-h2-ev10-final.csv`
- `gdn-h2-ev10-epoch-end-valid.csv`
- `gdn-h2-ev10-slice-level.csv`
- `gdn-h2-ev10-combined-with-prior.csv`
- `gdn-h2-ev10-run-manifest.json`
