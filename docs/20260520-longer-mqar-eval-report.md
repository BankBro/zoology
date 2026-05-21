# 2026-05-20 longer-MQAR eval-only report

## Scope

- 本报告是 eval-only 长度外推, 没有训练新模型, 没有修改核心模型源码, 没有重跑 GDN/Flash 训练.
- 所有 eval 使用 epoch4 `last.pt` final checkpoint, `run_type=longer_mqar_eval_only`.
- Smoke: num_examples=16, eval_batch_size=1, slices=[(1024, 256), (2048, 512), (4096, 512)].
- Formal: num_examples=500, eval_batch_size=1, slices=[(1024, 256), (2048, 256), (2048, 512), (4096, 512)]. 由于 12 个 ckpt 全量 formal 预计过长, formal 按用户给定优先级保留 5 个代表 ckpt.
- GDN eval 显式设置 `GDN_KERNEL_DTYPE=float32`.

## Artifacts

- `docs/artifacts/20260520-longer-mqar-eval/longer-mqar-smoke-final.csv`
- `docs/artifacts/20260520-longer-mqar-eval/longer-mqar-formal-final.csv`
- `docs/artifacts/20260520-longer-mqar-eval/longer-mqar-slice-level.csv`
- `docs/artifacts/20260520-longer-mqar-eval/longer-mqar-ckpt-manifest.json`
- `docs/artifacts/20260520-longer-mqar-eval/longer-mqar-run-manifest.json`
- `docs/artifacts/20260520-longer-mqar-eval/longer-mqar-oom-or-missing-ckpt.csv`

## Smoke

| run | status | 1024x256 | 2048x512 | 4096x512 | peak MiB | wall-clock |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gd-r10-s125-d123-b64-ga4-fp32-noearly4ep` | completed | 0.987305 | 0.942627 | 0.918945 | 1023 | 00:01:09 |
| `gd-r10-s124-d123-b64-ga4-fp32-noearly4ep` | completed | 0.986084 | 0.919922 | 0.885498 | 1011 | 00:01:12 |
| `gd-r10-s126-d123-b64-ga4-fp32-noearly4ep` | completed | 0.780273 | 0.369507 | 0.299438 | 1023 | 00:01:35 |
| `gd-cb256-r4-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 0.896484 | 0.664429 | 0.619141 | 753 | 00:01:41 |
| `gd-cb128-r8-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 0.782959 | 0.410156 | 0.355713 | 767 | 00:01:26 |
| `gd-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 0.969482 | 0.821411 | 0.782593 | 1015 | 00:01:06 |
| `gd-cb128-r10-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 0.709229 | 0.270264 | 0.217041 | 753 | 00:01:54 |
| `gdn-h2-ev8-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 0.839600 | 0.367920 | 0.322266 | 1391 | 00:00:48 |
| `gdn-h2-ev8-s125-d123-b64-ga4-fp32-noearly4ep` | completed | 0.852539 | 0.364136 | 0.305542 | 1391 | 00:00:32 |
| `gdn-h2-ev10-s123-d123-b64-ga4-fp32-noearly4ep` | completed | 0.854736 | 0.366211 | 0.282227 | 1387 | 00:00:37 |
| `gdn-h2-ev10-s124-d123-b64-ga4-fp32-noearly4ep` | completed | 0.827393 | 0.378540 | 0.322144 | 1373 | 00:00:30 |
| `gdn-h2-ev10-s127-d123-b64-ga4-fp32-noearly4ep` | completed | 0.836182 | 0.345947 | 0.285034 | 1387 | 00:00:30 |

## Formal

| run | role | status | 1024x256 | 2048x256 | 2048x512 | 4096x512 | peak MiB | wall-clock |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gd-r10-s125-d123-b64-ga4-fp32-noearly4ep` | flash_practical_strong | completed | 0.990383 | 0.984109 | 0.944473 | 0.926824 | 1025 | 00:39:31 |
| `gd-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep` | flash_capacity_matched_131k | completed | 0.968906 | 0.956484 | 0.823063 | 0.781027 | 1015 | 00:37:24 |
| `gd-cb128-r10-s123-d123-b64-ga4-fp32-noearly4ep` | flash_163k_candidate | completed | 0.704180 | 0.625180 | 0.266492 | 0.222008 | 755 | 01:05:59 |
| `gdn-h2-ev8-s125-d123-b64-ga4-fp32-noearly4ep` | gdn_stable_baseline | completed | 0.847117 | 0.793359 | 0.358199 | 0.301301 | 1391 | 00:00:41 |
| `gdn-h2-ev10-s123-d123-b64-ga4-fp32-noearly4ep` | gdn_upper_candidate | completed | 0.860078 | 0.791977 | 0.364219 | 0.291187 | 1387 | 00:00:39 |

## Answers

1. Smoke 是否全部通过: 是, OOM 数量=0.
2. Formal eval 参与 ckpt: `gd-r10-s125-d123-b64-ga4-fp32-noearly4ep`, `gd-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep`, `gd-cb128-r10-s123-d123-b64-ga4-fp32-noearly4ep`, `gdn-h2-ev8-s125-d123-b64-ga4-fp32-noearly4ep`, `gdn-h2-ev10-s123-d123-b64-ga4-fp32-noearly4ep`.
3. 1024x256 sanity check: `gd-r10-s125-d123-b64-ga4-fp32-noearly4ep` delta=-0.000348; `gd-cb64-r16-s123-d123-b64-ga4-fp32-noearly4ep` delta=+0.000195; `gd-cb128-r10-s123-d123-b64-ga4-fp32-noearly4ep` delta=+0.000020; `gdn-h2-ev8-s125-d123-b64-ga4-fp32-noearly4ep` delta=-0.000281; `gdn-h2-ev10-s123-d123-b64-ga4-fp32-noearly4ep` delta=+0.001156.
4. 2048/4096 长度衰减: Flash practical `r10-s125` 在 4096x512=0.926824, GDN upper `h2-ev10-s123` 在 4096x512=0.291187.
5. Capacity-matched 对比: Flash 131k best `cb64-r16` 在 4096x512=0.781027, GDN stable `h2-ev8-s125` 在 4096x512=0.301301.
6. Flash 在长长度仍明显领先于 GDN upper candidate, 这支持 VQ-indexed residual memory 的长程关联回忆优势, 但仍是 eval-only 外推证据.
7. 本报告明确是 eval-only 外推结果, 不是重新训练结果.
