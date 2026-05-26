# 20260526 Longer-MQAR official core 评估报告

## 摘要

本轮实验把 longer-MQAR eval-only preliminary 结果升级为 RNG-locked official core 结果. runner 已在 dataset 生成前固定 `random`, `numpy`, `torch`, `torch.cuda` RNG, 每个 eval slice 都记录 `dataset_hash`. 结果写入 `docs/artifacts/longer-mqar/official-core-20260526/`, 未追加到旧 preliminary 表.

本轮只评估 core subset: Flash `cb256-r10` seeds 123/124/125/126, Flash `cb256-r4` seed 123, Flash `cb64-r16` seed 123, GDN `h2-ev8` seeds 123/124/125/126/127, GDN `h2-ev10` seeds 123/124/125/126/127, GDN `h2-ev16` seeds 123/124/125. 全部 checkpoint 均满足 `b64_ga4 fp32 official` 训练口径: `train_batch_size=64`, `gradient_accumulation_steps=4`, `effective_train_batch_size=256`, `batch_accum_profile=b64_ga4`, `dtype_policy=float32`.

## artifact

- 逐条结果: `docs/artifacts/longer-mqar/official-core-20260526/longer-mqar-official-core-detail.csv`
- 聚合总表: `docs/artifacts/longer-mqar/official-core-20260526/longer-mqar-official-core-summary.csv`
- 状态表: `docs/artifacts/longer-mqar/official-core-20260526/status.csv`
- 验证摘要: `docs/artifacts/longer-mqar/official-core-20260526/verification.json`
- checkpoint manifest: `docs/artifacts/longer-mqar/official-core-20260526/manifest.csv`

## official core 结果

数值为 accuracy mean +/- population std. 单 checkpoint 组只列 mean.

| config | family | n | seeds | active cap | params | 1024x256 | 2048x512 | 4096x1024 | 8190x512 | 8190x2047 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cb256-r10 | flash | 4 | 123,124,125,126 | 327,680 | 1.184M | 0.9142 +/- 0.0824 | 0.7080 +/- 0.2360 | 0.4589 +/- 0.2716 | 0.6232 +/- 0.2605 | 0.2328 +/- 0.1658 |
| cb256-r4 | flash | 1 | 123 | 131,072 | 1.183M | 0.8942 | 0.6667 | 0.3807 | 0.5667 | 0.1717 |
| cb64-r16 | flash | 1 | 123 | 131,072 | 1.160M | 0.9696 | 0.8226 | 0.4689 | 0.7173 | 0.1623 |
| gdn-h2-ev8 | gdn | 5 | 123,124,125,126,127 | 65,536 | 1.368M | 0.8299 +/- 0.0130 | 0.3610 +/- 0.0117 | 0.0974 +/- 0.0130 | 0.2493 +/- 0.0176 | 0.0112 +/- 0.0040 |
| gdn-h2-ev10 | gdn | 5 | 123,124,125,126,127 | 81,920 | 1.435M | 0.8366 +/- 0.0152 | 0.3474 +/- 0.0177 | 0.0775 +/- 0.0151 | 0.2283 +/- 0.0157 | 0.0065 +/- 0.0027 |
| gdn-h2-ev16 | gdn | 3 | 123,124,125 | 131,072 | 1.635M | 0.7999 +/- 0.0675 | 0.3549 +/- 0.0488 | 0.1073 +/- 0.0164 | 0.2491 +/- 0.0285 | 0.0159 +/- 0.0026 |

主要结论不变: Flash 在 longer-MQAR core slices 上显著优于 GDN. `cb64-r16` 在 1024/2048 和 8190x512 上是当前单 checkpoint 最强 Flash 配置, `cb256-r10` 在最高 key density 的 8190x2047 上均值最高. GDN 三组在 4096 以上退化明显, `h2-ev16` 在 8190x2047 上略好于 `ev8/ev10`, 但整体仍远低于 Flash.

## 完成性和复现验证

- formal eval: 19 checkpoint/config x 5 slices = 95 条, 全部 `completed`.
- repro eval: 19 条, 全部 `completed`.
- formal 失败/OOM/missing checkpoint: 0 条.
- 使用 GPU: RTX 2080 Ti GPU0 跑 50 条 formal, GPU1 跑 45 条 formal.
- `eval_seed`: 全部为 123.
- `source_scope`: 全部为 `b64_ga4_fp32_official`.
- `source_batch_accum_profile`: 全部为 `b64_ga4`.
- `source_dtype_policy`: 全部为 `float32`.
- `official_core_constraint_status`: 全部为 `passed`.
- `selected_core_subset`: 全部为 `true`.

dataset hash 验证通过. 每个 slice 在所有 checkpoint 上只有一个 dataset hash:

| slice | unique dataset_hash | hash |
|---|---:|---|
| 1024x256 | 1 | `f30f1e09e1300deb2e0f430d7c50c47a77f58f43657aa4d6f425f938a91a9acb` |
| 2048x512 | 1 | `e446efc9f4dc774377c56c403a468c747870beb18ff0320f15df794dcb69f015` |
| 4096x1024 | 1 | `0981292ce8ebea1402a4c3a7a6655dfc1de42688c329bd5ea921541a0f1d16ed` |
| 8190x512 | 1 | `37a8533a985fabd3db03daf9dfd3a8e0936513ab724039c63b9a22bf352d002d` |
| 8190x2047 | 1 | `8c16a91ea127bb7bf2130238ecd56b49fb1fd73bb3e0c2f4c586ddba8e9b97a9` |

repro 验证通过: 19/19 的 `repro_check_dataset_hash_match=true`, 19/19 的 `repro_check_accuracy_match=true`.

## 与 preliminary 的差异

旧表 `docs/artifacts/longer-mqar/longer-mqar-eval-summary.csv` 保留为 preliminary. 它未在 dataset 生成前锁定 RNG, 不记录 `dataset_hash`, 且包含重复 eval attempt 和非本轮 core subset 行. 本报告对比时只取与 official core 相同的 19 个 source 和 5 个 slices, 并对 preliminary 中重复 attempt 先按 `source_run_id x slice` 去重平均.

| config | 最大均值差异 | 发生 slice |
|---|---:|---|
| cb256-r10 | +0.000551 | 1024x256 |
| cb256-r4 | -0.001187 | 2048x512 |
| cb64-r16 | -0.001066 | 8190x512 |
| gdn-h2-ev8 | +0.000768 | 8190x512 |
| gdn-h2-ev10 | +0.000570 | 1024x256 |
| gdn-h2-ev16 | +0.002456 | 1024x256 |

这些差异都很小, 符合 RNG 未锁定时不同 eval dataset 的轻微波动. official core 应作为后续引用口径, preliminary 仅用于历史追溯.

## batch-search 和异常说明

本轮 formal/repro 没有失败. adaptive batch-search 共 95 条, 均完成并选择可用 `eval_batch_size`. 其中 42 条记录了较大 batch candidate 的受控 OOM/fallback, 这是 batch-size 探测的一部分, 不计为 formal eval 失败. 最终 formal 的 `eval_batch_size` 主要为短 slice 的 32 和长 slice 的 16.

## 后续建议

下一步不建议继续扩大 longer-MQAR eval-only 表, 除非有新的训练 checkpoint 或新的 slice 设计. 更值得推进的是针对当前结论做机制实验: 对 Flash `cb64-r16` 和 `cb256-r10` 做更细的 density sweep, 对 GDN 做容量和 memory 更新机制诊断, 并在任何新正式对比中沿用本轮 runner 的 RNG lock, `dataset_hash`, checkpoint hash 和 b64_ga4/source dtype 约束.
