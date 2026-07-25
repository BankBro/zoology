# Longer-MQAR eval-only artifact 说明

本目录保存 checkpoint-only MQAR 长度外推评估结果. 它与 Flash-VQG 和 GDN 训练 canonical ledger 分开维护.

## 当前推荐引用结果

- `official-core-20260526/`: 当前推荐引用的 RNG-locked official core eval 结果.
- `official-core-20260526/longer-mqar-official-core-detail.csv`: official core 逐条 eval ledger, 包含 formal, repro 和 batch-search 记录.
- `official-core-20260526/longer-mqar-official-core-summary.csv`: official core 聚合总表.
- `official-core-20260526/status.csv`: official core formal/repro 状态表.
- `official-core-20260526/verification.json`: official core 完成性, dataset_hash 和 repro 验证摘要.
- `official-core-20260526/manifest.csv` 和 `manifest.json`: 19 个 official core checkpoint 来源清单.
- `kblocked-gdn-20260528/`: FLA K-blocked GDN research evidence. 该目录用于记录 true expanded-K GDN 的 longer-MQAR 结果, 但不属于 official core 推荐引用集.

## 当前基线重训对照

- [当前基线 artifact](../20260725-01-current-baselines-longer-mqar/README.md): 当前 Flash `baseline-r16-joint` 与 GDN `gdnxk-h2-ek4-ev4-usegate0` 在2080 Ti和3090上的独立三seed 4ep重训及跨GPU Longer-MQAR对照.
- [对应报告](../../20260725-01-current-baselines-longer-mqar-report.md).
- 本轮按机器独立保存last/best逻辑结果, 合并表不把同seed跨GPU结果汇总成`n=6`. 它没有改写`official-core-20260526/`或旧preliminary ledger, 也不把2026-05历史模型混入排名.

official core 结果使用 `eval_seed=123`, 在 dataset 生成前固定 `random`, `numpy`, `torch`, `torch.cuda` RNG, 并为每个 slice 记录 `dataset_hash`. 所有纳入 official core 的 checkpoint 均为 `b64_ga4 fp32 official` 训练口径: `source_train_batch_size=64`, `source_gradient_accumulation_steps=4`, `source_effective_train_batch_size=256`, `source_batch_accum_profile=b64_ga4`, `source_dtype_policy=float32`.

## 旧 preliminary 结果

- `longer-mqar-eval-summary.csv`: 旧 preliminary eval ledger. 该表来自 2026-05-20/2026-05-21 附近的 backfill 和 eval attempt, 未锁定 dataset 生成前 RNG, 且不记录 `dataset_hash`. 它保留用于历史追溯和与 official core 对比, 不作为当前推荐引用结果.

不要把 `official-core-20260526/` 的新结果追加回 `longer-mqar-eval-summary.csv`, 也不要写入 `gd-residual-v1` 或 `gdn` 训练 canonical ledger.

K-blocked GDN 结果应引用 `kblocked-gdn-20260528/` 和 `docs/artifacts/20260528-fla-kblocked-gdn-kernel/`. 由于 upstream-ready correctness 仍为 no-go, 这些结果只能作为 research evidence, 不应和 official core 混写成同一推荐总表.

## runner 路径

- `zoology/experiments/flash_vqg/scripts/20260521-longer-mqar-canonical/longer_mqar_eval_runner.py`

该 runner 是当前 longer-MQAR eval-only official core 的执行脚本. 本 artifact 目录只保存生成后的 ledger, summary, manifest 和验证文件, 不保存长期维护的执行脚本副本.

## 行语义

- `run_type` 正常为 `longer_mqar_eval_only`.
- `eval_event_id` 是单条 eval event 的稳定身份.
- `eval_batch_id` 标识同一批 eval.
- `source_*` 字段描述 checkpoint 来源训练 run 和 checkpoint 身份.
- `source_ckpt_sha256` 是 checkpoint 内容身份. 路径可能跨机器变化, hash 不应变化.
- `source_train_config_sha256` 记录训练配置内容 hash.
- `eval_protocol_id` 标识 eval dataset, slice 和 sample count.
- `eval_batch_size` 是执行吞吐设置, 不是任务定义. 质量比较应使用一致的 `source_ckpt_sha256`, `eval_protocol_id` 和 `eval_status=completed`.
- `eval_hardware_profile_id`, `gpu`, `cuda_device`, `gpu_name`, `peak_memory_mb` 等字段用于硬件和运行环境追溯. wall-clock 和 peak memory 只应在相同或可比硬件 profile 下比较.
- `dataset_hash` 是 slice dataset 的内容 hash. official core 要求同一 slice 在不同 checkpoint 上 hash 一致.
- `checkpoint_hash` 是本次 eval 使用的 checkpoint 内容 hash.
- `official_core_constraint_status=passed` 表示该 source 已通过 official core 约束筛选.
- `selected_core_subset=true` 表示该 source 属于本轮预先定义的 official core subset.

## batch-search 语义

Adaptive batch search 是硬件相关的吞吐探测. `status.csv` 只统计 formal/repro 完成状态. 受控 batch-search candidate OOM 或失败可以出现在 detail ledger 的 batch-search failure 字段中, 但不等价于 formal eval 失败. 当前 official core formal 和 repro 均为 completed.

未来 adaptive run 应继续记录:

- `batch_search_status`
- `batch_search_slice`
- `batch_search_candidates`
- `batch_search_best_eval_batch_size`
- `batch_search_peak_memory_mb`
- `batch_search_hardware_profile_id`
- `batch_search_reusable_scope=same_gpu_same_dtype_same_runner_only`

## official core 聚合口径

- 只使用 `official-core-20260526/longer-mqar-official-core-detail.csv` 中 `eval_mode=formal` 且 `eval_status=completed` 的行.
- 只使用 `source_scope=b64_ga4_fp32_official`, `source_batch_accum_profile=b64_ga4`, `source_dtype_policy=float32` 的 checkpoint.
- 每个 `source_run_id x slice` 只有一条 formal 结果.
- 对同一 config group 下的 checkpoint seed 做均值和 population std.
- 单 checkpoint config group 的 std 为空.

| config | family | n | seeds | active cap | params | 1024x256 | 2048x512 | 4096x1024 | 8190x512 | 8190x2047 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cb256-r10 | flash | 4 | 123,124,125,126 | 327,680 | 1.184M | 0.9142 ± 0.0824 | 0.7080 ± 0.2360 | 0.4589 ± 0.2716 | 0.6232 ± 0.2605 | 0.2328 ± 0.1658 |
| cb256-r4 | flash | 1 | 123 | 131,072 | 1.183M | 0.8942 | 0.6667 | 0.3807 | 0.5667 | 0.1717 |
| cb64-r16 | flash | 1 | 123 | 131,072 | 1.160M | 0.9696 | 0.8226 | 0.4689 | 0.7173 | 0.1623 |
| gdn-h2-ev10 | gdn | 5 | 123,124,125,126,127 | 81,920 | 1.435M | 0.8366 ± 0.0152 | 0.3474 ± 0.0177 | 0.0775 ± 0.0151 | 0.2283 ± 0.0157 | 0.0065 ± 0.0027 |
| gdn-h2-ev16 | gdn | 3 | 123,124,125 | 131,072 | 1.635M | 0.7999 ± 0.0675 | 0.3549 ± 0.0488 | 0.1073 ± 0.0164 | 0.2491 ± 0.0285 | 0.0159 ± 0.0026 |
| gdn-h2-ev8 | gdn | 5 | 123,124,125,126,127 | 65,536 | 1.368M | 0.8299 ± 0.0130 | 0.3610 ± 0.0117 | 0.0974 ± 0.0130 | 0.2493 ± 0.0176 | 0.0112 ± 0.0040 |
