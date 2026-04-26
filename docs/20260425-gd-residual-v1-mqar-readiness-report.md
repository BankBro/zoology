# gd_residual_v1 MQAR Readiness 报告

日期: 2026-04-26

## 1. 结论

本轮已把 PyTorch reference 路径推进到可复现实验和诊断状态: diagnostics opt-in, by-layer metrics, zoology 初始化检查, phase/event profile, 真实 MQAR short-run runner, JSON summary/JSONL records, whitelist 和 metrics.yaml 均已补齐. 本轮未改 legacy / `clr_v1` / `clr_delta_v1`, 未写 Triton/CUDA/custom backward.

Readiness 判定: 暂不达标, 不建议直接启动最终正式实验.

依据:

- `20` train batches x seeds `123,124`, batch=8, A/B/C + `local_only` + `legacy_fox` 全部完成, 无 OOM, 无 NaN/Inf.
- gd variants train loss 有下降, `inject_ratio` 稳定非零, `lambda_mean≈0.05`, `m_norm_max` 未爆炸, `L_state` 有非零累积.
- 但所有 run 的 `valid_accuracy=0.0`, gd variants `mu_valid_ratio=0.0`, valid loss 与 baseline 基本相同, 且当前 runner 只记录最终 valid, 还不能证明 valid loss 改善.
- baseline train loss 下降幅度明显大于 gd variants, 因此不满足 "启动最终正式实验" 的质量信号门槛.

## 2. 本轮改动

Flash-VQG:

- 保留 `FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS=1` 为唯一 debug 开关.
- 增加 `attn/gd_residual_debug_avg_events_per_group` alias, 与 `attn/gd_residual_debug_mean_events_per_group` 保持一致.
- 测试覆盖 diagnostics 默认关闭和开启状态, 并确认 `layer_0/attn/gd_residual_mu_valid_ratio` 与 `layer_0/attn/gd_residual_inject_ratio` 会进入 metrics.

zoology:

- `LanguageModel` wrapper 测试同时检查 `sigmoid(beta_proj.bias)=0.5` 和 `sigmoid(lambda_proj.bias)=0.05`.
- `profile_gd_residual_v1.py` 保持 `metrics_collect_sec`, 并支持 `PROFILE_ENABLE_GD_DIAGNOSTICS`.
- 新增 `short_run_gd_residual_v1.py` 和 `run_short_run.sh`, 支持 `SHORT_RUN_TRAIN_BATCHES`, `SHORT_RUN_SEEDS`, `SHORT_RUN_VARIANT`, `SHORT_RUN_OUTPUT_DIR`, `PROFILE_ENABLE_GD_DIAGNOSTICS`.
- short-run JSONL 每条 record 写 train/valid loss, accuracy, forward/backward/optimizer/metrics time, memory peak, gd metrics, layer metrics, event diagnostics, `L_state` diagnostics, NaN/Inf 状态.
- `metrics.yaml` 和 whitelist 增加 debug metrics, alias, `layer_*` 和 `valid/layer_*` patterns.

## 3. 测试结果

| 仓库 | 命令 | 结果 |
| --- | --- | --- |
| Flash-VQG | `pytest tests/test_fox_gd_residual_v1.py -q` | `11 passed in 0.23s` |
| Flash-VQG | `pytest tests/test_fox_guards.py tests/test_fox_dense_write.py tests/test_fox_clr_delta_v1.py tests/test_fox_phase2_metrics.py -q` | `66 passed in 18.71s` |
| zoology | `pytest tests/test_flash_vqg_wrapper.py tests/test_flash_vqg_scripts.py tests/test_flash_vqg_metrics_white_list.py tests/test_train_logging_steps.py tests/test_train_batch_order.py -q` | `77 passed, 1 warning in 11.59s` |

## 4. 诊断字段说明

| 字段 | 含义 |
| --- | --- |
| `attn/gd_residual_debug_event_pack_wall_sec` | event pack 同步 wall-time |
| `attn/gd_residual_debug_grouped_chunk_wall_sec` | grouped chunk reference 同步 wall-time |
| `attn/gd_residual_debug_phase2_residual_read_wall_sec` | phase2 residual read 同步 wall-time |
| `attn/gd_residual_debug_event_count` | 当前 batch 写入 event 数 |
| `attn/gd_residual_debug_group_count` | grouped chunk 分组数 |
| `attn/gd_residual_debug_mean_events_per_group` | 平均每组 event 数 |
| `attn/gd_residual_debug_avg_events_per_group` | mean alias, 便于 analysis 兼容 |
| `attn/gd_residual_debug_l_state_*` | coarse memory `L_state` 有效性统计 |
| `layer_*/attn/gd_residual_*` | layer-level metrics, 当前 MQAR 配置主要为 `layer_1` |

这些字段仅在 `collect_metrics=True` 且 `FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS=1` 时计算. 默认 diagnostics 关闭时不产生 debug metrics.

## 5. Phase Wall-Time

配置: `read_topk=2`, `rank=16`, `write_topk=4`, batch=8, seq_len=128, `dense_softmax`, `grouped_chunk_torch_ref`, `semivec_ref`, diagnostics=1. 表中均值排除第 0 个 warm-up microbatch.

| 项 | 数值 |
| --- | ---: |
| forward_sec | 2.427 |
| backward_sec | 4.989 |
| optimizer_sec | 0.006 |
| metrics_collect_sec | 0.006 |
| microbatch_sec | 7.428 |
| event_pack_wall_sec | 1.029 |
| grouped_chunk_wall_sec | 1.278 |
| phase2_residual_read_wall_sec | 0.00054 |
| event_count | 8192 |
| group_count | 4606 |
| mean_events_per_group | 1.779 |
| peak_allocated_GB | 1.277 |
| peak_reserved_GB | 1.646 |

结论: 当前耗时主要在 event pack/grouped chunk reference, phase2 residual read 不是主耗时.

## 6. Event Scaling

固定 `read_topk=2`, `rank=16`, batch=8, seq_len=128. 表中均值排除第 0 个 warm-up microbatch.

| write_topk | event_count | group_count | mean_events/group | event_pack_s | grouped_chunk_s | microbatch_s | peak_reserved_GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 1656 | 1.237 | 0.331 | 0.356 | 2.114 | 1.574 |
| 2 | 4096 | 2882 | 1.421 | 0.600 | 0.663 | 3.972 | 1.594 |
| 4 | 8192 | 4606 | 1.779 | 1.029 | 1.278 | 7.428 | 1.646 |

event_count 与 microbatch time 明显同向增长, 支持下一步优先优化 event_pack/grouped_chunk 的判断.

## 7. Short-Run Matrix

命令等价配置: `SHORT_RUN_TRAIN_BATCHES=20`, `SHORT_RUN_SEEDS=123,124`, batch=8, `read_topk=2`, diagnostics=1. 两个 GPU 并行分片执行: GPU0 跑 seed 123, GPU1 跑 seed 124.

| variant | runs | first_loss | last_loss | first5_loss | last5_loss | valid_loss | valid_acc | peak_alloc_GB | avg_step_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gd_r16_wk4` | 2/2 completed | 11.026 | 10.740 | 10.987 | 10.773 | 9.040 | 0.000 | 2.508 | 12.540 |
| `gd_r16_wk2` | 2/2 completed | 11.021 | 10.756 | 10.989 | 10.792 | 9.039 | 0.000 | 2.377 | 6.677 |
| `gd_r8_wk4` | 2/2 completed | 11.022 | 10.753 | 10.989 | 10.788 | 9.041 | 0.000 | 1.307 | 12.478 |
| `local_only` | 2/2 completed | 11.787 | 10.328 | 11.164 | 10.393 | 9.039 | 0.000 | 0.534 | 0.016 |
| `legacy_fox` | 2/2 completed | 11.787 | 10.329 | 11.164 | 10.393 | 9.039 | 0.000 | 0.534 | 0.020 |

说明: `peak_reserved_GB` 在同进程多 variant 顺序运行中受 CUDA caching allocator 影响, 各 run 都保持在约 3.264GB high watermark. 上表使用 `peak_alloc_GB` 比较实际峰值分配.

## 8. GD Metrics 走势

表中为两个 seed 的最终 train record 均值.

| variant | inject_ratio | lambda_mean | mu_valid_ratio | m_norm_max | L_state_mean | L_state_frac_ge_0_1 | event_count | grouped_chunk_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gd_r16_wk4` | 0.0165 | 0.0502 | 0.0000 | 0.327 | 0.126 | 0.750 | 8192 | 1.111 |
| `gd_r16_wk2` | 0.0172 | 0.0505 | 0.0000 | 0.427 | 0.122 | 0.750 | 4096 | 0.576 |
| `gd_r8_wk4` | 0.0175 | 0.0504 | 0.0000 | 0.293 | 0.124 | 0.750 | 8192 | 1.147 |

最终 valid record 均值:

| variant | valid_inject_ratio | valid_lambda_mean | valid_mu_valid_ratio | valid_m_norm_max | valid_L_state_mean | valid_L_state_frac_ge_0_1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gd_r16_wk4` | 0.0202 | 0.0503 | 0.0000 | 0.338 | 0.129 | 0.746 |
| `gd_r16_wk2` | 0.0209 | 0.0504 | 0.0000 | 0.450 | 0.123 | 0.746 |
| `gd_r8_wk4` | 0.0227 | 0.0505 | 0.0000 | 0.320 | 0.126 | 0.746 |

解读:

- `inject_ratio` 非零, 说明 residual path 确实参与输出.
- `lambda_mean` 没有塌缩, 与初始化附近一致.
- `m_norm_max` 在 0.29-0.45, 没有爆炸.
- `L_state` mean 和 `frac_ge_0_1` 显示 coarse memory 有累积.
- `mu_valid_ratio=0.0` 持续存在, 这是当前 readiness 的关键阻塞信号.

## 9. 产物位置

Profile:

- `tmp/20260425-gd-residual-v1-profile/phase-wk4/summary.json`
- `tmp/20260425-gd-residual-v1-profile/scaling-wk1/summary.json`
- `tmp/20260425-gd-residual-v1-profile/scaling-wk2/summary.json`

Short-run:

- `tmp/20260425-gd-residual-v1-short-run-gpu0-s123/summary.json`
- `tmp/20260425-gd-residual-v1-short-run-gpu0-s123/records.jsonl`
- `tmp/20260425-gd-residual-v1-short-run-gpu1-s124/summary.json`
- `tmp/20260425-gd-residual-v1-short-run-gpu1-s124/records.jsonl`

## 10. 回退方式

- 不启用 diagnostics: unset `PROFILE_ENABLE_GD_DIAGNOSTICS` 或设为 `0`; Flash-VQG debug metrics 不会计算.
- 不跑 short-run: 不调用 `run_short_run.sh` 或 `short_run_gd_residual_v1.py`.
- 代码回退范围:
  - Flash-VQG: `src/flash_vqg/nn/attn_fox.py`, `tests/test_fox_gd_residual_v1.py`.
  - zoology: wrapper test, scripts test, whitelist, metrics.yaml, README, `profile_gd_residual_v1.py`, `run_profile.sh`, `short_run_gd_residual_v1.py`, `run_short_run.sh`, 本报告.

## 11. 下一步决策

不进入最终正式实验. 建议下一步只做一个更聚焦的 readiness 修正循环:

1. short-run runner 增加 initial valid 和 periodic valid, 让 valid loss improvement 可判断.
2. 针对 `mu_valid_ratio=0` 做最小诊断, 确认是任务早期自然现象, mask/validity 条件过严, 还是 gd residual state 构造问题.
3. 若 `mu_valid_ratio` 能转为非零且 valid loss 出现改善, 再进入 PyTorch 侧 event_pack/grouped_chunk 等价优化.
