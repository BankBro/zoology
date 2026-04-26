# gd_residual_v1 MQAR 性能定位报告

## 1. 范围和结论

本轮只做 `gd_residual_v1` profiling, 低风险开关和短周期实验. 未改数学语义, 未改 legacy / `clr_v1` / `clr_delta_v1`, 未引入 Triton/CUDA/custom backward.

结论:

- 当前首要瓶颈是 `grouped_chunk_torch_ref` / event pack 的 PyTorch reference 实现, profiler 中 `gd_residual/grouped_chunk` 和 `gd_residual/event_pack` 合计占 B8/T128 细粒度 profile 的约 97% self CUDA time.
- `phase2_residual_read` 当前在 read_topk=2 主配置下不是主要耗时, 但 dense read 在 T=256,B=16 时把 reserved memory 从 5.58GB 拉到 10.46GB, 说明大张量 materialization 是明确显存风险.
- metrics gating 已修复: `enable_layer_metrics=False` 或 `fox_phase2_metrics_mode="off"` 时 gd residual metrics 不再扫描大状态.
- 2026-04-26 追加实现 opt-in diagnostics: `PROFILE_ENABLE_GD_DIAGNOSTICS=1` 时输出同步 wall-time phase timer, event 统计和 `L_state` 有效性统计. 默认关闭, 不增加常规训练扫描开销.
- 短 MQAR signal 有弱正向信号: all-segment short run loss `11.006 -> 10.975`, `inject_ratio=0.034`, `lambda_mean≈0.05`, `m_norm_max≈0.21`, 未出现 NaN/Inf. 但 valid accuracy 仍为 0, `mu_valid_ratio=0`, 不足以做正式质量结论.

建议:

- 继续小规模验证: 是, 但只做更长一点的短跑, 不进入大规模训练.
- 优先 Triton fused residual read: 暂不作为第一优先级. read_topk 已显著降低显存, phase2 read 时间不是当前主耗时. 若后续质量信号确认, 再做 fused residual read.
- 优先 grouped_chunk/event pack 优化: 是. 当前证据最强, 且 GPU 利用率粗采样平均约 22.7%, 符合 Python/event loop + 小 kernel 调度瓶颈.
- 下一步先补 phase wall-time 和 event_count scaling, 再进入 PyTorch 侧 event_pack / grouped_chunk 原型优化.

## 2. 环境和读过文件

仓库状态:

- Flash-VQG: branch `20260425-gd-residual-v1-codex`, commit `f4603bf92793005b4ad5dc8e8e062b6657824104`.
- zoology: branch `flash-vqg`, commit `667efbf2fc7e7f7fb76daa182c2fe80e45f979f4`.
- 两个仓库都有本轮未提交改动.

环境:

- Python `3.12.11`.
- PyTorch `2.6.0+cu118`.
- torch CUDA `11.8`, cuDNN `90100`.
- GPU: 2 x NVIDIA GeForce RTX 2080 Ti, driver `550.120`, 每张 11264MB.

实际读过的关键文件:

- `/home/lyj/mnt/project/Flash-VQG/docs/20260425-flash-vqg-gated-delta-v1-math-plan-final.md`
- `/home/lyj/mnt/project/Flash-VQG/docs/20260425-flash-vqg-gated-delta-v1-codex-blueprint.md`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/configuration_flash_vqg.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn_fox.py`
- `/home/lyj/mnt/project/Flash-VQG/tests/test_fox_gd_residual_v1.py`
- `zoology/mixers/flash_vqg.py`
- `zoology/experiments/models_repo.py`
- `zoology/train.py`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/*`

## 3. 低成本代码改动

- Flash-VQG:
  - 给 gd residual state build 和 phase2 residual read 增加 `collect_metrics`.
  - `collect_metrics = enable_layer_metrics and fox_phase2_metrics_mode != "off"`.
  - `collect_metrics=False` 时不计算 `lambda_mean`, `inject_ratio`, `write_strength`, `m_norm`, `mu_valid_ratio`.
  - 增加 profiler markers: `gd_residual/event_pack`, `gd_residual/grouped_chunk`, `gd_residual/build_metrics`, `gd_residual/phase2_residual_read`, `gd_residual/phase2_metrics`.
  - 追加 opt-in debug metrics: `event_pack` / `grouped_chunk` / `phase2_residual_read` 同步 wall-time, event/group 计数, `L_state` 有效性分布. 仅 `FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS=1` 时启用.
  - `FlashVQGAttention.__init__` 内按 config 初始化 beta/lambda projection, 并标记 `_flashvqg_custom_init`.

- zoology:
  - `_init_weights` 跳过带 `_flashvqg_custom_init` 的 Linear, 防止 wrapper 把 `lambda_init=0.05` 覆盖成 0 bias.
  - MQAR gd residual scripts 增加 `FOX_REMOTE_READ_TOPK`, 默认 `2`, 支持 `dense`.
  - run id 增加 read_topk/rank/write_topk/batch suffix.
  - 新增 `run_profile.sh` 和 `profile_gd_residual_v1.py`, 默认不连 SwanLab, 不保存 checkpoint.
  - `profile_gd_residual_v1.py` 将误导性的 `logged_step_sec` 改为 `metrics_collect_sec`, 并支持 `PROFILE_ENABLE_GD_DIAGNOSTICS`.
  - profiler 表现在输出 `self_cuda_time_total`, `cpu_time_total`, `cuda_memory_usage` 等排序.

关键代码位置:

- Metrics gating 和 profiler marker: `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn_fox.py:1026`, `:1090`, `:1114`, `:1212`, `:1300`, `:1344`.
- Phase2 materialization: `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn_fox.py:1317`, `:1322`, `:1331`, `:1333`.
- Attention 侧 `gd_collect_metrics`: `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py:1252`.
- zoology init skip: `zoology/model.py:139`, `:149`.
- profile runner timing/memory/profiler sort: `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/profile_gd_residual_v1.py:133`, `:180`, `:192`, `:230`.

追加诊断 diff 摘要:

- Flash-VQG: 2 files, `220 insertions`, `58 deletions`.
- zoology: 7 files, 主要是 profile runner, metrics whitelist, README, tests 和本报告.

## 4. 运行命令和测试结果

保护性测试:

| 命令 | 结果 |
| --- | --- |
| `pytest tests/test_fox_gd_residual_v1.py -q` | `11 passed in 0.34s` |
| `pytest tests/test_fox_guards.py tests/test_fox_dense_write.py tests/test_fox_clr_delta_v1.py tests/test_fox_phase2_metrics.py -q` | `66 passed in 19.11s` |
| `pytest tests/test_flash_vqg_wrapper.py tests/test_flash_vqg_scripts.py tests/test_flash_vqg_metrics_white_list.py tests/test_train_logging_steps.py tests/test_train_batch_order.py -q` | `77 passed, 1 warning in 11.59s` |

主要 profile 命令模板:

```bash
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
TRAIN_BATCH_SIZE=16 \
PROFILE_SEQ_LEN=128 \
PROFILE_MICROBATCHES=8 \
PROFILE_ENABLE_TORCH_PROFILER=0 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/20260425-gd-residual-v1-profile-final/A-baseline-r16-wk4-read2-b16 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

细粒度 profiler 使用 B8 可完成点:

```bash
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
TRAIN_BATCH_SIZE=8 \
PROFILE_SEQ_LEN=128 \
PROFILE_MICROBATCHES=1 \
PROFILE_ENABLE_TORCH_PROFILER=1 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/20260425-gd-residual-v1-profile-final/B8-profiler-rread-2-r16-wk4-b8 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

轻量 phase timer / event 诊断:

```bash
PROFILE_ENABLE_GD_DIAGNOSTICS=1 \
PROFILE_ENABLE_TORCH_PROFILER=0 \
PROFILE_SEQ_LEN=64 \
PROFILE_MICROBATCHES=1 \
TRAIN_BATCH_SIZE=1 \
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=4 \
FOX_GD_RESIDUAL_WRITE_TOPK=1 \
PROFILE_OUTPUT_DIR=tmp/20260426-gd-residual-diagnostics-smoke \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

B16/T128 torch profiler 单点超过 6 分钟仍未完成, 已中止并记录为 profiler overhead 不可接受. 因此细粒度表采用 B8/T128.

实验产物:

- `tmp/20260425-gd-residual-v1-profile-final/profile_runs.tsv`
- `tmp/20260425-gd-residual-v1-profile-final/*/summary.json`
- `tmp/20260425-gd-residual-v1-profile-final/B8-profiler-rread-2-r16-wk4-b8/profiler_*.txt`
- `tmp/20260425-gd-residual-v1-profile-final/local-mqar-signal*/summary.json`

## 5. Profiling 结果

### A baseline 和 T=256 read_topk

| run | read_topk | micro_s | fwd_s | bwd_s | peak_alloc_GB | peak_reserved_GB | inject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline T128 | 2 | 15.17 | 5.03 | 10.12 | 2.52 | 2.84 | 0.0176 |
| T256 read2 | 2 | 37.18 | 12.46 | 24.65 | 5.09 | 5.58 | 0.0348 |
| T256 read4 | 4 | 38.04 | 12.93 | 25.03 | 5.16 | 5.67 | 0.0476 |
| T256 read8 | 8 | 36.96 | 12.67 | 24.21 | 5.29 | 5.82 | 0.0616 |
| T256 read16 | 16 | 37.01 | 12.40 | 24.54 | 5.56 | 6.65 | 0.0786 |
| T256 dense | dense | 37.42 | 12.46 | 24.89 | 9.73 | 10.46 | 0.1388 |

T=256 dense read 已接近 11GB GPU 上限. read_topk=2 将 reserved memory 从 10.46GB 降到 5.58GB, 但单 microbatch 时间基本不变, 说明 read_topk 主要解决显存而非当前主耗时.

### read_topk sweep, T=128

| read_topk | batch | avg_micro_s | last_micro_s | peak_reserved_GB |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8 | 9.70 | 7.50 | 1.65 |
| 4 | 8 | 9.56 | 7.43 | 1.65 |
| 8 | 8 | 9.37 | 7.33 | 1.90 |
| 16 | 8 | 9.59 | 7.44 | 1.77 |
| dense | 8 | 9.65 | 7.56 | 2.83 |
| 2 | 16 | 15.75 | 15.17 | 2.84 |
| 4 | 16 | 17.17 | 14.96 | 2.91 |
| 8 | 16 | 16.62 | 14.53 | 2.97 |
| 16 | 16 | 17.04 | 14.95 | 5.12 |
| dense | 16 | 17.01 | 14.99 | 5.47 |

### rank / batch / write_topk sweep, T=128

| rank | read_topk | write_topk | batch | avg_micro_s | peak_reserved_GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 4 | 8 | 9.64 | 0.58 |
| 8 | 2 | 4 | 8 | 9.68 | 1.10 |
| 16 | 2 | 4 | 8 | 9.70 | 1.65 |

| batch | read_topk | rank | write_topk | avg_micro_s | peak_reserved_GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 16 | 4 | 5.92 | 1.08 |
| 8 | 2 | 16 | 4 | 9.70 | 1.65 |
| 16 | 2 | 16 | 4 | 15.75 | 2.84 |

| write_topk | read_topk | rank | batch | avg_micro_s | peak_reserved_GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 16 | 8 | 6.10 | 1.59 |
| 4 | 2 | 16 | 8 | 9.70 | 1.65 |

write_topk 从 4 降到 2 明显缩短时间, 但这是 efficiency 对照, 不替代主配置结论.

### Profiler 细表, B8/T128/read2/r16/wk4

Profiler 文件:

- `profiler_self_cuda_time_total.txt`
- `profiler_cpu_time_total.txt`
- `profiler_cuda_memory_usage.txt`

关键行:

| 项 | 证据 |
| --- | --- |
| `gd_residual/grouped_chunk` | self CUDA `2.571s`, `51.84%`, 4 calls |
| `gd_residual/event_pack` | self CUDA `2.240s`, `45.16%`, 4 calls |
| `gd_residual/phase2_residual_read` | self CUDA `0.941ms`, memory row约 `22.33MB` |
| `gd_residual/build_metrics` | self CUDA `0.349ms` |
| `gd_residual/phase2_metrics` | self CUDA `0.281ms` |
| `aten::copy_` | `163537` calls, self CUDA `483.7ms` |
| `aten::item` / `_local_scalar_dense` | `31368` calls, CUDA sync time约 `28.2ms` |
| `aten::outer` | `14336` calls |
| `aten::matmul` | `8197` calls |
| `aten::gather_backward` | CUDA memory usage约 `1.06GB` |
| `aten::gather` | CUDA memory usage约 `17.02MB` |

GPU 粗采样:

- `nvidia-smi dmon` baseline B16/T128/read2: SM utilization min/max/avg `0% / 84% / 22.7%`, memory utilization avg `15.4%`.
- 这与大量小 kernel 和 CPU 调度开销相符.

### Opt-in diagnostics smoke, 2026-04-26

`PROFILE_ENABLE_GD_DIAGNOSTICS=1` 的最小运行确认 summary 字段和模型 debug metrics 可以正常落盘:

| 项 | 数值 |
| --- | ---: |
| config | `T=64, B=1, rank=4, write_topk=1, read_topk=2` |
| microbatch_s | `4.52` |
| forward_s / backward_s / optimizer_s | `2.74 / 1.75 / 0.03` |
| metrics_collect_s | `0.0012` |
| peak_reserved_GB | `0.53` |
| event_pack_wall_s | `0.0461` |
| grouped_chunk_wall_s | `0.0220` |
| phase2_residual_read_wall_s | `0.00058` |
| event_count / group_count | `128 / 100` |
| max / mean events per group | `4.0 / 1.28` |
| L_state mean / max | `0.0677 / 0.1509` |
| L_state frac >= 0.1 / 0.5 / 1.0 | `0.50 / 0.0 / 0.0` |

说明: 这里使用的是同步 wall-time 计时, 不是 CUDA Event. 该诊断默认关闭, 只用于低开销复核 profiler 归因和 `mu_valid_ratio=0` 的原因.

## 6. MQAR 短周期训练信号

本地 MQAR short-signal 使用相同 config builder, 但限制样本数和 batch 数, 不连接 SwanLab, 不保存 checkpoint. first-segment sweep 只覆盖 64-token segment, all-segment baseline 覆盖原 smoke 的多个 segment 但每段样本极少. 这些数据只用于信号判断, 不是正式质量结论.

代表性 first-segment 结果:

| run | loss first -> last | delta | valid_loss | peak_reserved_GB | lambda | inject | m_norm_max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline r16/wk4/b16 | 10.993 -> 10.958 | -0.035 | 9.026 | 1.65 | 0.0500 | 0.0000 | 0.2064 |
| B read2/r16/wk4/b8 | 11.033 -> 11.025 | -0.008 | 9.028 | 1.09 | 0.0500 | 0.0000 | 0.1785 |
| B dense/r16/wk4/b8 | 11.036 -> 11.041 | 0.005 | 9.025 | 1.59 | 0.0500 | 0.0000 | 0.1760 |
| C rank4/read2/wk4/b8 | 11.028 -> 11.042 | 0.014 | 9.033 | 0.55 | 0.0500 | 0.0000 | 0.1869 |
| C rank8/read2/wk4/b8 | 11.034 -> 11.033 | -0.001 | 9.013 | 0.56 | 0.0500 | 0.0000 | 0.1792 |
| D batch4/read2/r16/wk4 | 11.034 -> 10.991 | -0.042 | 9.017 | 0.55 | 0.0500 | 0.0000 | 0.1695 |
| E wk2/read2/r16/b8 | 11.036 -> 11.041 | 0.005 | 9.025 | 1.06 | 0.0500 | 0.0000 | 0.3087 |

All-segment baseline short run:

| 参数 | 数值 |
| --- | ---: |
| read_topk / rank / write_topk / batch | `2 / 16 / 4 / 8` |
| train records | 5 |
| loss | `11.006 -> 10.975` |
| valid_loss | `9.043` |
| valid_accuracy | `0.0` |
| peak_reserved_GB | `2.88` |
| final lambda_mean | `0.04994` |
| final inject_ratio | `0.03399` |
| final write_strength_mean | `0.12484` |
| final m_norm_mean / max | `0.03376 / 0.20650` |
| final mu_valid_ratio | `0.0` |

解释:

- loss 有轻微下降, lambda 没有塌缩, write_strength 有效, m_norm 未爆炸.
- all-segment run 中 inject_ratio 非零, 说明残差注入路径在 MQAR 上可触发.
- 但 valid accuracy 仍为 0, mu_valid_ratio 未增长. 训练步数太少, 只能证明没有立刻崩坏, 不能证明质量有效.

## 7. 三个瓶颈假设回答

1. `grouped_chunk_torch_ref` 是否主要由 Python/event 循环和大量小 kernel 组成？

是. 证据包括:

- `gd_residual/grouped_chunk` self CUDA `2.571s`, `51.84%`.
- `gd_residual/event_pack` self CUDA `2.240s`, `45.16%`.
- `aten::copy_` 约 `163537` calls, `aten::item/_local_scalar_dense` 约 `31368` calls, `aten::outer` 约 `14336` calls.
- GPU SM utilization avg 约 `22.7%`, 符合 CPU 调度大量小 kernel 的模式.

2. `phase2 residual read` 是否 materialize 大中间张量并导致显存接近打满？

是, 但主要体现在 dense 或较高 read_topk. 证据包括:

- phase2 代码显式 materialize `C_sel`, `M_sel`, `proposal`.
- T=256,B=16 dense read: peak reserved `10.46GB`, 接近 11GB 卡上限.
- T=256,B=16 read_topk=2: peak reserved `5.58GB`, 说明 read_topk 对显存有效.
- B8 profiler 中 `aten::gather_backward` CUDA memory usage约 `1.06GB`.

3. `enable_layer_metrics=False` 时 gd residual metrics 是否仍会被计算？

旧实现会, 本轮已修复. 现在 `collect_metrics=False` 会跳过 gd residual build metrics 和 phase2 metrics. 新增测试覆盖:

- `enable_layer_metrics=False` 时 gd residual metrics 为空.
- `fox_phase2_metrics_mode="off"` 时 gd residual metrics 为空.
- `enable_layer_metrics=True` 时 6 个 gd residual 指标存在.

## 8. 瓶颈排序和决策

当前瓶颈排序:

1. `grouped_chunk_torch_ref` / event pack. 这是当前主耗时, 也是 GPU 利用率低的最直接原因.
2. phase2 residual read memory materialization. 对 dense/high read_topk/T=256 显存影响很大, 但在 read_topk=2 主配置下不是主耗时.
3. backward 总成本. B16/T256 backward 约 `24.65s`, 但 profiler 指向上游大量 reference 小算子.
4. metrics. 本轮 gating 后成本已降为可忽略, profiler 中 build/phase2 metrics 均小于 `1ms`.

决策:

- 继续小规模验证: 建议继续. 条件是下一个短跑要观察 `inject_ratio` 持续非零, loss 继续下降, `m_norm` 不爆炸.
- Triton fused residual read: 暂缓. 它对 dense/high read_topk 显存有价值, 但当前 read_topk=2 的主耗时不是 phase2 read.
- grouped_chunk/event pack 优化: 建议作为下一阶段第一优先级. 若要投入工程优化, 先减少 event pack loop, item sync, copy_/outer/matmul 小 kernel 数.

## 9. 回退方式

- Flash-VQG 回退: 删除 `attn.py` 中 gate direct init 和 `attn_fox.py` 中 `collect_metrics` / profiler marker 改动.
- zoology 回退: 删除 `_flashvqg_custom_init` 跳过逻辑, 去掉 `FOX_REMOTE_READ_TOPK` 透传和 profile 脚本.
- metrics 回退: 从 `metrics_white_list.py` 删除 `GD_RESIDUAL_BASE_KEYS` 和相关控制逻辑.
- profile/report 产物回退: 删除 `tmp/20260425-gd-residual-v1-profile-final/` 和本报告文件.

## 10. 下一步最小实验计划

1. 先用 `PROFILE_ENABLE_GD_DIAGNOSTICS=1` 复核 phase wall-time:
   - `read_topk=2, rank=16, write_topk=4, batch=8, T=128`.
   - 记录 `event_pack`, `grouped_chunk`, `phase2_residual_read`, `forward`, `backward`, `optimizer`, `metrics_collect`.
2. 做 event_count scaling:
   - 固定 `read_topk=2, rank=16, batch=8, T=128`.
   - 扫 `write_topk=1,2,4`, 记录 event/group 计数和 microbatch time 是否近似线性变化.
3. 不做 Triton/CUDA/custom backward, 跑 2-3 个更长的 all-segment MQAR short run:
   - `read_topk=2, rank=16, write_topk=4, batch=8`.
   - `read_topk=2, rank=16, write_topk=2, batch=8`.
   - `read_topk=2, rank=8, write_topk=4, batch=8`.
4. 每个 run 至少记录 20-50 个 train batch, 观察 loss 是否连续下降, `inject_ratio` 是否非零稳定, `mu_valid_ratio` 是否开始增长.
5. 补同 token budget baseline: local-only/no remote, legacy 或 `clr_delta_v1`, gd_residual_v1 主配置.
6. 若训练信号成立, 优先做 grouped_chunk/event pack 的 PyTorch 向量化或 padded grouped chunk 原型.
7. 只有当质量信号成立且 read_topk/dense 显存继续限制实验规模时, 再投入 fused residual read.
