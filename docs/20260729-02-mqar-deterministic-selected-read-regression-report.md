# MQAR 确定性 Selected-Read 回归实验报告

## 1. 结果概览

`20260729-02-mqar-deterministic-selected-read-regression` 已完成全部实现、门禁、6/6 个正式训练 run 和 60 个逻辑 Longer-MQAR 评估事件. 固定顺序修改显著缓解了历史 A1 质量退化, 但没有通过逐 seed 最终状态完全一致的严格门槛.

| 门禁 | 结果 | 门槛 | 裁决 |
|---|---:|---:|---|
| 正式训练 | 6/6 | 6/6 | 通过 |
| 最终 A0/A1 hash 相同 | 2/3 seeds | 3/3 | 失败 |
| 标准 MQAR delta 均值 | +0.00650 | >= -0.01 | 通过 |
| 四外推任务宏平均 delta | +0.01743 | >= -0.02 | 通过 |

正式终态为 `quality_recovered_but_not_deterministic`. A1 的质量非劣性已经恢复, 但仍不是可证明的长训练数值无操作, 因此不替代 A0, 不作为 300M 自然语言训练默认方案.

## 2. 实现与实验口径

修改仅位于 selected-read custom backward. 原实现通过 CUDA `index_select` backward 将重复 head 的 `addr_proj` 梯度归约到共享行, 累加顺序不固定. 新实现先计算每个 query 的 `grad_addr_selected`, 再使用已有 deterministic segment accumulation 按固定顺序回填 `grad_addr_proj`.

两组共同使用该修复, 唯一实验变量仍是 `fox_gd_residual_remat_mode`:

| Variant | Remat mode | Seeds |
|---|---|---|
| `a0-fixed-off` | `off` | 123, 124, 125 |
| `a1-fixed-post-phase1` | `post_phase1` | 123, 124, 125 |

正式配置固定为 RTX 3090、AMP BF16、FP32 master weights 和 optimizer state、`baseline-r16-joint`、B64、validation B16、GA4、4 epochs、data seed 123、canonical cache/init. 运行绑定 Zoology `22ed61c8b963` 和 Flash-VQG `d7dbb1282d20`.

## 3. 确定性结果

低层 selected-read backward 在默认全局 deterministic mode 关闭时重复 8 次, 输出和全部梯度逐位一致. seed124 的 128 optimizer-step A0/A1 lockstep 也逐位一致, 两个 variant 各自两次 fresh-process 32-step 结果同样一致.

完整四 epoch 训练揭示了短门禁未覆盖的问题:

| Seed | A0/A1 final hash | 标准 MQAR delta | 四外推宏平均 delta |
|---:|---|---:|---:|
| 123 | 相同 | 0 | 0 |
| 124 | 不同 | +0.01951 | +0.05228 |
| 125 | 相同 | 0 | 0 |

seed123 和 seed125 的 A0/A1 model-state hash 完全相同. seed124 的 A0 hash 为 `61717b06...424d`, A1 为 `681fb658...9ffd`. 当前证据说明 fixed `addr_proj` 归约消除了主要不稳定来源, 但不是所有长训练分叉的充分修复.

不能根据 128-step lockstep 推断 formal run 一定在 step128 之后才分叉, 因为 lockstep 是同进程交替执行, 正式 A0/A1 是独立进程. 本轮未保存逐 step formal state hash, 因而剩余分叉的首次位置和具体 tensor 仍未直接观测.

## 4. MQAR 与长度外推

### 4.1. 历史退化是否缓解

| 指标 | 历史 A1 - A0 | Fixed A1 - A0 | 改善 |
|---|---:|---:|---:|
| 标准 `1024x256` | -0.04020 | +0.00650 | +0.04670 |
| 四外推任务宏平均 | -0.10562 | +0.01743 | +0.12305 |

两个历史负向结果均已扭转. 尤其是历史退化最大的 seed125, 本轮 A0/A1 的完整状态和全部评估结果严格相同.

### 4.2. 正式 n=500 评估

| 任务 | A0 mean | A1 mean | A1 - A0 mean |
|---|---:|---:|---:|
| 1024x256 | 0.96721 | 0.97349 | +0.00627 |
| 2048x512 | 0.80671 | 0.82665 | +0.01995 |
| 4096x1024 | 0.44372 | 0.46412 | +0.02040 |
| 8190x512 | 0.67601 | 0.69730 | +0.02129 |
| 8190x2047 | 0.15061 | 0.15868 | +0.00807 |

预注册标准门槛使用训练末端 validation 的 `1024x256`, 因而其 `+0.00650` 与表中独立 n=500 评估的 `+0.00627` 口径不同. 两者方向一致.

当前正向 delta 全部来自 seed124 的分叉轨迹. 这能证明历史质量退化已经缓解, 不能证明 A1 稳定优于 A0. 在剩余非确定性消除前, 正向结果仍可能随运行方向改变.

## 5. 显存与速度

| Variant | Wall mean, min | Step p50 mean, s | Peak allocated max, MiB | Peak reserved max, MiB |
|---|---:|---:|---:|---:|
| A0 fixed off | 19.31 | 0.3180 | 2566.2 | 3158 |
| A1 fixed post-phase1 | 22.09 | 0.3747 | 2003.0 | 2426 |

A1 peak allocated 降低约 `22.0%`, peak reserved 降低约 `23.2%`. 代价是平均 wall time 增加约 `14.4%`, optimizer-step p50 增加约 `17.8%`. 三条 A1 均完成 11260 次 selected-read recompute, fallback 为 0.

固定顺序修改没有破坏 A1 原有显存收益, 也没有引入新的 OOM 或 checkpoint/resume 问题.

## 6. 解释与边界

本轮支持以下结论:

- `addr_proj` 的重复 head CUDA 归约是历史分叉的重要来源, 定向修复后 2/3 seeds 获得四 epoch 逐位一致, 两项质量 delta 大幅恢复.
- 它不是唯一来源. seed124 仍表明 remat 与 off 在长训练中可能进入不同轨迹.
- seed124 的 A1 恰好更好不改变确定性裁决. 未消除的分叉既可能放大为退化, 也可能放大为提升.
- 现有证据尚不能区分剩余来源是其他共享梯度归约、独立进程 CUDA 执行顺序, 还是更长时间尺度上的 remat 重算交互.

128-step lockstep 对快速阻止明显错误仍有价值, 但今后不能单独作为长训练逐位一致的充分门禁.

## 7. 决策与下一步

- A0 `off` 继续作为 canonical MQAR 和自然语言训练实现.
- A1 不晋升. Flash 修复分支保留为实验分支, 不合入当前 300M 默认显存优化分支.
- 下一轮先对 seed124 做独立进程长轨迹 probe, 在 128、256、512 及后续 optimizer steps 保存 model、gradient 和 optimizer hash, 定位首次分叉及首个 tensor.
- 定位后只修复剩余非确定性来源, 再从头执行同一三 seed 门禁. 不通过增加 seed 或放宽 hash 门槛掩盖问题.
- native-BF16 selected-read 仍可作为独立显存路线, 但也必须通过相同 MQAR 与 Longer-MQAR 回归.

3090 raw 共约 257 MiB. 除 checkpoint 外的 134 个文件和 12 个 resolved config 已镜像回 2080 Ti 并逐文件验证 SHA256 一致. 18 个 checkpoint tensor 保留在 3090. 完整数值见 [artifact README](artifacts/20260729-02-mqar-deterministic-selected-read-regression/README.md), [paired quality](artifacts/20260729-02-mqar-deterministic-selected-read-regression/paired-quality.csv), [final hash pairs](artifacts/20260729-02-mqar-deterministic-selected-read-regression/final-hash-pairs.csv) 和 [source manifest](artifacts/20260729-02-mqar-deterministic-selected-read-regression/source-manifest.csv).
