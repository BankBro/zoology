# MQAR GD Remat 回归实验计划

## 1. 实验登记

- Experiment ID: `20260729-01-mqar-gd-remat-regression`.
- 状态: `completed, quality_failed`.
- 执行机器: `mclab-3090`的`Flash-VQG-tun`容器.
- Flash-VQG固定commit: `79fef6a8e9d3f41dfcbf40bf668ec83286dd5d62`.
- zoology base: `flash-vqg@581329853fdbd94f9510c0e80481fccf63fb0add`.

## 2. 目标与假设

目标是验证A1 `post_phase1 remat`是否降低canonical `baseline-r16-joint`的MQAR训练质量和长度外推能力. A1只改变训练图生命周期, 预期不改变模型数学语义. 本实验是300M自然语言质量pilot的硬前置门禁, 不是新的GDN对照、dtype sweep或block length消融.

## 3. 固定口径

| 项目 | 固定值 |
|---|---|
| GPU与精度 | RTX 3090, AMP BF16, FP32 master weights和optimizer state |
| 模型 | `baseline-r16-joint`, 1,160,390参数 |
| GD配置 | codebook 64, rank 16, read top-k 16, write top-k 4 |
| Block | `block_len=32`, `local_num_blocks=2` |
| Backend | grouped Triton, selected-read Triton remat, `fp32_boundary` |
| 数据 | canonical MQAR cache, data seed 123 |
| 训练 | B64, validation B16, GA4, 4 epochs, early stopping关闭 |
| Seeds | 123, 124, 125 |
| Checkpoint | `last.pt`主结果, `best.pt`敏感性 |

唯一变量为`fox_gd_residual_remat_mode`:

| Variant | 值 | Formal runs |
|---|---|---:|
| `a0-off` | `off` | 3 |
| `a1-post-phase1` | `post_phase1` | 3 |

## 4. 执行流程

**(1)** Preflight锁定环境、GPU空闲、两个仓库clean commit、cache/init hash、六份配置和A0/A1单变量差异.

**(2)** Seed 124执行32 optimizer-step配对轨迹. Step 1严格比较BF16 forward、loss、梯度、参数和Adam状态, 容差为`atol=1e-5,rtol=1e-4`. Step 16/32只要求finite、状态键和逻辑计数一致, 误差作为诊断记录.

**(3)** A0/A1分别执行3-update integration smoke、受控optimizer-boundary恢复、validation、checkpoint save/load和两个Longer-MQAR slice smoke.

**(4)** 六条formal run按`A0-s123, A1-s123, A1-s124, A0-s124, A0-s125, A1-s125`串行执行. 任一失败立即停止, 不自动补seed或放宽门槛.

**(5)** Last与best分别在`1024x256`, `2048x512`, `4096x1024`, `8190x512`, `8190x2047`上执行500-example matching-BF16评估. Batch size固定为`128/64/32/16/16`, 相同model-state hash物理去重.

## 5. 裁决与产物

主门槛同时满足才通过:

```text
mean(A1 - A0) standard MQAR >= -0.01
mean(A1 - A0) four extrapolation slices >= -0.02
```

分析使用三seed配对delta、均值和population SD. Last失败时best不能挽救主结论. 同时记录wall time、step p50和peak allocated/reserved, 但不设置性能硬门槛.

Raw输出保留在实验目录`outputs/3090/<run-tag>/`. 终态artifact至少包含training、Longer-MQAR、paired quality、trajectory、system summary、source manifest和metadata. 正式完成、失败或中止后均生成report并追加`docs/EXPERIMENT_LOG.md`; 只有结论改变下一步时更新`docs/STATUS.md`.

预计总预算为3至4 GPU小时, checkpoint与raw输出低于2 GiB. 本实验不启动自然语言pilot或1B-token训练.

## 6. 执行结果

6/6 个正式训练 run 和 60/60 个逻辑评估事件已完成. A1 标准 MQAR delta 均值为 `-0.04020`, 四外推 slice 宏平均为 `-0.10562`, 均未通过预注册门槛. A1 不替代 A0, 详细结果见 [正式报告](../20260729-01-mqar-gd-remat-regression-report.md).
