# Residual Value Bottleneck MQAR 筛选报告

## 1. 结果概览

- Experiment ID: `20260731-02-residual-value-bottleneck-mqar`.
- 状态: `completed`, 终态为`quality_rejected_at_q0`.
- 执行机器: RTX 3090 GPU0.
- 训练source: Zoology `a18cd544462e2030900337bb953527c467272b04`, Flash-VQG `cc3f92b8a972f1c51c3deabeafd0d9f180bc2b16`.
- 评估兼容修复source: Zoology `30148ab`.
- Plan: [实验计划](plans/20260731-02-residual-value-bottleneck-mqar-plan.md).
- Artifact: [精简证据](artifacts/20260731-02-residual-value-bottleneck-mqar/README.md).

本实验验证仅压缩residual value通道是否能在保留粗记忆64维的情况下维持MQAR能力. U32在标准`1024x256`上相对U64下降`8.34%`, 尚处于Q0的`10%`容差内, 但四个真正外推slice宏平均下降`37.87%`. U16在标准任务下降`37.92%`, 外推宏平均下降`74.00%`. 两个候选均未通过预注册Q0, 因此三seed四epoch正式矩阵未启动.

## 2. 实验合同

三组共同使用canonical init、seed123、AMP BF16、block64、local2、rank16、write4/read16、`post_phase1` remat、grouped Triton builder、S1 exact selected backward和`fp32_boundary`. 仅residual value维度及候选新增的绑定投影不同.

| Variant | Residual value | Trainable参数 | 投影 |
|---|---:|---:|---|
| U64 | 64 | 1,160,390 | 无 |
| U32 | 32 | 1,164,486 | 每层每头`[64,32]` |
| U16 | 16 | 1,162,438 | 每层每头`[64,16]` |

U32和U16使用同一投影完成写入降维和读取升维. 所有共有初始tensor与U64逐位一致. 投影采用local RNG和dense signed Hadamard基, 两台机器的derived state hash逐位一致. Preflight确认候选相对U64的训练配置只改变`fox_gd_residual_value_dim`.

三组均完成3-update smoke和一轮704-update screen. Loss、checkpoint、FP32 master weight和optimizer state均finite, GradScaler skip为0, Triton fallback为0. 三组FLA fused-gate backward config一致.

## 3. 质量结果

### 3.1. 标准任务

标准指标取训练last checkpoint的locked validation `1024x256`.

| Variant | Accuracy | 相对U64变化 | Q0门禁 |
|---|---:|---:|---|
| U64 | 0.962172 | - | Reference |
| U32 | 0.881938 | -8.34% | 通过 |
| U16 | 0.597313 | -37.92% | 失败 |

### 3.2. Longer MQAR

| Shape | U64 | U32 | U32相对变化 | U16 | U16相对变化 |
|---|---:|---:|---:|---:|---:|
| `2048x512` | 0.831316 | 0.598859 | -27.96% | 0.273027 | -67.16% |
| `4096x1024` | 0.529271 | 0.251934 | -52.40% | 0.081477 | -84.61% |
| `8190x512` | 0.691324 | 0.488465 | -29.34% | 0.213922 | -69.06% |
| `8190x2047` | 0.217395 | 0.070709 | -67.47% | 0.021631 | -90.05% |
| 四slice宏平均 | 0.567327 | 0.352492 | **-37.87%** | 0.147514 | **-74.00%** |

U32并非只在最远的单个任务上失败. 四个外推slice全部负向, 且负载更高的`4096x1024`和`8190x2047`下降更大. 这与residual state容量被压缩的机制预期一致, 不能解释为一个边界batch或评估噪声.

## 4. 小模型资源信号

| Variant | Wall | Step p50 | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| U64 | 210.50 s | 0.22100 s | 1456.64 MiB | 2192 MiB |
| U32 | 217.87 s | 0.23408 s | 1456.71 MiB | 2194 MiB |
| U16 | 216.04 s | 0.23099 s | 1456.71 MiB | 2070 MiB |

小模型没有显示吞吐收益. 这不否定300M residual state缩小后的资源收益, 因为小模型中固定框架和其他算子占比更高. 300M资源结论应使用Flash-VQG仓的正式3×5测量, 不使用本表外推.

## 5. 失败闭环

**(1) 跨机初始化hash不一致.** 初版投影使用CPU QR, 两台机器虽都满足正交性, 逐位结果不同. Flash-VQG改为不依赖BLAS QR的dense signed Hadamard构造后, U32和U16 derived state hash在两台机器逐位一致.

**(2) 首次queue驱动失败.** Run tag `20260731-residual-value-mqar-01`在首个smoke前因上游callback被自身monkeypatch回调而停止. 修复保存原始上游函数引用并增加回归测试后, 使用新tag从头执行.

**(3) 首次longer评估加载失败.** `run02`的三组训练均已完成, 评估因旧`train_config.json`内三个`resume_identity`值为整数而无法重新加载. 修复从源头把identity序列化为字符串, 并按仓库既有机制建立checkpoint硬链接shadow, 只修正旁路config. Shadow metadata记录源checkpoint、源config和修复后config hash, checkpoint inode与SHA256不变. 15个正式评估事件随后全部完成.

这些基础设施问题均发生在科学结果之外, 没有放宽质量门槛或修改已训练权重.

## 6. 决策

1. U32和U16均拒绝进入三seed四epoch正式矩阵.
2. 不把residual value 32或16维加入当前质量canonical, 也不启动对应300M自然语言pilot或1B-token训练.
3. 现有U64模型和`A1 + S1 exact`质量路径保持不变.
4. 后续效率优化回到不缩减模型容量的fixed-slot custom VJP、backward scan-read fusion和remaining backward拆分.

本轮只有一个training seed, 但失败幅度远大于预注册容差, 且四个外推slice方向一致. 因此没有必要为U32/U16追加三seed确认. 若未来提出U48等新容量点, 必须登记为新的架构消融, 不能把本轮U32结果外推为已通过.

## 7. 原始证据

3090原始目录:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260731-02-residual-value-bottleneck-mqar/outputs/3090/
```

正式训练与评估使用run tag `20260731-residual-value-mqar-02`. `run01`和`run02`首次评估失败现场继续保留. Checkpoint不提交Git, 精简指标和source manifest保存在artifact目录.
