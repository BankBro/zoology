# Selected-read Warp MQAR 筛选计划

## 1. 实验登记

- Experiment ID: `20260731-01-selected-read-warp-mqar-screen`.
- 状态: `completed`, 终态为`quality_mixed`.
- Zoology分支: `20260731-082645-selected-read-warp-mqar-screen`.
- Zoology base: `flash-vqg@86ea8aa`.
- Flash-VQG source: 在正式运行前绑定selected-read性能实验的干净finalist commit.
- 执行机器: RTX 3090.
- 训练与评估精度: AMP BF16, FP32 master weights与optimizer state.

本实验只做seed123一轮MQAR质量筛选, 比较S1 exact、W2 direct和W2加preproject. 目标是判断减少state-owner kernel的warp数以及额外预投影是否会在训练中放大低层浮点差异. 本实验不启动三seed正式矩阵, 不替代300M自然语言paired pilot.

## 2. 固定合同与矩阵

三组共同使用`baseline-r16-joint`, canonical cache/init, data seed123, `block_len=64`, `local_num_blocks=2`, rank16, write top-k4, read top-k16, `post_phase1` remat, grouped builder, `fp32_boundary`, B64, validation B16和GA4.

| Variant | Selected backward | Chunk | 作用 |
|---|---|---:|---|
| `s1-head8192` | `triton_deterministic_s1_head` | 8192 | Exact质量对照 |
| `r1a-owner-w2` | `triton_state_owner_r1a_s1_w2` | 8192 | 隔离state-owner与warps2 |
| `r1b-preproject-w2` | `triton_state_owner_r1b_preproject_w2_fast` | 8192 | 追加query/code预投影 |

执行矩阵:

| 阶段 | Seed | 训练长度 | Runs |
|---|---:|---:|---:|
| Smoke | 123 | 3 optimizer updates | 3 |
| Q0筛选 | 123 | 1 epoch | 3 |
| Locked eval | 123 | 5个MQAR形状 | 15个checkpoint-case事件 |

三组从同一canonical init独立训练, 固定data order、FLA环境和数值策略. Q0只使用`last.pt`作主裁决.

## 3. 裁决与证据边界

两个candidate分别对S1计算:

```text
standard 1024x256 delta >= -0.01
four-slice extrapolation macro delta >= -0.02
```

同时要求loss、gradient和checkpoint finite, 三组均命中目标selected backward, 全部Triton fallback为0, FLA fused-gate backward config一致. 若FLA config不一致, 先做matched-config复验再解释质量.

W2候选已知在production-shape低层对照中确定性重复, 但`grad_addr_proj`最大绝对差约为`2e-4`, 超过原selected-read实验的`2e-5`门槛. 因此:

- MQAR失败时直接拒绝该candidate的质量路径.
- MQAR通过时只说明seed123一轮训练未观察到明显退化, 不会自动把candidate提升为质量canonical.
- 是否进入300M BF16自然语言paired pilot, 仍需结合端到端性能、低层误差和本实验结果另行裁决.

## 4. 执行与失败策略

**(1)** Preflight锁定两个仓库的干净commit、canonical Python、CUDA/NVML、3090、依赖版本、cache/init hash、参数量和配置差异.

**(2)** 任一smoke失败时停止其依赖训练, 保留现场并定位. 可修复的runner或audit bug采用新run tag重跑; 数值或质量门槛不事后放宽.

**(3)** 三组Q0均完成后统一评估. 单个candidate失败不阻止另一个candidate完成, 以便区分warp归约和preproject的影响.

**(4)** 3090保留checkpoint和大型raw. 轻量结果镜像回2080 Ti, 最终生成artifact、report并更新项目日志.

## 5. 预算

- 核心训练与评估预计低于0.5个3090 GPU-hour.
- API费用为0.
- 本实验不启动1B-token训练.

## 6. 执行终态

三组smoke和seed123一轮训练均完成. W2 direct的标准delta为`-0.005613`, 四外推宏平均delta为`-0.019200`, 两项通过. Preproject标准delta为`-0.006410`, 但外推宏平均delta为`-0.036765`, 已拒绝. 完整结论见[实验报告](../20260731-01-selected-read-warp-mqar-screen-report.md).
