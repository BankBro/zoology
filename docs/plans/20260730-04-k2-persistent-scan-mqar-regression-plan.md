# K2 Persistent Scan MQAR 质量回归计划

## 1. 实验登记

- Experiment ID: `20260730-04-k2-persistent-scan-mqar-regression`.
- 状态: `completed`, 终态为`quality_rejected_at_bf16_screen`.
- 登记日期: 2026-07-30.
- Zoology分支: `20260730-233021-k2-persistent-mqar-regression`.
- Zoology base: `8af50e0c9b02d4e76eabcdc82d6ebe307d34eefe`.
- Flash-VQG source: `20260730-171547-a1-persistent-scan@a6b1af3d8845caa7f317e7104a1af500a03b1c24`.
- 执行机器: RTX 3090.
- 训练与评估精度: AMP BF16, FP32 master weights与optimizer state.

本实验比较P0 A1与K2 P8在MQAR及长度外推任务上的质量. 两组只允许GD builder不同, 用于判断K2能否从资源可行候选提升为通过MQAR质量门禁的工程候选. 本实验不替代300M自然语言质量pilot, 不启动1B-token训练.

## 2. 固定合同与矩阵

两组共同使用`baseline-r16-joint`, canonical cache/init, data seed123, `block_len=64`, `local_num_blocks=2`, rank16, write top-k4, read top-k16, `post_phase1` remat, deterministic selected backward, `fp32_boundary`, B64, validation B16和GA4.

| Variant | Builder | Tile blocks | 其他差异 |
|---|---|---:|---|
| `p0-a1-block64` | `grouped_chunk_torch_ref` | 8 | 无 |
| `k2-persistent-p8` | `persistent_scan_triton` | 8 | 无 |

执行矩阵:

| 阶段 | Seeds | Epochs或steps | Runs |
|---|---|---:|---:|
| Smoke | 123 | 3 optimizer updates | 2 |
| Q0筛选 | 123 | 1 epoch | 2 |
| 正式回归 | 123, 124, 125 | 4 epochs | 6 |

Q0通过后正式训练从canonical init重新开始, 不续接Q0 checkpoint. 正式顺序固定为同seed配对的P0后K2.

## 3. 评估与裁决

`last.pt`是主裁决checkpoint, `best.pt`只作敏感性分析. 每个正式checkpoint评估500 examples的`1024x256`, `2048x512`, `4096x1024`, `8190x512`和`8190x2047`.

Q0和正式阶段均执行:

```text
每个seed standard delta >= -0.01
每个seed four-slice extrapolation macro delta >= -0.02
三seed standard mean delta >= -0.01
三seed extrapolation macro mean delta >= -0.02
```

同时要求loss, gradient和checkpoint finite, P0无persistent调用, K2有persistent调用, 全部Triton fallback为0. 最终model/optimizer hash和逐步loss必须记录, 但P0/K2不要求逐位一致.

默认保留FLA autotune行为. 若同seed两组选择不同fused-gate backward config或出现异常轨迹, 使用既有capture/replay做matched-config复验. BF16失败时补seed123 FP32配对用于区分K2实现问题与低精度交互, FP32不进入主质量矩阵.

## 4. 执行与失败策略

**(1)** Preflight锁定两个仓库的干净commit, canonical Python, CUDA/NVML, 3090, 依赖版本, cache/init hash, 参数量和唯一配置差异.

**(2)** Smoke失败时停止Q0. Q0失败时停止正式矩阵. 失败现场使用唯一run tag保留, 定位后只做有证据的最小修复.

**(3)** 若源码或配置改变, 受影响seed的P0/K2必须从canonical init成对重跑. 正式阶段的单seed质量下降不在矩阵中途修改配置, 仍完成全部预注册seed后统一裁决.

**(4)** 3090保留raw, checkpoint和日志. 轻量raw镜像回2080 Ti并校验SHA256. 最终生成artifact, report并更新STATUS与append-only EXPERIMENT_LOG.

## 5. 预算

- 核心实验预计3至4个3090 GPU-hours, 含编译和评估的上限为5 GPU-hours.
- 失败后FP32诊断最多额外2 GPU-hours.
- API费用为0, raw与checkpoint预计低于1 GiB.

## 6. 执行终态

两组smoke均通过. Seed123 AMP BF16一epochQ0的标准validation delta为`-0.010344`, 四外推宏平均delta为`-0.039300`, 均低于预注册门槛. 因此三seed四epoch正式矩阵按第4节停止规则未启动. 补充FP32配对只用于因果诊断, 不覆盖BF16主裁决. 完整结论见[实验报告](../20260730-04-k2-persistent-scan-mqar-regression-report.md).
