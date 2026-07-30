# K2 Persistent Scan MQAR 质量回归报告

## 1. 结果概览

- Experiment ID: `20260730-04-k2-persistent-scan-mqar-regression`.
- 状态: `completed`, 终态为`quality_rejected_at_bf16_screen`.
- 目标机器: RTX 3090.
- Flash-VQG source: `20260730-171547-a1-persistent-scan@a6b1af3`.
- Plan: [实验计划](plans/20260730-04-k2-persistent-scan-mqar-regression-plan.md).
- Artifact: [精简证据](artifacts/20260730-04-k2-persistent-scan-mqar-regression/README.md).

K2 P8通过全部smoke和runtime门禁, 但在seed123 AMP BF16一epochQ0中未通过标准及Longer-MQAR非劣门槛. 按预注册停止规则, 三seed四epoch正式矩阵未启动. 补充FP32诊断通过质量门槛, 但结果方向与BF16相反且最终状态仍分叉, 因此不能覆盖BF16拒绝结论.

## 2. 实验合同与执行

P0与K2共同使用`baseline-r16-joint`, canonical cache/init, data seed123, `block_len=64`, rank16, write4/read16, `post_phase1` remat, deterministic selected backward和`fp32_boundary`. 唯一模型配置差异为:

```text
P0: fox_gd_residual_builder=grouped_chunk_torch_ref
K2: fox_gd_residual_builder=persistent_scan_triton, tile_blocks=8
```

Preflight确认两个仓库干净、环境和cache/init hash匹配、参数量为1,160,390, 十个resolved config的审计通过, P0/K2 normalized config只差builder. Run02的两组smoke均完成3 updates, Q0两组均完成704个optimizer updates、checkpoint及10个固定评估事件. K2记录7,646次persistent调用, fallback为0.

## 3. BF16 Q0 质量结果

### 3.1. 标准与外推任务

| 任务 | P0 | K2 | Delta |
|---|---:|---:|---:|
| Validation `1024x256` | 0.962172 | 0.951828 | **-0.010344** |
| Locked eval `1024x256` | 0.961688 | 0.952977 | -0.008711 |
| `2048x512` | 0.831316 | 0.801977 | -0.029340 |
| `4096x1024` | 0.529271 | 0.471725 | -0.057547 |
| `8190x512` | 0.691324 | 0.667418 | -0.023906 |
| `8190x2047` | 0.217395 | 0.170990 | -0.046405 |
| 四外推宏平均 | 0.567327 | 0.528027 | **-0.039300** |

预注册门槛分别为`-0.01`和`-0.02`. 标准validation仅比门槛低`0.000344`, 但四个外推任务全部负向且宏平均明显失败, 不能解释为单个指标的边界抖动.

### 3.2. 训练轨迹

704条共同loss记录中, 前5步相同, step6首次出现`-2.29e-5`差异. 差异随后被训练放大, step394达到最大绝对差`4.4935`; 终态model和optimizer hash均不同.

两组FLA fused gate backward均选择`BT64, warps8`, 因而本轮没有历史warps4/warps8混杂. Runtime、finite、dtype和fallback门禁也全部通过. 当前证据支持真实的K2数值路径差异, 不支持把结果归因于FLA autotune或runner错误.

## 4. FP32诊断与根因

### 4.1. FP32质量只作因果证据

同一seed、同一一epoch合同改为FP32后:

| 指标 | P0 | K2 | Delta |
|---|---:|---:|---:|
| Validation `1024x256` | 0.957727 | 0.968668 | +0.010941 |
| 四外推宏平均 | 0.534157 | 0.609421 | +0.075264 |

FP32通过门槛, 但P0/K2最终model和optimizer hash仍不同, loss从step20开始分叉, 最大loss差为`2.8841`. 单seed正向delta不能证明K2更优; 它只说明同一个浮点顺序差异在不同精度下可进入相反训练轨迹.

### 4.2. 梯度累加树定位

生产shape standalone先确认P1/P2/P4/P8的最大梯度误差均为`7.629e-6`, 且集中在`W_blk`; tile大小不是原因.

随后分别对粗状态`G/L`和残差状态`M`反传:

- 粗状态分支的`W_blk`梯度逐位一致.
- 残差状态分支的`W_blk`梯度逐位一致.
- 两条分支同时反传时, 完整`W_blk`梯度差为`7.629e-6`.
- K2完整梯度严格等于两条分支梯度的显式FP32和.
- P0完整梯度与该显式和相差`7.629e-6`, 对应逐block计算图中的交错累加树.

因此K2与P0的公式及分支梯度一致, 差异来自多个正确梯度贡献在`W_blk`处的FP32归约顺序. 要逐位复刻P0需要重建跨block交错反向树, 不是减小tile或局部cast可以安全修正的问题.

## 5. 资源信号

| 精度 | Variant | Wall | Step p50 | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|---:|
| BF16 | P0 | 221.41 s | 0.2369 s | 1456.64 MiB | 2192 MiB |
| BF16 | K2 | 211.68 s | 0.2432 s | 1461.35 MiB | 2260 MiB |
| FP32 | P0 | 226.08 s | 0.2442 s | 1765.97 MiB | 2662 MiB |
| FP32 | K2 | 223.12 s | 0.2572 s | 1765.97 MiB | 2662 MiB |

小模型Q0中K2 wall略短, 但step p50略慢, 不能替代300M上已注册的1.545x性能结果. 本实验只用这些指标确认资源稳定, 不重新裁决300M吞吐.

## 6. 失败闭环

Run01中K2实际完成3/3 smoke updates, 但runner错误要求persistent audit包含旧grouped路径的`actual_core_dtype`字段, 因而误报失败. 修复仅调整audit合同, 新增测试后使用Run02从preflight和两组smoke完整重跑.

Run02在BF16 Q0质量门禁失败后按计划停止, 没有启动任何正式seed. Run03只执行预注册FP32诊断. 2080 Ti上的梯度分解因SM75不支持BF16 Triton指令而停止, 同一脚本随后在3090完成, 不构成科学结果缺失.

## 7. 结论与后续

K2仍是300M上最快且显存更低的资源候选, 但本轮证明其backward并非逐位E0. 更准确的分类是:

```text
forward: 当前production shape逐位一致
backward: 数学等价, FP32归约树不同, 属于E1
BF16 MQAR质量: seed123 Q0未通过
```

因此:

- K2不提升为MQAR质量canonical, 不直接进入300M BF16自然语言正式训练或1B-token训练.
- 当前质量路径继续使用P0 A1, 即`post_phase1 remat + triton_deterministic selected backward`.
- 若继续K2, 需要单独研究保持persistent数据流同时控制跨分支`W`梯度累加树的新backward, 完成低层门禁后重新从BF16 Q0开始.
- 也可以把K2明确视为E1候选并重新设计统计性质量协议, 但这属于新的科学决策, 不能事后放宽本实验阈值.

## 8. 原始证据

Run01、Run02和Run03除checkpoint外的轻量raw已镜像回2080 Ti, 文件数和aggregate SHA256逐项一致. 24个best/last/resume checkpoint文件保留在3090. 完整来源见[source manifest](artifacts/20260730-04-k2-persistent-scan-mqar-regression/source-manifest.csv).

