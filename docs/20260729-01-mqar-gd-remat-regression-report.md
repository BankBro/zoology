# MQAR GD Remat 回归实验报告

## 1. 结果概览

`20260729-01-mqar-gd-remat-regression` 已完成 6/6 个正式训练 run, 60 个逻辑 Longer-MQAR 评估事件和 30 个物理评估事件. A1 `post_phase1` remat 的显存优化成立, 但两个预注册质量门槛均失败, 因而不能替代 A0 `off`, 也不能作为 300M 自然语言正式训练的默认实现.

- 标准 MQAR `1024x256` 三 seed 配对 delta 均值为 `-0.04020`, 低于门槛 `-0.01`.
- 四个真正外推 slice 的 12 个配对 delta 宏平均为 `-0.10562`, 低于门槛 `-0.02`.
- A1 peak allocated 约降至 A0 的 `0.775x`, peak reserved 约降至 `0.768x`.
- A1 平均 wall time 为 A0 的 `1.149x`, optimizer-step p50 为 `1.185x`.
- 所有 runtime audit 均通过, fallback 为 0. 失败不是 OOM, fallback 或未执行 remat, 而是长训练轨迹发生数值分叉并造成质量退化.

因此当前决策是保留 A0 为 canonical MQAR 实现, 淘汰 A1 作为直接替代方案. 在新的显存方案通过同等质量门禁前, 不启动使用 A1 的自然语言正式预训练.

## 2. 实验口径

实验在 RTX 3090 上使用 AMP BF16, FP32 master weights 和 optimizer state. 模型固定为 `baseline-r16-joint`, 1,160,390 个可训练参数, `block_len=32`, codebook 64, rank 16, read top-k 16, write top-k 4. 两组均使用 B64, validation B16, GA4, 4 epochs, data seed 123 和 training seeds `123/124/125`.

唯一变量为:

| Variant | `fox_gd_residual_remat_mode` |
|---|---|
| A0 | `off` |
| A1 | `post_phase1` |

Preflight 绑定 Zoology `62985d0b4866`, Flash-VQG `79fef6a8e9d3`, cache `d9098e876a03` 和初始模型状态 `2a1107bf22d0`. Preflight、32-step trajectory、受控 resume smoke、checkpoint save/load 和评估 smoke 均通过后, 才启动六条正式训练.

主结果使用 last checkpoint. 六条 run 的 best 与 last model-state hash 分别相同, 因此 best 选择不会改变结论. 标准 MQAR 门槛使用训练端点 `1024x256`, 不是跨全部 validation slice 的 `valid/accuracy`.

## 3. 质量结果

### 3.1. 三 seed 门禁

| Seed | 标准 MQAR delta | 四外推 slice 宏平均 delta |
|---:|---:|---:|
| 123 | -0.01577 | -0.03569 |
| 124 | -0.01134 | -0.01597 |
| 125 | -0.09348 | -0.26520 |
| **均值** | **-0.04020** | **-0.10562** |
| **门槛** | **>= -0.01** | **>= -0.02** |

seed125 放大了退化, 但 seed123 和 seed124 的标准端点也均为负, 且 seed123 的外推宏平均已经低于门槛. 因此不能把失败仅解释为单个异常 seed.

### 3.2. 分 slice 均值

| 任务 | A0 mean | A1 mean | A1 - A0 mean |
|---|---:|---:|---:|
| 1024x256 | 0.97774 | 0.93755 | -0.04020 |
| 2048x512 | 0.83625 | 0.70751 | -0.12874 |
| 4096x1024 | 0.46855 | 0.34616 | -0.12239 |
| 8190x512 | 0.70655 | 0.58074 | -0.12580 |
| 8190x2047 | 0.15745 | 0.11190 | -0.04555 |

跨全部 validation slice 的 `valid/accuracy` delta 均值为 `-0.00584`, 但该汇总会被较容易的短序列任务稀释, 不是本实验的预注册标准门槛. 最难训练端点和真正外推任务均显示更明显退化.

## 4. 显存与速度

| Variant | Run | Wall time mean, min | Step p50 mean, s | Peak allocated max, MiB | Peak reserved max, MiB |
|---|---:|---:|---:|---:|---:|
| A0 off | 3 | 18.66 | 0.3054 | 2566.2 | 3158 |
| A1 post-phase1 | 3 | 21.44 | 0.3619 | 1988.4 | 2426 |

A1 减少约 `577.8 MiB` peak allocated 和 `732 MiB` peak reserved, 代价是 wall time 增加 `14.9%`, step p50 增加 `18.5%`. 三条 A1 均记录 `11260` 次 selected-read recompute, fallback 为 0, 说明测到的是实际 remat 路径.

这些结果证明 A1 的工程显存收益是真实的, 但显存收益不能覆盖质量门禁失败. 该实现只保留为诊断和后续修正对象.

## 5. 数值分叉分析

32-step probe 揭示了短测通过与长训练失败之间的差异:

| Step | Loss abs delta | Parameter max abs | Optimizer max abs | 裁决 |
|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 0 | 严格一致 |
| 16 | 0 | 2.38e-7 | 1.18e-12 | allclose, 但已非逐位一致 |
| 32 | 0 | 1.00e-6 | 8.49e-12 | allclose, 差异继续增大 |

最早可见差异集中在 `fox_gd_residual_addr_proj` 及其 Adam moments. 四个 epoch 后, 三个 seed 的 A0/A1 model-state hash 全部不同. seed125 的 A1 还出现了明显更晚的首轮收敛转折, 最终 `1024x256` delta 为 `-0.09348`.

直接证据支持以下结论: 当前 A1 不是长训练下的数值无操作. remat 重算引入的低阶差异会沿敏感训练轨迹被放大. 现有实验尚未进一步区分具体来源是 checkpoint 重算顺序, Triton reduction 数值路径, AMP BF16 交互或它们的组合, 因而不能把机制原因写成已经定位的确定 bug.

## 6. Collector 恢复与证据边界

首次 collect 在写 `system-summary.csv` 时因 A0/A1 行字段集合不同而失败. 队列把 collect 阶段的所有异常粗略标记为 `quality_failed`, 但当时 6/6 训练和 30/30 物理评估均已完成. commit `03f5d2583c96` 只为 A0 补充两个空 ratio 字段并增加回归测试, 然后复用原始结果生成 `final-summary-recovered`.

修复后的 collector 才给出本文的真实质量失败结论. 未重跑训练或评估, 未修改门槛, 未补 seed. 原始 collector 异常和恢复后结论均保留在 source manifest 中.

3090 raw 现场总计约 259 MiB. 除 checkpoint 外的 182 个轻量文件已镜像回 2080 Ti 工作区并逐文件 SHA256 一致. 12 个 best/last checkpoint 保留在 3090 原路径, file hash 与 model-state hash 已记录.

## 7. 决策与下一步

- A0 `off` 继续作为 canonical MQAR 质量基线.
- A1 `post_phase1` 不进入 300M 自然语言正式训练, 不以其显存收益作为放行理由.
- 不自动追加 seed, 不放宽非劣门槛, 不用 best checkpoint 覆盖 last 主结论.
- 后续若继续 remat, 应先定位并消除 `addr_proj` 早期数值分叉, 再从头执行同一三 seed 门禁.
- 若优先追求显存, 可转向不重算 GD 图的方案, 例如 native-BF16 selected-read, 但仍必须独立通过 MQAR 和 Longer-MQAR 回归后才能进入自然语言训练.

完整数值见 [artifact README](artifacts/20260729-01-mqar-gd-remat-regression/README.md), [paired quality](artifacts/20260729-01-mqar-gd-remat-regression/paired-quality.csv), [system summary](artifacts/20260729-01-mqar-gd-remat-regression/system-summary.csv) 和 [source manifest](artifacts/20260729-01-mqar-gd-remat-regression/source-manifest.csv).
