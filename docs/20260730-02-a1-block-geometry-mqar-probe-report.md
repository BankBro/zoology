# A1 Block Geometry MQAR 配对实验报告

## 1. 实验登记

- Experiment ID: `20260730-02-a1-block-geometry-mqar-probe`.
- 状态: `completed`, 终态为`quality_rejected`.
- 执行机器: RTX 2080 Ti GPU1.
- Seed和精度: seed 123, FP32.
- Zoology运行commit: `351b4d49ea3e20c745b37198d118bb5ea9e170fa`.
- Flash-VQG运行commit: `60a18b2969df67aace2ab5fa6c10280b766f05a3`.
- Plan: [实验计划](plans/20260730-02-a1-block-geometry-mqar-probe-plan.md).
- 上游报告: [原MQAR筛选](20260730-01-a1-acceleration-mqar-probe-report.md).
- Artifact: [精简证据](artifacts/20260730-02-a1-block-geometry-mqar-probe/README.md).

本实验检验上游大block退化是否只是因为训练序列过短、block数量不足. 它保持训练tokens、microbatch数、optimizer update数、block数及tokens-per-pair几何一致, 再比较`block32`与`block128`.

## 2. 固定合同与执行修复

三个variant都严格加载同一个1,160,390参数初始状态, 使用A1 `post_phase1` remat和`triton_deterministic` selected backward. 每组训练24,320,000 tokens, 2,812个microbatches, GA4. Candidate把序列长度与KV数量放大4倍, 样本数和batch缩小4倍.

执行中保留并修复了两项基础设施失败:

| Attempt | 失败原因 | 修复与结果 |
|---|---|---|
| 01 | `prepare-data`未向上游模块注入必要run tag | 临时注入并在调用后恢复环境变量, preflight及训练通过 |
| 02 | `summarize()`调用上游模块未导出的`load_json` | 改为直接读取UTF-8 JSON, 新增回归测试后摘要生成成功 |

Attempt 02只影响摘要生成. 三个训练结果在错误发生前均已完成, 未被覆盖或重跑.

## 3. 结果

| Variant | 比较任务 | Accuracy | 相对reference | 门禁 |
|---|---|---:|---:|---|
| `block32/write4/read16` | `1024x256` | 0.959813 | 0.000000 | reference |
| `block128/write4/read16` | `4096x1024` | 0.000236 | -0.959576 | rejected |
| `block128/write2/read8` | `4096x1024` | 0.000241 | -0.959571 | rejected |

两个候选的validation loss均约为`8.35`, 准确率接近8192词表的随机水平. 所有loss和checkpoint均finite, 三个3-step smoke及完整1-epoch训练都正常完成. 这不是OOM、NaN或摘要故障导致的假失败. 标准delta门槛为`-0.02`, 因此按计划不继续归一化Longer-MQAR.

## 4. 根因分析

### 4.1. 直接证据

上游固定任务实验中, 只改`block128`或`block256`已经严重退化, 只改`write2/read8`则保留`0.908125`. 本轮补齐block数量与总tokens后, 大block仍未恢复. 两轮结果共同说明当前最快候选的质量风险主要与逻辑block变化相关, 不是selected-read Triton backward本身造成.

### 4.2. 高置信解释

逻辑`block_len`不是纯kernel tile. 从32增至128会同时改变:

- state写回与remote可见边界的token频率.
- 两个local blocks覆盖的绝对token窗口.
- 每个block内参与局部竞争和打包的token及事件数量.
- 相同序列长度下的状态轨迹数量.

因此参数量不变不代表模型语义或有效容量不变. 3090上的约3.14倍性能提升主要来自逻辑block数下降4倍, 不能归类为严格等价优化.

### 4.3. 仍然存在的混杂

本轮为了匹配block数量, 同时把单样本长度和KV数量放大4倍. 它保持了tokens-per-pair, 但也把单样本绝对检索负载扩大4倍, 固定状态容量并未同步扩大. 因此本轮足以否决“简单改大block即可安全提速”, 却不能单独证明大block在所有自然语言任务上必然退化.

更严格的下一步不是继续扩大MQAR任务, 而是在固定自然语言序列和相同consumed tokens下做短paired pilot, 或实现保持逻辑`block_len=64`、仅扩大物理kernel tile的exact路径.

## 5. 结论

`block128/256`及其`write2/read8`组合不进入300M正式预训练配置, 不启动三seed扩展. 当前性能报告必须把这些候选标为`capacity_approx`并与exact候选分开.

保留的工程方向是逻辑block不变的物理tiling、persistent scan或state-build/read融合. 这些方向需要在每个逻辑边界保持相同state和remote索引, 才能合理主张数学等价.

## 6. 原始证据

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260730-02-a1-block-geometry-mqar-probe/outputs/2080ti/
20260730-a1-block-geometry-01
```

Raw目录包含preflight、三个smoke、三个完整训练、checkpoint、telemetry、两次失败现场及最终`summary.json`. 大型checkpoint不进入Git.
