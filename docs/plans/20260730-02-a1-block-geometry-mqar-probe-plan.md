# A1 Block Geometry MQAR 配对实验计划

## 1. 实验登记

- Experiment ID: `20260730-02-a1-block-geometry-mqar-probe`.
- 状态: registered.
- 执行机器: `mclab-2080ti`的`Flash-VQG-tun`, GPU1.
- Zoology base: `20260730-055000-a1-acceleration-mqar-probe`分支.
- Flash-VQG runtime commit: `60a18b2`, 其中相对性能run commit `114eadb`仅修正实验语义标签并新增测试, `src/`内容不变.
- 上游实验: [`20260730-01`计划](20260730-01-a1-acceleration-mqar-probe-plan.md).

本实验修正上游MQAR probe的block geometry混杂因素. 上游保持原始最长训练序列256, 使`block128/256`分别只产生2/1个block, remote path没有获得与canonical `block32`相当的训练. 本实验同步缩放序列长度, KV数量和batch, 再判断大block候选的质量变化.

## 2. 固定合同

| 项目 | Reference | Geometry candidate |
|---|---:|---:|
| `block_len` | 32 | 128 |
| Sequence scale | 1 | 4 |
| KV-pair scale | 1 | 4 |
| Train examples | 原始值 | 原始值除以4 |
| Train/eval batch | 64/16 | 16/4 |
| Gradient accumulation | 4 | 4 |
| 最长训练序列的block数 | 8 | 8 |
| 每段训练tokens | 原始值 | 与reference相同 |
| Seed/data seed | 123/123 | 123/123 |
| Precision | FP32 | FP32 |
| Remat/selected backward | `post_phase1`/`triton_deterministic` | 相同 |

候选包括`block128/write4/read16`和`block128/write2/read8`. 两组均从同一个canonical init严格加载. 本轮为单seed screen, 不替代三seed正式质量回归.

## 3. 任务映射与门禁

Reference任务`L x K`映射为候选`4L x 4K`. 例如标准任务从`1024x256`映射到`4096x1024`. 该映射保持序列内block数量, query密度和每个block的相对任务几何一致.

执行顺序为prepare-data、preflight、3-step smoke、1-epoch screen和标准任务比较. Candidate标准准确率相对reference的delta不低于`-0.02`时, 才继续5个归一化Longer-MQAR slice. 若失败, 先保留现场并分析; 可修复的基础设施或资源问题修复后重试, 不把第一次异常直接当作模型裁决.

## 4. 结论边界

通过本实验只表示大block在归一化MQAR curriculum下具备单seed非劣迹象. 它不能证明300M自然语言质量不下降, 也不能授权完整1B-token训练. 若通过, 仍需短自然语言pilot或三seed正式MQAR门禁.
