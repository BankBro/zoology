# 当前最快 Flash 与 GDN 的 MQAR 正式对照 Artifact

## 1. 结论

本实验在 RTX 3090 AMP BF16 下完成 `flash-fastest`, `flash-canonical` 和 capacity-matched GDN 的三 seed, 四 epoch 正式训练, 共 9/9 个 run, 234/234 个逻辑评估事件和 15/15 个 endpoint fresh-process 重复性检查.

预注册的 `last.pt` 主门禁中, Fastest 相对 Canonical 的标准端点均值 delta 为 `-0.015702`, 四外推宏平均 delta 为 `-0.000430`, 标准 8 任务宏平均 delta 为 `-0.002501`, 全部通过 5 个百分点容忍范围. 但两组 Flash 都在第 1 epoch 后明显退化. Last 四外推宏平均仅约 `0.083`, 低于 GDN 的 `0.214`.

`best.pt` 揭示另一面: Fastest, Canonical 和 GDN 的四外推宏平均分别为 `0.509`, `0.598` 和 `0.214`. 两个 Flash 均明显优于 GDN, 但 Fastest 相对 Canonical 的均值 delta 为 `-0.089462`, seed125 为 `-0.250114`. 因此本实验支持 Fastest 通过预注册 Last 主门禁, 但不支持宣称其训练轨迹或最佳 checkpoint 质量与 Canonical 等价.

## 2. 文件

| 文件 | 内容 |
|---|---|
| `training.csv` | 9 条正式训练的 last/best 指标, wall time, 显存和状态 hash |
| `evaluation-detail.csv` | 234 条 last/best 标准及 Longer-MQAR 逻辑评估明细 |
| `metrics.csv` | 每个 arm, seed 和 checkpoint role 的主要汇总指标 |
| `aggregate.csv` | 三 seed 均值, population SD 和 min/max |
| `paired-deltas.csv` | 同 seed 配对差值 |
| `paired-summary.csv` | 配对均值, population SD, 正向 seed 数和 worst delta |
| `system.csv` | 9 条 run 的运行时间, step p50, 显存和 runtime audit |
| `batch-profile.csv` | 39 个完整负载 capacity 与下一档 batch invariance 结果 |
| `failure-ledger.csv` | 初始 evaluator 口径错误及预注册 OOM 降档的恢复记录 |
| `metadata.json` | 终态, commit, 事件计数和 raw 镜像摘要 |
| `source-manifest.csv` | Raw, generated config 与 checkpoint 来源 |

## 3. Raw 与 checkpoint 边界

3090 原始输出位于:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260801-01-fastest-flash-vs-gdn-mqar/outputs/3090/
20260801-fastest-gdn-mqar-01/
```

除 checkpoint 外的 1712 个文件已镜像回本机相同相对路径, aggregate SHA256 为 `890e41edeacbeafe703ff4b6b559630aaf0d771931a8e770e1ec48bd508f7911`. 60 个 checkpoint 文件继续保留在 3090, 不进入 Git.

实验源码为 Zoology `00a19f291109d0dd1e50326d3005d8f8c8f4c8a7` 与 Flash-VQG `396ae65b89b53aad316fbbf7daf55a92a551d684`.
