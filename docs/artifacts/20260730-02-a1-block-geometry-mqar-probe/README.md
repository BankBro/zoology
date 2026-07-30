# A1 Block Geometry MQAR 配对 Artifact

## 1. 结论口径

本目录保存`20260730-02-a1-block-geometry-mqar-probe`的清洗后证据. 在等训练tokens、microbatch数和block几何的seed123实验中, 两个`block128`候选均接近随机, 因而被质量门禁拒绝.

该结果否决直接扩大逻辑block的当前候选, 但不等价于自然语言质量的普遍定理. Candidate同时承担了4倍单样本绝对长度与KV负载.

## 2. 文件说明

| 文件 | 内容 |
|---|---|
| `quality-summary.csv` | Reference与两个候选的标准任务结果 |
| `failure-ledger.csv` | 两次基础设施失败、修复和重试结果 |
| `source-manifest.csv` | 源码、cache和raw路径 |
| `metadata.json` | 机器可读实验合同与终态 |

详细解释见[正式报告](../../20260730-02-a1-block-geometry-mqar-probe-report.md).
