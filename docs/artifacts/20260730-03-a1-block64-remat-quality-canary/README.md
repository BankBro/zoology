# A1 Block64 Remat 质量门禁 Artifact

## 1. 结论口径

本目录保存`20260730-03-a1-block64-remat-quality-canary`的精简证据. 在seed123、FP32、block64的一epoch配对中, A0和A1的704-step loss、最终model/optimizer state及全部标准和外推质量指标完全一致.

该结果允许继续P0/P1和C1/K1工程探索, 但不替代300M BF16短自然语言paired pilot.

## 2. 文件说明

| 文件 | 内容 |
|---|---|
| `quality-summary.csv` | 标准与Longer-MQAR配对结果 |
| `trajectory-summary.csv` | 轨迹、state hash、FLA config与资源摘要 |
| `failure-ledger.csv` | 首次OOM、根因、修复和重跑终态 |
| `source-manifest.csv` | 源码与raw路径 |
| `metadata.json` | 机器可读实验合同与终态 |

详细解释见[正式报告](../../20260730-03-a1-block64-remat-quality-canary-report.md).
