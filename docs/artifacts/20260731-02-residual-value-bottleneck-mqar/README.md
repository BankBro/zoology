# Residual Value Bottleneck MQAR Artifact

## 1. 结论口径

U32标准MQAR相对U64下降`8.34%`, 但四外推宏平均下降`37.87%`. U16标准下降`37.92%`, 外推宏平均下降`74.00%`. 两个候选均在seed123 AMP BF16 Q0被拒绝, 三seed四epoch矩阵未启动. 详细解释见[正式报告](../../20260731-02-residual-value-bottleneck-mqar-report.md).

## 2. 文件说明

| 文件 | 内容 |
|---|---|
| `quality.csv` | 标准与四个外推slice的准确率和相对变化 |
| `system.csv` | 三组一轮训练的wall、step p50与显存 |
| `failure-ledger.csv` | 基础设施失败、修复和科学影响 |
| `source-manifest.csv` | Commit、run tag和raw目录 |
| `metadata.json` | 机器可读终态与门槛 |
