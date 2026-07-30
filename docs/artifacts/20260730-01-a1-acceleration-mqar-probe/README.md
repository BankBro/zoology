# A1 训练加速候选 MQAR 筛选 Artifact

## 1. 结论口径

本目录保存`20260730-01-a1-acceleration-mqar-probe`的清洗后证据. `block256/write2/read8`在seed123、FP32、1 epoch下被质量门禁拒绝. Exact `triton_deterministic` selected backward的reference达到`0.959813`, 但本实验不替代三seed正式回归.

## 2. 文件说明

| 文件 | 内容 |
|---|---|
| `quality-summary.csv` | 主候选标准与外推结果 |
| `diagnostic-summary.csv` | 三个单变量诊断 |
| `failure-ledger.csv` | 失败、根因、修复和重试结果 |
| `source-manifest.csv` | 源码与raw证据位置 |
| `metadata.json` | 机器可读终态与门槛 |

详细解释见[正式报告](../../20260730-01-a1-acceleration-mqar-probe-report.md). Raw checkpoint、日志和JSONL保留在实验脚本旁的ignored outputs.
