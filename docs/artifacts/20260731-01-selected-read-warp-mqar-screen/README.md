# Selected-read Warp MQAR 筛选 Artifact

## 1. 结论口径

W2 direct在seed123 AMP BF16一轮MQAR中通过注册门槛: 标准delta为`-0.005613`, 四外推宏平均delta为`-0.019200`. Preproject W2的外推宏平均delta为`-0.036765`, 已拒绝.

W2 direct仍是fast resource candidate, 不是质量canonical. S1 exact保持当前质量路径. 详细解释见[正式报告](../../20260731-01-selected-read-warp-mqar-screen-report.md).

## 2. 文件说明

| 文件 | 内容 |
|---|---|
| `quality.csv` | 两个candidate相对S1的标准及外推质量delta |
| `system.csv` | 三组一轮训练的wall、step p50与显存 |
| `source-manifest.csv` | Commit、run tag、raw镜像与checkpoint位置 |
| `metadata.json` | 机器可读终态与门槛 |
