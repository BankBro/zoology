# K2 Persistent Scan MQAR 质量回归 Artifact

## 1. 结论

K2 P8在RTX 3090 AMP BF16的seed123一epochQ0中未通过预注册质量门禁. 标准validation delta为`-0.010344`, 四个外推任务宏平均delta为`-0.039300`. 因此三seed四epoch正式矩阵按计划未启动, K2不能提升为MQAR质量canonical.

补充FP32同seed诊断通过门槛, 但P0/K2最终状态仍分叉且质量delta方向反转. 生产shape梯度分解进一步确认, K2和P0的粗状态与残差状态分支分别一致, 差异只出现在两条梯度于`W_blk`汇合时的FP32累加树. 当前K2应分类为forward exact、backward E1数值顺序候选.

## 2. 文件

| 文件 | 内容 |
|---|---|
| `quality-summary.csv` | BF16 Q0和FP32诊断的标准及五任务结果 |
| `trajectory-summary.csv` | 704条共同训练loss记录的首次、最大和终态差异 |
| `system-summary.csv` | wall time、step p50、显存和runtime audit |
| `gradient-diagnosis.csv` | tile sweep与梯度分支归因 |
| `failure-ledger.csv` | runner错误、质量拒绝和诊断恢复过程 |
| `metadata.json` | 终态、commit、门槛和raw镜像摘要 |
| `source-manifest.csv` | raw、generated config和checkpoint来源 |

## 3. Raw 边界

3090 raw保存在:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260730-04-k2-persistent-scan-mqar-regression/outputs/3090/
```

除checkpoint外的轻量raw已镜像到2080 Ti相同相对路径, 三个run tag的文件数和aggregate SHA256逐项一致. Checkpoint继续保留在3090, 不进入Git.

