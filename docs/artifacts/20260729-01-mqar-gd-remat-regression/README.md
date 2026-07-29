# 20260729-01 MQAR GD Remat 回归 artifact

本目录保存 A0 remat off 与 A1 post-phase1 remat 的三 seed 正式回归结果. 6/6 个正式训练 run, 60 个逻辑评估事件和 30 个物理评估事件全部完成. A1 的标准 MQAR delta 均值为 `-0.04020`, 四外推 slice 宏平均为 `-0.10562`, 两个预注册质量门槛均失败.

## 1. 结果文件

- `summary.json`: 质量门槛与终态.
- `training-final.csv`: 六条正式训练结果, checkpoint 和系统指标.
- `longer-mqar-detail.csv`: last/best 五 slice 逻辑评估明细.
- `paired-quality.csv`: 三 seed 逐项 A0/A1 配对结果.
- `paired-quality-summary.csv`: 每个任务的三 seed 均值与 population SD.
- `system-summary.csv`: A0/A1 wall time, step p50 和显存汇总.
- `trajectory-summary.json`: 32-step 数值轨迹和 runtime audit.
- `canonical-training-ledger.csv`: 本实验追加到 canonical ledger 的六条正式训练记录.
- `source-manifest.csv`: raw, resolved config 和 checkpoint 的来源与 SHA256.
- `metadata.json`: commit, cache, 镜像和 collector 恢复元数据.

## 2. 结论

A1 peak allocated 降至 A0 的约 `0.775x`, 但 wall time 增至 `1.149x`, 且质量显著退化. 32-step probe 在 step1 严格一致, step16 已出现 `2.38e-7` 参数差异, step32 增至 `1.00e-6`; 四个 epoch 后三个 seed 均发生 model-state hash 分叉. A1 因此不能替代 A0, 也不能作为自然语言正式训练默认项.

## 3. Raw 与恢复边界

实验实际运行于 Zoology `62985d0b4866`, Flash-VQG `79fef6a8e9d3`. 首次 collector 因 CSV schema bug 失败; `03f5d2583c96` 仅修复汇总字段并复用原始结果生成 recovered summary, 未重跑训练或评估.

3090 raw 路径为:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260729-01-mqar-gd-remat-regression/outputs/3090/20260729-mqar-remat-01/
```

除 checkpoint 外的 182 个轻量文件已镜像回本机同相对路径, 并逐文件 SHA256 一致. 12 个 best/last checkpoint 保留在 3090, 其 file hash 和 model-state hash 记录在 manifest.

详细解释见 [正式报告](../../20260729-01-mqar-gd-remat-regression-report.md).
