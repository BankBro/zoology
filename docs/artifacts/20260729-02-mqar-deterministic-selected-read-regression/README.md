# 20260729-02 MQAR 确定性 Selected-Read 回归 artifact

本目录保存 fixed A0 `off` 与 fixed A1 `post_phase1` 的三 seed 正式回归结果. 6/6 个正式训练 run, 60 个逻辑评估事件和 20 个物理评估事件全部完成. 标准 MQAR 和四外推任务的质量门槛均通过, 但 seed124 的最终 A0/A1 model-state hash 不同, 因而终态为 `quality_recovered_but_not_deterministic`.

## 1. 结果文件

- `summary.json`: 预注册门槛和终态.
- `training-final.csv`: 六条正式训练结果, checkpoint hash 和系统指标.
- `longer-mqar-detail.csv`: last/best 五个任务的逻辑评估明细.
- `paired-quality.csv`: 三 seed 逐项 A0/A1 配对结果.
- `paired-quality-summary.csv`: 每个任务的三 seed 均值和 population SD.
- `final-hash-pairs.csv`: 三个 seed 的最终 model-state hash 配对.
- `historical-comparison.csv`: 修复前后配对 delta 对比.
- `system-summary.csv`: A0/A1 wall time, step p50 和显存汇总.
- `determinism-summary.json`: 128-step lockstep 和 fresh-process 确定性门禁.
- `canonical-training-ledger.csv`: 本实验追加到 canonical ledger 的六条正式训练记录.
- `source-manifest.csv`: raw, resolved config 和 checkpoint 的来源与 SHA256.
- `metadata.json`: commit, cache, 镜像和证据边界.

## 2. 结论

固定顺序修改将历史标准 MQAR delta 从 `-0.04020` 改善为 `+0.00650`, 将四外推任务宏平均 delta 从 `-0.10562` 改善为 `+0.01743`. seed123 和 seed125 的四 epoch A0/A1 状态逐位一致, 历史退化最严重的 seed125 已完全恢复.

但 seed124 的最终 hash 仍不一致. A1 在该 seed 上反而更好, 不能把本轮的正向质量 delta 解释为已经获得稳定增益. A1 尚未满足严格确定性门槛, 不替代 A0, 也不进入 300M 自然语言正式训练默认配置.

## 3. Raw 边界

实验运行于 Zoology `22ed61c8b963`, Flash-VQG `d7dbb1282d20`. 3090 raw 路径为:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260729-02-mqar-deterministic-selected-read-regression/outputs/3090/
20260729-deterministic-selected-01/
```

除 checkpoint 外的 134 个文件和 12 个 resolved config 已镜像回 2080 Ti 工作区, 并逐文件验证 SHA256 一致. 18 个 checkpoint tensor 文件保留在 3090, 其 file hash 记录在 manifest.

详细解释见 [正式报告](../../20260729-02-mqar-deterministic-selected-read-regression-report.md).
