# MQAR GD Remat 回归实验

## 1. 实验登记

- Experiment ID: `20260729-01-mqar-gd-remat-regression`.
- 状态: `completed, quality_failed`.
- 目标: 在RTX 3090 BF16下, 对canonical `baseline-r16-joint`执行A0 remat off与A1 post-phase1 remat三seed配对回归.
- Plan: [`docs/plans/20260729-01-mqar-gd-remat-regression-plan.md`](../../../../../docs/plans/20260729-01-mqar-gd-remat-regression-plan.md).
- Report: [`docs/20260729-01-mqar-gd-remat-regression-report.md`](../../../../../docs/20260729-01-mqar-gd-remat-regression-report.md).
- Artifact: [`docs/artifacts/20260729-01-mqar-gd-remat-regression/`](../../../../../docs/artifacts/20260729-01-mqar-gd-remat-regression/README.md).

## 2. 固定矩阵

| Variant | `fox_gd_residual_remat_mode` | Seeds |
|---|---|---|
| `a0-off` | `off` | `123/124/125` |
| `a1-post-phase1` | `post_phase1` | `123/124/125` |

两组均固定3090 BF16, `block_len=32`, B64, validation B16, GA4, 4 epochs, canonical cache/init和data seed 123. `last.pt`为主结果, `best.pt`只做敏感性分析.

## 3. 阶段与入口

队列依次执行`preflight -> trajectory -> smoke -> formal -> evaluate -> collect`. 任一硬门禁失败立即停止并保留当前run目录.

3090的`Flash-VQG-tun`容器内启动:

```bash
cd /home/lyj/mnt/project/zoology
bash zoology/experiments/flash_vqg/scripts/20260729-01-mqar-gd-remat-regression/start_queue.sh 3090
```

原始输出位于:

```text
zoology/experiments/flash_vqg/scripts/20260729-01-mqar-gd-remat-regression/
outputs/3090/<run-tag>/
```

其中`status.json`, `heartbeat.json`和`logs/`用于监控与失败恢复. 队列幂等复用已经完成且身份匹配的阶段, 不覆盖失败run.

## 4. 裁决

A1必须同时满足:

```text
mean standard MQAR delta >= -0.01
mean four-slice extrapolation macro delta >= -0.02
```

质量通过前不启动300M自然语言质量pilot. 显存和吞吐只记录, 不作为本轮硬门槛.

## 5. 终态

6/6 个正式训练 run 和 60/60 个逻辑评估事件完成. A1 显存下降约 `22.5%`, 但标准 MQAR delta 均值为 `-0.04020`, 四外推 slice 宏平均为 `-0.10562`, 两个质量门槛均失败. A1 不替代 A0.

首次 collector 因 `system-summary.csv` 字段集合不一致而失败. 修复 commit `03f5d25` 仅重建汇总, 未重跑训练或评估. 恢复后的终态位于 raw 的 `final-summary-recovered/`.
