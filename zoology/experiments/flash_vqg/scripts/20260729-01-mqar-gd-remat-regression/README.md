# MQAR GD Remat 回归实验

## 1. 实验登记

- Experiment ID: `20260729-01-mqar-gd-remat-regression`.
- 状态: `planned`.
- 目标: 在RTX 3090 BF16下, 对canonical `baseline-r16-joint`执行A0 remat off与A1 post-phase1 remat三seed配对回归.
- Plan: [`docs/plans/20260729-01-mqar-gd-remat-regression-plan.md`](../../../../../docs/plans/20260729-01-mqar-gd-remat-regression-plan.md).
- Report: 实验终态后写入`docs/20260729-01-mqar-gd-remat-regression-report.md`.
- Artifact: 实验终态后写入`docs/artifacts/20260729-01-mqar-gd-remat-regression/`.

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
