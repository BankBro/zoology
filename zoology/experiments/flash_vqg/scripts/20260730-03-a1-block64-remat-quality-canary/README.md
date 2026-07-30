# A1 Block64 Remat 质量基线实验

## 1. 登记

- Experiment ID: `20260730-03-a1-block64-remat-quality-canary`.
- 状态: implementation.
- Plan: [`docs/plans/20260730-03-a1-block64-remat-quality-canary-plan.md`](../../../../../docs/plans/20260730-03-a1-block64-remat-quality-canary-plan.md).
- Report: 终态后写入`docs/20260730-03-a1-block64-remat-quality-canary-report.md`.
- GPU: 2080ti GPU1.
- Precision: FP32.
- Flash-VQG source: `0b50712576ee8f17499152f66e81a1b37ef67517`.

本实验比较block64 A0与A1, 唯一配置差异是`fox_gd_residual_remat_mode`. 默认FLA主运行若出现fresh-process分叉, 结果会登记为`requires_fla_replay`, 后续使用已存在的capture/replay机制做同config因果复验.

## 2. 运行

```bash
export MQAR_BLOCK64_REMAT_RUN_TAG=20260730-block64-remat-01
CUDA_VISIBLE_DEVICES=1 \
  /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260730-03-a1-block64-remat-quality-canary/run_queue.py
```

原始输出位于本目录`outputs/2080ti/<run-tag>/`. 正式运行前queue会执行preflight, 随后依次执行两组3-step smoke、两组1-epoch screen和Longer-MQAR评估. 任何失败保留现场且不覆盖已有run tag.
