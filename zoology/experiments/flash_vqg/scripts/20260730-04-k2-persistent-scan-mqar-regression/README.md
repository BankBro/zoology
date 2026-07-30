# K2 Persistent Scan MQAR 质量回归

## 1. 登记

- Experiment ID: `20260730-04-k2-persistent-scan-mqar-regression`.
- 状态: `implementation`.
- Plan: [`docs/plans/20260730-04-k2-persistent-scan-mqar-regression-plan.md`](../../../../../docs/plans/20260730-04-k2-persistent-scan-mqar-regression-plan.md).
- Report: 终态后写入`docs/20260730-04-k2-persistent-scan-mqar-regression-report.md`.
- GPU: RTX 3090.
- Precision: AMP BF16.
- Flash-VQG source: `a6b1af3d8845caa7f317e7104a1af500a03b1c24`.

本实验比较P0 A1与K2 P8. 两组除`fox_gd_residual_builder`外配置相同, 使用block64和三seed四epoch正式MQAR协议.

## 2. 运行

```bash
export MQAR_K2_PERSISTENT_RUN_TAG=20260730-k2-mqar-01
CUDA_VISIBLE_DEVICES=0 \
  /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260730-04-k2-persistent-scan-mqar-regression/run_queue.py
```

原始输出位于`outputs/3090/<run-tag>/`. 队列依次执行preflight, smoke, seed123一epochQ0, 三seed四epoch正式训练, Longer-MQAR评估和裁决. 任一硬失败保留现场且不覆盖已有run tag.

