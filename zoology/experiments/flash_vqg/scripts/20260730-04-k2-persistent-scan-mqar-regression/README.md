# K2 Persistent Scan MQAR 质量回归

## 1. 登记

- Experiment ID: `20260730-04-k2-persistent-scan-mqar-regression`.
- 状态: `completed`, 终态为`quality_rejected_at_bf16_screen`.
- Plan: [`docs/plans/20260730-04-k2-persistent-scan-mqar-regression-plan.md`](../../../../../docs/plans/20260730-04-k2-persistent-scan-mqar-regression-plan.md).
- Report: [`docs/20260730-04-k2-persistent-scan-mqar-regression-report.md`](../../../../../docs/20260730-04-k2-persistent-scan-mqar-regression-report.md).
- Artifact: [`docs/artifacts/20260730-04-k2-persistent-scan-mqar-regression/README.md`](../../../../../docs/artifacts/20260730-04-k2-persistent-scan-mqar-regression/README.md).
- GPU: RTX 3090.
- Precision: AMP BF16.
- Flash-VQG source: `a6b1af3d8845caa7f317e7104a1af500a03b1c24`.

本实验比较P0 A1与K2 P8. 两组除`fox_gd_residual_builder`外配置相同, 使用block64和三seed四epoch正式MQAR协议. Seed123 AMP BF16一epochQ0未通过预注册非劣门槛, 因而正式三seed矩阵按计划未启动.

## 2. 运行

```bash
export MQAR_K2_PERSISTENT_RUN_TAG=20260730-k2-mqar-01
CUDA_VISIBLE_DEVICES=0 \
  /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260730-04-k2-persistent-scan-mqar-regression/run_queue.py
```

原始输出位于`outputs/3090/<run-tag>/`. 队列依次执行preflight, smoke, seed123一epochQ0, 三seed四epoch正式训练, Longer-MQAR评估和裁决. 任一硬失败保留现场且不覆盖已有run tag.

实际run tag如下:

- `20260730-k2-mqar-01`: smoke audit误报现场, Q0未启动.
- `20260730-k2-mqar-02`: BF16 smoke与Q0, 质量门禁失败.
- `20260730-k2-mqar-03`: 预注册FP32因果诊断, 不进入主质量裁决.

BF16标准validation delta为`-0.010344`, 四外推宏平均delta为`-0.039300`. 根因定位为P0与K2在`W_blk`处使用不同的FP32梯度累加树. K2应分类为forward exact、backward E1; 当前质量路径仍为P0 A1.
