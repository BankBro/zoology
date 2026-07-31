# Residual Value Bottleneck MQAR

## 1. 登记

- Experiment ID: `20260731-02-residual-value-bottleneck-mqar`.
- 状态: `ready`.
- Plan: [`docs/plans/20260731-02-residual-value-bottleneck-mqar-plan.md`](../../../../../docs/plans/20260731-02-residual-value-bottleneck-mqar-plan.md).
- GPU: RTX 3090 GPU0.
- Precision: AMP BF16.
- Flash-VQG source: `cc3f92b8a972f1c51c3deabeafd0d9f180bc2b16`.

本实验在A1加S1 exact路径比较U64, U32和U16. Q0使用seed123一epochblock64 MQAR, 通过后再进入三seed四epoch正式矩阵.

## 2. 运行

```bash
export MQAR_RESIDUAL_VALUE_RUN_TAG=20260731-residual-value-mqar-01
CUDA_VISIBLE_DEVICES=0 \
  /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260731-02-residual-value-bottleneck-mqar/run_queue.py
```

原始输出位于`outputs/3090/<run-tag>/`.
