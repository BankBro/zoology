# Residual Value Bottleneck MQAR

## 1. 登记

- Experiment ID: `20260731-02-residual-value-bottleneck-mqar`.
- 状态: `completed`, 终态为`quality_rejected_at_q0`.
- Plan: [`docs/plans/20260731-02-residual-value-bottleneck-mqar-plan.md`](../../../../../docs/plans/20260731-02-residual-value-bottleneck-mqar-plan.md).
- GPU: RTX 3090 GPU0.
- Precision: AMP BF16.
- Flash-VQG source: `cc3f92b8a972f1c51c3deabeafd0d9f180bc2b16`.

本实验在A1加S1 exact路径比较U64, U32和U16. Seed123一轮Q0已完成. U32标准任务相对下降`8.34%`, 但四外推宏平均下降`37.87%`; U16标准下降`37.92%`, 外推宏平均下降`74.00%`. 两个候选均未进入三seed四epoch矩阵. 详细结论见[正式报告](../../../../../docs/20260731-02-residual-value-bottleneck-mqar-report.md).

## 2. 运行

```bash
export MQAR_RESIDUAL_VALUE_RUN_TAG=20260731-residual-value-mqar-01
CUDA_VISIBLE_DEVICES=0 \
  /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260731-02-residual-value-bottleneck-mqar/run_queue.py
```

原始输出位于`outputs/3090/<run-tag>/`.
