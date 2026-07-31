# Selected-read Warp MQAR 筛选

## 1. 登记

- Experiment ID: `20260731-01-selected-read-warp-mqar-screen`.
- 状态: `completed`, 终态为`quality_mixed`.
- Plan: [`docs/plans/20260731-01-selected-read-warp-mqar-screen-plan.md`](../../../../../docs/plans/20260731-01-selected-read-warp-mqar-screen-plan.md).
- GPU: RTX 3090.
- Precision: AMP BF16.
- Flash-VQG source: `efc75ad5539b636b026c76bedb70878bfe2390cf`.

本实验比较S1 exact、W2 direct和W2加preproject的seed123一轮block64 MQAR质量. 它是资源候选的低成本诊断, 不自动提升质量canonical.

## 2. 运行

```bash
export MQAR_SELECTED_WARP_RUN_TAG=20260731-selected-warp-mqar-01
CUDA_VISIBLE_DEVICES=0 \
  /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260731-01-selected-read-warp-mqar-screen/run_queue.py
```

原始输出位于`outputs/3090/<run-tag>/`. 队列依次执行preflight、三组smoke、三组一轮训练、locked eval和裁决.

实际run tag为`20260731-selected-warp-mqar-01`. W2 direct通过标准和外推门槛, preproject外推失败. 详细结果见[报告](../../../../../docs/20260731-01-selected-read-warp-mqar-screen-report.md).
