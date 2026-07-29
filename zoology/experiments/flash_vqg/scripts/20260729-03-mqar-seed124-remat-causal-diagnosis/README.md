# MQAR seed124 remat 数值分叉因果诊断

- Experiment ID: `20260729-03-mqar-seed124-remat-causal-diagnosis`.
- 状态: `completed`, 终态为 `causal_root_identified`.
- 目标: 定位 seed124 A0/A1 的首个数值分叉并完成单变量因果验证.
- Plan: [`docs/plans/20260729-03-mqar-seed124-remat-causal-diagnosis-plan.md`](../../../../../../docs/plans/20260729-03-mqar-seed124-remat-causal-diagnosis-plan.md).
- Report: [`docs/20260729-03-mqar-seed124-remat-causal-diagnosis-report.md`](../../../../../../docs/20260729-03-mqar-seed124-remat-causal-diagnosis-report.md).
- Artifact: [`docs/artifacts/20260729-03-mqar-seed124-remat-causal-diagnosis/`](../../../../../../docs/artifacts/20260729-03-mqar-seed124-remat-causal-diagnosis/README.md).
- 机器: RTX 3090, `CUDA_VISIBLE_DEVICES=0`.
- 环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`.

运行入口:

```bash
export MQAR_SEED124_DIAG_RUN_TAG=20260729-seed124-diag-rerun
/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python run_queue.py
```

有效运行绑定Zoology `8c8ceb3/662be93/b98bda6`和Flash-VQG `d7dbb12`. 根因是FLA 0.4.2 fused gate backward的fresh-process Triton autotune选择不同归约config. 固定`BT64, warps4`后,A0/A1 177-step和validation完全一致.
