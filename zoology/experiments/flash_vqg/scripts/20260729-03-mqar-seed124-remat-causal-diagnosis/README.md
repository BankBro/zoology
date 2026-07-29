# MQAR seed124 remat 数值分叉因果诊断

- Experiment ID: `20260729-03-mqar-seed124-remat-causal-diagnosis`.
- 状态: `implementing`.
- 目标: 定位 seed124 A0/A1 的首个数值分叉并完成单变量因果验证.
- Plan: [`docs/plans/20260729-03-mqar-seed124-remat-causal-diagnosis-plan.md`](../../../../../../docs/plans/20260729-03-mqar-seed124-remat-causal-diagnosis-plan.md).
- 机器: RTX 3090, `CUDA_VISIBLE_DEVICES=0`.
- 环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`.

运行入口:

```bash
export MQAR_SEED124_DIAG_RUN_TAG=20260729-seed124-diag-01
/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python run_queue.py
```

队列先执行 preflight, 再运行 A0/A1 16-step 初始定位、自重复和首个窗口详细定位. 任一硬门禁失败立即停止并保留现场.
