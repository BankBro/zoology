# A1训练加速候选MQAR筛选

本目录验证Flash-VQG训练加速实验中最快的`block256 + write2/read8`候选是否存在明显MQAR质量退化. 实验已完成, 终态为`quality_rejected`.

- Plan: [`docs/plans/20260730-01-a1-acceleration-mqar-probe-plan.md`](../../../../../docs/plans/20260730-01-a1-acceleration-mqar-probe-plan.md).
- Report: [`docs/20260730-01-a1-acceleration-mqar-probe-report.md`](../../../../../docs/20260730-01-a1-acceleration-mqar-probe-report.md).
- Artifact: [`docs/artifacts/20260730-01-a1-acceleration-mqar-probe/`](../../../../../docs/artifacts/20260730-01-a1-acceleration-mqar-probe/README.md).

```bash
export CUDA_VISIBLE_DEVICES=1
export MQAR_A1_ACCEL_RUN_TAG=20260730-a1-accel-mqar-01
/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python run_queue.py
```

原始输出写入`outputs/2080ti/<run-tag>/`, 默认不提交. 候选未通过单seed筛选, 因此未启动三seed4epoch正式实验.
