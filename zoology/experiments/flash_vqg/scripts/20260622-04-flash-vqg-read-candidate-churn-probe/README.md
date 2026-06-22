# Flash-VQG read candidate churn probe

对应计划:

```text
docs/plans/20260622-04-flash-vqg-read-candidate-churn-probe-plan.md
```

## 文件

- `metrics.yaml`: 本轮需要记录的 accuracy, read-side, write/state health 指标白名单.
- `config_runtime_smoke.py`: 轻量验证 churn probe 配置和 validation 指标链路.
- `run_config_runtime_smoke.sh`: 运行 smoke, 输出到 ignored `outputs/`.
- `run_churn_probe_train.sh`: 运行单个正式 target.
- `launch_wave_tmux.sh`: 在当前机器启动三条 3090 target.
- `collect_results.py`: 训练结束后从 generated manifest 和日志中整理轻量索引.

## 输出

中间文件写到:

```text
zoology/experiments/flash_vqg/scripts/20260622-04-flash-vqg-read-candidate-churn-probe/outputs/
```

`outputs/` 不提交. 训练完成后, 收尾报告和轻量 artifact 写到:

```text
docs/artifacts/20260622-04-flash-vqg-read-candidate-churn-probe/
docs/20260622-04-flash-vqg-read-candidate-churn-probe-report.md
```

## 运行

2080ti smoke:

```bash
cd /home/lyj/mnt/project/zoology
CUDA_VISIBLE_DEVICES=0 bash zoology/experiments/flash_vqg/scripts/20260622-04-flash-vqg-read-candidate-churn-probe/run_config_runtime_smoke.sh --device cuda
```

3090 三条长训:

```bash
cd /home/lyj/mnt/project/zoology
bash zoology/experiments/flash_vqg/scripts/20260622-04-flash-vqg-read-candidate-churn-probe/launch_wave_tmux.sh wave1-3090
```

