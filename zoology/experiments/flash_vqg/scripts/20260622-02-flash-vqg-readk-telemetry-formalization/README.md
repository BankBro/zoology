# Flash-VQG readk telemetry formalization scripts

本目录对应 `docs/plans/20260622-02-flash-vqg-readk-telemetry-formalization-plan.md`.

## 脚本

- `config_runtime_smoke.py`: 复用 config-to-runtime smoke, 增加 read-side telemetry required metrics.
- `run_config_runtime_smoke.sh`: 运行 smoke, 输出到 ignored `outputs/`.
- `run_fixed_readk4_train.sh`: 运行单个 fixed readk4 target.
- `launch_wave_tmux.sh`: 在当前机器启动三条 full run tmux session.

## 输出

中间输出写到:

```text
zoology/experiments/flash_vqg/scripts/20260622-02-flash-vqg-readk-telemetry-formalization/outputs/
```

`outputs/` 不提交. 训练完成后, 将轻量结果整理到:

```text
docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/
```

## 运行

2080ti smoke:

```bash
cd /home/lyj/mnt/project/zoology
CUDA_VISIBLE_DEVICES=0 bash zoology/experiments/flash_vqg/scripts/20260622-02-flash-vqg-readk-telemetry-formalization/run_config_runtime_smoke.sh --device cuda
```

3090 full runs:

```bash
cd /home/lyj/mnt/project/zoology
bash zoology/experiments/flash_vqg/scripts/20260622-02-flash-vqg-readk-telemetry-formalization/launch_wave_tmux.sh wave1-3090
```
