# E5-train Dense Teacher

本目录承载 train-time `dense_value teacher` 的最小可执行入口.

当前实现边界:

- teacher 固定为 `dense_value`
- loss 固定为 `sparse_top2_ce`
- 只监督 `layer_idx = 1`
- 只做 warm-start fine-tune, 不恢复 optimizer
- 不做 eval-time hard override

主要文件:

- `e5_train_builder.py`: 生成单个 E5-train 配置
- `metrics.yaml`: 建议观测的 train/valid 指标
- `run_e5_train_single.sh`: 通用单次启动入口
- `run_e5_train_control_single.sh`: 通用 paired control 入口, 固定 `lambda=0.0`
- `run_e5_train_smoke.sh`: 1 epoch smoke
- `run_e5_train_screening_s123.sh`: s123 上的 4-epoch screening, 含 control + lambda grid
- `run_e5_train_continuation_calibration.sh`: `lambda=0.0` 的 continuation calibration, 扫 `LR x max_epochs`
- `run_e5_train_rescreening_lr1e4_s123.sh`: 基于 `lr=1e-4, max_epochs=4` 新 control 底座的 teacher rescreening
- `run_e5_train_repro_with_control_s124.sh`: s124 上的 4-epoch repro, 含 control
- `run_e5_train_confirm_32epoch.sh`: 双 seed 的 32-epoch confirm, 含 control
- `run_e5_train_lambda_grid_s123.sh`: 兼容旧名字, 等价于 `run_e5_train_screening_s123.sh`
- `run_e5_train_repro_s124.sh`: 兼容旧名字, 等价于 `run_e5_train_repro_with_control_s124.sh`

最常用命令:

```bash
cd /home/lyj/mnt/project/zoology
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_smoke.sh
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_screening_s123.sh
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_continuation_calibration.sh
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_rescreening_lr1e4_s123.sh
BEST_LAMBDA=0.05 BEST_LAMBDA_TAG=005 \
  bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_repro_with_control_s124.sh
BEST_LAMBDA=0.05 BEST_LAMBDA_TAG=005 \
  bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_confirm_32epoch.sh
```

注意:

- 这里的 `E5-train` 指的是 train-time teacher-guided 路线.
- 它不是阶段 B 里的 eval-time partial override.
- 默认执行顺序是 `smoke -> screening -> repro -> confirm`.
- 从 `screening` 开始, 所有主脚本都自带 paired control, 不需要再手工补 `lambda=0.0`.
- 若 `P1` 显示 continuation 本身在掉点, 先运行 `run_e5_train_continuation_calibration.sh`, 再决定是否重开 teacher grid.
- 当前 continuation calibration 的最佳 control 是 `lr=1e-4, max_epochs=4`, 对应重筛入口是 `run_e5_train_rescreening_lr1e4_s123.sh`.
- `run_e5_train_screening_s123.sh` 默认使用 `SCREENING_GPUS=0,1` 做双卡分片:
  - GPU 0: control + `lambda=0.05`
  - GPU 1: `lambda=0.02` + `lambda=0.10`
- `run_e5_train_rescreening_lr1e4_s123.sh` 默认使用 `RESCREENING_GPUS=0,1` 做双卡分片:
  - GPU 0: control + `lambda=0.01`
  - GPU 1: `lambda=0.005` + `lambda=0.02`
- 如需退回单卡串行, 显式设置 `SCREENING_GPUS=0` 或 `RESCREENING_GPUS=0`.
