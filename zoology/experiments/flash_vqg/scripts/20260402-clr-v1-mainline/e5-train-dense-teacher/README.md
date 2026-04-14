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
- `run_e5_train_repro_with_control_s124.sh`: s124 上的 4-epoch repro, 含 control
- `run_e5_train_confirm_32epoch.sh`: 双 seed 的 32-epoch confirm, 含 control
- `run_e5_train_lambda_grid_s123.sh`: 兼容旧名字, 等价于 `run_e5_train_screening_s123.sh`
- `run_e5_train_repro_s124.sh`: 兼容旧名字, 等价于 `run_e5_train_repro_with_control_s124.sh`

最常用命令:

```bash
cd /home/lyj/mnt/project/zoology
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_smoke.sh
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_screening_s123.sh
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
