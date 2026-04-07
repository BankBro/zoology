# `e2-remote-interface`

实验2目录, 拆成两部分:

- `E2-main`: 去 remote denominator 的主实验
- `E2b`: 固定当前 denominator-aware selector 的 merge-only 补充实验

统一约定:

- 训练入口统一仍然是 `zoology.experiments.flash_vqg.run_flash_vqg_suite`
- 本目录通过 `--config-builder` 预组装配置矩阵
- smoke 必须先分别确定 `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`
- full train 默认用 `run_e2_dual_train.sh` 双卡并行执行

当前为了收尾实验2, 额外补充 2 个单 run probe 入口:

- `run_e2b_baseline_probe.sh`: 补 canonical E2b baseline 的 matched-seed 对照
- `run_e2b_2a_l100_probe.sh`: 补 `E2b-2A(l100)` 的单 seed probe 或 rerun

probe 约定:

- 只接受单个 `SEED_VALUES`
- 要求 `SEED_VALUES == DATA_SEED`
- batch / eval / GA 参数默认仍从 `e2b_smoke.env` 继承
- 这轮推荐的 `launch_id_prefix`:
  - `flash-vqg-20260402-clr-v1-e2b-baseline-s124`
  - `flash-vqg-20260402-clr-v1-e2b-baseline-s125`
  - `flash-vqg-20260402-clr-v1-e2b-2a-l100-s123-rerun`

最小执行顺序:

1. GPU0 跑 `baseline-s124`
2. GPU1 跑 `baseline-s125`
3. 任一空闲卡补 `e2b-2a-l100-s123-rerun`

示例:

```bash
GPU_ID=0 SEED_VALUES=124 DATA_SEED=124 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-s124 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe.sh

GPU_ID=1 SEED_VALUES=125 DATA_SEED=125 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-s125 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe.sh

GPU_ID=0 SEED_VALUES=123 DATA_SEED=123 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-2a-l100-s123-rerun \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_2a_l100_probe.sh
```
