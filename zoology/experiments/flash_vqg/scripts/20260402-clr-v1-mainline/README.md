# `20260402-clr-v1-mainline`

`Flash-VQG clr_v1` 主线实验脚本目录.

约定:

- 本目录承载参考文档中的实验 1 到实验 6
- 当前实验目录按 `e1` 到 `e6` 独立子目录逐步落地
- 统一通过 `zoology.experiments.flash_vqg.run_flash_vqg_suite` 发起训练与本地 analysis
- 各实验 wrapper 统一围绕 `run_flash_vqg_suite.py --config-builder ...` 预组装矩阵
- SwanLab 项目统一使用 `flash_vqg_clr_v1_mainline`
- analysis 统一优先使用本地产物

当前文件:

- `common_env.sh`: 实验公共环境变量
- `metrics_e1_soft_topk.yaml`: 实验 1 专用 metrics white list
- `smoke_e1_batch_accum.py`: 低成本端到端 smoke, 用于选择 `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`
- `run_e1_smoke.sh`: smoke 包装脚本
- `run_e1_train.sh`: 实验 1 正式训练入口
- `e2-remote-interface/`: 实验 2 目录, 同时包含 `E2-main` 与 `E2b`
- `e5-train-dense-teacher/`: train-time `dense_value teacher` 主线, 包含 smoke, screening, repro, confirm 与 paired control 包装脚本
