# `e2-remote-interface`

实验2目录, 拆成两部分:

- `E2-main`: 去 remote denominator 的主实验
- `E2b`: 固定当前 denominator-aware selector 的 merge-only 补充实验

统一约定:

- 训练入口统一仍然是 `zoology.experiments.flash_vqg.run_flash_vqg_suite`
- 本目录通过 `--config-builder` 预组装配置矩阵
- smoke 必须先分别确定 `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`
- full train 默认用 `run_e2_dual_train.sh` 双卡并行执行
