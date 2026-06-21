# Flash-VQG gd_residual_v1 init-transplant 探索报告

日期: 2026-06-05.

范围: 本实验只在 2080ti 的 `Flash-VQG-tun` 容器内执行, 只操作 `/home/lyj/mnt/project/zoology` 和 `/home/lyj/mnt/project/Flash-VQG`. 使用容器内两张 RTX 2080 Ti. 未 commit, 未 push, 未启用 `TORCH_DETERMINISTIC=1`. 结果按探索性 artifact 保存, 没有补到 canonical ledger `/home/lyj/mnt/project/zoology/docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`.

## 目标

问题是 Flash-VQG `gd_residual_v1` 中“什么样的 FlashVQG 初始化能带来稳定训练”. 实验拆成三个判断:

1. 如果同一个完整初始化状态重复训练, 分叉是否仍显著.
2. 如果只把 good run 的 FlashVQG 初始化移植到 bad/boundary recipient, 是否能把结果救回 good basin.
3. 初始化几何审计指标能否在 step 0 区分 good/bad 初始化.

## 初始化相关路径

初始化路径已落到 `docs/artifacts/20260603-gd-init-transplant/init-path-notes.json`:

- `/home/lyj/mnt/project/zoology/zoology/mixers/flash_vqg.py`: zoology 的 `FlashVQGMixer` wrapper, 构造 `FlashVQGConfig`, 传递 codebook, addr, beta/lambda 和 `gd_residual_v1` 参数.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py`: `FlashVQGAttention` 的 `qkvg_proj`, `res_proj`, quantizer, `fox_gd_residual_addr_proj`, beta/lambda projection.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/vq.py`: `LearnableVQ`/`RoutingVQ` 创建 codebook.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/vq_init.py`: codebook RNG 策略, 包括 `global`, `local_burn`, `local_noburn` 和 scale/bootstrap.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/model.py`: model custom init, 包括 `gd_residual_beta/lambda` 和常规 `Linear`/`Embedding`.
- `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py`: `gd_residual_v1` MQAR config builder.
- `/home/lyj/mnt/project/zoology/zoology/train.py`: 训练入口, `set_determinism(seed, deterministic=TORCH_DETERMINISTIC==1)`, 模型构造和 `init_checkpoint_path` 加载.

## 实验脚本与产物

脚本目录为 `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260603-gd-init-transplant/`. 主要脚本:

- `save_init_snapshot.py`: 保存训练前初始化快照.
- `make_transplant_checkpoint.py`: 生成 `full_model`, `flash_only`, `non_flash_only` transplant checkpoint.
- `audit_init_geometry.py`: 审计初始化几何和初始 probe.
- `build_transplant_configs.py`: 生成 early/train core 矩阵 config.
- `local_parallel_launch.py`: 本机两卡并行启动, 支持 `--only-indices` 断点续跑剩余 run, 并拒绝 `TORCH_DETERMINISTIC=1`.
- `collect_results.py`: 汇总 final CSV.

探索性 artifact 目录为 `/home/lyj/mnt/project/zoology/docs/artifacts/20260603-gd-init-transplant/`, 只保存 CSV/JSON/README 等轻量索引和结果. 初始化 `.pt` 快照和 transplant checkpoint 保存在 `/home/lyj/mnt/project/zoology/checkpoints/20260603-gd-init-transplant/`.

## 最小矩阵

初始化快照覆盖 `cb64-r16-s124`, `cb64-r16-s125`, `cb256-r4-s123`, `cb256-r4-s124`, `cb256-r4-s125`, 每个 target 保存 `full_model`, `flash_only`, `non_flash_only`. core train 矩阵实际跑 9 个 4 epoch run:

- `normal-cb64-r16-s124`: cb64 boundary 对照.
- `normal-cb64-r16-s125`: cb64 good 对照.
- `fullrerun-cb64-r16-s125`: good full-model 初始化 rerun.
- `flashdonor-cb64-r16-s125-to-s124`: good flash-only donor -> boundary recipient.
- `flashdonor-cb64-r16-s124-to-s125`: boundary flash-only donor -> good recipient.
- `nonflashdonor-cb64-r16-s125-to-s124`: good non-flash donor -> boundary recipient.
- `normal-cb256-r4-s123`: cb256 historical-good 对照.
- `normal-cb256-r4-s124`: cb256 bad 对照.
- `flashdonor-cb256-r4-s123-to-s124`: good flash-only donor -> bad recipient.

early core 矩阵同样 9 个 run, `max_epochs=1`, 用于 early probe. 两个矩阵均完成, failed run 为 0.

## 训练结果

核心 CSV: `/home/lyj/mnt/project/zoology/docs/artifacts/20260603-gd-init-transplant/train-core-final.csv`.

| run | valid/loss | valid/accuracy | 1024x256 accuracy |
|---|---:|---:|---:|
| normal-cb64-r16-s124 | 0.089076 | 0.991254 | 0.952305 |
| normal-cb64-r16-s125 | 0.055727 | 0.996036 | 0.981039 |
| fullrerun-cb64-r16-s125 | 0.058445 | 0.995200 | 0.974852 |
| flashdonor-cb64-r16-s125-to-s124 | 0.203710 | 0.971069 | 0.836082 |
| flashdonor-cb64-r16-s124-to-s125 | 0.071784 | 0.994267 | 0.969512 |
| nonflashdonor-cb64-r16-s125-to-s124 | 0.333867 | 0.944754 | 0.661695 |
| normal-cb256-r4-s123 | 0.093231 | 0.986751 | 0.941930 |
| normal-cb256-r4-s124 | 0.294885 | 0.948059 | 0.747195 |
| flashdonor-cb256-r4-s123-to-s124 | 0.337730 | 0.944706 | 0.679957 |

关键对比:

- `fullrerun-cb64-r16-s125` 相比 `normal-cb64-r16-s125`, `valid/accuracy` 只低 `0.000836`, loss 只高 `0.002718`, 1024x256 accuracy 低 `0.006187`. 这说明不启用 `TORCH_DETERMINISTIC=1` 时, 这个设置下的训练非确定性底噪很小.
- `flashdonor-cb64-r16-s125-to-s124` 相比 `normal-cb64-r16-s124`, `valid/accuracy` 低 `0.020185`, loss 高 `0.114634`, 1024x256 accuracy 低 `0.116223`. good flash-only 初始化不能把 boundary recipient 救回 good basin, 甚至比 boundary normal 更差.
- `flashdonor-cb64-r16-s124-to-s125` 保持在较好区间, 但相比 `normal-cb64-r16-s125` 仍低 `0.001769` accuracy, loss 高 `0.016057`. good recipient 即使用 boundary flash-only donor, 也没有明显崩掉.
- `nonflashdonor-cb64-r16-s125-to-s124` 最差, 相比 boundary normal 的 1024x256 accuracy 低 `0.290609`. 这说明非 Flash 初始化状态也强烈影响最终训练.
- `flashdonor-cb256-r4-s123-to-s124` 没有救回 cb256 bad recipient, 相比 `normal-cb256-r4-s124` 还低 `0.003353` accuracy, loss 高 `0.042845`, 1024x256 accuracy 低 `0.067238`.

## 初始化几何审计

核心 CSV: `/home/lyj/mnt/project/zoology/docs/artifacts/20260603-gd-init-transplant/init-geometry-audit.csv`.

layer 1 的 full-model 初始化摘要:

| target | nearest cos | nearest L2 | proj-code K cos | proj-code Q cos | addr condition | addr nearest L2 | head code cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| cb64-r16-s124 | 0.287759 | 3.373456 | 0.579792 | 0.590897 | 2.676849 | 1.231312 | 0.007676 |
| cb64-r16-s125 | 0.300949 | 3.341366 | 0.576022 | 0.584489 | 2.484449 | 1.241580 | 0.010527 |
| cb256-r4-s123 | 0.342796 | 3.240982 | 0.731559 | 0.725351 | 1.375387 | 0.225765 | -0.010073 |
| cb256-r4-s124 | 0.351677 | 3.218761 | 0.734280 | 0.733411 | 1.454997 | 0.232355 | 0.000557 |

这些指标显示 capacity 设置之间差异明显: cb256 的 codebook 更密, `proj-code` covariance cosine 更高, addr coordinate 最近邻距离更小. 但同一 capacity 内 good/bad 或 boundary 的差异很小, 没有一个简单 scalar 在初始化时把最终 good/bad 清楚分开. head diversity 相关 off-diagonal cosine 也接近 0, 不是明显区分项.

## early probe

核心 CSV: `/home/lyj/mnt/project/zoology/docs/artifacts/20260603-gd-init-transplant/init-geometry-probe.csv` 和 `/home/lyj/mnt/project/zoology/docs/artifacts/20260603-gd-init-transplant/early-core-final.csv`.

layer 1 初始 probe 摘要:

| target | top1-top2 margin | top1 weight | write entropy/head | active ratio | VQ relative err | write entropy |
|---|---:|---:|---:|---:|---:|---:|
| cb64-r16-s124 | 0.045172 | 0.044983 | 4.158710 | 1.000000 | 0.949958 | 4.036716 |
| cb64-r16-s125 | 0.045199 | 0.044849 | 4.158720 | 1.000000 | 0.950906 | 4.036789 |
| cb256-r4-s123 | 0.037474 | 0.013982 | 5.544973 | 1.000000 | 0.940548 | 5.421594 |
| cb256-r4-s124 | 0.037390 | 0.014013 | 5.545020 | 1.000000 | 0.940929 | 5.420947 |

early 训练结果也没有形成可用分界: cb64 good 和 full-model rerun 的 early metrics 基本完全一致, donor rows 在 early 阶段也很接近. 因此, step 0 或 1 epoch 的单点 probe 不能可靠预测最终 4 epoch 的稳定性.

## 结论

本实验最强的证据是: FlashVQG 初始化几何不是稳定训练的充分条件. 同一个 good full initialized state 的 rerun 几乎不分叉, 说明训练非确定性不是主要解释; 但是只移植 good flash-only 初始化到 bad/boundary recipient 并不能稳定救回训练. 相反, recipient-side 非 Flash 初始化状态和后续训练动力学会强烈影响最终 basin.

更准确的表述是: FlashVQG 初始化几何可能与稳定 basin 相关, 但它不是一个独立可移植的充分干预. 稳定训练需要 FlashVQG 初始化, 非 Flash 初始化和早期训练轨迹之间匹配.

## 后续建议

如果继续做第二轮, 优先扩展三个方向:

1. 增加多 seed 的同类 transplant, 用统计而不是单对 good/bad 判断.
2. 对前几百 step 记录 FlashVQG codebook/projection/addr/write 分布的动态变化, 而不是只看 step 0 scalar.
3. 增加 full-model donor-to-recipient 和分组模块 transplant, 细分非 Flash 初始化中 embedding, MLP, LayerNorm, output head 对稳定性的贡献.
