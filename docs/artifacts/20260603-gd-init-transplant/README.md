# Flash-VQG gd_residual_v1 init-transplant 探索 artifact

本目录保存 `20260603-gd-init-transplant` 的探索性结果, 用于分析 Flash-VQG `gd_residual_v1` 中什么样的 FlashVQG 初始化能带来稳定训练. 本实验只在 2080ti 的 `Flash-VQG-tun` 容器内执行, 只操作 `/home/lyj/mnt/project/zoology` 和 `/home/lyj/mnt/project/Flash-VQG`.

本目录不是 canonical ledger. `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv` 未更新.

## 主要文件

- `metadata.json`: 实验范围, 约束, launch id, raw analysis 路径和结论摘要.
- `init-transplant-source-manifest.csv`: 初始化快照来源清单, 包括 `full_model`, `flash_only`, `non_flash_only`.
- `init-path-notes.json`: FlashVQG 初始化相关代码路径记录.
- `init-geometry-audit.csv`: 初始化几何审计, 包括 codebook 分散度, projection-codebook 匹配度, addr slot 分离度, head diversity, beta/lambda 初始值.
- `init-geometry-probe.csv`: 初始 probe, 包括 top-k candidate margin, 初始 write 分布覆盖, VQ relative error 和 write entropy.
- `early-core-final.csv`: 1 epoch smoke/early probe 矩阵结果.
- `train-core-final.csv`: 4 epoch core train 矩阵结果.
- `early-core-matrix.csv`, `train-core-matrix.csv`: 实验矩阵规格.
- `init-transplant-source-manifest.csv`: 记录训练前初始化快照索引, 实际 `.pt` 文件保存在 `/home/lyj/mnt/project/zoology/checkpoints/20260603-gd-init-transplant/init_snapshots/`.
- `init_checkpoints/*/metadata.json`: transplant checkpoint 的轻量 metadata; 实际 `.pt` 文件保存在 `/home/lyj/mnt/project/zoology/checkpoints/20260603-gd-init-transplant/init_checkpoints/`.

## 运行与分析路径

- 实验脚本: `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260603-gd-init-transplant/`.
- early generated configs: `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated/flash-vqg-20260603-gd-init-transplant-early-core-2026-06-03-08-56-37/`.
- train generated configs: `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated/flash-vqg-20260603-gd-init-transplant-train-core-2026-06-03-09-07-13/`.
- early raw analysis: `/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/flash-vqg-20260603-gd-init-transplant-early-core-2026-06-03-08-56-37/`.
- train raw analysis: `/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/flash-vqg-20260603-gd-init-transplant-train-core-2026-06-03-09-07-13/`.
- 人读报告: `/home/lyj/mnt/project/zoology/docs/20260603-gd-init-transplant-report.md`.
- large artifact root: `/home/lyj/mnt/project/zoology/checkpoints/20260603-gd-init-transplant/`.

## 核心结论

本矩阵不支持“good FlashVQG flash-only 初始化单独决定稳定训练”的解释. `full_model` 同 seed rerun 与 normal good 很接近, 说明训练非确定性底噪很小; 但 good flash-only donor 移植到 bad/boundary recipient 后不能把结果稳定救回 good basin. 更合理的解释是: FlashVQG 初始化几何可能与稳定 basin 相关, 但 recipient-side 非 Flash 初始化状态和后续训练动力学同样关键.
