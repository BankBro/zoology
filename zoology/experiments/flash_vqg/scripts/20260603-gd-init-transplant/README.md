# gd_residual_v1 init-transplant 探索实验

本目录用于分析 Flash-VQG `gd_residual_v1` 中什么样的 FlashVQG 初始化会带来稳定训练. 实验只在当前 2080ti `Flash-VQG-tun` 容器内运行, 只操作 `/home/lyj/mnt/project/zoology` 和 `/home/lyj/mnt/project/Flash-VQG`.

约束:

- 不 commit, 不 push.
- 不启用 `TORCH_DETERMINISTIC=1`.
- 可用 GPU 0,1.
- 结果先写独立 artifact: `/home/lyj/mnt/project/zoology/docs/artifacts/20260603-gd-init-transplant/`.
- 暂不写 canonical ledger: `/home/lyj/mnt/project/zoology/docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`.

主要脚本:

- `save_init_snapshot.py`: 保存刚初始化后的 `full_model`, `flash_only`, `non_flash_only` snapshot.
- `audit_init_geometry.py`: 审计 codebook 分散度, projection-codebook 匹配度, addr slot 分离度, head diversity, top-k candidate margin, 初始 write 分布覆盖.
- `make_transplant_checkpoint.py`: 把 donor snapshot 覆盖到 recipient 初始模型, 生成可被 `init_checkpoint_path` strict 加载的完整 checkpoint.
- `build_transplant_configs.py`: 生成 core/extended 矩阵的 generated configs 和 manifest, 可选择直接 launch.
- `run_early_probe.sh`: smoke 数据的 early probe.
- `run_train.sh`: 4 epoch no-early-stopping core 矩阵.
- `poll_launch.sh`: 进入稳定训练后显式 sleep 定时轮询.

core 矩阵:

- `normal-cb64-r16-s124`, `normal-cb64-r16-s125`.
- `fullrerun-cb64-r16-s125`: 固定 full init, 测 nondeterminism floor.
- `flashdonor-cb64-r16-s125-to-s124`: good Flash init transplant 到 seed124 背景.
- `flashdonor-cb64-r16-s124-to-s125`: boundary Flash init transplant 到 seed125 背景.
- `nonflashdonor-cb64-r16-s125-to-s124`: 非 Flash 背景 transplant 对照.
- `normal-cb256-r4-s123`, `normal-cb256-r4-s124`.
- `flashdonor-cb256-r4-s123-to-s124`: historical-good Flash init transplant 到 bad seed124 背景.

推荐执行顺序:

```bash
cd /home/lyj/mnt/project/zoology
/home/lyj/miniconda3/envs/flash-vqg/bin/python zoology/experiments/flash_vqg/scripts/20260603-gd-init-transplant/save_init_snapshot.py
/home/lyj/miniconda3/envs/flash-vqg/bin/python zoology/experiments/flash_vqg/scripts/20260603-gd-init-transplant/audit_init_geometry.py --probe
GPUS=0,1 bash zoology/experiments/flash_vqg/scripts/20260603-gd-init-transplant/run_early_probe.sh
GPUS=0,1 bash zoology/experiments/flash_vqg/scripts/20260603-gd-init-transplant/run_train.sh
```
