# Flash-VQG MQAR Experiments

这个目录提供 Flash-VQG 在 Zoology MQAR 任务上的正式实验入口.

## 文件说明

- `run_flash_vqg_suite.py`
  主入口脚本. 负责解析命令行参数, 生成临时配置脚本, 然后调用 `python -m zoology.launch` 真正启动实验.
- `flash_vqg_suite.py`
  配置生成器. 负责定义 MQAR 的训练集 / 测试集, 以及 Flash-VQG 和 Gated DeltaNet 的 sweep 配置.
- `generated/`
  存放由 `run_flash_vqg_suite.py` 自动生成的配置脚本. 目录已通过 `.gitignore` 忽略, 不需要手动提交.

## 推荐用法

在仓库根目录下运行:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite
```

这会使用当前默认配置启动一组实验:

- `backend=accel`
- `include_gdn=True`
- `block_len=8`
- `dmodels=128`
- `learning_rates=1e-4,3e-4,1e-3,3e-3`
- `max_epochs=32`

## 常用命令

只跑 Flash-VQG, 不带 Gated DeltaNet 对照:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --flash-only
```

把 `block_len` 改成 `32`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --block-len 32
```

切到 Torch 后端:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --backend torch
```

同时扫多个 `d_model`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --dmodels 64,128,256
```

指定自定义学习率:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --learning-rates 1e-4,3e-4,1e-3
```

并行启动:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite -p
```

## 参数入口

如果你只是想改实验启动参数, 优先改命令行参数:

- `--backend`
- `--flash-only`
- `--block-len`
- `--dmodels`
- `--learning-rates`
- `--project`
- `--entity`
- `--max-epochs`
- `--name`
- `--outdir`
- `--gpus`
- `-p`

如果你要改实验矩阵本身, 比如训练 / 测试 segment, `num_heads`, `local_num_blocks`, `if_remote_enabled`, 或 Flash-VQG 的 recipe, 请改 `flash_vqg_suite.py` 里的 `build_configs()`.

## 当前默认实验设置

`flash_vqg_suite.py` 当前默认构造的是 Zoology MQAR 合成任务:

- train segments:
  - `(64, 4, 100k)`
  - `(128, 8, 20k)`
  - `(256, 16, 20k)`
  - `(256, 32, 20k)`
  - `(256, 64, 20k)`
- test segments:
  - `(64, 4)`
  - `(64, 8)`
  - `(64, 16)`
  - `(128, 32)`
  - `(256, 64)`
  - `(512, 64)`
  - `(512, 128)`
  - `(1024, 256)`

当前 Flash-VQG 默认配置包含:

- `num_heads=2`
- `if_remote_enabled=True`
- `num_codebook_vectors={64: 64, 128: 128, 256: 256}`
- `local_num_blocks=2`
- `use_time_mixing="kv_shift"`
- `vq_score_mode="l2"`
- `vq_weight_mode="one-hot"`
- `vq_update_mode="ema"`

## 说明

- 这个目录下的正式入口已经替代早期 `mqar_example_configs/` 里的临时 Flash-VQG 调试脚本.
- 如果只想看配置生成逻辑, 直接读 `flash_vqg_suite.py`.
- 如果只想发起实验, 直接跑 `run_flash_vqg_suite.py`.
