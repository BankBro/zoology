# Flash-VQG MQAR Experiments

这个目录提供 Flash-VQG 在 Zoology MQAR 任务上的正式实验入口.

## 文件说明

- `run_flash_vqg_suite.py`
  主入口脚本. 负责解析命令行参数, 生成临时配置脚本, 然后调用 `python -m zoology.launch` 真正启动实验.
- `flash_vqg_suite.py`
  配置生成器. 负责定义 MQAR 的训练集 / 测试集, 以及 Flash-VQG 和 Gated DeltaNet 的 sweep 配置.
- `generated/`
  存放由 `run_flash_vqg_suite.py` 自动生成的 launch 目录. 每个 `launch_id` 下至少包含 `launch_configs.py` 和 `manifest.json`. 目录已通过 `.gitignore` 忽略, 不需要手动提交.

## 推荐用法

在仓库根目录下运行:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite
```

这会使用当前默认配置启动一组实验:

- `backend=accel`
- `logger_backend=wandb`
- `include_gdn=True`
- `block_len=8`
- `dmodels=128`
- `learning_rates=1e-4,3e-4,1e-3,3e-3`
- `if_remote_enabled=true`
- `local_num_blocks=2`
- `train_batch_order=sequential`
- `cache_dir=./data/flash_vqg`
- `max_epochs=32`
- `launch_id_prefix=flash-vqg-suite`

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

训练日志切到 SwanLab:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --logger-backend swanlab
```

训练成功结束后自动执行 analysis:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --logger-backend swanlab \
  --analysis remote
```

如果希望自动 analysis 直接走本地 SwanLab 日志:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --logger-backend swanlab \
  --analysis local
```

禁用训练日志写出:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --logger-backend none
```

同时扫多个 `d_model`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --dmodels 64,128,256
```

指定自定义学习率:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --learning-rates 1e-4,3e-4,1e-3
```

同时扫描 `if_remote_enabled`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --if-remote-enabled true,false
```

同时扫描 `local_num_blocks`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --local-num-blocks 1,2
```

指定数据缓存目录:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --cache-dir /path/to/data_cache
```

E0 sampler 对照建议固定 `block_len=32`, 只切 `train_batch_order`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --block-len 32 --train-batch-order sequential
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --block-len 32 --train-batch-order global_shuffle
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --block-len 32 --train-batch-order balanced_interleave
```

如果你希望在同一个 `launch_id` 里一次跑完 3 个 sampler, 可以直接传逗号列表:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --block-len 32 \
  --train-batch-order sequential,global_shuffle,balanced_interleave \
  --launch-id-prefix flash-vqg-e0-all
```

这会生成同一个 `launch_id` 下的 3 个 run, 它们共享:

- `sweep_id = flash-vqg-e0-all`
- 同一个 timestamped `launch_id`

但 `run_id` 会继续按 sampler 和结构标签区分, 例如 `...-local2-remote1-sampler-gshuffle`.

并行启动:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite -p
```

指定实验组名前缀:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --launch-id-prefix flash-vqg-e0-seq
```

命名规则如下:

- `sweep_id = launch_id_prefix`
- `launch_id = launch_id_prefix-<timestamp>`
- generated 配置入口 = `generated/<launch_id>/launch_configs.py`
- generated manifest = `generated/<launch_id>/manifest.json`

## 参数入口

如果你只是想改实验启动参数, 优先改命令行参数:

- `--backend`
- `--logger-backend`
- `--flash-only`
- `--block-len`
- `--dmodels`
- `--learning-rates`
- `--if-remote-enabled`
- `--local-num-blocks`
- `--train-batch-order`
- `--cache-dir`
- `--project`
- `--entity`
- `--max-epochs`
- `--launch-id-prefix`
- `--outdir`
- `--gpus`
- `-p`
- `--analysis`

如果你要改实验矩阵本身, 比如训练 / 测试 segment, `num_heads`, `local_num_blocks`, `if_remote_enabled`, 或 Flash-VQG 的 recipe, 请改 `flash_vqg_suite.py` 里的 `build_configs()`.

E0 的 sampler 对照优先通过 `--train-batch-order` 切换, 不需要手改 `flash_vqg_suite.py`.
`--train-batch-order` 支持单值和逗号分隔多值.

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
- `enable_layer_metrics=True`
- `use_time_mixing="kv_shift"`
- `vq_score_mode="l2"`
- `vq_weight_mode="one-hot"`
- `vq_update_mode="ema"`

## 说明

- 这个目录下的正式入口已经替代早期 `mqar_example_configs/` 里的临时 Flash-VQG 调试脚本.
- 如果只想看配置生成逻辑, 直接读 `flash_vqg_suite.py`.
- 如果只想发起实验, 直接跑 `run_flash_vqg_suite.py`.
- 当前仅训练日志支持 `wandb`, `swanlab`, `none` 切换.
- `zoology.analysis.flash_vqg` 现在只支持 SwanLab 数据源.
- `manifest.json` 是后续 analysis 的唯一 run 发现入口.
- 默认不会自动 analysis. 只有显式传入 `--analysis remote` 或 `--analysis local` 时, wrapper 才会在训练成功完成后继续执行 analysis.
