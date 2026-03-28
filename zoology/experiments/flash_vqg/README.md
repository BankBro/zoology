# Flash-VQG MQAR Experiments

这个目录提供 Flash-VQG 在 Zoology MQAR 任务上的正式实验入口.

## 文件说明

- `run_flash_vqg_suite.py`
  主入口脚本. 负责解析命令行参数, 生成临时配置脚本, 然后调用 `python -m zoology.launch` 真正启动实验.
- `flash_vqg_suite.py`
  配置生成器. 负责定义 MQAR 的训练集 / 测试集, 以及 Flash-VQG 和 Gated DeltaNet 的 sweep 配置.
- `generated/`
  存放由 `run_flash_vqg_suite.py` 自动生成的 launch 目录. 每个 `launch_id` 下至少包含 `launch_configs.py` 和 `manifest.json`. 目录已通过 `.gitignore` 忽略, 不需要手动提交.
  manifest 现在还会在每个 run 的 `local` 节点下记录 checkpoint 路径, 包括 `best_checkpoint` 和 `train_config_json`.
- `metrics_white_lists/`
  存放文档实验默认使用的 metrics 白名单模板. `scripts/` 下的 E0-E5 命令默认会引用这里对应的 `e0.yaml` 到 `e5.yaml`.

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

同时扫描多个 `block_len`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite --block-len 8,16,32,64
```

按 `(block_len, local_num_blocks)` 做一一配对扫描:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --paired-block-local 8:8,16:4,32:2,64:1
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

按白名单只记录和分析指定指标:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --logger-backend swanlab \
  --analysis remote \
  --metrics-white-list-file zoology/experiments/flash_vqg/metrics_white_lists/e1.yaml
```

在模板基础上临时追加几个指标:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --logger-backend swanlab \
  --analysis remote \
  --metrics-white-list-file zoology/experiments/flash_vqg/metrics_white_lists/e1.yaml \
  --metrics-white-list valid/vq/relative_err_mean,vq/relative_err_mean
```

E4-A eval-only, 直接加载历史 checkpoint 做固定测试:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --eval-only e4a \
  --logger-backend swanlab \
  --analysis remote \
  --checkpoint-launch-id <launch_id> \
  --checkpoint-run-id <run_id> \
  --gpus 0
```

这个模式也会像 E0-E3 一样生成新的 eval `launch_id` / `sweep_id` / `run_id`.
其中 eval run 的命名规则是 `eval_<eval_task>_<checkpoint_run_id>`.

E7 eval-only, 使用同一个 checkpoint 对比 `dense`, `top2`, `top4` 三种 remote read:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
 --eval-only e7 \
  --logger-backend swanlab \
  --analysis local \
  --checkpoint-launch-id <launch_id> \
  --checkpoint-run-id <run_id> \
  --metrics-white-list-file zoology/experiments/flash_vqg/metrics_white_lists/e7.yaml \
  --gpus 0
```

这个模式会生成一个新的 eval `launch_id`, 但在 launch 下拆成 3 个独立 run:

- `eval_e7_dense_<checkpoint_run_id>`
- `eval_e7_top2_<checkpoint_run_id>`
- `eval_e7_top4_<checkpoint_run_id>`

同时会把跨 mode 的聚合结果直接写到 `launch_analysis/`, 命名风格与通用 analysis 保持一致:

- `launch_analysis/metrics.csv`
- `launch_analysis/summary.json`
- `launch_analysis/valid__accuracy.png`
- `launch_analysis/valid__loss.png`
- 若白名单里包含对应指标, 还会额外生成对应 telemetry 图, 例如 `launch_analysis/valid__attn__o_remote_energy_ratio.png`

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

按列表扫描 codebook size:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --dmodels 128 \
  --num-codebook-vectors 64,128,256,512
```

按 `d_model -> codebook size` 指定固定映射:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --dmodels 128,256 \
  --num-codebook-vectors-map 128:128,256:256
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

E5 codebook size sweep 建议固定 `full_K` 基线, 只扫 `num_codebook_vectors`:

```bash
python -m zoology.experiments.flash_vqg.run_flash_vqg_suite \
  --flash-only \
  --logger-backend swanlab \
  --analysis remote \
  --block-len 32 \
  --dmodels 128 \
  --learning-rates 1e-3 \
  --local-num-blocks 2 \
  --if-remote-enabled true \
  --train-batch-order global_shuffle \
  --num-codebook-vectors 64,128,256,512 \
  --metrics-white-list-file zoology/experiments/flash_vqg/metrics_white_lists/e5.yaml \
  --launch-id-prefix flash-vqg-e5
```

或者直接使用脚本:

```bash
bash zoology/experiments/flash_vqg/scripts/run_flash_vqg_e5.sh
```

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
- `--paired-block-local`
- `--eval-only`
- `--checkpoint-launch-id`
- `--checkpoint-run-id`
- `--dmodels`
- `--learning-rates`
- `--num-codebook-vectors`
- `--num-codebook-vectors-map`
- `--if-remote-enabled`
- `--local-num-blocks`
- `--train-batch-order`
- `--cache-dir`
- `--metrics-white-list`
- `--metrics-white-list-file`
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
`--block-len` 也支持单值和逗号分隔多值.
`--paired-block-local` 用于配对扫描 `(block_len, local_num_blocks)`, 不走笛卡尔积.
`--num-codebook-vectors` 支持逗号分隔的正整数列表, 会对当前 `--dmodels` 做笛卡尔积 sweep.
`--num-codebook-vectors-map` 只接受 `d_model:num_codebook_vectors` 格式, 用于给不同 `d_model` 指定固定 codebook size.
`--num-codebook-vectors` 和 `--num-codebook-vectors-map` 不能同时使用.
`--num-codebook-vectors-map` 可以是 `--dmodels` 的超集, 但不能缺少本次实际要跑的 `d_model`.
`--eval-only` 用于跳过训练, 从 manifest 记录的 `best_checkpoint` 直接执行指定 eval task. 当前支持 `e4a` 和 `e7`, 并可像普通 launch 一样写 SwanLab 和自动 analysis.
`--metrics-white-list-file` 支持 JSON/YAML 文件. 顶层可以是 list, 也可以是 `{metrics_white_list: [...]}`.
`--metrics-white-list` 支持逗号分隔的模式串. 支持 `*` 通配.
如果同时传入文件和命令行白名单, 会先读文件, 再把命令行项追加进去并去重.
如果不传任何白名单, 行为保持为当前 legacy full, 也就是继续记录和分析全部默认指标.
Flash-VQG 的 `run_id` 现在会带上 `cb{size}` 标签, 例如 `...-dmodel128-cb256-lr1.0e-03-...`, 方便区分 codebook size sweep.

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
- manifest 的 `local` 节点会记录本地 logger 路径和 checkpoint 路径, 供后续 eval-only / analysis 复用.
- 默认不会自动 analysis. 只有显式传入 `--analysis remote` 或 `--analysis local` 时, wrapper 才会在训练成功完成后继续执行 analysis.
- `metrics_white_list` 是新实验控制日志和 analysis 范围的唯一事实来源. 新结果目录里只会出现白名单命中的指标.
- `scripts/run_flash_vqg_e5.sh` 默认按 `full_K` 基线固定其余条件, 只扫 `num_codebook_vectors=64,128,256,512`.
