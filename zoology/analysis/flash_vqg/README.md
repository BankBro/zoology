# Flash-VQG Analysis

这个目录提供 Flash-VQG 在 Zoology MQAR 实验上的正式分析入口.

## 文件说明

- `run_flash_vqg_analysis.py`
  主入口脚本. 负责解析命令行参数, 然后调用分析函数执行汇总和画图.
- `flash_vqg_analysis_suite.py`
  分析实现. 负责从 WandB 拉取 run, 过滤目标模型, 选择最佳 learning rate, 并导出表格和图.
- `results/`
  默认输出目录. 如果不显式传 `--output-dir`, 分析结果会写到这个目录下的时间戳子目录.

## 支持的分析模式

### `d128`

聚焦 `d_model=128` 的 Flash-VQG vs Gated DeltaNet 对比.

默认会导出:

- `full_runs.csv`
- `best_runs.csv`
- `best_valid_accuracy.png`
- `best_num_kv_slices.png`

### `dmodel`

对 `d_model in {64, 128, 256}` 做跨模型规模对比.

这个模式会优先按 `selection_metric` 为每个模型和 `d_model` 选择最佳 run, 然后围绕指定的 `target_case` 画图和汇总.

## 推荐用法

在仓库根目录下运行.

按 `launch_id` 做 `d128` 分析:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis \
  --mode d128 \
  --launch-id <launch_id>
```

按 `sweep_id` 做 `d128` 分析:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis \
  --mode d128 \
  --sweep-id <sweep_id>
```

做跨 `d_model` 分析:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis \
  --mode dmodel \
  --sweep-id <sweep_id>
```

指定目标 case:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis \
  --mode dmodel \
  --sweep-id <sweep_id> \
  --target-case 512x64
```

指定自定义输出目录:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis \
  --mode d128 \
  --launch-id <launch_id> \
  --output-dir /path/to/output
```

## 常用参数

- `--mode {d128,dmodel}`
- `--project`
- `--entity`
- `--launch-id`
- `--sweep-id`
- `--output-dir`
- `--selection-metric`
- `--metric`
- `--target-case`
- `--expected-runs`

注意:

- `--launch-id` 和 `--sweep-id` 至少要传一个.
- `--metric` 是 `d128` 模式的兼容别名, 如果传了, 会覆盖 `--selection-metric`.
- 默认项目是 `scu-mclab/flash_vqg_vs_gdn`.

## 输出行为

如果不传 `--output-dir`, 脚本会自动在 `results/` 下创建一个带时间戳的目录, 目录名类似:

```text
results/20260325-123456-d128
results/20260325-123456-dmodel
```

输出内容通常包括:

- 原始 run 汇总 CSV
- 最优 run 汇总 CSV
- accuracy 对比图
- 按 `num_kv_pairs` 或目标 case 生成的切片图

## 说明

- 这个目录只负责分析已有 WandB run, 不负责启动训练.
- 训练入口在 `zoology/experiments/flash_vqg/`.
- 如果要改分析逻辑本身, 直接改 `flash_vqg_analysis_suite.py`.
