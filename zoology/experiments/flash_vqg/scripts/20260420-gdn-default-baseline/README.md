# GatedDeltaNet 默认对齐基线

这组脚本用于启动一个 `GDN-only` 单点基线, 目标是和当前 Flash-VQG 主线 baseline 做可比对照.

固定口径:

- 模型: `BaseConv + GatedDeltaNet`
- `state_mixer=Identity`
- `vocab_size=8192`
- `d_model=128`
- `n_layers=2`
- `num_heads=2`
- `use_gate=false`
- `use_short_conv=true`
- `conv_size=4`
- `learning_rate=1e-3`
- `train_batch_order=global_shuffle`
- `train_batch_size=64`
- `eval_batch_size=16`
- `gradient_accumulation_steps=4`
- `max_epochs=32`
- `seed=123`
- `data_seed=123`

统一约定:

- logger 使用 `SwanLab`
- analysis 固定读本地产物, 即 `--analysis local`
- 正式入口是 `run_train.sh`

默认 run_id:

- `gated_delta_net-default-s123-d123`

启动:

```bash
bash zoology/experiments/flash_vqg/scripts/20260420-gdn-default-baseline/run_train.sh
```

主要结果位置:

- generated manifest: `zoology/experiments/flash_vqg/generated/<launch_id>/manifest.json`
- checkpoint: `checkpoints/<launch_id>/gated_delta_net-default-s123-d123/`
- local analysis: `zoology/analysis/flash_vqg/results/<launch_id>/launch_analysis/run_summary.csv`

对照对象:

- Flash-VQG baseline run: `dense-t025-s123-d123`
- baseline reference launch: `flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12`
