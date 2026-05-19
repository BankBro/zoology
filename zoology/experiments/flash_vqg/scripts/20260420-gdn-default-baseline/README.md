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
- `validations_per_epoch=1`
- `max_epochs=32`
- `seed=123`
- `data_seed=123`

可覆盖项:

- `DISABLE_EARLY_STOPPING`: 默认 `false`, 设为 `true` 时将 early stopping metric/threshold 置空.
- `RUN_ID_OVERRIDE`: 默认空, 设定后覆盖默认 `run_id`, 用于 batch/accum 诊断等不应覆盖旧 baseline 名称的 run.
- `GDN_KERNEL_DTYPE`: 默认 `auto`, 控制进入 FLA gated-delta kernel 的 `q/k/v/beta/g` dtype. 可选 `auto|input|float32|float16|bfloat16`. `auto` 在 CUDA sm80+ 使用 bf16, sm80 以下使用 fp16, CPU 使用 fp32; `input` 表示保持 GDN 内部当前输入 dtype, 不额外 cast.

统一约定:

- logger 使用 `SwanLab`
- analysis 固定读本地产物, 即 `--analysis local`
- 正式入口是 `run_train.sh`
- 后续完整 MQAR official 实验的 dtype 口径按项目规则执行: RTX 2080 Ti/sm75 默认优先 float32, 支持 bf16 的 GPU 默认优先 bfloat16. 当前 GDN 代码的 `GDN_KERNEL_DTYPE=auto` 历史行为在 sm75 上会走 fp16, 因此 2080 Ti 上的 official GDN 可比实验需要显式设置 `GDN_KERNEL_DTYPE=float32`, 除非后续运行时默认已改为 float32.
- 只有相同 dtype policy 和实际 kernel dtype 的结果可以作为 direct quality comparison. `auto`, `float32`, `float16`, `bfloat16`, `input` 等不同 dtype 口径只能作为 dtype probe / hardware profile / ablation 解释, 不要混入同一 official baseline 或 hparam 表作为同口径结果.
- 非 official dtype 或 dtype 诊断实验必须写入 artifact/report/summary, 至少记录 `GDN_KERNEL_DTYPE`, outer model dtype, FLA kernel input dtype, GPU 型号和 compute capability.
- GDN canonical ledger: `docs/artifacts/gdn/gdn-hparam-effect-summary.csv`. 当前 2080 Ti float32 baseline 是 `gated_delta_net-default-s123-d123-noearly4ep-b64-ga4-gdnkernel-fp32`.

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
