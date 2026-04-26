# gd_residual_v1 MQAR 实验

这组脚本用于在 `zoology` 侧以最小接线方式启动 `Flash-VQG + gd_residual_v1` 的 MQAR 单点实验.

固定口径:

- `d_model=128`
- `num_codebook_vectors=128`
- `block_len=32`
- `local_num_blocks=2`
- `vq_score_mode=codebook_dot`
- `vq_weight_mode=dense_softmax`
- `vq_update_mode=grad`
- `vq_topk=4`
- `seed=123`
- `data_seed=123`
- `max_epochs=32`
- `fox_remote_formula=gd_residual_v1`
- `fox_remote_read_topk=2`, 可用 `FOX_REMOTE_READ_TOPK=dense` 切回 dense read
- 其余 `fox_gd_residual_*` 默认值按蓝图:
  - `rank=16`
  - `write_topk=4`
  - `builder=grouped_chunk_torch_ref`
  - `pack_mode=semivec_ref`
  - `chunk_size=64`
  - `mu_min_count=1.0`
  - `addr_eps=1e-6`
  - `den_eps=1e-6`
  - `rho_eps=1e-12`
  - `beta_init=0.5`
  - `lambda_init=0.05`
  - `norm_with_gain=false`
  - `use_separate_addr_codebook=false`

统一约定:

- SwanLab 项目名固定为 `flash_vqg_gd_residual_v1_mqar`
- 新脚本默认 `SWANLAB_MODE=cloud`, 直接走在线上报
- 如需离线落本地, 可显式覆盖: `SWANLAB_MODE=offline bash .../run_train.sh`
- analysis 默认走本地产物, 即 `--analysis local`
- `run_smoke.sh` 用于 1 epoch 冒烟
- `run_train.sh` 用于正式训练
- `run_profile.sh` 用于短周期 profiling, 默认 `logger=none`, 不连接 SwanLab, 不保存 checkpoint
- `run_short_run.sh` 用于真实 MQAR readiness 短跑, 默认 `20` train batches x seeds `123,124`, 不连接 SwanLab, 不保存 checkpoint

启动:

```bash
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh
PROFILE_ENABLE_TORCH_PROFILER=1 bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
PROFILE_ENABLE_GD_DIAGNOSTICS=1 bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
SHORT_RUN_VARIANT=all PROFILE_ENABLE_GD_DIAGNOSTICS=1 bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_short_run.sh
```

默认 launch 前缀:

- smoke: `flash-vqg-20260425-gd-residual-v1-mqar-smoke`
- train: `flash-vqg-20260425-gd-residual-v1-mqar-train`
- profile: `flash-vqg-20260425-gd-residual-v1-mqar-profile`

主要产物位置:

- generated manifest: `zoology/experiments/flash_vqg/generated/<launch_id>/manifest.json`
- generated config: `zoology/experiments/flash_vqg/generated/<launch_id>/generated_config.py`
- local analysis: `zoology/analysis/flash_vqg/results/<launch_id>/launch_analysis/run_summary.csv`
- checkpoint: `checkpoints/<launch_id>/gd-residual-v1-*/`
- profiling summary: `tmp/20260425-gd-residual-v1-profile/summary.json`
- short-run summary: `tmp/20260425-gd-residual-v1-short-run/summary.json`
- short-run records: `tmp/20260425-gd-residual-v1-short-run/records.jsonl`
- `PROFILE_ENABLE_GD_DIAGNOSTICS=1` 会额外输出同步 wall-time phase timer, event 统计和 `L_state` 有效性诊断

Short-run 参数:

- `SHORT_RUN_TRAIN_BATCHES`: 默认 `20`
- `SHORT_RUN_SEEDS`: 默认 `123,124`
- `SHORT_RUN_VARIANT`: 默认 `gd_r16_wk4`, 可选 `gd_r16_wk4`, `gd_r16_wk2`, `gd_r8_wk4`, `local_only`, `legacy_fox`, `all`
- `SHORT_RUN_OUTPUT_DIR`: 默认 `tmp/20260425-gd-residual-v1-short-run`
- `PROFILE_ENABLE_GD_DIAGNOSTICS`: 默认 `1`, 显式控制 gd residual debug metrics
