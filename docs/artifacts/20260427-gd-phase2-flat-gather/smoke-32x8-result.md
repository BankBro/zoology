# 32x8 smoke result

日期: 2026-04-27

## 命令

```bash
SWANLAB_MODE=offline \
ANALYSIS_SOURCE=off \
GPU_ID=0 \
TRAIN_BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=8 \
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
VQ_WEIGHT_MODE=dense_softmax \
VQ_SOFTMAX_TAU=0.25 \
FOX_GD_RESIDUAL_MU_MIN_COUNT=0.1 \
RUN_ID_OVERRIDE=gd-r16-wk4-mu01-t025-cb256-s123-d123-flatgather-smoke-tbs32-ga8 \
EXPERIMENT_MODE_OVERRIDE=smoke_mu01_flatgather \
LAUNCH_ID_PREFIX_SMOKE=flash-vqg-20260425-gd-residual-v1-flatgather-smoke-mu01-tbs32-ga8 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh
```

## 结果

- 退出码: 0.
- GPU: NVIDIA GeForce RTX 2080 Ti, 11GB.
- 完成 `Train Epoch 0/1: 12/12`.
- 中途 validation 完成 `32/32`, `valid/loss=9.05`.
- 最终 validation 完成 `32/32`, `valid/loss=9.04`.
- 最终 `valid/accuracy=0`, 这是 smoke 训练时长下的预期质量状态, 不是本次显存优化的判定指标.
- 无 OOM.
- 无 NaN/Inf 输出.

## 最终 validation 关键指标

```text
valid/loss=9.04
valid/accuracy=0
valid/attn/gd_residual_write_strength_mean=0.125
valid/attn/gd_residual_m_norm_mean=0.0159
valid/attn/gd_residual_m_norm_max=0.187
valid/attn/gd_residual_mu_valid_ratio=0.0936
valid/attn/gd_residual_lambda_mean=0.05
valid/attn/gd_residual_inject_ratio=0.0107
valid/vq/relative_err_mean=0.932
valid/vq/c_entropy=5.54
valid/vq/write_entropy_mean=5.41
valid/vq/write_top1_mass_mean=0.0148
```
