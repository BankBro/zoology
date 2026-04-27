# 64x4 smoke result

日期: 2026-04-27

用途: 记录 gd_residual_v1 phase2 residual read flat gather 后, official candidate 64x4 smoke gate 的关键结果. 该文件是 GitHub 可见的精简数据附件, 不包含完整 terminal log, SwanLab 本地目录, checkpoint 或 generated launch config.

## 配置

- `TRAIN_BATCH_SIZE=64`
- `GRADIENT_ACCUMULATION_STEPS=4`
- `FOX_REMOTE_READ_TOPK=2`
- `FOX_GD_RESIDUAL_RANK=16`
- `FOX_GD_RESIDUAL_WRITE_TOPK=4`
- `NUM_CODEBOOK_VECTORS=256`
- `VQ_WEIGHT_MODE=dense_softmax`
- `VQ_SOFTMAX_TAU=0.25`
- `FOX_GD_RESIDUAL_MU_MIN_COUNT=0.1`
- `builder=grouped_chunk_torch_ref`
- `pack=semivec_ref`
- GPU: NVIDIA GeForce RTX 2080 Ti 11GB

## 运行结果

| 项目 | 结果 |
| --- | --- |
| OOM | 否 |
| train 完成情况 | 完整完成, `Train Epoch 0/1: 6/6` |
| validation 完成情况 | 完整完成, final validation `32/32` |
| train wall time | tqdm 显示 `12:37` |
| smoke 平均 step time | tqdm 显示 `126.30s/it`, 混合 seq length, 包含训练循环中的 validation 影响 |
| sampled peak memory | 手动 `nvidia-smi` 最高采样约 `7945 MiB`, 非 profiler peak |
| final valid/loss | `9.04` |
| final valid/accuracy | `0` |
| final valid/attn/gd_residual_inject_ratio | `0.0107` |
| final valid/attn/gd_residual_mu_valid_ratio | `0.0927` |
| final valid/attn/gd_residual_m_norm_mean | `0.0159` |
| final valid/attn/gd_residual_m_norm_max | `0.184` |
| NaN/Inf | 完整日志中未见 `Traceback`, `RuntimeError`, `CUDA out of memory`, `NaN`, `Inf` 错误匹配 |

## 说明

- `valid/accuracy=0` 是短 smoke 结果, 不能作为正式质量结论.
- 该 smoke gate 的作用是确认 flat gather 后 64x4 official candidate 是否可运行, 不是质量评估.
- 对应 profile 数据见 `docs/artifacts/20260427-gd-flatgather-64x4/profile-b64-t256-mb8-summary.json`.
