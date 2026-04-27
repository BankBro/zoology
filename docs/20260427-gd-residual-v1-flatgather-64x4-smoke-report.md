# gd_residual_v1 flat gather 64x4 gate report

日期: 2026-04-27

范围: 只跑实验和整理结果, 未修改 `Flash-VQG` 代码, 未改变 official candidate 配置. 当前配置保持 `rank=16`, `write_topk=4`, `read_topk=2`, `cb256`, `vq_weight_mode=dense_softmax`, `vq_softmax_tau=0.25`, `builder=grouped_chunk_torch_ref`, `pack=semivec_ref`.

## 结论

64x4 smoke gate 已通过. flat gather 后, official candidate 的 64x4 口径在本机 RTX 2080 Ti 11GB 上不再触发 OOM, 能完整完成 train 和 validation, loss finite, 日志中未见 NaN/Inf 错误.

64x4, T256, 8 microbatches profile 也通过. profile peak_reserved 为 8.54 GiB, peak_allocated 为 6.78 GiB, 8 个 microbatch loss 均 finite. 主要问题已经从 backward 显存转为吞吐, T256 平均 microbatch_sec 为 197.81s, backward 平均 153.58s, 是主要耗时来源.

建议可以启动 1 次正式 4 epoch 64x4 对齐 run 做质量验证, 但不建议直接扩大 sweep. 若需要高频实验或多配置对比, 下一步应优先优化 `event_pack/grouped_chunk_torch_ref` 的吞吐, 而不是继续改 official candidate 的 rank/write_topk/read_topk/cb/tau.

注意: smoke 的 `valid/accuracy=0` 只说明短 smoke 未训练出任务质量, 不能作为正式质量结论.

## 提交的数据附件

为便于 GitHub 侧复核, 本次只提交必要的精简实验数据:

- `docs/artifacts/20260427-gd-flatgather-64x4/smoke-64x4-result.md`: 64x4 smoke gate 的关键配置和最终指标.
- `docs/artifacts/20260427-gd-flatgather-64x4/profile-b64-t256-mb8-summary.json`: 64x4, T256, 8 microbatches profile 的原始 summary JSON, 包含 peak memory 和逐 microbatch timing/loss.

未提交完整 terminal log, SwanLab 本地日志, checkpoints, generated launch config, profiler trace 或 `tmp` 目录. 这些文件体积或噪声较大, 对网页 ChatGPT 分析当前结论不是必要输入.

## 实验命令

64x4 smoke gate:

```bash
cd /home/lyj/mnt/project/zoology

SWANLAB_MODE=offline \
ANALYSIS_SOURCE=off \
GPU_ID=0 \
TRAIN_BATCH_SIZE=64 \
GRADIENT_ACCUMULATION_STEPS=4 \
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
VQ_WEIGHT_MODE=dense_softmax \
VQ_SOFTMAX_TAU=0.25 \
FOX_GD_RESIDUAL_MU_MIN_COUNT=0.1 \
RUN_ID_OVERRIDE=gd-r16-wk4-mu01-t025-cb256-s123-d123-flatgather-smoke-tbs64-ga4 \
EXPERIMENT_MODE_OVERRIDE=smoke_mu01_flatgather \
LAUNCH_ID_PREFIX_SMOKE=flash-vqg-20260425-gd-residual-v1-flatgather-smoke-mu01-tbs64-ga4 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh
```

64x4, T256, 8 microbatches profile:

```bash
cd /home/lyj/mnt/project/zoology

FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
TRAIN_BATCH_SIZE=64 \
PROFILE_SEQ_LEN=256 \
PROFILE_MICROBATCHES=8 \
PROFILE_ENABLE_TORCH_PROFILER=0 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/gd_phase2_flat_gather_ab/new-b64-t256-mb8 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

## 64x4 smoke 结果

GitHub 可见摘要: `docs/artifacts/20260427-gd-flatgather-64x4/smoke-64x4-result.md`

本地完整日志: `tmp/gd_phase2_flat_gather_ab/smoke-64x4.log`, 未提交.

生成配置:

`zoology/experiments/flash_vqg/generated/flash-vqg-20260425-gd-residual-v1-flatgather-smoke-mu01-tbs64-ga4-2026-04-27-16-28-03/`

SwanLab 本地日志:

`swanlog/run-20260427_162807-fmmiapdvclq6df2a6uj41`

| 项目 | 结果 |
| --- | --- |
| 是否 OOM | 否 |
| train 是否完整完成 | 是, `Train Epoch 0/1: 6/6` |
| validation 是否完整完成 | 是, final validation `32/32` |
| train wall time | tqdm 显示 `12:37` |
| smoke 平均 step time | tqdm 显示 `126.30s/it`, 混合 seq length, 包含训练循环中的 validation 影响 |
| sampled peak memory | 手动 `nvidia-smi` 最高采样约 `7945 MiB`, 该值不是 profiler peak |
| final valid/loss | `9.04` |
| final valid/accuracy | `0` |
| final valid/attn/gd_residual_inject_ratio | `0.0107` |
| final valid/attn/gd_residual_mu_valid_ratio | `0.0927` |
| final valid/attn/gd_residual_m_norm_mean | `0.0159` |
| final valid/attn/gd_residual_m_norm_max | `0.184` |
| NaN/Inf | 日志中未见 `Traceback`, `RuntimeError`, `CUDA out of memory`, `NaN`, `Inf` 错误匹配 |

## 64x4 profile 结果

GitHub 可见 summary: `docs/artifacts/20260427-gd-flatgather-64x4/profile-b64-t256-mb8-summary.json`

本地 profile log: `tmp/gd_phase2_flat_gather_ab/profile-new-b64-t256-mb8.log`, 未提交.

| 指标 | 数值 |
| --- | ---: |
| peak_reserved_GB | `8.544921875` |
| peak_allocated_GB | `6.781586170196533` |
| avg_forward_sec | `43.74714952742215` |
| avg_backward_sec | `153.58020816236967` |
| avg_microbatch_sec | `197.80990810511867` |

逐 microbatch:

| microbatch | forward_sec | backward_sec | microbatch_sec | loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 45.9845 | 151.4236 | 197.6635 | 9.0392 |
| 1 | 42.5186 | 153.3470 | 195.9617 | 8.9500 |
| 2 | 41.4980 | 153.9501 | 196.2510 | 8.8633 |
| 3 | 42.6744 | 154.0173 | 196.7383 | 8.7749 |
| 4 | 43.9876 | 153.3299 | 198.1800 | 8.6889 |
| 5 | 44.1958 | 154.5188 | 199.5790 | 8.6037 |
| 6 | 44.6531 | 153.6932 | 199.2337 | 8.5199 |
| 7 | 44.4652 | 154.3618 | 198.8721 | 8.4378 |

Profile 中所有 loss 均为 finite. 训练 metrics 中 `attn/gd_residual_mu_valid_ratio` 为 `0.0`, 但 smoke final validation 的 `valid/attn/gd_residual_mu_valid_ratio` 为 `0.0927`; 二者数据分布和记录口径不同, 不应直接混作质量结论.

## 4 epoch 时间估算

按 profile 的固定 T256 平均 microbatch_sec 估算:

- B64 smoke 数据集每 epoch 为 6 个 train microbatch.
- 4 epoch 为 24 个 train microbatch.
- `24 * 197.81s = 4747.44s`, 约 `79.1min` train compute.
- smoke validation 每次约 `21s`, 若按 2 次 validation/epoch, 4 epoch 约 8 次 validation, 额外约 `2.8min`.
- 保守估算总 wall-clock 约 `82min`, 再加启动, checkpoint, 日志开销.

按本次 smoke 混合 seq length 的实际 1 epoch 观察值估算:

- train tqdm 到 6/6 为 `12:37`, final validation 约 `20s`.
- 线性估算 4 epoch 约 `52min`.

因此, 当前 64x4 正式 4 epoch 单 run 预计约 `52-82min`. 固定 T256 profile 是保守上界, smoke 混合长度更接近当前 smoke 配置.

## 建议

建议启动 1 次正式 4 epoch 64x4 run, 用于确认 flat gather 后 official candidate 能否恢复到 baseline 对齐口径, 并观察正式训练质量曲线. 不要把本次 smoke 的 `valid/accuracy=0` 作为质量否定依据.

不建议在吞吐优化前扩大多配置 sweep. 当前显存 gate 已经通过, profile 显示主要成本在 T256 backward, 平均 `153.58s`, 下一步应转向 `event_pack/grouped_chunk_torch_ref` 吞吐优化. 优先建议:

1. 开启更细粒度 diagnostics 或 torch profiler, 重新分解 `event_pack`, `grouped_chunk_torch_ref`, `phase2_residual_read` 的 forward/backward wall time.
2. 先优化 `event_pack/grouped_chunk_torch_ref` 的 Python/torch indexing 和分组路径, 目标是降低 T256 backward 主耗时.
3. 保留 official candidate 配置不变, 避免把吞吐改进和模型配置改动混在同一轮实验里.
4. 在优化后复跑 B64,T256,mb8 profile 和 64x4 smoke gate, 再决定是否扩大正式实验.

本轮未重跑 baseline.
