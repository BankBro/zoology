# gd_residual_v1 4 Epoch MQAR 候选实验报告

日期: 2026-04-26

## 1. 结论

本轮完成了 `gd_residual_v1` 4 epoch 候选实验所需的 zoology 接口改动, baseline 读取, MQAR cache 检查, smoke gate 和本地 analysis. 正式 4 epoch 主训练没有启动.

原因是实测 gate 结果不满足可执行性要求:

- 原计划对齐 baseline 的 `train_batch_size=64`, `gradient_accumulation_steps=4` 在 backward OOM, 需要分配 32.00 GiB.
- 降到 `32x8` 仍 OOM, 需要分配 16.00 GiB.
- `16x16` 在两张 2080 Ti 上均稳定完成 smoke, 但 24 个 micro-batch 约 9.7 分钟. 按 baseline 每 epoch 约 704 个 optimizer step 推算, `16x16` 每 epoch 约 11264 个 micro-batch, 约 76 小时/epoch/run, 4 epoch 约 304 小时/run. 两个 run 并行也需要约 12.7 天 wall-clock.

因此本轮停止在 smoke gate, 不生成 gd 4 epoch 曲线. 当前候选更适合先转到 `gd_residual_v1` reference 路径的效率/显存 debug, 或降低 write/event 规模后再重新进入 4 epoch 筛选.

## 2. 代码改动

zoology:

- `TrainConfig` 增加 `validations_per_epoch: int = 1`.
- `Trainer` 支持每 epoch 多次 validation, 本轮设置 `2`, 即每 0.5 epoch 额外 valid 一次.
- checkpoint 和 early stopping 仍只使用 epoch 末 validation, 不改变旧 checkpoint 语义.
- Flash-VQG suite 和 CLI 透传 `validations_per_epoch`, 并支持显式 `run_id` 和 `experiment_mode` override.
- `20260425-gd-residual-v1-mqar` 脚本默认值改为本轮口径: `max_epochs=4`, `cb256`, `tau=0.25`, `train_batch_size=64`, `eval_batch_size=16`, `gradient_accumulation_steps=4`, `validations_per_epoch=2`.

未改动:

- 未改 `legacy`, `clr_v1`, `clr_delta_v1` 语义.
- 未写 Triton/CUDA/custom backward.
- 未优化 `event_pack` 或 `grouped_chunk`.

## 3. 测试结果

| 仓库 | 命令 | 结果 |
| --- | --- | --- |
| zoology | `pytest tests/test_flash_vqg_wrapper.py tests/test_flash_vqg_scripts.py tests/test_flash_vqg_metrics_white_list.py tests/test_train_logging_steps.py tests/test_train_batch_order.py -q` | `82 passed, 1 warning in 13.01s` |
| Flash-VQG | `pytest tests/test_fox_gd_residual_v1.py -q` | `11 passed` |
| Flash-VQG | `pytest tests/test_fox_guards.py tests/test_fox_dense_write.py tests/test_fox_clr_delta_v1.py tests/test_fox_phase2_metrics.py -q` | `66 passed` |

## 4. Baseline

只读取 baseline, 没有重跑.

路径:

- metadata: `zoology/analysis/flash_vqg/results/flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45/dense-t025-cb256-s123-d123/data/metadata.json`
- history: `zoology/analysis/flash_vqg/results/flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45/dense-t025-cb256-s123-d123/data/history.csv`

配置摘要:

| 项 | 值 |
| --- | --- |
| run_id | `dense-t025-cb256-s123-d123` |
| formula | `clr_v1` |
| d_model / n_layers | `128 / 2` |
| heads / key_dim / value_dim | `2 / 64 / 64` |
| block_len / local_num_blocks | `32 / 2` |
| codebook | `256` |
| vq | `codebook_dot`, `dense_softmax`, `grad`, `tau=0.25`, `topk=4` |
| remote read topk | `2` |
| batch / eval / ga | `64 / 16 / 4` |
| lr / wd | `1e-3 / 0.1` |
| seed / data_seed | `123 / 123` |
| train_batch_order | `global_shuffle` |
| max_epochs | `32` |

前 4 epoch:

| epoch | step | valid/loss | valid/accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 704 | 1.492455 | 0.776821 |
| 2 | 1409 | 0.519401 | 0.914123 |
| 3 | 2114 | 0.299164 | 0.951626 |
| 4 | 2819 | 0.237862 | 0.961423 |

epoch 4 slice-level valid accuracy:

| slice | accuracy |
| --- | ---: |
| `input_seq_len=64` | 0.999687 |
| `input_seq_len=128` | 0.991625 |
| `input_seq_len=256` | 0.990531 |
| `input_seq_len=512` | 0.967660 |
| `input_seq_len=1024` | 0.774844 |
| `num_kv_pairs=4` | 0.999750 |
| `num_kv_pairs=8` | 0.999625 |
| `num_kv_pairs=16` | 0.999687 |
| `num_kv_pairs=32` | 0.991625 |
| `num_kv_pairs=64` | 0.986688 |
| `num_kv_pairs=128` | 0.952477 |
| `num_kv_pairs=256` | 0.774844 |
| `1024x256` | 0.774844 |
| `512x128` | 0.952477 |
| `512x64` | 0.982844 |
| `256x64` | 0.990531 |
| `128x32` | 0.991625 |
| `64x16` | 0.999687 |
| `64x8` | 0.999625 |
| `64x4` | 0.999750 |

## 5. 候选配置

目标候选:

| GPU | run_id | mu_min_count |
| --- | --- | ---: |
| GPU0 | `gd-r16-wk4-mu015-t025-cb256-s123-d123` | 0.15 |
| GPU1 | `gd-r16-wk4-mu01-t025-cb256-s123-d123` | 0.1 |

共同配置:

| 项 | 值 |
| --- | --- |
| formula | `gd_residual_v1` |
| rank / write_topk | `16 / 4` |
| builder / pack / chunk | `grouped_chunk_torch_ref / semivec_ref / 64` |
| lambda_init / beta_init | `0.05 / 0.5` |
| norm_with_gain | `false` |
| use_separate_addr_codebook | `false` |
| addr_eps / den_eps / rho_eps | `1e-6 / 1e-6 / 1e-12` |
| d_model / n_layers | `128 / 2` |
| heads / key_dim / value_dim | `2 / 64 / 64` |
| block_len / local_num_blocks | `32 / 2` |
| codebook | `256` |
| vq | `codebook_dot`, `dense_softmax`, `grad`, `tau=0.25`, `topk=4` |
| remote read topk | `2` |
| seed / data_seed | `123 / 123` |
| train_batch_order | `global_shuffle` |
| intended batch / eval / ga | `64 / 16 / 4` |
| runnable smoke batch / eval / ga | `16 / 1 / 16` |
| validations_per_epoch | `2` |

已对齐 baseline:

- 模型尺寸, head/key/value, block/local 设置.
- `cb256`, `tau=0.25`, VQ score/weight/update/topk.
- `remote_read_topk=2`.
- `lr=1e-3`, `weight_decay=0.1`.
- `seed=123`, `data_seed=123`, `global_shuffle`.
- effective train batch 仍为 `256` when using `16x16`.

未对齐项:

- `train_batch_size` 从 `64` 降为 `16`, 因为 `64x4` 和 `32x8` 均 OOM.
- smoke 的 `eval_batch_size` 实际为 `1`, 因为 smoke builder 会缩小 test budget 并设置 eval micro-batch 为 1.
- 主训练样本预算未执行, 因为 `16x16` 线性估算耗时不可接受.
- 没有 gd 4 epoch valid 曲线, 只能报告 smoke gate 指标.

## 6. Cache 检查

正式执行前检查了 MQAR 数据 cache. 目标配置需要的 13 个 cache 文件均存在, 未触发双进程同时写同一个 cache 的风险.

## 7. 运行命令

OOM gate:

```bash
SWANLAB_MODE=offline ANALYSIS_SOURCE=off GPU_ID=0 FOX_GD_RESIDUAL_MU_MIN_COUNT=0.15 RUN_ID_OVERRIDE=gd-r16-wk4-mu015-t025-cb256-s123-d123-smoke EXPERIMENT_MODE_OVERRIDE=smoke_mu015 LAUNCH_ID_PREFIX_SMOKE=flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015 bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh
```

```bash
SWANLAB_MODE=offline ANALYSIS_SOURCE=off GPU_ID=0 TRAIN_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=8 FOX_GD_RESIDUAL_MU_MIN_COUNT=0.15 RUN_ID_OVERRIDE=gd-r16-wk4-mu015-t025-cb256-s123-d123-smoke-tbs32-ga8 EXPERIMENT_MODE_OVERRIDE=smoke_mu015 LAUNCH_ID_PREFIX_SMOKE=flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-tbs32-ga8 bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh
```

通过的 smoke:

```bash
SWANLAB_MODE=offline ANALYSIS_SOURCE=off GPU_ID=0 TRAIN_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=16 FOX_GD_RESIDUAL_MU_MIN_COUNT=0.15 RUN_ID_OVERRIDE=gd-r16-wk4-mu015-t025-cb256-s123-d123-smoke-tbs16-ga16 EXPERIMENT_MODE_OVERRIDE=smoke_mu015 LAUNCH_ID_PREFIX_SMOKE=flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-tbs16-ga16 bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh
```

```bash
SWANLAB_MODE=offline ANALYSIS_SOURCE=off GPU_ID=1 TRAIN_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=16 FOX_GD_RESIDUAL_MU_MIN_COUNT=0.1 RUN_ID_OVERRIDE=gd-r16-wk4-mu01-t025-cb256-s123-d123-smoke-tbs16-ga16 EXPERIMENT_MODE_OVERRIDE=smoke_mu01 LAUNCH_ID_PREFIX_SMOKE=flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu01-tbs16-ga16 bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh
```

本地 analysis:

```bash
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis --launch-id flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-tbs16-ga16-2026-04-26-19-37-51 --source local
python -m zoology.analysis.flash_vqg.run_flash_vqg_analysis --launch-id flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu01-tbs16-ga16-2026-04-26-19-48-21 --source local
```

## 8. Smoke 结果

| run | status | train iters | train wall | avg iter | sampled peak memory | total wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mu=0.15`, `64x4` | OOM | - | - | - | - | 57s |
| `mu=0.15`, `32x8` | OOM | - | - | - | - | 33s |
| `mu=0.15`, `16x16` | completed | 24 | 9m50s | 24.59s | ~10363 MiB | 10m17s |
| `mu=0.1`, `16x16` | completed | 24 | 9m40s | 24.19s | ~10351 MiB | 10m07s |

OOM 摘要:

| run | OOM |
| --- | --- |
| `mu=0.15`, `64x4` | backward tried to allocate 32.00 GiB on 10.74 GiB GPU |
| `mu=0.15`, `32x8` | backward tried to allocate 16.00 GiB on 10.74 GiB GPU |

通过的 smoke train/valid:

| run | train/loss first | train/loss last | valid/loss mid | valid/loss final | valid/accuracy final |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mu=0.15`, `16x16` | 10.803198 | 10.789975 | 9.044239 | 9.043460 | 0.000000 |
| `mu=0.1`, `16x16` | 10.803198 | 10.789976 | 9.044238 | 9.043459 | 0.000000 |

通过的 smoke final gd residual metrics:

| run | mu_valid_ratio | inject_ratio | lambda_mean | write_strength_mean | m_norm_mean | m_norm_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mu=0.15` | 0.000013 | 0.010754 | 0.050035 | 0.125037 | 0.015931 | 0.184106 |
| `mu=0.1` | 0.093386 | 0.010713 | 0.050037 | 0.125038 | 0.015924 | 0.184506 |

slice-level smoke valid accuracy:

- 两个 `16x16` smoke 在 final valid 的 `input_seq_len/*`, `num_kv_pairs/*`, `mqar_case/*` accuracy 全部为 `0.0`.
- 这不作为失败判据, 因为 smoke 只跑了 24 个 micro-batch, 但它也不能支持进入正式候选结论.

NaN/Inf:

- 通过的 smoke 未出现 NaN/Inf loss 或异常终止.
- 本轮 smoke history 未记录显式 `attn/nan_inf_count`, 因此 NaN/Inf 检查以 loss/metric 有限值和进程完成状态为准.

## 9. 与 Baseline 对比

当前没有 gd 4 epoch 主训练结果, 因此不能做 epoch 1/2/3/4 的候选曲线对比.

可对比项:

| 项 | baseline epoch 1 | gd smoke final |
| --- | ---: | ---: |
| valid/loss | 1.492455 | ~9.04346 |
| valid/accuracy | 0.776821 | 0.000000 |

这个比较仅说明 smoke 没有训练到可比阶段, 不说明候选最终质量. 真正阻断项是 `gd_residual_v1` reference 路径在 `cb256`, `rank=16`, `write_topk=4`, `read_topk=2` 下的显存和吞吐.

## 10. 产物位置

成功 smoke analysis:

- `zoology/analysis/flash_vqg/results/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-tbs16-ga16-2026-04-26-19-37-51/gd-r16-wk4-mu015-t025-cb256-s123-d123-smoke-tbs16-ga16/data/history.csv`
- `zoology/analysis/flash_vqg/results/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-tbs16-ga16-2026-04-26-19-37-51/gd-r16-wk4-mu015-t025-cb256-s123-d123-smoke-tbs16-ga16/data/metadata.json`
- `zoology/analysis/flash_vqg/results/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu01-tbs16-ga16-2026-04-26-19-48-21/gd-r16-wk4-mu01-t025-cb256-s123-d123-smoke-tbs16-ga16/data/history.csv`
- `zoology/analysis/flash_vqg/results/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu01-tbs16-ga16-2026-04-26-19-48-21/gd-r16-wk4-mu01-t025-cb256-s123-d123-smoke-tbs16-ga16/data/metadata.json`

Manifests:

- `zoology/experiments/flash_vqg/generated/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-2026-04-26-19-35-45/manifest.json`
- `zoology/experiments/flash_vqg/generated/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-tbs32-ga8-2026-04-26-19-37-05/manifest.json`
- `zoology/experiments/flash_vqg/generated/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu015-tbs16-ga16-2026-04-26-19-37-51/manifest.json`
- `zoology/experiments/flash_vqg/generated/flash-vqg-20260425-gd-residual-v1-4epoch-t025-cb256-smoke-mu01-tbs16-ga16-2026-04-26-19-48-21/manifest.json`

## 11. 候选判断

不建议用当前 `gd_residual_v1` reference 路径继续正式 4 epoch.

如果必须在两个 smoke 候选中保留一个, 优先保留 `mu=0.1` 做后续 debug, 因为它的 `valid_mu_valid_ratio` 约 0.093, 明显高于 `mu=0.15` 的约 0.000013. 但这不是 4 epoch 质量结论, 只是 residual validity 信号更可观.

## 12. 下一步建议

1. 先 profile `event_pack/grouped_chunk_torch_ref` 在 `batch=16`, `seq_len=256`, `cb256`, `write_topk=4` 的 backward 显存来源, 找出 16/32 GiB allocation 的张量形状.
2. 尝试降低 event 规模的候选 gate, 例如 `write_topk=2` 或 `rank=8`, 目标先恢复 `32x8` 或 `64x4`.
3. 如果要保持 `rank=16`, `write_topk=4`, 则先做 reference 路径效率修正, 不建议直接消耗约 12.7 天跑两个 4 epoch 候选.
4. 当至少 `32x8` 通过且每 epoch wall-clock 可接受时, 再按原计划启动 `mu=0.15` 和 `mu=0.1` 的正式 4 epoch 并和 baseline 对齐比较.
