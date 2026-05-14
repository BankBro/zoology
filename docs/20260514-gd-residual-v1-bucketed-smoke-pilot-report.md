# gd_residual_v1 bucketed 后续 smoke/pilot 报告

日期: 2026-05-14

## 1. 摘要

本次在 `grouped_chunk_torch_ref` bucketed 优化后继续完成三件事:

1. 补强 Flash-VQG correctness tests, 覆盖 CUDA forward/backward oracle 对齐和 zero-count group branch.
2. 补跑 current 64x4 smoke, 使用 `mu_min_count=0.1` 和 official runtime 配置.
3. 跑 `mu=0.1` 的 100-step short pilot, 验证真实训练路径上的稳定性和短程信号.

结论: correctness 通过, 64x4 smoke 完成, 100-step pilot 完成, 无 OOM, 无 NaN/Inf failure record. 这仍然不是 official 4 epoch 质量结论.

## 2. 仓库状态

| repo | branch | commit | 说明 |
|---|---|---|---|
| Flash-VQG | `20260428-gd-residual-v1-sync` | `811e1ce` | 已补充 grouped_chunk CUDA / zero-count 测试并推送 |
| zoology | `flash-vqg` | `0e87a26` | 本报告和 artifacts 基于此提交追加 |

Flash-VQG 测试补强:

- `test_gd_grouped_chunk_bucketed_handles_zero_count_groups`
- `test_gd_grouped_chunk_bucketed_cuda_matches_loop_oracle_forward_backward`

## 3. Correctness gate

在 `/home/lyj/mnt/project/Flash-VQG` 运行:

| command | result |
|---|---|
| `pytest tests/test_fox_gd_residual_v1.py -q -k "grouped_chunk_bucketed"` | `6 passed, 11 deselected` |
| `pytest tests/test_fox_gd_residual_v1.py -q` | `17 passed` |
| `pytest tests/test_attn_fox_compat.py -q` | `5 passed` |

新增 CUDA 测试在本机 RTX 2080 Ti 上执行, 覆盖 zero-count group 和不均匀 event count 的 forward/backward grad close.

## 4. 64x4 smoke

运行配置:

| item | value |
|---|---|
| run id | `gd-r16-wk4-mu01-t025-cb256-s123-d123-bucketed-smoke-tbs64-ga4` |
| launch id | `flash-vqg-20260514-gd-residual-v1-bucketed-smoke-mu01-tbs64-ga4-2026-05-14-14-29-41` |
| `rank / write_topk / read_topk` | `16 / 4 / 2` |
| `num_codebook_vectors` | `256` |
| `vq_weight_mode / tau` | `dense_softmax / 0.25` |
| `builder / pack` | `grouped_chunk_torch_ref / semivec_ref` |
| `mu_min_count` | `0.1` |
| `TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS` | `64 / 4` |
| status | `completed` |

最终 smoke 指标:

| metric | value |
|---|---:|
| `train/loss` | `10.7865066528` |
| `valid/loss` | `9.0444288254` |
| `valid/accuracy` | `0.0` |
| `valid/attn/gd_residual_mu_valid_ratio` | `0.0926723480` |
| `valid/attn/gd_residual_inject_ratio` | `0.0106633850` |
| `valid/attn/gd_residual_lambda_mean` | `0.0499641919` |
| `valid/attn/gd_residual_m_norm_mean` | `0.0159287043` |
| `valid/attn/gd_residual_m_norm_max` | `0.1839346427` |

判断: smoke runtime gate 通过. `valid/accuracy=0.0` 不作为失败判据, 不能解读为正式训练质量结论.

## 5. 100-step short pilot

第一次直接调用脚本时缺少 `PYTHONPATH`, 在 import 阶段失败: `ModuleNotFoundError: No module named 'zoology'`. 未进入训练. 随后使用相同训练配置加上 `PYTHONPATH=/home/lyj/mnt/project/zoology` 重新运行并完成.

运行配置:

| item | value |
|---|---|
| variant | `gd_r16_wk4` |
| seed / data seed | `123 / 123` |
| train batches | `100` |
| valid every | `20` |
| eval batches | `2` |
| `mu_min_count` | `0.1` |
| diagnostics | enabled |
| status | `completed` |

summary:

| metric | value |
|---|---:|
| train records | `100` |
| valid records | `6` |
| NaN/Inf records | `0` |
| first train loss | `10.9784460068` |
| final train loss | `9.5773897171` |
| initial valid loss | `9.0094652176` |
| final valid loss | `8.7230839729` |
| final valid accuracy | `0.0` |
| avg step sec | `2.5489988219` |
| peak allocated GiB | `6.1026496887` |
| peak reserved GiB | `8.4375` |

最后一个 train record 的 gd metrics:

| metric | value |
|---|---:|
| `attn/gd_residual_lambda_mean` | `0.0365100354` |
| `attn/gd_residual_inject_ratio` | `0.0012330422` |
| `attn/gd_residual_m_norm_mean` | `0.0042590229` |
| `attn/gd_residual_m_norm_max` | `0.8968794346` |
| `attn/gd_residual_mu_valid_ratio` | `0.8509368896` |
| `attn/gd_residual_debug_event_pack_wall_sec` | `0.0146427266` |
| `attn/gd_residual_debug_grouped_chunk_wall_sec` | `0.2391663492` |
| `attn/gd_residual_debug_phase2_residual_read_wall_sec` | `0.0036487114` |

最终 valid gd metrics:

| metric | value |
|---|---:|
| `valid/attn/gd_residual_lambda_mean` | `0.0361594651` |
| `valid/attn/gd_residual_inject_ratio` | `0.0` |
| `valid/attn/gd_residual_m_norm_mean` | `0.0057036867` |
| `valid/attn/gd_residual_m_norm_max` | `0.8313809335` |
| `valid/attn/gd_residual_mu_valid_ratio` | `0.3944396973` |

判断: short pilot runtime/stability gate 通过. 100-step 下 train loss 和 valid loss 都下降, 但 valid accuracy 仍为 `0.0`; 这不是 official 4 epoch 质量结论.

## 6. Artifacts

已保存小体积 artifacts:

- `docs/artifacts/20260514-gd-bucketed-smoke-pilot/smoke-64x4-history.csv`
- `docs/artifacts/20260514-gd-bucketed-smoke-pilot/smoke-64x4-metadata.json`
- `docs/artifacts/20260514-gd-bucketed-smoke-pilot/smoke-64x4-summary.json`
- `docs/artifacts/20260514-gd-bucketed-smoke-pilot/smoke-64x4-run_summary.csv`
- `docs/artifacts/20260514-gd-bucketed-smoke-pilot/short-pilot-mu01-steps100-summary.json`
- `docs/artifacts/20260514-gd-bucketed-smoke-pilot/short-pilot-mu01-steps100-records.jsonl`

未提交:

- `tmp/` 全目录.
- `zoology/experiments/flash_vqg/generated/` 新生成目录.
- SwanLab 本地日志目录.
- checkpoint.
- 数据 cache.

## 7. 下一步建议

现在可以进入单候选 official 4 epoch 前的最后决策点. 推荐下一步只跑一个 official candidate:

- `gd-r16-wk4-mu01-t025-cb256-s123-d123`

不要同时启动 `mu=0.1` 和 `mu=0.15`. `event_pack` 暂不作为第一优先级继续优化; 若 official run wall-clock 仍然不满意, 再回到 `event_pack` Phase-B profiler.

明确声明:

- 本次没有重跑 baseline.
- 本次没有启动 official full 4 epoch.
- 本次没有修改 official runtime 超参, 只使用 `mu=0.1` candidate 做 smoke/pilot.
- 本次没有改变 `gd_residual_v1` 数学.
- 本次 smoke/pilot 不是正式训练质量结论.
