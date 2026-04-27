# gd_residual_v1 phase2 flat gather 显存优化实验报告

日期: 2026-04-27

## 1. 给网页版 ChatGPT 的阅读入口

本报告用于分析 `gd_residual_v1` 的 phase2 residual read 显存优化. 代码改动在 Flash-VQG 仓库, 实验脚本和数据附件在 zoology 仓库.

核心问题: 旧实现用 expanded source 做 gather:

- `codebook.view(...).expand(...).gather(...)`
- `M_remote.unsqueeze(...).expand(...).gather(...)`

这会让 backward 在 expanded 大张量形状上 materialize 梯度, 在主配置 `rank=16`, `write_topk=4`, `read_topk=2`, `cb256`, `dense_softmax`, `tau=0.25` 下导致显存膨胀.

本次只优化 `Flash-VQG/src/flash_vqg/nn/attn_fox.py` 的 `phase2_fox_gd_residual_correction_torch`, 不改 `gd_residual_v1` 数学, 不改 block-entry frozen baseline, 不改 rank/write_topk/read_topk/cb/tau, 不写 Triton/CUDA/custom backward, 不改 legacy / `clr_v1` / `clr_delta_v1`.

## 2. 代码改动

文件:

- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn_fox.py`

改动位置:

- `phase2_fox_gd_residual_correction_torch`
- residual read 内 `C_sel` 和 `M_sel` 的选择逻辑

旧逻辑:

```python
codebook_exp = codebook.view(1, H, 1, 1, num_codes, -1).expand(B, H, N, L, num_codes, -1)
C_sel = torch.gather(codebook_exp, dim=4, index=...)
M_sel = torch.gather(
    M_remote.unsqueeze(3).expand(B, H, N, L, num_codes, d_v, r),
    dim=4,
    index=...,
)
```

新逻辑:

```python
head_offsets = torch.arange(H, device=Q_blk.device, dtype=torch.long).view(1, H, 1, 1, 1)
code_flat_idx = (top_idx + head_offsets * num_codes).reshape(-1)
C_sel = codebook.reshape(H * num_codes, codebook.size(-1)).index_select(0, code_flat_idx)
C_sel = C_sel.view(B, H, N, L, read_topk, codebook.size(-1))

b_offsets = torch.arange(B, device=Q_blk.device, dtype=torch.long).view(B, 1, 1, 1, 1)
n_offsets = torch.arange(N, device=Q_blk.device, dtype=torch.long).view(1, 1, N, 1, 1)
M_flat_idx = (((b_offsets * H + head_offsets) * N + n_offsets) * num_codes + top_idx).reshape(-1)
M_sel = M_remote.reshape(B * H * N * num_codes, d_v, r).index_select(0, M_flat_idx)
M_sel = M_sel.view(B, H, N, L, read_topk, d_v, r)
```

输出 shape 保持不变:

- `C_sel: [B,H,N,L,K_read,d_k]`
- `M_sel: [B,H,N,L,K_read,d_v,r]`

## 3. 提交的数据文件

为了让网页版 ChatGPT 能在 GitHub 代码仓中直接查看关键证据, 本次选择提交小体积、可复核的数据附件:

- `docs/artifacts/20260427-gd-phase2-flat-gather/old-t256-rread2-b16-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-t256-rread2-b16-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/old-prof-b8-t128-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/old-prof-b8-t128-profiler_cuda_memory_usage.txt`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-profiler_cuda_memory_usage.txt`
- `docs/artifacts/20260427-gd-phase2-flat-gather/smoke-32x8-result.md`

没有提交以下产物:

- torch profiler trace JSON, 文件大且环境相关.
- `swanlog/`, checkpoint, `__pycache__`, generated launch config.
- 整个 `tmp/` 目录.

## 4. 测试结果

Flash-VQG:

| 命令 | 结果 |
| --- | --- |
| `pytest tests/test_fox_gd_residual_v1.py -q` | `11 passed in 0.29s` |
| `pytest tests/test_fox_guards.py tests/test_fox_dense_write.py tests/test_fox_clr_delta_v1.py tests/test_fox_phase2_metrics.py -q` | `66 passed in 17.96s` |

额外做了一个不落盘的小 shape old/new 对照:

- old expanded gather 和 new flat gather 的 `C_sel/M_sel` forward `assert_close` 通过.
- backward 后 `codebook/M_remote` 梯度 `assert_close` 通过.
- 梯度 finite 且非零.

## 5. Profile A/B 结果

### B16,T256, profiler off

命令:

```bash
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
TRAIN_BATCH_SIZE=16 \
PROFILE_SEQ_LEN=256 \
PROFILE_MICROBATCHES=1 \
PROFILE_ENABLE_TORCH_PROFILER=0 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/gd_phase2_flat_gather_ab/new-b16-t256 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

对比:

| 版本 | peak_reserved_GB | peak_allocated_GB | forward_sec | backward_sec | microbatch_sec | loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old expanded gather | 5.583984 | 5.094963 | 12.464064 | 24.649285 | 37.183079 | 9.036685 |
| new flat gather | 2.154297 | 1.711079 | 13.505596 | 26.200924 | 39.781209 | 9.036441 |

结论:

- `peak_reserved_GB` 从 5.58 GiB 降到 2.15 GiB, 下降约 61.4%.
- `peak_allocated_GB` 从 5.09 GiB 降到 1.71 GiB, 下降约 66.4%.
- loss finite.
- 时间略慢, 但本次目标是验证 backward 显存是否下降.

### B8,T128, profiler on

命令:

```bash
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
TRAIN_BATCH_SIZE=8 \
PROFILE_SEQ_LEN=128 \
PROFILE_MICROBATCHES=1 \
PROFILE_ENABLE_TORCH_PROFILER=1 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/gd_phase2_flat_gather_ab/new-prof-b8-t128 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

对比:

| 版本 | peak_reserved_GB | peak_allocated_GB | forward_sec | backward_sec | microbatch_sec | loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old expanded gather | 1.644531 | 1.268227 | 7.671179 | 14.724265 | 22.451003 | 9.033426 |
| new flat gather | 0.710938 | 0.551205 | 8.131222 | 15.145080 | 23.331968 | 9.033280 |

`profiler_cuda_memory_usage.txt` 关键行:

| 版本 | operator | CUDA memory usage | calls |
| --- | --- | ---: | ---: |
| old expanded gather | `aten::gather_backward` | 1.06 Gb | 4 |
| new flat gather | `aten::gather_backward` | 2.00 Mb | 2 |
| new flat gather | `aten::index_select_backward` | 132.21 Mb | 8 |

结论:

- residual read 的 expanded gather backward 显存热点被消除.
- 新增的 `index_select_backward` 仍有开销, 但远小于旧 `gather_backward`.
- 新版本还保留 `omega_sel = torch.gather(omega, dim=-1, index=top_idx)`, 所以 profiler 中仍有小额 `gather_backward`.

## 6. 32x8 smoke

触发条件: B16,T256 显存明显下降后运行.

命令见:

- `docs/artifacts/20260427-gd-phase2-flat-gather/smoke-32x8-result.md`

结果:

- 退出码 0.
- `TRAIN_BATCH_SIZE=32`, `GRADIENT_ACCUMULATION_STEPS=8`.
- 完成 `Train Epoch 0/1: 12/12`.
- 最终 validation 完成 `32/32`.
- 最终 `valid/loss=9.04`.
- 无 OOM, 无 NaN/Inf 输出.

这说明在本地 2080 Ti 11GB 环境下, new flat gather 版本可以跑过之前关注的 32x8 smoke.

## 7. 结论和建议

结论:

- 本次实验支持原假设: phase2 residual read 的 expanded gather backward materialization 是显存异常放大的主要来源之一.
- 将 `C_sel/M_sel` 改为 compact source 上的 flat `index_select` 后, B16,T256 peak reserved 从 5.58 GiB 降到 2.15 GiB.
- B8,T128 profiler 中 `aten::gather_backward` CUDA memory usage 从 1.06 Gb 降到 2.00 Mb.
- 32x8 smoke 通过.

建议:

- 可以保留这次 flat gather 改动.
- 下一步继续优化 `event_pack/grouped_chunk_torch_ref`, 因为 phase2 residual read 不再是当前最明显的显存热点, 而 previous profiler 已显示 event pack/grouped chunk 是主要时间瓶颈.
- 如果继续压低 residual read 的小额 profiler 项, 可以单独评估 `omega_sel` 的 gather, 但优先级低于 event pack/grouped chunk.

## 8. 回退方式

只回退本次代码改动时, 在 Flash-VQG 仓库执行:

```bash
git checkout -- src/flash_vqg/nn/attn_fox.py
```

若只想手动回退核心 hunk, 将 `phase2_fox_gd_residual_correction_torch` 内 `C_sel/M_sel` 的 flat `index_select` 逻辑替回 expanded `torch.gather` 逻辑即可.
