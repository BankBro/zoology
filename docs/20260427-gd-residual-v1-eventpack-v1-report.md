# gd_residual_v1 event_pack v1 throughput report

日期: 2026-04-27

范围: 本次只优化 `Flash-VQG/src/flash_vqg/nn/attn_fox.py` 中 `_build_gd_event_pack_semivec_ref`, 不改 `grouped_chunk_torch_ref` 递推本身, 不改 token-step reference, 不改 block-entry frozen baseline, 不改 legacy/clr 路径, 不改 rank/write_topk/read_topk/cb/tau 等 official candidate 配置.

## 结论

event_pack v1 优化有效. 在 B64,T256,8 microbatches profile 中, avg_microbatch_sec 从 `197.81s` 降到 `80.82s`, 下降 `59.14%`; avg_backward_sec 从 `153.58s` 降到 `57.10s`, 下降 `62.82%`; peak_reserved 从 `8.54 GiB` 到 `8.50 GiB`, 没有升高. 8 个 microbatch loss 均 finite.

B8,T128 profiler 显示主要收益来自 event_pack metadata autograd 和 Python scalarization 被移除: `gd_residual/event_pack` CPU total 从 `2.549s` 降到 `87.172ms`, `aten::copy_` calls 从 `182866` 降到 `51314`, `aten::item/_local_scalar_dense` calls 从 `37243` 降到 `11681`.

当前最可能的下一步瓶颈是 `grouped_chunk_torch_ref`, 而不是 event_pack. 本轮没有实现 grouped_chunk 优化.

## 代码改动

代码提交在 `BankBro/Flash-VQG`, 文件:

- `src/flash_vqg/nn/attn_fox.py`
- commit: `705bcd7 Optimize gd residual event pack throughput`

核心改动:

- `_build_gd_event_pack_semivec_ref` 中 metadata 构造进入 `torch.no_grad()`: `event_mask`, `event_pos`, `b_e/h_e/ell_e/kw_e`, `s_e`, `group_id_e`, `sort_key`, `order`, `group_ids`, `cu_seqlens`.
- 保留梯度路径: `rho_e`, `K_e`, `V_e`, `codebook[h_e, s_e]`, `addr_proj[h_e]`, `beta_blk[b_e, h_e, ell_e]`, `M_ent_pack`.
- `logabar_pack` 和 `alpha_tail_pack` 改为基于 `log_alpha` prefix sum 的向量化计算.
- `M_ent_pack` 改为从 decoded `group_ids` 向量化 gather.
- `GDEventPack` ABI 未变.

## 测试结果

在 `/home/lyj/mnt/project/Flash-VQG` 运行:

```bash
pytest tests/test_fox_gd_residual_v1.py -q
pytest tests/test_fox_guards.py tests/test_fox_dense_write.py tests/test_fox_clr_delta_v1.py tests/test_fox_phase2_metrics.py -q
```

结果:

- `tests/test_fox_gd_residual_v1.py`: `11 passed`.
- guards/dense_write/clr_delta/phase2_metrics: `66 passed`.
- 额外不落盘梯度探针通过, 确认 `zeta_pack` 对 `W_blk` 仍有梯度, `D_pack/U_pack/M_ent_pack` 没有被误 detach.

## 提交的数据附件

为便于 GitHub 侧复核, 本次提交精简实验数据:

- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-b64-t256-mb8-summary.json`: B64,T256,8 microbatches profile summary.
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-summary.json`: B8,T128 profiler summary.
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-profiler_cpu_time_total.txt`: B8 profiler CPU total 表.
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-profiler_cuda_time_total.txt`: B8 profiler CUDA total 表.
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-profiler_cuda_memory_usage.txt`: B8 profiler CUDA memory 表.
- `docs/artifacts/20260427-gd-eventpack-v1/hotspot-summary.md`: 手工整理的关键热点对比.

对比用 baseline 已在仓库中:

- `docs/artifacts/20260427-gd-flatgather-64x4/profile-b64-t256-mb8-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-profiler_cuda_memory_usage.txt`

未提交完整 terminal log, torch profiler trace, SwanLab 本地日志, checkpoint, generated launch config 或 `tmp` 目录.

## B64,T256 Profile

命令:

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
PROFILE_OUTPUT_DIR=tmp/gd_event_grouped_opt/eventpack-v1-b64-t256-mb8 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

| 指标 | flat-gather baseline | eventpack-v1 | 变化 |
| --- | ---: | ---: | ---: |
| peak_reserved_GB | `8.5449` | `8.5020` | `-0.50%` |
| peak_allocated_GB | `6.7816` | `6.7499` | `-0.47%` |
| avg_forward_sec | `43.7471` | `23.6432` | `-45.95%` |
| avg_backward_sec | `153.5802` | `57.0983` | `-62.82%` |
| avg_microbatch_sec | `197.8099` | `80.8248` | `-59.14%` |

eventpack-v1 的 8 个 microbatch loss:

`9.0392`, `8.9500`, `8.8633`, `8.7749`, `8.6889`, `8.6037`, `8.5199`, `8.4378`.

## B8,T128 Profiler

命令:

```bash
cd /home/lyj/mnt/project/zoology

FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
TRAIN_BATCH_SIZE=8 \
PROFILE_SEQ_LEN=128 \
PROFILE_MICROBATCHES=1 \
PROFILE_ENABLE_TORCH_PROFILER=1 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/gd_event_grouped_opt/eventpack-v1-prof-b8-t128 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

| 指标 | flat-gather baseline | eventpack-v1 | 变化 |
| --- | ---: | ---: | ---: |
| peak_reserved_GB | `0.7109` | `0.7090` | `-0.27%` |
| peak_allocated_GB | `0.5512` | `0.5492` | `-0.36%` |
| forward_sec | `8.1312` | `5.6942` | `-29.97%` |
| backward_sec | `15.1451` | `8.0379` | `-46.93%` |
| microbatch_sec | `23.3320` | `13.7807` | `-40.94%` |
| loss | `9.0333` | `9.0333` | unchanged |

关键热点见 `docs/artifacts/20260427-gd-eventpack-v1/hotspot-summary.md`.

## 下一步建议

建议继续做下一轮吞吐优化, 重点转向 `grouped_chunk_torch_ref`. B8 profiler 中 `gd_residual/grouped_chunk` CUDA total 约 `2.802s`, 已经成为 gd residual phase1 的主要热点之一. 相关现象包括大量 `SelectBackward0/select_backward`, `copy_/CopySlices`, `zero_/fill_`, per-event `mv/outer`, 以及 group/event loop 中的标量索引.

下一轮建议保持 official candidate 配置不变, 先只优化 grouped_chunk 的执行形态或 metadata/autograd 开销, 并继续用 B64,T256,mb8 profile 和 B8,T128 profiler 做 A/B gate.

## 回退方式

Flash-VQG 代码回退:

```bash
git revert <eventpack-v1-code-commit>
```

zoology 文档和 artifacts 回退:

```bash
git revert <eventpack-v1-doc-commit>
```
