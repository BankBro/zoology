# gd_residual_v1 grouped_chunk bucketed reference report

日期: 2026-05-14

## 1. 摘要

本次完成 `gd_residual_v1` 中 `grouped_chunk_torch_ref` 的 PyTorch reference 执行形态优化: public builder 名称保持不变, 内部从逐 group / 逐 event loop 改为按 event count 分桶的 batched recurrence. 本次没有改变 recurrence 数学, 没有修改 official candidate 超参, 没有写 Triton / CUDA / custom backward.

结果: correctness tests 通过. strict `tau=0.25` B8 profiler 与 B64/T256/8 microbatches profile 均完成, loss finite, 不 OOM. B64 平均 microbatch 从文档 baseline `80.005957s` 降到 `1.195595s`, peak reserved 从 `8.501953 GiB` 降到 `8.431641 GiB`.

## 2. 仓库状态

| repo | branch | commit | status |
|---|---|---|---|
| Flash-VQG | `20260428-gd-residual-v1-sync` | `06d4b80` | modified: `src/flash_vqg/nn/fox/gd_residual.py`, `tests/test_fox_gd_residual_v1.py` |
| zoology | `flash-vqg` | `ce186fa` | new report/artifacts, existing untracked implementation plan |

修改文件:

- Flash-VQG: `src/flash_vqg/nn/fox/gd_residual.py`
- Flash-VQG: `tests/test_fox_gd_residual_v1.py`
- zoology: `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-report.md`
- zoology: `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/`

## 3. 实现说明

- 新增 `_grouped_chunk_torch_loop_oracle`, 完整保留旧逐 group / 逐 event loop 逻辑, 只作为 correctness oracle 和紧急回退参考.
- 新增 `_grouped_chunk_torch_bucketed_ref`, 按 `cu_seqlens` 计算每个 group 的 event count, 将相同 event count 的 group 分桶, 对每个 bucket 一次性 gather `D/U/zeta/logabar/state/tail`, 只在 event step 上循环.
- bucket 内用 `torch.bmm` 和广播 outer update 替代逐 event `matmul/outer/addmv` 小算子, 输出用 bucket 级 `index_copy_` 写回.
- `grouped_chunk_torch_ref` 现在直接调用 bucketed 实现. `chunk_size` 语义保持原状: 只校验正数, 不改变 recurrence.

## 4. Correctness tests

在 `/home/lyj/mnt/project/Flash-VQG` 运行:

| command | result |
|---|---|
| `pytest tests/test_fox_gd_residual_v1.py -q -k "grouped_chunk_bucketed"` | `4 passed, 11 deselected` |
| `pytest tests/test_fox_gd_residual_v1.py -q` | `15 passed` |
| `pytest tests/test_attn_fox_compat.py -q` | `5 passed` |

新增覆盖:

- bucketed forward 对齐 old loop oracle.
- bucketed backward gradients 对齐 old loop oracle, 覆盖 `M_ent_pack`, `D_pack`, `U_pack`, `zeta_pack`, `logabar_pack`, `alpha_tail_pack`.
- empty pack 行为保持 clone/shape/dtype/device.
- `chunk_size=1/2/64` 结果一致, `chunk_size=0` 抛错.

## 5. B8,T128 profiler gate

命令保持 strict official profile 配置: `rank=16`, `write_topk=4`, `read_topk=2`, `cb=256`, `dense_softmax`, `tau=0.25`, `builder=grouped_chunk_torch_ref`, `pack=semivec_ref`.

summary:

| item | value |
|---|---:|
| `vq_softmax_tau` | `0.25` |
| loss finite | `true` |
| `microbatch_sec` | `5.715510` |
| `forward_sec` | `3.563703` |
| `backward_sec` | `2.082318` |
| `peak_allocated_GiB` | `0.542016` |
| `peak_reserved_GiB` | `0.681641` |

profiler 对比:

| metric | baseline | current |
|---|---:|---:|
| `gd_residual/grouped_chunk` CUDA total | `2.783s` | `76.141ms` |
| `gd_residual/event_pack` CUDA total | `84.894ms` | `136.367ms` |
| `gd_residual/phase2_residual_read` CUDA total | `1.220ms` | `1.458ms` |
| `aten::copy_` calls | `51314` | `1283` |
| `aten::item` calls | `11681` | not in top-100 profiler table |
| `aten::_local_scalar_dense` calls | `11681` | not in top-100 profiler table |
| `aten::outer` calls | `14336` | not in top-100 profiler table |
| `aten::mv` calls | `14336` | not in top-100 profiler table |
| `aten::matmul` calls | `8197` | `5` |
| `aten::addmv_` calls | `14336` | not in top-100 profiler table |
| `aten::select_backward` calls | `31863` | `272` |
| `torch::autograd::CopySlices` calls | `4360` | `16` |
| `aten::gather_backward` memory | `2.00 MB` | `2.00 MB` |
| `aten::index_select_backward` memory | `132.19 MB` | `206.06 MB` |

说明: `grouped_chunk` 已从主要瓶颈降到 `76.141ms`. `event_pack` 本轮没有改代码, 但 B8 profiler 中绝对 CUDA total 高于文档 baseline, 且由于 `grouped_chunk` 大幅下降, 它现在成为 gd residual 三个 range 中最大的后续优化候选. 这不是模型质量结论.

## 6. B64,T256,8 microbatches profile gate

summary:

| metric | baseline | current |
|---|---:|---:|
| `vq_softmax_tau` | `0.25` | `0.25` |
| `avg_microbatch_sec` | `80.005957s` | `1.195595s` |
| `avg_forward_sec` | `22.903922s` | `0.592819s` |
| `avg_backward_sec` | `57.025518s` | `0.594279s` |
| `peak_reserved_GiB` | `8.501953` | `8.431641` |
| `peak_allocated_GiB` | `6.749899` | `6.609921` |
| losses finite | `true` | `true` |

第一 microbatch 包含初始化开销, `microbatch_sec=5.746792s`; 后 7 个 microbatches 均约 `0.53s` 到 `0.56s`. 以上平均值按全部 8 个 records 计算.

## 7. Artifacts

已保存小体积 artifacts:

- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/b64-t256-mb8-summary.json`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-summary.json`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-profiler_cpu_time_total.txt`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-profiler_cuda_time_total.txt`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-profiler_cuda_memory_usage.txt`

未提交 profiler trace 大文件, `tmp/` 全目录, SwanLab 本地日志, checkpoint, generated launch config, `__pycache__`.

## 8. 结论

本阶段 hard correctness gate 通过, strict `tau=0.25` profile summary 记录正确, B64 loss finite, 不 OOM, peak reserved 未高于 `9.0 GiB`. soft performance success 明确成立: `grouped_chunk` CUDA total, total microbatch time, backward time, `copy_`, `matmul`, `select_backward`, `CopySlices` 均显著下降.

current 64x4 smoke 未执行, 不能声明 current 64x4 smoke 已通过.

建议进入下一阶段前先保留本 patch, 并将 `event_pack` 作为后续 profiler 优化候选. 若要进入正式训练, 仍需单独执行 current 64x4 smoke 或 official 4 epoch gate. 本报告中的 smoke/profile 不是正式训练质量结论.

明确声明:

- 没有重跑 baseline.
- 没有启动 official full 4 epoch.
- 没有修改 official 超参.
- 没有重新设计 `gd_residual_v1` 数学.
- 没有写 Triton / CUDA / custom backward.
- 没有改 legacy / `clr_v1` / `clr_delta_v1`.
- smoke/profile 不是正式训练质量结论.

## 9. Web ChatGPT 阅读入口

推荐网页版 ChatGPT 从以下远端内容阅读本次实验:

| repo | branch | commit / reference | files |
|---|---|---|---|
| `BankBro/Flash-VQG` | `20260428-gd-residual-v1-sync` | `af7f5e1` | `src/flash_vqg/nn/fox/gd_residual.py`, `tests/test_fox_gd_residual_v1.py` |
| `BankBro/zoology` | `flash-vqg` | 本报告推送后的 commit | `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-implementation-plan.md`, `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-report.md`, `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/` |

已选择提交的实验数据文件:

- `b64-t256-mb8-summary.json`: B64/T256/8 microbatches profile summary.
- `prof-b8-t128-summary.json`: B8/T128 profiler summary.
- `prof-b8-t128-profiler_cpu_time_total.txt`: B8 profiler CPU total table.
- `prof-b8-t128-profiler_cuda_time_total.txt`: B8 profiler CUDA total table.
- `prof-b8-t128-profiler_cuda_memory_usage.txt`: B8 profiler CUDA memory table.

未提交的内容:

- `tmp/` 下完整运行目录.
- PyTorch profiler trace JSON 大文件.
- checkpoint, SwanLab 本地日志, generated launch config, `__pycache__`.
