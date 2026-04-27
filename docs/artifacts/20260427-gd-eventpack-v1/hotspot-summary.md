# gd_residual_v1 event_pack v1 hotspot summary

日期: 2026-04-27

用途: 记录 event_pack metadata no_grad 和 logabar/tail vectorization 后, B8,T128 profiler 中关键热点相对 flat-gather baseline 的变化. baseline 数据来自 `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-summary.json` 和 `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-profiler_cuda_memory_usage.txt`; eventpack-v1 数据来自本目录 profiler 表.

## 关键指标

| 指标 | flat-gather baseline | eventpack-v1 | 变化 |
| --- | ---: | ---: | ---: |
| B8,T128 microbatch_sec | `23.331968s` | `13.780667s` | `-40.94%` |
| B8,T128 forward_sec | `8.131222s` | `5.694159s` | `-29.97%` |
| B8,T128 backward_sec | `15.145080s` | `8.037872s` | `-46.93%` |
| peak_reserved_GB | `0.710938` | `0.708984` | `-0.27%` |
| peak_allocated_GB | `0.551205` | `0.549210` | `-0.36%` |

## Profiler 热点

| 热点 | flat-gather baseline | eventpack-v1 | 变化 |
| --- | ---: | ---: | ---: |
| `gd_residual/event_pack` CPU total | `2.549s` | `87.172ms` | `-96.58%` |
| `gd_residual/event_pack` CUDA total | `2.548s` | `85.748ms` | `-96.63%` |
| `gd_residual/event_pack` self CPU | `726.939ms` | `3.679ms` | `-99.49%` |
| `aten::copy_` CPU total | `2.441s` | `711.946ms` | `-70.83%` |
| `aten::copy_` self CUDA | `607.581ms` | `211.780ms` | `-65.14%` |
| `aten::copy_` calls | `182866` | `51314` | `-71.94%` |
| `aten::item` CPU total | `530.681ms` | `175.455ms` | `-66.94%` |
| `aten::item` calls | `37243` | `11681` | `-68.64%` |
| `aten::_local_scalar_dense` CPU total | `483.143ms` | `160.498ms` | `-66.78%` |
| `aten::_local_scalar_dense` calls | `37243` | `11681` | `-68.64%` |
| `gd_residual/grouped_chunk` CUDA total | `2.723s` | `2.802s` | `+2.90%` |

## 解释

event_pack v1 主要减少 metadata/indexing autograd 图和 per-event Python scalarization. 因此 `event_pack`, `copy_`, `item`, `_local_scalar_dense` 都明显下降. `grouped_chunk_torch_ref` 本身没有优化, profiler 中它已经成为下一阶段主要热点.
