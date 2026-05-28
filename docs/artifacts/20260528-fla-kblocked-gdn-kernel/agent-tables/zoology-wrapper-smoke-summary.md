# zoology wrapper adaptation and smoke summary

updated_at_utc: 2026-05-27T20:28:55Z
status: passed
gate: `zoology_wrapper_adaptation_and_smoke_stage`

## 适配

- 修改范围仅限 3090 隔离 zoology worktree: `/home/lyj/mnt/project/worktrees/fla-kblocked/zoology/zoology/mixers/gated_delta_net.py`.
- diff stat: `1 file changed, 3 deletions(-)`.
- 删除 `head_first=False` kwarg:
  - `GatedDeltaNet` chunk 调用.
  - `GatedDeltaNet` fused_recurrent 调用.
  - `GatedDeltaNetBankedK` chunk 调用.
- `grep head_first zoology/mixers/gated_delta_net.py` 无残留.

## 检查

- `git diff --check`: passed.
- `py_compile zoology/mixers/gated_delta_net.py`: passed.
- import source sanity: passed.
  - `zoology` -> `/home/lyj/mnt/project/worktrees/fla-kblocked/zoology/zoology/__init__.py`.
  - `flash_vqg` -> `/home/lyj/mnt/project/worktrees/fla-kblocked/Flash-VQG/src/flash_vqg/__init__.py`.
  - `fla` -> `/home/lyj/mnt/project/worktrees/fla-kblocked/flash-linear-attention/fla/__init__.py`.

## CUDA smoke

配置:

- 3090 单卡, `CUDA_VISIBLE_DEVICES=0`.
- `GDN_KERNEL_DTYPE=float16`, `PYTHONPATH` 为空.
- `torch=2.6.0+cu118`, CUDA `11.8`, device `NVIDIA GeForce RTX 3090`.
- `GatedDeltaNetExpandedK`, `mode=chunk`, `use_gate=True`, `use_short_conv=True`, `B=1,T=128`.
- timings 包含 first-run/JIT, 只作为 smoke 粗略耗时, 不作为 benchmark.

结果:

| case | output | kernel dtype | path hit | loss | max allocated | wall |
|---|---:|---|---|---:|---:|---:|
| `H=2,K=512,V=128` | `[1,128,256]` | `torch.float16` | true large-K fwd/bwd | `4.2106e-04` | `280.7085 MiB` | `19371.0895 ms` |
| `H=2,K=1024,V=64` | `[1,128,128]` | `torch.float16` | true large-K fwd/bwd | `5.0817e-04` | `282.6108 MiB` | `72322.2521 ms` |

Instrumentation:

- public `chunk_gated_delta_rule` call saw `head_first_kwarg_present=false`.
- each case hit `hidden_state_fwd_h` twice, expected because backward recomputes fwd state.
- each case hit `hidden_state_bwd_dhu` once.
- all hidden-state hits had `K>256`, matching the Phase A true large-K path.

Post-run GPU:

- `0, NVIDIA GeForce RTX 3090, 24576 MiB total, 28 MiB used, util 0%, P8`.
- no compute process listed.

## 边界

- 未运行正式训练.
- 未运行新 benchmark.
- 未 push.
- 未写正式 docs/ledger/weekly slices.
- 未改 2080ti 主仓库.
- 未改 Flash-VQG.
