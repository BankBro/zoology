# True Triton K-blocked Phase A CUDA correctness

updated_at_utc: 2026-05-27T20:02:24Z
owner: fla-kblocked-kernel
gate: `true_triton_kblocked_cuda_correctness_phaseA`
status: passed

## 结论

Phase A true Triton large-K hidden-state path CUDA correctness 通过. 共 11 个 case:

- extended: `K=512,V=128` 和 `K=1024,V=64`, 各覆盖 `initial_state none/provided` x `output_final_state false/true`.
- regression: `K=64/128/256,V=64` 旧 Triton path.

所有 case 均使用独立 `naive_recurrent_gated_delta_rule` reference, 对齐 output, final_state, `dq`, `dk`, `dv`, `dbeta`, `dg`, `dh0`. 未出现 Triton compile error, OOM, CUDA error, NaN/Inf, dtype fallback, Traceback/RuntimeError, import 来源错误或超容忍差异.

## 环境

| 项目 | 值 |
|---|---|
| worktree | `/home/lyj/mnt/project/worktrees/fla-kblocked/flash-linear-attention` |
| branch | `codex/fla-kblocked-gdn-kernel` |
| commit | `19b5a3f411ecea6cdda62c6cc65cdae55ed2dec5` |
| dirty | ` M fla/ops/common/chunk_delta_h.py`; ` M fla/ops/gated_delta_rule/chunk.py` |
| diff stat | `2 files changed, 327 insertions(+), 2 deletions(-)` |
| device | NVIDIA GeForce RTX 3090 |
| torch | 2.6.0+cu118 |
| CUDA | 11.8 |
| dtype | torch.float16 |
| Triton note | 3.2.0 below recommended 3.3.0, recorded as JIT risk |

Import source sanity passed for `fla`, `fla.ops.common.chunk_delta_h`, and `fla.ops.gated_delta_rule.chunk`; all point to the kblocked FLA worktree.

## Overall Maxima

| tensor | max_abs | max_rel_to_ref_max |
|---|---:|---:|
| output | 7.7248e-05 | 8.3720e-04 |
| final_state | 3.5697e-04 | 5.7761e-04 |
| dq | 3.0518e-05 | 1.5835e-03 |
| dk | 9.7656e-04 | 1.0488e-03 |
| dv | 1.9531e-03 | 9.5147e-04 |
| dbeta | 1.9581e-03 | 3.6758e-03 |
| dg | 6.8598e-04 | 1.7183e-03 |
| dh0 | 1.2249e-05 | 1.1407e-03 |

Max torch allocated across cases: 275.6934 MiB.

## Runtime Notes

These times are correctness/JIT observations, not benchmark data.

- Total script wall time: 90.6507s.
- First `K=512,V=128` large-K case included Triton JIT: 30.998047s; later K=512 cases were about 1.19-1.38s.
- First `K=1024,V=64` large-K case included Triton JIT: 28.818362s; later K=1024 cases were about 0.93-0.99s.
- Regression `K=64/128/256` old path took 6.677368s, 6.652138s, and 6.152516s.

## GPU Snapshots

| snapshot | value |
|---|---|
| pre-run nvidia-smi | `0, NVIDIA GeForce RTX 3090, 24576, 28, 0, P8` |
| post-run nvidia-smi | `0, NVIDIA GeForce RTX 3090, 24576, 28, 0, P8` |
| post-run compute process | none |

## Boundary

This validates Phase A numerical correctness for the true Triton large-K source draft. It is not a kernel benchmark and does not authorize zoology smoke, training probe, push, or formal docs/ledger/weekly writes.
