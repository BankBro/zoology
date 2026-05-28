# Shared-Memory Repair Attempt Summary

updated_at_utc: 2026-05-27T21:08:55Z
gate: `kblocked_shared_memory_repair_stage`
status: failed_waiting_controller_review

## Attempt

- Modified only the 3090 FLA isolated worktree source file `fla/ops/common/chunk_delta_h.py` in this gate.
- Repair was the minimal shared-memory reduction: large-K backward kernel launch `num_stages=3 -> 2`.
- Removed the temporary untracked backup file created during repair staging, so FLA source dirty range is back to the two previously authorized files.

## Checks

- `git diff --check`: passed.
- `py_compile` for FLA files: passed.
- `py_compile` for the existing zoology wrapper file: passed.
- Validation script imported the kblocked worktree modules before entering CUDA correctness.

## Failure

Required fp32 correctness case `K=1024,V=64,H=2,T=128`, with `initial_state` provided and `output_final_state=True`, hit the true large-K path and did not reproduce the shared-memory error. However, gradient correctness failed under the validation threshold:

- `dk`: max_abs `0.008891`, tolerance `0.005`.
- `dbeta`: max_abs `0.009246`, tolerance `0.005`.

Output, final_state, `dq`, `dv`, `dg`, and `dh0` passed the same absolute threshold. After the correctness failure, I stopped as required and did not run K<=256 regression sanity or `GatedDeltaNetExpandedK` fp32 smoke.

## Post State

- 3090 GPU returned idle: 28 MiB used, utilization 0%, no compute process.
- No training retry, new benchmark, push, formal docs/ledger/weekly write, or 2080ti main repo edit was performed.
- Current blocker: controller review needed before any further repair or validation.
