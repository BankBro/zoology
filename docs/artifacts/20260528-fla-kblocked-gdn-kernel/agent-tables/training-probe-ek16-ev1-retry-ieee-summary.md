# ek16-ev1 retry-ieee training probe summary

updated_at_utc: 2026-05-28T02:22:30Z

## Scope

This is tmp-only training probe readiness evidence for `true_expanded_k_gdn_training_probe_ek16_ev1_retry_ieee_tmp_only`. It is not official ledger evidence and not an upstream-ready correctness conclusion.

## Run

- Config: `ek16-ev1`, `H=2,K=1024,V=64`, seed=123, data_seed=123, b64_ga4, fp32, 4 epochs, no early stop, `GDN_KERNEL_DTYPE=float32`.
- Run id: `gdnxk-h2-ek16-ev1-s123-d123-b64-ga4-fp32-noearly4ep-retry-ieee-20260528T020346Z`.
- Launch id: `flash-vqg-20260528-kblocked-probe-ek16-ev1-retry-ieee-20260528T020346Z-2026-05-28-02-06-08`.
- Manifest status: `completed`.
- Started: `2026-05-28T02:06:15.162758+00:00`.
- Ended: `2026-05-28T02:18:36.417194+00:00`.
- Wall clock: `741.254436` sec.

## Final Valid Metrics

- `valid/loss`: 0.251.
- `valid/accuracy`: 0.974.
- `valid/input_seq_len/accuracy-1024`: 0.792.
- `valid/mqar_case/accuracy-1024x256`: 0.792.
- `valid/mqar_case/accuracy-512x128`: 0.998.
- `valid/mqar_case/accuracy-512x64`: 0.999.
- `valid/mqar_case/accuracy-256x64`: 1.0.

## Artifacts

- Log: `/home/lyj/mnt/project/worktrees/fla-kblocked/zoology/artifacts/fla-kblocked-kernel/training-probe-ek16-ev1-retry-ieee-20260528T020346Z/train.log`.
- Generated manifest: `/home/lyj/mnt/project/worktrees/fla-kblocked/zoology/zoology/experiments/flash_vqg/generated/flash-vqg-20260528-kblocked-probe-ek16-ev1-retry-ieee-20260528T020346Z-2026-05-28-02-06-08/manifest.json`.
- Checkpoint dir: `/home/lyj/mnt/project/worktrees/fla-kblocked/zoology/checkpoints/flash-vqg-20260528-kblocked-probe-ek16-ev1-retry-ieee-20260528T020346Z-2026-05-28-02-06-08/gdnxk-h2-ek16-ev1-s123-d123-b64-ga4-fp32-noearly4ep-retry-ieee-20260528T020346Z`.
- `best.pt`, `last.pt`, and `train_config.json` exist.
- SwanLab: `https://swanlab.cn/@scu-mclab/flash_vqg_gdn_expanded_k/runs/sxmswam493s14kgs8slm8`.

## Health

- No OOM, CUDA error, NaN/Inf, dtype fallback, shared-memory error, or training failure was observed before completion.
- After completion, the wrapper Python processes stayed alive with about 3.8 GiB GPU memory at 0 percent utilization. I sent Ctrl-C to the completed tmux session to release GPU memory.
- That cleanup added a post-completion `KeyboardInterrupt` traceback to the log. The generated manifest already recorded `completed`, so this is recorded as a cleanup artifact rather than a training failure.
- Post-cleanup GPU state: 28 MiB used, util 0%, P8, no compute process.

## Limits

- No second probe was launched.
- No fallback `ek8-ev2` was launched.
- No benchmark, push/commit, formal docs/ledger/weekly write, 2080ti main repo edit, or Flash-VQG official ledger write was performed.
