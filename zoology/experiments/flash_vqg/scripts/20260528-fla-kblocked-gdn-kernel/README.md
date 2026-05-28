# 20260528 FLA K-blocked GDN kernel scripts

This directory holds reusable entrypoints for the FLA K-blocked GDN research line.

The experiment evidence is stored under:

- `docs/artifacts/20260528-fla-kblocked-gdn-kernel/`
- `docs/artifacts/longer-mqar/kblocked-gdn-20260528/`

These scripts are not an instruction to rerun experiments. They require the 3090 K-blocked worktree and conda env, or an equivalent future environment with the K-blocked FLA patch applied.

Expected 3090 environment:

- zoology worktree: `/home/lyj/mnt/project/worktrees/fla-kblocked/zoology`
- conda env: `/home/lyj/miniconda3/envs/flash-vqg-kblocked`
- FLA branch used for evidence: `codex/fla-kblocked-gdn-kernel`

Training probe wrappers:

- `run_ek16_ev1_training_probe.sh`: reproduces the true `K=1024,V=64` training probe shape.
- `run_ek8_ev2_training_probe.sh`: reproduces the true `K=512,V=128` training probe shape.
- `run_kblocked_training_probe.sh`: shared implementation used by the wrappers.

Run only from inside the target 3090 `Flash-VQG-tun` container, for example:

```bash
cd /home/lyj/mnt/project/worktrees/fla-kblocked/zoology
zoology/experiments/flash_vqg/scripts/20260528-fla-kblocked-gdn-kernel/run_ek8_ev2_training_probe.sh
```

Current caveat:

- This kernel is research-stage. `K=1024,V=64,fp32` still exceeds strict `abs_tol=0.005` on `dk/dbeta`, while passing FLA-style tolerance.
- Do not use these scripts for official ledger runs unless a new gate explicitly approves that.
