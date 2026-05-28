# FLA K-blocked GDN kernel evidence bundle

This artifact preserves the important evidence from the 3090 temporary K-blocked worktrees before those worktrees and the temporary conda environment are removed.

It is not a canonical ledger entry, not a weekly report, and does not include checkpoints or full swanlog directories.

Key files:

- `fla-kblocked-gdn-kernel-final.csv`: compact result and comparison table.
- `fla-kblocked-gdn-kernel-source-manifest.csv`: copied evidence and original raw path index.
- `raw-3090/`: selected 3090 result, training, correctness, and generated config artifacts.
- `source-snapshots/`: FLA, zoology, and Flash-VQG branch/status/diff snapshots.
- `env/`: conda, pip, and import source snapshots.
- `agent-tables/`: pass/no-go, patch manifest, and earlier correctness/training summaries.

Reusable script entrypoints are maintained in `zoology/experiments/flash_vqg/scripts/20260528-fla-kblocked-gdn-kernel/`. Any `.sh` or generated `launch_configs.py` files under `raw-3090/` are preserved as historical evidence, not as maintained source code.

Standard generated launch configs were also copied back to `zoology/experiments/flash_vqg/generated/` so the repository follows the usual experiment layout. The copies under `raw-3090/` remain the original evidence snapshots from the 3090 worktree.

Main conclusion at preservation time:

- `ek8-ev2-kblocked` improves over `ek16-ev1-kblocked`, which suggests the `K=1024,V=64` shape is a poor fit for this training setup.
- Both K-blocked runs remain far below Flash longer-MQAR OOD performance.
- The kernel remains research-stage. Upstream-ready correctness is still no-go because strict fp32 `K=1024,V=64` gradient tolerance is not fully satisfied.
