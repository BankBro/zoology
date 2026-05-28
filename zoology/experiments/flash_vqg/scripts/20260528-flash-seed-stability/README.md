# 20260528 Flash seed stability scripts

This directory records the reusable launch entrypoint for the strict official Flash seed stability补跑.

Scope:

- `data_seed=123`.
- `train_batch_size=64`, `eval_batch_size=16`, `gradient_accumulation_steps=4`.
- fp32 official/default, 4 epochs, early stopping disabled.
- `fox_remote_formula=gd_residual_v1`, `read_topk=2`, `write_topk=4`, `builder=grouped_chunk_torch_ref`, `pack_mode=semivec_ref`, `chunk_size=64`, `mu_min_count=0.1`.

Targets:

- `cb256-r4-s124`
- `cb256-r4-s125`
- `cb64-r16-s124`
- `cb64-r16-s125`

Usage:

```bash
cd /home/lyj/mnt/project/zoology
GPU_ID=0 bash zoology/experiments/flash_vqg/scripts/20260528-flash-seed-stability/run_train.sh cb64-r16-s124
GPU_ID=1 bash zoology/experiments/flash_vqg/scripts/20260528-flash-seed-stability/run_train.sh cb64-r16-s125
```

This script launches a new reproduction run. It does not rewrite the completed 2026-05-28 artifacts.

Recorded results:

- Report: `docs/20260528-flash-seed-stability-report.md`
- Formal artifact: `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv`
- Source manifest: `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-source-manifest.csv`
- Canonical ledger: `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`

Original completed launches:

- `flash-vqg-20260527-seed-stability-tmux-20260527T182923Z-cb256-r4-s124-2026-05-27-18-32-20`
- `flash-vqg-20260527-seed-stability-tmux-20260527T182923Z-cb256-r4-s125-2026-05-27-18-32-20`
- `flash-vqg-20260528-seed-stability-wave2-corrected-tmux-20260528T021416Z-cb64-r16-s124-2026-05-28-02-16-23`
- `flash-vqg-20260528-seed-stability-wave2-corrected-tmux-20260528T021416Z-cb64-r16-s125-2026-05-28-02-16-23`
