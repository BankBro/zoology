# 20260529 Flash local window fairness scripts

This directory contains eval-only tooling for the first stages of
`20260529-flash-local-window-fairness`.

Primary entrypoint:

```text
/home/lyj/miniconda3/envs/flash-vqg/bin/python local_window_bucket_eval.py prepare-sources
/home/lyj/miniconda3/envs/flash-vqg/bin/python local_window_bucket_eval.py init-readme
/home/lyj/miniconda3/envs/flash-vqg/bin/python local_window_bucket_eval.py eval-buckets --variants full
```

Use the `flash-vqg` conda environment. The system `/usr/bin/python3` on 3090
does not provide the project dependencies.

`prepare-sources` must generate `source_checkpoints.csv` from the official
longer-MQAR manifest before any bucket eval runs. Checkpoint paths must not be
hand-maintained.

`eval-buckets` computes MQAR distance from generation-time position metadata:
`query_pos - value_pos`. It does not recover positions by token lookup.

Eval-time override variants such as `local_only`, `local1`, and `local4` are
diagnostic only. They must not be mixed with formal training results.

## Stage 3 formal training ablation

Stage 3 uses the same cb64-r16 seed123 gd-residual training budget as the
existing official anchor, but changes only the Flash local/remote structure:

```text
GPU_ID=0 bash run_stage3_train.sh local-only
GPU_ID=0 bash run_stage3_train.sh local1
GPU_ID=0 bash run_stage3_train.sh local4
```

Targets:

- `local-only`: `local_num_blocks=2`, `if_remote_enabled=false`.
- `local1`: `local_num_blocks=1`, `if_remote_enabled=true`.
- `local4`: `local_num_blocks=4`, `if_remote_enabled=true`.

Formal runs must be launched one at a time on the 3090 GPU. Their checkpoint,
manifest, logs, and analysis output are recorded under
`docs/artifacts/20260529-flash-local-window-fairness/` after each run completes.
