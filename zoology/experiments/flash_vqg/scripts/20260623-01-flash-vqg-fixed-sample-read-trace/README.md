# Flash-VQG fixed-sample read trace

Plan:

```text
docs/plans/20260623-01-flash-vqg-fixed-sample-read-trace-plan.md
```

This experiment records fixed validation sample/query read candidates as JSONL. It traces validation batch `441`, the first `1024x256` hard-slice batch under eval batch size 16. Raw trace files are written under:

```text
zoology/experiments/flash_vqg/scripts/20260623-01-flash-vqg-fixed-sample-read-trace/outputs/traces/
```

Run one target:

```bash
GPU_ID=0 bash zoology/experiments/flash_vqg/scripts/20260623-01-flash-vqg-fixed-sample-read-trace/run_read_trace_train.sh cb256r8-readk2-s125-trace
```

Launch the 3090 wave:

```bash
bash zoology/experiments/flash_vqg/scripts/20260623-01-flash-vqg-fixed-sample-read-trace/launch_wave_tmux.sh wave1-3090
```

Collect after completion:

```bash
python zoology/experiments/flash_vqg/scripts/20260623-01-flash-vqg-fixed-sample-read-trace/collect_results.py
```
