# 20260625-01 Flash-VQG Early-Window Trace

本目录是 `20260625-01-flash-vqg-early-window-trace` 的脚本入口。本轮是 diagnostic / exploratory, 不写 official ledger。

## 目标

- 在 `cb64-r16`, `data_seed=123`, `read_topk=2` 下, 精确按 optimizer step `0,64,130,203,352,448,705` 触发 fixed-sample read trace.
- 比较 `s123 default`, `s123 hard04`, `s124 default`, 并做 `s123/s124 default` cross-machine repeat.
- 本轮不跑 `cap0405`, `caprel0406late`, `s124-hard04`, `s125`.

## P0 Smoke

2080ti:

```bash
MACHINE_NAME=2080ti bash zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/start_early_trace_queue.sh 2080ti-smoke
```

3090 容器内:

```bash
MACHINE_NAME=3090 bash zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/start_early_trace_queue.sh 3090-smoke
```

smoke 成功后用 `config_runtime_smoke.py` 检查 `early_window_metrics.jsonl` 和 `train_step_*/read_trace.jsonl`。

## Wave 1

2080ti:

```bash
MACHINE_NAME=2080ti bash zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/start_early_trace_queue.sh 2080ti-wave1
```

3090 容器内:

```bash
MACHINE_NAME=3090 bash zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/start_early_trace_queue.sh 3090-wave1
```

Wave 1 排布:

| 机器 | GPU | target | 目的 |
|---|---:|---|---|
| 3090 | 0 | `default-s123` | P1 low seed baseline |
| 3090 | 0 | `hard04-s123` | P1 low seed pressure-control |
| 3090 | 0 | `default-s124` | P2 high seed cross-machine repeat |
| 2080ti | 0 | `default-s124` | P1 healthy seed baseline |
| 2080ti | 1 | `default-s123` | P2 low seed cross-machine repeat |

## 收集

训练完成后:

```bash
python zoology/experiments/flash_vqg/scripts/20260625-01-flash-vqg-early-window-trace/collect_early_trace_results.py
```

大型 raw trace, checkpoints, swanlog 默认不提交。轻量 summary 写入 `docs/artifacts/20260625-01-flash-vqg-early-window-trace/`。
