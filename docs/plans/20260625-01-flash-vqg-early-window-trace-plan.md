# 20260625-01 Flash-VQG Early-Window Trace Plan

## Summary

本轮实验统一命名为 `20260625-01-flash-vqg-early-window-trace`, 定位是 diagnostic / exploratory, 不写 official ledger。目标不是实现 guard, 而是抓住 `cb64-r16` 下 `s123` low basin 与 `s124` high basin 在训练早期的分叉证据。

## Implementation

- `zoology` 增加 `read_trace_train_steps`, CLI 通过 `--read-trace-train-steps` 传入.
- `Trainer` 使用纯 optimizer step 计数触发 trace-only validation probe.
- `step 0` 定义为训练前 probe, 其余 step 定义为完成对应 optimizer step 后 probe.
- 每个 step 的 raw trace 写入 `read_trace_output_dir/train_step_<step>/read_trace.jsonl`.
- trace-only probe 不参与 checkpoint, early stopping, scheduler 或 official validation aggregate.
- Flash-VQG read trace schema 已够用, 默认不改 Flash-VQG 代码.

## Runs

P0 smoke:

- 2080ti GPU0: `smoke-default-s123`
- 3090 GPU0: `smoke-default-s123`
- trace steps: `0,1,2`
- validation batch: `441`
- `read_trace_max_samples=1`, `read_trace_max_queries_per_sample=1`

Wave 1:

| 机器 | GPU | run | 目的 |
|---|---:|---|---|
| 3090 | 0 | `s123 default` | P1 low seed baseline |
| 3090 | 0 | `s123 hard04` | P1 low seed pressure-control |
| 3090 | 0 | `s124 default` | P2 high seed cross-machine repeat |
| 2080ti | 0 | `s124 default` | P1 healthy seed baseline |
| 2080ti | 1 | `s123 default` | P2 low seed cross-machine repeat |

Wave 1 公共配置:

- layout: `cb64-r16`
- data_seed: `123`
- read_topk: `2`
- trace steps: `0,64,130,203,352,448,705`
- validation batch: `441`
- `read_trace_max_samples=4`
- `read_trace_max_queries_per_sample=8`
- `validations_per_epoch=4`

## Handoff

Wave 1 启动后, 等所有 run 进入稳定训练状态再退出会话。稳定状态要求进程存活, GPU 占用合理, log 持续追加, 无 traceback/OOM, 至少看到 `train_step_0` 和 `train_step_64` trace, 优先等到 `train_step_130`。

## Report Questions

- `s123` 在 3090/2080ti 是否都低.
- `s124` 在 3090/2080ti 是否都高.
- low/high basin 最早在哪个 train step 分叉.
- 分叉先体现在 update pressure, `m_norm`, lambda/inject, 还是 read-side.
- `hard04` 是否压低 early update pressure.
- `hard04` 是否降低 pressure 但未解除 read-side lock-in.
- 是否已有足够证据进入下一轮 guarded cap release.
