# 20260625-02 Flash-VQG 1-Epoch Repro Screen Report

## 1. 状态

本轮实验 `20260625-02-flash-vqg-1epoch-repro-screen` 定位为 diagnostic / exploratory, 不写 official ledger。

当前状态:

- zoology: pending
- Flash-VQG: pending
- dtype policy: default torch/zoology runtime dtype; no explicit AMP, bf16, or fp16 override in launch configs
- execution path: host-side runner container on both machines
- preflight: in progress
- smoke: pending
- main queues: pending

## 2. 目标

本轮要回答:

1. `2080ti GPU0` 上 `s123` / `s124` 的 `1 epoch` repeat 是否稳定.
2. `3090 GPU0` 上 `s123` / `s124` 的 `1 epoch` repeat 是否稳定.
3. 两机在 `1 epoch` 时的 seed 排序是否一致.
4. early-window 异常是否能延续到 `1 epoch` 附近.

## 3. 执行矩阵

主矩阵:

| machine | gpu | target |
|---|---:|---|
| 2080ti | 0 | `default-s123-r1` |
| 2080ti | 0 | `default-s124-r1` |
| 2080ti | 0 | `default-s123-r2` |
| 2080ti | 0 | `default-s124-r2` |
| 3090 | 0 | `default-s123-r1` |
| 3090 | 0 | `default-s124-r1` |
| 3090 | 0 | `default-s123-r2` |
| 3090 | 0 | `default-s124-r2` |

公共配置:

- layout: `cb64-r16`
- `data_seed=123`
- `train_batch_order=global_shuffle`
- `train_batch_size=64`
- `eval_batch_size=16`
- `gradient_accumulation_steps=4`
- `max_epochs=1`
- `validations_per_epoch=4`
- `read_trace_train_steps=0,64,130,203,352,448,704`

## 4. 结果

待 `collect_repro_screen_results.py` 汇总后回填:

- `preflight-summary.csv`
- `run-summary.csv`
- `repeat-summary.csv`
- `step-window-summary.csv`
- `read-trace-summary.csv`
- `cache-hash-summary.csv`

## 5. 判读

待回填:

- same-card repeat 是否 stable
- cross-machine seed 排序是否 flip
- 是否建议进入 `4 epoch` confirm run

## 6. 备注

当前已确认的环境现象:

- `2080ti` 宿主机 `nvidia-smi` 正常, 但常驻 `Flash-VQG-tun` 容器内 `nvidia-smi` 报 `Failed to initialize NVML: Unknown Error`, `torch.cuda.is_available() == False`, `device_count == 0`.
- `3090` 常驻容器内 `nvidia-smi` 与 `torch.cuda` 当前正常。
- 两台机器使用同镜像 `flash-vqg-tun-snapshot:0.1` 新起 runner 容器时, `torch.cuda` 都正常。

本报告必须在训练完成后补齐:

- final `1024x256` 精度表
- repeat gap 表
- cache hash 对照
- source manifest 审计说明
