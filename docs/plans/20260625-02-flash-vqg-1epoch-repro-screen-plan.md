# 20260625-02 Flash-VQG 1-Epoch Repro Screen Plan

updated: 2026-06-25
status: launched; container-side main queues running; stable-training gate satisfied
experiment_id: `20260625-02-flash-vqg-1epoch-repro-screen`

## 目标

本轮目标不是直接给出 `4 epoch` final basin 结论, 而是先用更便宜的 `1 epoch` same-card repeat screen 回答:

```text
1. 同机同卡同 seed 到 1 epoch 时是否稳定.
2. 跨机器的 seed 排序是否已经在 1 epoch 时发生翻转.
3. early-window 异常是否能延续到 1 epoch 附近.
```

本轮定位是 diagnostic / exploratory, 不写 official ledger。

## 非目标

- 不在本轮里做 `4 epoch x 2 repeat` 主结论矩阵.
- 不在本轮里引入 `2080ti GPU1`.
- 不在本轮里重新展开 hard04 / cap0405 / caprel0406late.
- 不在 reproducibility 尚未过关时回到 guard 设计.

## 预检

两机都必须先跑 preflight:

- 记录 `machine`, `hostname`, `driver`, `torch`, `cuda`, `cudnn`, branch, commit.
- 用与正式训练同一 builder 口径确认:
  - `train_batches=2815`
  - `gradient_accumulation_steps=4`
  - `num_optimizer_steps=704`
- preflight 不通过则本轮中止.

执行约束:

- 默认执行路径必须是目标机器的常驻 `Flash-VQG-tun` 容器。
- 启动任何需要 GPU 的 preflight, smoke 或 main queue 前, 必须在目标机器的 `Flash-VQG-tun` 容器内确认 `nvidia-smi` / NVML 和 `torch.cuda.is_available()` 均可用。
- 若容器内 NVML/CUDA 不可用, 必须提醒用户并暂停实验启动, 不得自动改用宿主机直接运行, 临时 `docker run --gpus`, host-side runner 或其他绕过路径。
- `run_repro_screen_host.sh` 只作为用户明确授权后的应急入口, 默认拒绝执行。
- 2026-06-25 早先因 `2080ti` 常驻容器 GPU runtime 失效曾临时使用 host-side runner; 用户随后要求暂停并重启容器。重启后 `2080ti` 容器内 `runtime_ready=true`, 后续 2080ti 主实验改回常驻容器路径。
- 为避免混用执行路径, 用户随后要求 3090 也重启。本轮两个早先 host-side queue 均已停止, partial trace 移入 `outputs/interrupted-traces/`, 后续正式 screen 只采用从 `Flash-VQG-tun` 容器启动的 `20260625T145800Z` queues。

## Smoke

两机都只跑:

- `smoke-default-s123`

配置:

- `max_train_steps=2`
- `max_validation_batches=1`
- `read_trace_train_steps=0,1,2`

smoke 成功标准:

- `config_runtime_smoke.py` 通过
- `early_window_metrics.jsonl` 已写出
- `train_step_0/1/2` trace 已落盘

## 主矩阵

两机都只用 `GPU0`, 单卡串行, 单进程, 单 config, fail-fast queue。

固定顺序:

1. `default-s123-r1`
2. `default-s124-r1`
3. `default-s123-r2`
4. `default-s124-r2`

公共配置:

- layout: `cb64-r16`
- `data_seed=123`
- `train_batch_order=global_shuffle`
- `train_batch_size=64`
- `eval_batch_size=16`
- `gradient_accumulation_steps=4`
- `max_epochs=1`
- `validations_per_epoch=4`
- `disable_early_stopping=true`
- `read_trace_enabled=true`
- `read_trace_valid_batches=441`
- `read_trace_train_steps=0,64,130,203,352,448,704`

## 退出会话条件

只有在以下条件同时满足后才允许退出会话:

```text
1. 2080ti queue 和 3090 queue 都已成功启动.
2. 两个 queue supervisor 均存活.
3. 两个 queue 的 queue-status.tsv 已写出全量 target.
4. 两台机器当前 active run 都已进入训练循环.
5. 当前 active run 日志持续追加, 无 Traceback/OOM/nan/inf.
6. 当前 active run 至少已写出 train_step_0, 优先等到 train_step_64.
7. 当前 active run 的 cache path 已从日志抽取, 可用于后续 sha256 审计.
```

## 产物

脚本目录:

```text
zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/
```

artifact:

```text
docs/artifacts/20260625-02-flash-vqg-1epoch-repro-screen/
```

脚本旁 raw / 临时 / 中断输出:

```text
zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/outputs/
```

其中暂停前的 2080ti host-side partial trace 归档为:

```text
zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/outputs/interrupted-traces/2080ti-hostrunner-20260625T135615Z/
```

暂停前的 3090 host-side partial trace 归档为:

```text
zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/outputs/interrupted-traces/3090-hostrunner-20260625T135615Z/
```

report:

```text
docs/20260625-02-flash-vqg-1epoch-repro-screen-report.md
```

collector 必须生成:

- `preflight-summary.csv`
- `machine-summary.csv`
- `run-summary.csv`
- `repeat-summary.csv`
- `step-window-summary.csv`
- `read-trace-summary.csv`
- `cache-hash-summary.csv`
- `invalid-runs.csv`
- `source-manifest.csv`
- `metadata.json`

## 判读规则

主判据:

- 同机同 seed `repeat_gap <= 0.02` 记为 stable.
- 同机同 seed `repeat_gap > 0.02` 记为 unstable.

后续分流:

```text
1. 若 same-card repeat 已 unstable:
   先查 nondeterminism / runtime / launch-state, 不进 4 epoch.

2. 若 same-card stable, 但两机 seed 排序相反:
   主线转 machine/GPU/runtime robustness, 不进 guard.

3. 若 same-card stable, 且两机 seed 排序一致:
   再进入更小规模 4 epoch confirm run.
```
