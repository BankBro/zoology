# 20260625-02 Flash-VQG 1-Epoch Repro Screen

本目录是 `20260625-02-flash-vqg-1epoch-repro-screen` 的脚本入口。本轮是 diagnostic / exploratory, 不写 official ledger。

## 目标

- 在 `cb64-r16`, `data_seed=123`, `read_topk=2` 下, 用 `1 epoch` 层级做 same-card reproducibility screen。
- 在 `2080ti GPU0` 和 `3090 GPU0` 上分别跑 `s123 x2`, `s124 x2`。
- 每条 run 单独进程, 单卡串行, 不并发, 不跨卡混跑。
- 统一保留 trace steps `0,64,130,203,352,448,704`。

## 执行路径

默认执行路径是目标机器的常驻 `Flash-VQG-tun` 容器。

启动任何需要 GPU 的 preflight, smoke 或 main queue 前, 必须先在目标机器的 `Flash-VQG-tun` 容器内确认:

- `nvidia-smi` / NVML 可用
- `torch.cuda.is_available() == True`
- `torch.cuda.device_count() > 0`

`start_repro_screen_queue.sh`, `run_repro_screen_queue.sh` 和 `run_repro_screen_train.sh` 已包含容器内 GPU ready 检查。`run_repro_screen_host.sh` 只作为显式授权后的应急绕过入口, 默认拒绝执行; 只有用户知情并明确要求时才允许设置 `ALLOW_HOST_SIDE_RUNNER=1`。

## Preflight

在对应机器的 `Flash-VQG-tun` 容器内执行:

```bash
cd /home/lyj/mnt/project/zoology
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/outputs/${MACHINE_NAME}-container-preflight-${TS}
mkdir -p "${OUT}"
ZOOLOGY_REPO_ROOT=/home/lyj/mnt/project/zoology \
FLASH_VQG_ROOT=/home/lyj/mnt/project/Flash-VQG \
/home/lyj/miniconda3/envs/flash-vqg/bin/python \
  zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/preflight_repro_screen.py \
  --machine-name "${MACHINE_NAME}" \
  --mode train \
  --output-json "${OUT}/preflight-train.json"
```

其中 `MACHINE_NAME` 为 `2080ti` 或 `3090`。

preflight 必须确认:

- `train_batches=2815`
- `gradient_accumulation_steps=4`
- `num_optimizer_steps=704`
- `runtime_ready=true`

## Smoke

在对应机器的 `Flash-VQG-tun` 容器内执行:

```bash
cd /home/lyj/mnt/project/zoology
MACHINE_NAME=2080ti \
LOGGER_BACKEND=none \
PYTHON_BIN=/home/lyj/miniconda3/envs/flash-vqg/bin/python \
ZOOLOGY_REPO_ROOT=/home/lyj/mnt/project/zoology \
FLASH_VQG_ROOT=/home/lyj/mnt/project/Flash-VQG \
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/start_repro_screen_queue.sh \
  2080ti-smoke
```

```bash
cd /home/lyj/mnt/project/zoology
MACHINE_NAME=3090 \
LOGGER_BACKEND=none \
PYTHON_BIN=/home/lyj/miniconda3/envs/flash-vqg/bin/python \
ZOOLOGY_REPO_ROOT=/home/lyj/mnt/project/zoology \
FLASH_VQG_ROOT=/home/lyj/mnt/project/Flash-VQG \
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/start_repro_screen_queue.sh \
  3090-smoke
```

smoke 成功后用 `config_runtime_smoke.py` 检查 `early_window_metrics.jsonl` 和 `train_step_*/read_trace.jsonl`。

## Main Queues

在对应机器的 `Flash-VQG-tun` 容器内执行:

```bash
cd /home/lyj/mnt/project/zoology
MACHINE_NAME=2080ti \
LOGGER_BACKEND=none \
PYTHON_BIN=/home/lyj/miniconda3/envs/flash-vqg/bin/python \
ZOOLOGY_REPO_ROOT=/home/lyj/mnt/project/zoology \
FLASH_VQG_ROOT=/home/lyj/mnt/project/Flash-VQG \
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/start_repro_screen_queue.sh \
  2080ti-gpu0
```

```bash
cd /home/lyj/mnt/project/zoology
MACHINE_NAME=3090 \
LOGGER_BACKEND=none \
PYTHON_BIN=/home/lyj/miniconda3/envs/flash-vqg/bin/python \
ZOOLOGY_REPO_ROOT=/home/lyj/mnt/project/zoology \
FLASH_VQG_ROOT=/home/lyj/mnt/project/Flash-VQG \
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/start_repro_screen_queue.sh \
  3090-gpu0
```

两个 queue 都是固定顺序:

1. `default-s123-r1`
2. `default-s124-r1`
3. `default-s123-r2`
4. `default-s124-r2`

## 稳定后退出会话

允许退出会话的最小条件:

- 两个 tmux queue 都已启动且 supervisor 存活
- 两个 queue 的 `queue-status.tsv` 已写出全量 target
- 当前 active run 都已进入训练循环
- 当前 active run 无 traceback / OOM / nan / inf
- 当前 active run 至少落出 `train_step_0`, 并优先等到 `train_step_64`
- 当前 active run 的 cache path 已从日志抽取并可用于后续 hash 审计

## 收集

训练完成后:

```bash
python zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/collect_repro_screen_results.py
```

collector 会生成:

- `run-summary.csv`
- `repeat-summary.csv`
- `step-window-summary.csv`
- `read-trace-summary.csv`
- `cache-hash-summary.csv`
- `preflight-summary.csv`
- `source-manifest.csv`

轻量 summary 写入 `docs/artifacts/20260625-02-flash-vqg-1epoch-repro-screen/`。
