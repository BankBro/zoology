# 20260625-02 Flash-VQG 1-Epoch Repro Screen

本目录是 `20260625-02-flash-vqg-1epoch-repro-screen` 的脚本入口。本轮是 diagnostic / exploratory, 不写 official ledger。

## 目标

- 在 `cb64-r16`, `data_seed=123`, `read_topk=2` 下, 用 `1 epoch` 层级做 same-card reproducibility screen。
- 在 `2080ti GPU0` 和 `3090 GPU0` 上分别跑 `s123 x2`, `s124 x2`。
- 每条 run 单独进程, 单卡串行, 不并发, 不跨卡混跑。
- 统一保留 trace steps `0,64,130,203,352,448,704`。

## Host Runner

本轮默认通过 host-side runner 容器执行, 不直接复用常驻 `Flash-VQG-tun` 容器。

原因:

- `2080ti` 当前常驻容器内 `nvidia-smi` / `torch.cuda` 不可用, 但同镜像新起 runner 容器 GPU runtime 正常。
- `3090` 常驻容器当前是健康的, 但为了执行口径一致, 本轮两机统一走同一个 runner 入口。

入口脚本:

```bash
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/run_repro_screen_host.sh
```

## Preflight

2080ti:

```bash
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/run_repro_screen_host.sh \
  preflight 2080ti train 0
```

3090 宿主机:

```bash
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/run_repro_screen_host.sh \
  preflight 3090 train 0
```

preflight 必须确认:

- `train_batches=2815`
- `gradient_accumulation_steps=4`
- `num_optimizer_steps=704`
- `runtime_ready=true`

## Smoke

2080ti:

```bash
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/run_repro_screen_host.sh \
  queue 2080ti-smoke 0
```

3090 宿主机:

```bash
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/run_repro_screen_host.sh \
  queue 3090-smoke 0
```

smoke 成功后用 `config_runtime_smoke.py` 检查 `early_window_metrics.jsonl` 和 `train_step_*/read_trace.jsonl`。

## Main Queues

2080ti:

```bash
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/run_repro_screen_host.sh \
  queue 2080ti-gpu0 0
```

3090 宿主机:

```bash
bash zoology/experiments/flash_vqg/scripts/20260625-02-flash-vqg-1epoch-repro-screen/run_repro_screen_host.sh \
  queue 3090-gpu0 0
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
