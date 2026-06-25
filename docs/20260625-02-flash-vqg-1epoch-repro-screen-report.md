# 20260625-02 Flash-VQG 1-Epoch Repro Screen Report

## 1. 状态

本轮实验 `20260625-02-flash-vqg-1epoch-repro-screen` 定位为 diagnostic / exploratory, 不写 official ledger。

当前状态:

- zoology: 2080ti and 3090 host-side runs stopped; both machines restarted from persistent `Flash-VQG-tun` containers
- Flash-VQG: both machines are running commit `020c1d4` from their `Flash-VQG-tun` container paths
- dtype policy: default torch/zoology runtime dtype; no explicit AMP, bf16, or fp16 override in launch configs
- execution path: default path is target machine `Flash-VQG-tun` container; host-side runner now requires explicit user authorization
- preflight: passed on 3090; 2080ti host-side preflight passed earlier, container-side preflight passed after restart
- smoke: passed on 3090; 2080ti host-side smoke passed earlier, container-side smoke passed after restart
- main queues: previous host-side queues stopped before completion; container-side queues started with `RSCREEN_TIMESTAMP=20260625T145800Z`

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

- `2080ti` 宿主机 `nvidia-smi` 正常, 但重启前常驻 `Flash-VQG-tun` 容器内 `nvidia-smi` 报 `Failed to initialize NVML: Unknown Error`, `torch.cuda.is_available() == False`, `device_count == 0`.
- `3090` 常驻容器内 `nvidia-smi` 与 `torch.cuda` 当前正常。
- 两台机器使用同镜像 `flash-vqg-tun-snapshot:0.1` 新起 runner 容器时, `torch.cuda` 都正常。
- `2080ti` 宿主机用户 uid/gid 为 `1002:1002`, `3090` 为 `1001:1001`. runner 容器内需要显式注入 `USER` / `LOGNAME` 与 `TORCHINDUCTOR_CACHE_DIR`, 否则 `torch.compile` 侧的用户名解析会因为 `getpwuid(uid)` 无记录而失败。
- 主实验首次启动时默认 `LOGGER_BACKEND=swanlab`, 但 runner 容器内没有 cloud API key, 因此改为 `LOGGER_BACKEND=none` 重新启动主队列。该调整不影响本轮 diagnostic 所需的 trace, early-window metrics 和 final stdout metrics 抽取。
- 用户要求后续遵守容器 GPU 可用性硬门槛: 若 `Flash-VQG-tun` 容器内 NVML/CUDA 不可用, 必须提醒用户并暂停, 不得自动改用 host-side runner。
- 2080ti 常驻容器已重启并恢复 GPU runtime: `nvidia-smi` 可见两张 `NVIDIA GeForce RTX 2080 Ti`, `torch.cuda.is_available() == True`, `device_count == 2`。
- 用户决定采用严格方案: 3090 早先 host-side run 也停止, 后续 2080ti 和 3090 都从各自常驻 `Flash-VQG-tun` 容器重新启动主队列。

当前运行状态:

- commit: `020c1d4`
- 2080ti preflight: passed, output `outputs/2080ti-preflight-20260625T134932Z/`
- 2080ti container-side preflight after restart: passed, output `outputs/2080ti-container-preflight-20260625T144118Z/`
- 3090 preflight: passed, output `outputs/3090-preflight-20260625T135155Z/`
- 2080ti smoke: passed, output `outputs/2080ti-smoke-20260625T135329Z/`
- 2080ti container-side smoke after restart: passed, output `outputs/2080ti-smoke-20260625T144200Z/`
- 3090 smoke: passed, output `outputs/3090-smoke-20260625T135155Z/`
- 2080ti container main queue: `outputs/2080ti-gpu0-20260625T145800Z/`, tmux session `rscreen-2080ti-gpu0-20260625T145800Z`, active target `default-s123-r1`, pid `2285`
- 3090 container main queue: `outputs/3090-gpu0-20260625T145800Z/`, tmux session `rscreen-3090-gpu0-20260625T145800Z`, active target `default-s123-r1`, pid `3457013`
- 两台当前 active run 均已进入训练循环并写出 `train_step_0/read_trace.jsonl`
- 3090 `default-s123-r1` 已写出 `train_step_64/read_trace.jsonl`
- 2080ti `default-s123-r1` 已写出 `train_step_64/read_trace.jsonl`
- 2026-06-25 23:05 CST 检查时, 两台 tmux supervisor 均存活; 两台当前 active run 错误扫描未见 `Traceback`, `RuntimeError`, `CUDA out of memory`, `loss=nan`, `loss=inf` 或独立 `nan` / `inf` token
- 两台日志显示训练 loss 持续下降, 且数据均从 `./data/flash_vqg/*.pt` on-disk cache 读取

稳定阶段检查结论:

- 本轮设定的退出门槛为: queue supervisor 已启动, `queue-status.tsv` 已写入全量 target, 当前 active run 已进入训练循环, 无 traceback / OOM / nan / inf, 且至少写出 `train_step_0`, 优先等到 `train_step_64`
- 截至本次检查, 2080ti 与 3090 均满足上述门槛
- 因此当前从常驻 `Flash-VQG-tun` 容器启动的主实验已进入稳定阶段训练
- 2026-06-25 22:31 CST 左右, 2080ti 本次 host-side runner container `10e7799877c0` 已执行 `docker stop --time 30`; `host-runner.pid=3470811` 随后退出, 未清理输出目录
- 2080ti `queue-status.tsv` 仍保留暂停前的 `started` 行, 后续判读时应按本报告外部状态视为 interrupted / paused, 不是 completed
- 2080ti 暂停前的 partial trace 已从正式 trace 路径移入 `outputs/interrupted-traces/2080ti-hostrunner-20260625T135615Z/`, 只进入 source manifest 审计, 不进入正式 metrics 汇总
- 2080ti 重启后的 container-side smoke 已完成且没有残留训练进程
- 2026-06-25 22:55 CST 左右, 3090 早先 host-side runner container `da45fc19e440` 已执行 `docker stop --time 30`; tmux 会话 `hrscreen-3090-gpu0-20260625T135615Z` 已停止
- 3090 暂停前的 partial trace 已从正式 trace 路径移入 `outputs/interrupted-traces/3090-hostrunner-20260625T135615Z/`, 只进入 source manifest 审计, 不进入正式 metrics 汇总
- 2080ti 和 3090 主队列已从 container-side 路径重启, 本轮正式 screen 判读只使用 `20260625T145800Z` container queues; earlier host-side partials 仅作审计记录

本报告必须在训练完成后补齐:

- final `1024x256` 精度表
- repeat gap 表
- cache hash 对照
- source manifest 审计说明
