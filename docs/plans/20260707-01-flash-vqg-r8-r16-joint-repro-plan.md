# 20260707-01 Flash-VQG r8/r16 joint control same-seed stability rerun plan

## 目标

本轮只验证 `update softcap + injection warmup` 这条正信号的复现性, 不做新机制修改, 不开 D-geometry, 不开 read trace, 不跑 4ep 或多 seed.

实验分两批自动接续:

1. Batch R8: 三张卡同时跑 `r8-update-softcap0p5-injwarm512-rerun`.
2. Batch R16: R8 三个 run 都结束后, 自动启动三张卡同时跑 `r16-update-softcap0p5-injwarm512-rerun`.

三张卡为:

- 2080ti GPU0.
- 2080ti GPU1.
- 3090 GPU0.

## 固定条件

- seed: `124`.
- data_seed: `123`.
- canonical MQAR cache, 内容 hash 必须 match.
- canonical seed124 init checkpoint, tensor hash 必须 match.
- same batch order, hash 必须 match.
- default dropout:
  - `embed_dropout=0.1`.
  - `resid_dropout=0.0`.
  - `drop_path=0.0`.
- model: `cb64-r16`.
- `fox_gd_residual_write_topk=4`.
- `max_epochs=1`.
- `max_train_steps=704`.
- `grad_accumulation_steps=4`.
- `fox_gd_residual_update_norm_softcap=0.5`.
- `fox_gd_residual_update_norm_softcap_mode=smooth_p4`.
- residual injection warmup: optimizer step `0 -> 512`, 即 train-forward step `0 -> 2048`.
- formal runs 关闭 D-geometry, read trace, train-inline event trace 和 hash probe.

## 执行方式

新实验单元:

- script dir: `zoology/experiments/flash_vqg/scripts/20260707-01-flash-vqg-r8-r16-joint-repro/`.
- artifact dir: `docs/artifacts/20260707-01-flash-vqg-r8-r16-joint-repro/`.
- report: `docs/20260707-01-flash-vqg-r8-r16-joint-repro-report.md`.

脚本:

- `r8_r16_joint_repro.py`: 复用上一轮训练和 collect 逻辑, 只暴露 r8/r16 两个 target.
- `run_queue.sh`: 单个 GPU 跑单个 target.
- `run_master.sh`: 负责三卡 batch 并行和 R8 -> R16 自动接续.
- `start_master.sh`: 后台启动 master 并记录 PID/log.

启动前硬门槛:

1. 两机代码同步到同一 commit.
2. 两机容器内 `nvidia-smi` 通过.
3. 2080ti 容器内 `torch.cuda` 能看到 GPU0/GPU1.
4. 3090 容器内 `torch.cuda` 能看到 GPU0.
5. cache/init/batch order 全部 match.
6. 6 个 smoke run 全部通过.
7. smoke 出现 NaN, OOM, Traceback 或非零退出时, 不启动 formal.

Codex 退出会话前必须确认:

- 6 个 smoke run 全部 completed.
- formal master 已后台启动.
- formal R8 batch 已进入实际 train loop, 日志持续推进.
- master 脚本会在 R8 batch 结束后自动启动 R16 batch.
- `master.pid`, `master-status.tsv`, queue logs 和输出路径已明确记录.

## 判定标准

每个 variant 都有 3 个 final `1024x256` 结果.

| variant | pass 条件 |
|---|---|
| r8 | 三次 final `1024x256 >= 0.85`, 且 max-min gap `<=4pp` |
| r16 | 三次 final `1024x256 >= 0.85`, 且 max-min gap `<=4pp` |

报告需要额外比较:

- 2080ti GPU0 vs GPU1, 判断同机器同架构波动.
- 2080ti 两卡 vs 3090, 判断跨硬件波动.
- 与上一轮结果对比:
  - r8: `0.930/0.943`, gap `1.3pp`.
  - r16: `0.901/0.945`, gap `4.4pp`.

## 输出

收尾时生成:

- `run-summary.csv`.
- `batch-summary.csv`.
- `variant-stability-summary.csv`.
- `mechanism-metrics-summary.csv`.
- `cache-init-preflight-summary.csv`.
- `batch-order-summary.csv`.
- `master-status-summary.csv`.
- `source-manifest.csv`.
- `metadata.json`.
- `README.md`.

## 预期解释口径

- 如果 r8 三卡稳定, 下一步补 `r8-injwarm512-only`, 判断收益来自 joint control 还是 read support + warmup.
- 如果 r16 继续边界或失败, 暂不推进 r16 4ep.
- 如果 r8 也不稳定, 说明 joint control 仍不足, 下一步转向 support-aware / read confidence guard.
- 本轮“续训”指 master 自动启动下一个 run, 不是 checkpoint resume. 所有 formal run 都从同一 canonical init 开始.
