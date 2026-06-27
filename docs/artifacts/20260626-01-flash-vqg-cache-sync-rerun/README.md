# 20260626-01-flash-vqg-cache-sync-rerun

本 artifact 记录 3090 使用 2080ti canonical cache 的 r1-r4 1-epoch screen, 并与 2080ti canonical r1-r4 做跨机器对照。

## Cache

- canonical cache 来源: 2080ti 当前 13 个 `data_*.pt`.
- 3090 旧 cache quarantine: `data/flash_vqg/quarantine-20260626-01-cache-sync-rerun-20260626-133919/`.
- content-level cache 验证: `13/13` match=true.
- 3090 r1-r4 日志实际加载 cache 集合均匹配 canonical: `True`.

## 3090 Canonical r1-r4

- run 数: 8, completed: 8.
- 错误扫描: not found for Traceback/RuntimeError/OOM/loss=nan/loss=inf.
- 两批队列合计训练 wall time: 501.1 min.

## 关键表

- `canonical-3090-run-summary.csv`: 3090 canonical r1-r4 明细.
- `canonical-cross-machine-run-summary.csv`: 2080ti 与 3090 canonical run-level 对照.
- `canonical-cross-machine-repeat-summary.csv`: 每台机器, 每个 seed 的 r1-r4 mean/gap/std.
- `canonical-cross-machine-seed-summary.csv`: seed 级跨机器均值与稳定性对照.
- `noncanonical-vs-canonical-comparison.csv`: 3090 旧非 canonical cache 与新 canonical cache 的 run-level 对比.
- `source-manifest.csv`: 已镜像轻量 evidence 的来源, 大小和 sha256.

## 注记

`r3/r4` 补跑使用同 seed/config 的训练入口并通过 `base_target` 映射到 `r1` 入口, 但 `run_id`, `launch_id`, `trace_output_dir` 和 queue target 均标记为 `r3/r4`. 因此它们是 same-seed independent rerun, 用于观察训练数值路径稳定性, 不是不同 seed.
