# 20260624-02 Flash-VQG pressure telemetry guard artifact

本 artifact 记录第一阶段 telemetry 补齐与 config-to-runtime smoke 结果.

本阶段没有启动完整 MQAR 训练, 没有实现 guarded release. 目标只是确认新增 pressure telemetry 能从 runtime 传出, 且 2080ti 和 3090 在同一代码提交下都能通过 smoke.

## 文件

| 文件 | 用途 |
|---|---|
| `smoke-summary-2080ti.json` | 2080ti 本地 smoke 原始轻量 summary |
| `smoke-summary-3090.json` | 3090 smoke 原始轻量 summary, 已镜像回 2080ti |
| `smoke-machine-summary.csv` | 双机设备, 状态, 输出路径摘要 |
| `smoke-case-summary.csv` | 每个 smoke case 的核心 telemetry 摘要 |
| `source-manifest.csv` | source path, mirror path, sha256 |
| `metadata.json` | commit, 状态和归档说明 |

## 结论

- 2080ti 和 3090 均通过 smoke.
- `update_norm_mean/p95/max` 可传出.
- `update_norm_cap_active/effective_cap/hit_ratio` 可传出, 且低 cap case 能触发正 hit ratio.
- write cap 的 `effective_cap`, `scheduled_cap`, `release_progress` 可传出.
- 本阶段只说明观测链路可用, 不说明 guard 策略已经有效.
