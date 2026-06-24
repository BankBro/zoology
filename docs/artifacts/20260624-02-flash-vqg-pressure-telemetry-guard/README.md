# 20260624-02 Flash-VQG pressure telemetry guard artifact

本 artifact 记录 `20260624-02-flash-vqg-pressure-telemetry-guard` 的阶段 1 smoke 和阶段 2 最小 telemetry probe.

阶段 1 只验证新增 pressure telemetry 与 config-to-runtime 链路. 阶段 2 在 `cb64-r16` 上跑 `default`, `hard04`, `cap0405`, `caprel0406late` 的最小矩阵, 用来判断 guard 实现前应该观察哪些 pressure 指标. 本实验没有实现 guarded release, 也不是 official MQAR 结果.

## 文件

| 文件 | 用途 |
|---|---|
| `smoke-summary-2080ti.json` | 2080ti 本地 smoke 原始轻量 summary |
| `smoke-summary-3090.json` | 3090 smoke 原始轻量 summary, 已镜像回 2080ti |
| `smoke-machine-summary.csv` | 双机设备, 状态, 输出路径摘要 |
| `smoke-case-summary.csv` | 每个 smoke case 的核心 telemetry 摘要 |
| `source-manifest.csv` | 阶段 1 source path, mirror path, sha256 |
| `metadata.json` | 阶段 1 commit, 状态和归档说明 |
| `stage2-key-metrics.csv` | 阶段 2 关键结果与 pressure 指标摘要 |
| `stage2-run-summary.csv` | 阶段 2 有效 run 的完整摘要 |
| `stage2-variant-summary.csv` | 阶段 2 按 variant 聚合的 two-seed 摘要 |
| `stage2-invalid-runs.csv` | 阶段 2 被排除的无效 run 及原因 |
| `stage2-source-manifest.csv` | 阶段 2 generated/analysis/log 轻量 evidence 的 source/mirror/hash |
| `stage2-metadata.json` | 阶段 2 状态, 机器和归档说明 |

## 结论

- 2080ti 和 3090 均通过 smoke.
- `update_norm_mean/p95/max` 可传出.
- `update_norm_cap_active/effective_cap/hit_ratio` 可传出, 且低 cap case 能触发正 hit ratio.
- write cap 的 `effective_cap`, `scheduled_cap`, `release_progress` 可传出.
- 阶段 2 的 8 条有效训练均完成. 3090 跑 `s123`, 2080ti 跑 `s124`; 2080ti 上 `cap0405-s124` 首次因 GPU0 重叠占用 OOM, 已排除并单独 rerun 成功.
- `hard04` 在 `s123` 上把 final hard 从 `0.776699` 提到 `0.852586`, 同时把 `m_norm_max` 从 `7.122381` 压到 `3.451890`, 说明 write/state pressure 控制确实影响低 seed 轨迹.
- `cap0405` 和 `caprel0406late` 在本轮没有出现 `m_norm > 8` 或 `m_norm > 12` 红线, 但也没有把 `s123` 救到 `s124` 的高位.
- 本轮不能把问题简化成 “只防 m_norm 爆”. `s123 default` 最低, 但 `m_norm_max=7.122381`, 没过 12 的红线.
