# Eval-only 补强状态

- 2026-05-29T03:58:54+00:00 启动 stage3 best/last longer-MQAR bucket eval 和 near-distance enriched eval 串行任务.
- 2026-05-29T05:15:25+00:00 stage3 longer-MQAR bucket eval 完成.
- 2026-05-29T06:32:34+00:00 初版 near-distance enriched eval 完成, 但 33-64 和 65-128 使用固定上边界代表距离, 不作为最终 near 结论.
- 2026-05-29T07:03:54+00:00 更新 near generator 为 33-64/65-128 桶内均匀 odd-distance 采样后, 启动 near rerun.
- 2026-05-29T08:22:22+00:00 near-distance enriched rerun 完成, 作为最终 near 结果.
