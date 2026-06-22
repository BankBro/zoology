# Flash-VQG readk 边界历史审计报告

## 摘要

这次审计没有启动新训练, 只整理 `20260530-gd-seed-diag` 及后续研究报告中已有的 fixed readk2/readk4 证据. 审计结论是: 历史数据已经足够支持 fixed `read_topk=4` 不是全局默认解, 也足够支持 `cb256-r4/cb256-r8` 是 readk4 的局部正例. 但如果要把 `cb256-r4/cb256-r8` fixed readk4 写成正式三 seed 候选, 仍需要补 seed123 和更完整的 telemetry.

本审计输出位于:

- `docs/plans/20260622-01-flash-vqg-readk-boundary-audit-plan.md`
- `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/`
- `docs/20260622-01-flash-vqg-readk-boundary-audit-report.md`

## 关键结论

1. **不需要重跑就能确认的结论:** fixed readk4 不能作为全局默认. `cb64-r16` 和 `cb128-r8` 已经给出明确反例.
2. **readk4 的正证据是局部的:** `cb256-r8` readk4 all-completed spread 是 `0.010`, `cb256-r4` formal readk4 spread 是 `0.015`.
3. **cb128-r8 是最高优先级风险点:** readk4 main pair 是 `0.973/0.972`, 但 s125 rerun 是 `0.609`, 说明 main pair 不能单独作为稳定证据.
4. **cb64-r16 说明 readk4 会伤 high path:** s124 在 readk2 replacement 下是 `0.959`, 在 readk4 r1/r2 下是 `0.831/0.849`.
5. **下一步不是大扫参:** 如果要补训, 应做 targeted formalization, 不是重复跑所有旧矩阵.

## 审计表

### Run-level

完整表见 `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-final.csv`.

| 配置 | 条件 | 已审计 completed rows | 核心数值 |
|---|---:|---:|---|
| `cb256-r4` | readk2 | 2 | s124=`0.772465`, s125=`0.956375` |
| `cb256-r4` | readk4 | 3 | s124-r1c=`0.943`, s124-r2c=`0.958`, s125-r1c=`0.944` |
| `cb64-r16` | readk2 | 2 | s124-r1b=`0.959`, s125-r1=`0.915` |
| `cb64-r16` | readk4 | 3 | s124-r1=`0.831`, s124-r2=`0.849`, s125-r1=`0.965` |
| `cb128-r8` | readk2 | 2 | s124=`0.956`, s125=`0.956` |
| `cb128-r8` | readk4 | 3 | s124=`0.973`, s125-r1=`0.972`, s125-r2=`0.609` |
| `cb256-r8` | readk2 | 2 | s124=`0.988`, s125=`0.804` |
| `cb256-r8` | readk4 | 4 | s124-r1=`0.982`, s124-r2=`0.982`, s125-r1=`0.988`, s125-r2=`0.992` |

### Spread-level

完整表见 `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-spread-summary.csv`.

| 配置 | 对比 | spread | 审计判断 |
|---|---|---:|---|
| `cb256-r4` | readk2 r1 cross seed | `0.183910` | fail, ordinary readk2 path-sensitive |
| `cb256-r4` | readk4 formal all completed | `0.015` | pass_local |
| `cb64-r16` | readk4 main cross seed | `0.134` | fail, high-path damage |
| `cb64-r16` | readk4 s124 rerun | `0.018` | fail_reproduced, 两次都低 |
| `cb128-r8` | readk4 main cross seed | `0.001` | initial_pass_only |
| `cb128-r8` | readk4 all completed | `0.364` | fail, rerun collapse |
| `cb256-r8` | readk2 main cross seed | `0.184` | fail |
| `cb256-r8` | readk4 all completed | `0.010` | pass_local |

## 与 roadmap 的关系

`docs/plans/20260622-flash-vqg-seed-stability-roadmap.md` 中把 P0 read-side 任务拆成 `cb128-r8` rerun triage 和 `cb256-r4/r8` readk2 vs readk4 formalization. 这次审计说明:

- roadmap 的 qualitative 判断已经被历史数据支持: fixed readk4 只应作为局部候选, 不能作为全局默认.
- `cb256-r8` 的正证据强, 但还不是严格三 seed formalization, 因为 fixed-readk4 表里缺 s123.
- `cb256-r4` 的正证据也强, 但 formal readk4 表里同样缺 s123.
- `cb128-r8` 的反例足以阻止 fixed readk4 默认化, 但如果要统计失败频率, 还需要追加 repeat 或补 s123.

因此, 当前不建议重跑所有旧实验. 更合适的是先把后续问题收窄成两类:

1. 如果目标是写清楚 readk4 适用边界, 目前历史审计已经足够.
2. 如果目标是把 cb256-like readk4 推成正式候选, 再补最小 targeted runs.

## 最小补训建议

### 不启动新训练也可以继续推进的部分

- 把 fixed readk4 正反边界写入后续 read-side plan.
- 设计 read schedule 或 margin-aware gate 前, 先补 instrumentation 需求清单.
- 后续报告统一把 `cb64-r16` 和 `cb128-r8` 放进 fixed readk4 反例表, 防止只引用 `cb256-r8` 正例.

### 只有 formalization 时才补训

| 优先级 | 建议 run | 目的 |
|---|---|---|
| P0 | `cb256-r8 readk4 s123` | 补齐最强正例的三 seed fixed-readk4 表. |
| P0 | `cb256-r4 readk4 s123` | 补齐 cb256-r4 formal readk4 三 seed 表. |
| P0 optional | `cb256-r8 readk2 s123` | 只在要求 strict same-wave baseline 时运行. |
| P0 optional | `cb128-r8 readk4 s123` | 完整三 seed risk map. |
| P0 diagnostic | `cb128-r8 readk4 s125` repeat | 估计 rerun collapse 的频率, 需要增强 telemetry. |

## 机制层面的限制

这次审计支持的是边界判断, 不是完整机制证明. 目前仍缺:

- read top1-top2 margin.
- read entropy.
- candidate churn.
- remote candidate coverage proxy.
- per-head/per-code read usage.
- step-level early split telemetry.

所以后续不应把结论写成“readk4 已证明解决 read-side basin lock-in”. 更准确的写法是: fixed readk4 在 cb256-like 配置上强烈支持 read-side 候选覆盖不足这个解释, 但 cb64/cb128 反例说明 fixed topk 本身不是稳定机制, 后续需要 schedule/gate/margin-aware 控制.

## 最终判断

当前审计满足 `20260622-01-flash-vqg-readk-boundary-audit` 的目标. 现有历史数据足以停止“是否要重跑全部 readk2/readk4 旧矩阵”的争论: 不需要. 下一步应先决定目标是写清边界, 还是正式化 cb256-like readk4 候选. 只有后者需要少量 targeted 补训.
