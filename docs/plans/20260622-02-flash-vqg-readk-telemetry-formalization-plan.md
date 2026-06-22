# Flash-VQG read-side telemetry + readk4 formalization 计划

## 基本信息

- 实验 ID: `20260622-02-flash-vqg-readk-telemetry-formalization`
- 日期: 2026-06-22
- 分支: `flash-vqg`
- 目标机器: 2080ti 做实现和 smoke, 3090 并行跑完整训练
- 状态: planned

## 目标

本轮不是重跑 readk2/readk4 全矩阵, 而是补齐两个缺口:

1. 在 Flash-VQG phase2 residual read 中新增最小 read-side scalar telemetry.
2. 用带 telemetry 的 targeted runs 补 fixed readk4 在 `cb256-r4/r8` 上的 seed123 formalization 缺口, 同时追加 `cb128-r8/readk4/s125` 反例 repeat.

## 新增 telemetry

新增指标只做观测, 不改变模型语义:

- `attn/gd_residual_read_margin_top1_top2_mean`
- `attn/gd_residual_read_margin_top1_top2_p05`
- `attn/gd_residual_read_entropy_mean`
- `attn/gd_residual_read_selected_mass_mean`
- `attn/gd_residual_read_selected_mass_p05`

暂不实现严格 `candidate_churn`, 因为它需要跨 forward 缓存 candidate id, 容易引入状态污染.

## 运行矩阵

### 2080ti smoke

| target | 配置 | 目的 |
|---|---|---|
| `smoke-readk4-cb256r8-s123` | `cb256-r8`, seed `123`, fixed readk4, very short | 验证新增 telemetry 出现在 train/valid logs, 且 artifact extraction 可抽取. |

### 3090 full runs

| target | 配置 | 目的 |
|---|---|---|
| `cb256r8-readk4-s123` | `cb256-r8`, seed `123`, fixed readk4, 4ep | 补齐最强 readk4 正例的三 seed formalization. |
| `cb256r4-readk4-s123` | `cb256-r4`, seed `123`, fixed readk4, 4ep | 补齐 cb256-r4 formal readk4 三 seed 表. |
| `cb128r8-readk4-s125-repeat` | `cb128-r8`, seed `125`, fixed readk4, 4ep | 带 telemetry 复查 cb128 rerun collapse 风险. |

## 通过与停止条件

smoke 通过条件:

- train 和 valid 日志中出现新增 read-side telemetry.
- `attn/gd_residual_remote_read_topk_effective=4`.
- 没有 NaN/inf.
- `git diff --check` 通过.

full run 启动条件:

- 2080ti smoke 通过.
- 3090 的 zoology 和 Flash-VQG 已同步到本轮代码.
- 3090 最多并行 3 条 run.

full run 记录要求:

- 保存 generated config, manifest, raw log path, checkpoint path, status.
- artifact 汇总 final, best, best-final gap, validation history, source manifest.
- 同时报告 historical outcome evidence 和本轮 instrumented mechanism evidence, 不把单条 telemetry 写成完整因果证明.
