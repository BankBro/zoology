# Flash-VQG readk 边界历史审计 artifact

## 基本信息

- Artifact ID: `20260622-01-flash-vqg-readk-boundary-audit`
- 创建日期: 2026-06-22
- 类型: 历史 artifact 审计
- 是否启动新训练: 否
- 主指标: `valid/mqar_case/accuracy-1024x256`
- 主要来源: `docs/artifacts/20260530-gd-seed-diag/`

## 文件说明

| 文件 | 说明 |
|---|---|
| `readk-boundary-final.csv` | run-level 审计表, 覆盖 `cb256-r4`, `cb64-r16`, `cb128-r8`, `cb256-r8` 的 completed readk2/readk4 历史结果. |
| `readk-boundary-spread-summary.csv` | spread-level 汇总, 直接标记 fixed readk4 的正例, 反例和 rerun 风险. |
| `readk-boundary-gap-table.csv` | 缺口表, 区分 qualitative boundary 已足够和 formalization 仍需补训的部分. |
| `readk-boundary-source-manifest.csv` | 本审计使用的源文件清单. |
| `metadata.json` | 审计元数据, 主要结论和已知缺口. |

## 主要结论

1. 现有历史数据已经足够支持: fixed `read_topk=4` 不是全局默认解.
2. `cb256-r8` 是 fixed readk4 最强局部正例: readk2 s124/s125 为 `0.988/0.804`, spread `0.184`; readk4 completed runs 为 `0.982/0.982/0.988/0.992`, spread `0.010`.
3. `cb256-r4` 也是局部正例: ordinary readk2 r1 s124/s125 为 `0.772465/0.956375`, spread `0.183910`; formal readk4 completed runs 为 `0.943/0.958/0.944`, spread `0.015`.
4. `cb64-r16` 是 high-path damage 反例: readk4 s124 r1/r2 为 `0.831/0.849`, 而 readk2 s124 replacement 是 `0.959`.
5. `cb128-r8` 是 reproducibility 反例: readk4 main pair 为 `0.973/0.972`, 但 s125 rerun 掉到 `0.609`.
6. 因此下一步不应立即大规模重跑 readk2/readk4; 只有要把 `cb256-r4/r8` fixed readk4 升级为正式三 seed 局部候选时, 才需要补最小 targeted runs.

## 是否需要补训

不需要补训即可成立的结论:

- fixed readk4 不能作为全局默认.
- read-side control 值得继续机制化研究.
- cb64-r16 和 cb128-r8 必须保留为 fixed readk4 的反例或风险点.

只有 formalization 时才建议补训:

- `cb256-r8 readk4 s123`, 用来补齐最强正例的三 seed 表.
- `cb256-r4 readk4 s123`, 用来补齐 cb256-r4 局部正例的三 seed 表.
- `cb128-r8 readk4 s123` 或 `cb128-r8 readk4 s125` 追加 repeat, 仅当需要更完整地估计失败频率时运行.
- `cb256-r8 readk2 s123` 只在要求严格 same-wave readk2/readk4 baseline 时运行; 旧 capacity sweep 中的 s123 baseline 只能作为背景, 不应混成同一张 formal boundary 表.

## 注意事项

- 本审计从历史 validation curve 重构 `best_hard_1024x256` 和 `best_final_gap`; 未来实验应在 artifact 中直接记录 best 和 final.
- 历史表缺少 read margin, read entropy, candidate churn, per-head coverage 等 telemetry; 因此 read-side 机制仍应写成候选解释, 不应写成已完全证明.
- `cb64-r16/readk2/s124/r1` 曾 OOM, 本审计使用有效 replacement `r1b=0.959`.
- 大型原始日志, checkpoint 和 swanlog 仍按原 artifact 约定原位保留.
