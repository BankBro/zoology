# Flash-VQG readk 边界历史审计计划

## 基本信息

- 审计 ID: `20260622-01-flash-vqg-readk-boundary-audit`
- 日期: 2026-06-22
- 类型: 历史 artifact 审计, 不启动新训练
- 目标问题: 判断 fixed `read_topk=4` 的既有正反证据是否足够支撑下一步实验决策, 并列出真正需要补训的缺口.

## 背景

`20260622-flash-vqg-seed-stability-roadmap.md` 把 read-side 方向定位为 P0/P1 重点: fixed `read_topk=4` 在 `cb256-r4` 和 `cb256-r8` 上有正证据, 但在 `cb64-r16` 和 `cb128-r8` 上有反例. 用户明确希望先使用历史数据做边界复核, 不重复运行已经跑过的实验.

## 审计范围

主要读取:

- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-key-metrics.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-spread-summary.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-final.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-spread-summary.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-source-manifest.csv`
- `docs/artifacts/20260530-gd-seed-diag/metadata.json`
- `docs/20260530-gd-seed-diag-report.md`
- `docs/20260605-flash-vqg-stability-direction-independent-review.md`
- `docs/20260605-flash-vqg-stability-research-direction-report.md`
- `docs/plans/20260622-flash-vqg-seed-stability-roadmap.md`

## 输出

- `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-final.csv`
- `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-spread-summary.csv`
- `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-gap-table.csv`
- `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-source-manifest.csv`
- `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/metadata.json`
- `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/README.md`
- `docs/20260622-01-flash-vqg-readk-boundary-audit-report.md`

## 审计判据

- fixed readk4 若要升级为局部候选, 至少需要同一 layout 下低 spread, worst repeat 稳定, 且不出现明显 high-path damage.
- fixed readk4 若在某 layout 出现 reproducibility collapse 或稳定伤害 high path, 该 layout 不得用来支持 fixed readk4 默认化.
- 历史证据如果缺少 seed123, best-final gap 或 early read margin/churn telemetry, 只标为 formalization 缺口, 不自动要求立即重训.

## 预期结论形态

- 可以回答: 现有数据是否足够支持“fixed readk4 不是全局默认”.
- 可以回答: 现有数据是否足够支持“cb256-like layout 上 fixed readk4 是局部正候选”.
- 必须列出: 若后续要进入正式 P0 formalization, 最小补训集合是什么.
