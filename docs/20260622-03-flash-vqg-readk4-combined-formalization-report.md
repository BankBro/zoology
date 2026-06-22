# Flash-VQG readk4 combined formalization 报告

updated: 2026-06-22

## 摘要

本次没有启动新训练, 只是把 `20260622-01` 的历史 readk4 审计和 `20260622-02` 新补的 seed123/telemetry 结果合并. 目的是给 fixed `read_topk=4` 一个当前可执行的判断: 哪些配置可以当局部候选, 哪些配置必须继续当反例或风险样例.

结论很直接: `cb256-r8 fixed readk4` 是当前最强局部候选; `cb256-r4 fixed readk4` 也可以进入局部候选池; `cb128-r8 fixed readk4` 有明确 rerun/path instability; fixed readk4 不能作为全局默认.

## 来源

- plan: `docs/plans/20260622-03-flash-vqg-readk4-combined-formalization-plan.md`
- artifact: `docs/artifacts/20260622-03-flash-vqg-readk4-combined-formalization/`
- 输入:
  - `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-final.csv`
  - `docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/final.csv`
  - `docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/readk4-context-summary.csv`

## Combined 结果

| config | combined values | worst | spread | 当前判断 |
|---|---|---:|---:|---|
| `cb256-r8` | `s123=0.992`, `s124=0.982/0.982`, `s125=0.988/0.992` | 0.982 | 0.010 | strong local candidate |
| `cb256-r4` | `s123=0.965`, `s124=0.943/0.958`, `s125=0.944` | 0.943 | 0.022 | local candidate |
| `cb128-r8` | `s124=0.973`, `s125=0.972/0.609/0.967` | 0.609 | 0.364 | rerun/path instability |
| `cb64-r16` | `s124=0.831/0.849`, `s125=0.965` | 0.831 | 0.134 | counterexample |

## 判断

`cb256-r8 fixed readk4` 现在证据最强. 历史 rows 已经显示 s124/s125 全部 high path, 新补的 s123=`0.992` 也在 high path, combined worst=`0.982`. 这足够把它放进 cb256-like 局部候选池.

`cb256-r4 fixed readk4` 也可以进入局部候选池. 它的 combined worst=`0.943`, 低于 `cb256-r8`, 但没有出现 collapse, 新补的 s123=`0.965` 也比历史 rows 更高. 它可以作为次级 cb256-like candidate.

`cb128-r8 fixed readk4` 不能进候选池. 同一 seed=125 在 fixed readk4 下出现了 `0.972/0.609/0.967`: 这说明它不是必坏, 但 rerun/path instability 明确存在. 我们已核对 `0.609` 那条历史 run 的 raw log, manifest 和 checkpoint `train_config.json`; available resolved config 显示它确实是 `cb128-r8`, `read_topk=4`, `seed=125`, `data_seed=123`, 而不是简单跑成 `cb256`, `readk2` 或其他 seed. 同时本轮新 repeat 的 `m_norm_max=13.8`, 不符合之前对 official 候选的 state health 红线. 所以更准确的表述是: `cb128-r8/readk4` 是 rerun/path instability case, 不是 clean positive.

`cb64-r16 fixed readk4` 继续作为反例. 历史 s124 两次 readk4 是 `0.831/0.849`, 而 readk2 replacement s124 是 `0.959`, 说明 fixed readk4 会伤某些 high path.

因此 fixed readk4 的最终当前定位是: **cb256-like 局部候选, 不是全局默认.**

## 对后续实验的影响

当前不需要马上重跑 readk4 大矩阵. 如果只是为了决定 fixed readk4 的位置, 这个 combined artifact 已经足够.

下一步更值得做的是补机制指标:

- candidate churn.
- early-step read margin / read entropy / selected mass.
- step `0,130,203,352,353,448,705` 的 lambda, inject ratio, zeta, M norm.

这些指标补上后, 再做 margin-aware read gate 或 read/write confidence gate. 继续盲目扩大 fixed readk4 矩阵的价值不高.

## Caveat

- 这是 combined evidence, 不是 strict same-wave full matrix.
- 历史 rows 没有本轮新增 read telemetry, 所以 telemetry 解释只适用于 `20260622-02` 新 rows.
- `cb128-r8/readk4/s125-r2=0.609` 已核对 raw log, manifest 和 train_config, 没发现简单配置错跑; 但这不等于完全排除历史代码 runtime bug.
- 本分析没有更新 official ledger.
