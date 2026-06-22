# Flash-VQG readk4 combined formalization 计划

created: 2026-06-22

## 目标

本计划只做分析整理, 不启动新训练. 目标是合并两个已有 artifact:

- `20260622-01-flash-vqg-readk-boundary-audit`
- `20260622-02-flash-vqg-readk-telemetry-formalization`

最终回答 fixed `read_topk=4` 当前应该如何归类:

- `cb256-r8` 是否进入局部候选池.
- `cb256-r4` 是否进入局部候选池.
- `cb128-r8` 是否仍是边界风险样例.
- fixed readk4 是否可以作为全局默认.

## 输入

使用以下现有文件:

```text
docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-final.csv
docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/final.csv
docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/readk4-context-summary.csv
```

## 输出

Artifact 路径:

```text
docs/artifacts/20260622-03-flash-vqg-readk4-combined-formalization/
```

输出文件:

- `combined-readk4-final.csv`: 合并后的 run-level readk4 表.
- `combined-readk4-summary.csv`: 每个 config 的汇总和当前解释.
- `combined-readk4-decision.csv`: 决策表, 包含局部候选和全局默认判断.
- `candidate-pool.csv`: 简化候选池表.
- `metadata.json`: 来源, 状态, caveat.
- `README.md`: artifact 说明.

报告路径:

```text
docs/20260622-03-flash-vqg-readk4-combined-formalization-report.md
```

## 判断规则

- `cb256-r8`: 如果 combined readk4 包含 seed `123/124/125`, 且 worst final hard 不低于 `0.95`, 判为强局部候选.
- `cb256-r4`: 如果 combined readk4 包含 seed `123/124/125`, 且没有低于 `0.90` 的 collapse, 判为局部候选.
- `cb128-r8`: 如果历史 `0.609` collapse 和本轮 `0.967` repeat 同时存在, 判为边界风险样例, 不进入候选池.
- `cb64-r16`: 保留为 high-path damage 反例.
- `global_default`: 只要 `cb64-r16` 或 `cb128-r8` 保留反例/风险, fixed readk4 不能作为全局默认.

## 校验

- 输入 CSV 可正常解析.
- `cb256-r8` 汇总包含 seed `123/124/125`.
- `cb256-r4` 汇总包含 seed `123/124/125`.
- `cb128-r8` 汇总同时包含 `0.609` 和 `0.967`.
- `metadata.json` 通过 `python3 -m json.tool`.
- `git diff --check` 通过.

## 注意事项

- 这是 combined evidence, 不是 strict same-wave full matrix.
- 历史 rows 没有本轮新增 read telemetry.
- 本 artifact 不更新 official ledger.
- 不启动训练, 不修改 Flash-VQG 代码.
