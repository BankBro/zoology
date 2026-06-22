# Flash-VQG readk4 combined formalization artifact

## 基本信息

- Artifact ID: `20260622-03-flash-vqg-readk4-combined-formalization`
- 创建日期: 2026-06-22
- 类型: combined evidence analysis
- 是否启动新训练: 否
- 主指标: `valid/mqar_case/accuracy-1024x256`

## 文件说明

| 文件 | 说明 |
|---|---|
| `combined-readk4-final.csv` | 历史 readk4 rows 和 20260622-02 新 rows 的合并 run-level 表. |
| `combined-readk4-summary.csv` | 每个 config 的 combined final 值, spread, risk 和解释. |
| `combined-readk4-decision.csv` | 当前决策表, 包含局部候选, 风险样例和全局默认判断. |
| `candidate-pool.csv` | 简化候选池表, 方便后续报告引用. |
| `metadata.json` | 来源, 结论和 caveat. |
| `README.md` | 本说明文件. |

## 输入来源

```text
docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/readk-boundary-final.csv
docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/final.csv
docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/readk4-context-summary.csv
```

## 主要结论

| scope | 当前判断 | 依据 |
|---|---|---|
| `cb256-r8 fixed readk4` | strong local candidate | 历史 `s124/s125=0.982/0.982/0.988/0.992`, 新补 `s123=0.992`, combined worst=`0.982`. |
| `cb256-r4 fixed readk4` | local candidate | 历史 `s124/s125=0.943/0.958/0.944`, 新补 `s123=0.965`, combined worst=`0.943`. |
| `cb128-r8 fixed readk4` | boundary risk | 历史有 `s125=0.609` collapse, 新 repeat `0.967` 没复现, 但新 run `m_norm_max=13.8`. |
| `cb64-r16 fixed readk4` | counterexample | 历史 `s124=0.831/0.849`, 低于 readk2 replacement `s124=0.959`. |
| global fixed readk4 | reject | `cb64-r16` 和 `cb128-r8` 反例/风险仍成立. |

## 候选池

当前可以进入局部候选池:

- `cb256-r8 fixed readk4`
- `cb256-r4 fixed readk4`

当前不能进入候选池:

- `cb128-r8 fixed readk4`
- `cb64-r16 fixed readk4`
- global fixed readk4 default

## Caveat

- 这是 combined evidence, 不是 strict same-wave full matrix.
- 历史 rows 没有新增 read telemetry; telemetry 只来自 `20260622-02`.
- candidate churn 还没有实现.
- 本 artifact 不更新 official ledger.
