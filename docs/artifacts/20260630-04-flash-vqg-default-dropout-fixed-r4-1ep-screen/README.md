# 20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen

本 artifact 收尾 `seed=124` default-dropout fixed-r4 1 epoch screen, 并包含 2080ti fixed-r2 supplemental baseline. 本轮是 diagnostic / exploratory screen, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `max_epochs=1`, `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.

主跨机器 variant 是 `fixed-r4`; `fixed-r2-baseline` 只作为 2080ti 单机 supplemental read_topk baseline.

## 结果摘要

主指标是 `valid/mqar_case/accuracy-1024x256`.

| machine | variant | read_topk | final 1024x256 | best 1024x256 |
|---|---|---:|---:|---:|
| 2080ti | `fixed-r4` | 4 | 0.284 | 0.284 |
| 3090 | `fixed-r4` | 4 | 0.135 | 0.135 |
| 2080ti | `fixed-r2-baseline` | 2 | 0.877 | 0.877 |

`fixed-r4` cross-machine gap 是 `14.9pp`, 超过 4pp 容忍线, 因此本轮不支持直接进入 default-dropout `fixed-r4` 4 epoch confirm. `fixed-r2-baseline` 只有 2080ti 一条, 只能作为下一步补 3090 同口径 baseline 的依据.

## 文件

- `run-summary.csv`: per-run final/best metrics.
- `variant-summary.csv`: fixed-r4 的 2080ti/3090 成对结果.
- `cross-machine-comparison.csv`: fixed-r4 的 1024x256 cross-machine gap.
- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.
- `queue-summary.csv`: queue 状态.
- `invalid-runs.csv`: failed/interrupted/pending run.
- `source-manifest.csv`: mirrored raw evidence 路径和 sha256.
- `metadata.json`: 收尾元数据.

注: `best_*` 是日志观测到的 best validation metric, 不是 saved-best checkpoint 复评. 本轮只回答 default dropout 下 1 epoch 是否值得继续 4 epoch confirm。
