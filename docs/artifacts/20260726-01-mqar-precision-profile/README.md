# 20260726-01 MQAR precision profile artifact

本目录保存双机 MQAR 低精度训练与长度泛化实验的轻量可审计产物. 大型 checkpoint, progress, per-sample raw result 和日志保留在各 source machine 的实验脚本 `outputs/` 下.

预期收尾文件:

- `final.csv`.
- `source-manifest.csv`.
- `metadata.json`.
- `machines/2080ti/` 与 `machines/3090/` 的轻量 evidence.
- `combined/` 的 matching dtype 与 off-diagonal 汇总.
- `figures/` 下分开的 last 与 best 图.

当前状态: 自动化实现与 smoke 进行中, 尚无正式结果. 首轮 pre-formal 在 canary 数据口径审计处 fail-fast, formal 从未启动; 旧commit产物已归档为双机 `outputs/invalidated-80483073-canary-generated-data/`. 标准 canary 已改为复用 checkpoint 原始 validation cache, 严格相等单事件验证通过, 双机将从新commit完整重跑全部 gate.
