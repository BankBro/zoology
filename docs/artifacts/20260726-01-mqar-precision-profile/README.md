# 20260726-01 MQAR precision profile artifact

本目录保存双机 MQAR 低精度训练与长度泛化实验的轻量可审计产物. 大型 checkpoint, progress, per-sample raw result 和日志保留在各 source machine 的实验脚本 `outputs/` 下.

预期收尾文件:

- `final.csv`.
- `source-manifest.csv`.
- `metadata.json`.
- `machines/2080ti/` 与 `machines/3090/` 的轻量 evidence.
- `combined/` 的 matching dtype 与 off-diagonal 汇总.
- `figures/` 下分开的 last 与 best 图.

当前状态: 自动化实现与 smoke 进行中, 尚无正式结果.
