# 20260725-01 当前基线 Longer-MQAR: 3090

本目录由完整 `DONE.json` 后的审计 collector生成. `last.pt`为预注册主结果, `best.pt`为 epoch-end checkpoint敏感性结果.

本目录仅记录该机器上的原始正式结果和机器内统计. 跨机器比较与结论在上级 `combined/` 目录及正式 report 中给出.

主要文件:

- `training-final.csv`: 6条 epoch4训练和时间/dtype/GPU/checkpoint信息.
- `longer-mqar-detail.csv`: 60条 last/best逻辑 formal结果.
- `longer-mqar-summary.csv`: checkpoint role × model × slice三 seed汇总.
- `paired-deltas.csv`: 同 seed Flash-GDN paired delta和预注册分类.
- `checkpoint-role-comparison.csv`: best-last敏感性.
- `source-manifest.csv`: 12个逻辑角色的 checkpoint来源、hash和大小.
- `batch-sizes.csv`, `repro-verification.csv`, `verification.json`, `metadata.json`: 执行与审计证据.
- `raw-evidence-manifest.csv`: 状态、命令、日志、config及其ignored raw镜像路径和hash.
- `flash-ledger-rows.csv`, `gdn-ledger-rows.csv`: 等待主工作区统一写入 canonical ledger 的候选记录.

本目录只包含 `3090` 的机器级结果. 完整解释见 `docs/20260725-01-current-baselines-longer-mqar-report.md`. Raw输出保留在 `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260725-01-current-baselines-longer-mqar/outputs/machines/3090`. 本轮使用专用 collector直接生成统计, 未另跑 analysis suite, 因而没有 `zoology/analysis/flash_vqg/results/<launch_id>/` 目录.
