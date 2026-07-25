# 20260725-01 当前基线 Longer-MQAR

本目录由完整 `DONE.json` 后的审计 collector生成. `last.pt`为预注册主结果, `best.pt`为 epoch-end checkpoint敏感性结果.

主结果显示, Flash在 `1024x256` 不支持领先; 四个真正外推 slice中, 三个为 3/3 seeds稳健领先, `8190x512`为均值领先但 2/3 seeds的混合领先. `best.pt`敏感性在全部四个外推 slice为 3/3 seeds稳健领先. 主要方差来源是 Flash seed124.

主要文件:

- `training-final.csv`: 6条 epoch4训练和时间/dtype/GPU/checkpoint信息.
- `longer-mqar-detail.csv`: 60条 last/best逻辑 formal结果.
- `longer-mqar-summary.csv`: checkpoint role × model × slice三 seed汇总.
- `paired-deltas.csv`: 同 seed Flash-GDN paired delta和预注册分类.
- `checkpoint-role-comparison.csv`: best-last敏感性.
- `source-manifest.csv`: 12个逻辑角色的 checkpoint来源、hash和大小.
- `batch-sizes.csv`, `repro-verification.csv`, `verification.json`, `metadata.json`: 执行与审计证据.
- `figures/`: 当前两模型 `last.pt` 三 seed Longer-MQAR曲线的 PDF/PNG/SVG、绘图数据.

完整解释见 `docs/20260725-01-current-baselines-longer-mqar-report.md`. Raw输出保留在 `zoology/experiments/flash_vqg/scripts/20260725-01-current-baselines-longer-mqar/outputs/`. 本轮使用专用 collector直接生成统计, 未另跑 analysis suite, 因而没有 `zoology/analysis/flash_vqg/results/<launch_id>/` 目录.
