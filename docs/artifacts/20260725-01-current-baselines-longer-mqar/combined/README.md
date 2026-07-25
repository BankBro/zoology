# 跨GPU合并结果

本目录合并2080 Ti和3090的机器级正式artifact. 相同seed是跨GPU独立重训配对, 统计始终在每张GPU内按3个seed计算, 不合并为n=6.

- `longer-mqar-detail.csv`: 120条机器×模型×seed×checkpoint role×slice逻辑结果.
- `longer-mqar-summary.csv`: 每张机器内的三seed mean和population std.
- `paired-deltas.csv`: 每张机器内Flash-GDN同seed差值.
- `cross-machine-deltas.csv`: 同模型、seed、role、slice的3090-2080Ti差值.
- `verification.json`: 行数、唯一键、dataset hash和机器artifact hash审计.

验收结果为120条唯一逻辑结果、12条训练、24条source manifest和60条跨机器delta. 五个formal dataset hash在两机一致. `verification.json`还记录了Flash/GDN canonical ledger各新增3条3090记录; 图位于上级`figures/`目录.
