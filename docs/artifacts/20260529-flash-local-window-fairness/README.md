# 20260529 Flash local window fairness formal artifacts

本目录保存 Flash local window fairness 的正式训练阶段 artifact.

当前阶段:

- 阶段 0-2 eval-only 诊断结果保存在 `docs/artifacts/longer-mqar/local-window-fairness-20260529/`.
- 阶段 3 计划在 3090 上顺序训练 cb64-r16 seed123 的三个最小 ablation: `local-only`, `local1`, `local4`.
- `local-only`: `local_num_blocks=2`, `if_remote_enabled=false`.
- `local1`: `local_num_blocks=1`, `if_remote_enabled=true`.
- `local4`: `local_num_blocks=4`, `if_remote_enabled=true`.

记录规则:

- 正式训练 3090 GPU 独占, 不并跑正式训练.
- 训练完成到 final checkpoint 后才标记 `completed`.
- checkpoint path, manifest path, analysis path, swanlog path 和日志路径必须写入 `train_runs.csv`.
- 失败, 中断, smoke, diagnostic 不写入正式结果表, 只在 status 或独立记录中说明.

创建时间: 2026-05-28T19:13:53.033928+00:00
