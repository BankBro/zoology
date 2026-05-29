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

## 阶段 3 当前结果

- `local-only` 已完成正式训练, manifest status `completed`, final `last.pt` 已保存.
- 训练 wrapper 在 SwanLab finalization 后未自然退出, 在 manifest 完成后 SIGTERM 释放 GPU. 本地 `backup.swanlab` 校验失败, remote analysis 成功.
- final validation: overall `0.404470`, `1024x256=0.000262`, `512x128=0.000219`, `512x64=0.025156`, `128x32=0.203969`, `64x16=1.0`.
- 当前观察: exact local-only 训练可以解决短配置, 但不能解决长 MQAR. 这进一步支持 Flash 的长距离优势不是由 64-token local exact window 单独解释.
- `local1` 已完成正式训练和 local analysis, manifest status `completed`, final `last.pt` 已保存.
- final validation: overall `0.917821`, `1024x256=0.508379`, `512x128=0.877281`, `512x64=0.977297`, `128x32=0.997844`, `64x16=0.999563`.
- 当前观察: 将 local window 从 2 blocks 缩到 1 block 后, 训练后的 remote-on 模型仍显著强于 `local-only`, 说明当前长距离能力主要依赖 remote/VQ/GD residual 路径, 不是 local exact attention 单独造成.
- `local4` 已完成正式训练和 local analysis, manifest status `completed`, final `last.pt` 已保存.
- final validation: overall `0.967501`, `1024x256=0.758496`, `512x128=0.986875`, `512x64=0.996828`, `128x32=0.999969`, `64x16=1.0`. Best validation peak 为 `0.991211`, 高于 final, 对应 `best.pt`.
- 当前观察: 放大 local window 到 4 blocks 明显提升训练速度和 final 长配置表现, 但 final `1024x256` 仍低于早期 best peak. 这提示 local exact attention 会影响训练动态和中短距离样本, 但 `local-only` 与 `local1/local4` 的差距仍说明 remote/VQ/GD residual 是长距离能力的必要路径.
