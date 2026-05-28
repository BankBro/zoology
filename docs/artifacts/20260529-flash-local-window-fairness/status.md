# 20260529 Flash local window fairness status

- 2026-05-28T19:13:53.033928+00:00: 阶段 3 artifact 初始化. 三个正式训练 ablation 均为 planned.

下一步: 校验 stage3 builder, 提交脚本, 然后在 3090 GPU 上先启动 `local-only`.

- 2026-05-28T19:17:13.870993Z: `local-only` 正式训练已启动, launch_id `flash-vqg-20260529-local-window-fairness-stage3-localonly-2026-05-28-19-16-28`, run_id `gd-cb64-r16-s123-localonly-d123-b64-ga4-fp32-noearly4ep`, supervisor PID `52535`, log `tmp/20260529-flash-local-window-fairness-stage3-logs/local-only-20260528T191626Z.log`.

- 2026-05-28T20:55:53.297719Z: `local-only` 训练完成. manifest status `completed`, ended_at `2026-05-28T20:43:43.168439+00:00`, wall_clock_sec `5228.583029`. Wrapper 在 SwanLab finalization 后未自然退出, 已在 manifest 完成和 final checkpoint 写入后 SIGTERM 释放 GPU. local analysis 因 `backup.swanlab` checksum invalid 失败, remote analysis 成功. Final valid accuracy `0.40447021484375`, `1024x256` `0.00026171875`, `512x128` `0.00021875`, `64x16` `1.0`.
