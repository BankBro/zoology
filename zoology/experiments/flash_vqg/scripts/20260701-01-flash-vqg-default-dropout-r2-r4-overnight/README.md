# 20260701-01 Flash-VQG default-dropout r2/r4 overnight

本目录是 default-dropout `fixed-r2` / `fixed-r4` overnight diagnostic 的实验入口。

核心入口:

- `default_dropout_r2_r4_overnight.py`: 生成配置, preflight, train, collect.
- `run_default_dropout_r2_r4_overnight_queue.sh`: 单机容器内队列执行.
- `start_default_dropout_r2_r4_overnight_queue.sh`: 后台启动队列.

本轮先跑 `p0-3090-fixed-r2`. 通过后再启动 fixed-r2 4ep paired confirm; 不通过则启动 bounded diagnostic queue.

所有完整正式 MQAR 结果是否写 ledger 由收尾报告决定; probe/失败/中断 run 不写 official ledger.
