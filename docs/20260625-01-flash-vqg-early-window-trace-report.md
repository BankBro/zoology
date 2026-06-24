# 20260625-01 Flash-VQG Early-Window Trace Report

状态: planned / running.

本轮是 diagnostic / exploratory, 不写 official ledger。训练完成并收集 summary 后, 本报告需要回答:

1. `s123` 在 3090/2080ti 是否都低.
2. `s124` 在 3090/2080ti 是否都高.
3. low/high basin 最早在哪个 train step 分叉.
4. 分叉先体现在 `update_norm_p95/max`, uncapped write pressure, `m_norm`, lambda/inject, 还是 read-side.
5. `hard04` 是否压低 early update pressure.
6. `hard04` 是否降低 pressure 但仍未解除 read-side lock-in.
7. 是否已有足够证据进入下一轮 guarded cap release.

当前结论: 等待 P0 smoke 和 Wave 1 训练产物。
