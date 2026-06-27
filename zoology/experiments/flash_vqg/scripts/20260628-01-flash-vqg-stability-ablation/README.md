# 20260628-01 Flash-VQG Stability Ablation

本目录服务于 `docs/plans/20260628-01-flash-vqg-stability-ablation-plan.md`.

本轮定位是 diagnostic / exploratory:

- 复用 `20260627-02` 的 canonical MQAR cache.
- 复用 `20260627-02` 的 canonical init checkpoint.
- 只改 `embed_dropout=0.0`.
- 跑 `2080ti x1 + 3090 x2` 的 1 epoch screen.
- 主指标是 `valid/mqar_case/accuracy-1024x256`.
- 进入稳定训练后按 20 分钟低频轮询, 避免持续刷日志.

不写 official MQAR ledger.
