# 实验日志

本文件按时间追加关键实验进展. 详细配置和指标见对应 report 与 artifact.

## 1. 2026-07-24: Flash-VQG 效率优化完成

- `experiment_id`: `20260724-01-flash-vqg-gd-residual-efficiency`.
- 目的: 在不改变模型数学语义和超参数的前提下降低显存与运行时间.
- 结果: 优化通过等价性和正式质量回归; Flash-VQG 相对同量级 GDN 的核心时间与显存比值均不超过 `2x`.
- 输出: [报告](20260724-01-flash-vqg-gd-residual-efficiency-report.md), [artifact](artifacts/20260724-01-flash-vqg-gd-residual-efficiency/README.md).
- 下一步: 解决 GDN `ek4-ev4` 在 RTX 3090 上的 FLA kernel 兼容性.

## 2. 2026-07-24 至 2026-07-25: GDN FLA 兼容性闭环

- `experiment_id`: `20260724-02-gdn-ek4-fla-compatibility`.
- 目的: 解决 GDN `ek4-ev4` 在 RTX 3090 上的 shared-memory kernel 启动失败, 并确定双 GPU 共同环境.
- 结果: 选择官方 FLA 0.4.2; 2080 Ti 和 RTX 3090 的 production shape、正式 1ep 质量及完整 epoch 效率门槛全部通过.
- 输出: [报告](20260724-02-gdn-ek4-fla-compatibility-report.md), [artifact](artifacts/20260724-02-gdn-ek4-fla-compatibility/README.md).
- 下一步: 后续 Flash-VQG/GDN 实验统一使用 `flash-vqg-fla042` 环境和当前基线.
