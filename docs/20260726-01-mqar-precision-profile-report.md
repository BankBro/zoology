# MQAR 低精度与长度泛化实验报告

## 1. 实验状态

状态: 实现与 smoke 验证进行中. 正式结果仅在双机 global gate 通过后写入本报告.

- 实验计划: [20260726-01-mqar-precision-profile-plan.md](plans/20260726-01-mqar-precision-profile-plan.md).
- 正式 artifact: [20260726-01-mqar-precision-profile](artifacts/20260726-01-mqar-precision-profile/README.md).
- Flash-VQG baseline: `baseline-r16-joint`.
- GDN baseline: `gdnxk-h2-ek4-ev4-usegate0`.

## 2. 实验口径

本实验在 RTX 2080 Ti 上比较 FP32 与 AMP-FP16, 在 RTX 3090 上比较 FP32, AMP-FP16 与 AMP-BF16. 每个模型和 dtype 使用 seeds `123,124,125`, 固定 B64, GA4, 4 epochs 和每 epoch 4 次 validation.

Flash-VQG 使用 hybrid precision: 外围投影与局部路径遵循 AMP dtype, grouped update 与 selected-read 两个 Triton core 在 kernel 外建立 FP32 boundary. GDN 外围遵循 AMP dtype, FLA kernel dtype 与实验 dtype 显式匹配.

matching train/eval dtype 为主结果. Off-diagonal train x eval dtype 网格只用于机制分析. 两张 GPU 分别统计 3 seeds 的 mean, population std 和 seed-paired delta, 不合并为 `n=6`.

## 3. Smoke 与恢复验证

待自动填充:

- 双机环境, cache, init 与 commit gate.
- 30 个 descriptor 的 3-update train/validation/eval smoke.
- 15 个 Flash 满注入 stress smoke.
- controlled training resume 与 `8190x2047` eval resume.
- 全量 batch 容量搜索与 batch invariance.
- 4 个历史 FP32 canary.

## 4. 正式结果

待正式队列完成后填充 last 与 best 的 matching dtype 主表, off-diagonal 网格和 longer-MQAR 分析.

## 5. 结论

待实验完成.
