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

首次完整 pre-formal 队列在源码 `80483073` 下执行. 2080 Ti 的 12 个普通训练 smoke, 6 个 Flash stress, 52 个容量 profile, 52 个 batch invariance, 312 个 eval smoke 和 26 个 canary 物理事件均完成, 但 canary 汇总正确地 fail-fast, 正式训练未启动. 失败原因是 canary 生成数据与旧 checkpoint validation cache 不是同一数据集, 因而不应进行逐项精确指标比较. 旧结果保存在 `outputs/invalidated-80483073-canary-generated-data/`.

修正后, 标准 n=1000 canary 只读 checkpoint 的原始 test cache, longer n=500 仍使用锁定 dataset hash 的生成数据. GDN `64x4` 与 `1024x256` 两个首尾 test segment 的单事件验证均与旧指标完全一致, delta=0. 修复后的双机全部 gate 将从头重跑, 不复用旧 commit 的 smoke gate.

## 4. 正式结果

待正式队列完成后填充 last 与 best 的 matching dtype 主表, off-diagonal 网格和 longer-MQAR 分析.

## 5. 结论

待实验完成.
