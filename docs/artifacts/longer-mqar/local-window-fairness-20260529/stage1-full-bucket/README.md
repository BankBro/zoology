# 阶段 1 full-only distance bucket eval

本目录保存 `20260529-flash-local-window-fairness` 的阶段 1 longer-MQAR full-only distance bucket 诊断结果.

运行口径:

- 机器: mclab-3090, RTX 3090.
- checkpoint 来源: `../source_checkpoints.csv`.
- slices: `1024x256`, `2048x512`, `4096x512`, `4096x1024`.
- variants: `full`.
- num_examples: 500.
- distance 定义: `query_pos - value_pos`, 来自 MQAR 生成时 position metadata.
- bucket 输出: `distance_bucket.csv`, 含 `n`, `correct`, `accuracy`, `stderr`, `ci95_low`, `ci95_high`.

Sanity 结果:

- Flash rows: strict official sanity 全部通过, 与 RTX 2080 Ti official ref 差值为 0.
- GDN rows: 数据 hash 与 official ref 一致, 但 strict `1e-4` accuracy sanity 有 8 行 invalid, 最大差值 `0.0005703125`.
- GDN 复查: 单独用 official batch size 32 复跑 `gdn-h2-ev8-s123 1024x256`, 差值保持不变, 因此不是 batch size 造成.
- 解释: official longer-MQAR ref 来自 RTX 2080 Ti, torch 2.6.0+cu118, CUDA 11.8, driver 550.120. 当前阶段 1 在 RTX 3090 上跑, GDN FLA chunk kernel 即使 `GDN_KERNEL_DTYPE=float32` 也出现小的硬件/kernel 数值差异.

使用约束:

- Flash full rows 可作为本阶段 strict sanity 通过的 official 复现.
- GDN full rows 先作为 3090 diagnostic bucket baseline 保留, 不作为 strict official reproduction.
- 后续 Flash local ablation 可继续, 但 Flash-vs-GDN 距离 bucket 解读必须注明 GDN strict sanity invalid 的限制.
