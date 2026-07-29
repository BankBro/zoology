# Seed124 remat 数值分叉因果诊断 Artifact

本 artifact 保存实验 `20260729-03-mqar-seed124-remat-causal-diagnosis` 的精简结论证据. 终态为 `causal_root_identified`.

## 1. 关键结论

seed124 剩余分叉不是已证实的 remat 数学语义变化. 根因是 FLA 0.4.2 `FusedRMSNormGated` backward 在每个 fresh process 中重新执行 Triton autotune, 不同获胜配置改变 `output_gate_fused.weight` 的 FP32 归约顺序.

- 首个不同量是 window1、microbatch0 的 layer1 output gate weight gradient.
- `BT64, warps4` 与 `BT64, warps8` 使 42/64 个元素出现差异, 最大绝对差 `1.8189894e-12`.
- 差异在 step1 进入 Adam state, step4 改变参数, window10 首次改变 loss.
- A0/A1 同时固定 `BT64, warps4` 后, 177 steps 的 1947 个训练事件、最终 model/optimizer hash和两次 validation 质量指标全部一致.

## 2. 文件

| 文件 | 内容 |
|---|---|
| `first-divergence.json` | gradient、optimizer、model和loss的首次分叉时间线 |
| `autotune-gradient-groups.csv` | 12个默认 fresh-process run 的 autotune config 与梯度 hash |
| `exact-gradient-difference.json` | 两个固定 config 的逐元素差异统计 |
| `causal-validation.json` | 固定 config 后的177-step与validation门禁 |
| `replay-summary.json` | 真实算子 replay 的 forward、输入梯度和weight梯度结果 |
| `metadata.json` | 终态、源码和主结论 |
| `source-manifest.csv` | 3090 raw、主工作区镜像、大小与SHA256 |

8.1 MiB replay capsule和约1.2 GiB checkpoint仍保留在3090 raw目录, 不提交Git. 10个关键raw文件已镜像回主工作区并逐文件验证SHA256一致.

两机实际安装的FLA `fused_norm_gate.py` SHA256均为`e620731d73fd069944b37d5b4a76cf00adcaba2b6745f5600351751cec85afe1`, 已写入`metadata.json`.
