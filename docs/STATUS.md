# 项目状态

更新时间: 2026-07-25.

## 1. 当前基线

- Flash-VQG: `baseline-r16-joint`, `gd_rank=16`, `read_topk=16`, `write_topk=4`, `smooth_p4` softcap `0.5`, injection warmup `0->512`.
- GDN 对照: `gdnxk-h2-ek4-ev4-usegate0`, active state capacity `131072`.
- 默认环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`, PyTorch 2.6.0+cu118, Triton 3.2.0, FLA 0.4.2.
- 已验证代码: Zoology `7eab136`, Flash-VQG `ec770f3`.
- 依据: [效率报告](20260724-01-flash-vqg-gd-residual-efficiency-report.md), [FLA 兼容性报告](20260724-02-gdn-ek4-fla-compatibility-report.md).

## 2. 当前进展

- Flash-VQG 显存与运行时间优化已完成, 未改变模型数学语义.
- GDN `ek4-ev4` 的 RTX 3090 兼容性已解决, 两条实验分支已合入各自活跃基线.
- 双 GPU 正式质量回归通过; Flash-VQG 相对同量级 GDN 的训练、eval 和显存比值均不超过 `2x`.

## 3. 下一步

- 后续实验默认以上述 Flash-VQG、GDN 和 Conda 环境为基线.
- 如需更换基线, 先完成正式对照实验并更新本页与实验日志.
