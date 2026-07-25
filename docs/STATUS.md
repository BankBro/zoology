# 项目状态

更新时间: 2026-07-25.

## 1. 当前基线

- Flash-VQG: `baseline-r16-joint`, `gd_rank=16`, `read_topk=16`, `write_topk=4`, `smooth_p4` softcap `0.5`, injection warmup `0->512`.
- GDN 对照: `gdnxk-h2-ek4-ev4-usegate0`, active state capacity `131072`.
- 默认环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`, PyTorch 2.6.0+cu118, Triton 3.2.0, FLA 0.4.2.
- 已验证代码: Zoology训练 `d6616f9`, Longer-MQAR eval恢复runner `ed95ec2`, Flash-VQG `ec770f3`.
- 依据: [效率报告](20260724-01-flash-vqg-gd-residual-efficiency-report.md), [FLA 兼容性报告](20260724-02-gdn-ek4-fla-compatibility-report.md), [Longer-MQAR报告](20260725-01-current-baselines-longer-mqar-report.md).

## 2. 当前进展

- Flash-VQG 显存与运行时间优化已完成, 未改变模型数学语义.
- GDN `ek4-ev4` 的 RTX 3090 兼容性已解决, 两条实验分支已合入各自活跃基线.
- 双 GPU 正式质量回归通过; Flash-VQG 相对同量级 GDN 的训练、eval 和显存比值均不超过 `2x`.
- 当前基线已在2080 Ti和3090分别完成三seed 4ep Longer-MQAR. 两机训练长度端点均不支持Flash领先; 2080 Ti的四个外推slice为三个稳健领先、一个混合领先, 3090四个均稳健领先.
- GDN同seed跨GPU结果高度稳定; Flash存在更明显的seed×GPU数值路径敏感性, 主要表现为seed124在2080 Ti退化而3090未复现. 同seed跨GPU结果不合并为`n=6`.

## 3. 下一步

- 后续实验默认以上述 Flash-VQG、GDN 和 Conda 环境为基线.
- 优先定位Flash跨GPU训练轨迹的首次state-hash分叉; 若需加强结论, 增加新的独立training seeds.
- 如需更换基线, 先完成正式对照实验并更新本页与实验日志.
