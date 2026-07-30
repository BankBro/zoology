# 项目状态

更新时间: 2026-07-30.

## 1. 当前基线

- Flash-VQG: `baseline-r16-joint`, `gd_rank=16`, `read_topk=16`, `write_topk=4`, `smooth_p4` softcap `0.5`, injection warmup `0->512`.
- GDN 对照: `gdnxk-h2-ek4-ev4-usegate0`, active state capacity `131072`.
- 默认环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`, PyTorch 2.6.0+cu118, Triton 3.2.0, FLA 0.4.2.
- 已验证代码: 当前 precision profile 的训练/评估绑定 Zoology `e56fa9a`, Flash-VQG `9a8bf70`; 历史 Longer-MQAR 恢复 runner 为 `ed95ec2`.
- 依据: [效率报告](20260724-01-flash-vqg-gd-residual-efficiency-report.md), [FLA 兼容性报告](20260724-02-gdn-ek4-fla-compatibility-report.md), [Longer-MQAR报告](20260725-01-current-baselines-longer-mqar-report.md), [低精度与长度泛化报告](20260726-01-mqar-precision-profile-report.md), [Remat回归报告](20260729-01-mqar-gd-remat-regression-report.md), [seed124因果诊断](20260729-03-mqar-seed124-remat-causal-diagnosis-report.md).

## 2. 当前进展

- Flash-VQG 显存与运行时间优化已完成, 未改变模型数学语义.
- GDN `ek4-ev4` 的 RTX 3090 兼容性已解决, 两条实验分支已合入各自活跃基线.
- 双 GPU 正式质量回归通过; Flash-VQG 相对同量级 GDN 的训练、eval 和显存比值均不超过 `2x`.
- 当前基线已在2080 Ti和3090分别完成三seed 4ep Longer-MQAR. 两机训练长度端点均不支持Flash领先; 2080 Ti的四个外推slice为三个稳健领先、一个混合领先, 3090四个均稳健领先.
- GDN同seed跨GPU结果高度稳定; Flash存在更明显的seed×GPU数值路径敏感性, 主要表现为seed124在2080 Ti退化而3090未复现. 同seed跨GPU结果不合并为`n=6`.
- MQAR低精度profile已完成30/30个正式训练run和2028个逻辑checkpoint-eval事件. GDN的FP16/BF16 matching质量与FP32近乎一致; Flash低精度变化方向随GPU改变, 但在四个真正外推slice上, 全部60/60个`GPU x dtype x seed x shape`配对仍高于GDN. 固定checkpoint只改变eval dtype的最大accuracy跨度为`0.002328`.
- 低精度训练显存收益明确: Flash peak allocated约降至FP32的`0.819x`, GDN约为`0.800x`. 3090上的Flash-BF16平均wall time为FP32的`0.862x`; 30个run全部保持FP32 master weights与optimizer state, 仅1次可接受的GradScaler skip.
- GD `post_phase1` remat的确定性selected-read修复已完成RTX 3090 BF16三seed回归. 标准`1024x256` delta从历史`-0.04020`恢复为`+0.00650`, 四外推slice宏平均从`-0.10562`恢复为`+0.01743`; seed123和seed125的四epoch A0/A1状态逐位一致. 但seed124最终hash仍分叉, 终态为`quality_recovered_but_not_deterministic`, A1仍不替代A0.
- Seed124剩余分叉已完成因果定位: FLA 0.4.2 `FusedRMSNormGated` backward在fresh process中选择不同Triton autotune归约config, 使layer1 output gate weight梯度产生最大`1.82e-12`差异. 固定config后,A0/A1 177-step、最终model/optimizer hash和两次validation质量指标完全一致. 当前不支持“remat数学语义改变”解释.
- A1训练加速MQAR筛选已完成. 新的`triton_deterministic` selected backward在seed123 reference上达到标准`0.959813`, 说明exact kernel可以正常学习; 但最快的`block256/write2/read8`近似候选仅为`0.218785`, 四外推slice宏平均delta为`-0.573613`.
- 为修正原curriculum中大block只有1至2个block的混杂, 追加了等tokens、等microbatch数和等block几何实验. `block128`与`block128/write2/read8`在对应任务上均接近随机. 该结果否决当前大block配置, 但由于单样本绝对长度和KV负载同步放大4倍, 不能外推为所有自然语言任务的普遍失败定理.

## 3. 下一步

- 后续实验默认以上述 Flash-VQG、GDN 和 Conda 环境为基线.
- 自然语言300M训练仍优先在RTX 3090使用BF16. `post_phase1` remat和Triton selected backward目前只有单seed新kernel canary及既有remat质量证据, 在完整1B-token训练前仍需三seed或短自然语言paired pilot.
- 不采用`block128/256`或`write2/read8`作为当前正式质量路径. 性能工程优先保持逻辑block不变, 研究物理tiling、persistent scan和state-build/read融合, 不把逻辑block增大产生的加速标记为exact.
- 若继续研究MQAR数值稳定性, 优先定位Flash跨GPU训练轨迹的首次state-hash分叉; 若需加强结论, 增加新的独立training seeds.
- 如需更换基线, 先完成正式对照实验并更新本页与实验日志.
