# 项目状态

更新时间: 2026-08-02.

## 1. 当前基线

- Flash-VQG: `baseline-r16-joint`, `gd_rank=16`, `read_topk=16`, `write_topk=4`, `smooth_p4` softcap `0.5`, injection warmup `0->512`.
- GDN 对照: `gdnxk-h2-ek4-ev4-usegate0`, active state capacity `131072`.
- 当前质量canonical为`baseline-r16-joint + A1 post-phase1 remat + S1 exact`; 当前最快MQAR稳定候选为`baseline-r16-joint + A1 + K2 P8 + W2 direct + K3 fixed-slot VJP + G1 head-grouped geometry + F1 hoisted selected forward + block64/local1`.
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
- A1 block64低成本质量门禁已通过. 在seed123、FP32、1 epoch且FLA fused gate配置相同的条件下, A0/A1的704-step loss、最终model/optimizer hash、标准MQAR和4个外推任务全部完全一致. 该结果允许继续C1/K1工程探索, 但不替代300M BF16短自然语言paired pilot.
- K2 P8 persistent scan的RTX 3090 AMP BF16质量筛选已完成. Seed123一epoch标准validation delta为`-0.010344`, 四外推宏平均delta为`-0.039300`, 未通过预注册门槛; 三seed正式矩阵按计划未启动. 梯度分解确认K2与P0的粗状态和残差状态分支分别一致, 差异来自`W_blk`处不同的FP32累加树. K2应分类为forward exact、backward E1资源候选, 不提升为质量canonical.
- Selected-read W2筛选已完成. W2 direct在seed123 AMP BF16一轮block64 MQAR中以标准delta`-0.005613`和四外推宏平均delta`-0.019200`通过门槛; preproject的外推宏平均delta为`-0.036765`, 已拒绝. W2 direct仍超过低层绝对误差门槛且只有单seed证据, 因此是fast resource candidate, 不替换S1 exact质量canonical.
- [当前最快Flash与GDN正式MQAR对照](20260801-01-fastest-flash-vs-gdn-mqar-report.md)已完成. RTX 3090 AMP BF16下完成9/9条三seed四epoch训练、234/234逻辑评估和15/15 endpoint重复性检查. 预注册Last主门禁通过: Fastest相对Canonical的标准端点均值delta为`-0.015702`, 四外推宏平均delta为`-0.000430`; 但两组Flash均在epoch1后明显退化, Last四外推宏平均约`0.083`, 低于GDN的`0.214`.
- Best checkpoint下, Fastest、Canonical和GDN的四外推宏平均分别为`0.509`, `0.598`和`0.214`. Fastest相对Canonical的best均值delta为`-0.089462`, seed125为`-0.250114`; 因而Last主门禁通过不等于训练轨迹或最佳质量等价. Fastest仍是资源候选, S1 exact继续作为质量canonical.
- [Flash后期退化因果诊断](20260801-02-flash-late-degradation-causal-diagnosis-report.md)已完成. Block32/local2桥接组两seed四epochpeak-to-final drop仅为`-0.005219/-0.004484`, block64/local2则为`-0.247516/-0.161031`. 2x2机制矩阵将主要因素定位为`local_num_blocks`控制的近场/远场可见跨度由64扩到128 token, 而不是64-token block边界本身; fixed与default FLA均复现.
- 最快栈改为block64/local1后, 两seed validation peak均值相对local2仅低`0.021062`, final均值提高`0.132854`, drop从`-0.157150`缩小到`-0.003234`; 四外推last宏平均从`0.092636`恢复到`0.451911`, 略高于local2 best的`0.441406`. 小模型四epochwall同时改善约`1.9%`. 因此block64/local1替换local2成为最快MQAR稳定候选.
- Seed123 fresh-per-epoch补充没有消除block64/local2退化: final仅提高`0.031805`, drop仍为`-0.200363`. 终态为`persistent_window_dynamics`, 不再简单称为重复MQAR cache导致的传统过拟合. `local_num_blocks`同时改变local window和remote boundary offset, 当前尚未将两者进一步解耦.

## 3. 下一步

- 后续实验默认以上述 Flash-VQG、GDN 和 Conda 环境为基线.
- 自然语言300M训练仍优先在RTX 3090使用BF16. 当前质量路径为P0 A1加S1 exact selected backward; W2 direct仅作为fast resource candidate. 在完整1B-token训练前仍需300M短自然语言paired pilot.
- 不采用`block128/256`、`write2/read8`或当前K2 P8作为正式质量路径. 若继续K2, 应先控制跨分支`W`梯度累加树并重新执行BF16 Q0, 或预注册新的E1统计性质量协议, 不事后放宽本次门槛.
- 若继续研究MQAR数值稳定性, 优先定位Flash跨GPU训练轨迹的首次state-hash分叉; 若需加强结论, 增加新的独立training seeds.
- 如需更换基线, 先完成正式对照实验并更新本页与实验日志.
- Fastest若进入自然语言质量路径, 应做同初始化、同data order的300M BF16短自然语言paired pilot, 至少配对比较block64/local2与block64/local1并保留当前质量参考. 需要同时测资源、validation NLL、多阶段checkpoint、下游指标和checkpoint选择敏感性; 本次MQAR结果不自动授权1B-token训练.
- 后续MQAR稳定性screen不得只停在step1232. Fastest seed123在该点仍稳定, 四epoch末端却下降`0.154855`; 应运行完整四epoch或采用peak后追加固定验证窗口的预注册规则.
