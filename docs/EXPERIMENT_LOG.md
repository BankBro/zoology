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

## 3. 2026-07-25: 当前 Flash/GDN 基线 Longer-MQAR

- `experiment_id`: `20260725-01-current-baselines-longer-mqar`.
- 目的: 以三 training seeds、4ep重训和 RNG-locked五 slice评估, 比较当前 Flash `baseline-r16-joint` 与 GDN `gdnxk-h2-ek4-ev4-usegate0` 的长度泛化.
- 结果: 6/6正式训练和全部 formal/repro完成. `last.pt`主结果在四个外推 slice中有三个为 Flash 3/3 seeds稳健领先, `8190x512`为混合领先; `1024x256`不支持 Flash领先. Flash seed124是主要方差来源.
- 输出: [报告](20260725-01-current-baselines-longer-mqar-report.md), [artifact](artifacts/20260725-01-current-baselines-longer-mqar/README.md).
- 启动记录: 首次 preflight因入口文件遮蔽Python标准库失败, 在任何 smoke/formal启动前由 commit `0dd9572`修复; 后续全流程通过.
- 下一步: 分析 Flash seed124的 epoch3->epoch4退化, 必要时增加独立 seeds; 不自动更换当前 baseline.

## 4. 2026-07-25: 当前基线 Longer-MQAR 3090扩展首次formal eval中断

- `experiment_id`: `20260725-01-current-baselines-longer-mqar`.
- 目的: 补充3090独立重训及与2080 Ti相同口径的跨GPU长度泛化对照.
- 结果: 6/6正式训练和checkpoint审计完成; 首次formal eval在Flash s123 `8190x512`、batch 32处OOM并fail-fast. 原因是batch-search只用32 examples, 未覆盖500-example多batch allocator碎片化; 已完成训练不受影响.
- 输出: 3090 ignored raw的`outputs/machines/3090/queue/FAILED.json`和`formal-eval`事件日志.
- 下一步: formal batch-search改用完整500 examples, 全部测试和smoke通过后从已有checkpoint幂等恢复.

## 5. 2026-07-25: 当前基线 Longer-MQAR 跨GPU扩展完成

- `experiment_id`: `20260725-01-current-baselines-longer-mqar`.
- 目的: 完成3090恢复eval, 与2080 Ti按机器分层汇总, 生成last/best跨GPU双图.
- 结果: 3090恢复队列重新通过全部preflight和smoke, 6条训练经hash审计后复用; 30/30 formal和6/6 repro完成. 两机合并为120条唯一逻辑结果, 五个dataset hash完全一致. 2080 Ti外推为三个稳健领先、一个混合领先; 3090四个外推slice均为Flash 3/3 seeds稳健领先. GDN跨GPU稳定, Flash存在明显seed×数值路径敏感性.
- 输出: [报告](20260725-01-current-baselines-longer-mqar-report.md), [artifact](artifacts/20260725-01-current-baselines-longer-mqar/README.md), [last图](artifacts/20260725-01-current-baselines-longer-mqar/figures/longer-mqar-accuracy-last.pdf), [best图](artifacts/20260725-01-current-baselines-longer-mqar/figures/longer-mqar-accuracy-best.pdf).
- 下一步: 对Flash增加跨GPU早期step state/kernel hash probe; 不把相同seed跨GPU结果合并为`n=6`, 不自动替换baseline.

## 6. 2026-07-26: MQAR 低精度与长度泛化实验启动

- `experiment_id`: `20260726-01-mqar-precision-profile`.
- 目的: 在2080 Ti上比较FP32与AMP-FP16, 在3090上比较FP32, AMP-FP16与AMP-BF16, 对当前Flash/GDN基线完成三seed重训, 标准MQAR和longer-MQAR全精度网格评估.
- 当前结果: 实验计划, Flash Triton FP32 boundary, AMP/GradScaler, 精确`resume.pt`, 可恢复逐batch eval, batch capacity/invariance和双机global gate自动化已完成. 2080 Ti与3090环境/cache/init/commit preflight通过; 正式训练尚未启动.
- 输出: [计划](plans/20260726-01-mqar-precision-profile-plan.md), [报告骨架](20260726-01-mqar-precision-profile-report.md), [artifact](artifacts/20260726-01-mqar-precision-profile/README.md).
- 下一步: 双机完成全部descriptor train/validation/eval smoke和Flash满注入stress smoke; global gate通过后自动启动30个正式run.

## 7. 2026-07-26: MQAR 低精度实验 canary 数据口径修正

- `experiment_id`: `20260726-01-mqar-precision-profile`.
- 目的: 在正式 gate 前用4个历史FP32 checkpoint验证新 evaluator 与旧 validation 指标严格一致.
- 结果: 旧commit `80483073` 下, 2080 Ti完成312/312 eval smoke和26/26全量canary后, canary汇总按设计fail-fast; 正式训练从未启动. 原因是canary重新生成了`random_non_queries=True`数据, 却与旧checkpoint在原始validation cache上的指标做逐项精确比较, 最大差为`6.5625e-4`, 不是同数据集上的模型回归. 3090旧队列在326/702 eval smoke处主动停止.
- 修正: 标准n=1000 canary改为按checkpoint完整test-config顺序恢复原始segment seed并只读加载对应`data_*.pt` cache; longer n=500仍使用锁定hash的生成数据. `64x4`与test-config最后一个`1024x256`单事件复验均与旧指标严格相等, delta为0; 严格相等门槛未放宽.
- 输出: 旧结果非破坏性归档在双机`outputs/invalidated-80483073-canary-generated-data/`; debug单事件保存在2080 Ti `outputs/dev-cache-canary-*`.
- 下一步: 提交并同步修复后, 以新commit从头重跑双机全部preflight, train/validation/eval smoke, capacity/invariance和canary; global gate通过前继续禁止formal.

## 8. 2026-07-26: MQAR 低精度与长度泛化正式实验完成

- `experiment_id`: `20260726-01-mqar-precision-profile`.
- 目的: 在2080 Ti上比较FP32与AMP-FP16, 在RTX 3090上比较FP32, AMP-FP16与AMP-BF16, 对当前Flash/GDN基线完成三seed重训及标准/longer-MQAR全精度网格评估.
- 门禁: 2080 Ti通过52/52 capacity与batch invariance, 312/312 eval smoke, 26/26 canary和16/16标准accuracy审计; RTX 3090通过78/78 capacity与batch invariance, 702/702 eval smoke, 26/26 canary和16/16标准accuracy审计. 两机controlled eval resume均实际通过, global commit/cache/config gate全部为true.
- 结果: 30/30正式训练run和2028个逻辑checkpoint-eval事件完成, 其中1066个物理执行, 962个best/last state-hash去重. GDN低精度matching质量与FP32近乎一致; Flash低精度变化方向随GPU改变, 但在四个真正外推slice的60/60个`GPU x dtype x seed x shape`配对中均高于GDN. 固定checkpoint只改变eval dtype的最大accuracy跨度为`0.002328`.
- 效率: Flash低精度peak allocated约为FP32的`0.819x`; 3090 Flash-BF16平均wall time为FP32的`0.862x`. GDN低精度peak allocated约为FP32的`0.800x`, 并获得更明显训练加速. 30个run仅记录1次允许范围内的GradScaler skip, 全部正常完成epoch 4.
- 审计: 独立artifact包含30条canonical training ledger, 780条canonical longer-MQAR ledger, 60个checkpoint file hash, 18个双机gate/status JSON及30个resolved config镜像hash. 本dtype probe不混入历史FP32推荐总表.
- 基础设施修正: formal启动前发现coordinator的SSH argv在远端重组时丢失`bash -lc`命令边界; 修复为`shlex.join`单一远端命令后, 用真实3090 gate完成读写round-trip smoke, 再由原`build_global_gate`五项校验自动放行. 未绕过任何实验门禁.
- 输出: [报告](20260726-01-mqar-precision-profile-report.md), [artifact](artifacts/20260726-01-mqar-precision-profile/README.md), [last图](artifacts/20260726-01-mqar-precision-profile/figures/matching-precision-last.pdf), [best图](artifacts/20260726-01-mqar-precision-profile/figures/matching-precision-best.pdf).
- 下一步: 自然语言300M训练优先以RTX 3090 BF16做任务级容量与完整smoke, 再启动正式下游训练; 2080 Ti保留为FP16 B1/GA兼容路径.

## 9. 2026-07-29: GD post-phase1 remat MQAR 回归失败

- `experiment_id`: `20260729-01-mqar-gd-remat-regression`.
- 目的: 在RTX 3090 BF16下, 以三seed配对验证A1 `post_phase1` remat是否保持canonical `baseline-r16-joint`的MQAR质量和长度外推.
- 门禁: Preflight, 32-step轨迹, controlled resume, checkpoint和eval smoke均通过; 6/6正式训练及60/60逻辑评估完成, 其中30个物理执行、30个best/last state-hash去重.
- 结果: A1 peak allocated约降至A0的`0.775x`, wall time增至`1.149x`; 标准`1024x256` delta均值为`-0.04020`, 四外推slice宏平均为`-0.10562`, 两个质量门槛均失败. A1不替代A0.
- 数值审计: step1严格一致; step16参数max abs差为`2.38e-7`, step32为`1.00e-6`; 四epoch后三seed model-state hash均分叉, seed125退化最明显.
- Collector修复: 首次collect因A0/A1 CSV字段集合不同而失败; commit `03f5d25`只修复汇总schema并复用已有训练/评估结果, 最终质量失败结论不受该基础设施bug影响.
- 输出: [报告](20260729-01-mqar-gd-remat-regression-report.md), [artifact](artifacts/20260729-01-mqar-gd-remat-regression/README.md).
- 下一步: 禁止A1进入自然语言正式训练; 优先定位remat数值分叉或验证不重算GD图的显存方案, 再执行同口径回归.

## 10. 2026-07-29: 确定性 Selected-Read 回归质量恢复但未完全确定

- `experiment_id`: `20260729-02-mqar-deterministic-selected-read-regression`.
- 目的: 将selected-read backward中`addr_proj`重复head梯度改为固定顺序归约, 重新验证A0与A1的三seed MQAR和长度外推.
- 门禁: 低层8次重复、seed124的128-step lockstep、fresh-process、controlled resume和smoke均通过; 6/6正式训练与60/60逻辑评估完成, 20个物理评估按model-state hash去重.
- 结果: 标准`1024x256` delta均值为`+0.00650`, 四外推slice宏平均为`+0.01743`, 均通过非劣门槛. seed123和seed125的四epochA0/A1最终hash相同, seed124仍分叉, 终态为`quality_recovered_but_not_deterministic`.
- 效率: A1 peak allocated降低约`22.0%`, peak reserved降低约`23.2%`; 平均wall time增加约`14.4%`, optimizer-step p50增加约`17.8%`.
- 输出: [报告](20260729-02-mqar-deterministic-selected-read-regression-report.md), [artifact](artifacts/20260729-02-mqar-deterministic-selected-read-regression/README.md).
- 下一步: A1不晋升; 对seed124执行更长的独立进程state/gradient/optimizer hash轨迹, 定位剩余首次分叉后再重跑同口径回归.

## 11. 2026-07-29: Seed124 remat 剩余分叉完成因果定位

- `experiment_id`: `20260729-03-mqar-seed124-remat-causal-diagnosis`.
- 目的: 定位seed124 A0/A1的首个剩余分叉, 并以单变量干预完成因果验证.
- 结果: 首个差异是window1、microbatch0的layer1 `output_gate_fused.weight`梯度. FLA 0.4.2 fused gate backward的fresh-process Triton autotune在`BT64 warps4/warps8`间选择, 改变FP32 weight归约顺序; 42/64个元素不同, 最大绝对差`1.82e-12`.
- 因果门禁: 固定`BT64, warps4`后, A0/A1的177-step共1947个训练事件、最终model/optimizer hash和两次validation质量指标全部一致; 真实算子replay精确复现两个梯度hash.
- 输出: [报告](20260729-03-mqar-seed124-remat-causal-diagnosis-report.md), [artifact](artifacts/20260729-03-mqar-seed124-remat-causal-diagnosis/README.md).
- 下一步: A1仍不晋升; 先生产化确定性output gate backward并评估吞吐, 再执行三seed正式质量回归.

## 12. 2026-07-30: 训练加速近似候选 MQAR 筛选失败

- `experiment_id`: `20260730-01-a1-acceleration-mqar-probe`.
- 目的: 以seed123、FP32、1 epoch筛选300M性能实验中最快的`block256/write2/read8`近似候选, 并验证新的Triton deterministic selected backward能否正常学习.
- 结果: Exact reference标准`1024x256`为`0.959813`; 组合候选为`0.218785`, delta为`-0.741027`, 四外推slice宏平均delta为`-0.573613`. `block128`, `block256`和`write2/read8`单变量分别为`0.060676`, `0.199504`和`0.908125`.
- 失败修复: 5次评估identity、batch或allocator失败均保留现场、根因修复后完成重试, 不影响质量拒绝结论.
- 输出: [报告](20260730-01-a1-acceleration-mqar-probe-report.md), [artifact](artifacts/20260730-01-a1-acceleration-mqar-probe/README.md).
- 下一步: 不启动该近似候选的三seed扩展; 追加block几何归一化实验排查短curriculum混杂.

## 13. 2026-07-30: Block Geometry 归一化 MQAR 仍拒绝大 block

- `experiment_id`: `20260730-02-a1-block-geometry-mqar-probe`.
- 目的: 在等训练tokens、microbatch数、optimizer更新及block几何下, 判断上游大block退化是否只是训练序列过短.
- 结果: Reference `1024x256`为`0.959813`; `block128`及`block128/write2/read8`在对应`4096x1024`上分别为`0.000236`和`0.000241`, 接近随机. 两者均未进入Longer-MQAR.
- 失败修复: `prepare-data`缺少上游run tag及summarizer错误均保留现场、最小修复并通过回归测试后重试; 三个正式训练均在摘要错误前完成.
- 输出: [报告](20260730-02-a1-block-geometry-mqar-probe-report.md), [artifact](artifacts/20260730-02-a1-block-geometry-mqar-probe/README.md).
- 下一步: 当前不采用扩大逻辑block的性能路径; 优先保持逻辑block不变的物理tiling和kernel融合.

## 14. 2026-07-30: A1 Block64 Remat 单seed质量门禁通过

- `experiment_id`: `20260730-03-a1-block64-remat-quality-canary`.
- 目的: 在进入C1/K1性能工程前, 以seed123、FP32和1 epoch验证block64下A0与A1的训练轨迹及Longer-MQAR质量.
- 结果: 完整run中A0/A1的704-step loss最大差为0, 最终model/optimizer hash相同; 标准和4个外推任务delta均为0. FLA fused gate backward配置同为`BT32/warps4`.
- 资源: A1 wall time增加约14.3%; validation peak reserved下降470 MiB. 该小模型FP32结果不外推为300M吞吐结论.
- 失败修复: 首次run在8190-token FP32 batch16评估时因4 GiB logits申请OOM; 仅将两个最长任务batch降为4后, 以新tag从头完整重跑通过.
- 输出: [报告](20260730-03-a1-block64-remat-quality-canary-report.md), [artifact](artifacts/20260730-03-a1-block64-remat-quality-canary/README.md).
- 下一步: 允许继续P0/P1和C1/K1; 1B-token训练前仍需300M BF16短自然语言paired pilot.

## 15. 2026-07-30: K2 Persistent Scan BF16 MQAR 筛选拒绝

- `experiment_id`: `20260730-04-k2-persistent-scan-mqar-regression`.
- 目的: 在RTX 3090 AMP BF16下比较P0 A1与K2 P8, 判断persistent scan与bounded backward能否通过block64 MQAR及长度外推质量门禁.
- 门禁: 两组smoke、runtime、finite、dtype和fallback审计通过; 两组FLA fused gate backward均为`BT64/warps8`. Seed123一epochQ0完成704个optimizer updates及全部固定评估.
- 结果: 标准validation delta为`-0.010344`, 四外推宏平均delta为`-0.039300`, 均低于预注册门槛. 三seed四epoch正式矩阵按计划未启动. 补充FP32同seed诊断方向反转为正, 但状态仍分叉, 不覆盖BF16拒绝结论.
- 根因: P0与K2的粗状态和残差状态分支梯度分别逐位一致; 差异只在两条贡献于`W_blk`汇合时出现, 对应P0逐block交错与K2分支分离的FP32累加树. Tile P1/P2/P4/P8误差相同, 排除tile大小为根因.
- 输出: [报告](20260730-04-k2-persistent-scan-mqar-regression-report.md), [artifact](artifacts/20260730-04-k2-persistent-scan-mqar-regression/README.md).
- 下一步: K2保留为forward exact、backward E1资源候选; 当前质量路径回到P0 A1. 未经新的低层修复和BF16 Q0, 不启动K2自然语言正式训练或1B-token训练.

## 16. 2026-07-31: Selected-read W2 MQAR 筛选质量混合

- `experiment_id`: `20260731-01-selected-read-warp-mqar-screen`.
- 目的: 在RTX 3090 AMP BF16下, 以seed123一轮block64 MQAR比较S1 exact、W2 direct和W2加preproject.
- 结果: W2 direct标准delta为`-0.005613`, 四外推宏平均delta为`-0.019200`, 两项通过预注册门槛. Preproject标准通过, 外推宏平均delta为`-0.036765`, 已拒绝. 三组FLA config一致, fallback为0.
- 决策: S1 exact继续作为质量canonical; W2 direct保留为fast resource candidate, preproject只保留资源上限证据并从最终生产源码删除.
- 输出: [报告](20260731-01-selected-read-warp-mqar-screen-report.md), [artifact](artifacts/20260731-01-selected-read-warp-mqar-screen/README.md).
- 下一步: 完成Flash-VQG中的K2组合资源测量和生命周期闭环; 是否提升W2证据等级由多seed或300M BF16短自然语言paired pilot决定.

## 17. 2026-08-01: 当前最快 Flash 与 GDN 的 MQAR 正式对照完成

- `experiment_id`: `20260801-01-fastest-flash-vs-gdn-mqar`.
- 目的: 在RTX 3090 AMP BF16下, 以三seed四epoch正式比较当前最快Flash资源组合、A1+S1质量canonical和capacity-matched GDN的标准MQAR及长度外推表现.
- 完成度: 9/9正式训练、234/234逻辑评估、195个物理评估和15/15 endpoint fresh-process重复性检查完成; 39/39完整负载batch profile与下一档batch invariance通过, fallback为0.
- Last主结果: Fastest相对Canonical的标准端点均值delta为`-0.015702`, 四外推宏平均delta为`-0.000430`, 通过预注册5个百分点门禁. 但两组Flash均在epoch1后退化, Last四外推宏平均约`0.083`, 低于GDN的`0.214`.
- Best敏感性: Fastest、Canonical和GDN四外推宏平均分别为`0.509`, `0.598`和`0.214`. Fastest相对Canonical均值delta为`-0.089462`, seed125为`-0.250114`, 说明Fastest会改变本就敏感的Flash训练轨迹.
- 资源: 小模型协议下Fastest平均wall为`554.53 s`, 相对Canonical的`830.23 s`加速`1.497x`, 但仍比GDN的`236.69 s`慢`2.343x`.
- 决策影响: Last主门禁判定通过, 但Fastest不替换S1 exact质量canonical, 不自动进入1B-token训练. 后续若进入自然语言路径, 必须执行多checkpoint的300M BF16 paired pilot.
- 输出: [报告](20260801-01-fastest-flash-vs-gdn-mqar-report.md), [artifact](artifacts/20260801-01-fastest-flash-vs-gdn-mqar/README.md).

## 18. 2026-08-01 至 2026-08-02: Flash 后期退化因果诊断完成

- `experiment_id`: `20260801-02-flash-late-degradation-causal-diagnosis`.
- 目的: 复现当前Flash四epoch后期退化, 在固定初始化、数据、精度和优化器下拆分block长度、近场/远场可见跨度、selected backend与FLA autotune因素, 并将修复迁移到最快K2/W2/K3栈.
- 完成度: 主队列36/36作业、8条两seed四epoch正式训练、208条best/last standard与Longer-MQAR评估、8/8 batch invariance及2条fresh-per-epoch补充训练全部完成; fallback、NaN、OOM和checkpoint/dtype审计失败均为0.
- 根因: Block32/local2两seeddrop仅`-0.005219/-0.004484`, block64/local2为`-0.247516/-0.161031`. 机制矩阵支持`local_num_blocks`控制的近场/远场跨度由64扩到128 token是主要因素, 不支持64-token block边界本身为根因. Fixed与default FLA方向一致.
- 修复: 最快栈使用block64/local1后, 两seedvalidation peak均值仅下降`0.021062`, final提高`0.132854`, drop由`-0.157150`缩小为`-0.003234`; 四外推last宏平均由`0.092636`恢复为`0.451911`, 小模型wall约再快`1.9%`.
- Fresh-data结论: Epoch0与fixed对照逐值相同, epoch1至epoch3共20个epoch-segment hash全部唯一且跨结构一致. Fresh data只将退化组final提高`0.031805`, drop仍为`-0.200363`, 因此分类为`persistent_window_dynamics`, 不是重复cache导致的传统过拟合.
- 证据边界: `local_num_blocks`同时改变local window与remote boundary offset, 本实验没有进一步解耦两者. Step1232不足以筛除Fastest后期退化; 自然语言路径仍需300M BF16 block64/local2与local1配对pilot.
- 输出: [报告](20260801-02-flash-late-degradation-causal-diagnosis-report.md), [artifact](artifacts/20260801-02-flash-late-degradation-causal-diagnosis/README.md).
- 下一步: 在300M自然语言短pilot中同时测local1/local2资源和质量, 保存多阶段checkpoint; 未通过前不启动1B-token正式训练.
