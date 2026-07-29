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
