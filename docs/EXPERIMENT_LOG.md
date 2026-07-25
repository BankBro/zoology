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
