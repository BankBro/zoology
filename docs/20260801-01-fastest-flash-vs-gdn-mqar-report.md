# 当前最快 Flash 与 GDN 的 MQAR 正式对照报告

## 1. 结果概览

- `experiment_id`: `20260801-01-fastest-flash-vs-gdn-mqar`.
- 状态: `completed`.
- 目标机器: RTX 3090 GPU0.
- Zoology source: `20260801-121200-fastest-flash-vs-gdn-mqar@00a19f291109d0dd1e50326d3005d8f8c8f4c8a7`.
- Flash-VQG source: `20260801-103002-selected-read-native-load@396ae65b89b53aad316fbbf7daf55a92a551d684`.
- Plan: [实验计划](plans/20260801-01-fastest-flash-vs-gdn-mqar-plan.md).
- Artifact: [精简证据](artifacts/20260801-01-fastest-flash-vs-gdn-mqar/README.md).

实验完成 9/9 条三 seed, 四 epoch AMP BF16 正式训练, 234/234 个 last/best 逻辑评估事件和 15/15 个 endpoint fresh-process 重复性检查. 预注册的 `last.pt` 主门禁全部通过, 但该结果主要说明 Fastest 与 Canonical 在共同发生训练后期退化后仍接近, 不能单独证明两者训练轨迹或最佳 checkpoint 质量等价.

最终结论分成两部分:

- 固定四 epoch 的 `last.pt` 下, GDN 的标准端点和四外推宏平均均明显优于两个 Flash. Fastest 与 Canonical 的 last 质量接近.
- 验证集选择的 `best.pt` 下, 两个 Flash 的四外推宏平均明显高于 GDN, 但 Fastest 相对 Canonical 存在明显 seed 敏感性, seed125 四外推宏平均下降 `0.250114`.

因此, 当前 Fastest 组合通过了预注册 Last 主门禁, 但仍只能保留为快速资源候选, 不能据此替换 A1+S1 质量 canonical.

## 2. 实验合同

三组共同使用 training seeds `123/124/125`, data seed `123`, canonical cache/init, train batch `64`, gradient accumulation `4`, effective batch `256`, 四 epochs 和每 epoch 四次 validation. 正式训练与评估均为 AMP BF16, FP32 master weights 和 optimizer state, TF32关闭.

| Arm | 配置 | 参数量 | Active state capacity |
|---|---|---:|---:|
| `flash-fastest` | `baseline-r16-joint + A1 + K2 P8 + W2 direct + K3 VJP + G1 + F1` | 1,160,390 | 131,072 |
| `flash-canonical` | `baseline-r16-joint + A1 + S1 exact` | 1,160,390 | 131,072 |
| `gdn` | `gdnxk-h2-ek4-ev4-usegate0` | 1,335,942 | 131,072 |

GDN与Flash只匹配active state capacity, 参数量不完全一致, 因此本报告不称其为严格参数匹配. 300M资源实验中的B4/GA8没有移植到小模型MQAR, 避免batch暴露成为质量混杂变量.

标准MQAR包含8个任务, Longer-MQAR包含`1024x256`曲线起点和4个真正外推任务. `last.pt`为主结果, `best.pt`为checkpoint选择敏感性结果.

## 3. Last 主结果

### 3.1. 三 seed 汇总

| Arm | 标准8任务macro | 标准`1024x256` | 四外推macro | Population SD, 外推 |
|---|---:|---:|---:|---:|
| Fastest Flash | 0.968107 | 0.761488 | 0.082973 | 0.004581 |
| Canonical Flash | 0.970608 | 0.777190 | 0.083402 | 0.005167 |
| GDN | **0.995949** | **0.968426** | **0.213550** | 0.002789 |

GDN在固定四 epoch 终点的标准端点和外推宏平均均明显领先. 两个 Flash 的标准简单任务仍接近满分, 但`1024x256`与更长序列显著退化.

Fastest相对Canonical的三 seed均值 delta 为:

| 指标 | Mean delta | Worst seed | 预注册门槛 | 结果 |
|---|---:|---:|---:|---|
| 标准8任务macro | -0.002501 | -0.005391 | -0.05 | 通过 |
| 标准`1024x256` | -0.015702 | -0.035961 | -0.05 mean, -0.10 seed | 通过 |
| 四外推macro | -0.000430 | -0.004510 | -0.05 mean, -0.10 seed | 通过 |

### 3.2. Fastest与Canonical逐seed配对

| Seed | 标准macro delta | 端点delta | 四外推macro delta |
|---:|---:|---:|---:|
| 123 | +0.000016 | +0.004078 | +0.004062 |
| 124 | -0.002128 | -0.015223 | -0.004510 |
| 125 | -0.005391 | -0.035961 | -0.000840 |

三 seed方向并不完全一致, 但均未触发Last主门禁. 需要注意, 这个通过结论发生在Fastest和Canonical都已经严重后期退化的背景下, 因而属于较弱的非劣证据.

## 4. Best checkpoint 敏感性

### 4.1. 三 seed 汇总

| Arm | 标准8任务macro | 标准`1024x256` | 四外推macro | Population SD, 外推 |
|---|---:|---:|---:|---:|
| Fastest Flash | 0.988433 | 0.937408 | 0.508548 | 0.101751 |
| Canonical Flash | 0.993771 | 0.967103 | **0.598011** | 0.038157 |
| GDN | **0.995949** | **0.968426** | 0.213550 | 0.002789 |

两个Flash的best checkpoint都在第1 epoch, GDN的best与last相同且位于第4 epoch. 相对GDN, Canonical Flash的best四外推宏平均提高`0.384461`, Fastest提高`0.294998`. 这说明Flash在未发生后期退化时确有更强的长度外推信号.

### 4.2. Fastest与Canonical逐seed配对

| Seed | 标准macro delta | 端点delta | 四外推macro delta |
|---:|---:|---:|---:|
| 123 | -0.000475 | -0.002844 | -0.038144 |
| 124 | +0.001452 | +0.005711 | +0.019871 |
| 125 | -0.016990 | -0.091953 | **-0.250114** |
| Mean | -0.005338 | -0.029695 | **-0.089462** |

Seed124略有正向变化, seed123小幅负向, seed125则明显退化. Fastest不是稳定地比Canonical更差或更好, 而是改变了本就敏感的Flash训练轨迹. Seed125的best外推下降远超5个百分点, 因此不能把Last主门禁通过解释为“训练质量完全不变”.

## 5. 训练稳定性与效率

### 5.1. Last与Best分离

| Arm | Best端点 | Last端点 | Best外推macro | Last外推macro |
|---|---:|---:|---:|---:|
| Fastest Flash | 0.937408 | 0.761488 | 0.508548 | 0.082973 |
| Canonical Flash | 0.967103 | 0.777190 | 0.598011 | 0.083402 |
| GDN | 0.968426 | 0.968426 | 0.213550 | 0.213550 |

两个Flash均在第1 epoch达到best, 随后在端点和外推任务上大幅回落. GDN持续训练到第4 epoch且best=last. 因而当前主要科学问题不是单纯“Flash能否外推”, 而是“如何在保留其外推能力的同时避免后期训练崩落”.

### 5.2. 逐epoch训练动态

按三seed对epoch-end telemetry取均值后, 两个Flash均表现为训练loss持续下降, 但验证loss和高难度`1024x256`端点在早期达到峰值后退化. GDN则在相同四epoch暴露下持续改善.

| Arm | Epoch | Train loss | Valid loss | `1024x256`端点 |
|---|---:|---:|---:|---:|
| Fastest Flash | 1 | 5.115179 | 0.190863 | 0.936617 |
| Fastest Flash | 4 | 0.037838 | 0.217582 | 0.760089 |
| Canonical Flash | 1 | 4.259112 | 0.132055 | 0.967223 |
| Canonical Flash | 4 | 0.034826 | 0.203793 | 0.776695 |
| GDN | 1 | 2.311091 | 0.308335 | 0.919052 |
| GDN | 4 | 0.008093 | 0.168816 | 0.968198 |

该现象在操作意义上符合有限MQAR训练集上的过拟合: Flash的训练目标继续改善, held-out验证目标却恶化. 但“过拟合”仍不足以概括全部现象. Fastest从第1到第2 epoch的平均验证loss由`0.190863`改善至`0.165316`, 同期端点却由`0.936617`下降至`0.864107`; 这说明模型在改善平均或简单样本loss时, 已经开始遗忘高KV、长距离关联能力. 因此更准确的描述是**早期能力形成较快, 随后发生任务选择性遗忘和后期训练退化**, 而不是已经稳定收敛.

按相同epoch或训练样本暴露衡量, Canonical Flash在第1 epoch已经达到接近GDN第4 epoch的标准端点, 且best checkpoint的长度外推明显更强, 表明Flash具有潜在的早期学习与外推优势. 但从wall time看, GDN完成四epoch只需`236.69 s`, 与Flash完成约一个epoch处于相近量级; 同时GDN的能力持续保持和改善. 因而当前结果不能解释为Flash已经获得明确的“更快收敛”优势. 在现有训练协议下, 短暂高峰是科学上的潜力信号, 依赖早停且不能保持则是训练系统上的劣势.

### 5.3. 与优化前历史结果的对照

优化前Flash通常在第4 epoch取得best checkpoint的记忆与历史artifact一致. `20260726-01-mqar-precision-profile`中, `block_len=32`的Flash共有13/15个run在第4 epoch最优, 其余2个在第3 epoch最优; 其中3090上的FP32、FP16和BF16共9/9个run均在第4 epoch最优. `20260729-02-mqar-deterministic-selected-read-regression`中, 同为`block_len=32`的A0/A1 BF16三seed共6/6个run也是best=epoch4. 历史文件名中的`b64ga4`表示train batch 64和gradient accumulation 4, 不表示`block_len=64`.

本实验两个Flash arm则共同使用`block_len=64`, 并在三seed下全部于第1 epoch达到best. 从历史A1在block32下能够稳定训练到第4 epoch可知, A1 remat本身不足以解释当前退化. Fastest和Canonical又出现相同方向的后期崩落, 因而K2、W2、K3、G1和F1等Fastest专属优化也不是共同退化的充分原因. 不过, 当前Canonical仍不是优化前A0的逐项复刻: 除`block_len`由32变为64外, 还使用了当前S1和相关backend. 因此现有证据不能把退化确定归因于某一个实现.

在这些共同变化中, `block_len=64`是优先级最高的待验证因素. 它减少逻辑记忆边界数量, 改变局部窗口、远端状态可见性和记忆更新节奏, 属于模型语义变化, 而不是纯kernel等价加速. Fastest在best checkpoint上额外表现出的seed125退化, 则说明效率backend的数值路径可能进一步放大训练轨迹敏感性, 但不能解释两个Flash共有的后期退化. 最小因果实验应在当前源码下固定初始化、数据、精度、优化器和backend, 只比较A1+S1 exact的`block_len=32`与`block_len=64`, 完成三seed四epoch配对; 只有该对照完成后, 才能判断问题主要来自block语义还是其他共享实现变化.

MQAR协议会对有限训练集重复四次, 而1B-token自然语言预训练通常接近单遍或低重复数据暴露. 因而该结果是自然语言正式训练前必须处理的稳定性风险, 但不能直接推出自然语言训练也会在相同token位置发生崩落. 自然语言路径仍需要保存多个训练阶段checkpoint, 进行token-aligned验证NLL与下游任务配对评估.

### 5.4. 小模型训练资源

| Arm | Mean wall | Mean step p50 | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Fastest Flash | 554.53 s | 0.15583 s | 1456.64 MiB | 2134.00 MiB |
| Canonical Flash | 830.23 s | 0.22902 s | 1456.64 MiB | 2195.33 MiB |
| GDN | 236.69 s | 0.05446 s | 1535.74 MiB | 2108.00 MiB |

Fastest相对Canonical的完整四 epoch wall time为`1.497x`加速, peak reserved降低约`61 MiB`. 但Fastest仍比GDN慢约`2.343x`. 这是小模型MQAR协议下的结果, 不能替代300M自然语言训练吞吐测量.

## 6. 评估与失败闭环

- 9/9正式run均完成2816个optimizer updates, loss/gradient finite, runtime fallback为0.
- 39/39完整负载batch profile和下一档batch invariance通过. 三组使用相同选定batch: 标准至`1024x256`为128, `2048x512`为64, `4096x1024`为32, 两个8190任务为16.
- Capacity search共记录27次预注册允许的OOM降档, 没有非OOM失败.
- 234个逻辑评估由195个物理执行完成; GDN best/last相同的39个事件按model-state hash复用.
- 15/15个标准`1024x256` fresh-process重复性检查与正式结果精确一致.
- 13个case的dataset hash在全部arm, seed和checkpoint role之间一致.

首次batch profile曾错误使用全序列argmax hash检查batch invariance. 未评分的non-query位置在不同batch下可产生不同预测, 但query预测和accuracy完全相同. 修复后只对`targets != -100`位置计算prediction hash, 同时保留逐样本accuracy精确比较和loss容差. 修复发生在formal训练启动前, 随后39/39 profile从头通过, 不影响科学结果.

## 7. 决策

**(1)** 预注册Last主门禁判定为`quality_retained=true`. Fastest在固定四 epoch终点相对Canonical没有超过5个百分点的均值损失.

**(2)** 该通过结论不足以提升Fastest为质量canonical. 两个Flash共同后期退化使Last差值接近, 而Best结果显示Fastest存在显著seed敏感性, 尤其seed125外推下降`0.250114`.

**(3)** 当前质量canonical继续保持`baseline-r16-joint + A1 + S1 exact`. Fastest组合继续作为300M快速资源候选, 不以本实验自动启动1B-token训练.

**(4)** 如果Fastest进入自然语言质量路径, 下一门禁应使用相同初始化和data order的短自然语言paired pilot, 保存多个训练阶段checkpoint, 同时比较validation NLL曲线, downstream指标和checkpoint选择敏感性, 不能只比较固定终点.

**(5)** GDN继续作为capacity-matched外部baseline. 它在固定训练预算下稳定性和速度明显更好; Flash的潜在优势集中在best checkpoint的长度外推能力, 两者应同时报告.

**(6)** 下一项最小稳定性诊断应优先比较当前A1+S1 exact的`block_len=32`与`block_len=64`, 而不是先将后期退化归因于Fastest专属kernel. 该对照应保持三seed、四epoch、初始化、data order和训练超参完全一致, 同时记录逐epoch训练loss、验证loss、标准端点和长度外推.

## 8. 原始证据

除checkpoint外的1712个raw文件已镜像回本机, 与3090来源的文件数, bytes和aggregate SHA256完全一致. 60个checkpoint文件保留在3090. 详细来源见[source manifest](artifacts/20260801-01-fastest-flash-vs-gdn-mqar/source-manifest.csv).
