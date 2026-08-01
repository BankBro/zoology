# Flash 后期退化因果诊断报告

## 1. 结果概览

- `experiment_id`: `20260801-02-flash-late-degradation-causal-diagnosis`.
- 状态: `completed`.
- 目标机器: RTX 3090 GPU0.
- 主实验源码: Zoology `20260801-185852-flash-late-degradation-causal-diagnosis@68d9e8e52288cf83149630825d6df37f0ebe8450`.
- Fresh-data补充源码: Zoology `20260801-185852-flash-late-degradation-causal-diagnosis@1c295200e67cffd8599183cd39965f27a7c602b2`.
- Flash-VQG源码: `20260801-103002-selected-read-native-load@182180fd7a0770caf72b2dec6e6d27616dfd31a3`.
- Plan: [实验计划](plans/20260801-02-flash-late-degradation-causal-diagnosis-plan.md).
- Artifact: [精简证据](artifacts/20260801-02-flash-late-degradation-causal-diagnosis/README.md).

主队列完成36/36个作业, 包含8条两seed四epoch正式训练、208条standard/Longer-MQAR best/last评估和8条batch-size invariance检查. 预注册条件触发后, 又完成2条seed123 fresh-per-epoch四epoch训练.

结论如下:

1. `block64/local2`的严重后期退化可在fixed与default FLA、诊断backend与最快K2/W2/K3栈中重复观察.
2. 根因主要是`local_num_blocks`控制的近场/远场可见跨度由64扩到128 token, 不是64-token block边界本身. `local_num_blocks`同时改变local window与remote boundary offset, 当前证据不能把两者进一步拆开.
3. 最快栈改为`block64/local1`后, 两seed 1024x256 validation peak均值只降低`0.021062`, final均值提高`0.132854`, peak-to-final drop从`-0.157150`缩小到`-0.003234`.
4. Fresh data只将退化组final提高`0.031805`, drop仍为`-0.200363`. 因此该现象不是重复cache导致的传统过拟合, 更准确的名称是**近场/远场可见跨度引发的持久后期动力学退化**.
5. 事后路径覆盖审计进一步解释了该现象: `block64/local2`将训练目标中按microbatch等权计量的remote-required比例从`30.578%`降到`16.355%`, 但1024及更长评估几乎`100%`依赖remote路径. 最符合当前证据的机制是**对训练距离分布和可用计算路径的捷径过拟合, 导致长程remote能力选择性遗忘**.

`block64/local1`可替代`block64/local2`成为当前最快MQAR候选, 但这属于模型语义修改. 本实验不自动授权将其用于300M自然语言正式训练或1B-token训练.

## 2. 实验合同

所有训练使用canonical init与validation cache, data seed `123`, train batch `64`, gradient accumulation `4`, AMP BF16, FP32 master weights和optimizer state, TF32关闭. 每条正式训练完成4 epochs、2816个optimizer updates和16次固定验证, early stopping关闭.

### 2.1. 因果诊断矩阵

| Arm | `block_len` | `local_num_blocks` | 有效近场跨度 | Selected backward |
|---|---:|---:|---:|---|
| `ctrl-bridge` | 32 | 2 | 64 | `torch_chunked`, chunk2048 |
| `factor-block` | 64 | 2 | 128 | `torch_chunked`, chunk2048 |
| `mechanism-window128` | 32 | 4 | 128 | `torch_chunked`, chunk2048 |
| `mechanism-boundary64` | 64 | 1 | 64 | `torch_chunked`, chunk2048 |

因果筛选固定FLA fused-gate backward为`BT64/warps4/stages2`, 再以默认autotune复核`ctrl-bridge`与`factor-block`. Seeds123/125用于确认; 只有结果不明确时才启用seed124.

### 2.2. 最快栈迁移

`fastest-current`为:

```text
baseline-r16-joint
+ A1 post-phase1 remat
+ K2 P8 persistent scan
+ W2 direct selected backward
+ K3 fixed-slot custom VJP
+ G1 head-grouped geometry
+ F1 hoisted selected forward
+ block64/local2
```

修复组仅将最后一项改为`block64/local1`; 参数量、rank、read/write top-k、优化器、精度和效率kernel全部不变.

### 2.3. Fresh-data补充

固定数据组每个epoch重复同一训练cache. Fresh组epoch0复用相同cache, epoch1至epoch3为5个训练segment分别使用独立seed生成新样本. 数据生成使用forked CPU Torch RNG并恢复NumPy状态, 不推进模型训练RNG; 两个结构组复用完全相同的20个epoch-segment cache hash.

## 3. 根因定位

### 3.1. 历史现象复现

Seed123短筛选中:

| Arm | Step707端点 | Step1238端点 | Retention |
|---|---:|---:|---:|
| `ctrl-current`, block64/local2+S1 | 0.957457 | 0.904152 | -0.053305 |
| `ctrl-bridge`, block32/local2+torch | 0.921844 | 0.946324 | +0.024480 |

当前路径复现了历史`-5.33pp`回落, 同源码下的block32桥接组则继续改善, 因而不是统计脚本或环境错误.

### 3.2. Block单因素确认

只把桥接组的`block_len`从32改为64时:

| Seed | Bridge retention | Block64 retention | 配对效应 |
|---:|---:|---:|---:|
| 123 | +0.024480 | -0.099934 | **-0.124414** |
| 125 | +0.017258 | -0.085496 | **-0.102754** |
| Mean | +0.020869 | -0.092715 | **-0.113584** |

两个seed均远低于预注册`-0.05`强因果线. Default FLA复核的两seed均值效应为`-0.093561`, 排除固定warps4是共同根因.

### 3.3. 近场跨度与block边界

机制矩阵的retention如下:

| 配置 | 近场跨度 | `idx_remote` | Seed123 | Seed125 | Mean |
|---|---:|---|---:|---:|---:|
| block32/local2 | 64 | `n-1` | +0.024480 | +0.017258 | +0.020869 |
| block32/local4 | 128 | `n-3` | -0.029426 | -0.066812 | -0.048119 |
| block64/local1 | 64 | `n` | +0.435746 | +0.041352 | +0.238549 |
| block64/local2 | 128 | `n-1` | -0.099934 | -0.085496 | -0.092715 |

在block32下将近场跨度从64扩到128的配对效应为`-0.053906/-0.084070`, 两seed均达到strong cause. 保持64-token近场跨度并改成block64没有退化证据. `block64/local1`的seed123 retention很大主要因为它在step707尚未收敛, 不能解释为绝对质量提高; 完整四epoch结果见第5节.

严格来说, `local_num_blocks`同时控制:

```python
W = local_num_blocks * block_len
idx_remote = n - (local_num_blocks - 1)
```

所以本实验定位到的是**近场/远场可见跨度合同**, 不是单独的local softmax窗口或remote read算子. 若需要更细机制, 必须设计仍保持无重叠、无缺口可见性的受控实验; 不能朴素拆开`W`与`idx_remote`, 也不能从本矩阵事后推断某一个算子单独有错.

## 4. 四epoch后期退化

### 4.1. Canonical诊断backend

| Arm | Seed | Validation peak | Final | Peak-to-final drop |
|---|---:|---:|---:|---:|
| block32/local2 | 123 | 0.966016 | 0.960797 | -0.005219 |
| block32/local2 | 125 | 0.980289 | 0.975805 | -0.004484 |
| block64/local2 | 123 | 0.958980 | 0.711465 | **-0.247516** |
| block64/local2 | 125 | 0.967816 | 0.806785 | **-0.161031** |

Block32两seed训练到末端基本保持峰值. Block64/local2两seed则分别下降`24.75pp`和`16.10pp`. 同时, block64/local2的terminal train loss为`0.032419/0.031661`, 低于block32的`0.044568/0.037767`, 但terminal valid loss更差. 这说明训练目标继续改善, 高难度held-out关联回忆却被选择性遗忘.

### 4.2. 短screen的边界

Fastest-current seed123在step1232时retention仍为`+0.014180`, 看起来稳定; 完整四epoch后却从validation peak `0.969008`跌到`0.814152`. 因此step1232不足以筛除该退化. 后续同类稳定性screen至少应运行完整四epoch, 或预注册“达到峰值后再观察两个验证窗口”的自适应停止规则.

## 5. 最快栈修复

### 5.1. 训练曲线与效率

| Arm | Seed | Validation peak | Final | Drop | Wall, s |
|---|---:|---:|---:|---:|---:|
| Fastest block64/local2 | 123 | 0.969008 | 0.814152 | -0.154855 | 543.78 |
| Fastest block64/local2 | 125 | 0.917633 | 0.758188 | -0.159445 | 545.43 |
| Fastest block64/local1 | 123 | 0.931207 | 0.929012 | -0.002195 | 533.72 |
| Fastest block64/local1 | 125 | 0.913309 | 0.909035 | -0.004273 | 535.34 |

两seed均值下:

- Validation peak: local1比local2低`0.021062`, 在用户允许的5个百分点范围内.
- Final: local1提高`0.132854`.
- Drop: 从`-0.157150`缩小到`-0.003234`.
- 小模型四epochwall: `544.60 s -> 534.53 s`, local1约`1.019x`更快.

因此local1不是用明显峰值质量损失换稳定性, 而是在当前MQAR协议下同时稳定末端并略微缩短运行时间. 该wall结果只适用于1.16M小模型, 不能替代300M资源测量.

### 5.2. Standard与Longer-MQAR

以下结果对seeds123/125取均值; `true extrapolation`不含1024x256锚点, 只包含4个更长任务.

| Arm | Checkpoint | Standard 8 macro | Standard 1024x256 | True extrapolation 4 macro |
|---|---|---:|---:|---:|
| Fastest block64/local2 | Best | 0.985667 | 0.922254 | 0.441406 |
| Fastest block64/local2 | Last | 0.971403 | 0.787400 | 0.092636 |
| Fastest block64/local1 | Best | **0.986781** | 0.919836 | **0.451911** |
| Fastest block64/local1 | Last | **0.986781** | **0.919836** | **0.451911** |

Local1的saved best=last, 第4 epoch仍保持质量. 相对local2 best, local1的standard端点只低`0.002418`, standard macro提高`0.001114`, 四外推macro提高`0.010505`. 相对local2 last, 四外推macro提高`0.359275`.

这支持`block64/local1`替换`block64/local2`成为当前最快MQAR候选. 但它不自动替换自然语言质量canonical, 因为近场/远场跨度是模型语义, 自然语言最优值可能与MQAR不同.

## 6. Fresh-data因果补充

### 6.1. Loader canary与数据审计

- 两个fresh arm的epoch0四次validation在1024x256 accuracy、总体accuracy和valid loss上分别与fixed-data逐值完全相等.
- 两个arm使用相同的4 epochs × 5 segments数据manifest.
- 20条epoch-segment记录具有20个唯一cache SHA256.
- 两条训练均完成2816 updates, checkpoint、FP32 model/optimizer dtype、GradScaler和runtime fallback审计通过.

### 6.2. 结果

| Arm | Exposure | Peak | Final | Drop | Terminal train loss | Terminal valid loss |
|---|---|---:|---:|---:|---:|---:|
| block32/local2 | Fixed repeat | 0.966016 | 0.960797 | -0.005219 | 0.044568 | 0.071263 |
| block32/local2 | Fresh per epoch | 0.968926 | 0.965371 | -0.003555 | 0.043977 | 0.064735 |
| block64/local2 | Fixed repeat | 0.958980 | 0.711465 | -0.247516 | 0.032419 | 0.255386 |
| block64/local2 | Fresh per epoch | 0.943633 | 0.743270 | -0.200363 | 0.031518 | 0.225542 |

Fresh data使退化组final提高`0.031805`, drop改善`0.047152`, 但未达到预注册5个百分点改善线, 更没有将drop恢复到`-0.02`以内. 训练loss仍继续下降而验证能力显著回落.

因此, 数据重复可能轻微放大退化, 但不是充分原因. 终态分类为`persistent_window_dynamics`, 不再将该问题简单写成“Flash在有限MQAR cache上过拟合”.

## 7. 局部路径捷径机制审计

本节是正式训练完成后的只读结构审计, 不属于预注册因果矩阵, 也没有修改模型或实验结果. 审计使用正式训练cache `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`、已登记的standard/Longer-MQAR数据和训练器真实loss聚合方式. 汇总数据见[路径覆盖率Artifact](artifacts/20260801-02-flash-late-degradation-causal-diagnosis/local-path-coverage.csv).

### 7.1. 路径语义

Flash的local路径不是任意滑动窗口, 而是按逻辑block对齐的因果窗口. `local_num_blocks`同时控制:

```python
local_window = local_num_blocks * block_len
idx_remote = n - (local_num_blocks - 1)
```

`state[..., n]`表示进入第`n`个block前的记忆状态. 因此, 在当前`block_len=64`下:

| 配置 | Local可见内容 | 写入首次可被remote读取 |
|---|---|---|
| `local1` | 当前block内的因果前缀 | 下一block, 即`j + 1` |
| `local2` | 前一完整block加当前因果前缀 | 下下个block, 即`j + 2` |

这不是单纯“多看64个token”. `local2`一方面给local分支增加了短距离直接访问, 另一方面把相同信息交给remote分支的时间推迟一个block. Local与remote coarse路径还共享softmax分母, 因而local候选增多会改变coarse路径竞争. Residual `M`读取也复用该权重, 但随后RMSNorm会抵消大部分公共标量, 所以不能把其退化简单归因于分母缩放; 对`M`更可信的影响是读取边界更旧、监督延迟更长和训练期有效读取次数减少.

### 7.2. 训练目标的路径覆盖

正式训练cache只有64、128和256三类序列长度. 对每个监督query, 若其目标value位置落入block-aligned local因果窗口, 记为`local-visible`; 否则记为`remote-required`. `block64`下的精确覆盖如下:

| 训练segment | 监督query数 | `local1` local-visible | `local2` local-visible |
|---|---:|---:|---:|
| 64x4 | 400,000 | 100.000% | 100.000% |
| 128x8 | 160,000 | 76.159% | 100.000% |
| 256x16 | 320,000 | 48.838% | 76.885% |
| 256x32 | 640,000 | 0.000% | 60.418% |
| 256x64 | 1,280,000 | 0.000% | 15.606% |
| **按query数汇总** | **2,800,000** | **24.219%** | **49.731%** |

`local2`将`714,323`个训练目标, 即全部监督query的`25.511`个百分点, 从remote-required变为local-visible. 训练器先对每个microbatch内的监督token取mean, gradient accumulation再等权累积microbatch, 因而真实训练目标不能只按query总数加权. 按2815个训练microbatch等权计算:

| 配置 | 近场跨度 | Query-count remote-required | Microbatch-mean remote-required | 四epoch结果 |
|---|---:|---:|---:|---|
| block32/local2 | 64 | 72.322% | 28.951% | 稳定 |
| block64/local1 | 64 | 75.781% | 30.578% | 稳定 |
| block32/local4 | 128 | 41.016% | 13.645% | 退化 |
| block64/local2 | 128 | 50.269% | 16.355% | 退化 |

从`block64/local1`改成`block64/local2`后, remote-required的有效训练权重下降`14.223`个百分点, 只剩原来的约`53.5%`. 两种64-token近场配置均稳定, 两种128-token近场配置均退化, 与第3节的因果矩阵一致.

### 7.3. 训练与评估的路径错配

高难度评估的目标几乎完全落在local窗口外:

| 评估case | 监督query数 | `local1` remote-required | `local2` remote-required |
|---|---:|---:|---:|
| Standard 1024x256 | 256,000 | 100.000% | 99.9996% |
| Longer 1024x256 | 128,000 | 100.000% | 100.000% |
| Longer 2048x512 | 256,000 | 100.000% | 100.000% |
| Longer 4096x1024 | 512,000 | 100.000% | 100.000% |
| Longer 8190x512 | 256,000 | 100.000% | 99.9910% |
| Longer 8190x2047 | 1,023,500 | 100.000% | 100.000% |

因此, `local2`在短训练序列中经常能从local路径直接取得正确value, 但在1024及更长评估中, 扩大的窗口几乎只增加干扰项, 正确答案仍必须依靠remote记忆. Fresh-per-epoch只更换token身份, 没有改变这种距离分布和路径覆盖, 所以不能消除退化.

现有结果还显示, 退化模型在短任务和部分512长度任务上保持较高准确率, 到1024及更长任务才明显崩塌. 这更像有效记忆时域、抗碰撞或遗忘稳定性下降, 而不是remote机制整体失效.

### 7.4. 机制结论与证据边界

当前最合理的解释是: 128-token近场为短训练样本提供了更容易优化的local捷径, 同时减少并延迟remote路径收到的有效监督. 随着训练继续, 优化器仍能降低短分布train loss, 但remote routing、write/read和forget相关参数受到的约束变弱, 最终表现为长程能力选择性遗忘. 这应称为**对训练距离分布和可用计算路径的捷径过拟合**, 而不是传统的固定样本记忆过拟合.

直接证据只包括路径语义、覆盖率、训练器加权方式以及稳定/退化结果的对应关系. 本实验没有逐epoch测量local/remote质量占比、`Den_far / Den_total`、selected mass、路径梯度、forget half-life或最先漂移的组件, 因而“优化器逐渐依赖local捷径”仍是高一致性的机制解释, 不是已直接观测的内部状态. 此外, local window宽度和`idx_remote`共同构成无重叠、无缺口的路径划分, 不能用简单拆开其中一个参数的方式声称完成单因素验证. 该结论也尚不能直接外推到300M自然语言训练.

## 8. 审计与失败边界

- 主preflight的cache、init、model parameter/state hash、Python/Torch/CUDA/Triton/FLA版本、GPU、源码clean、FLA config和17个arm合同全部通过.
- 主队列36/36作业返回0, 无训练、checkpoint或evaluation失败.
- 8条正式训练均完成2816 updates, model与optimizer state保持FP32, fallback为0.
- 208条formal evaluation记录覆盖4个arm × 2 seeds × best/last × 13 cases.
- 8/8条batch-size invariance检查通过; query prediction、accuracy、dataset hash和loss均满足门槛.
- Fresh-data两条训练均完成, 20个数据cache hash唯一且跨结构完全对应.

主队列运行于Zoology `68d9e8e`; fresh loader是预注册条件触发后的独立补充, 仅新增于commit `1c29520`, 没有修改主实验模型或已有结果.

## 9. 决策与下一步

**(1)** 当前不再把Flash后期下降解释为K2/W2/K3效率kernel的共同副作用. 诊断backend和最快backend都支持近场/远场跨度是主要因素.

**(2)** 当前最快MQAR候选更新为:

```text
baseline-r16-joint
+ A1 post-phase1 remat
+ K2 P8 persistent scan
+ W2 direct selected backward
+ K3 fixed-slot custom VJP
+ G1 head-grouped geometry
+ F1 hoisted selected forward
+ block64/local1
```

**(3)** `block64/local2`不再作为四epochMQAR稳定配置. 若必须使用, 应保存早期checkpoint并明确其任务选择性遗忘风险.

**(4)** 自然语言300M下一门禁应在相同初始化、data order和token预算下配对比较`block64/local2`与`block64/local1`, 同时保留当前质量参考. 必须报告validation NLL、多阶段checkpoint和下游任务; MQAR结论不自动决定自然语言最佳近场跨度.

**(5)** 若继续追究微观机制, 不应直接拆开local window宽度与`idx_remote`偏移, 因为朴素拆分会引入路径重叠或可见性缺口. 更安全的低成本验证是将local2的best/last checkpoint分别在local1/local2推理合同下交叉评估: 若last在local1下仍差, 更支持remote参数已经遗忘; 若明显恢复, 更支持前向路径竞争是主要因素.

**(6)** 未来低成本screen不能停在step1232. Fastest seed123证明退化可在该点之后出现; 应使用完整4 epochs或peak后追加固定验证窗口.

**(7)** 后续机制trace应至少记录`Den_far / Den_total`、coarse local/remote输出范数、selected mass、residual injection比例、`logf`对应的记忆半衰期、routing entropy和路径梯度. 这些量能区分“local前向压制remote”和“remote参数随训练遗忘”, 但不影响当前将`block64/local1`作为MQAR稳定候选的决策.

## 10. 原始证据

Raw、checkpoint和fresh cache保留在3090:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260801-02-flash-late-degradation-causal-diagnosis/outputs/3090/
20260801-late-degradation-01/
```

Git中的精简artifact包含208条evaluation明细、训练曲线、机制效应、fresh-data配对、local/remote路径覆盖率与源码/作业审计. Checkpoint和大体积raw日志不进入Git.
