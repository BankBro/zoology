# 面向长上下文建模的量化索引残差动态记忆机制研究

**论文关键词:** 长上下文建模, 高效序列模型, 关联回忆, 量化索引, 残差动态记忆, 信息保持与检索

# 一、课题来源及研究的目的和意义

## 1.1 课题来源

本课题来源于导师指导下的长上下文语言模型与高效序列建模研究方向, 并由本人结合相关文献调研和前期原型探索进一步凝练形成。长上下文建模在长文档问答, 代码理解, 检索增强生成和长对话建模等场景中具有重要应用价值。这类任务的难点不只是输入文本长度变长, 更在于关键信息常常被埋在大量上下文和干扰信息之中, 模型需要在后续查询时准确保持并读出前文中的实体关系, 变量绑定, 局部事实和上下文约束。

例如在真实长文档问答中, 用户可能只关心长篇技术文档中的某一个字段, 某一项结论或某一组实体关系。模型如果只是能够接收很长的输入, 但无法在后文准确找回前文出现过的关键事实, 仍然不能真正满足长上下文任务需求。因此, 长上下文能力可以进一步抽象为一种关联回忆能力: 模型需要把前文中出现的键值绑定或事实关系保存下来, 并在后续查询时稳定读出正确答案。

标准 Transformer 的全注意力机制可以让每个 token 直接访问所有历史 token, 因而在信息访问和关联回忆方面具有较强能力。但全注意力的计算和显存开销会随序列长度增加而快速增长, 在更长上下文场景中训练和部署成本较高。线性注意力, 状态空间模型, 门控循环模型等高效序列模型通过有限状态或压缩注意力降低成本, 但在高负载长上下文中可能出现不同信息混在同一状态中, 旧信息被覆盖, 或查询时读出错误等问题。基于这一背景, 本课题拟研究一种面向长上下文建模的量化索引残差动态记忆机制, 探索如何在计算成本可控的条件下更有效地组织, 校正和利用历史信息。

## 1.2 研究目的

本课题旨在面向长上下文建模场景, 设计并验证一种能够提升历史信息保持与检索能力的动态记忆机制。研究重点不是单纯扩大上下文窗口, 也不是单纯增加模型参数或记忆容量, 而是关注高效序列模型在压缩历史信息之后能否仍然准确找回关键信息。

围绕这一目标, 本研究拟回答三个层层递进的问题。第一, 当上下文变长, 关键信息变多时, 如何避免不同历史信息在有限状态中混在一起。第二, 当多个 token 被分配到同一个记忆槽后, 如何校正粗粒度记忆带来的读出误差。第三, 如何通过公平基线对比, 模块消融, 多随机种子和长度外推实验, 证明性能提升来自机制设计本身, 而不是单纯容量增加或实验偶然。

为回答上述问题, 本课题将把量化索引用于历史信息的分槽组织, 把粗粒度记忆用于低成本保存和读出远程信息, 并在每个记忆槽内引入残差动态记忆来补偿槽内冲突和近似误差。最终目标是形成一种结构清晰, 可实验验证, 能够与现有高效序列模型进行系统对比的长上下文动态记忆建模方法。

## 1.3 研究意义

本课题具有一定的理论研究意义和工程应用价值。理论上, 长上下文建模的关键不仅是扩展输入长度, 还包括如何在有限计算成本下有效保持, 组织和读取历史信息。现有高效序列模型虽然降低了长序列计算开销, 但通常需要将历史信息压缩到有限状态中, 这会带来记忆冲突, 信息遗忘和读出误差。本课题从“压缩后还能不能准确找回”这一角度出发, 研究量化索引分槽组织和槽内残差动态校正的结合方式, 可为高效序列模型的记忆组织机制提供新的设计思路。

工程上, 长上下文能力直接影响长文档问答, 代码理解, 检索增强生成, 长对话建模和专业文档分析等实际应用。如果模型能够在较低计算和显存成本下更稳定地利用前文信息, 将有助于提升长文本处理系统的效率和可用性。特别是在推理成本, token 开销和部署资源受到限制的场景中, 一种更有效的动态记忆机制有望在保持效率优势的同时改善远距离信息检索能力。

# 二、国内外研究现状评述及存在问题分析

## 2.1 国内外研究现状评述

近年来, 长上下文建模逐渐成为语言模型研究中的重要方向。标准 Transformer 依靠全注意力机制显式访问上下文中的所有历史 token, 在长文本理解, 信息检索和上下文依赖建模中表现较强, 但其计算和显存开销会随序列长度增加而快速增长。因此, 如何在降低长序列计算成本的同时保持较强建模能力, 成为高效序列模型研究的核心问题。

为解决全注意力的高开销问题, 研究者提出了线性注意力, 状态空间模型和门控循环模型等高效序列建模方法。Katharopoulos 等人在 ICML 2020 提出的 Linear Transformer 将自回归注意力改写为线性递推形式, 使模型能够用有限矩阵状态压缩历史信息; Schlag 等人在 ICML 2021 从 fast-weight 视角进一步指出, 线性注意力可以理解为一种动态更新临时矩阵记忆的快权重模型。

状态空间模型方面, Gu 和 Dao 提出的 Mamba 通过选择性状态空间机制增强模型对输入内容的选择能力; Dao 和 Gu 提出的 Mamba2 / SSD 框架进一步将状态空间模型与注意力形式联系起来, 提升了效率和可扩展性。这类方法强调通过递推状态承载历史信息, 在长序列建模中具有较好的效率优势。

在动态记忆更新方面, GLA, DeltaNet 和 GDN 等方法具有代表性。Yang 等人在 ICML 2024 提出的 GLA 在矩阵状态中加入数据相关门控, 使模型能够动态控制历史信息的保留程度; Yang 等人在 NeurIPS 2024 提出的 DeltaNet 使用 delta rule 对已有 key-value 记忆进行定向更新, 缓解普通线性注意力只累加, 不精确改写的问题; Yang, Kautz 和 Hatamizadeh 在 ICLR 2025 提出的 GDN 进一步结合门控和 delta rule, 兼顾快速遗忘与精确记忆修改, 是本课题最直接相关的高效动态记忆基线之一。

测试时记忆也是近年来长上下文建模中的重要方向。Sun 等人提出的 TTT Layers 通过在测试序列上更新可学习的 hidden state 来增强上下文记忆; Behrouz, Zhong 和 Mirrokni 提出的 Titans 引入 neural long-term memory 来存储更长期的历史信息。这类方法与本课题都关注动态记忆, 但更侧重测试时学习和长期神经记忆; 本课题则聚焦于前向建模过程中的分槽记忆组织和残差校正。

此外, 向量量化为长上下文信息压缩与索引组织提供了重要思路。Lingle 提出的 Transformer-VQ 通过量化 key 和缓存机制实现线性时间的 dense self-attention; Liu 等人提出的 LongVQ 将 VQ 与 structured memory 结合, 用固定长度 codebook 压缩全局信息。与这些方法不同, 本课题不只关注量化压缩本身, 而是进一步研究多个 token 被组织到同一记忆槽之后, 如何利用残差动态记忆校正粗粒度读出误差和槽内冲突。

## 2.2 存在问题分析

现有研究虽然推动了长上下文高效建模, 但从信息保持与检索角度看, 仍存在三个需要进一步解决的问题。

第一, 有限状态压缩容易导致记忆混叠。线性注意力, 状态空间模型和门控循环模型通常需要将历史信息压缩到有限状态中。当上下文变长, 关键信息数量增加时, 不同实体关系, 变量绑定和键值关系可能被压入相同或相近的状态表示, 造成记忆冲突, 信息遗忘和读出误差。也就是说, 高效模型虽然降低了计算成本, 但需要回答“压缩后还能不能准确找回”的问题。

第二, 现有动态记忆方法对历史信息的显式分槽组织仍不足。GLA, DeltaNet, Mamba2 和 GDN 等方法通过门控, delta rule 或状态递推改善记忆更新, 但多数方法仍主要依赖整体状态来承载历史信息, 对大量历史事实如何被组织到不同可读写位置中的问题关注不足。在高负载关联回忆场景中, 如果缺少显式组织机制, 不同信息仍可能相互干扰。

第三, 量化索引方法对槽内误差校正关注不足。Transformer-VQ, LongVQ 等方法说明了量化索引在长序列压缩和缓存组织中的价值, 但多个 token 被映射到同一码本槽后, 槽内仍可能存在粗粒度近似误差和细节冲突。如果只进行量化分槽, 而不进一步校正每个槽内部的读出偏差, 模型在复杂查询和高负载上下文中仍可能读错。

基于上述问题, 本课题拟研究一种量化索引残差动态记忆机制。其核心思路是: 先通过量化索引把历史信息分配到不同记忆槽中, 降低不同信息之间的全局混叠; 再在每个记忆槽内部维护粗粒度记忆和残差动态记忆, 用残差校正项补偿量化聚合带来的读出误差。这样既保留高效模型的低成本优势, 又增强模型在长上下文场景中的信息保持与检索能力。

# 三、课题主要研究内容和拟解决的关键问题

## 3.1 课题主要研究内容

本课题围绕长上下文建模中的历史信息保持与检索问题, 研究一种“量化索引分槽组织 + 槽内残差动态校正”的动态记忆机制。主要研究内容包括以下四个方面。

第一, 设计量化索引残差动态记忆机制。针对长上下文中历史信息数量多, 容易混叠的问题, 本课题拟通过量化索引将历史信息组织到不同记忆槽中, 形成分槽动态记忆结构。在每个记忆槽内, 模型同时维护粗粒度统计记忆和残差动态记忆。粗粒度记忆负责低成本保存和读取历史信息的整体趋势, 残差动态记忆负责校正量化聚合后的近似误差和槽内冲突。

第二, 在可控任务中验证信息保持与检索能力。本课题将以 MQAR 等关联回忆任务作为主要可控验证场景, 分析模型在不同上下文长度, 信息负载和查询数量下的表现。通过高负载切片, 长度外推和不同容量设置, 观察量化索引与残差校正是否能够缓解有限状态压缩下的记忆混叠和读出误差。

第三, 开展公平基线对比和模块消融实验。本课题将本方法与 GLA, DeltaNet, Gated DeltaNet, Mamba2 和全注意力 Transformer 等模型进行对比, 并通过去除残差动态记忆, 仅保留粗粒度量化记忆, 调整码本规模, 调整残差记忆 rank, 控制 active capacity 和补充多随机种子等实验, 分析模型效果提升究竟来自量化分槽, 残差校正, 记忆容量还是训练偶然。

第四, 开展自然语言及长上下文验证。除 MQAR 等机制任务外, 本课题计划进一步在自然语言场景中验证方法的泛化能力。基础实验包括在 FineWeb-Edu 等语料上进行小规模语言建模训练, 在 PG-19 等长文档数据上评估困惑度, 位置相关损失和远距离上下文利用情况, 并结合 AR-hit, 重复 n-gram, 实体回忆等文本切片分析模型对历史信息的保持与检索能力。若实验条件允许, 将进一步选取 RULER 或 LongBench 的部分代表性子任务进行补充评估。

## 3.2 拟解决的关键问题

本课题拟解决的关键问题凝练为以下三个方面, 并与后续方法设计和实验验证一一对应。

第一, 长上下文历史信息被压缩到有限状态后, 如何减少不同信息之间的混叠和冲突。现有高效序列模型为了降低成本, 通常不再逐 token 保存全部历史, 而是把历史写入有限状态。本课题拟研究如何利用量化索引把不同历史信息分配到不同记忆槽中, 从组织方式上降低不同事实和键值绑定之间的相互干扰。

第二, 多个 token 被组织到同一记忆槽后, 如何校正粗粒度量化记忆带来的读出误差。量化分槽可以改善全局组织, 但同一个槽内仍然可能聚合多个相近但不完全相同的信息。本课题拟研究如何在每个槽内引入残差动态记忆, 用 gated-delta 式更新对粗粒度读出与真实 value 之间的误差进行建模和补偿, 从而提升读出准确性。

第三, 如何证明方法优势来自机制设计本身, 而非单纯容量增加, 随机种子偶然或任务偏置。本课题拟通过 capacity-fair 基线对比, 多随机种子复现实验, 码本规模与残差 rank 消融, 去残差和粗粒度-only 变体, 长度外推评估, 自然语言长上下文验证以及效率分析, 系统判断量化索引残差动态记忆机制的有效性, 稳定性和工程可行性。

# 四、拟采取的研究方法, 研究方案及其可行性分析

## 4.1 总体研究路线

本课题采用“问题分析 - 方法机制 - 可控验证 - 真实场景验证 - 机制与效率分析”的总体研究路线。首先从现有高效长上下文模型的不足出发, 将问题凝练为有限状态混叠, 槽内读出误差和机制有效性验证三个层面; 随后设计量化索引分槽, 粗粒度远程记忆, 残差动态校正和输出融合机制; 再通过 MQAR 机制验证, capacity-fair 基线对比, 模块消融, 多随机种子和 longer-MQAR 长度外推进行验证; 最后补充自然语言长上下文实验和效率分析, 形成从问题到方法再到验证的闭环。

总体研究路线图如下。

[此处插入图 1: 总体研究路线图]

图 1 不是单纯的实验流程图, 而是从现有不足到关键问题, 方法机制, 验证方案和预期目标的完整研究路线。现有不足包括有限状态压缩导致记忆冲突, 缺少显式历史信息分槽组织, 以及粗粒度量化读出存在槽内误差; 对应的关键问题是历史信息如何有效组织, 读出误差如何校正, 以及如何区分机制优势与容量优势; 对应方法为量化索引分槽, 粗粒度远程记忆, 残差动态校正和输出融合; 验证方案包括 MQAR, capacity-fair 对比, 消融稳定性, longer-MQAR 长度外推和自然语言长上下文验证。

## 4.2 具体研究方案

第一, 设计并实现量化索引残差动态记忆机制。本课题拟构建一种“局部上下文建模 + 量化索引远程记忆 + 残差动态校正”的长上下文建模结构, 并将该结构命名为 Flash-VQRM, 即 Flash Vector-Quantized Residual Memory Attention。模型首先将输入 token 投影为 query, key 和 value 表示, 分别记为 $$q_t$$, $$k_t$$ 和 $$v_t$$。近邻上下文由局部建模模块处理, 远距离历史信息则通过量化索引记忆进行压缩存储与读取。

模型维护一个可学习 codebook, 每个 codeword 对应一个记忆槽中心 $$c_s$$。历史 token 的 key 根据与 codeword 的匹配关系被分配到不同记忆槽中, 从而实现历史信息的分槽组织。在每个记忆槽内, 模型维护粗粒度统计状态 $$\mathcal{G}_s$$ 和 $$\mathcal{L}_s$$, 其中 $$\mathcal{G}_s$$ 表示该槽内 value 的加权累积, $$\mathcal{L}_s$$ 表示对应写入质量。粗粒度记忆的读出可写为 $$\mu_s=\frac{\mathcal{G}_s}{\mathcal{L}_s+\varepsilon}$$。该粗粒度记忆提供低成本远程读出, 但由于多个 token 可能被聚合到同一记忆槽中, 其读出结果可能存在近似误差。

为校正这类误差, 本研究进一步引入槽内残差动态记忆。具体而言, 模型将当前 value 与粗粒度读出之间的差异视为残差信息, 可表示为 $$u_{t,s}=v_t-\operatorname{sg}(\mu^{pre}_{t,s})$$, 并在每个记忆槽内维护低秩残差矩阵 $$M_s$$。写入时, 模型根据 $$k_t-c_s$$ 构造写入地址, 以当前残差为目标对 $$M_s$$ 进行 gated-delta 式误差校正更新; 读出时, 模型根据 $$q_t-c_s$$ 构造读出地址, 从 $$M_s$$ 中读取残差校正项, 用于补偿粗粒度量化记忆的读出误差。最终输出由局部上下文信息, 粗粒度远程记忆和残差校正信息融合得到, 可概括为 $$o_t=o_t^{base}+\lambda_t\operatorname{RMSNorm}(u_t^{res})$$。

具体模型结构如下。

[此处插入图 2: 模型结构示意图]

第二, 开展机制验证, 基线对比与模块消融实验。本课题将以 MQAR 等任务作为可控验证场景, 分析模型在不同上下文长度, 信息负载和查询数量下的表现; 同时将本方法与 Gated DeltaNet, DeltaNet, Mamba2, GLA 和全注意力 Transformer 等模型进行对比。为了避免将容量增加误认为机制改进, 本课题将采用 active capacity, 参数量和训练预算等口径进行公平比较, 并补充 no-residual, coarse-only, 不同 codebook size, 不同 residual rank 和不同 seed 的消融实验。

第三, 开展自然语言及长上下文验证。除 MQAR 等机制任务外, 本课题计划进一步在自然语言场景中验证方法的泛化能力。基础实验包括在 FineWeb-Edu 等语料上进行小规模语言建模训练, 在 PG-19 等长文档数据上评估困惑度和位置相关损失, 并结合 AR-hit, 重复 n-gram, 实体回忆等文本切片分析模型对历史信息的保持能力。若实验条件允许, 将进一步选取 RULER 或 LongBench 的部分代表性子任务进行补充评估。

第四, 开展机制指标与效率分析。本课题将通过记忆槽使用分布, 残差注入强度, 记忆状态范数, 长上下文切片表现等指标, 分析量化索引和残差动态记忆是否真正参与历史信息组织与校正; 同时统计训练速度, 显存占用和不同序列长度下的运行表现, 评估该机制在实际训练和部署中的工程可行性。

## 4.3 可行性分析

从理论基础看, 线性注意力, 状态空间模型, 门控 Delta 网络, Transformer-VQ, LongVQ 等相关工作已经表明, 动态记忆, 量化索引和高效长上下文建模具有明确研究价值。本课题在已有研究基础上进一步关注“分槽组织”和“槽内残差动态校正”, 研究目标较为清晰。

从工程基础看, 前期已完成方法原型实现和初步测试, 原型中已包含量化路由, 粗粒度记忆状态, 残差动态记忆状态, 残差写入, 残差读出和输出融合等核心模块。后续实验以小规模可控模型为主, 通过统一训练预算, 参数量, active capacity 和评估设置进行比较, 整体实验规模可控。

从数据与评估条件看, MQAR 可用于机制验证, FineWeb-Edu, PG-19 等公开语料可用于自然语言训练和长文档评估, RULER, LongBench 等公开评测可作为补充验证。因此, 本课题具备较明确的数据来源和评估路径。

从风险控制看, 本课题可能面临自然语言训练成本较高, 部分基线实现难度较大, 机制任务结果无法完全迁移到真实文本等风险。对此, 本课题将采用分阶段实验策略: 优先完成 MQAR, 基线对比和消融实验, 再开展小规模自然语言训练, 最后根据实验条件选择部分公开长上下文任务进行补充评估, 保证课题在毕业设计周期内可完成, 可分析, 可落地。

# 五、课题研究的特色与创新之处

本课题的特色与创新主要体现在将量化索引的“分槽组织能力”和动态记忆的“误差校正能力”结合起来, 面向长上下文高效模型中的信息保持与检索瓶颈进行机制设计。已有 VQ 长序列方法多侧重 key 量化, 缓存压缩或全局信息抽象; 本研究关注的不是量化索引本身, 而是量化索引之后每个记忆槽内部如何继续处理多个 token 聚合带来的误差。

具体而言, 本课题在每个量化记忆槽中同时维护粗粒度记忆和残差动态记忆。粗粒度记忆用于以较低成本保存和读取历史信息的整体趋势, 残差动态记忆用于补偿同槽信息聚合后的细节误差和读出偏差。通过这种方式, 模型不只是把历史信息压缩得更省, 还尝试让压缩后的信息更容易被准确找回。

从更高层面看, 该机制有望为低成本长上下文模型提供一种可解释的记忆组织方式: 量化索引回答“信息放在哪里”, 粗粒度记忆回答“槽内大致保存什么”, 残差动态记忆回答“读错的细节如何校正”。如果该机制能够在 MQAR, 长度外推和自然语言长上下文任务中保持稳定优势, 将有助于提升长文档问答, 代码理解和检索增强生成等应用中的信息检索效率, 并为降低长上下文模型的计算, 显存和 token 开销提供参考。

# 六、计划进度和预期研究成果

## 6.1 计划进度

2026 年 5 月至 2026 年 7 月: 完成开题报告和文献调研, 明确模型结构, 实验任务和基线设置; 整理已有 MQAR 与 longer-MQAR 预研结果, 固化实验记录和评估口径。

2026 年 8 月至 2026 年 10 月: 完成模型原型完善, MQAR 正式实验, capacity-fair 基线对比和模块消融实验; 补充 codebook size, residual rank, seed 稳定性和效率统计。

2026 年 11 月至 2027 年 1 月: 开展自然语言建模和长文档验证实验, 包括 FineWeb-Edu 小规模语言建模, PG-19 长文档评估和文本切片分析; 条件允许时补充 RULER 或 LongBench 部分任务。

2027 年 2 月至 2027 年 5 月: 整理实验结果, 完成论文撰写, 修改, 送审和答辩准备; 完成代码, 配置, checkpoint, 实验表格和可复现说明的归档。

## 6.2 预期研究成果

本课题预期形成一种面向长上下文建模的量化索引残差动态记忆机制, 完成与 GDN, DeltaNet, GLA, Mamba2 和全注意力 Transformer 等基线的系统对比, 并形成可复现实验配置, 主要实验结果表格, 模型代码和学位论文。若进展顺利, 进一步争取形成发明专利一项, 会议论文一篇, GitHub 开源代码和 Hugging Face 开源权重。

# 七、前期预研工作基础

前期已围绕长上下文建模, 高效序列模型, 动态记忆机制和量化索引方法开展了较系统的文献调研, 重点学习了 Linear Transformer, Mamba/Mamba2, GLA, DeltaNet, GDN, Transformer-VQ, LongVQ, TTT, Titans, FwPKM 等相关工作, 初步明确了本课题与现有方法的关系和差异。

在方法设计方面, 已完成量化索引残差动态记忆机制的初步方案设计, 明确了局部上下文建模, 量化索引远程记忆和残差动态校正的整体结构。前期原型中已包含量化路由, 粗粒度记忆状态, 残差动态记忆状态, 残差写入, 残差读出和输出融合等核心模块, 为后续系统实验提供了工程基础。

在 MQAR 预研方面, 已完成一组 capacity-fair 的初步实验。在 131k active capacity 档, Flash-VQRM 单 seed 结果已明显高于 GDN multi-seed mean, 尤其是在 1024x256 hard slice 上表现更突出。其中 Flash-VQRM cb64-r16 配置在 1024x256 上达到约 0.9687 accuracy, 而 GDN 131k 档 h1-ev8 和 h2-ev16 的均值约为 0.81 和 0.80。该结果说明量化索引残差动态记忆在高负载关联回忆任务中具有较好的潜力, 但当前 131k 档结果仍属于 single-seed trend, 后续需要补充多随机种子来固化稳定性。

在长度外推预研方面, 已使用 standard MQAR 训练 checkpoint 进行 longer-MQAR eval-only 测试。在 2048x512, 4096x1024 和 8190x2047 等更长切片上, GDN accuracy 快速下降并接近失效区间, 而 Flash-VQRM 仍保留明显可用的关联回忆能力。例如 Flash-VQRM cb256-r10 配置在 1024x256, 2048x512, 4096x1024 和 8190x2047 上分别取得约 0.9137, 0.7081, 0.4590 和 0.2327 的平均 accuracy; Flash-VQRM cb64-r16 配置在 131k active capacity 下分别取得约 0.9691, 0.8230, 0.4689 和 0.1622 的 single-seed trend。相比之下, GDN h2-ev10 在相同四个长度上的均值约为 0.8360, 0.3478, 0.0772 和 0.0063。该结果为本课题继续开展长度外推和长上下文验证提供了初步依据。

目前课题已具备继续推进的基础, 但仍需要完成更严格的多 seed 复现实验, 模块消融, 自然语言建模训练, 长文档评估, 公开长上下文任务测试和完整基线对比。后续将基于已有原型和初步实验结果, 进一步完善模型结构, 固化实验口径, 补充机制分析与效率分析。

# 八、涉及实验项目的安全风险分析与防控

本课题主要开展计算机模型训练, 数据处理和算法实验, 不涉及化学, 生物, 动物, 人体或临床实验, 整体安全风险较低。主要风险包括服务器长时间运行导致的设备过热, 进程异常, 数据误删, 结果覆盖, 账号密钥泄露和公开数据合规问题。

针对设备和运行风险, 实验过程中将通过 GPU 温度和显存监控, 训练日志记录, 失败任务自动退出和关键 checkpoint 备份等方式降低风险。针对数据和结果风险, 将使用版本管理, 配置归档, manifest 记录和 artifact 目录保存关键结果, 避免结果覆盖或口径混乱。针对账号和密钥风险, 将避免在公开仓库中保存 token, app secret 和私人数据, 并仅使用合规公开数据集开展训练和评估。

# 九、主要参考文献

[1] VASWANI A, SHAZEER N, PARMAR N, 等. Attention is All you Need[C/OL]//Advances in Neural Information Processing Systems: 卷 30. Curran Associates, Inc., 2017. https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[2] ZHAO T, JONES L. Fast-weight Product Key Memory[J]. arXiv preprint arXiv:2601.00671, 2026.

[3] YANG S, KAUTZ J, HATAMIZADEH A. Gated Delta Networks: Improving Mamba2 with Delta Rule[C/OL]//International Conference on Learning Representations. 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf.

[4] YANG S, WANG B, SHEN Y, 等. Gated Linear Attention Transformers with Hardware-Efficient Training[C/OL]//Proceedings of the 41st International Conference on Machine Learning. PMLR, 2024: 56501-56523. https://proceedings.mlr.press/v235/yang24ab.html.

[5] SUN Y, LI X, DALAL K, 等. Learning to (Learn at Test Time): RNNs with Expressive Hidden States[C/OL]//Proceedings of the 42nd International Conference on Machine Learning. PMLR, 2025: 57503-57522. https://proceedings.mlr.press/v267/sun25h.html.

[6] SCHLAG I, IRIE K, SCHMIDHUBER J. Linear Transformers Are Secretly Fast Weight Programmers[C/OL]//Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021: 9355-9366. https://proceedings.mlr.press/v139/schlag21a.html.

[7] LIU Z, WANG L, LI S, 等. LongVQ: long sequence modeling with vector quantization on structured memory[C/OL]//Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence. 2024. https://doi.org/10.24963/ijcai.2024/510.

[8] GU A, DAO T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces[C/OL]//First Conference on Language Modeling. 2024. https://openreview.net/forum?id=tEYskw1VY2.

[9] YANG S, WANG B, ZHANG Y, 等. Parallelizing Linear Transformers with the Delta Rule over Sequence Length[C/OL]//Advances in Neural Information Processing Systems: 卷 37. Curran Associates, Inc., 2024: 115491-115522. https://proceedings.neurips.cc/paper_files/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf.

[10] HSIEH C P, SUN S, KRIMAN S, 等. RULER: What’s the Real Context Size of Your Long-Context Language Models?[C/OL]//First Conference on Language Modeling. 2024. https://openreview.net/forum?id=kIoBbc76Sy.

[11] BEHROUZ A, ZHONG P, MIRROKNI V. Titans: Learning to Memorize at Test Time[C/OL]//Advances in Neural Information Processing Systems: 卷 38. Curran Associates, Inc., 2025: 113506-113543. https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf.

[12] LINGLE L D. Transformer-VQ: Linear-Time Transformers via Vector Quantization[C/OL]//International Conference on Learning Representations. 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/18eb80b9faaed5d003b31574bd2a3e9d-Paper-Conference.pdf.

[13] KATHAROPOULOS A, VYAS A, PAPPAS N, 等. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention[C/OL]//Proceedings of the 37th International Conference on Machine Learning. PMLR, 2020: 5156-5165. https://proceedings.mlr.press/v119/katharopoulos20a.html.

[14] DAO T, GU A. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality[J]. Proceedings of Machine Learning Research, 2024, 235: 10041-10071.

[15] ARORA S, EYUBOGLU S, TIMALSINA A, 等. Zoology: Measuring and Improving Recall in Efficient Language Models[C/OL]//International Conference on Learning Representations. 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/448fc91f669c15d10364ee01d512cc10-Paper-Conference.pdf.
