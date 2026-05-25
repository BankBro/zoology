# Flash-VQG Baseline 组会 PPT 文案

- 生成时间: `2026-04-20 08:13:19 UTC`
- 适用形式: HTML 幻灯片正文文案
- 主题: `Flash-VQG 当前 baseline 的结构, 证据与下一步方向`
- 备注: 本稿按 `4 个章节, 7 页正文` 组织. 实验数据优先使用本地已验证的 MQAR 结果, 外部 Gated DeltaNet 数据只作为公开 retrieval 参照, 不伪装成同口径 MQAR 榜单.

## 数据与材料来源

- `docs/reference/20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md`
- `docs/reference/20260413-e5a-top2-audit.md`
- `docs/reference/20260416-flash-vqg-vnext-architecture-directions.md`
- `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e3-dense-routing/config_builder.py`

## PPT 总体结构

### 第 1 章: 任务背景与汇报目标

- 第 1 页: 当前已经定下来的 baseline 是什么
- 第 2 页: MQAR 任务是什么

### 第 2 章: baseline 的结构与原理

- 第 3 页: 当前 baseline 的整体架构
- 第 4 页: 当前 baseline 的数学原理

### 第 3 章: MQAR 视角下的比较与结论

- 第 5 页: MQAR 视角下: Flash-VQG baseline 与 Gated DeltaNet 的对比分析
- 第 6 页: 当前 baseline 的一句话总结

### 第 4 章: 下一步实验

- 第 7 页: Gated-Delta Remote Memory

## 第 1 章: 任务背景与汇报目标

### 第 1 页: 当前已经定下来的 baseline 是什么

#### 标题

**Flash-VQG 当前 baseline: 先讲清楚“现在在用什么”**

#### 正文

- 当前已经收口的 baseline, 可以用一句中文概括为:
  **稠密软写入 + 远端只读两个候选槽 + 选槽时兼顾匹配分和有效内容量 + 本地/远端共享归一化**
- 如果需要在页脚保留配置名, 可以小字写成:
  **dense-t025 + top2 read + den_aware + shared_den**
- 现有文档已经把它当作当前稳定工作点, 而不是还在摇摆的候选项
- 这次汇报分四章:
  1. 任务是什么
  2. baseline 怎么工作
  3. MQAR 视角下如何看 baseline 与 Gated DeltaNet
  4. 下一步实验准备改什么

#### 这一页想让听众记住的话

**这次不是讲新方案, 而是讲: Flash-VQG 现在已经定下来的 baseline, 到底是什么.**

#### 建议配图

- 一张总览图: 左边“局部窗口”, 右边“远端记忆槽”, 底部“统一融合”

#### 建议展示数据

- 这一页只放 baseline anchor 的 4 个 KPI 小卡片, 不放大表

| 指标 | 数值 |
| --- | ---: |
| `valid/accuracy` | `0.981208` |
| `valid/loss` | `0.081622` |
| `acc@512` | `0.991031` |
| `acc@1024` | `0.874887` |

#### 页面元素建议

- 左侧: baseline 一句话定义
- 右下: 4 个 KPI 卡片
- 页脚: `当前默认工作点: dense-t025 + top2 read + den_aware + shared_den`

---

### 第 2 页: MQAR 任务是什么

#### 标题

**MQAR: 为什么它适合拿来分析长程记忆与检索**

#### 正文

- MQAR 可以简单理解为:
  **前面先给若干组“键-值”配对, 后面在不同位置给出多个“键”的查询, 模型要把对应“值”找回来**
- Zoology 将这个问题 formalize 成**多查询联想召回**. 相比更早的单查询版本, 它更接近真实语言建模, 因为一次前向里经常要做**多次召回**, 而且查询位置和交互距离都在变化
- 这个任务真正考察的, 不只是一般短程建模能力, 而是:
  **能不能把较早写进去的内容, 在需要的时候准确取出来**
- 因为 MQAR 同时考察长距离, 内容检索和多次召回, 所以非常适合分析 Flash-VQG 这类**带远端 memory 的结构**
- 用一句最朴素的话讲:
  **它测的是模型会不会“记住前面存过的东西, 并在后面正确取回”.**

#### 建议配图

- 一条序列示意图: 前半段若干“键-值”对, 后半段多个 query 位置

#### MQAR 作图建议

- 建议直接用一条从左到右的 toy sequence 来画, 上方写“已写入的键值配对”, 右侧写“后续多个查询”
- 可以直接使用下面这组例子:

```text
已写入的键值配对                         后续多个查询
序列:   [A][4]   [B][3]   [C][6]   [F][1]   [E][2]    [A][?]  [C][?]  [F][?]  [E][?]  [B][?]

输出:                                                    [4]     [6]     [1]     [2]     [3]
```

- 作图时建议把左半段 5 组键值对画成已经写入 memory 的历史内容, 右半段 5 个 query 位置画成待召回的位置
- 每个 query 下方用一根竖向箭头指向正确答案, 让听众一眼看出“这是一个从前文取回 value 的任务”
- 建议用两种颜色区分:
  - 键 `A/B/C/F/E` 用一种颜色
  - 值 `4/3/6/1/2` 用另一种颜色
- `?` 要明显画成查询位, 让人立即理解“这里不是继续写入, 而是在问前面对应的 value 是什么”
- 如果版面允许, 可以在 toy sequence 下方补一句解释:
  **模型需要记住前面写入的键值配对, 并在后面的多个查询位置正确取回对应的值**

#### 页面元素建议

- 顶部: 问题定义
- 中部: 一条玩具例子序列
- 底部: 为什么这个任务适合检验长程 memory

## 第 2 章: baseline 的结构与原理

### 第 3 页: 当前 baseline 的整体架构

#### 标题

**Flash-VQG 当前 baseline: 局部精确注意力 + 远程压缩记忆**

#### 核心思想

- 当前 baseline 将上下文建模分成两条路径:
  - **局部路径**: 直接对最近窗口做精确注意力, 负责短程, 细粒度的信息交互
  - **远程路径**: 将更久远的历史压缩到一组记忆槽中, 再根据当前查询按需读出
- 可以把它概括为:
  **近处的信息精确看, 远处的信息压缩存**

#### 整体流程

- 对于每个位置 `t`, 先得到查询, 键, 值表示:
  `q_t, k_t, v_t`
- 然后将键向量量化到码本中心:
  `k_hat_t = VQ(k_t)`
- 并保留量化残差:
  `r_t = k_t - k_hat_t`
- 这里:
  - `k_hat_t` 决定该 token 属于哪个记忆槽
  - `r_t` 描述该 token 相对槽中心的细节偏差

#### 模型的两条路径

- 局部路径:
  **近处的信息不压缩, 直接精确计算**
- 远程路径:
  **历史内容被聚合到码本槽中, 每个槽同时保存平均原型和细节修正**

#### 最终输出

- 最后把:
  - 局部路径的精确结果
  - 远程路径的压缩读出结果
- 做统一融合, 得到最终输出 `o_t`

#### 这一页的总结句

**当前 baseline 的本质是: 局部路径负责精确建模, 远程路径负责压缩存储; 远程记忆既保存“平均原型”, 也保存“细节修正”.**

#### 建议配图

- 中间画一条 token 流
- 左支: “局部窗口精确注意力”
- 右支: “码本槽级记忆”
- 最后在底部合流

---

### 第 4 页: 当前 baseline 的数学原理

#### 标题

**Flash-VQG 当前 baseline: 远程记忆的写入, 读出与融合**

#### 1. 残差表征

- 键向量被量化到码本中心:
  `k_hat_t = VQ(k_t)`
- 量化误差定义为:
  `r_t = k_t - k_hat_t`
- 再将残差投影到每个记忆槽自己的局部坐标系中:
  `h_(t,s) = B_s^T r_t`

#### 2. 远程记忆写入

- 对每个记忆槽 `s`, 维护四类状态:
  - `g_s`: 值的粗粒度累积
  - `L_s`: 槽权重或计数累积
  - `R_s`: 值的残差修正器
  - `a_s`: 分母的残差修正器
- 写入时按权重 `w_(t,s)` 累计:
  - `g_s <- gamma g_s + sum_t w_(t,s) v_t`
  - `L_s <- gamma L_s + sum_t w_(t,s)`
  - `R_s <- gamma R_s + sum_t w_(t,s) v_t h_(t,s)^T`
  - `a_s <- gamma a_s + sum_t w_(t,s) h_(t,s)`

#### 3. 远程记忆读出

- 给定查询 `q_t`, 在槽 `s` 内得到查询相关坐标 `alpha_s(q_t)`
- 用槽状态恢复内容:
  - `Num_s(q_t) = g_s + R_s alpha_s(q_t)`
  - `Den_s(q_t) = L_s + a_s^T alpha_s(q_t)`

#### 4. 槽选择

- 当前 baseline 选择槽时, 不只看相似度, 还考虑该槽中“有效可读内容”的多少:
  `selector_s(q_t) = score_s(q_t) + log Den_s(q_t)`
- 直观含义:
  **既看当前查询与该槽是否匹配, 也看该槽里是否真的有足够内容可以读出**

#### 5. 局部与远程的统一融合

- 局部路径与远程路径最终共用一个归一化分母:
  `o_t = (Num_local(t) + Num_remote(t)) / (Den_local(t) + Den_remote(t))`
- 这就是 `shared_den` 的直观含义

#### 这一页的总结句

**当前 baseline 的数学结构可以概括为: 先把长程历史压到若干记忆槽, 每个槽同时存“平均原型”和“残差修正”, 再按查询相关方式读出, 最后与局部精确注意力统一融合.**

#### 建议配图

- 这一页以公式分块为主
- 每一块公式右边只配一句中文解释

## 第 3 章: MQAR 视角下的比较与结论

### 第 5 页: MQAR 视角下: Flash-VQG baseline 与 Gated DeltaNet 的对比分析

#### 标题

**MQAR 视角下: Flash-VQG baseline 与 Gated DeltaNet 的对比分析**

#### 正文

- **先说明口径**: 目前公开可见材料里, 还没有“Flash-VQG baseline 与 Gated DeltaNet 在同一套 MQAR 配置, 同一训练预算, 同一评测脚本下”的直接数值表
- 所以这一页更适合写成:
  **“MQAR 视角下的对比分析 + 外部公开 retrieval 结果参照”**
- Flash-VQG baseline 的路线是:
  **压缩存储 + 内容检索**
- Gated DeltaNet 的路线是:
  **可遗忘 + 可重写的状态更新**
- 这一页最终想传达的不是:
  **“Gated DeltaNet 已经在同口径 MQAR 上打败了 Flash-VQG”**
- 而是:
  **从 MQAR 视角看, 它瞄准的正是 stale memory, memory collision, 状态难以重写这些当前最值得怀疑的问题**

#### 这一页的内部实验证据

##### A. `dense-t025` 为什么优于 `dense-t050`

| axis | `dense-t025` acc | `dense-t050` acc | delta | `dense-t025` acc@1024 | `dense-t050` acc@1024 | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | `0.981208` | `0.966792` | `+1.44pp` | `0.874887` | `0.776309` | `+9.86pp` |
| seed axis | `0.980097` | `0.966789` | `+1.33pp` | `0.860320` | `0.777801` | `+8.25pp` |
| seed axis | `0.975150` | `0.962067` | `+1.31pp` | `0.858090` | `0.764488` | `+9.36pp` |
| data axis | `0.977402` | `0.970286` | `+0.71pp` | `0.874203` | `0.817445` | `+5.68pp` |
| data axis | `0.975364` | `0.969170` | `+0.62pp` | `0.856121` | `0.811410` | `+4.47pp` |

- 可讲成一句话:
  - `dense-t025` 在 `5/5` 组 matched validation 上都优于 `dense-t050`
  - 平均 `valid/accuracy` 提升约 `+1.08pp`
  - 平均 `acc@1024` 提升约 `+7.52pp`

##### B. 为什么保留 `top2 read`, 不退回 dense read

| config | `valid/accuracy` | delta vs baseline | `valid/loss` | `acc@1024` | delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense-t025 + top2 read` | `0.981208` | `reference` | `0.081622` | `0.874887` | `reference` |
| `dense-t025 + dense read` | `0.965413` | `-1.58pp` | `0.158107` | `0.765508` | `-10.94pp` |

##### C. 为什么说当前主瓶颈不像 selector

| run | mode | `acc_total` | `acc_1024x256` | 结论 |
| --- | --- | ---: | ---: | --- |
| `e5a-run-001` | `student` | `0.9302` | `0.8728` | baseline 参照 |
| `e5a-run-001` | `ref` | `0.7869` | `0.6932` | 比 student 低 `14.33pp` |
| `e5a-run-002` | `student` | `0.9336` | `0.8760` | baseline 参照 |
| `e5a-run-002` | `ref` | `0.7552` | `0.6311` | 比 student 低 `17.85pp` |

#### 对外部 Gated DeltaNet 数据的使用说明

- 建议右半边单独放论文原表或你整理后的引用小表
- 页脚必须标注:
  **“注意: 此页不是同一套 MQAR 脚本下的直接 apples-to-apples 数值对比”**

#### 建议版式

- 左半边: 我们自己的 MQAR 证据
- 右半边: Gated DeltaNet 的机制要点 + 外部 retrieval 参照
- 页脚: 口径说明

#### 推荐展示方式

- `dense-t025 vs dense-t050`: 适合用 `slope chart`
- `top2 read vs dense read`: 适合用 `2 行小表`
- `student vs ref`: 适合用 `裁决表`

---

### 第 6 页: 当前 baseline 的一句话总结

#### 标题

**一句话记住当前 baseline, 以及它为什么被留下来**

#### 正文

- 当前 baseline 不是简单地“更稀疏”或者“更稠密”
- 更准确的说法是:
  **近处信息直接精确看, 远处信息压缩存; 写入更平滑, 读出更集中, 最后统一归一化**
- 对应到结构上就是:
  **码本量化键 + 局部精确窗口 + 残差修正状态 + 远端只读两个候选槽**
- 如果整场汇报只留一句话给听众, 可以写成:
  **Flash-VQG 当前 baseline = 局部精确注意力 + 远程压缩记忆**

#### 这一页建议放的裁决表

| 维度 | 当前结论 | 处理 |
| --- | --- | --- |
| 写入温度 | `t025 > t050` | 保留 `t025` |
| 读出方式 | `top2 > dense` | 保留 `top2` |
| codebook 容量 | `128` 最稳 | 保留 `cb128` |

#### 支撑 `cb128` 的数据

| codebook size | seed123 `valid/accuracy` | seed124 `valid/accuracy` | seed123 `acc@1024` | seed124 `acc@1024` |
| --- | ---: | ---: | ---: | ---: |
| `64` | `0.923458` | `0.956714` | `0.579992` | `0.749629` |
| `128` | `0.981208` | `0.980097` | `0.874887` | `0.860320` |
| `256` | `0.981071` | `0.939210` | `0.871535` | `0.627754` |

#### 这一页的结论句

**当前 baseline 被留下来, 不是因为它看起来顺眼, 而是因为在温度, 读出方式和容量三个关键维度上, 它都是当前最稳的工作点.**

#### 建议配图

- 左侧: 一句话结构图
- 右侧: 3 行裁决表
- 底部: `cb64 / cb128 / cb256` 双 seed 柱状图
  - 柱高统一表示 `acc@1024`
  - 棕色表示 `seed123`, 蓝色表示 `seed124`
  - 图上直接标数值 `0.580 / 0.750 / 0.875 / 0.860 / 0.872 / 0.628`
  - 图例和纵轴刻度要明确, 用这张图直接讲清:
    `64` 明显不够, `256` 不稳定, `128` 最稳

## 第 4 章: 下一步实验

### 第 7 页: Gated-Delta Remote Memory

#### 标题

**Flash-VQG: Gated-Delta Remote Memory**

#### 核心思想

- 从“加法累计 memory”升级为“可遗忘, 可重写的 memory”
- 当前 remote residual memory 更像“加法累计器”
- 下一步的目标不是再多加一点信息, 而是:
  **把已经写旧, 写脏, 写错的 memory 改干净**

#### 左侧: 现有问题

- 当前 residual 来源合理:
  - `r_t = k_t - k_hat_t`
  - `h_(t,s) = B_s^T r_t`
- 但 residual 后续主要按加法方式累计进 memory
- 旧记忆主要只能靠统一衰减变弱, 很难定向清除
- 多个 token 反复写入同一个 code 时, 容易出现:
  **旧信息残留, 内容冲突, 记忆变脏**

#### 中间: 关键改动

- 只更新真正命中的 code
- 先减去 coarse baseline, 只学习“修正量”
- 用 gate 控制遗忘强度
- 一句话理解:
  - **Gate 决定忘多少**
  - **Delta 决定怎么改**

#### 右侧: 新方案

- 不再只是把新信息加进去, 而是对命中的 code 做**误差驱动的重写**
- 写入前先通过 gate 做遗忘, 降低旧内容残留
- 让 remote memory 从“累加器”变成“可编辑 memory”

#### 这一页必须附的实验矩阵

| 组别 | residual update | forget | 目的 |
| --- | --- | --- | --- |
| `additive` | additive | global | 当前 writer 对照组 |
| `gated-only` | additive | code-aware | 单独看 forget 是否有益 |
| `delta-only` | delta | global | 单独看 delta update 是否有益 |
| `gated-delta` | delta | code-aware | 看两者是否可叠加 |

#### 主看指标

- `valid/accuracy`
- `acc@1024`
- `valid/loss`
- `valid/attn/den_min`
- `valid/attn/nan_inf_count`
- `valid/vq/relative_err_mean`

#### 底部总结语

**原来: 先算 residual, 再不断累加**
**现在: 先遗忘旧内容, 再写入残差修正**

## 实验数据使用清单

| 页面 | 要证明什么 | 建议数据 | 建议展示方式 |
| --- | --- | --- | --- |
| 第 1 页 | baseline 已经是稳定 anchor | `dense-t025-s123-d123` 的 `valid/accuracy`, `valid/loss`, `acc@512`, `acc@1024` | `4 个 KPI 卡片` |
| 第 5 页 | baseline 为什么合理, 为什么下一步更该怀疑 memory mechanics | `dense-t025 vs dense-t050`, `top2 read vs dense read`, `E5A student vs ref`, 外部 Gated DeltaNet retrieval 结果 | `左边内部证据小表 + 右边外部参照小表` |
| 第 6 页 | baseline 的一句话总结背后有硬裁决 | `dense write vs sparse write`, `cb64/cb128/cb256` | `3 行决策表 + 1 张点图` |
| 第 7 页 | 下一步实验怎么设计 | `additive / gated-only / delta-only / gated-delta` 4 组矩阵 | `2x2 矩阵表` |

## 不建议放进正文的大表

- 训练 run 跟踪表
- 所有 telemetry 全字段表
- launch_id 和 queue / session 细节
- 过长的 audit mode 明细表

这些内容如果需要, 更适合放到 HTML 幻灯片的 appendix 或 speaker notes.

## 组会口头版 3 分钟摘要

- 目前 Flash-VQG 已经稳定收口到一个 baseline: `dense-t025 + top2 read + den_aware + shared_den`
- 这套 baseline 的核心特点是: 近处信息直接精确看, 远处信息压缩到码本槽里, 再按当前 query 从少量候选槽中读出, 最后和局部路径统一融合
- 它被留下来不是拍脑袋决定的, 而是因为三类证据同时成立:
  - `dense-t025` 在 matched validation 上稳定优于 `dense-t050`
  - `top2 read` 明显优于 dense read
  - `cb128` 比 `cb64` 更强, 也比 `cb256` 更稳
- 同时 E5A audit 给了一个很关键的负结论:
  **即使把 top2 code set 换成更接近 oracle 的版本, 最终准确率也没有更好**
- 所以当前更合理的判断是:
  **瓶颈不像只在 selector, 更像在 remote memory 的写法, 存法和遗忘法**
- 因而下一步最自然的方向, 就是把当前远端 residual memory 从“加法累计器”升级成“可遗忘, 可重写的 Gated-Delta Remote Memory”
