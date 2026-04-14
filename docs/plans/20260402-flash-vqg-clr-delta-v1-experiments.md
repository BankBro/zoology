# Flash-VQG CLR Delta V1 实验记录

> status: validating
> doc_id: flash-vqg-clr-delta-v1-experiments
> version: v1
> owner: lyj
> reviewers: pending
> created: 2026-04-02
> updated: 2026-04-02
> parent: none
> supersedes: none

## 1. 背景现状与目标

### 1.1. 背景

当前主问题不是单独验证 `clr_delta_v1` 能否跑通, 而是判断它是否值得作为 `clr_v1` 的后续主线方向继续推进. 这件事需要把两类信息区分开: 一类是工程代价, 例如显存和吞吐; 另一类是任务效果, 例如 MQAR 上的 `valid/accuracy` 和 hard case 表现. 如果只看单次默认点, 很容易把"坏默认值"误判成"方向本身错误".

### 1.2. 现状

- 主仓实验路径固定在 `/home/lyj/mnt/project/Flash-VQG`, 当前对照分支为 `20260401-clr-delta-v1-compare`.
- 主线比较固定使用 stable non-remat slice: `block32 + local2 + remote1 + global_shuffle + cb128 + remat=off + den1`.
- 已完成三轮方向性实验:
  - 第一轮: `clr_v1` vs `clr_delta_v1` 的 `r={2,4,8}` 一轮 smoke.
  - 第二轮: 仅对 `clr_delta_v1-r4` 做 `beta_init x y_den = 3 x 3` 粗筛.
  - 第三轮: 将 `r4` 粗筛挑出的 `4` 个候选点带去 `r2/r8` 做跨 rank 短复核.
- 当前最重要的新事实是: `clr_delta_v1` 默认点表现很差, `r4` 在调过 `beta_init/y_den` 后可以被显著拉回, 但这一改善目前没有泛化到 `r8`, 且在 `r2` 上也未超过 `clr_v1` baseline.

### 1.3. 目标

- 用同一篇文档持续记录 `clr_delta_v1` 相关实验, 避免结论分散在临时消息和日志里.
- 先明确哪些结论已经成立, 哪些还只是方向性信号.
- 为后续 `r8` 定点诊断和更长实验提供单一更新入口.

## 2. 推荐方案概览

### 2.1. 一句话方案

以"当前有效结论 + 证据表"的方式维护 `clr_delta_v1` 实验文档: 先沉淀 stable slice 下的默认对照, 再沉淀 `r4` 超参粗筛, 后续同主题实验继续追加到相应专题, 不新开零散文档.

### 2.2. 核心流程

1. 先记录默认配置下的 smoke 结果, 明确默认点不能直接作为结构结论.
2. 再记录 `r4` 的 `beta_init/y_den` 粗筛, 提取当前最有希望的候选组合.
3. 再记录 `r2/r8` 跨 rank 复核, 判断 `r4` 的正向信号是否具有迁移性.
4. 后续只把 `r8` 定点诊断和更长实验继续追加到同一文档, 并明确标注其 baseline 身份和结论状态.

### 2.3. 关键接口与数据

- 代码仓:
  - `Flash-VQG`: `/home/lyj/mnt/project/Flash-VQG`
  - `zoology`: `/home/lyj/mnt/project/zoology`
- 关键运行约束:
  - `fox_clr_remat_mode=off`
  - `fox_clr_use_den_residual=True`, 除非专题内显式说明变更
  - `data_seed=123`, `train_batch_order=global_shuffle`
- 当前主指标:
  - `valid/accuracy`
  - `valid/loss`
  - `valid/num_kv_pairs/accuracy-*`
  - `valid/input_seq_len/accuracy-*`
- 已有关键 launch:
  - 默认 smoke: `flash-vqg-clr-delta-v1-smoke-non-remat-2026-04-01-18-47-09`
  - `r4` 粗筛: `flash-vqg-clr-delta-v1-r4-beta-yden-screen-2026-04-01-19-56-19`
  - `r2/r8` 跨 rank 复核: `flash-vqg-clr-delta-v1-r2-r8-crossrank-review-2026-04-02-04-22-45`

### 2.4. 边界条件与不变式

- 本文当前所有结果都属于 `performance baseline1: stable non-remat clr compare`.
- `remat` 相关结论不混入本文主线; 如需记录 `remat` 实验, 应单独标明为补充对照.
- `v1/v2/...` 在专题内表示设计或结论版本, 不表示同一设计下的第几轮实验.
- `result=pass/fail` 表达是否支持当前结论, 不表达命令是否执行成功.

## 3. 专题设计

### 3.1. 专题清单与边界

| topic | goal | current_version | status | note |
|---|---|---|---|---|
| topic-a | 固定 stable non-remat slice, 记录 `clr_v1` 与 `clr_delta_v1` 默认点对照 | v1 | validating | 当前结论是"默认点负向, 但不能直接否定结构本身" |
| topic-b | 记录 `clr_delta_v1-r4` 的 `beta_init/y_den` 联合粗筛 | v1 | validating | 当前结论是"`r4` 可被调回, 但同配置存在方差, 仍需跨 rank 复核" |
| topic-c | 记录 `r4` 候选点在 `r2/r8` 上的跨 rank 短复核 | v1 | validating | 当前结论是"`r4` 正向信号未泛化到 `r8`, `r2` 也尚未超过 baseline" |

### 3.2. 专题 A: 默认点 smoke 对照

**(1)** 当前状态与采纳版本

- status: validating
- current_version: v1
- summary: 默认 `clr_delta_v1` 在 stable non-remat slice 下同时表现出更高工程代价和更差任务效果, 但这个结论只适用于默认点, 不能直接外推成"delta 结构无效".

**(2)** 当前设计

使用与 `clr_v1` 相同的 stable non-remat slice 做最小对照, 只比较 `fox_remote_formula in {clr_v1, clr_delta_v1}` 与 `r in {2,4,8}`. 当前这一专题只记录默认 delta 超参:

- `fox_clr_delta_beta_mode=learned_scalar`
- `fox_clr_delta_beta_init=0.5`
- `fox_clr_delta_y_den=1.0`
- `fox_clr_use_den_residual=True`

这轮实验的作用是提供统一基线, 让后续所有超参调整都有一个可回看的默认对照点.

**(3)** 历史版本演进

| version | date | status | summary | change_reason |
|---|---|---|---|---|
| v1 | 2026-04-02 | validating | 固定 stable non-remat slice, 记录默认 `clr_delta_v1` 与 `clr_v1` 的 smoke 对照 | 需要先判断默认点是方向问题还是超参问题 |

**(4)** 实验记录

| version | date | experiment | metric | baseline | current | result | note |
|---|---|---|---|---|---|---|---|
| v1 | 2026-04-01 18:20:00 | `24-step` OOM smoke | `clr_v1` 峰值显存约 `2712/2946/3418 MB`; `clr_delta_v1` 峰值显存约 `4546/6587/10669 MB`, 其中 `delta-r8` 默认点 OOM | `performance baseline1: stable non-remat clr compare` | 默认 `clr_delta_v1` | fail | 说明默认 delta 工程代价显著更高, 且 `r8` 默认 batch 设定不可用 |
| v1 | 2026-04-01 18:47:09 | `1-epoch` smoke, `r={2,4,8}` | `clr_v1`: `0.729/0.807/0.820`; 默认 `clr_delta_v1`: `0.668/0.400/0.403` | `performance baseline1: stable non-remat clr compare` | 默认 `clr_delta_v1` | fail | 默认点显著退化, 尤其 `r4/r8`; 但该结果后续被 topic-b 证明不能直接解释为结构必坏 |

**(5)** 风险与兼容性

- 默认点结果容易被误读成"delta 路线整体失败".
- `clr_delta_v1` 的显存和吞吐成本显著高于 `clr_v1`, 后续任何收益判断都必须同时考虑工程代价.
- topic-a 只包含默认点, 不足以直接锁定后续主线.

**(6)** 未决问题与下一步

- 需要用调参后的候选点复核 `r2` 和 `r8`, 判断默认点的负向结论有多少来自坏超参.
- 需要把同名义配置的重复结果纳入判断, 防止把运行方差误判为参数主效应.

- 相关附件目录: `docs/plans/artifacts/flash-vqg-clr-delta-v1-experiments/topic-a/v1/`

### 3.3. 专题 B: `r4` 的 `beta_init/y_den` 粗筛

**(1)** 当前状态与采纳版本

- status: validating
- current_version: v1
- summary: `clr_delta_v1-r4` 对 `beta_init/y_den` 非常敏感. 当前观察到的最佳点是 `beta_init=0.75, y_den=1.0`, 其 `valid_acc=0.809`, 已接近 `clr_v1-r4=0.807`. 但同名义默认点 `(0.5, 1.0)` 在重复实验中从 `0.400` 变到 `0.583`, 说明该专题同时暴露了真实的运行方差.

**(2)** 当前设计

先用 `r4` 作为便宜的 canary, 固定 `fox_clr_use_den_residual=True`, 只扫:

- `beta_init in {0.25, 0.5, 0.75}`
- `y_den in {0.5, 1.0, 2.0}`

这轮实验不追求定最终默认值, 只回答 3 个问题:

1. `clr_delta_v1-r4` 是否可以靠少量超参从明显异常状态中被拉回.
2. `beta` 和 `y_den` 是否存在清晰主效应或强交互.
3. 哪 `2-3` 个点值得带去 `r2/r8` 复核.

完整 `9` 点结果如下:

| beta_init | y_den | valid_acc | valid_loss | kv32 | kv64 | kv128 | kv256 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.5 | 0.436 | 4.80 | 0.287 | 0.0868 | 0.0312 | 0.00639 |
| 0.25 | 1.0 | 0.682 | 2.17 | 0.885 | 0.620 | 0.291 | 0.0628 |
| 0.25 | 2.0 | 0.412 | 4.95 | 0.225 | 0.0313 | 0.00737 | 0.00172 |
| 0.5 | 0.5 | 0.671 | 2.35 | 0.917 | 0.606 | 0.215 | 0.0315 |
| 0.5 | 1.0 | 0.583 | 3.03 | 0.820 | 0.382 | 0.0856 | 0.00802 |
| 0.5 | 2.0 | 0.400 | 4.95 | 0.172 | 0.0152 | 0.000984 | 0.000566 |
| 0.75 | 0.5 | 0.421 | 4.72 | 0.290 | 0.0360 | 0.00321 | 0.00114 |
| 0.75 | 1.0 | 0.809 | 1.38 | 0.974 | 0.868 | 0.604 | 0.163 |
| 0.75 | 2.0 | 0.590 | 3.12 | 0.836 | 0.399 | 0.0914 | 0.00795 |

当前最重要的结构性观察:

- `y_den=1.0` 是明显更稳的区域.
- `beta` 不是单调越小越好; 它和 `y_den` 存在强交互.
- `(0.75, 1.0)` 在当前 `1 epoch` 结果里最优, 但其优势量级还不足以在单次点上宣称优于 `clr_v1-r4`.

**(3)** 历史版本演进

| version | date | status | summary | change_reason |
|---|---|---|---|---|
| v1 | 2026-04-02 | validating | 先用 `r4` 做 `beta_init/y_den` 粗筛, 再决定是否带去 `r2/r8` 复核 | 需要尽快判断默认点负向结果是否来自坏超参 |

**(4)** 实验记录

| version | date | experiment | metric | baseline | current | result | note |
|---|---|---|---|---|---|---|---|
| v1 | 2026-04-01 19:56:19 | `r4`, `beta_init x y_den = 3 x 3`, `1 seed`, `1 epoch` | 最优点 `(0.75, 1.0)` 为 `acc=0.809`, `loss=1.38`; 同名义默认点 `(0.5, 1.0)` 在本轮为 `0.583`, 与 topic-a smoke 的 `0.400` 不一致 | `performance baseline1: stable non-remat clr compare`, `supplementary control: r4 hyperparam screen` | `clr_delta_v1-r4` 的 `beta_init/y_den` 粗筛 | pass | 支持"默认点不能代表结构本身"这一结论; 但同时也暴露出同配置结果存在方差 |

**(5)** 风险与兼容性

- 当前最佳点来自 `1 seed + 1 epoch`, 只适合做方向性筛选, 不适合直接当正式结论.
- `(0.5, 1.0)` 的重复结果与 topic-a 不一致, 说明这条线存在非忽略性的运行方差.
- 即使 `r4` 被救回, 也不能自动外推到 `r2/r8`.

**(6)** 未决问题与下一步

- 优先带以下组合去做 `r2/r8` 复核:
  - `(0.75, 1.0)`: 当前最优点
  - `(0.25, 1.0)`: 同为 `y_den=1.0` 的较保守点
  - `(0.5, 0.5)`: 第二梯队中较稳定的备选点
- 保留 `(0.5, 1.0)` 作为方差对照, 用于判断当前改善到底来自超参本身, 还是来自单次运行波动.
- 如果跨 rank 复核仍然正向, 再考虑更长 `12-16 epoch` 验证; 否则不要直接推进 `clr_delta_v1` 成为主线.

- 相关附件目录: `docs/plans/artifacts/flash-vqg-clr-delta-v1-experiments/topic-b/v1/`

### 3.4. 专题 C: `r2/r8` 跨 rank 短复核

**(1)** 当前状态与采纳版本

- status: validating
- current_version: v1
- summary: `r4` 上最优的 `(beta_init=0.75, y_den=1.0)` 在 `r2` 上只能接近 baseline, 在 `r8` 上则完全失效. 继续做 `r8 + den_residual=False` 诊断后, 两个 probe 仍然停在 `acc=0.402-0.403`, 说明 `r8` 问题也不是简单关掉 denominator residual 就能解决.

**(2)** 当前设计

基于 topic-b 的结果, 选取 `4` 个 delta 候选点:

- `(0.75, 1.0)`: `r4` 当前最优点
- `(0.25, 1.0)`: 同一 `y_den` 下的保守点
- `(0.5, 0.5)`: 第二梯队备选点
- `(0.5, 1.0)`: 方差对照点

在 `r2` 和 `r8` 上分别与同批 `clr_v1` baseline 做 `1 epoch, 1 seed` 短复核, 并保持:

- `fox_clr_use_den_residual=True`
- `fox_clr_remat_mode=off`
- `r2`: `tbs64, ga4`
- `r8 clr_v1`: `tbs64, ga4`
- `r8 clr_delta_v1`: `tbs32, ga8`

结果如下:

| formula | rank | beta_init | y_den | valid_acc | valid_loss | note |
|---|---|---|---|---|---|---|
| `clr_v1` | 2 | - | - | 0.749 | 1.65 | baseline |
| `clr_delta_v1` | 2 | 0.75 | 1.0 | 0.720 | 1.93 | 当前 `r2` 最佳 delta 点, 但仍低于 baseline |
| `clr_delta_v1` | 2 | 0.25 | 1.0 | 0.404 | 5.00 | 明显异常 |
| `clr_delta_v1` | 2 | 0.50 | 0.5 | 0.649 | 2.46 | 低于 baseline |
| `clr_delta_v1` | 2 | 0.50 | 1.0 | 0.690 | 2.20 | 接近 baseline, 但未超过 |
| `clr_v1` | 8 | - | - | 0.830 | 1.16 | baseline |
| `clr_delta_v1` | 8 | 0.75 | 1.0 | 0.405 | 4.86 | 明显异常 |
| `clr_delta_v1` | 8 | 0.25 | 1.0 | 0.401 | 4.95 | 明显异常 |
| `clr_delta_v1` | 8 | 0.50 | 0.5 | 0.403 | 4.85 | 明显异常 |
| `clr_delta_v1` | 8 | 0.50 | 1.0 | 0.401 | 4.94 | 明显异常 |

当前最重要的结构性观察:

- `r4` 上的调参成功目前更像 rank-specific 信号, 不是可直接迁移的通用解.
- `r2` 上最好的 delta 点只能接近 `clr_v1`, 没有形成明确优势.
- `r8` 上四个点都系统性失败, 且 aggregate 和 hard case 都崩, 说明主问题可能更靠近 denominator residual 动力学, 而不是单独的 `beta/y_den` 数值选择.

作为 hard case 对照, `clr_v1-r8` 和 `clr_delta_v1-r8-(0.75, 1.0)` 的容量轴差异非常明显:

| config | kv32 | kv64 | kv128 | kv256 |
|---|---|---|---|---|
| `clr_v1-r8` | 0.964 | 0.875 | 0.654 | 0.286 |
| `clr_delta_v1-r8-(0.75, 1.0)` | 0.205 | 0.0158 | 0.000977 | 0.000508 |

在此基础上又补了一个更窄的 `r8 + den_residual=False` 诊断, 只测:

- `(beta_init=0.75, y_den=1.0, den_residual=False)`
- `(beta_init=0.5, y_den=1.0, den_residual=False)`

结果如下:

| formula | rank | beta_init | y_den | den_residual | valid_acc | valid_loss |
|---|---|---|---|---|---|---|
| `clr_delta_v1` | 8 | 0.75 | 1.0 | False | 0.403 | 4.88 |
| `clr_delta_v1` | 8 | 0.50 | 1.0 | False | 0.402 | 5.00 |

这个诊断的意义在于:

- 训练过程本身比 `den_residual=True` 时更稳, 后半程 train loss 明显下降.
- 但最终 `valid_acc/valid_loss` 几乎没有改善, 仍然和先前 `0.401-0.405` 的失败区间重合.
- 因此, 当前不能再把 `r8` 的失败主要归因于 denominator residual 这一条更新.

**(3)** 历史版本演进

| version | date | status | summary | change_reason |
|---|---|---|---|---|
| v1 | 2026-04-02 | validating | 将 `r4` 粗筛选出的 `4` 个点带去 `r2/r8` 做跨 rank 短复核 | 需要判断 `r4` 的正向信号是否具有迁移性 |

**(4)** 实验记录

| version | date | experiment | metric | baseline | current | result | note |
|---|---|---|---|---|---|---|---|
| v1 | 2026-04-02 04:22:45 | `r2/r8` cross-rank review, `10 runs`, `1 seed`, `1 epoch` | `r2`: 最优 delta 点 `(0.75, 1.0)` 为 `0.720`, 低于 `clr_v1-r2=0.749`; `r8`: 四个 delta 点全部在 `0.401-0.405`, 远低于 `clr_v1-r8=0.830` | `performance baseline1: stable non-remat clr compare`, `supplementary control: topic-b selected candidates` | `clr_delta_v1` 的跨 rank 迁移性复核 | fail | 否定了"`r4` 最优点可直接迁移到 `r2/r8`"这一假设; 下一步不应直接进长实验 |
| v1 | 2026-04-02 07:48:42 | `r8 den_residual=False` probe, `2 runs`, `1 seed`, `1 epoch` | `(0.75, 1.0, denoff) -> 0.403/4.88`; `(0.5, 1.0, denoff) -> 0.402/5.00` | `performance baseline1: stable non-remat clr compare`, `supplementary control: r8 failure diagnosis` | `clr_delta_v1-r8` 的 `den_residual=False` 定点诊断 | fail | 否定了"关掉 denominator residual 可直接救回 `r8`"这一假设; 训练更稳, 但验证结果仍然失败 |

**(5)** 风险与兼容性

- 当前 cross-rank 结论仍是 `1 seed + 1 epoch`, 但负向信号已经足够强, 尤其 `r8` 四点同时失败.
- `r8` 上的失败具有系统性, 因此继续扩大 `beta/y_den` 网格的优先级不高.
- 即使补了 `den_residual=False`, `r8` 仍然没有被救回; 因此 `fox_clr_use_den_residual` 已经不是当前最高优先级嫌疑点.

**(6)** 未决问题与下一步

- 当前最稳妥的动作是暂停 `clr_delta_v1` 的主线推进:
  - 保持 `clr_v1` 为当前主线
  - `clr_delta_v1` 降级为补充探索
- 如果后续还要继续追 `clr_delta_v1`, 下一轮应该换问题, 不再继续做同类短网格:
  - 优先检查 `r8` 上训练目标与验证目标脱钩的实现原因
  - 或回到算法层重新审视 `clr_delta_v1` 在高 rank 下的读写语义
- 在没有新的结构性假设之前, 不建议继续给 `clr_delta_v1` 投多 seed 或长 epoch 预算.

- 相关附件目录: `docs/plans/artifacts/flash-vqg-clr-delta-v1-experiments/topic-c/v1/`

## 4. 集成验证与落地

### 4.1. 集成验证与回归

当前主结论以 topic-c 为准, 但必须带着 topic-a 和 topic-b 的边界一起解释:

- 已成立:
  - 默认 `clr_delta_v1` 点不能代表结构本身.
  - `r4` 在合适 `beta_init/y_den` 下可以被明显拉回.
  - `r4` 的正向信号目前没有泛化到 `r8`, `r2` 上也未形成明确优势.
- 尚未成立:
  - `clr_delta_v1` 是否能在 `r8` 上通过更窄的结构诊断被救回.
  - `clr_delta_v1` 是否能在更长训练里稳定优于或追平 `clr_v1`.

后续更新规则:

- 同主题新实验继续追加到本文件.
- `r8` 定点诊断优先追加到 topic-c; 若后续形成独立长期主线, 再在 `专题设计` 新增 topic-d.
- 新实验若覆盖旧实验, 正文要明确哪一轮是当前主结论依据.

### 4.2. 发布与回退

当前阶段不涉及发布. 对实验主线的"推进 / 回退"规则如下:

- 若 `r8` 定点诊断失败, 保持 `clr_v1` 为主线, `clr_delta_v1` 降为补充探索.
- 若 `r8` 定点诊断成功但方差较大, 先补短重复实验再决定是否进入长跑.
- 若后续更长实验也支持当前候选点, 再进入正式多 seed 对照.

## 5. 参考

### 5.1. 参考资料

- `docs/reference/20260401-clr-delta-v1-algorithm-full.md`
- `Flash-VQG` 分支: `20260401-clr-delta-v1-compare`

### 5.2. 相关文档与实验附件

- 文档: 本文档用于统一记录 `clr_delta_v1` 同主题实验.
- 附件目录: `docs/plans/artifacts/flash-vqg-clr-delta-v1-experiments/<topic>/<version>/`
- 默认 smoke 结果目录: `/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/flash-vqg-clr-delta-v1-smoke-non-remat-2026-04-01-18-47-09`
- `r4` 粗筛结果目录: `/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/flash-vqg-clr-delta-v1-r4-beta-yden-screen-2026-04-01-19-56-19`
- `r2/r8` 跨 rank 复核结果目录: `/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/flash-vqg-clr-delta-v1-r2-r8-crossrank-review-2026-04-02-04-22-45`
