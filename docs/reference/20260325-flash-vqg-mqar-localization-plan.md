# Flash-VQG 在 Zoology MQAR 上的定位实验整理

## 背景

当前 `d_model=128` 的 Zoology MQAR 对比实验已经给出一个相对稳定的现象.

- Gated DeltaNet 在当前配置上可以基本学会任务.
- Flash-VQG 在 `block_len=8` 时表现很差, 切换 Triton / Torch 后端都没有本质改善.
- 把 `block_len` 提到 `32` 后, Flash-VQG 会明显提升, 但仍然显著低于 Gated DeltaNet.

结合 Zoology 对 MQAR 的解释, 这个任务的难点不只是长距离, 还在于同一条样本内要完成多次, 且距离不同的 recall. 对 gated-conv / recurrence 类结构, 典型差距往往来自 insufficient data-dependent mixing. 只在少量 recall-sensitive 位置加入 selective look-up / sparse attention, 往往就能缩小 gap.

因此, 当前更适合走"先洗环境, 再做定位"的顺序, 而不是直接大改模型.

## 当前代码语境

在 Zoology 当前数据路径里, `prepare_data()` 会构造 `_SyntheticDataset`, 而 `_SyntheticDataset.batches` 按 segment 顺序展开批次, 同时 `DataLoader(..., shuffle=False)` 不再打乱这些批次. 这意味着每个 epoch 默认会按固定 segment 顺序喂数据, 容易把 sampler / curriculum artifact 和模型结构问题混在一起.

此外, 当前这版 Flash-VQG 虽然已经有 `if_remote_enabled` 开关, 可以直接关闭 remote 读出, 但当前 FoX materialize 训练路径并不支持严格的 `strict remote-only` 作为主训练消融. 原因是本地窗口校验要求 `local_num_blocks >= 1`, 所以这轮最自然, 也最有信息量的远程定位路径, 不是 `local-only / remote-only / full` 三分, 而是仓库远程分支消融文档给出的最小四配置矩阵:

- `full_K`
- `remote_hist_1`
- `local_only_K`
- `local_only_1`

相关代码位置:

- `zoology/data/utils.py`
- `_SyntheticDataset.__init__()`
- `prepare_data()`
- `Flash-VQG/src/flash_vqg/nn/attn.py`
- `Flash-VQG/src/flash_vqg/nn/attn_fox.py`

因此, 这一轮不再纠结 strict remote-only. 第一优先级是先改 sampler, 然后围绕上述两对主对照来判断 remote 到底有没有净贡献.

## 定位实验总原则

- 先做最小改动, 优先排除数据顺序和观测方式带来的伪现象.
- 每一轮实验尽量只改一类因素, 不同时引入多个结构变化.
- 所有实验尽量复用同一组核心指标, 保证横向可比.
- 先确认"谁在做题", 再确认"为什么做不好", 最后再决定是否值得改结构.

## 实验阶段与顺序

这轮实验主线收束成 4 个阶段. 为了和既有讨论保持一致, 下面仍沿用 `E0-E8` 的编号. 执行上可以把 `E7 / E8` 视作最后一组收尾 probe, 因此整体仍然是一条 8 步主线.

### 阶段 A. 先把训练环境洗干净

#### E0. Sampler 重跑 baseline

**目的**

先排除"按长度顺序喂数据"带来的锯齿和 curriculum artifact.

**改动**

- 只改 dataloader / sampler.
- 方案 A: 全局 shuffle.
- 方案 B: balanced interleave, 让 `(64,4)`, `(128,8)`, `(256,16)`, `(256,32)`, `(256,64)` 在每个 epoch 内尽量均匀混合.
- 模型固定为当前 best baseline, 即 `block_len=32`, 其余超参先不动.

**记录指标**

- `train/loss`
- 分 segment 的 `train/loss`
- `valid/accuracy`
- `valid/input_seq_len/*`
- `valid/num_kv_pairs/*`
- step-to-`0.2 / 0.3 / 0.4`

**预期现象**

- 如果锯齿明显减弱, 但 `128+` 仍然差, 说明 sampler 只是表象, 主问题仍然是结构失配.
- 如果 `128+` 也明显改善, 说明原训练流程确实被顺序喂数污染了.

**下一步动作**

- 无论结果如何, 后续实验都固定用新的 sampler, 然后进入阶段 B.

### 阶段 B. 先确认 remote 到底有没有真贡献

仓库里的远程分支消融文档已经把最小实验矩阵明确成 4 个配置: `full_K`, `remote_hist_1`, `local_only_K`, `local_only_1`. 这一阶段的目标, 就是先回答 remote 有没有在做题.

#### E1. `full_K` vs `local_only_K`

**目的**

在相同 local horizon 下, 看 remote 分支有没有净贡献.

**改动**

- `full_K`: `local_num_blocks=2`, `if_remote_enabled=true`
- `local_only_K`: `local_num_blocks=2`, `if_remote_enabled=false`

**记录指标**

- `valid/accuracy`
- `valid/input_seq_len/*`
- `valid/mqar_case/*`
- `valid/num_kv_pairs/*`
- `attn/remote_win_rate`
- `attn/den_cache_ratio`
- `attn/o_remote_energy_ratio`
- `attn/o_remote_local_cos`
- `attn/remote_dominance_rate`

**预期现象**

- 如果 `full_K ≈ local_only_K`, 说明 remote 基本没帮上忙.
- 如果 `full_K > local_only_K`, 但提升只出现在少量短 case, 说明 remote 有贡献但不够强.
- 如果提升主要出现在 `128+` 或大 `num_kv_pairs` case, 说明 remote 是有效的, 只是还不够.

**下一步动作**

- 若差距很小, 后面优先修 remote read / routing.
- 若差距明显, 再继续看它到底受限于 local horizon, 还是 VQ 寻址.

#### E2. `remote_hist_1` vs `local_only_1`

**目的**

当跨块能力几乎主要依赖 remote 时, 看 remote 有没有真正把跨块 recall 接住.

**改动**

- `remote_hist_1`: `local_num_blocks=1`, `if_remote_enabled=true`
- `local_only_1`: `local_num_blocks=1`, `if_remote_enabled=false`

**记录指标**

- `valid/accuracy`
- `valid/input_seq_len/*`
- `valid/mqar_case/*`
- `valid/num_kv_pairs/*`
- `attn/remote_win_rate`
- `attn/den_cache_ratio`
- `attn/o_remote_energy_ratio`
- `attn/o_remote_local_cos`
- `attn/remote_dominance_rate`

**预期现象**

- 如果 `remote_hist_1` 只比 `local_only_1` 稍高一点, 说明 remote 有一点作用, 但远不足以承担 MQAR 的跨块召回.
- 如果差距很大, 说明 remote 是有效的, 只是当前 `K=2` 可能被别的因素限制住了.
- 如果两者几乎没差, 说明当前瓶颈更偏 local path 主导, remote 基本形同虚设.

**下一步动作**

- 若这组差距也很小, 后面就不用再纠结"是不是 local 不够", 而应该直接修 remote 机制.
- 若差距明显, 再进入阶段 C 分析 local horizon / block effect.

### 阶段 C. 分清楚是 local horizon 问题, 还是 block 机制本身的问题

#### E3. Fixed-local-coverage 对照

**目的**

区分 `block_len=32` 的提升, 到底来自更大的 local coverage, 还是来自 block 切分本身.

**改动**

尽量让有效 local coverage 接近, 例如:

- `(block_len=8, local_num_blocks=8)`
- `(16,4)`
- `(32,2)`
- `(64,1)`

如果代码语义不是简单乘法, 就找一组有效覆盖范围尽量接近的组合.

**记录指标**

- `valid/input_seq_len/*`
- `valid/mqar_case/*`
- `valid/num_kv_pairs/*`
- 各长度下的收敛速度

**预期现象**

- 如果 coverage 接近时效果也接近, 说明主要问题是 local horizon.
- 如果 `block_len` 越大越好, 即使 coverage 差不多, 说明 block 切分或 block 内 exact mixing 本身就重要.

**下一步动作**

- 如果主要是 horizon, 不要把继续增大 `block_len` 当成正式解法, 只把它当成诊断边界.
- 如果主要是 block effect, 后面更应该修 remote, 而不是继续堆本地窗口.

#### E4. 单轴 MQAR 诊断

**目的**

拆开"长度变长"和"`num_kv_pairs` 变多"这两种难度.

**改动**

- 固定 `num_kv_pairs`, 扫 `seq_len`
- 固定 `seq_len`, 扫 `num_kv_pairs`

**记录指标**

- `valid/input_seq_len/*`
- `valid/num_kv_pairs/*`
- `accuracy vs seq_len`
- `accuracy vs num_kv_pairs`

**预期现象**

- 如果随长度掉得更快, 主瓶颈偏 distance coverage.
- 如果随 `num_kv_pairs` 掉得更快, 主瓶颈偏 memory capacity / collision.
- 如果两边都掉, 说明两类问题同时存在.

**下一步动作**

- 长度轴更差, 优先 exact rescue / selective lookup.
- 容量轴更差, 优先 routing / residual / state 表达力.

### 阶段 D. 开始打 VQ / remote 的内部诊断, 并用小改动做机理探针

#### E5. Codebook size sweep

**目的**

判断问题是不是"寻址太粗, collision 太重".

**改动**

- 只扫 `num_codebook_vectors = 64 / 128 / 256 / 512`
- 其他配置保持不变

**记录指标**

- `valid/accuracy`
- `valid/input_seq_len/*`
- `valid/num_kv_pairs/*`
- code usage entropy
- dead code ratio
- mean / max code load
- routing entropy
- top1-top2 margin
- quantization residual norm

**预期现象**

- 如果 codebook 变大后, `128+` 与大 `num_kv_pairs` case 明显上涨, 说明主瓶颈偏寻址粗糙.
- 如果几乎不涨, 说明不是单纯 codebook 太小.

**下一步动作**

- 若明显有效, 后面优先做 routing 改进.
- 若不明显, 转去看桶内表达力或精确救援.

#### E6. VQ telemetry 常驻化

**目的**

不要只盯最终 accuracy, 要知道 remote 里到底发生了什么. 这一组指标应该作为后续所有实验的固定监控面板, 并在 train / val / test 三个 phase 都收集.

**改动**

- 把 VQ 指标和 attn 指标长期打出来.
- 尽量统一日志前缀, 避免 train / val / test 三个 phase 的命名不一致.

**记录指标**

- `attn/remote_win_rate`
- `attn/den_cache_ratio`
- `attn/o_remote_energy_ratio`
- `attn/o_remote_local_cos`
- `attn/remote_dominance_rate`
- `attn/q_rms_mean`
- `attn/k_rms_mean`
- `vq/k_norm_mean`
- `vq/k_hat_norm_mean`
- `vq/c_rms_mean`
- code usage entropy
- dead code ratio
- mean / max code load
- routing entropy
- top1-top2 margin
- quantization residual norm

**预期现象**

- 如果远程能量占比长期很低, 且 dominance rate 很低, 说明 remote 基本是陪跑.
- 如果远程占比不低, 但结果仍差, 说明不是没用, 而是 remote 读错, 混叠过重, 或表达力不足.

**下一步动作**

- 把 E6 作为之后所有实验的固定监控面板.

#### E7. Read-side top-k / soft routing probe

**目的**

验证问题是不是"单 code 读取太脆".

**改动**

- 写入保持 top1 / one-hot 不变
- 读取改成 `top-k=2`
- 读取改成 `top-k=4`
- 先只改 read-side, 不改 write-side

**记录指标**

- 全部 validation 指标
- `128+` 与大 `num_kv_pairs` case 的提升幅度
- routing entropy 与 top-k gain 的相关性

**预期现象**

- 如果明显上涨, 说明主瓶颈是 read-side routing error.
- 如果不怎么涨, 说明问题更多在桶内表达力或缺少 exact rescue.

**下一步动作**

- 若有效, 把 top-k routing 纳入基础版.
- 若弱效, 优先去做 E8 或后续 residual correction.

#### E8. Oracle / heuristic exact rescue probe

**目的**

验证"少量 exact lookup"能不能明显补 MQAR.

**改动**

先不做 learned rescue, 先做 heuristic:

- routing entropy 高时触发
- top1-top2 margin 小时触发
- residual 大时触发
- repeated token / repeated bigram 命中时触发

只对少量 token 走 small exact path.

**记录指标**

- 长度分桶提升
- 大 `num_kv_pairs` 提升
- 触发比例
- 每触发 `1%` token 带来的 gain
- 额外 FLOPs / latency

**预期现象**

- 若只触发很少 token 就涨很多, 说明 MQAR 最缺的是 recall-sensitive token 上的输入相关精确读.
- 若收益很小, 说明 remote 不是偶尔错, 而是整体表示能力不够.

**下一步动作**

- 若收益明显, 下一步优先做正式版 `Uncertainty-triggered Exact Rescue`.
- 若收益很小, 说明应继续修 remote state / routing 本体.

## 统一监控面板

后续所有实验尽量统一保留下面 7 组指标.

### 1. 主指标

- `valid/accuracy`
- `valid/loss`

### 2. 长度分桶

- `valid/input_seq_len/accuracy-{64,128,256,512,1024}`

### 3. case 分桶

- `valid/mqar_case/accuracy-*`

### 4. 记忆容量分桶

- `valid/num_kv_pairs/accuracy-{4,8,16,32,64,128,256}`

### 5. 训练效率

- step-to-`0.2 / 0.3 / 0.4`
- fixed step budget 下的 best accuracy
- train loss AUC

### 6. 远程贡献指标

对 `full_K vs local_only_K` 和 `remote_hist_1 vs local_only_1` 这两对实验, 优先看下面 5 类指标:

- `attn/remote_win_rate`
- `attn/den_cache_ratio`
- `attn/o_remote_energy_ratio`
- `attn/o_remote_local_cos`
- `attn/remote_dominance_rate`

这组指标比只看总 accuracy 更能判断 remote 到底有没有在工作.

### 7. VQ / routing 内部指标

- `attn/q_rms_mean`
- `attn/k_rms_mean`
- `vq/k_norm_mean`
- `vq/k_hat_norm_mean`
- `vq/c_rms_mean`
- code usage entropy
- dead code ratio
- max / mean code load
- routing entropy
- top1-top2 margin
- residual norm
- local / remote output norm ratio

这组指标尽量在 train / val / test 三个 phase 都收集, 并统一命名前缀, 方便跨实验横向比较.

## 结果决策树

- 如果 `full_K ≈ local_only_K`, 下一步优先修 remote, 不要再继续调 learning rate.
- 如果 `remote_hist_1 > local_only_1`, 但提升不大, 说明 remote 有一定价值, 但不足以支撑长距 / 大容量 case.
- 如果两对主对照都几乎没差, 说明当前瓶颈更偏 local path 主导, remote 基本形同虚设.
- 如果 codebook size 或 top-k routing 很有效, 下一步优先做 `top-k soft routing`, 再考虑 `PQ / RQ`.
- 如果 top-k 不怎么涨, 但 collision / residual 很差, 下一步优先做 `Code-local Residual Correction`.
- 如果长度轴稍有改善, 但大 `num_kv_pairs` 仍明显掉, 下一步做 `Per-code Low-rank Delta Memory`.
- 如果 exact rescue probe 只触发少量 token 就能明显上涨, 下一步优先做 `Uncertainty-triggered Exact Rescue`.

## 建议执行顺序

建议不要同时开太多线, 按下面顺序推进:

1. `E0`, 先做 sampler 重跑 baseline.
2. `E1`, 再做 `full_K` vs `local_only_K`.
3. `E2`, 再做 `remote_hist_1` vs `local_only_1`.
4. `E3`, 再做 fixed-local-coverage 对照.
5. `E4`, 再做单轴 MQAR 诊断.
6. `E5`, 再做 codebook size sweep.
7. `E6`, 把 telemetry 常驻化并纳入固定监控面板.
8. `E7 / E8`, 最后再做 top-k probe 和 exact-rescue probe.

## 补充说明

- 这份文档的目标是给 Flash-VQG 在 Zoology MQAR 上的定位实验提供统一顺序和统一判据.
- 这不是最终模型改造方案, 而是一份"先把问题定位清楚"的实验执行说明.
- 如果后续确认 sampler artifact 很重, 或 remote path 几乎不工作, 应优先更新本文件, 再继续扩展新实验.
