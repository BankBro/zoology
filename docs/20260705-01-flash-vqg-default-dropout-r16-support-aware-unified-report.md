# 20260705-01 Flash-VQG default-dropout r16 support-aware unified screen report

## 摘要

本轮实验按 plan 统一执行 P0-P3: 复跑 `fixed-r16`, 扩大 read support, 做 256 step read/write/injection trace, 并筛查几条 support-aware 稳定化干预. 训练协议保持 default dropout, 即 `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`; 数据和初始化保持 canonical cache/init; 任务仍然是 `cb64-r16`, `write_topk=4`, seed=124, data_seed=123.

结果比较明确: 这批没有找到可推进的稳定方案. 更重要的是, 它排除了几条看起来可能有用的路线. 之前看起来不错的 `fixed-r16` 在 same-seed paired 1ep 复跑中没有稳定复现, 结果为 2080ti `0.861`, 3090 `0.725`, gap `13.6pp`. 更大的 read support 或简单 read schedule 也没有解决问题. 当前实现的 read-confidence injection, softread, write-mass scaling 也都没有过线, 其中 softread 反而非常不稳定.

本轮最有价值的结论是: 问题不能再简化成“把 read_topk 调大一点就行”. 在 default dropout 下, read support 翻转确实是放大器之一, 但即使 read support 消除翻转, 不同机器的 residual state, injection, write/update 轨迹仍可能明显不同. 后续应该停止大范围 K 值扫参, 转向更原则性的 residual memory 稳定化设计.

## 实验设置

- zoology branch: `flash-vqg`
- zoology commit: `e9f1651`
- Flash-VQG branch: `20260428-gd-residual-v1-sync`
- Flash-VQG commit: `dc558cf`
- canonical init state sha256: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`
- canonical init match: 2080ti `true`, 3090 `true`
- training dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`
- VQ/GD residual: `num_codebook_vectors=64`, `fox_gd_residual_rank=16`, `fox_gd_residual_write_topk=4`
- main metric: `final_1024x256_accuracy`

正式队列状态:

| machine | formal runs | completed | failed | note |
|---|---:|---:|---:|---|
| 3090 | 15 | 15 | 0 | 全部 formal variants 完成 |
| 2080ti | 12 | 12 | 0 | `fixed-r24`, `fixed-r32`, `sched32to16-linear512` 在 smoke 阶段 OOM, 因此没有正式启动 |

256 step trace rows 只用于诊断 early-window 指标, 不用于判断 1ep 训练效果. 它们在主结果表里的 1024x256 near-zero 是因为只训练到 256 optimizer steps, 不能解释为该模型配置完整 1ep 失败.

## 主结果

| variant | 2080ti final | 3090 final | gap | judgement |
|---|---:|---:|---:|---|
| `p0-fixed-r16-repro` | 0.861 | 0.725 | 13.6pp | fail |
| `fixed-r24` | - | 0.86 | - | 3090 only; 2080ti skipped/OOM |
| `fixed-r32` | - | 0.667 | - | 3090 only; 2080ti skipped/OOM |
| `sched32to16-linear512` | - | 0.764 | - | 3090 only; 2080ti skipped/OOM |
| `sched16to8-linear512` | 0.863 | 0.503 | 36.0pp | fail |
| `trace-r2-readwrite-256` | 0.00025 | 0.00025 | 0.0pp | diagnostic only, 256 steps |
| `trace-r4-read-256` | 0.00025 | 0.000254 | 0.0pp | diagnostic only, 256 steps |
| `trace-r16-readwrite-256` | 0.00025 | 0.000254 | 0.0pp | diagnostic only, 256 steps |
| `trace-r64-read-256` | 0.00025 | 0.000254 | 0.0pp | diagnostic only, 256 steps |
| `r16-injconf` | 0.618 | 0.568 | 5.0pp | fail |
| `r16-softread` | 0.0148 | 0.9 | 88.5pp | fail |
| `r16-softread-injconf` | 0.8 | 0.402 | 39.8pp | fail |
| `r2-injconf` | 0.42 | 0.897 | 47.7pp | fail |
| `r16-write-mass` | 0.505 | 0.623 | 11.8pp | fail |
| `r16-write-mass-injconf` | 0.376 | 0.643 | 26.7pp | fail |

筛选标准仍按之前讨论的口径: 1024x256 hard slice 不低于 `0.85`, 且 paired gap 不超过 `4pp`. 本轮 paired 候选没有一个通过.

## 结果解读

### 1. `fixed-r16` 不能作为已验证稳定配置

之前 `fixed-r16` 有过 `0.912/0.850` 的较好信号, 因此本轮 P0 首先做 same-seed paired 1ep 复跑. 复跑结果是 `0.861/0.725`, gap `13.6pp`. 这说明 `fixed-r16` 至少在 default dropout + same cache/init + seed124 条件下还不能稳定复现.

这不是说 `read_topk=16` 完全没价值. 更准确的判断是: 单次较好结果不能被当作候选方案成立, 后续任何候选都需要至少 same-seed paired rerun 过线, 再谈换 seed.

### 2. 继续增大 read support 不是充分解

3090 单机上 `fixed-r24=0.860`, 但 `fixed-r32=0.667`, `sched32to16=0.764`. 2080ti 对这三条在 smoke 阶段 OOM, 所以没有 paired 结论. `sched16to8` 能在两机跑, 但结果是 `0.863/0.503`, gap `36.0pp`.

这说明“大 K”不是一个稳妥的单调改进方向. 它可能在某些轨迹上帮助 read support 覆盖, 也可能引入更多不可靠 residual candidate, 或者改变 residual branch 的输出分布. 在 2080ti 上, 大 K 还有明确显存约束.

### 3. 当前 support-aware 小补丁没有解决问题

本轮 P3 测了三类简单干预:

- `read-confidence-gated residual injection`: 根据 read margin 降低 residual 注入强度.
- `softread`: 在 selected top-k 内部做 softmargin/temperature 型平滑.
- `write-mass`: 用已有 `topk_mass_scaled` 改写 residual write strength.

结果都没有过线. `r16-injconf` 是 `0.618/0.568`, 分数太低; `r2-injconf` 是 `0.420/0.897`, 机器差异极大; `r16-softread` 是 `0.0148/0.900`, 说明当前 softread 设计本身很危险; `write-mass` 系列也没有稳定化效果.

因此, 当前这些实现不能作为后续主线. 特别是 softread, 不能因为概念上“更平滑”就默认有效. 这条路径需要先重新检查数学定义和数值范围, 否则会把 residual read 分布改坏.

### 4. Trace 显示 read support 是放大器, 但不是全部

P2 trace 在 step 0, 16, 64, 128, 256 上记录了 read support churn/top1 flip, selected mass, update norm, M norm, lambda 和 injection ratio. step 256 摘要如下:

| machine | variant | loss | churn | top1 flip | selected mass | update max | M max | inject |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2080ti | `trace-r16-readwrite-256` | 8.390 | 0.065 | 0.001 | 0.902 | 2.591 | 4.554 | 0.008 |
| 3090 | `trace-r16-readwrite-256` | 8.388 | 0.134 | 0.181 | 0.737 | 1.026 | 3.369 | 0.003 |
| 2080ti | `trace-r2-readwrite-256` | 8.388 | 0.510 | 0.694 | 0.477 | 0.790 | 2.302 | 0.001 |
| 3090 | `trace-r2-readwrite-256` | 8.388 | 0.184 | 0.132 | 0.551 | 0.830 | 2.866 | 0.002 |
| 2080ti | `trace-r4-read-256` | 8.391 | 0.193 | 0.327 | 0.791 | 1.850 | 3.624 | 0.003 |
| 3090 | `trace-r4-read-256` | 8.393 | 0.080 | 0.218 | 0.677 | 1.360 | 3.282 | 0.002 |
| 2080ti | `trace-r64-read-256` | 8.390 | 0.000 | 0.000 | 0.834 | 1.035 | 3.981 | 0.004 |
| 3090 | `trace-r64-read-256` | 8.389 | 0.000 | 0.000 | 0.909 | 0.937 | 1.971 | 0.002 |

几个观察:

- `r64` 的 read churn/top1 flip 为 0, 符合预期, 因为 dense/full read 没有 top-k support 翻转.
- 但是 `r64` 的 M norm, injection ratio, selected mass 等仍存在跨机器差异. 这说明 read support 翻转不是唯一放大器.
- `r16` 在 2080ti 的 `update_norm_max=2.591`, 高于 3090 的 `1.026`; M norm 也更高. 这支持之前“residual memory update/state 本身会放大早期扰动”的判断.
- `r2` 和 `r4` 的 read support 也有明显 churn/top1 flip, 但不同机器的方向并不一致. 这说明简单地按 K 值排序不能解释全部现象.

这些 trace 结果更像在支持一个组合结论: default dropout 下, early hidden-state 扰动进入 VQ routing, residual write, M_state 累积, residual read/injection 之后, 多个机制一起放大轨迹差异. read top-k 是其中一个放大器, 但不是唯一控制旋钮.

## 和历史实验的关系

历史上 no-dropout + canonical cache/init 下, `fixed-r4` 是强信号, 跨机器 gap 很小. default dropout 加回后, `fixed-r4` 直接崩到很低分. 后续 `update_norm_cap=0.5` 和 injection warmup 曾经给出过缓解信号, 但复跑稳定性也不足. 本轮进一步说明, 仅靠扩大 read_topk 或简单 support-aware gating, 不能替代对 residual memory 写入, 状态增长, 读出注入这几条路径的系统性稳定化.

因此, 当前最稳的判断不是“某一个 K 值最好”, 而是:

> Flash-VQG 的 `gd_residual_v1` 在 default dropout 下仍然过度敏感. 低位数值差异和 dropout 扰动会进入 VQ-indexed residual memory, 再经 sparse read/write 和 residual injection 放大. 现有简单补丁不能可靠抑制这种放大.

## 结论

1. `fixed-r16` same-seed paired 1ep 没有稳定复现, 不能推进为候选默认配置.
2. 大 read support 不是单调有效方案. `r24` 只在 3090 单机达到 `0.860`, `r32` 和 schedule 表现较差, 且 2080ti 大 K 有 OOM 约束.
3. 当前 read-confidence injection, softread, write-mass scaling 都没有解决 default dropout 下的跨机器不稳定. softread 尤其危险.
4. 256 step trace 说明 read support 翻转确实存在, 但 full read 消除 support 翻转后仍保留 residual state/injection 差异. 问题不止 read candidate flip.
5. 后续不应继续做大范围 K 值扫参. 更合理的方向是围绕 residual memory 的写入幅度, 状态范数, 注入强度和 support 置信度做更原则性的设计.

## 下一步建议

短期不要再开 4ep 或大网格. 下一轮应以“先验证机制, 再扩实验”为原则:

1. 回到历史上相对有信号的两类干预, 做 same-seed paired rerun:
   - `update_norm_cap` 或更平滑的 scheduled update cap.
   - residual injection warmup / residual branch schedule.

2. 如果做新机制, 不要沿用本轮 softread 的实现口径. 更合适的是先设计更保守的机制:
   - residual branch 早期 schedule: 先降低读出来的 residual correction 对最终输出的影响, 不改 M_state 写入语义.
   - M_state update soft cap: 限制单次写入尖峰, 但保留梯度和连续性.
   - support confidence guard: 在 read/write margin 很低时降低 residual 贡献, 但必须保留 floor, 避免整条 residual branch 被关死.
   - code/head local diagnostics: 继续看大 update 是否集中在少数 code/head, 再决定是否做 code-aware 控制.

3. 每个候选先只做 paired 1ep, 且同 seed 至少复跑一次. 只有两个 same-seed paired run 都过 `0.85` 和 `4pp` 线, 才值得扩到不同 seed 或 4ep.

## Artifact

- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/run-summary.csv`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/cross-machine-comparison.csv`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/variant-summary.csv`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/formal-early-window-summary.csv`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/formal-early-window-cross-machine.csv`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/formal-early-window-step256-summary.csv`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/cache-init-preflight-summary.csv`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/source-manifest.csv`
