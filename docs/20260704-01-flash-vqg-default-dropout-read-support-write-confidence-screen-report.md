# 20260704-01 Flash-VQG default-dropout read support/write confidence screen report

## 结论

本轮 14 条 paired 1ep run 全部完成, 没有 NaN, OOM 或 Traceback. 共同口径是 same MQAR cache, same canonical seed124 init, same batch order, default dropout, `cb64-r16`, `write_topk=4`, `seed=124`, `data_seed=123`.

本轮定位是 exploratory 1ep screen, 不写 official MQAR ledger, 不进入正式推荐总表. 目的只是缩小后续稳定化方向.

本轮筛选目标是:

```text
final valid/mqar_case/accuracy-1024x256 两机都高,
且 2080ti vs 3090 gap <= 4pp.
```

没有任何 variant 完全通过这个标准. 但结果不是全失败, 而是给出一个很清楚的方向:

```text
read_topk=16 是当前 default dropout 下最值得继续看的 read support 宽度.
```

`fixed-r16` 两机 final hard slice 是:

```text
2080ti: 0.912
3090:   0.850
gap:    6.2pp
```

它没有进入 4pp 容忍线, 但明显优于这批其他配置, 并且两机整体 accuracy 也都高:

```text
2080ti final accuracy: 0.983
3090 final accuracy:   0.971
gap:                   1.2pp
```

所以它是候选方向, 不是最终结论. 下一步可以围绕 `read_topk=16` 做更小的稳定化和复现实验, 但不能直接升格为默认配置.

## 共同设置

| 项目 | 设置 |
|---|---|
| zoology commit | `d27ca50` |
| Flash-VQG commit | `94c1591` |
| seed | `124` |
| data seed | `123` |
| MQAR cache | canonical 13 files |
| init | canonical seed124 init |
| batch order | 两机一致 |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| train write | `write_topk=4` |
| epoch | 1 epoch, `704` optimizer steps |
| trace | disabled |

硬门槛:

- cache combined content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`
- init model state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`
- `read_trace_enabled=false`, `read_trace_train_steps=[]`, `train_inline_event_trace_enabled=false`

## 实验矩阵

本轮每台机器 7 个 run, 两台机器共 14 个 run.

| variant | read_topk | write strength | injection warmup | 目的 |
|---|---:|---|---:|---|
| `baseline-r2` | 2 | `renorm_topk` | 0 | 当前 default-dropout r2 baseline |
| `baseline-r4` | 4 | `renorm_topk` | 0 | 检查 default dropout 下 r4 是否仍危险 |
| `fixed-r8` | 8 | `renorm_topk` | 0 | read support 宽度 sweep |
| `fixed-r16` | 16 | `renorm_topk` | 0 | read support 宽度 sweep |
| `fixed-r64` | 64 | `renorm_topk` | 0 | dense/full-read 上界, chunked path |
| `write-mass-r2` | 2 | `topk_mass_scaled` | 0 | 保留 r2 read, 降低低置信 top-k write 过度放大 |
| `write-mass-injwarm512-r2` | 2 | `topk_mass_scaled` | 512 | write mass scaling + residual injection warmup |

`fixed-r64` 在正式启动前做过两机 smoke, 2080ti 和 3090 都通过, 因此本轮使用 r64, 没有降级到 r32.

## 主结果

核心指标使用 final validation 的 `valid/mqar_case/accuracy-1024x256`.

| variant | read_k | 2080ti | 3090 | gap | 判定 |
|---|---:|---:|---:|---:|---|
| `baseline-r2` | 2 | 0.823 | 0.452 | 37.1pp | fail |
| `baseline-r4` | 4 | 0.753 | 0.056 | 69.7pp | fail |
| `fixed-r8` | 8 | 0.865 | 0.000 | 86.5pp | fail |
| `fixed-r16` | 16 | 0.912 | 0.850 | 6.2pp | best signal, not pass |
| `fixed-r64` | 64 | 0.015 | 0.710 | 69.5pp | fail |
| `write-mass-r2` | 2 | 0.371 | 0.741 | 37.0pp | fail |
| `write-mass-injwarm512-r2` | 2 | 0.060 | 0.708 | 64.8pp | fail |

整体 accuracy:

| variant | 2080ti | 3090 | gap |
|---|---:|---:|---:|
| `baseline-r2` | 0.966 | 0.894 | 7.2pp |
| `baseline-r4` | 0.943 | 0.582 | 36.1pp |
| `fixed-r8` | 0.974 | 0.009 | 96.5pp |
| `fixed-r16` | 0.983 | 0.971 | 1.2pp |
| `fixed-r64` | 0.522 | 0.939 | 41.7pp |
| `write-mass-r2` | 0.850 | 0.953 | 10.3pp |
| `write-mass-injwarm512-r2` | 0.644 | 0.949 | 30.5pp |

## 结果解释

### 1. `read_topk=16` 是强信号, 但还不是稳定解

`fixed-r16` 是本轮唯一同时满足“两边都学得好”的配置. 它的 hard slice 为 `0.912/0.850`, 比 `baseline-r2` 的 `0.823/0.452` 明显更接近, 也比 r4/r8/r64 稳得多.

这支持一个判断:

```text
default dropout 下, 过窄 read support 会让 residual read 对早期扰动太敏感;
适度扩大 read support 可以缓解一部分 support flip 和错误候选截断.
```

但 `fixed-r16` 的 1024x256 gap 仍然是 6.2pp, 超过 4pp 容忍线. 所以它只能作为下一轮重点候选, 不能作为收敛结论.

### 2. `read_topk` 不是越大越好

`fixed-r64` 是 full-code/dense read 上界, 但它没有稳定:

```text
2080ti: 0.015
3090:   0.710
gap:    69.5pp
```

这说明问题不是简单地“读越多越稳”. 全读改变了 residual proposal 的组合方式, 也改变了 residual injection 的统计分布. 在 default dropout 下, 过宽 support 可能把更多不可靠 residual correction 注入输出.

这一点和 no-dropout 阶段的结论要区分开:

```text
no-dropout dense/full read 可以作为去掉 top-k 离散 flip 的 diagnostic control;
default dropout 下 full read 不自动等于稳定训练方案.
```

### 3. `read_topk=4/8` 在 default dropout 下仍然危险

本轮 `baseline-r4` 和 `fixed-r8` 都失败, 且呈现强机器依赖:

```text
r4: 2080ti 0.753 vs 3090 0.056
r8: 2080ti 0.865 vs 3090 0.000
```

这进一步确认: no-dropout 下 r4 强, 不代表 default dropout 下 r4/r8 可直接使用. dropout 是正常训练协议, 问题不是把 dropout 当 bug, 而是当前 gd_residual_v1 的 read/write/state/residual 注入链路对 dropout 引入的训练扰动过敏.

### 4. `topk_mass_scaled` write strength 没有解决问题

`write-mass-r2` 和 `write-mass-injwarm512-r2` 都没有通过. 它们在 3090 上分数较高, 但 2080ti 明显低:

```text
write-mass-r2:              0.371 vs 0.741
write-mass-injwarm512-r2:   0.060 vs 0.708
```

这说明简单把 residual write strength 按 top-k mass 缩放, 不能稳定地把两机带到同一个高分 basin. 结合历史 update cap/injection warmup 实验, 这更像是:

```text
标量限制或标量缩放可以改变轨迹,
但不足以解决 read/write support 和 M_state 递推之间的耦合不稳定.
```

### 5. 诊断指标显示轨迹差异不只是最终 accuracy 差异

几个例子:

- `baseline-r2`: 3090 的 `update_norm_max=2.46`, `m_norm_max=7.09`, 高于 2080ti 的 `1.01/4.37`, 同时 1024x256 低很多.
- `fixed-r16`: 3090 的 `update_norm_max=3.92`, `m_norm_max=9.97`, 明显高于 2080ti 的 `0.152/2.04`, 但仍能保持高分. 这说明大 state/update 指标不是单独充分条件, 还要看 read support 和 injection 组合.
- `fixed-r64`: 两机 read selected mass 接近, 但效果相反, 说明只看 selected mass 不够, full read 的 residual proposal 质量和后续轨迹也关键.

所以当前不能把根因简化为单一指标, 更准确的表述是:

```text
default dropout 扰动进入 Flash-VQG 后,
read support 宽度, write support/strength, M_state update, residual injection
共同决定是否把扰动放大成明显效果差异.
```

## 和历史实验的关系

这轮结果补上了一个重要空白:

- 之前 no-dropout 下 `fixed-r4` 很强, 但 default dropout 下 r4 崩.
- 之前 dense/full read 作为 diagnostic 能缓解部分 no-dropout cross-machine gap, 但本轮 default dropout 下 r64 不是稳态解.
- 之前 injection warmup/update cap/write limiter 都有局部信号, 但复现不稳.
- 本轮显示 `read_topk=16` 是当前 default dropout 下最强的单一候选, 比 r2/r4/r8/r64 更值得继续.

这不是说最终方案就是固定 r16. 更合理的理解是:

```text
模型需要一个“适度宽但不过宽”的 read support 区间.
```

后续应该围绕这个区间做稳定化, 而不是继续在 r2/r4 上硬补 scalar limiter, 也不是直接 full read.

## 下一步建议

### P0: 复现 `fixed-r16`

先不要跑 4ep. 先做低成本 paired 1ep 复现:

```text
seed=123 或 seed=125
same cache/init/batch
default dropout
read_topk=16
machines=2080ti + 3090
```

目的:

```text
确认 r16 是稳定趋势, 还是本轮 seed124 的偶然好轨迹.
```

判定:

- 如果 r16 多 seed 仍高分且 gap 接近或低于 4pp, 再考虑 4ep confirm.
- 如果 r16 复现失败, 说明 read support 宽度本身仍然不够, 要转向 support guard/adaptive read.

### P1: read support schedule, 不要直接固定 full read

围绕 r16 做 schedule:

```text
sched32to16-linear512
sched16to8-linear512
```

目标是训练早期给更宽 support, 后期收回到更便宜的 support. 但不能再用 `64 -> small` 作为优先方向, 因为 r64 在本轮 default dropout 下强机器依赖.

### P2: support-aware residual injection

历史 warmup 说明 residual injection 参与分叉, 但单独 warmup 不稳定. 后续应把 injection 强度和 support confidence 绑定, 例如:

```text
read margin 小, read entropy 高, 或 selected mass 低时, 降低 residual injection.
```

这比固定 scalar cap 更符合当前证据, 因为不稳定不是单纯幅度问题, 而是“低置信 support 下仍然强注入 residual”.

### 暂不建议

- 不建议把 `fixed-r64` 当默认方案.
- 不建议推进 `write-mass-r2` 或 `write-mass-injwarm512-r2` 的 4ep confirm.
- 不建议继续密集 sweep scalar cap/softcap.
- 不建议把 r4 default dropout 继续长训.

## Artifact

核心文件:

- `docs/artifacts/20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen/final-summary.csv`
- `docs/artifacts/20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen/paired-summary.csv`
- `docs/artifacts/20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen/source-manifest.csv`
- `docs/artifacts/20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen/metadata.json`
- `docs/artifacts/20260704-01-flash-vqg-default-dropout-read-support-write-confidence-screen/README.md`
