# 20260702-02 Flash-VQG training-minibatch residual event trace 报告

status: completed_diagnostic
ledger: not written

## 结论摘要

本轮完成了真实 training-minibatch inline trace. 它比上一轮 fixed validation-batch snapshot 更接近真实训练现场.

核心结论是:

```text
大 M_state residual update 确实发生在真实训练 batch 里,
而且这些 update 会在跨机器轨迹中表现出明显不同的 hit pattern 和 code/head 热点.

但是 hard update_norm_cap=0.5 不是稳定解法.
它不是只挡少数尖峰, 而是会在中后期大面积介入训练,
并且在不同机器轨迹上介入程度不同.
```

本轮主结果没有通过用户接受的 `4pp` hard-slice gap 线:

| variant | cap | 2080ti `1024x256` | 3090 `1024x256` | gap | within 4pp |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline-r2` | unset | `0.664` | `0.786` | `12.2pp` | no |
| `ucap0p5-r2` | `0.5` | `0.827` | `0.542` | `28.5pp` | no |

这说明两件事要分开看:

1. `M_state` update 幅度是当前证据支持的放大环节之一. 真实训练 batch 里能看到大量超过 `0.5` 的候选 update, 且集中在少数 layer/head/code 上.
2. `cap=0.5` 这个 hard cap 不是可直接推进的解决方案. 它会改变训练, 但改变方式是轨迹依赖的, 不是稳定地把两台机器拉近.

一句话:

```text
这轮把问题从 "validation snapshot 看到大 update" 推进到了
"真实训练 batch 确实发生大 update, 且 hard cap 会强烈、轨迹依赖地介入这些 update".

下一步不应该继续把 hard cap 当候选默认配置,
而应该设计更原则的 soft/scheduled residual control 和 read-support 稳定化.
```

## 实验设置

共同条件:

| 项 | 值 |
| --- | --- |
| experiment id | `20260702-02-flash-vqg-training-minibatch-event-trace` |
| zoology branch / commit | `flash-vqg` / `8a14276` |
| Flash-VQG branch / commit | `20260428-gd-residual-v1-sync` / `0117dcd` |
| seed | `124` |
| data seed | `123` |
| data/init | canonical MQAR cache + canonical seed124 init |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| residual read/write | `read_topk=2`, `write_topk=4` |
| train length | 1 epoch, `704` optimizer steps |
| machines | `2080ti` + `3090`, both inside `Flash-VQG-tun` container |

Variants:

| variant | `fox_gd_residual_update_norm_cap` | hypothetical cap | 目的 |
| --- | ---: | ---: | --- |
| `baseline-r2` | unset | `0.5` | 观察真实训练 batch 中如果套 `cap=0.5` 会命中哪些 update |
| `ucap0p5-r2` | `0.5` | `0.5` | 观察 actual cap 在真实训练 batch 中实际拦截了哪些 update |

本轮是 diagnostic/probe, 不写 official MQAR ledger. Artifact 目录:

```text
docs/artifacts/20260702-02-flash-vqg-training-minibatch-event-trace/
```

## Trace 口径

本轮有两类 trace, 必须区分:

| trace 类型 | 文件 | 语义 |
| --- | --- | --- |
| validation snapshot trace | `read-trace-*`, `early-window-*`, `cap-metrics-*` | 在指定训练进度上额外跑 fixed validation batch 的 eval forward, 用于和历史 read-support 证据对齐 |
| training-minibatch inline trace | `train-inline-event-*`, `cap-hit-timeline.csv`, `code-head-hotspot-summary.csv` | 在真实 training batch forward 中记录 top residual update event, 该 forward 继续参与 backward 和 optimizer step |

训练 forward 发生在 optimizer step 递增前, 所以 inline `train_step=703` 表示产生第 `704` 次 optimizer update 的训练窗口.

## 前置一致性

cache/init/batch order 全部通过跨机器检查:

| target | field | all match | hash |
| --- | --- | --- | --- |
| `baseline-r2` | MQAR cache content | true | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| `baseline-r2` | init model state | true | `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0` |
| `baseline-r2` | batch order | true | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |
| `ucap0p5-r2` | MQAR cache content | true | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| `ucap0p5-r2` | init model state | true | `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0` |
| `ucap0p5-r2` | batch order | true | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

因此本轮结果不能解释为数据, 初始权重, 或 batch order 不一致.

## First Mismatch

hash probe 的 first mismatch:

| target | first mismatch stage | optimizer step | micro step | field | module |
| --- | --- | ---: | ---: | --- | --- |
| `baseline-r2` | `forward_before_backward_step0_micro0` | `0` | `0` | `module_output_sha256` | `backbone.layers.0.dropout1` |
| `ucap0p5-r2` | `forward_before_backward_step0_micro0` | `0` | `0` | `module_output_sha256` | `backbone.layers.0.dropout1` |

这是正常训练协议下的预期现象, 不是 bug. 本轮保留 default dropout, 重点不是让 dropout 消失, 而是判断 Flash-VQG/GD residual 后续机制是否能承受这种正常训练扰动.

## 主结果

`1024x256` hard slice:

| variant | machine | duration min | final valid acc | final valid loss | final `1024x256` |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline-r2` | 2080ti | `140` | `0.940` | `0.570` | `0.664` |
| `baseline-r2` | 3090 | `80` | `0.957` | `0.497` | `0.786` |
| `ucap0p5-r2` | 2080ti | `155` | `0.966` | `0.358` | `0.827` |
| `ucap0p5-r2` | 3090 | `110` | `0.914` | `0.738` | `0.542` |

解读:

- `baseline-r2` 仍然跨机器不稳, gap 为 `12.2pp`.
- `ucap0p5-r2` 在 2080ti 上比 baseline 高, 但在 3090 上明显低, gap 扩大到 `28.5pp`.
- 这和 `20260701-04` 中 `cap=0.5` 曾经 `2.8pp` 的结果不同. 所以 `cap=0.5` 不能被视为稳健候选, 只能作为 diagnostic intervention.

## Read Support

fixed validation snapshot 的跨机器 read support 对齐情况:

| target | step | top1 match | top-k exact match | top-k overlap |
| --- | ---: | ---: | ---: | ---: |
| `baseline-r2` | `0` | `1.000` | `1.000` | `1.000` |
| `baseline-r2` | `16` | `0.656` | `0.312` | `0.688` |
| `baseline-r2` | `128` | `0.531` | `0.172` | `0.438` |
| `baseline-r2` | `192` | `0.109` | `0.000` | `0.242` |
| `baseline-r2` | `704` | `0.047` | `0.000` | `0.156` |
| `ucap0p5-r2` | `0` | `1.000` | `1.000` | `1.000` |
| `ucap0p5-r2` | `16` | `0.656` | `0.312` | `0.688` |
| `ucap0p5-r2` | `128` | `0.531` | `0.172` | `0.438` |
| `ucap0p5-r2` | `192` | `0.016` | `0.000` | `0.172` |
| `ucap0p5-r2` | `704` | `0.141` | `0.016` | `0.141` |

这个表说明:

```text
read support 分叉发生得很早.
到 step16, top-k exact match 已经只有 31.2%.
到 step128, top-k exact match 只有 17.2%.
```

同时, `cap=0.5` 并没有修复 read support 对齐. 它不把两台机器拉回同一条离散 read path.

## Training-Minibatch Inline Event

真实 training minibatch 的 aggregate update event:

| variant | machine | records | update max | update mean | update p95 | actual cap hit | hypothetical cap hit | top code | top code share | top head | top head share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-r2` | 2080ti | `4288` | `2.918` | `0.521` | `2.235` | `0.000` | `0.210` | `39` | `0.324` | `0` | `0.583` |
| `baseline-r2` | 3090 | `4288` | `4.083` | `0.522` | `1.470` | `0.000` | `0.298` | `39` | `0.499` | `0` | `0.688` |
| `ucap0p5-r2` | 2080ti | `4288` | `2.468` | `0.381` | `1.805` | `0.125` | `0.125` | `50` | `0.339` | `0` | `0.509` |
| `ucap0p5-r2` | 3090 | `4288` | `3.979` | `0.558` | `2.194` | `0.298` | `0.298` | `39` | `0.338` | `0` | `0.528` |

这里的 `update max/p95` 是 cap 前候选 update 的 norm. `actual cap hit` 表示实际启用 cap 后该 event 是否被缩放.

直接结论:

```text
大 update 不是 validation snapshot 假象.
它们真实出现在 training minibatch 的 forward 中.
```

而且, 如果使用 `0.5` 作为阈值, baseline 中有相当比例的 training event 会被命中:

- 2080ti hypothetical hit ratio: `21.0%`.
- 3090 hypothetical hit ratio: `29.8%`.

这已经不是极少数单点 outlier.

术语说明:

| 指标 | 含义 |
| --- | --- |
| `actual_cap_hit` | 当前 run 真实启用了 `fox_gd_residual_update_norm_cap`, 该 update 在训练中实际被 cap 缩放 |
| `hypothetical_cap_hit` | 当前 run 不一定启用 cap, 但 trace 事后假设阈值为 `0.5`, 计算如果启用这个 cap, 该 update 是否会被缩放 |

因此 `baseline-r2` 的 `hypothetical_cap_hit_ratio=0.298` 不是说 baseline 训练真的被 cap 了, 而是说:

```text
baseline 真实训练没有 cap,
但如果当时使用 cap=0.5,
被记录的 top residual update event 中约 29.8% 会被截断.
```

这个指标的价值是估计 hard cap 会影响多少真实训练 event. 如果 hit ratio 很低, cap 更像 spike guard; 如果 hit ratio 很高, cap 更像大面积改变训练语义的 intervention.

## Cap Hit Timeline

真实 training minibatch 中的 selected steps:

| variant | step | update max 2080ti | update max 3090 | hypothetical hit 2080ti | hypothetical hit 3090 | actual hit 2080ti | actual hit 3090 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-r2` | `0` | `0.219` | `0.223` | `0.000` | `0.000` | `0.000` | `0.000` |
| `baseline-r2` | `48` | `0.603` | `0.614` | `0.023` | `0.215` | `0.000` | `0.000` |
| `baseline-r2` | `192` | `1.123` | `1.756` | `0.117` | `1.000` | `0.000` | `0.000` |
| `baseline-r2` | `256` | `1.980` | `2.495` | `0.613` | `1.000` | `0.000` | `0.000` |
| `baseline-r2` | `384` | `2.918` | `1.281` | `1.000` | `1.000` | `0.000` | `0.000` |
| `baseline-r2` | `512` | `1.728` | `1.331` | `1.000` | `1.000` | `0.000` | `0.000` |
| `baseline-r2` | `703` | `2.358` | `4.083` | `1.000` | `1.000` | `0.000` | `0.000` |
| `ucap0p5-r2` | `0` | `0.219` | `0.223` | `0.000` | `0.000` | `0.000` | `0.000` |
| `ucap0p5-r2` | `48` | `0.603` | `0.613` | `0.023` | `0.215` | `0.023` | `0.215` |
| `ucap0p5-r2` | `192` | `1.174` | `1.690` | `0.156` | `1.000` | `0.156` | `1.000` |
| `ucap0p5-r2` | `256` | `0.866` | `1.988` | `0.480` | `1.000` | `0.480` | `1.000` |
| `ucap0p5-r2` | `384` | `2.468` | `2.538` | `1.000` | `1.000` | `1.000` | `1.000` |
| `ucap0p5-r2` | `512` | `0.358` | `1.205` | `0.000` | `1.000` | `0.000` | `1.000` |
| `ucap0p5-r2` | `703` | `0.723` | `3.979` | `0.542` | `1.000` | `0.542` | `1.000` |

这个 timeline 是本轮最关键的定位证据之一.

它说明:

1. training inline trace 中 `step0` 到 `step32` 基本没有 cap hit; fixed validation snapshot 的 read support 在同一训练进度区间已开始分叉.
2. `step48` 开始出现 cap hit, 而且 3090 的 hit ratio 明显高于 2080ti.
3. 到 `step192/256` 以后, 3090 很多 traced top events 全部超过 `0.5`.
4. 到中后期, cap 不是只挡几个尖峰, 而是在大量 top event 上介入训练.

所以 `cap=0.5` 更像是:

```text
一个会大面积改变 residual write/state 轨迹的 hard intervention,
不是只对罕见异常尖峰做保护的 spike guard.
```

这提供了一个合理解释: 它在某些 run 里可能缓解下游破坏性, 但本轮显示这种 hard intervention 本身也会随轨迹变化而不稳.

## Post-Hoc Observations

在收尾后继续从已有 CSV 做了几项事后统计. 这些不是新的训练实验, 只是对本轮 artifact 的二次分析.

### read support 早于大规模 cap hit 分叉

跨机器 read support 的分叉很早:

| 事件 | baseline-r2 | ucap0p5-r2 |
| --- | ---: | ---: |
| top-k exact match 低于 `50%` | step `8` | step `8` |
| top-k exact match 低于 `25%` | step `64` | step `64` |
| top-k exact match 接近 `0` | step `192` | step `192` |
| cap/hypothetical cap 开始 hit | step `48` | step `48` |
| 3090 cap hit 达到 `100%` | step `192` | step `192` |
| 2080ti cap hit 达到 `100%` | step `384` | step `384` |

这说明:

```text
read support 分叉早于大规模 cap hit.
hard cap 主要处理下游 residual update/state 的破坏性,
不是阻止早期 read path 分叉.
```

### cap 的早期影响不会立刻改变 read-support summary

`baseline-r2` 和 `ucap0p5-r2` 在 step `0-128` 的 fixed validation snapshot read-support summary 完全一致. 但 training inline trace 显示 `ucap0p5-r2` 从 step `48` 开始已经出现 actual cap hit.

这说明:

```text
早期少量 cap hit 没有立刻改变 fixed validation read-support summary.
真正明显的轨迹重塑发生在 step192 以后,
尤其是 cap hit 变成大面积之后.
```

### 3090 更早进入 cap-active update regime

`ucap0p5-r2` 的 actual cap hit ratio:

| step | 2080ti | 3090 |
| ---: | ---: | ---: |
| `48` | `0.023` | `0.215` |
| `192` | `0.156` | `1.000` |
| `256` | `0.480` | `1.000` |
| `384` | `1.000` | `1.000` |
| `512` | `0.000` | `1.000` |
| `703` | `0.542` | `1.000` |

这进一步说明 hard cap 不是在两台机器上提供相同训练语义. 3090 更早进入几乎所有 traced top update 都被 cap 的状态, 这会把两台机器推向不同的 residual write/state 轨迹.

### loss 明显拉开集中在 step384 到 step512

fixed validation snapshot loss gap:

| variant | step384 gap | step512 gap | step704 gap |
| --- | ---: | ---: | ---: |
| `baseline-r2` | `0.044` | `1.939` | `0.705` |
| `ucap0p5-r2` | `0.099` | `3.477` | `1.625` |

读法:

```text
早期 read support 已经分叉, 但 loss 仍接近.
真正可见的 loss 分叉集中出现在 step384 -> step512.
```

如果后续要做更细 event trace, 不需要全程高频 dump; 应优先盯 `step256-512` 这个窗口.

### code/head hotspot 是轨迹现象, 不是固定 code bug

top event 的主热点会迁移:

| target | machine | 关键迁移 |
| --- | --- | --- |
| `baseline-r2` | 2080ti | early scattered -> head0/code39 -> head1/code50 -> head0/code39 |
| `baseline-r2` | 3090 | early scattered -> head0/code39 -> head1/code50 -> head0/code39 |
| `ucap0p5-r2` | 2080ti | early scattered -> head0/code39 -> head1/code50 -> head0/code39 -> head1/code50 |
| `ucap0p5-r2` | 3090 | early scattered -> head0/code39 -> head1/code50 -> head1/code12 |

所以本轮不能说 `code39`, `code50`, 或 `code12` 是固定 bug. 更合理的判断是:

```text
某些 code/head bucket 会在特定训练轨迹中成为 residual update pressure 的吸收点.
```

如果后续做 code-aware 控制, 应该按 bucket pressure 自适应, 不是手工处理某个 code.

### lambda/inject 也是下一步应拆的环节

step `512` 时, loss 已明显拉开, 同时 residual injection 相关指标也明显不同:

| variant | metric | 2080ti | 3090 |
| --- | --- | ---: | ---: |
| `baseline-r2` | `lambda_mean` | `0.125` | `0.023` |
| `baseline-r2` | `inject_ratio` | `0.696` | `0.114` |
| `baseline-r2` | loss | `6.554` | `8.494` |
| `ucap0p5-r2` | `lambda_mean` | `0.131` | `0.099` |
| `ucap0p5-r2` | `inject_ratio` | `0.689` | `0.542` |
| `ucap0p5-r2` | loss | `3.259` | `6.736` |

这说明 final gap 不只是 write/update 的事. Residual branch 读出来以后如何通过 `lambda` 和 injection 加回主输出, 也在 loss 分叉窗口中表现出明显跨机器差异.

因此下一步做 `residual injection / lambda warmup` 是有数据支撑的, 不是单纯调参.

## Validation Snapshot Scalar Metrics

step704 fixed validation snapshot scalar:

| variant | machine | loss | cap hit | update p95 | update max | M max | lambda mean | inject ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-r2` | 2080ti | `2.521` | `0.000` | `0.485` | `2.091` | `6.726` | `0.090` | `0.622` |
| `baseline-r2` | 3090 | `1.815` | `0.000` | `0.361` | `4.024` | `10.869` | `0.199` | `0.206` |
| `ucap0p5-r2` | 2080ti | `1.467` | `0.000076` | `0.133` | `0.617` | `4.773` | `0.115` | `0.397` |
| `ucap0p5-r2` | 3090 | `3.092` | `0.053` | `0.527` | `4.460` | `5.176` | `0.089` | `0.425` |

读法:

- validation snapshot 和 training inline trace 方向一致: 3090 `ucap0p5-r2` 仍进入更 cap-active 的 update regime.
- `ucap0p5-r2` 的 2080ti final snapshot 几乎不 hit cap, 但 3090 snapshot hit ratio 为 `5.3%`.
- 这说明 cap 对不同机器轨迹的实际介入程度并不一样.

## Code/Head Hotspot

真实 training inline event 的 top hotspots:

| variant | machine | head | code | event share | update max | update p95 | cap/hyp hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-r2` | 2080ti | `0` | `39` | `0.323` | `2.918` | `2.422` | `0.355` hypothetical |
| `baseline-r2` | 2080ti | `1` | `50` | `0.264` | `1.655` | `0.594` | `0.105` hypothetical |
| `baseline-r2` | 3090 | `0` | `39` | `0.498` | `4.083` | `2.913` | `0.534` hypothetical |
| `baseline-r2` | 3090 | `1` | `50` | `0.159` | `2.129` | `0.981` | `0.202` hypothetical |
| `ucap0p5-r2` | 2080ti | `1` | `50` | `0.337` | `1.174` | `0.531` | `0.103` actual |
| `ucap0p5-r2` | 2080ti | `0` | `39` | `0.317` | `2.468` | `2.010` | `0.283` actual |
| `ucap0p5-r2` | 3090 | `0` | `39` | `0.337` | `1.577` | `0.847` | `0.313` actual |
| `ucap0p5-r2` | 3090 | `1` | `50` | `0.265` | `3.979` | `2.147` | `0.521` actual |

本轮可以说:

```text
大 update 不是均匀撒在所有 code/head 上,
而是明显集中在少数 layer1 head/code bucket.
```

但不能说:

```text
code39 或 code50 本身是 bug.
```

更准确的说法是: 在这个 seed/config/trajectory 下, layer1 的少数 code/head 成为 residual update hotspot. 这提示后续可以做 code/head-aware diagnostics 或 normalization, 但需要更多 seed 证据.

## Plan 问题逐项回答

| plan 问题 | 本轮回答 | 判定 |
| --- | --- | --- |
| 大 update 是否发生在真实 training batch | 是. training inline 中 baseline max 达 `2.918/4.083`, `cap=0.5` hypothetical hit ratio 为 `21.0%/29.8%`. | 直接支持 |
| `cap=0.5` 是挡少数尖峰还是大面积改训练 | 更接近大面积介入. 中后期很多 traced top events hit ratio 达到 `1.0`. | 直接支持 |
| 问题更偏 write/update, state/read, 还是 injection | read support 分叉很早, update 大幅化随后出现, cap 不修复 read support. 当前证据支持 read-support 分叉 + residual update/state 放大, 但还不能单独归因到 injection. | 部分定位 |
| 大 update 是否 code/head 局部集中 | 是. layer1 head0/code39 和 head1/code50 占很大比例, 且跨机器轨迹不同. | 直接支持 |
| `cap=0.5` 是否可推进 | 否. 本轮 gap `28.5pp`, 且 hit pattern 轨迹依赖强. | 不通过 |

## 与前两轮的关系

最近三轮关于 `cap=0.5` 的结果:

| experiment | baseline gap | `cap=0.5` gap | 结论 |
| --- | ---: | ---: | --- |
| `20260701-04` | `43.7pp` | `2.8pp` | 当轮明显缓解 gap, 但仍是 diagnostic |
| `20260702-01` | `11.9pp` | `5.9pp` | partial mitigation, 未过 `4pp` |
| `20260702-02` | `12.2pp` | `28.5pp` | hard cap 不稳, 真实 training trace 显示 cap 是大面积且轨迹依赖的干预 |

综合判断:

```text
update 幅度仍是有实验证据支持的放大环节之一,
但 hard cap=0.5 不是稳定方案.
```

这不是推翻 `M_state update` 方向, 而是把结论推进了一步:

```text
问题不是 "随便限制一下 update 就能解决",
而是当前 residual write/state/read 机制需要更原则的阻尼和路径稳定设计.
```

## 当前已经知道什么

直接支持:

| 判断 | 证据 |
| --- | --- |
| dropout 是正常训练扰动入口 | first mismatch 在 `backbone.layers.0.dropout1` |
| cache/init/batch order 不是污染源 | 三类 hash 跨机器全部 match |
| read support 很早就分叉 | step16 top-k exact match `0.312`, step128 `0.172` |
| 大 update 真实发生在 training minibatch | inline update max 达 `4.083`, hypothetical cap hit ratio 达 `29.8%` |
| hard cap 不是 rare spike guard | 多个中后期 step hit ratio 为 `1.0` |
| update hotspots 有 layer/head/code 集中性 | code39/code50 等 bucket 占比很高 |
| `cap=0.5` 不是稳定方案 | 本轮 `ucap0p5-r2` gap `28.5pp` |

仍未证明:

| 未证明点 | 为什么 |
| --- | --- |
| 某个具体 token/code event 直接导致最终 hard gap | inline trace 记录 top events, 但不是完整因果回放 |
| read support 是唯一上游根因 | read support 早分叉, 但 gate, write, state, optimizer 也共同参与 |
| residual injection 是主因 | 本轮只有 lambda/inject scalar, 没有单独关 injection 或做 injection warmup |
| code39/code50 是通用问题 code | 当前只是一组 seed/config/trajectory 热点 |
| hard cap 只要调数值就能解决 | 三轮结果已经显示 cap=0.5 轨迹敏感 |

## 机制解释

目前最稳的解释是:

```text
default dropout 是正常训练扰动入口.
它很早改变 hidden state, 进而影响 Q/K/V, VQ routing, forget gate, beta/lambda 和 residual read/write.

read support 在早期已经发生跨机器分叉.
随后 M_state residual update 在某些 layer/head/code 上变大并集中.
这些 update 写入长期 residual memory, 后续被反复读取.

hard cap 可以改变这条链路, 但它不会修复 read support 分叉,
而且会按不同机器轨迹在不同 step/head/code 上大面积介入训练.
因此 cap=0.5 不稳.
```

这比简单说 "dropout 造成问题" 更准确. Dropout 是正常训练协议, 问题在于 Flash-VQG/GD residual 后面的 sparse read/write/state/residual injection 对这种扰动过敏.

## 下一步建议

不要继续把 `cap=0.5` 当作候选默认配置长跑. 也不要去掉 dropout, 因为 dropout 属于公平训练协议.

建议下一步围绕三个方向做最小 paired 1ep screen:

1. `residual injection / lambda warmup`.
   目标是控制 residual 读出后注入主输出的强度, 而不是在 write 端硬裁剪所有大 update. 如果它有效, 说明主要问题偏 read-out/injection 放大.

2. `soft or scheduled update control`.
   保留 "限制早期大 update" 的思想, 但不要用固定 hard cap. 更合理的是早期强一点, 后期逐步放松, 或使用平滑饱和函数, 减少轨迹依赖的硬切换.

3. `read-support stabilization`.
   read support 在 cap hit 大量出现前已经分叉, 所以 update control 只能处理下游破坏性. 仍需要继续评估 margin-aware/adaptive read, 避免 score 接近时过早硬切断 candidate.

下一轮不建议:

- 不建议直接跑 `cap=0.5` 4ep confirm.
- 不建议把 `cap=0.5` 写成最终方案.
- 不建议移除 default dropout 来得到漂亮结果.
- 不建议先做大 seed 网格, 因为当前更缺机制拆解, 不是统计重复.

更具体的下一步实验可以是:

```text
20260702-03 default-dropout residual-control 1ep screen

variants:
  baseline-r2                 当前已有, 可作为历史对照
  residual-injection-warmup   控制 O_res_added / lambda 的早期注入
  soft-update-cap             用平滑 scale 替代 hard min(1, c / norm)
  scheduled-update-control    早期限制, 后期放松

machines:
  2080ti + 3090

判定:
  hard 1024x256 高, 且 gap <= 4pp.
  同时 read support, update hit, M_state norm, inject ratio 不能出现明显单机异常.
```

如果下一轮 residual injection warmup 有效, 说明写入不一定要硬裁剪, 重点是降低 residual branch 早期影响. 如果 soft/scheduled update 有效, 说明 `M_state` 写入幅度控制方向成立, 但需要更平滑的设计. 如果两者都无效, 就应把优先级转回 read-support stabilization.
