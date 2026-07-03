# 20260703-02 Flash-VQG injection warmup reproducibility rerun report

## 结论摘要

本轮做两件事:

1. 把 `read_trace_train_steps` 改成显式 diagnostic 开关, 默认关闭.
2. 在 no-trace 模式下精确重跑上一轮接近可用的两个 residual injection warmup variant.

本轮正式 run 均确认:

```text
read_trace_enabled = false
read_trace_train_steps = []
```

也就是说, 训练期间没有插入 read-trace eval snapshot, 训练后也没有跑 hash-probe. 这次结果仍然没有稳定复现上一轮的接近过线信号:

| variant | 2080ti final 1024x256 | 3090 final 1024x256 | gap | <=4pp |
| --- | ---: | ---: | ---: | --- |
| `inj-warmup-linear512-r2` | `0.846` | `0.771` | `7.5pp` | false |
| `inj-warmup-silent64-linear512-r2` | `0.816` | `0.748` | `6.8pp` | false |

所以本轮结论是:

```text
residual injection warmup 是有效的放大器控制点,
但 linear512 / silent64-linear512 单独作为 default-dropout 稳定方案不够稳,
不应该进入 4ep confirm, 也不应该被当成最终配置.
```

更直白地说: warmup 能帮忙, 但不够硬. 当前问题仍然需要更直接地控制 `M_state` 写入尖峰, learned `lambda`, 或最终 residual injection 的实际幅度.

## 为什么做这轮

`20260702-03` 里, default dropout 下两个 injection warmup variant 看起来接近可用:

| previous variant | 2080ti | 3090 | gap |
| --- | ---: | ---: | ---: |
| `inj-warmup-linear512-r2` | `0.871` | `0.814` | `5.7pp` |
| `inj-warmup-silent64-linear512-r2` | `0.775` | `0.819` | `4.4pp` |

但当时训练中启用了 `read_trace_train_steps`, 训练后还跑了 hash-probe. 这些定位流程理论上不应该作为正式训练路径的一部分. 因此本轮先把 read trace 做成显式开关, 默认关闭, 再无 trace 重跑这两个配置, 看上一轮信号是否稳.

## 实验设置

共同条件:

| 项 | 值 |
| --- | --- |
| experiment id | `20260703-02-flash-vqg-injection-warmup-repro-rerun` |
| seed | `124` |
| data seed | `123` |
| model | `cb64-r16` |
| train length | 1 epoch, `704` optimizer steps |
| gradient accumulation | `4` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| residual read/write | `read_topk=2`, `write_topk=4` |
| machines | `2080ti` + `3090`, both in `Flash-VQG-tun` container |
| read trace | disabled |
| hash-probe | disabled |

实现版本:

| repo | branch | commit |
| --- | --- | --- |
| `zoology` | `flash-vqg` | `3ba3e6e` |
| `Flash-VQG` | `20260428-gd-residual-v1-sync` | `a51b6b0` |

Variants:

| variant | residual injection warmup |
| --- | --- |
| `inj-warmup-linear512-r2` | optimizer step `0 -> 512`, factor `0 -> 1` |
| `inj-warmup-silent64-linear512-r2` | optimizer step `0-64`, factor `0`; optimizer step `64 -> 512`, factor `0 -> 1` |

这里的 optimizer step 会在 Flash-VQG 内部转换成 train-forward step. 本轮 `gradient_accumulation_steps=4`, 所以 optimizer step `512` 对应 train-forward step `2048`.

## 启动前一致性

所有正式 run 都通过启动前检查:

| field | all match | sha256 |
| --- | --- | --- |
| cache content | true | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| init model state | true | `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0` |
| batch order | true | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

首 16 个 batch index 为:

```text
[487, 968, 1599, 205, 1685, 1567, 1443, 72, 1336, 2219, 756, 2572, 1778, 2588, 2430, 2464]
```

这说明本轮 paired comparison 不是由 cache, init, batch order 不一致造成的.

## Trace 开关检查

本轮已把 read trace 变成默认关闭的定位流程:

| 模式 | 启动方式 | 行为 |
| --- | --- | --- |
| default | 不传开关 | `read_trace_enabled=false`, `read_trace_train_steps=[]` |
| diagnostic | `ENABLE_READ_TRACE=1` 或 `--enable-read-trace` | 恢复旧 17 个 train-step read trace snapshot |

本次四个正式 run 的配置均为:

| variant | machine | read_trace_enabled | read_trace_train_steps | hash_probe_steps | event trace |
| --- | --- | --- | --- | --- | --- |
| `inj-warmup-linear512-r2` | 2080ti | false | `[]` | null | null |
| `inj-warmup-linear512-r2` | 3090 | false | `[]` | null | null |
| `inj-warmup-silent64-linear512-r2` | 2080ti | false | `[]` | null | null |
| `inj-warmup-silent64-linear512-r2` | 3090 | false | `[]` | null | null |

之前 timestamp `20260703T022900Z` 启动过一批 trace-on run, 已在发现后中止, 并写入 abort marker. 这些 run 不进入本轮结果解释.

## 主结果

主指标仍然是 `valid/mqar_case/accuracy-1024x256`.

| variant | machine | final 1024x256 | valid accuracy | valid loss | lambda mean | inject ratio | update norm max | M norm max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inj-warmup-linear512-r2` | 2080ti | `0.846` | `0.972` | `0.307` | `0.407` | `0.165` | `4.56` | `13.2` |
| `inj-warmup-linear512-r2` | 3090 | `0.771` | `0.959` | `0.385` | `0.725` | `0.357` | `14.6` | `29.3` |
| `inj-warmup-silent64-linear512-r2` | 2080ti | `0.816` | `0.967` | `0.336` | `0.715` | `0.384` | `0.708` | `5.07` |
| `inj-warmup-silent64-linear512-r2` | 3090 | `0.748` | `0.952` | `0.405` | `0.247` | `0.155` | `2.86` | `5.76` |

Paired gap:

| variant | 2080ti | 3090 | gap | 判断 |
| --- | ---: | ---: | ---: | --- |
| `inj-warmup-linear512-r2` | `0.846` | `0.771` | `7.5pp` | 未过线 |
| `inj-warmup-silent64-linear512-r2` | `0.816` | `0.748` | `6.8pp` | 未过线 |

本轮 `results/*.json` 中 `train_result` 为空, 因此以上 final metric 是从每个 log 的 final validation progress line 解析得到. 对应 log hash 已写入 artifact 的 `run-summary.csv` 和 `source-manifest.csv`.

## 与上一轮对比

| variant | previous 2080ti | previous 3090 | previous gap | current 2080ti | current 3090 | current gap | gap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inj-warmup-linear512-r2` | `0.871` | `0.814` | `5.7pp` | `0.846` | `0.771` | `7.5pp` | `+1.8pp` |
| `inj-warmup-silent64-linear512-r2` | `0.775` | `0.819` | `4.4pp` | `0.816` | `0.748` | `6.8pp` | `+2.4pp` |

这个对比要谨慎解释. 本轮不能证明 read trace 导致上一轮结果更好或更差, 因为两轮训练本身已经是随机 dropout 下的独立运行轨迹. 但它可以说明一件事:

```text
去掉 read trace 和 hash-probe 之后,
linear512 / silent64-linear512 没有稳定复现到 <=4pp.
```

所以, 上一轮 injection warmup 的方向仍然有价值, 但 single-factor warmup 本身不够可靠.

## 这说明了什么

本轮不是推翻 residual injection warmup. 它更像是把 warmup 的位置放准了:

- warmup 确实能控制 residual correction 进入输出的时间曲线.
- 但它不控制 `M_state` 怎么写, 不控制 learned `lambda` 怎么走, 不控制 read/write support 分叉, 也不限制最终 `O_res_added` 的实际大小.
- 因此它只能缓解一部分 early amplification, 不能保证 default dropout 下跨机器稳定.

尤其是 `linear512` 这次很直观:

| machine | lambda mean | inject ratio | update norm max | M norm max | final 1024x256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2080ti | `0.407` | `0.165` | `4.56` | `13.2` | `0.846` |
| 3090 | `0.725` | `0.357` | `14.6` | `29.3` | `0.771` |

两边 nominal warmup schedule 一样, 但 learned residual strength 和 state/update 指标走到了明显不同轨迹. 这支持之前的判断: 只控制时间开关不够, 后面还需要控制实际写入或实际注入强度.

## 对后续实验的影响

本轮之后不建议做:

- 不建议把 `linear512` 或 `silent64-linear512` 直接推进 4ep.
- 不建议继续只拉长或改 warmup 曲线.
- 不建议把 read trace 打开后得到的训练结果当作正式候选结果.

更合理的下一步是 1ep paired screen, 直接测试更强的稳定器:

1. `M_state update_norm_cap + injection warmup`: 同时限制写入尖峰和输出注入.
2. `lambda/inject soft cap`: 限制 learned `lambda` 或最终 residual 注入比例, 而不是只控制时间 factor.
3. `residual_scale=0.5`: 低成本检查整体 residual branch 缩放是否比复杂 warmup 更稳.

判定标准仍然应该保持严格:

```text
default dropout
same cache/init/batch
1ep final 1024x256 hard slice 高
paired gap <= 4pp
```

低分但稳定没有意义, 单机高分但跨机器 gap 大也不能进入 4ep confirm.

## Artifact

主要 artifact 位于:

```text
docs/artifacts/20260703-02-flash-vqg-injection-warmup-repro-rerun/
```

关键文件:

- `run-summary.csv`
- `cross-machine-comparison.csv`
- `preflight-summary.csv`
- `prelaunch-consistency-summary.csv`
- `trace-mode-summary.csv`
- `previous-comparison.csv`
- `source-manifest.csv`
- `aborted-runs.csv`
- `metadata.json`

大型 raw 输出, logs, configs, result JSON 保留在 ignored `zoology/experiments/flash_vqg/scripts/20260703-02-flash-vqg-injection-warmup-repro-rerun/outputs/` 下. 3090 轻量 raw evidence 已镜像回当前主工作区并在 `source-manifest.csv` 中记录 sha256.
