# 20260702-03 Flash-VQG residual injection warmup screen report

## 结论摘要

本轮在 default dropout 协议下验证了一个很小的机制干预: 不改 `M_state` build/write/read, 不改 `read_topk/write_topk`, 只把 GD residual correction 注入输出的强度做早期 warmup.

核心公式是:

```python
O_res_added = alpha_inj(t) * lambda_blk * residual_scale * O_res_norm
Out_f32 = O_base + O_res_added
```

其中 `alpha_inj(t)` 是本轮新增的 residual injection warmup factor. 这只控制 residual correction 进入输出的强度, 不会阻止 `M_state` 被写入, 也不会改变 residual read 的候选集合.

主要结果:

- `baseline-r2` 在相同 cache/init/batch 下仍然严重跨机器分叉: 2080ti `0.818`, 3090 `0.480`, gap `33.8pp`.
- `inj-warmup-linear512-r2` 明显改善: 2080ti `0.871`, 3090 `0.814`, gap `5.7pp`.
- `inj-warmup-silent64-linear512-r2` gap 更接近阈值: 2080ti `0.775`, 3090 `0.819`, gap `4.4pp`, 但 2080ti 绝对分下降.

因此, 本轮说明 residual injection 是一个真实的放大环节: 降低早期 residual 注入强度可以大幅缓解 default dropout 下的 1ep 跨机器 gap. 但两种 warmup 都没有严格通过 `<=4pp` 标准, 所以它还不是最终方案, 不应该直接推进 4ep confirm 或设为默认配置.

## 实验设置

共同条件:

| 项 | 值 |
| --- | --- |
| seed | `124` |
| data seed | `123` |
| data/init | canonical MQAR cache + canonical seed124 init |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| residual read/write | `read_topk=2`, `write_topk=4` |
| train length | 1 epoch, `704` optimizer steps |
| machines | `2080ti` + `3090`, both in `Flash-VQG-tun` container |

Variants:

| variant | residual injection warmup |
| --- | --- |
| `baseline-r2` | no warmup, factor = `1` |
| `inj-warmup-linear512-r2` | optimizer step `0 -> 512`, factor linearly `0 -> 1` |
| `inj-warmup-silent64-linear512-r2` | optimizer step `0-64`, factor `0`; optimizer step `64 -> 512`, factor linearly `0 -> 1` |

本轮使用 Flash-VQG train-forward counter. `gradient_accumulation_steps=4`, 所以 optimizer step `64` 对应 train-forward step `256`, optimizer step `512` 对应 train-forward step `2048`.

实现版本:

| repo | branch | commit |
| --- | --- | --- |
| `zoology` | `flash-vqg` | `69fdb60` |
| `Flash-VQG` | `20260428-gd-residual-v1-sync` | `a51b6b0` |

## 启动前一致性

所有 paired run 的 cache, init, batch order 都一致:

| field | all match | sha256 |
| --- | --- | --- |
| cache content | true | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| init model state | true | `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0` |
| batch order | true | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

首个 hash mismatch 均出现在:

| variant | first mismatch |
| --- | --- |
| `baseline-r2` | `forward_before_backward_step0_micro0`, `backbone.layers.0.dropout1` |
| `inj-warmup-linear512-r2` | `forward_before_backward_step0_micro0`, `backbone.layers.0.dropout1` |
| `inj-warmup-silent64-linear512-r2` | `forward_before_backward_step0_micro0`, `backbone.layers.0.dropout1` |

这说明本轮不是数据或初始化不一致. default dropout 下, `layer0.dropout1` 是正常训练扰动入口, 不是 bug. 问题是 Flash-VQG 后面的 read/write/state/residual injection 机制是否会把这个正常扰动放大成明显效果差异.

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`.

| variant | 2080ti final | 3090 final | gap | <=4pp | 判断 |
| --- | ---: | ---: | ---: | --- | --- |
| `baseline-r2` | `0.818` | `0.480` | `33.8pp` | false | default dropout 下严重不稳 |
| `inj-warmup-linear512-r2` | `0.871` | `0.814` | `5.7pp` | false | 明显改善, 但未过线 |
| `inj-warmup-silent64-linear512-r2` | `0.775` | `0.819` | `4.4pp` | false | gap 接近过线, 但 2080ti 绝对分下降 |

本轮只有 1 epoch, 所以 best 与 final 相同.

## Warmup 行为检查

`inj-warmup-linear512-r2` 的 factor 按预期从 0 线性升到 1:

| machine | step0 | step64 | step256 | step512 | step704 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2080ti factor | `0.000` | `0.125` | `0.500` | `1.000` | `1.000` |
| 3090 factor | `0.000` | `0.125` | `0.500` | `1.000` | `1.000` |
| 2080ti inject ratio | `0.000` | `0.009` | `0.061` | `0.277` | `0.210` |
| 3090 inject ratio | `0.000` | `0.009` | `0.473` | `0.582` | `0.438` |

`inj-warmup-silent64-linear512-r2` 在 step64 前保持静默, 之后升到 1:

| machine | step0 | step64 | step128 | step512 | step704 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2080ti factor | `0.000` | `0.000` | `0.143` | `1.000` | `1.000` |
| 3090 factor | `0.000` | `0.000` | `0.143` | `1.000` | `1.000` |
| 2080ti inject ratio | `0.000` | `0.000` | `0.082` | `0.677` | `0.700` |
| 3090 inject ratio | `0.000` | `0.000` | `0.041` | `0.328` | `0.287` |

这里有一个重要现象: factor 本身跨机器完全一致, 但 learned `lambda` 和实际 `inject_ratio` 后续仍会分叉. 也就是说, warmup 能降低早期 residual 注入冲击, 但不能自动约束后续 learned residual strength 的跨机器轨迹.

## Read support 对齐情况

Read support 仍然大量分叉:

| variant | step16 top1 | step16 exact | step16 overlap | step704 top1 | step704 exact | step704 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-r2` | `0.656` | `0.312` | `0.688` | `0.172` | `0.000` | `0.203` |
| `inj-warmup-linear512-r2` | `0.672` | `0.344` | `0.703` | `0.062` | `0.031` | `0.312` |
| `inj-warmup-silent64-linear512-r2` | `0.672` | `0.344` | `0.703` | `0.578` | `0.141` | `0.445` |

这说明 warmup 没有消除 read support 分叉. 但 `linear512` 在 late read support 仍然很差的情况下, final hard slice 从 baseline 3090 `0.480` 提升到 `0.814`. 因此 read support 分叉很重要, 但不是唯一充分解释. 下游 residual injection 强度本身也是效果差异的放大器.

`silent64` 的 late read support match 比 `linear512` 好, 并且 gap 更小, 但 2080ti 绝对分低到 `0.775`. 这说明简单静默前 64 step 可能减少一部分跨机器差异, 但也可能损失有用 residual 学习或让单机轨迹变弱.

## 对问题链条的判断

本轮支持以下链条:

```text
default dropout 是正常训练扰动入口
-> Flash-VQG read/write/state 轨迹发生分叉
-> residual correction 通过 lambda/residual_scale 注入输出
-> 早期注入过强时, 分叉更容易变成明显 loss/accuracy gap
```

这不是说 dropout 是 bug, 也不是说 read support 不重要. 更准确地说:

- dropout 是正常训练协议的一部分, 必须保留.
- read support/top-k 分叉仍然存在, 仍然是重要风险.
- residual injection 是一个下游放大器, 控制它可以显著缓解 gap.
- 目前只靠 injection warmup 还不够稳定, 需要进一步约束 learned `lambda`/injection 轨迹或与 read/write update 控制组合.

## 局限

- 本轮是 1 seed, 1 epoch diagnostic screen, 不是正式 MQAR ledger.
- 两个 warmup 都没有严格达到 `<=4pp`.
- 本轮没有证明 4ep 或多 seed 稳定.
- 本轮只控制输出注入强度, 没有控制 `M_state` 写入幅度, `M_state` norm, read candidate stability, 或 learned `lambda` 上界.

## 下一步建议

不要直接跑 `inj-warmup-linear512-r2` 或 `silent64` 的 4ep. 当前更合理的是先做 1ep refinement:

1. 试更平滑的 injection schedule, 例如 `inj-warmup-linear1024-r2`. 目的: 看 `linear512` 的 3090/2080ti 差异是否来自 residual 注入放开太快.
2. 试 `lambda` 或 residual injection bounded schedule. 目的: warmup factor 到 1 后, learned `lambda/inject_ratio` 仍明显分叉, 需要看 bounded injection 是否比只 warmup factor 更稳.
3. 若已有稳定的 `M_state` update soft control, 再试 `linear512 + soft update control`. 目的: 同时降低写入尖峰和输出注入放大, 但避免 hard cap 大面积改变训练.

判定仍然保持简单: default dropout, same cache/init/batch, 1ep final hard slice 高, 且 2080ti/3090 gap `<=4pp`. 只有满足这个条件的候选才值得进入 4ep confirm.

## Artifact

主要 artifact 位于:

```text
docs/artifacts/20260702-03-flash-vqg-injection-warmup-screen/
```

关键文件:

- `variant-gap-summary.csv`
- `run-summary.csv`
- `injection-warmup-summary.csv`
- `read-trace-cross-machine-summary.csv`
- `hash-probe-comparison-summary.csv`
- `preflight-effective-summary.csv`
- `source-manifest.csv`
- `metadata.json`
