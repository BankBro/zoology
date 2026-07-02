# 20260703-01 Flash-VQG injection warmup refinement screen report

## 结论摘要

本轮继续在 default dropout 协议下测试 residual injection warmup, 但只做 refinement, 不改变 `M_state` build/write/read, 不改变 `read_topk/write_topk`, 不改变 dropout 训练协议.

控制的仍然只是 residual correction 加到 base output 上的强度:

```python
O_res_added = alpha_inj(t) * lambda_blk * residual_scale * O_res_norm
Out_f32 = O_base + O_res_added
```

最终结果比较明确:

- 三个 variant 都完成了 2080ti + 3090 paired 1ep.
- cache, init, batch order 都完全一致.
- 三个 variant 都没有达到 `gap <= 4pp`.
- 最好的 `inj-warmup-linear704-r2` 是 2080ti `0.921`, 3090 `0.798`, gap `12.3pp`.
- 更慢的 `linear1024` 和更短静默的 `silent32-linear704` 都没有改善, 反而更差.

所以本轮结论是:

```text
residual injection warmup 仍然是有意义的放大器控制点,
但单纯把 warmup 拉长, 或简单静默一小段, 不能稳定 default-dropout r2.
```

这轮不应该进入 4ep confirm. 下一步应转向更直接的 bounded residual injection, 例如 `lambda/inject soft cap`, 或把 injection 控制和 `M_state update_norm_cap` 组合起来验证.

## 实验设置

共同条件:

| 项 | 值 |
| --- | --- |
| experiment id | `20260703-01-flash-vqg-injection-warmup-refinement-screen` |
| seed | `124` |
| data seed | `123` |
| model | `cb64-r16` |
| train length | 1 epoch, `704` optimizer steps |
| gradient accumulation | `4` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| residual read/write | `read_topk=2`, `write_topk=4` |
| machines | `2080ti` + `3090`, both in `Flash-VQG-tun` container |

Warmup counter 说明:

```text
Flash-VQG 内部使用 train-forward counter.
本轮 gradient_accumulation_steps=4.
所以 optimizer step 704 对应 train-forward step 2816.
optimizer step 1024 对应 train-forward step 4096.
```

实现版本:

| repo | branch | commit |
| --- | --- | --- |
| `zoology` | `flash-vqg` | `c3122f7` |
| `Flash-VQG` | `20260428-gd-residual-v1-sync` | `a51b6b0` |

## Variants

| variant | optimizer warmup | train-forward warmup | 目的 |
| --- | --- | --- | --- |
| `inj-warmup-linear704-r2` | `0 -> 704` | `0 -> 2816` | 1ep 结束刚好完全放开, 比上一轮 `linear512` 慢 |
| `inj-warmup-linear1024-r2` | `0 -> 1024` | `0 -> 4096` | 1ep 结束 factor 约 `0.687`, 测持续低注入 |
| `inj-warmup-silent32-linear704-r2` | `32 -> 704` | `128 -> 2816` | 前 32 optimizer step 静默, 再线性放开到 1ep 结束 |

本轮没有 baseline, 因为上一轮 `20260702-03` 已经给出同条件 baseline:

| previous variant | 2080ti | 3090 | gap |
| --- | ---: | ---: | ---: |
| `baseline-r2` | `0.818` | `0.480` | `33.8pp` |
| `inj-warmup-linear512-r2` | `0.871` | `0.814` | `5.7pp` |
| `inj-warmup-silent64-linear512-r2` | `0.775` | `0.819` | `4.4pp` |

## 启动前一致性

所有 paired run 的 cache, init, batch order 都 match:

| field | all match | sha256 |
| --- | --- | --- |
| cache content | true | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| init model state | true | `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0` |
| batch order | true | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

首个 hash mismatch 均出现在:

| variant | first mismatch |
| --- | --- |
| `inj-warmup-linear704-r2` | `forward_before_backward_step0_micro0`, `backbone.layers.0.dropout1` |
| `inj-warmup-linear1024-r2` | `forward_before_backward_step0_micro0`, `backbone.layers.0.dropout1` |
| `inj-warmup-silent32-linear704-r2` | `forward_before_backward_step0_micro0`, `backbone.layers.0.dropout1` |

这符合 default dropout 训练协议. 它说明本轮不是 cache/init/batch 不一致导致的差异. `dropout1` 是正常训练扰动入口, 不是 bug. 问题仍然是 Flash-VQG 后续机制是否会把正常扰动放大.

## 主结果

主指标是 `valid/mqar_case/accuracy-1024x256`.

| variant | 2080ti final | 3090 final | gap | <=4pp | 判断 |
| --- | ---: | ---: | ---: | --- | --- |
| `inj-warmup-linear704-r2` | `0.921` | `0.798` | `12.3pp` | false | 单机分数高, 跨机器 gap 大 |
| `inj-warmup-linear1024-r2` | `0.863` | `0.666` | `19.7pp` | false | 过慢释放, 3090 明显低 |
| `inj-warmup-silent32-linear704-r2` | `0.909` | `0.757` | `15.2pp` | false | 静默 32 step 不够, gap 仍大 |

本轮 1ep 中 best 与 final 相同.

与上一轮相比, `linear704` 的 2080ti 绝对分比 `linear512` 更高, 但 3090 低于 `linear512`, 所以 gap 变大. `linear1024` 说明持续压低 residual 注入并不自动更稳, 反而可能让学习变弱. `silent32-linear704` 也没有复现上一轮 `silent64-linear512` 接近过线的效果.

## Warmup 行为检查

Warmup factor 按预期生效. 但是 factor 一致不代表实际注入强度一致, 因为实际注入还受 learned `lambda` 和 residual output norm 影响.

Step704 摘要:

| variant | factor | 2080ti inject ratio | 3090 inject ratio | 2080ti lambda | 3090 lambda |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linear704` | `1.000` | `0.212` | `0.331` | `0.373` | `0.567` |
| `linear1024` | `0.687` | `0.344` | `0.256` | `0.715` | `0.348` |
| `silent32-linear704` | `1.000` | `0.508` | `0.419` | `0.729` | `0.679` |

几个关键观察:

- `linear704` 虽然 factor 到 1, 但 3090 的 `lambda` 和 inject ratio 都明显高于 2080ti, 同时 final hard slice 更低.
- `linear1024` 在 1ep 结束时 factor 只有约 `0.687`, 但它没有带来更小 gap, 说明简单持续压低注入不是充分方案.
- `silent32-linear704` 到后期 inject ratio 很高, 但两边仍然明显分叉.

所以更准确的说法是:

```text
warmup factor 可以控制名义注入开关,
但不能约束 learned lambda / 实际 inject_ratio 的跨机器轨迹.
```

## Read Support

Read support 仍然大量分叉. 这里的 exact match 指两台机器选出的 `read_topk=2` code 集合完全一致.

| variant | step16 top1 | step16 exact | step16 overlap | step128 top1 | step128 exact | step128 overlap | step704 top1 | step704 exact | step704 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear704` | `0.672` | `0.344` | `0.703` | `0.422` | `0.047` | `0.328` | `0.031` | `0.000` | `0.047` |
| `linear1024` | `0.672` | `0.344` | `0.703` | `0.422` | `0.047` | `0.328` | `0.547` | `0.125` | `0.375` |
| `silent32-linear704` | `0.672` | `0.344` | `0.703` | `0.422` | `0.047` | `0.344` | `0.047` | `0.016` | `0.203` |

这里有两个信息:

1. 三个 variant 在早期 read support 基本一样, 说明 injection warmup 没有修复早期 read support 分叉.
2. `linear1024` 的 step704 read support 比另外两个更好, 但 final hard slice 更差. 这说明 late read support 对齐不是最终效果的唯一解释, residual 注入强度和训练轨迹本身也很关键.

## 本轮说明了什么

本轮不是证明 residual injection warmup 没价值. 上一轮已经证明它可以把 `baseline-r2` 的 gap 从 `33.8pp` 降到 `5.7pp/4.4pp`. 本轮更具体地说明:

```text
只调 warmup 时间曲线不够.
```

原因是:

- default dropout 下, 首个可观测分叉仍然从 `layer0.dropout1` 开始.
- read support 分叉仍然存在.
- `M_state` 仍照常 build/write/read.
- learned `lambda` 和实际 `inject_ratio` 后续仍会跨机器走到不同轨迹.
- 只把 `alpha_inj(t)` 放慢, 不能限制这些后续放大器.

所以当前链条更像:

```text
normal dropout perturbation
-> read/write/state trajectory divergence
-> learned lambda / inject_ratio diverges
-> residual correction injection amplifies the trajectory gap
-> hard-slice accuracy diverges
```

`alpha_inj(t)` 可以降低一部分早期冲击, 但它不是足够强的稳定化方案.

## 对上一轮结果的修正

上一轮 `inj-warmup-linear512-r2` 和 `silent64-linear512-r2` 看起来接近可用, 但本轮没有复现“更慢或更温和就更好”的趋势. 因此不能把上一轮理解成:

```text
只要 warmup 更慢, default dropout 就会稳定.
```

更合理的解释是:

```text
residual injection 是真实放大器,
但单独 warmup 的效果对 schedule 和训练轨迹敏感.
要成为方案, 需要直接控制实际注入强度或和 state/update 控制组合.
```

## 局限

- 本轮是 1 seed, 1 epoch diagnostic screen, 不是 official MQAR ledger.
- 本轮没有跑 4ep, 因为 1ep 没有候选通过 `gap <= 4pp`.
- 本轮只控制 `O_res_added` 的 warmup factor, 没有限制 `lambda`, `inject_ratio`, `M_state update_norm`, `M_state norm`, read candidate margin.
- hash-probe 会额外重放训练步骤, 所以 wall time 比单纯 1ep train 更长.

## 下一步建议

不建议继续沿着“单纯拉长 warmup”跑大实验. 更建议做 1ep paired screen, 目标是直接控制实际 residual 注入强度, 而不是只控制时间因子.

优先级:

1. `lambda/inject soft cap`: 限制 `lambda_blk * residual_scale * alpha_inj` 或最终 `O_res_added / O_base` 的有效幅度. 目的: 验证实际注入强度分叉是不是主要放大器.
2. `linear512 + update_norm_cap`: 组合上一轮接近有效的 injection warmup 和已有的 `M_state update_norm_cap`. 目的: 同时控制写入尖峰和输出注入.
3. `residual_scale=0.5` 或等效静态缩放: 作为低成本 sanity check. 目的: 看减少 residual branch 全局强度是否比复杂 warmup 更稳.

判定仍然保持:

```text
default dropout
same cache/init/batch
1ep final 1024x256 hard slice on both machines >= 0.82
paired gap <= 4pp
```

只有满足这个条件的候选才值得进入 4ep confirm.

## Artifact

主要 artifact 位于:

```text
docs/artifacts/20260703-01-flash-vqg-injection-warmup-refinement-screen/
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
