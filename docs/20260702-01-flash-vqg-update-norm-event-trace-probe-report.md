# 20260702-01 Flash-VQG update-norm event trace probe 报告

status: completed_diagnostic
ledger: not written

## 结论摘要

本轮完成了 `M_state` residual update event trace probe. 结论比上一轮更谨慎:

```text
update_norm_cap=0.5 仍然是有效的稳定化方向,
但这轮没有通过 4pp 跨机器容忍线,
所以不能把 cap=0.5 直接推进成 4ep 候选或最终方案.
```

主结果是:

| variant | 2080ti 1024x256 | 3090 1024x256 | gap | within 4pp |
| --- | ---: | ---: | ---: | --- |
| baseline-r2 | 0.596 | 0.477 | 11.9pp | False |
| ucap0p5-r2 | 0.695 | 0.754 | 5.9pp | False |

相比 baseline, `cap=0.5` 同时提高了两台机器的 hard slice, 并把 gap 从 `11.9pp` 缩到 `5.9pp`. 这支持 "M_state residual update 幅度是放大器之一". 但 `5.9pp` 仍超过用户可接受的 `4pp` 线, 所以它只是 partial mitigation.

一句话:

```text
限制 M_state 单次 residual update 幅度确实有帮助,
但单个 hard cap 不够稳, 也不是最终设计.
下一步应该做更原则的 soft/scheduled update control, 而不是直接长跑 cap=0.5.
```

## 实验设置

共同条件:

| 项 | 值 |
| --- | --- |
| experiment id | `20260702-01-flash-vqg-update-norm-event-trace-probe` |
| zoology commit | `65510b1` |
| Flash-VQG commit | `e376f4d` |
| seed | `124` |
| data seed | `123` |
| data/init | canonical MQAR cache + canonical seed124 init |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| residual read/write | `read_topk=2`, `write_topk=4` |
| train length | 1 epoch, `704` optimizer steps |
| trace steps | `0,1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,704` |
| machines | `2080ti` + `3090`, both inside `Flash-VQG-tun` |

Variants:

| variant | `fox_gd_residual_update_norm_cap` | 作用 |
| --- | ---: | --- |
| `baseline-r2` | unset | default-dropout r2 baseline |
| `ucap0p5-r2` | `0.5` | 限制单次 `M_state` residual update 幅度 |

本轮是 diagnostic/probe, 不写 official MQAR ledger. Artifact 目录:

```text
docs/artifacts/20260702-01-flash-vqg-update-norm-event-trace-probe/
```

## Trace 口径

`update_event_trace.jsonl` 不是直接记录实际 training minibatch 的每一次写入. 它是在指定 optimizer step 上, 对 fixed validation batch 做 eval forward, 记录当时模型状态下最大的 residual update event.

因此它适合回答:

```text
在同一训练进度下, 当前模型会产生多大的 residual update?
cap 会命中哪些大 update?
两台机器在 update event 分布上是否已经明显不同?
```

它不能直接证明:

```text
最终 gap 一定由某一个具体 training minibatch 的某一个 event 触发.
```

另外, event trace 中的 `update_norm_uncapped` / `update_norm_max` 是 cap 前的候选 update norm. `actual_cap_hit` 和 `actual_cap_scale` 才表示实际是否被 cap 缩放.

## 前置一致性

cache/init/batch order 在两台机器上均一致:

| target | cache match | init match | batch order match |
| --- | --- | --- | --- |
| baseline-r2 | True | True | True |
| ucap0p5-r2 | True | True | True |

固定 hash:

- MQAR cache content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.

所以本轮不是数据, 初始权重, 或 batch order 不一致导致的对比污染.

## First Mismatch

| target | first mismatch stage | step | micro | field | module |
| --- | --- | ---: | ---: | --- | --- |
| baseline-r2 | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |
| ucap0p5-r2 | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |

这符合预期. 本轮保留正常训练 dropout, 训练态两台机器不追求 bitwise 一致. 重点不是消灭 dropout, 而是看 Flash-VQG/GD residual 后续机制能不能承受正常训练扰动.

## 主结果

| variant | machine | duration min | final valid acc | final 1024x256 | final valid loss |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline-r2 | 2080ti | 140 | 0.930 | 0.596 | 0.655 |
| baseline-r2 | 3090 | 95 | 0.908 | 0.477 | 0.806 |
| ucap0p5-r2 | 2080ti | 185 | 0.941 | 0.695 | 0.529 |
| ucap0p5-r2 | 3090 | 125 | 0.955 | 0.754 | 0.429 |

`cap=0.5` 的效果:

- 2080ti hard slice: `0.596 -> 0.695`, 提升 `9.9pp`.
- 3090 hard slice: `0.477 -> 0.754`, 提升 `27.7pp`.
- 跨机器 gap: `11.9pp -> 5.9pp`, 明显缩小, 但仍未进入 `4pp` 容忍线.

与上一轮 `20260701-04` 的关系:

- 上一轮 `cap=0.5` 曾达到 `0.807/0.779`, gap `2.8pp`.
- 本轮方向一致, 但没有复现 `<=4pp`.
- 因此更稳妥的判断是: `cap=0.5` 是强诊断信号, 但还不是稳定配置.

## Update Norm 证据

step `704` 的 scalar 指标:

| variant | machine | cap | cap hit | update p95 | update max | M max | lambda mean | inject ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-r2 | 2080ti |  | 0.000 | 0.600 | 1.680 | 6.370 | 0.086 | 0.734 |
| baseline-r2 | 3090 |  | 0.000 | 0.742 | 2.249 | 9.936 | 0.095 | 0.680 |
| ucap0p5-r2 | 2080ti | 0.5 | 0.000 | 0.090 | 0.465 | 3.603 | 0.090 | 0.550 |
| ucap0p5-r2 | 3090 | 0.5 | 0.036 | 0.245 | 4.173 | 4.755 | 0.091 | 0.084 |

读法:

1. baseline 在 step704 的 update norm 明显超过 `0.5`, 且 3090 的 `M max` 更高.
2. `cap=0.5` 后, 两台机器的 hard slice 都提高, `M max` 也低于 baseline.
3. 3090 在 `cap=0.5` 下仍有很大的 uncapped candidate update, 但 actual cap 会缩放它, 例如 top event 的 `actual_cap_scale=0.120`.
4. 2080ti 在 final trace 上没有 cap hit, 说明 cap 主要改变的是训练历史轨迹, 不一定在 final snapshot 仍持续大量命中.

## Event Trace

全 trace 聚合:

| target | machine | records | update max | update mean | update p95 | actual cap hit | hypothetical cap hit | actual scale mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-r2 | 2080ti | 1088 | 2.821 | 0.510 | 2.305 | 0.000 | 0.206 | 1.000 |
| baseline-r2 | 3090 | 1088 | 2.249 | 0.406 | 1.752 | 0.000 | 0.153 | 1.000 |
| ucap0p5-r2 | 2080ti | 1088 | 2.313 | 0.357 | 1.846 | 0.063 | 0.063 | 0.956 |
| ucap0p5-r2 | 3090 | 1088 | 4.173 | 0.547 | 2.679 | 0.192 | 0.192 | 0.887 |

top event:

| target | machine | step | uncapped update | actual cap hit | actual scale | err norm | zeta before | zeta after | code | token |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-r2 | 2080ti | 384 | 2.821 | False | 1.000 | 3.052 | 0.924 | 0.924 | 39 | 31 |
| baseline-r2 | 3090 | 704 | 2.249 | False | 1.000 | 5.149 | 0.437 | 0.437 | 14 | 763 |
| ucap0p5-r2 | 2080ti | 384 | 2.313 | True | 0.216 | 2.597 | 0.891 | 0.193 | 39 | 7 |
| ucap0p5-r2 | 3090 | 704 | 4.173 | True | 0.120 | 6.780 | 0.616 | 0.074 | 39 | 271 |

这张表说明 cap 实际做的事情是:

```text
当候选 update norm 过大时, 缩小 zeta, 从而缩小写入 M_state 的 delta_M.
```

例如 3090 `ucap0p5-r2` 的最大 event 中, uncapped update 为 `4.173`, cap scale 为 `0.120`, `zeta` 从 `0.616` 缩到 `0.074`. 这正是预期的阻尼行为.

## Read Support

跨机器 read support 对齐:

| target | step | top1 match | top-k exact match | top-k overlap |
| --- | ---: | ---: | ---: | ---: |
| baseline-r2 | 0 | 1.000 | 1.000 | 1.000 |
| baseline-r2 | 1 | 0.906 | 0.875 | 0.969 |
| baseline-r2 | 16 | 0.656 | 0.312 | 0.688 |
| baseline-r2 | 128 | 0.531 | 0.172 | 0.438 |
| baseline-r2 | 704 | 0.156 | 0.000 | 0.203 |
| ucap0p5-r2 | 0 | 1.000 | 1.000 | 1.000 |
| ucap0p5-r2 | 1 | 0.906 | 0.875 | 0.969 |
| ucap0p5-r2 | 16 | 0.656 | 0.312 | 0.688 |
| ucap0p5-r2 | 128 | 0.531 | 0.172 | 0.438 |
| ucap0p5-r2 | 704 | 0.047 | 0.000 | 0.078 |

这里有一个重要结论:

```text
update_norm_cap 没有让 read support 重新一致.
```

`cap=0.5` 下 step704 的 top-k exact match 仍为 `0.000`, top1 match 甚至只有 `0.047`. 所以 cap 的作用不是把两台机器拉回同一条离散 read path, 而是降低不同 path 写入和读取 residual memory 后的破坏性.

## 这轮说明了什么

直接支持:

| 判断 | 证据 |
| --- | --- |
| normal dropout 下 first mismatch 仍在 layer0 dropout | 两个 target 都是 `backbone.layers.0.dropout1` |
| read support 仍会快速跨机器分叉 | step16 top-k exact match `0.312`, step704 `0.000` |
| `cap=0.5` 有稳定化价值 | hard gap `11.9pp -> 5.9pp`, 两边 hard 都提高 |
| `cap=0.5` 不是充分解法 | gap 仍为 `5.9pp`, 超过 4pp |
| cap 不通过恢复 read support 起效 | `ucap0p5-r2` step704 top-k exact match `0.000` |
| 大 update 是放大器之一 | top event 会被显著缩放, 分数和 gap 同时改善 |

仍未证明:

| 判断 | 状态 |
| --- | --- |
| 最终 gap 一定由某几个具体 training events 触发 | 未直接证明, 当前 trace 是 validation-batch snapshot |
| hard cap 是最终落地方案 | 不支持 |
| 只靠 update cap 就能解决 default dropout 稳定性 | 不支持 |
| read support flip 不重要 | 不支持, 它仍然明显存在 |

## 与上一轮结果的关系

`20260701-04` 已经证明 `update_norm_cap` 是强方向: `cap=0.5` 曾把 gap 从 `43.7pp` 降到 `2.8pp`. 本轮复跑加入了更细 event trace 后, baseline gap 为 `11.9pp`, cap gap 为 `5.9pp`.

这说明两点:

1. `cap=0.5` 的方向不是偶然无效, 因为本轮仍然提高两机 hard slice 并缩小 gap.
2. `cap=0.5` 的稳定性还不够, 因为本轮没有通过 4pp.

所以当前不能说:

```text
cap=0.5 已经解决 default-dropout r2.
```

只能说:

```text
限制 M_state update 幅度是有效但不充分的稳定化手段.
```

## 下一步建议

不要马上跑 `cap=0.5` 4ep. 当前 1ep 还没有稳定通过 4pp, 直接长跑会浪费算力.

更合理的下一步是小范围 1ep paired screen, 目标是找到 "既保留 hard slice, 又稳定过 4pp" 的控制方式:

1. `soft update cap`.
   用连续缩放代替硬截断, 例如让过大 update 逐渐饱和, 减少 hard cap 的非光滑性.

2. `scheduled cap`.
   早期限制强一些, 后期逐步放松. 这更符合当前现象: 训练早期扰动进入 M_state 后会被长期放大, 但后期 residual memory 仍需要表达能力.

3. `cap + residual injection warmup`.
   cap 管写入幅度, warmup 管 residual branch 注入输出的强度. 如果只管写入还不够稳, 需要同时降低早期读出的破坏性.

4. `cap value neighborhood`.
   不要大网格, 只试 `0.4/0.6/0.8` 或对应 soft/scheduled 版本. 判定标准必须同时满足:
   - 1ep `1024x256` hard slice 高.
   - 2080ti/3090 gap `<=4pp`.
   - read support 可以分叉, 但 final loss/hard 不崩.

当前优先级:

```text
P1: soft/scheduled update control
P2: residual injection warmup
P3: 再迁移到 read_topk=4
P4: 只有通过 1ep paired screen 的候选才跑 4ep confirm
```

## Artifact

核心文件:

- `run-summary.csv`: 每条 run 的 final/best 指标.
- `variant-gap-summary.csv`: 两机 hard slice gap.
- `cap-metrics-summary.csv`: step704 cap/update/M/lambda/inject 指标.
- `update-event-step-summary.csv`: 每个 trace step 的 event 聚合.
- `update-event-trace-summary.csv`: 每个 variant/machine 的 event 总聚合.
- `update-event-cross-machine-summary.csv`: event 聚合的跨机器比较.
- `update-event-top.csv`: top event 明细.
- `read-trace-cross-machine-summary.csv`: read support 跨机器 match.
- `hash-probe-comparison-summary.csv`: train-mode hash probe.
- `source-manifest.csv`: 回收的轻量 raw evidence.

