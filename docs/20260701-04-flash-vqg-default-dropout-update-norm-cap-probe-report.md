# 20260701-04 Flash-VQG default-dropout update-norm-cap probe 报告

## 结论

本轮结论很明确: 在正常训练 dropout 打开的条件下, `fox_gd_residual_update_norm_cap=0.5` 能把 default-dropout `read_topk=2` 的跨机器 1ep hard-slice gap 从 `43.7pp` 压到 `2.8pp`, 同时保留还可用的 hard accuracy.

这说明 `M_state` residual 写入幅度是当前“微小扰动被放大成明显效果差异”的关键放大点之一. 它不是唯一因素, 也不是最终落地方案, 但已经从“合理怀疑”变成了有直接实验支撑的方向.

一句话解释:

```text
dropout 是正常训练扰动入口, 不是 bug.
问题是 gd_residual_v1 的 sparse residual write/state/read 路径对扰动太敏感.
update_norm_cap 没有让两台机器走同一条路径, 但显著降低了路径分叉后的破坏性.
```

## 实验设置

共同条件:

| 项 | 值 |
|---|---|
| branch | `flash-vqg` |
| zoology commit | `d693a0a` |
| Flash-VQG commit | `bc391c0` |
| seed | `124` |
| data seed | `123` |
| data/init | canonical MQAR cache + canonical seed124 init |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| VQ weight | `dense_softmax` |
| residual write | `fox_gd_residual_write_topk=4` |
| residual read | `fox_remote_read_topk=2` |
| train length | 1 epoch, `704` optimizer steps |
| trace steps | `0,16,64,128,256,384,512,704` |
| machines | `2080ti` + `3090`, both inside `Flash-VQG-tun` container |

Variants:

| variant | `fox_gd_residual_update_norm_cap` | 作用 |
|---|---:|---|
| `baseline-r2` | unset | default-dropout r2 失败对照 |
| `ucap0p5-r2` | `0.5` | 温和限制单步 `M_state` update |
| `ucap0p25-r2` | `0.25` | 更强限制单步 `M_state` update |

本轮是 diagnostic/probe, 不是正式 MQAR final 记录, 因此不写正式 ledger. 原始轻量 evidence 已整理到 `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/`, 3090 产物已镜像回主工作区并记录在 `source-manifest.csv`.

## Preflight

cache/init 前置检查通过:

| machine | init expected | init embedded | init actual | match |
|---|---|---|---|---|
| 2080ti | `2a1107bf...` | `2a1107bf...` | `2a1107bf...` | true |
| 3090 | `2a1107bf...` | `2a1107bf...` | `2a1107bf...` | true |

hash probe 的 first mismatch 都在:

```text
forward_before_backward_step0_micro0
backbone.layers.0.dropout1
```

这是本轮预期现象, 因为实验保留了正常训练 dropout. 本轮不是追求 bitwise 一致, 而是看 Flash-VQG 后续机制能不能承受正常训练扰动.

## 主结果

`1024x256` hard slice:

| variant | cap | 2080ti final | 3090 final | gap | within 4pp |
|---|---:|---:|---:|---:|---|
| `baseline-r2` | unset | `0.881` | `0.444` | `43.7pp` | no |
| `ucap0p5-r2` | `0.5` | `0.807` | `0.779` | `2.8pp` | yes |
| `ucap0p25-r2` | `0.25` | `0.568` | `0.554` | `1.4pp` | yes |

valid 指标:

| variant | machine | final valid acc | final valid loss | final `1024x256` |
|---|---|---:|---:|---:|
| `baseline-r2` | 2080ti | `0.977` | `0.294` | `0.881` |
| `baseline-r2` | 3090 | `0.901` | `0.846` | `0.444` |
| `ucap0p5-r2` | 2080ti | `0.962` | `0.418` | `0.807` |
| `ucap0p5-r2` | 3090 | `0.959` | `0.385` | `0.779` |
| `ucap0p25-r2` | 2080ti | `0.913` | `0.679` | `0.568` |
| `ucap0p25-r2` | 3090 | `0.917` | `0.685` | `0.554` |

解读:

- `baseline-r2` 复现了 default-dropout 下的严重跨机器失败, 不是偶然单次记录.
- `cap=0.5` 是本轮最有价值的点: 两边都还能学, 且 gap 回到 `4pp` 容忍线内.
- `cap=0.25` 更稳定, 但 hard slice 明显掉分, 说明 cap 太强会把有用的 residual memory 学习也压掉.
- 稳定但低分没有意义, 所以后续不应推进 `0.25` 作为候选主线.

## update norm 证据

step `704` 的 residual update/state 统计:

| variant | machine | loss | cap hit | update p95 | update max | M max | lambda mean | inject ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline-r2` | 2080ti | `1.188` | `0.000` | `0.063` | `0.529` | `2.633` | `0.122` | `0.327` |
| `baseline-r2` | 3090 | `3.731` | `0.000` | `0.889` | `2.106` | `7.780` | `0.092` | `0.695` |
| `ucap0p5-r2` | 2080ti | `1.687` | `0.000` | `0.073` | `0.302` | `2.212` | `0.116` | `0.779` |
| `ucap0p5-r2` | 3090 | `1.548` | `0.032` | `0.284` | `2.132` | `7.890` | `0.101` | `0.141` |
| `ucap0p25-r2` | 2080ti | `2.752` | `0.057` | `0.264` | `0.873` | `3.369` | `0.098` | `0.316` |
| `ucap0p25-r2` | 3090 | `2.838` | `0.042` | `0.208` | `5.485` | `4.168` | `0.117` | `0.379` |

这里要谨慎读:

- `cap=0.5` 的 cap hit ratio 不高, 但足以改变训练轨迹, 尤其 3090 从 hard `0.444` 恢复到 `0.779`.
- `cap=0.25` hit ratio 更高, 但性能明显被过度阻尼.
- 这说明 `M_state` update 中少数大更新或早期关键更新可能很重要, 不一定需要全局大面积裁剪才会影响结果.

## 为什么说 update 幅度是放大器

这条结论来自三组结果的组合, 不是只看单个指标.

第一, 不限制 update 时, default-dropout `r2` 出现严重跨机器分叉:

```text
baseline-r2:
2080ti = 0.881
3090   = 0.444
gap    = 43.7pp
```

这说明在 cache/init/batch 都锁住后, 正常训练 dropout 加上当前 `gd_residual_v1` 路径, 足以把两台机器推到明显不同的训练结果.

第二, 温和限制 update 到 `0.5` 后, 两边仍然能学, 但 gap 大幅缩小:

```text
ucap0p5-r2:
2080ti = 0.807
3090   = 0.779
gap    = 2.8pp
```

这直接说明“限制 `M_state` residual update 幅度”切中了某个真实放大环节. 如果 update 幅度和分叉无关, 不应该把 `43.7pp` 的 gap 压到 `2.8pp`.

第三, 继续压到 `0.25` 时, gap 更小, 但两边 hard slice 都明显变差:

```text
ucap0p25-r2:
2080ti = 0.568
3090   = 0.554
gap    = 1.4pp
```

这说明不是“update 越小越好”. residual update 本身是有用的, 过强限制会把模型能力也压掉. 因此更合理的解释是:

```text
大部分 residual update 是有用的,
但部分偏大或关键时刻的 update 会把早期扰动写进长期 M_state,
后续被 recurrent state 和 residual read 反复使用,
最终放大成明显训练效果差异.
```

这里还有一个重要边界: 本轮没有证明“两机变稳定是因为 read support 重新一致”. 后面的 read trace 显示, `cap=0.5` 时 step704 的 top-k exact match 仍约 `0.016`, 和 baseline 几乎一样低. 所以 cap 的作用更像是:

```text
不消灭路径分叉,
但降低路径分叉写入 M_state 后的破坏性.
```

因此本轮可以直接说:

```text
update_norm_cap=0.5 显著缓解 default-dropout r2 的跨机器结果差异.
```

也可以较强地说:

```text
M_state residual update 幅度是重要放大器之一.
```

但还不能写成:

```text
已经证明最终 gap 一定由少数某几个大 update 直接触发.
```

要证明最后这一点, 下一步还需要补时间序列证据: 哪些 step/token/code 触发 cap, 触发前后 `M_state` norm, residual read output, loss 和 grad 是否开始明显分叉.

## read support trace

跨机器 read support 对齐情况:

| variant | step | top1 match | topk exact match | topk overlap |
|---|---:|---:|---:|---:|
| `baseline-r2` | `0` | `1.000` | `1.000` | `1.000` |
| `baseline-r2` | `704` | `0.172` | `0.016` | `0.203` |
| `ucap0p5-r2` | `0` | `1.000` | `1.000` | `1.000` |
| `ucap0p5-r2` | `704` | `0.109` | `0.016` | `0.250` |
| `ucap0p25-r2` | `0` | `1.000` | `1.000` | `1.000` |
| `ucap0p25-r2` | `704` | `0.141` | `0.016` | `0.203` |

这个表很关键:

```text
update_norm_cap 没有让 read support 重新一致.
```

也就是说, 它并不是通过“让两台机器选同样的 code”来修复问题. 两边的 top-k read support 仍然大量不同, 但 `cap=0.5` 之后最终效果接近了.

所以更准确的结论是:

```text
read/write support 分叉仍会发生,
但不受控的 residual M_state update 会把这种分叉放大成训练效果崩坏.
限制 update 幅度可以降低分叉后的破坏性.
```

这比简单说“read support flip 是根因”更具体.

## 对当前机制的判断

本轮支持以下判断:

1. `gd_residual_v1` 的方向仍然有价值.
   no-dropout 和部分 default-dropout 配置都显示 residual memory 能带来收益, 不能简单说机制错了.

2. 当前 V1 的 residual update/state 注入缺少足够阻尼.
   default dropout 是正常训练噪声, 但它进入 Flash-VQG 后会同时影响 Q/K/V, VQ routing, forget gate, beta/lambda 和 residual read/write. 如果 `M_state` update 太自由, 这些早期扰动会被写进长期 residual memory.

3. `update_norm_cap=0.5` 是 diagnostic stabilization, 不是最终方法.
   hard cap 太机械, 梯度和优化语义也不够优雅. 它的价值是证明“限制 residual update 幅度”这条路值得继续, 后续应改成更原则的 soft cap, warmup 或 adaptive control.

4. `update_norm_cap=0.25` 过强.
   它让两机 gap 更小, 但 hard slice 掉到 `0.568/0.554`, 说明只追求稳定会牺牲模型能力.

## 直接证据和仍未证明的点

| 判断 | 证据状态 | 说明 |
|---|---|---|
| default-dropout r2 baseline 可严重跨机器分叉 | 直接支持 | 本轮 `0.881` vs `0.444`, gap `43.7pp` |
| `cap=0.5` 能显著降低 1ep gap | 直接支持 | gap `2.8pp`, 且两边 hard 都接近 `0.8` |
| `cap=0.25` 过度阻尼 | 直接支持 | gap 小, 但 hard 明显低 |
| cap 通过恢复 read support 一致性来起效 | 不支持 | step704 top-k exact match 仍约 `0.016` |
| `M_state` update 幅度是放大器之一 | 较强支持 | cap 改变 final gap, 但 support divergence 仍在 |
| `cap=0.5` 是最终落地方案 | 不支持 | 只跑 1 seed/1 epoch, 且 hard cap 仍是诊断控制 |

## 后续建议

下一步不要回到“让两台 GPU bitwise 一样”的方向. 现在更应该沿着“让 residual write/state/read 对正常扰动不那么敏感”继续.

优先建议:

1. 做 `cap=0.5` 附近的小范围 1ep paired screen.
   例如 `0.4`, `0.6`, `0.8`, 看能否在保持 gap `<=4pp` 的同时把 hard slice 从 `0.807/0.779` 往 baseline 2080ti 的 `0.881` 靠近.

2. 做 scheduled cap 或 soft cap.
   训练早期强一点, 后期逐步放松, 目标是保留早期稳定性, 减少长期性能税.

3. 再迁移到 `read_topk=4`.
   no-dropout 下 `r4` 是强配置, default dropout 下崩过. 如果 `update_norm_cap` 能稳定 r2, 下一步要看它能不能把 default-dropout r4 从失败区拉回来.

4. 只对通过 1ep screen 的候选跑 4ep confirm.
   当前 `cap=0.5` 有资格进入 4ep confirm, 但更节省算力的做法是先做一轮邻域/调度 screen, 再挑最强配置长跑.

不建议:

- 不推进 `cap=0.25` 作为候选主线, 因为效果损失太大.
- 不把 hard `update_norm_cap` 直接写成最终方法.
- 不取消正常训练 dropout. dropout 是公平训练协议的一部分, 要解决的是后续机制的抗扰动能力.

## Artifact

主要文件:

- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/run-summary.csv`
- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/variant-summary.csv`
- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/variant-gap-summary.csv`
- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/cap-metrics-summary.csv`
- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/read-trace-cross-machine-summary.csv`
- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/cache-init-preflight-summary.csv`
- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/source-manifest.csv`
- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/metadata.json`
