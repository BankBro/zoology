# 20260701-03 Flash-VQG default-dropout 1ep bridge trace 报告

status: completed_diagnostic
ledger: not written

## 结论摘要

本轮完成了 `128 -> 704` optimizer step 的桥接实验. 6 条 run 全部完成, 共同使用相同 MQAR cache, 相同 canonical init, 相同 batch order. 结论很明确:

```text
default dropout 下, 当前 gd_residual_v1 的跨机器训练轨迹仍然严重不稳定.
这不是 read_topk=4 独有问题.
把 embed_dropout 从 0.1 降到 0.05 也不能当作稳定方案.
```

主指标 `valid/mqar_case/accuracy-1024x256`:

| target | embed dropout | read topk | 2080ti final | 3090 final | gap | 4pp 内 |
|---|---:|---:|---:|---:|---:|---|
| `default-r2` | 0.1 | 2 | 0.869 | 0.445 | 42.4pp | False |
| `default-r4` | 0.1 | 4 | 0.818 | 0.128 | 69.0pp | False |
| `dropout005-r4` | 0.05 | 4 | 0.738 | 0.859 | 12.1pp | False |

这里最重要的新信息有三点:

1. `default-r2` 也失败, 所以问题不能简化成 "`read_topk=4` 单独坏".
2. `default-r4` 仍然最差, 说明较大的 residual read support 在 default dropout 下确实更危险.
3. `dropout005-r4` 这次跨机器 gap 仍有 12.1pp, 且方向反过来是 3090 更好. 所以 `embed_dropout=0.05` 只能算扰动强度对照, 不能说它稳定.

本轮更像证明了:

```text
正常训练 dropout 引起的早期路径差异,
会在 gd_residual_v1 的 read support, residual state, injection 和训练 loss 轨迹中持续放大.
```

但还不能正面证明某一个单独指标, 比如 read top-k flip, M_state norm, 或 inject ratio, 就是充分根因.

## 实验口径

- experiment id: `20260701-03-flash-vqg-default-dropout-1ep-bridge-trace`.
- zoology: branch `flash-vqg`, commit `2a71c31`.
- Flash-VQG: branch `20260428-gd-residual-v1-sync`, commit `bc391c0`.
- common config: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, `vq_weight_mode=dense_softmax`.
- dropout: `embed_dropout` 按 target 设置, `resid_dropout=0.0`, `drop_path=0.0`.
- max train steps: `704`, 即 1 epoch optimizer steps.
- trace steps: `0,16,64,128,256,384,512,704`.
- machines: `2080ti` + `3090`, 均在 `Flash-VQG-tun` 容器内运行.
- 本轮是 diagnostic/probe, 不是 official MQAR 正式实验, 不写 ledger.

Variants:

| target | 配置 | 作用 |
|---|---|---|
| `default-r2` | `embed_dropout=0.1`, `read_topk=2` | 判断 default dropout 下 r2 是否跨机器稳定 |
| `default-r4` | `embed_dropout=0.1`, `read_topk=4` | default dropout 下已知高风险 r4 路径 |
| `dropout005-r4` | `embed_dropout=0.05`, `read_topk=4` | 扰动强度边界对照, 不作为最终协议 |

## 前置一致性

三组 target 的 cache, init, batch order 都跨机器一致:

| target | cache match | init match | batch order match |
|---|---|---|---|
| `default-r2` | True | True | True |
| `default-r4` | True | True | True |
| `dropout005-r4` | True | True | True |

固定 hash:

- MQAR cache content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init model state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.

所以这轮不能再把差异归因到数据 cache, 初始权重, 或 batch 顺序不一致.

## First Mismatch

训练态 hash probe 里, 三组 target 的第一处跨机器 mismatch 都一样:

| target | first mismatch stage | step | micro | field | module |
|---|---|---:|---:|---|---|
| `default-r2` | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |
| `default-r4` | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |
| `dropout005-r4` | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |

这不表示 dropout 配置错误. 这是正常训练协议: 训练时 dropout 开启, 评估时 `model.eval()` 关闭 dropout. 这里的意义是, 跨机器路径差异在进入第一层 Flash-VQG mixer 之前就已经出现.

## Read Support

固定 valid batch 的 read support 对比显示, step0 两机完全一致, 之后快速分叉. 到 step704 时, 三组 target 的 read support 基本都已经高度不一致:

| target | step | top1 match | top-k exact match | top-k overlap |
|---|---:|---:|---:|---:|
| `default-r2` | 0 | 100.0% | 100.0% | 100.0% |
| `default-r2` | 128 | 53.1% | 17.2% | 43.8% |
| `default-r2` | 704 | 3.1% | 0.0% | 10.9% |
| `default-r4` | 0 | 100.0% | 100.0% | 100.0% |
| `default-r4` | 128 | 57.8% | 0.0% | 48.0% |
| `default-r4` | 704 | 3.1% | 0.0% | 20.3% |
| `dropout005-r4` | 0 | 100.0% | 100.0% | 100.0% |
| `dropout005-r4` | 128 | 53.1% | 1.6% | 51.2% |
| `dropout005-r4` | 704 | 6.2% | 0.0% | 14.5% |

这直接支持 "read support 是重要放大器信号". 但是它仍然不是充分解释. 原因是 `dropout005-r4` 的 read support 也严重分叉, 但 final 不是双边都崩, 而是 3090 0.859, 2080ti 0.738. 也就是说, support 分叉说明轨迹已经不同, 但最终学得好坏还取决于后续 residual state, injection, optimizer trajectory 等因素.

## Bridge Scalar

step128 时, 三组 target 的 loss 基本仍然都在 8.46 附近. 这说明 128-step probe 只能证明路径分叉, 还没有抓到最终 metric collapse.

到 step512/704, loss 轨迹已经明显分开:

| target | machine | step128 loss | step512 loss | step704 loss | final 1024x256 |
|---|---|---:|---:|---:|---:|
| `default-r2` | 2080ti | 8.461 | 4.468 | 1.283 | 0.869 |
| `default-r2` | 3090 | 8.460 | 7.821 | 3.753 | 0.445 |
| `default-r4` | 2080ti | 8.461 | 8.374 | 1.754 | 0.818 |
| `default-r4` | 3090 | 8.460 | 8.375 | 6.398 | 0.128 |
| `dropout005-r4` | 2080ti | 8.463 | 2.520 | 1.910 | 0.738 |
| `dropout005-r4` | 3090 | 8.465 | 2.791 | 1.325 | 0.859 |

这张表回答了本轮的核心问题: 128 step 时还看不出最终分数差异, 但 512/704 之间 loss 已经和 final hard slice 方向基本对上. 尤其:

- `default-r2`: 2080ti 在 step512 已经明显学起来, 3090 仍然高 loss.
- `default-r4`: 两机到 step512 仍都高 loss, 但 step704 2080ti 突然学起来, 3090 仍然很差.
- `dropout005-r4`: 两机都学起来, 但收敛速度和最终 hard slice 仍不一致.

M/update/inject 指标能说明 residual state 轨迹确实不同, 但不能单独解释最终分数:

| target | machine | step704 lambda | step704 inject | step704 M max | step704 update max |
|---|---|---:|---:|---:|---:|
| `default-r2` | 2080ti | 0.248 | 0.830 | 5.708 | 0.564 |
| `default-r2` | 3090 | 0.111 | 0.734 | 8.683 | 1.928 |
| `default-r4` | 2080ti | 0.443 | 0.276 | 11.718 | 3.578 |
| `default-r4` | 3090 | 0.197 | 0.344 | 21.503 | 4.978 |
| `dropout005-r4` | 2080ti | 0.098 | 0.148 | 12.741 | 3.586 |
| `dropout005-r4` | 3090 | 0.241 | 0.552 | 1.630 | 0.167 |

直接读法:

1. `default-r2/default-r4` 里, 3090 更差, 且 step704 的 M max/update max 更大.
2. `dropout005-r4` 里, 2080ti 更差, 但它的 M max/update max 反而更大.
3. 因此 M_state magnitude 或 update_norm 不是单独充分根因. 更准确的说法是: residual state 和 injection 轨迹已经分叉, 它们和 read support, optimizer trajectory 共同决定最后落在哪条训练轨迹上.

## 和上一轮结论的关系

上一轮 `20260701-02` 的 128-step trace 说的是:

```text
step128 已经出现 read support / M_state 相关分叉,
但 loss 还没明显分开.
```

本轮补上的信息是:

```text
到 512/704 step, loss 和 final hard slice 开始显著分开.
```

所以当前因果链更完整:

```text
train-mode dropout 在 layer0 dropout1 产生早期路径差异
-> read support 很快跨机器分叉
-> residual state / injection / optimizer trajectory 继续走向不同路径
-> 512/704 step loss 分开
-> 1ep hard slice 出现大 gap
```

但是最后两步之间还没有精确定位到单个充分原因. 现在能排除的比能肯定的更多:

- 不是 cache/init/batch 不一致.
- 不是评估时误开 dropout.
- 不是 `read_topk=4` 独有, 因为 `default-r2` 也失败.
- 不是简单把 `embed_dropout` 改成 0.05 就能解决, 因为本轮 `dropout005-r4` 仍有 12.1pp gap.
- 不是关掉 residual 的方向, 因为前一轮 residual-zero 基本学不动 hard slice.

## 当前结论边界

本轮能直接支持:

- default dropout 训练协议下, 当前 gd_residual_v1 跨机器 1ep 不稳定.
- `read_topk=2` 和 `read_topk=4` 都不能在 default dropout 下通过 4pp 稳定线.
- `read_topk=4` 比 `read_topk=2` 更危险, 但不是唯一问题.
- `embed_dropout=0.05` 不是可宣布的稳定方案.
- 128-step 之前主要是路径分叉, 512/704 才更清楚地表现为 loss/metric 分叉.

本轮不能直接支持:

- 不能说 dropout 应该从正式训练协议里拿掉.
- 不能说 read top-k flip 是唯一根因.
- 不能说 M_state norm 或 update norm 单独就是根因.
- 不能说 3090 一定更差. `dropout005-r4` 中 3090 更好, 说明这是轨迹敏感性, 不是固定机器强弱排序.

## 下一步建议

现在不建议继续重复同口径 diagnostic. 已经有足够证据进入最小稳定化 probe, 但必须仍在正常训练协议 `embed_dropout=0.1` 下做.

建议下一轮只做 2 到 3 个最小干预, 每个仍跑 1ep paired:

| probe | 目的 | 判定 |
|---|---|---|
| residual injection warmup | 训练前期先降低 residual branch 对输出的影响, 看能否避免 512/704 分叉 | hard slice 高, 且 gap <= 4pp 才算有效 |
| beta/lambda cap 或 warmup | 限制 residual 写入/注入强度突然变大 | 如果 loss 分叉推迟或消失, 说明注入强度是关键放大器 |
| M_state update norm cap | 限制一次 residual 写入对 M_state 的冲击 | 如果 r2/r4 都更稳, 说明 state update 是主要放大点 |

不建议下一步做:

- 不要把 `dropout=0.05` 当成正式方案.
- 不要再只跑 `read_topk` 网格.
- 不要直接跑 4ep confirm, 因为 1ep 已经大幅失败.
- 不要改评估 dropout, 因为评估本来就关闭 dropout.

下一轮的成功标准必须同时满足:

```text
1ep 1024x256 hard slice 高.
2080ti/3090 gap <= 4pp.
不是靠关掉 residual 或降低正式 dropout 协议实现.
```

Artifact 目录: `docs/artifacts/20260701-03-flash-vqg-default-dropout-1ep-bridge-trace/`.
