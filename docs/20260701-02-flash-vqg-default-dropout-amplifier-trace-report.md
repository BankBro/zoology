# 20260701-02 Flash-VQG default-dropout amplifier trace 报告

status: completed_diagnostic
ledger: not written

## 结论摘要

这轮实验完成了 `default-dropout amplifier trace` 的定位任务: 在相同 MQAR cache, 相同 canonical init, 相同 batch order 下, 训练态跨机器 first mismatch 对三个 target 都发生在 `optimizer_step=0`, `micro_step=0`, `backbone.layers.0.dropout1` 的 `module_output_sha256`.

这件事的含义很具体: 在 zoology 正常训练协议里, `embed_dropout=0.1` 位于第一层 Flash-VQG mixer 之前. 训练时 dropout 正常开启, 两台 GPU 的 train-mode dropout/RNG 轨迹不要求 bitwise 相同, 所以第一层入口前就会产生路径差异. 这不是 bug, 也不是说应该把 dropout 去掉. 评估时 `model.eval()` 会关闭 dropout.

这轮更重要的新信号是: 这个早期 dropout 路径差异进入 `gd_residual_v1` 后, read support 在 128 step 内快速跨机器分叉. 但它没有直接证明 "read support 分叉本身足以导致 1ep 失败". 原因是 `dropout005-r4` 在 step128 同样有明显 read support 分叉, 但上一轮 1ep paired screen 仍然进入 4pp 容忍线. 所以下一步不能只盯 `read_topk`, 还要看 default dropout 强度下 residual injection, M_state/update 和写入轨迹如何在 128 step 之后继续放大。

## 实验口径

- experiment id: `20260701-02-flash-vqg-default-dropout-amplifier-trace`.
- zoology: branch `flash-vqg`, run-start commit `2ddbe23`.
- Flash-VQG: branch `20260428-gd-residual-v1-sync`, commit `bc391c0`.
- common config: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, `vq_weight_mode=dense_softmax`, `resid_dropout=0.0`, `drop_path=0.0`.
- max train steps: `128`.
- trace steps: `0,1,4,16,64,128`.
- machines: `2080ti` + `3090`, 均在 `Flash-VQG-tun` 容器内运行.
- 本轮是 diagnostic/probe, 不是 official MQAR 正式实验, 不写 ledger.

Variants:

| target | embed_dropout | read_topk | 作用 |
|---|---:|---:|---|
| `default-r4` | 0.1 | 4 | default dropout 下已知失败链路的主定位对象 |
| `dropout005-r4` | 0.05 | 4 | 上一轮 1ep 过线的扰动强度边界对照 |
| `default-r2` | 0.1 | 2 | 判断 default dropout 问题是否只和 `read_topk=4` 有关 |

## 执行状态

6 条有效运行全部完成: 3 个 target x 2 台机器. 部分后台 wrapper 在训练和 hash probe 成功后没有把 `queue-status.tsv` 写成 `completed`, 因此本轮判断有效完成状态时以 `execution-status-summary.csv` 为准: 它要求同时存在 `result.json`, 日志 `[done] target=... train_status=0 hash_status=0`, 以及 `hash_probe.json`.

`run-summary.csv` 不是本轮主判定表. 一方面这轮只跑 128 optimizer steps, 不是 1ep; 另一方面部分 wrapper status 没有写 completed, base collector 的 `run-summary.csv` 不能代表所有有效运行. 本轮主判定表是:

- `execution-status-summary.csv`.
- `preflight-effective-summary.csv`.
- `first-mismatch-summary.csv`.
- `read-trace-cross-machine-summary.csv`.
- `early-window-summary.csv`.
- `hash-probe-comparison-summary.csv`.

## 前置一致性

三组 target 的 cache/init/batch order 都是跨机器一致:

| target | cache match | init match | batch order match |
|---|---|---|---|
| `default-r2` | True | True | True |
| `default-r4` | True | True | True |
| `dropout005-r4` | True | True | True |

固定 hash:

- MQAR cache content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init model state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.

因此本轮不是数据不一致, 初始权重不一致, 或 batch order 不一致造成的对比污染。

## First Mismatch

| target | first mismatch stage | step | micro | field | module |
|---|---|---:|---:|---|---|
| `default-r2` | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |
| `default-r4` | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |
| `dropout005-r4` | `forward_before_backward_step0_micro0` | 0 | 0 | `module_output_sha256` | `backbone.layers.0.dropout1` |

这说明在正常 train-mode dropout 下, 第一处可观测跨机器分叉发生在 Flash-VQG mixer 之前. 因为 `layer0.dropout1 = embed_dropout`, 它会直接改变随后进入 Flash-VQG 的 hidden state. 这个 hidden state 再同时影响 Q/K/V, VQ routing, fox gate/logf, beta/lambda, G/L coarse state, M_state residual write/read 等路径。

## Read Support Trace

这里的比较口径是同一个 target 的 `2080ti` vs `3090`, 不是 r2 和 r4 互相比.

- `top1 match`: 两台机器选出的第 1 个 read code 是否一致.
- `top-k exact match`: 两台机器选出的整个 top-k code 集合是否完全一致.
- `top-k overlap`: 两台机器 top-k code 集合的平均重叠比例.

| target | step | top1 match | top-k exact match | top-k overlap |
|---|---:|---:|---:|---:|
| `default-r2` | 16 | 65.6% | 31.2% | 68.8% |
| `default-r2` | 128 | 53.1% | 17.2% | 43.8% |
| `default-r4` | 16 | 65.6% | 3.1% | 75.8% |
| `default-r4` | 128 | 57.8% | 0.0% | 48.0% |
| `dropout005-r4` | 16 | 76.6% | 9.4% | 78.1% |
| `dropout005-r4` | 128 | 53.1% | 1.6% | 51.2% |

直接读法:

1. 三个 target 从 step16 开始就已经明显 read support 分叉, 到 step128 更严重.
2. `default-r2` 也分叉, 所以 default dropout 问题不是 `read_topk=4` 独有.
3. `default-r4` 的 exact-set match 更低, 但这是 larger k 的集合完全一致判定更苛刻; 它的 overlap 在 step16/64 并不比 r2 更差. 因此不能只用 exact-set match 得出 "`r4` 一定更坏" 的结论.
4. `dropout005-r4` 在 step128 仍然明显分叉, 但上一轮 1ep paired 通过 4pp 线. 所以 read support 分叉是放大器证据, 但还不是 1ep accuracy gap 的充分解释。

## Early Scalar Signals

step128 的 loss 在两台机器和三个 target 之间仍然很接近:

| target | 2080ti loss | 3090 loss |
|---|---:|---:|
| `default-r2` | 8.461 | 8.460 |
| `default-r4` | 8.461 | 8.460 |
| `dropout005-r4` | 8.463 | 8.465 |

step128 的 residual/read-write 指标:

| target | machine | retention | churn | top1 flip | margin | entropy | selected mass | lambda | inject | update max | M max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default-r2` | 2080ti | 0.244 | 0.756 | 0.889 | 1.079 | 2.281 | 0.416 | 0.003 | 0.034 | 0.488 | 0.971 |
| `default-r2` | 3090 | 0.387 | 0.613 | 0.759 | 2.945 | 1.784 | 0.512 | 0.003 | 0.025 | 0.366 | 0.832 |
| `default-r4` | 2080ti | 0.364 | 0.636 | 0.850 | 0.817 | 2.588 | 0.470 | 0.003 | 0.035 | 0.476 | 0.879 |
| `default-r4` | 3090 | 0.470 | 0.530 | 0.785 | 2.876 | 1.771 | 0.569 | 0.003 | 0.028 | 0.341 | 0.734 |
| `dropout005-r4` | 2080ti | 0.261 | 0.739 | 0.959 | 2.749 | 1.176 | 0.684 | 0.003 | 0.046 | 0.619 | 0.803 |
| `dropout005-r4` | 3090 | 0.432 | 0.568 | 0.786 | 0.853 | 2.660 | 0.452 | 0.003 | 0.040 | 0.704 | 0.781 |

这张表的重点不是判断哪个 target 最好, 而是说明: 到 128 step 时, read support 和 residual state 指标已经明显不同, 但 loss 还没有明显分开. 因此早期 128-step trace 能证明"路径已经分叉", 不能单独证明"失败已经发生". 真正需要定位的是 128 step 之后, 这些 state/read/write 差异什么时候变成 1ep hard slice 崩溃。

## 和最近几轮实验合并判读

已有 1ep 证据:

| 实验 | 配置 | 2080ti 1024x256 | 3090 1024x256 | gap | 判定 |
|---|---|---:|---:|---:|---|
| `20260630-04` | `embed_dropout=0.1`, `read_topk=4` | 0.284 | 0.135 | 14.9pp | fail |
| `20260701-01` | `embed_dropout=0.1`, `read_topk=2` | 0.857 | 0.462 | 39.5pp | fail |
| `20260701-01` | `embed_dropout=0.05`, `read_topk=4` | 0.872 | 0.841 | 3.1pp | pass screen |
| `20260701-01` | `embed_dropout=0.1`, `read_topk=4`, residual zero | 0.010 | 0.096 | 8.61pp | fail, 仅诊断 |

合并后的结论是:

1. no-dropout 下 `fixed-r4` 强, default dropout 下 `fixed-r4` 1ep 崩, 所以问题和正常训练 dropout 引入的早期扰动有关.
2. default dropout 下 `fixed-r2` 跨机器也失败, 所以不能把问题简化成 "`read_topk=4` 单独有问题".
3. `dropout=0.05 fixed-r4` 通过 1ep paired, 说明扰动强度会影响当前 gd_residual_v1 是否进入不稳定区间. 但 `dropout=0.05` 不是最终方案, 因为公平训练协议仍应回到 `embed_dropout=0.1`.
4. residual zero 学不动 hard slice, 说明 M/residual branch 是能力来源之一. 解决方向不能是关掉 residual, 而是稳定 residual read/write/state 的早期训练过程.
5. 这轮 128-step trace 进一步确认: 正常 train dropout 造成 first mismatch 后, read support 和 M_state 相关指标会快速分叉. 但 128 step loss 尚未明显分开, 所以下一步要桥接短 trace 和 1ep metric collapse。

## 当前结论边界

本轮能支持:

- train-mode first mismatch 在 `layer0 dropout1`.
- cache/init/batch order 已排除为原因.
- default dropout 下 read support 很早就跨机器分叉.
- r2 和 r4 都会分叉, 所以不是 r4 独有.
- read support 分叉是重要放大器信号, 但不是唯一充分解释.

本轮不能支持:

- 不能说 dropout 是错误配置. 它是正常训练协议.
- 不能说应该把 `embed_dropout` 从 0.1 改成 0.05 作为最终方案.
- 不能说 `read_topk=4` 本身坏. no-dropout 下它是强配置.
- 不能说 `read_topk=2` 是最终方案. default dropout r2 paired 1ep 已经失败.
- 不能说关掉 residual 是方案. residual zero hard slice 基本学不动.

## 下一步建议

下一轮不要马上跑 4ep, 也不要直接做大网格. 应该先补一轮 "128 step 到 1ep 之间的桥接 trace":

1. 仍保持正常训练协议 `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
2. 选择两个主 target: `default-r2` 和 `default-r4`. 可选保留 `dropout005-r4` 作为边界对照.
3. 跑到 1ep, 但在 optimizer step `128,256,384,512,704` 做稀疏 trace, 同时保留 final hard slice.
4. 必须记录 read support, write support, M_state norm/update norm, residual injection ratio, lambda/beta, loss, grad/model/optimizer hash.
5. 目标是找出: 128 step 时 loss 还接近, 但 1ep hard slice 已经崩, 中间是哪一类指标先出现异常增益.

只有在这轮桥接 trace 找到明确转折点后, 再进入最小稳定化 probe:

- residual injection warmup 或 cap.
- beta/lambda warmup 或 cap.
- M_state update norm cap.
- read/write support margin guard.

这些稳定化 probe 必须在 default `embed_dropout=0.1` 下验证. `dropout=0.05` 只能作为扰动边界诊断, 不能替代正常训练协议。

Artifact 目录: `docs/artifacts/20260701-02-flash-vqg-default-dropout-amplifier-trace/`.

