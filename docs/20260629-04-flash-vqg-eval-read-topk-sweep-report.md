# 20260629-04 Flash-VQG Eval Read-TopK Sweep Report

## 目的

本实验不重新训练模型, 只加载 `20260629-03` dense-read 4 epoch 实验留下的 8 个 checkpoint, 在评估阶段覆盖 `fox_remote_read_topk`, 检查 read top-k 对最终 MQAR validation 指标的影响。

核心问题是: 已训练好的 dense-read checkpoint 在 eval 时是否必须使用 dense read, 以及不同机器 eval 是否会给出不同结论。

## 实验设置

- checkpoint: 4 个 run x `best,last` = 8 个 checkpoint.
- checkpoint 来源: `2080ti-r1`, `2080ti-r2`, `3090-r1`, `3090-r2`.
- eval machine: `2080ti`, `3090`.
- eval read_topk: `1,2,4,8,16,32,64`.
- 总有效记录: 8 个 checkpoint/kind x 2 台 eval machine x 7 个 topk = 112.
- 评估数据: 使用同一份 canonical MQAR cache. 本轮启动前做过跨机器 cache 内容 hash 校验, 13/13 match.
- 训练代码 commit: `708180d Add eval read-topk sweep tooling`.

本实验只读 checkpoint 和 validation cache, 不训练, 不保存新 checkpoint.

cache preflight 的轻量汇总保存在 `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/cache-hash-summary.csv`. 13 个 cache 文件的 file sha256 与 tensor content sha256 均为 13/13 match.

## 运行状态

四组 eval 全部完成, failed=0:

- `2080ti-eval-2080ti-source`: 28/28.
- `2080ti-eval-3090-mirror`: 28/28.
- `3090-eval-3090-source`: 28/28.
- `3090-eval-2080ti-mirror`: 28/28.

注意: 3090 source eval 早期曾误启动过一个重复后台进程, 导致 raw JSONL 为 39 行。最终 collector 按 `(checkpoint_id, checkpoint_kind, eval_read_topk, eval_machine)` 去重, 有效记录为 28 条。`metadata.json` 中保留了 `raw_records=123`, `total_records=112` 以便审计。

## 总体结果

下面按 eval read_topk 汇总 16 条 eval 记录, 即 8 个 checkpoint/kind 组合 x 2 台 eval machine. 注意这 16 条不是 16 个独立训练 run; 两台 eval machine 是用来验证同一 checkpoint 的 eval runtime 一致性.

`delta vs topk64 mean` 的口径是: 同一 checkpoint, 同一 checkpoint kind, 同一 eval machine 下, 先计算 `topk - topk64`, 再对这些差值求平均.

| eval read_topk | n | hard 1024x256 mean | min | max | delta vs topk64 mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.888998 | 0.866508 | 0.909191 | -0.028783 |
| 2 | 16 | 0.917787 | 0.897074 | 0.935711 | +0.000006 |
| 4 | 16 | 0.922252 | 0.901383 | 0.940055 | +0.004471 |
| 8 | 16 | 0.919947 | 0.898055 | 0.938570 | +0.002166 |
| 16 | 16 | 0.918023 | 0.896156 | 0.937102 | +0.000242 |
| 32 | 16 | 0.917768 | 0.895910 | 0.936945 | -0.000014 |
| 64 | 16 | 0.917781 | 0.895926 | 0.936984 | +0.000000 |

在本轮 checkpoint 和 canonical cache 上, 结论一致:

- `topk=1` 明显不够, 平均比 `topk=64` 低 2.878pp.
- `topk=2` 基本等于 `topk=64`.
- `topk=4` 最好, 平均比 `topk=64` 高 0.447pp.
- `topk=8` 也有小幅收益, 但弱于 `topk=4`.
- `topk=16/32/64` 基本持平.

`topk=4` 的胜出不是单个 checkpoint 拉高均值. 在 16 个 checkpoint/kind/eval-machine 组合中, `topk=4` 全部优于 `topk=64`; 单条 `topk=4 - topk64` 的 hard accuracy 提升范围是 +0.307pp 到 +0.607pp. 相对每个组合里的次优 topk, `topk=4` 的 margin 范围是 +0.148pp 到 +0.335pp.

## 辅助指标

下面补充 overall valid accuracy, loss 和 selected mass. selected mass 随 topk 增加而增加, 但 hard accuracy 和 loss 在 `topk=4` 最好, 说明读到更多 residual mass 不等价于更好的预测。

| eval read_topk | n | valid accuracy mean | hard 1024x256 mean | valid loss mean | selected mass mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.980827 | 0.888998 | 0.175589 | 0.173698 |
| 2 | 16 | 0.985994 | 0.917787 | 0.141790 | 0.250511 |
| 4 | 16 | 0.986636 | 0.922252 | 0.137573 | 0.318467 |
| 8 | 16 | 0.986261 | 0.919947 | 0.140887 | 0.358869 |
| 16 | 16 | 0.985970 | 0.918023 | 0.143328 | 0.393043 |
| 32 | 16 | 0.985931 | 0.917768 | 0.143784 | 0.415020 |
| 64 | 16 | 0.985932 | 0.917781 | 0.143808 | 0.423879 |

## 分机器结果

| eval machine | topk | n | hard mean | min | max |
|---|---:|---:|---:|---:|---:|
| 2080ti | 1 | 8 | 0.888998 | 0.866508 | 0.909191 |
| 2080ti | 2 | 8 | 0.917787 | 0.897074 | 0.935711 |
| 2080ti | 4 | 8 | 0.922251 | 0.901383 | 0.940051 |
| 2080ti | 8 | 8 | 0.919947 | 0.898055 | 0.938570 |
| 2080ti | 16 | 8 | 0.918023 | 0.896156 | 0.937102 |
| 2080ti | 32 | 8 | 0.917768 | 0.895910 | 0.936945 |
| 2080ti | 64 | 8 | 0.917781 | 0.895926 | 0.936984 |
| 3090 | 1 | 8 | 0.888998 | 0.866508 | 0.909191 |
| 3090 | 2 | 8 | 0.917787 | 0.897074 | 0.935711 |
| 3090 | 4 | 8 | 0.922252 | 0.901383 | 0.940055 |
| 3090 | 8 | 8 | 0.919947 | 0.898055 | 0.938570 |
| 3090 | 16 | 8 | 0.918023 | 0.896156 | 0.937102 |
| 3090 | 32 | 8 | 0.917768 | 0.895910 | 0.936945 |
| 3090 | 64 | 8 | 0.917781 | 0.895926 | 0.936984 |

两台机器的趋势完全一致。`topk=4` 在 2080ti 和 3090 上都最好。

## 单个 checkpoint 对比

| checkpoint | kind | source | 2080ti topk4 | 2080ti topk64 | 3090 topk4 | 3090 topk64 |
|---|---|---|---:|---:|---:|---:|
| 2080ti-r1 | best | 2080ti | 0.920750 | 0.915859 | 0.920750 | 0.915859 |
| 2080ti-r1 | last | 2080ti | 0.920750 | 0.915859 | 0.920750 | 0.915859 |
| 2080ti-r2 | best | 2080ti | 0.940051 | 0.936984 | 0.940055 | 0.936984 |
| 2080ti-r2 | last | 2080ti | 0.940051 | 0.936984 | 0.940055 | 0.936984 |
| 3090-r1 | best | 3090 | 0.921883 | 0.917941 | 0.921883 | 0.917941 |
| 3090-r1 | last | 3090 | 0.901383 | 0.895926 | 0.901383 | 0.895926 |
| 3090-r2 | best | 3090 | 0.926266 | 0.921883 | 0.926266 | 0.921883 |
| 3090-r2 | last | 3090 | 0.906879 | 0.900813 | 0.906879 | 0.900813 |

所有 checkpoint/kind/eval machine 组合中, `topk=4` 都是本轮 sweep 的最佳 eval read_topk。

额外观察:

- 2080ti 的两个 run 中, `best` 与 `last` 指标相同, 说明 final 没有相对 best 掉分.
- 3090 的两个 run 中, `last` 明显低于 `best`, 约低 2.1-2.2pp. 这个现象和之前看到的 final/best gap 一致.

## 跨机器 eval 一致性

同一个 checkpoint, 同一个 eval read_topk, 分别在 2080ti 和 3090 上评估, `1024x256` 差值如下:

| topk | n | mean delta 3090-2080ti | max abs delta |
|---:|---:|---:|---:|
| 1 | 8 | +0.00000000 | 0.00000000 |
| 2 | 8 | +0.00000000 | 0.00000000 |
| 4 | 8 | +0.00000098 | 0.00000391 |
| 8 | 8 | +0.00000000 | 0.00000000 |
| 16 | 8 | +0.00000000 | 0.00000000 |
| 32 | 8 | +0.00000000 | 0.00000000 |
| 64 | 8 | +0.00000000 | 0.00000000 |

这说明 eval 本身几乎不是跨机器差异来源。给定同一个 checkpoint 和同一份 cache, 2080ti 与 3090 的 validation 结果基本一致。

因此, 当前主要差异更可能来自训练过程形成的 checkpoint 质量差异, 而不是 eval runtime 把同一个 checkpoint 评估出明显不同的 accuracy。

## 解释

这次结果说明, dense-read 训练出来的模型在 eval 时并不一定需要 dense read。

`topk=1` 太窄, 会漏掉必要 code, 所以明显掉分。`topk=2` 已经接近 dense read。`topk=4` 反而更好, 可能是因为它保留了足够候选, 同时过滤掉一部分低质量 residual contribution。`topk=64` 读全 code, 从 `valid_read_selected_mass_mean` 看 selected mass 更高, 但不一定带来更高 accuracy。

这个结果不直接证明训练时也应该用 `read_topk=4`; 它只证明 eval/read 阶段存在一个可测的候选宽度效应。训练阶段的最优 read/write 支持还需要单独实验。

因此, 报告中的 `topk=4` 应被理解为本轮 dense-read checkpoint 的 eval-time 最优点, 不是训练阶段默认配置, 也不是所有 seed/长度/容量下的通用结论。

## artifact

- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/eval-summary.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/topk-vs-64.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/cross-machine-eval-comparison.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/aggregate-by-topk.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/aggregate-by-topk-extended.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/topk4-win-margins.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/cache-hash-summary.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/checkpoint-manifest.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/source-manifest.csv`
- `docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/metadata.json`
