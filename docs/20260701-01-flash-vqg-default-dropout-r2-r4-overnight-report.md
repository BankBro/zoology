# 20260701-01 Flash-VQG default-dropout r2/r4 overnight diagnostic 报告

## 结论摘要

这轮实验没有支持马上跑 `default-dropout fixed-r2/fixed-r4` 的 4ep confirm. P0 判定失败: 在相同 MQAR cache, 相同 canonical init, 相同 seed=124/data_seed=123 下, `default-dropout fixed-r2` 的 1ep hard slice 为 2080ti `0.857` vs 3090 `0.462`, gap `39.5pp`, 明显超过 4pp 容忍线.

最有用的新信号是 `embed_dropout=0.05, read_topk=4` 的 1ep paired 结果: 2080ti `0.872`, 3090 `0.841`, gap `3.1pp`, 进入 4pp 容忍线. 这说明问题不应该再简单说成 "fixed-r4 有问题" 或 "dropout 一开就不行"; 更准确是 default `embed_dropout=0.1` 的早期扰动强度会把当前 gd_residual_v1 的 read/write/state 路径推入不稳定区间, 降低到 `0.05` 后这个区间明显缓和.

`residual_norm_mode=zero` 不是解决方案. 它在 1ep paired 中 hard slice 为 2080ti `0.010`, 3090 `0.0961`; 关掉 residual contribution 后模型基本学不动 hard slice. 这说明 M/residual branch 是能力来源之一, 不能通过直接禁用来换稳定性.

## 实验条件

- zoology: branch `flash-vqg`, commit `6eee4c8`.
- Flash-VQG: branch `20260428-gd-residual-v1-sync`, commit `bc391c0`.
- common config: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, `resid_dropout=0.0`, `drop_path=0.0`.
- cache: 13 个 MQAR cache 内容 hash 均为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- init: canonical init state hash 为 `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- 本轮是 diagnostic/exploratory screen, 不写 official MQAR ledger.

## 关键结果

| 问题 | 配置 | 2080ti 1024x256 | 3090 1024x256 | gap | 判定 |
|---|---|---:|---:|---:|---|
| P0: default fixed-r2 是否可复现 | `embed_dropout=0.1`, `read_topk=2` | 0.857 | 0.462 | 39.5pp | fail, 不跑 4ep |
| 降低 dropout 是否恢复稳定 | `embed_dropout=0.05`, `read_topk=4` | 0.872 | 0.841 | 3.1pp | pass screen |
| 关闭 residual 是否解决 | `embed_dropout=0.1`, `read_topk=4`, residual zero | 0.010 | 0.0961 | 8.61pp | fail, 仅诊断 |

注意: `variant-summary.csv` 会按 variant 聚合所有队列, 其中 `fixed-r2` 同时包含 128-step probe 和 1ep run, 因此主判定不要直接读它. 主判定表是 `decision-summary.csv`.

## Early Probe 信号

128-step read-support probe 显示, 即使 cache/init 一致, early read candidate support 也会快速跨机器分叉:

| target | step | top1 match | top-k exact match | top-k overlap |
|---|---:|---:|---:|---:|
| fixed-r2 | 16 | 65.6% | 31.3% | 68.8% |
| fixed-r2 | 64 | 51.6% | 12.5% | 53.1% |
| fixed-r2 | 128 | 53.1% | 17.2% | 43.8% |
| fixed-r4 | 16 | 65.6% | 3.1% | 75.8% |
| fixed-r4 | 64 | 50.0% | 1.6% | 62.1% |
| fixed-r4 | 128 | 57.8% | 0.0% | 48.0% |

这支持 "read top-k 是放大器" 这个判断, 但它不是唯一充分解释. 因为 `dropout=0.05 fixed-r4` 在 step128 也有明显分叉, top1 match `53.1%`, top-k exact match `1.6%`, top-k overlap `51.2%`, 但 1ep paired 仍然进入 4pp 容忍线. 所以下一步不应只盯 read_topk, 还要围绕 early dropout perturbation strength, residual injection strength, M_state 写入强度和训练早期 schedule 做拆解.

## 对当前问题的解释

当前最合理的解释更新为:

1. `embed_dropout=0.1` 在 layer0 进入 Flash-VQG 前施加强扰动, 同时影响 Q/K/V, VQ routing, forget gate, beta/lambda 和 residual read/write.
2. gd_residual_v1 的 sparse read/write support 和 recurrent M_state 会把这些早期扰动变成路径差异.
3. 但路径差异本身并不必然导致失败. `embed_dropout=0.05 fixed-r4` 说明只要早期扰动强度降低, 同样的 read_topk=4 可以恢复到可接受的跨机器 gap.
4. M/residual branch 不能直接关闭. residual-zero 的低分说明 residual correction 对 hard slice 能力是必要的, 后续应做稳定化, 不是删掉.

## 执行偏差和无效 run

- `fixed-r4-default-2080ti-gpu1` 是额外手动补跑, 因为没有用 `setsid`, 会话结束后进程被带掉, 日志停在训练起点, 不纳入结论.
- `probe-3090-gpu0 fixed-r4` 产出了 trace/result, 但为了避免和 3090 的 `dropout=0.05` 1ep 抢同一张 GPU, 队列在写 completed 状态前被中断; 这部分只作为 early trace 诊断参考.
- P0 失败后没有启动 4ep confirm, 符合 plan.

## 下一步建议

优先做一轮低成本确认, 不要直接跑大网格:

1. 跑 `embed_dropout=0.05, read_topk=4` 的 4ep paired confirm, 2080ti + 3090, 继续使用 canonical cache/init. 这是当前唯一同时有高分和跨机器 4pp 内信号的配置.
2. 同时或随后做 `embed_dropout` sweep: `0.0, 0.025, 0.05, 0.075, 0.1`, 先单机 1ep, 再对临界点做 paired. 目标是找出 default dropout 扰动强度的稳定边界.
3. 设计原则性稳定化 probe: residual injection warmup, beta/lambda cap 或 warmup, M_state update norm cap. 重点不是关掉 residual, 而是降低训练早期 residual state 被 dropout 扰动带偏的强度.
4. read candidate 稳定化仍然重要, 但不要单独作为下一步主线. 当前证据更像是 "dropout 强度 + sparse residual read/write/state" 的耦合问题.

Artifact 目录: `docs/artifacts/20260701-01-flash-vqg-default-dropout-r2-r4-overnight/`.
