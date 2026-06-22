# Flash-VQG seed stability next read schedule 报告

updated: 2026-06-22

## 摘要

本轮完成了 `gd_residual_v1` read-side schedule 的首轮验证. 机制是 early `read_topk=4` 到 late `read_topk=2`, release window 为 train forward count `200->800`, schedule 为 `linear_int`, eval policy 为 `scheduled`.

五条训练都 completed, 没有 traceback, OOM 或 runtime error. 但是结果是负的: `cb256-r8` 三 seed hard spread 为 `0.171`, `cb128-r8` 两 seed hard spread 为 `0.295`, 都远高于本轮 `<=0.03` 的稳定准入门槛. 因此这版固定时间窗 read schedule 不应进入 official longer-MQAR.

## 代码与来源

- zoology branch: `flash-vqg`, head `4d8e4d3`.
- Flash-VQG branch: `20260428-gd-residual-v1-sync`, head `e717489`.
- plan: `docs/plans/20260622-flash-vqg-seed-stability-next-run-plan.md`.
- artifact: `docs/artifacts/20260622-flash-vqg-seed-stability-next/`.
- 3090 raw generated/logs 已同步回 2080ti 工作区, 统一从本地抽取 artifact.

正式准入 smoke 是 `config-runtime-smoke-20260621T194116Z`, CUDA 5/5 passed, 覆盖:

- fixed `read_topk=2`.
- fixed `read_topk=4`.
- `read_topk 4->2` schedule.
- write cap `0.04`.
- bounded beta + orthogonal addr init.

## 训练矩阵

共同配置:

```text
data_seed=123
d_model=128
max_epochs=4
validations_per_epoch=2
train_batch_size=64
eval_batch_size=16
gradient_accumulation_steps=4
fox_remote_formula=gd_residual_v1
fox_gd_residual_rank=8
fox_gd_residual_write_topk=4
fox_remote_read_topk_initial=4
fox_remote_read_topk_final=2
fox_remote_read_topk_release_start_train_steps=200
fox_remote_read_topk_release_end_train_steps=800
fox_remote_read_topk_schedule=linear_int
fox_remote_read_topk_eval_policy=scheduled
write cap=None
beta_control=hard_cap
```

| 机器 | target | seed | codebook | rank |
|---|---|---:|---:|---:|
| 3090 | `cb256r8-sched-s123` | 123 | 256 | 8 |
| 3090 | `cb256r8-sched-s124` | 124 | 256 | 8 |
| 3090 | `cb256r8-sched-s125` | 125 | 256 | 8 |
| 2080ti | `cb128r8-sched-s124` | 124 | 128 | 8 |
| 2080ti | `cb128r8-sched-s125` | 125 | 128 | 8 |

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`.

| config | seed | best hard | final hard | valid acc | max `m_norm_max` over valid | best-final gap |
|---|---:|---:|---:|---:|---:|---:|
| `cb256-r8` | 123 | 0.935 | 0.935 | 0.988 | 3.95 | 0.000 |
| `cb256-r8` | 124 | 0.820 | 0.820 | 0.972 | 5.75 | 0.000 |
| `cb256-r8` | 125 | 0.991 | 0.991 | 0.997 | 1.70 | 0.000 |
| `cb128-r8` | 124 | 0.681 | 0.681 | 0.949 | 3.40 | 0.000 |
| `cb128-r8` | 125 | 0.976 | 0.976 | 0.995 | 4.58 | 0.000 |

Spread summary:

| config | values | worst | spread | verdict |
|---|---|---:|---:|---|
| `cb256-r8` | s123=`0.935`, s124=`0.820`, s125=`0.991` | 0.820 | 0.171 | fail |
| `cb128-r8` | s124=`0.681`, s125=`0.976` | 0.681 | 0.295 | fail |

## 与既有 readk4 证据的关系

这轮 schedule 没有复现 fixed readk4 在 `cb256-r8` 上的强正结果. 之前 `20260530-gd-seed-diag` 中:

- `cb256-r8` readk2 s124/s125 为 `0.988/0.804`, spread `0.184`.
- `cb256-r8` fixed readk4 四条 completed run 为 `0.982/0.982/0.988/0.992`, spread `0.010`.

本轮 `cb256-r8 schedule` 为 `0.935/0.820/0.991`, spread `0.171`. 这说明固定窗口 `4->2` 不能等价替代 fixed readk4 的候选覆盖效果, 至少当前 release window 和 eval policy 不足以稳定 weak seed.

`cb128-r8` 也没有改善. 之前 readk2 main pair 为 `0.956/0.956`, fixed readk4 main pair 为 `0.973/0.972`, 但 readk4 s125 rerun 掉到 `0.609`. 本轮 schedule 是 s124=`0.681`, s125=`0.976`, 说明它仍处在 read-side boundary-sensitive 区间, 只是低分 seed 从旧 rerun 的 s125 转到了这轮 s124.

## Failure 分类

这轮不是 late drift. 五条 run 的 best-final gap 都是 `0`, validation hard curve 基本单调向上或持平, final 没有从高点掉下去.

这轮也不像简单的 state norm explosion. `max_m_norm_max_over_valid` 最大是 `5.75`, 低于本轮红线 `8`, 也没有 `m_norm>12` 这类不可接受状态.

更合理的解释是 early basin selection 仍然存在, 而固定时间窗 schedule 太粗. 它没有根据 read margin, candidate churn 或 proposal confidence 自适应调整候选覆盖和 residual injection, 所以仍然会在不同 seed/layout 上进入不同 basin.

还有一个 telemetry caveat: full training stdout 里的 validation `gd_residual_remote_read_topk_effective` 全部是 `2`. 这不是直接反证 schedule, 因为第一次 validation 已经在 release window `200->800` 之后. 但是它意味着本轮 artifact 不能单靠 validation stdout 审计 early top4 训练窗口. 后续要新增 train-step early telemetry 或在 release window 内做显式 probe.

## 决策

不把这版 `read_topk 4->2, 200->800, linear_int` schedule 送入 official longer-MQAR.

本轮应作为负结果记录:

- `cb256-r8` 上不能替代 fixed readk4.
- `cb128-r8` 上不能解除边界风险.
- failure 与 late drift 或 m_norm redline 无直接对应, 需要更细的 read-side confidence telemetry.

## 下一步

1. 补 train-step early telemetry: 在 step `0,130,203,352,353,448,705` 或 release window 前后记录 effective readk, read margin, read entropy, candidate churn, topk mass, lambda/inject ratio, zeta, M norm.
2. 不再继续扩大固定 `read_topk` 或固定时间窗 schedule 的大矩阵. 下一版应做 margin-aware read gate: margin 低时扩大候选或降低 lambda, margin 高时回到 top2.
3. 把 read gate 与轻量 write guard 或 beta/lambda confidence gate 组合验证, 因为本轮 m_norm 不爆但 basin 仍分叉.
4. `cb128-r8` 保留为 failure case 和边界测试, 不作为当前 read schedule 的正证据.
