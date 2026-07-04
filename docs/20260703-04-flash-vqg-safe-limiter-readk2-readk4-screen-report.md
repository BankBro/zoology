# 20260703-04 Flash-VQG safe limiter readk2/readk4 screen report

## 结论

本轮 20 条 paired 1ep run 全部完成, 没有 NaN, OOM 或 Traceback, 但没有任何一个 variant 通过筛选标准.

筛选标准是:

```text
2080ti 和 3090 final valid/mqar_case/accuracy-1024x256 都 >= 0.82
且两机 gap <= 4pp
```

结果说明:

1. `safe residual injection limit` 没有解决 default dropout 下的稳定性问题. ratio=1.0/2.0 不再像旧 `inject-softcap0p5` 那样 NaN, 但分数和跨机器 gap 都不可接受.
2. `scheduled update_norm hard cap 0.5->0.8/1.0` 也没有通过. 它能改变轨迹, 但要么压低两机分数, 要么仍保留很大 gap.
3. `read_topk=4` 在 default dropout 下仍然非常危险. 本轮 r4 baseline 是 2080ti 0.894 vs 3090 0.013, gap 88.1pp; r4 的 limiter 版本也全部低分或高 gap.
4. `read_topk=2` 比 r4 更可控, 但当前这批 limiter 仍没有把 r2 变成稳定方案. r2 baseline 是 0.849 vs 0.480, gap 36.9pp; 最好的 r2 平均分 variant 是 `r2-updatecap-0p5to1p0-linear512`, 但 gap 仍有 13.5pp.

所以这轮要直接判定为 failed screen. 不建议把 safe injection limit 或 scheduled update cap 作为下一轮 4ep confirm 候选.

## 共同设置

本轮是 diagnostic/exploratory screen, 不写 official MQAR ledger.

| 项目 | 设置 |
|---|---|
| zoology commit | `9776938` |
| Flash-VQG commit | `94c1591` |
| seed | `124` |
| data seed | `123` |
| MQAR cache | canonical 13 files |
| init | canonical seed124 init |
| batch order | r2/r4 两机一致 |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| train write | `write_topk=4` |
| train read | `read_topk=2` 或 `read_topk=4` |
| epoch | 1 epoch, `704` optimizer steps |
| extra trace | disabled |

Preflight/guard:

- cache combined content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`
- init model state hash: `2a1107bf22d0804ed485ab94bdc7af8004efb0`
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`
- `train` 入口内置 `pretrain_data_guard=PASS`, 缺 cache, cache hash 不匹配, 或 init hash 不匹配时会直接失败.
- `read_trace_enabled=false`, `read_trace_train_steps=[]`, `train_inline_event_trace_enabled=false`, no shadow dense read, no hash probe.

## 方案定义

### Safe residual injection limit

作用位置:

```text
O_res_added 算出后, Out_f32 = O_base + O_res_added 之前.
```

形式:

```python
ratio = norm(O_res_added.detach()) / sqrt(norm(O_base.detach()) ** 2 + eps)
scale = (1 + (ratio / cap) ** 4) ** (-1 / 4)
O_res_added = O_res_added * scale.detach()
```

本轮测试:

```text
cap = 1.0
cap = 2.0
```

它是默认关闭的候选稳定化控制, 不是默认训练语义.

### Scheduled update hard cap

作用位置:

```text
residual GD 写入 M_state 之前.
```

形式:

```text
fox_gd_residual_update_norm_cap = 0.5
fox_gd_residual_update_norm_cap_final = 0.8 or 1.0
release = 512 optimizer steps = 2048 train forward steps
schedule = linear
```

目标是训练早期限制单次 residual update, 后期逐步放开.

## 主结果

| variant | read_k | 2080ti final | 3090 final | gap | 结果 |
|---|---:|---:|---:|---:|---|
| `r2-baseline` | 2 | 0.849 | 0.480 | 36.9pp | fail |
| `r2-safe-inj-ratio1p0` | 2 | 0.560 | 0.010 | 55.0pp | fail |
| `r2-safe-inj-ratio2p0` | 2 | 0.682 | 0.925 | 24.3pp | fail |
| `r2-updatecap-0p5to0p8-linear512` | 2 | 0.435 | 0.497 | 6.2pp | fail |
| `r2-updatecap-0p5to1p0-linear512` | 2 | 0.745 | 0.880 | 13.5pp | fail |
| `r4-baseline` | 4 | 0.894 | 0.013 | 88.1pp | fail |
| `r4-safe-inj-ratio1p0` | 4 | 0.110 | 0.440 | 33.0pp | fail |
| `r4-safe-inj-ratio2p0` | 4 | 0.084 | 0.487 | 40.3pp | fail |
| `r4-updatecap-0p5to0p8-linear512` | 4 | 0.017 | 0.099 | 8.2pp | fail |
| `r4-updatecap-0p5to1p0-linear512` | 4 | 0.025 | 0.165 | 14.0pp | fail |

注意: `r2-updatecap-0p5to0p8-linear512` 的 gap 最小, 但两机分数只有 0.435/0.497, 是“低分但稍微接近”, 不能解释为稳定成功.

`r2-updatecap-0p5to1p0-linear512` 的两机分数相对最高, 0.745/0.880, 但 gap 13.5pp, 也不能推进.

## Limiter 指标

| variant | machine | final | lambda | inject | inj hit | inj scale | update hit |
|---|---|---:|---:|---:|---:|---:|---:|
| `r2-baseline` | 2080ti | 0.849 | 0.223 | 0.153 | 0 |  | 0 |
| `r2-baseline` | 3090 | 0.480 | 0.0826 | 0.389 | 0 |  | 0 |
| `r2-safe-inj-ratio1p0` | 2080ti | 0.560 | 0.394 | 0.163 | 0.225 | 0.797 | 0 |
| `r2-safe-inj-ratio1p0` | 3090 | 0.010 | 0.235 | 0.147 | 0.154 | 0.911 | 0 |
| `r2-safe-inj-ratio2p0` | 2080ti | 0.682 | 0.108 | 0.301 | 0.0173 | 0.991 | 0 |
| `r2-safe-inj-ratio2p0` | 3090 | 0.925 | 0.399 | 0.130 | 0.191 | 0.854 | 0 |
| `r2-updatecap-0p5to0p8-linear512` | 2080ti | 0.435 | 0.0894 | 0.304 | 0 |  | 0 |
| `r2-updatecap-0p5to0p8-linear512` | 3090 | 0.497 | 0.0887 | 0.0552 | 0 |  | 0.139 |
| `r2-updatecap-0p5to1p0-linear512` | 2080ti | 0.745 | 0.262 | 0.319 | 0 |  | 0.0184 |
| `r2-updatecap-0p5to1p0-linear512` | 3090 | 0.880 | 0.467 | 0.199 | 0 |  | 0 |
| `r4-baseline` | 2080ti | 0.894 | 0.576 | 0.174 | 0 |  | 0 |
| `r4-baseline` | 3090 | 0.013 | 0.150 | 0.131 | 0 |  | 0 |
| `r4-safe-inj-ratio1p0` | 2080ti | 0.110 | 0.676 | 0.219 | 0.169 | 0.912 | 0 |
| `r4-safe-inj-ratio1p0` | 3090 | 0.440 | 0.521 | 0.432 | 0.388 | 0.749 | 0 |
| `r4-safe-inj-ratio2p0` | 2080ti | 0.084 | 0.281 | 0.440 | 0.0737 | 0.960 | 0 |
| `r4-safe-inj-ratio2p0` | 3090 | 0.487 | 0.103 | 0.0437 | 0.0454 | 0.972 | 0 |
| `r4-updatecap-0p5to0p8-linear512` | 2080ti | 0.017 | 0.208 | 0.175 | 0 |  | 0.00358 |
| `r4-updatecap-0p5to0p8-linear512` | 3090 | 0.099 | 0.184 | 0.169 | 0 |  | 0.0648 |
| `r4-updatecap-0p5to1p0-linear512` | 2080ti | 0.025 | 0.231 | 0.258 | 0 |  | 0.000181 |
| `r4-updatecap-0p5to1p0-linear512` | 3090 | 0.165 | 0.194 | 0.174 | 0 |  | 0.0526 |

这些指标支持两个判断:

1. safe injection limit 确实被激活并发生限制, 但限制强度和最终效果没有稳定对应关系. 例如 `r2-safe-inj-ratio2p0` 在 3090 上是 0.925, 但 2080ti 只有 0.682; `r4-safe-inj-*` 两机都不达标.
2. scheduled update cap 也确实被激活, 但没有形成有效稳定化. `0.5->0.8` 更像过度限制, 两机接近但都低; `0.5->1.0` 分数好一些但 gap 仍大.

## 执行状态

- 20/20 run completed.
- `grep` 检查未发现 `Traceback`, `RuntimeError`, `CUDA out of memory`, `loss=nan`, `loss=inf`, `valid/loss=nan`, `valid/loss=inf`.
- 2080ti 和 3090 GPU 均已空闲.
- 3090 轻量 raw evidence 已镜像回主工作区.
- 本轮没有启用 read trace/hash probe, 所以它是方案筛选实验, 不是 first-mismatch 定位实验.

## 判断

这轮给出的最重要信息不是“某个 cap 数值还可以微调”, 而是:

```text
单独限制 residual injection 幅度, 或单独 schedule M_state update hard cap,
不足以让 default dropout 下的 gd_residual_v1 跨机器稳定.
```

更具体地说:

- r2 仍有明显跨机器分叉, 说明 read_topk=2 不是最终稳定方案.
- r4 在 default dropout 下仍然非常不稳, 即使 no-dropout 下 r4 曾经很好, 也不能直接把 r4 带回 default dropout 长训.
- limiter 可以改变轨迹, 但没有把两机压到同一个高分 basin. 这更支持之前的判断: 问题不是单一幅度过大, 而是 dropout 扰动, read/write support, M_state 写入和 residual 注入之间的耦合.

## 下一步建议

不要继续扩展这批 limiter 的 4ep confirm, 也不要继续只调 cap 数值.

下一步应该回到更机制化的方案, 但仍用小成本 1ep screen:

1. `read/write support stabilization`: 重点处理硬 top-k support flip, 比如 read margin guard, adaptive read_topk, 或 early read support smoothing. 当前证据显示 r4 default dropout 极不稳, 单纯限制 update/injection 不能救回来.
2. `residual branch schedule + support guard` 联合控制: 之前 injection warmup 有过较好但复现不稳的信号. 现在应把 residual injection warmup 和 read/write support 稳定化放在一起, 而不是只做 scalar cap.
3. `training trace targeted`: 如果要继续定位, 应只对少数代表配置做 training-minibatch event trace, 例如 `r2-baseline`, `r2-updatecap-0p5to1p0-linear512`, 以及一个未来的 support-stabilized variant, 看 read/write support 是否在真实训练 batch 中分叉.

当前不建议马上跑:

- r4 default dropout 4ep.
- safe injection limit ratio=1/2 的 repeat.
- scheduled update cap 的 4ep confirm.
- 更密的 cap sweep.

## Artifact

核心文件:

- `docs/artifacts/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/run-summary.csv`
- `docs/artifacts/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/cross-machine-comparison.csv`
- `docs/artifacts/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/limiter-metrics-summary.csv`
- `docs/artifacts/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/cache-init-preflight-summary.csv`
- `docs/artifacts/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/source-manifest.csv`
- `docs/artifacts/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/metadata.json`
