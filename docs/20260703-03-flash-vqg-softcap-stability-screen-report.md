# 20260703-03 Flash-VQG smooth cap stability screen report

## 结论

本轮没有找到可推进的 softcap 方案. 在 `default dropout`, `read_topk=2`, `write_topk=4`, canonical cache/init/batch order 全部一致的条件下, 5 个 paired 1ep variant 都没有通过 4pp 跨机器稳定线.

最重要的结果:

- `injection softcap ratio=0.5` 两个 variant 双机都 NaN, 不能继续推进.
- `update_norm smooth_p4 softcap=0.5` 本身没有改善跨机器稳定性, gap 为 29.8pp.
- `update_norm smooth_p4 softcap=0.5 + injection linear512 warmup` 是本轮分数最高的 variant, 2080ti 为 0.903, 3090 为 0.708, 但 gap 仍有 19.5pp, 不通过.

所以这轮说明: 简单把 hard cap 换成 smooth_p4 softcap, 并不能解决当前 default-dropout 下的跨机器放大问题. 后续不应该继续围绕 `cap=0.5` 单独微调, 应回到机制拆解: read/write support, update spike 的训练时序, 以及 residual injection 的更温和 schedule.

## 共同设置

本轮是 diagnostic/exploratory screen, 不写 official MQAR ledger.

| 项目 | 设置 |
|---|---|
| zoology commit | `6ff03df` |
| Flash-VQG commit | `4f85186` |
| seed | `124` |
| data seed | `123` |
| MQAR cache | canonical 13 files |
| init | canonical seed124 init |
| batch order | 两机一致 |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| train read/write | `read_topk=2`, `write_topk=4` |
| epoch | 1 epoch, `704` optimizer steps |
| trace | `read_trace_train_steps=[]`, 训练 trace 关闭 |

Preflight 通过:

- cache combined content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`
- init model state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`

## 方案定义

本轮 softcap 使用:

```python
scale = (1 + (x / cap) ** 4) ** (-1 / 4)
```

`injection softcap` 控制 residual 加回主输出的相对强度:

```python
ratio = norm(O_res_added) / (norm(O_base) + eps)
O_res_added = O_res_added * scale(ratio, cap=0.5)
```

`update_norm softcap` 控制单次写入 `M_state` 的 update 强度:

```python
update_norm = abs(zeta) * norm(err)
zeta = zeta * scale(update_norm, cap=0.5).detach()
```

## 主结果

| variant | 2080ti final 1024x256 | 3090 final 1024x256 | gap | 结果 |
|---|---:|---:|---:|---|
| `baseline-r2-no-trace` | 0.740 | 0.460 | 28.0pp | fail |
| `inject-softcap0p5-r2` | 0.000 | 0.000 | 0.0pp | fail, NaN |
| `inject-softcap0p5-linear512-r2` | 0.000 | 0.000 | 0.0pp | fail, NaN |
| `update-softcap0p5-r2` | 0.806 | 0.508 | 29.8pp | fail |
| `update-softcap0p5-linear512-r2` | 0.903 | 0.708 | 19.5pp | fail |

注意: injection softcap 两条的 gap 是 0, 但这是因为两机都 NaN 后 accuracy 变成 0, 不是稳定成功.

## Softcap 指标

| variant | machine | final loss | final 1024x256 | inject ratio | update hit | update scale min | update norm max | M norm max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2080ti | 0.490 | 0.740 | 0.325 | 0 | 1.000 | 1.90 | 3.53 |
| baseline | 3090 | 0.803 | 0.460 | 0.393 | 0 | 1.000 | 1.66 | 4.80 |
| update-softcap | 2080ti | 0.392 | 0.806 | 0.124 | 0.000351 | 0.719 | 0.646 | 3.52 |
| update-softcap | 3090 | 0.796 | 0.508 | 0.481 | 0.0718 | 0.152 | 3.33 | 2.98 |
| update-softcap+linear512 | 2080ti | 0.240 | 0.903 | 0.254 | 0 | 0.997 | 0.166 | 1.15 |
| update-softcap+linear512 | 3090 | 0.476 | 0.708 | 0.198 | 0.0695 | 0.0582 | 8.64 | 8.06 |

这里能看到一个有用信号: `update-softcap+linear512` 在 2080ti 上把 `update_norm_max` 压到 0.166, `M_norm_max` 压到 1.15, 分数升到 0.903. 但 3090 上同样配置仍有 `update_norm_max=8.64`, `M_norm_max=8.06`, final 只有 0.708. 这说明当前 softcap 方案没有把两机训练轨迹压到同一稳定区域.

## 执行说明

本轮实际执行中, 原始 queue 由于 20 分钟轮询会在 target 间空等, 所以对部分已经完成目标后的 idle queue 做了停止, 并用 remainder queue 补齐剩余目标. 另外 `update-softcap0p5-linear512-r2` 在 2080ti 上曾短暂启动一个重复进程, 已停止该重复进程; 正式结果采用更早启动并完整完成的 GPU0 run.

最终 artifact 中 `run-summary.csv`, `cross-machine-comparison.csv`, `softcap-metrics-summary.csv` 均以完整 result/log 为准, 10 个正式 paired run 全部完成. 被停止的重复进程只保留在 raw output/log 中用于审计, 不计入主结果.

本轮还暴露一个监控问题: 原 wrapper 的 NaN grep 没能捕获 `valid/loss=nan`, 因此 injection softcap 的 NaN run 被 wrapper 标成 completed. 已修复后续 queue monitor 的 grep, 但本轮结果仍按 NaN failure 解释.

## 判断

1. `injection softcap ratio=0.5` 不是可用方案. 它不是轻微稳定 residual 注入, 而是直接导致双机 NaN collapse. 这个方向如果要继续, 需要先单独做更小 ratio 或更稳的 eps/norm 保护 smoke, 不能直接放回 paired 1ep.

2. `update_norm smooth_p4 softcap=0.5` 方向仍然比 injection softcap 合理, 但本轮没有通过. 它在某些机器上能提高分数, 但不能保证跨机器稳定.

3. `update_norm softcap + residual injection warmup` 有正向信号, 但不足以作为解决方案. 它把 2080ti 推到 0.903, 3090 也比 baseline 的 0.460 提高到 0.708, 说明限制 update 幅度和推迟 residual 注入确实影响放大链路. 但 19.5pp gap 太大, 不能进入 4ep confirm.

4. 当前更像是 residual memory 的写入幅度, residual 注入强度, read/write support 分叉三者耦合. 单独一个 smooth cap 不够.

## 下一步建议

不要继续盲调 `smooth_p4 cap=0.5`.

建议下一轮做两个更有判别力的方向:

1. 训练时序定位: 对 `baseline-r2`, `update-softcap+linear512` 做 training-minibatch event trace, 只记录真实训练 batch 中 update spike 的 step/layer/code/head, 并对齐 loss/read output/M_state norm 变化. 目标是回答 3090 上为什么仍有 `update_norm_max=8.64` 和 `M_norm_max=8.06`.

2. 机制型稳定化: 先设计 read/write support 稳定方案, 例如 margin-aware read 或 write spike guard. 本轮已经说明单独 update softcap 不能把轨迹稳定住, 后续应该把 support flip 和 update spike 一起处理.

如果还想继续 softcap 线, 最小可做的是把 `update-softcap+linear512` 重复一轮 seed124 no-trace paired 1ep, 验证 0.903/0.708 是否稳定复现. 但即使复现, 它也只是部分改善, 不是最终方案.

## Artifact

核心文件:

- `docs/artifacts/20260703-03-flash-vqg-softcap-stability-screen/run-summary.csv`
- `docs/artifacts/20260703-03-flash-vqg-softcap-stability-screen/cross-machine-comparison.csv`
- `docs/artifacts/20260703-03-flash-vqg-softcap-stability-screen/softcap-metrics-summary.csv`
- `docs/artifacts/20260703-03-flash-vqg-softcap-stability-screen/cache-init-preflight-summary.csv`
- `docs/artifacts/20260703-03-flash-vqg-softcap-stability-screen/source-manifest.csv`
- `docs/artifacts/20260703-03-flash-vqg-softcap-stability-screen/metadata.json`
