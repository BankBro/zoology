# 20260623-01 Flash-VQG 固定样本 read trace 报告

## 结论

这次实验完成了 3090 上三条 fixed-sample read trace run, 有效波次为 `2026-06-23-07-19-43`, trace batch 为 `441`. `batch 441` 对应 eval batch size 16 下 `1024x256` hard slice 的第一批, 避免了之前 batch 0 落在 `64x4` 短序列导致 remote read 被 mask 掉, `selected_mass=0` 的问题.

最重要结果是: `cb256-r8 s125` 上 fixed `read_topk=4` 明显优于 `read_topk=2`, hard accuracy 从 `0.8844` 提升到 `0.9888`. 这复现了 `readk4` 对该 weak seed 的 rescue 现象. 但 fixed-sample trace 不支持一个过度简单的解释: `readk4` 并不是让候选集合更稳定. 它的 fixed-sample churn 和 top1 flip 反而略高于 `readk2`. 更合理的解释是 `readk4` 增加了 early candidate coverage, 改变了后续 residual read/write 轨迹, 但不是单纯通过降低候选翻转来工作.

`cb128-r8 s125 readk4` 本次没有 collapse, final hard 为 `0.9578`, 但 `m_norm_max=8.539`, 高于当前经验红线 `8`. 所以它仍然是边界 layout: 本次 run 成功不等于这个 layout 上 readk4 稳定, 后续仍应做 repeat 或转向 guarded cap / read schedule, 而不是把 fixed readk4 设为全局默认.

## 运行范围

| 项目 | 值 |
|---|---|
| 机器 | `mclab-3090` / `Flash-VQG-tun` |
| 有效 launch 后缀 | `2026-06-23-07-19-43` |
| 数据 seed | `123` |
| model seed | `125` |
| 训练 epoch | `4` configured, checkpoint epoch `3` |
| validation | 每 epoch 4 次, trace 覆盖 16 个 global step |
| trace batch | `441`, 即 `1024x256` hard slice |
| trace 样本 | batch 内前 4 个样本, 每样本最多 8 个有效 query, 所有 layer/head |
| artifact | `docs/artifacts/20260623-01-flash-vqg-fixed-sample-read-trace/` |

前两波 batch 0 run 只是暴露了观测点选错的问题, 不纳入正式分析. 本报告只使用第三波有效结果.

## Final 与 trace 汇总

| target | hard final | valid acc | m_norm_max | trace selected_mass | trace margin | trace entropy | trace churn | trace top1 flip | trace steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cb256r8-readk2-s125-trace` | `0.8844` | `0.9797` | `2.616` | `0.513` | `1.626` | `2.768` | `0.199` | `0.151` | `16` |
| `cb256r8-readk4-s125-trace` | `0.9888` | `0.9969` | `6.191` | `0.475` | `1.520` | `3.292` | `0.223` | `0.196` | `16` |
| `cb128r8-readk4-s125-trace` | `0.9578` | `0.9914` | `8.539` | `0.683` | `1.340` | `2.243` | `0.221` | `0.160` | `16` |

补充说明:

- `cb256-r8 readk4` 的 hard final 为 `0.9888`, 明显高于 `cb256-r8 readk2` 的 `0.8844`.
- `cb256-r8 readk4` 的 trace selected mass 均值低于 readk2, 但 final 更高. 因此 selected mass 均值本身不是充分解释变量.
- `cb256-r8 readk4` 的 trace churn `0.223` 高于 readk2 的 `0.199`, top1 flip `0.196` 高于 readk2 的 `0.151`. 因此 rescue 不是因为候选更少变化.
- `cb128-r8 readk4` 的 fixed-sample selected mass 高, churn 与 `cb256-r8 readk4` 接近, 但 `m_norm_max` 达到 `8.539`, 说明风险更像 write/state 健康问题, 不只是 read candidate 问题.

## 按 step 的 trace 片段

| target | global_step | selected_mass | margin | entropy | records |
|---|---:|---:|---:|---:|---:|
| `cb256r8-readk2-s125-trace` | `176` | `0.509` | `2.571` | `2.713` | `64` |
| `cb256r8-readk2-s125-trace` | `353` | `0.940` | `4.246` | `0.417` | `64` |
| `cb256r8-readk2-s125-trace` | `530` | `0.657` | `2.231` | `1.973` | `64` |
| `cb256r8-readk2-s125-trace` | `2831` | `0.388` | `1.173` | `3.379` | `64` |
| `cb256r8-readk4-s125-trace` | `176` | `0.623` | `3.100` | `2.320` | `64` |
| `cb256r8-readk4-s125-trace` | `353` | `0.958` | `5.099` | `0.324` | `64` |
| `cb256r8-readk4-s125-trace` | `530` | `0.306` | `0.752` | `4.053` | `64` |
| `cb256r8-readk4-s125-trace` | `2831` | `0.563` | `1.097` | `2.961` | `64` |
| `cb128r8-readk4-s125-trace` | `176` | `0.558` | `1.933` | `2.570` | `64` |
| `cb128r8-readk4-s125-trace` | `353` | `0.792` | `3.352` | `1.520` | `64` |
| `cb128r8-readk4-s125-trace` | `530` | `0.686` | `2.708` | `1.866` | `64` |
| `cb128r8-readk4-s125-trace` | `2831` | `0.700` | `1.089` | `2.230` | `64` |

关键观察:

- 在 `global_step=353`, `cb256-r8 readk4` 的 selected mass `0.958`, margin `5.099`, entropy `0.324`, 比 readk2 的 selected mass `0.940`, margin `4.246`, entropy `0.417` 更集中. 这个窗口仍然符合之前报告里早期分叉窗口的判断.
- 到 `global_step=530`, `cb256-r8 readk4` 的 selected mass 降到 `0.306`, entropy 升到 `4.053`, 但 final 仍然高. 说明单点均值波动不能直接等价为失败, 需要和 state/write 指标一起看.
- `cb128-r8 readk4` 后期 selected mass 稳在约 `0.70`, margin 约 `1.09`, 但 `m_norm_max` 偏高. 这支持继续把 cb128-r8 作为边界/风险 case, 而不是 positive proof.

## 对当前假设的影响

这次实验收紧了 read-side 假设:

1. `read_topk=4` 对 `cb256-r8 s125` 是有效 rescue, 不是 batch 0 trace 误读造成的假象.
2. rescue 不能简单归因于 candidate churn 下降. fixed-sample 候选仍会换, 且 readk4 的候选变化不小.
3. 更可能的机制是: early top4 给了模型更大的候选覆盖和更好的早期集中度, 让 weak seed 避开 read_topk=2 的坏路径. 但这个机制在不同 layout 上可能带来不同 state 压力.
4. `cb128-r8 readk4` 这次 final 成功, 但 state 健康指标越过经验线. 这解释了为什么历史上它可能 rerun collapse: 问题可能不在 read candidate 是否一直翻, 而在 read/write feedback 后的 state 放大边界.

## 下一步建议

下一步不建议继续只做 fixed `read_topk=4` repeat. 更应该实现或验证两个方向:

1. `read_topk` schedule: early `4` -> late `2`. 目的不是让候选完全稳定, 而是在早期给覆盖, 后期减少 over-read 和 state pressure.
2. guarded write cap: 对 `m_norm_max > 8` 的 layout 加 release guard. `cb128-r8` 已经显示 readk4 可以高分, 但 state 健康不足, 更像需要 write/state guard.

建议首批矩阵:

| target | control | 目的 |
|---|---|---|
| `cb256-r8 s125` | readk2, readk4, read schedule 4->2 | 验证 schedule 是否保留 rescue |
| `cb128-r8 s125` | readk4, read schedule 4->2, readk4+guarded cap | 验证能否降低 `m_norm_max` 和 rerun 风险 |
| `cb128-r8 s125 repeat` | 至少 1 次 repeat | 确认本次 high run 是否稳定 |

进入 longer-MQAR 之前仍需满足: 三 seed spread 小, worst repeat gap 小, best-final gap 小, `m_norm_max < 8`, 且 good-seed ceiling tax 不超过约 `0.02`.

## 产物

- `final.csv`: final checkpoint 指标.
- `final_best_metrics.csv`: final 与 best checkpoint 指标.
- `trace_summary.csv`: fixed-sample candidate 级汇总.
- `trace_step_summary.csv`: fixed-sample 按 validation step 汇总.
- `source_manifest.csv`: 轻量 source 和 hash.
- `metadata.json`: 收集参数和 trace row.
- `raw_summary.json`: manifest, checkpoint 指标, 日志摘要.
- `traces/*.jsonl.gz`: 压缩后的 fixed-sample trace 明细.
