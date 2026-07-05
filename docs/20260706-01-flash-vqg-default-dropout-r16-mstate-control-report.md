# 20260706-01 Flash-VQG default-dropout r16 M_state control report

## 结论摘要

本轮固定 `read_topk=16`, `write_topk=4`, default dropout, canonical cache/init/batch order, 做 2080ti + 3090 paired 1ep screen. 目的不是继续扫 `read_topk`, 而是检查 `M_state` 写入幅度, `M_state` 状态大小, residual 输出注入时机这三类控制是否能缓解跨机器训练不稳定.

核心结果很明确:

1. `fixed-r16-baseline` 本轮没有复现上一轮的弱正信号. 2080ti hard slice 为 `0.803`, 3090 为 `0.261`, gap `54.2pp`, 失败.
2. 单独 `update-softcap0p5` 失败. 它没有让两机进入同一个高分 basin, 2080ti 甚至降到 `0.151`.
3. 单独 `m_norm_cap6` 失败. 这说明粗暴限制 `M_state` 整体范数不是当前可用方案.
4. `update-softcap0p5 + injwarm512` 是本轮唯一过线配置: 2080ti `0.901`, 3090 `0.923`, gap `2.2pp`, 两机 overall accuracy 也都高.

这支持一个更具体的判断: **只控制 M_state 写入幅度不够, 只控制 M_state 范数也不够; 至少在本轮 seed124 / r16 / default dropout 下, 写入幅度控制需要和 residual injection 延迟联合, 才能把两机带进同一个高分区域.**

但这还不是最终方案. 它只是 same-seed paired 1ep 的强正信号. 下一步应该先 same-seed paired rerun, 不应该直接推进 4ep 或多 seed.

## 实验条件

所有 formal run 固定:

| item | value |
|---|---|
| seed | `124` |
| data seed | `123` |
| cache | canonical MQAR cache, 13 files |
| init | canonical seed124 init checkpoint |
| batch order | fixed, hash matched across machines |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| model | `cb64-r16` |
| residual read | `fox_remote_read_topk=16` |
| residual write | `fox_gd_residual_write_topk=4` |
| training | `max_epochs=1`, `max_train_steps=704`, `grad_accumulation_steps=4` |
| machines | 2080ti + 3090 |
| trace | heavy read trace and train inline event trace disabled |

Preflight 结果:

- MQAR cache content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`, 两机 match.
- init tensor hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`, 两机 match.
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`, 两机 match.
- 8 个 formal run 全部 completed, 无 `Traceback`, OOM, NaN/Inf.

注意: 本轮 artifact 只统计两个 formal output dirs:

- `outputs/mstate-2080ti-gpu0-20260705T074648Z`
- `outputs/mstate-3090-gpu0-20260705T074648Z`

较早的 smoke output dirs 已排除, 不计入 `run-summary.csv` 或判定.

指标口径: 本轮 `results/*.json` 中 `train_result` 为 `null`, 因此 final/best valid accuracy, final/best `1024x256`, loss 等指标均从 formal log 的 validation summary 解析得到. 原始日志中 tqdm 重绘会让同一次 validation summary 出现多条文本行; collector 去重后记录为 4 次 validation summary, 不把重绘行当作额外 eval.

## 结果总表

| variant | 2080ti 1024x256 | 3090 1024x256 | gap | overall acc, 2080ti / 3090 | pass |
|---|---:|---:|---:|---:|---|
| `fixed-r16-baseline` | 0.803 | 0.261 | 54.2pp | 0.954 / 0.834 | False |
| `r16-update-softcap0p5` | 0.151 | 0.465 | 31.4pp | 0.696 / 0.895 | False |
| `r16-mnorm-cap6` | 0.700 | 0.029 | 67.0pp | 0.940 / 0.558 | False |
| `r16-update-softcap0p5-injwarm512` | 0.901 | 0.923 | 2.2pp | 0.981 / 0.985 | True |

Screen pass 条件为两机 final `1024x256 >= 0.85` 且 gap `<=4pp`. 因此只有 `r16-update-softcap0p5-injwarm512` 过线.

## 每个 variant 的解释

### fixed-r16-baseline

结果: 2080ti `0.803`, 3090 `0.261`, gap `54.2pp`.

这说明 `fixed-r16` 本身并不稳定. 它上一轮曾出现 `0.912/0.850` 的弱正信号, 但本轮 same-seed paired repeat 没有复现. 因此不能把 `read_topk=16` 本身升格为默认稳定配置.

### r16-update-softcap0p5

结果: 2080ti `0.151`, 3090 `0.465`, gap `31.4pp`.

这个配置用 smooth_p4 softcap 控制单次 `M_state` residual update 幅度, 但单独使用失败. 它确实改变了 M_state 轨迹, 但没有稳定训练, 也没有同时保持高分. 这说明“只把单次 update 缩小”不是充分方案.

### r16-mnorm-cap6

结果: 2080ti `0.700`, 3090 `0.0295`, gap `67.1pp`.

这个配置用 hard cap 限制 `M_state` 整体范数, 也失败. 3090 几乎崩到低分. 因此 `m_norm_cap=6` 只能作为诊断反例, 不能作为候选方案推进.

### r16-update-softcap0p5-injwarm512

结果: 2080ti `0.901`, 3090 `0.923`, gap `2.2pp`, 过线.

这个配置同时做两件事:

1. 用 smooth_p4 softcap 限制单次 `M_state` update 过大.
2. residual injection warmup 从 optimizer step 0 线性升到 512. 因为 grad accumulation 为 4, 实际 train-forward step 是 `0 -> 2048`.

它是本轮唯一同时满足高分和低 gap 的配置. 这说明 residual branch 的问题不是单独的“写入过大”或“状态过大”, 而更像是: **早期 M_state 还不可靠时, 如果 residual correction 太早强注入输出, 会把扰动放大到训练轨迹; 写入控制和输出注入时机需要联合处理.**

## 机制指标

以下为 final validation 日志解析, 每个单元为 `2080ti / 3090`:

| variant | update_norm_p95 | softcap_hit_ratio | m_norm_max | lambda_mean | inject_ratio |
|---|---:|---:|---:|---:|---:|
| `fixed-r16-baseline` | 0.761 / 0.164 | 0.000 / 0.000 | 9.560 / 3.470 | 0.227 / 0.091 | 0.098 / 0.256 |
| `r16-update-softcap0p5` | 0.644 / 0.675 | 0.075 / 0.065 | 3.550 / 3.290 | 0.256 / 0.106 | 0.145 / 0.346 |
| `r16-mnorm-cap6` | 0.057 / 0.633 | 0.000 / 0.000 | 4.810 / 6.000 | 0.353 / 0.157 | 0.152 / 0.123 |
| `r16-update-softcap0p5-injwarm512` | 0.136 / 1.330 | 0.000 / 0.096 | 1.320 / 12.300 | 0.256 / 0.813 | 0.213 / 0.242 |

几点需要谨慎解释:

- `update-softcap0p5` 单独启用时, 两机 softcap hit ratio 约 `6-8%`, 但结果仍失败. 这说明 cap 到了部分 update, 但没有解决残差分支何时影响输出的问题.
- `update-softcap0p5-injwarm512` 过线, 但 3090 final validation 的 `m_norm_max=12.3`, 并不是“状态范数越小越好”. 因此本轮不支持简单 hard `M_state` norm cap.
- 过线配置的 `injection_warmup_factor=1` 是 final eval 时的值. 它不表示训练全程都满注入; 训练早期 warmup factor 从 0 线性升到 1. 因此这个 variant 的作用主要在训练早期, 而不是 final eval 静态指标.
- 过线配置两机的 `lambda_mean` 差异仍较大 (`0.256 / 0.813`), 但最终 hard slice 都高. 这提示目标不是让所有内部指标跨机器完全一致, 而是让机制不要把早期扰动带到低分 basin.

## 回答 plan 中的问题

1. `fixed-r16-baseline` 是否复现?  
   没有. 本轮 `0.803/0.261`, gap `54.2pp`, 不稳定.

2. update softcap 是否比 baseline 更稳?  
   单独看不是. `r16-update-softcap0p5` 两机都是低/中分, gap 仍 `31.4pp`. 它不是可推进方案.

3. m_norm_cap 是否有效?  
   无效. `r16-mnorm-cap6` 是失败配置, 特别是 3090 `0.0295`.

4. update softcap + injection warmup 是否比单独 update softcap 更好?  
   是, 而且差异很大. 单独 softcap 是 `0.151/0.465`; 联合 warmup 后变成 `0.901/0.923`, gap `2.2pp`.

5. 过线 variant 是否值得 same-seed paired rerun?  
   值得, 但只做同 seed paired rerun. 还不应直接 4ep 或多 seed, 因为历史上 injection warmup 和 cap 类方法有过复跑波动.

6. 如果都不过线是否转向 support confidence guard?  
   本轮不是全失败. 因此下一步优先复跑 `r16-update-softcap0p5-injwarm512`. 如果复跑不稳定, 再转向 support confidence guard 或更细的 read/write/state 解耦.

7. 本轮是否支持继续从 M_state 写入, 状态, 注入控制入手?  
   支持, 但重点应从“单独控制 M_state 写入/范数”转向“写入控制 + residual injection schedule/confidence 的联合设计”.

## 与历史结果的关系

本轮与之前几条观察一致:

- default dropout 是正常训练扰动入口, 不是要被关掉的 bug.
- `gd_residual_v1` 的 residual read/write/state/injection 路径会放大早期扰动.
- 单独 scalar limiter 很难稳定复现. 例如 hard update cap 曾有一次强正信号, 后续复跑不稳.
- injection warmup 过去已经显示能缓解 gap, 但单独 warmup 也不够稳. 本轮显示它与 update softcap 组合后有更强信号.

因此当前更合理的研究方向不是继续调一个固定 cap 数值, 而是设计更机制化的联合控制:

```text
早期 M_state 写入不要太猛;
早期 residual correction 不要太早强注入输出;
当 read/write support 低置信时, residual branch 应该更保守.
```

## 下一步建议

下一步不要直接跑 4ep. 建议按这个顺序:

1. same-seed paired 1ep rerun `r16-update-softcap0p5-injwarm512`.  
   目标是确认 `0.901/0.923`, gap `2.2pp` 是否可复现.

2. 如果复跑仍过线, 再做一个很小的 ablation:  
   - `r16-injwarm512-only`.
   - `r16-update-softcap0p5-injwarm512` repeat.  
   目标是确认这次收益来自联合, 还是其实主要由 warmup 驱动.

3. 如果 same-seed repeat 仍稳定, 再考虑 alternate seed paired 1ep.  
   这一步通过后, 才值得考虑 4ep confirm.

4. 如果 repeat 不稳定, 不继续扫 cap 数值. 转向 support-aware 机制:  
   - read confidence gated injection.
   - write confidence gated update.
   - code/head-aware residual control.

## Artifact

主要文件:

- `docs/artifacts/20260706-01-flash-vqg-default-dropout-r16-mstate-control/run-summary.csv`
- `docs/artifacts/20260706-01-flash-vqg-default-dropout-r16-mstate-control/cross-machine-comparison.csv`
- `docs/artifacts/20260706-01-flash-vqg-default-dropout-r16-mstate-control/mechanism-metrics-summary.csv`
- `docs/artifacts/20260706-01-flash-vqg-default-dropout-r16-mstate-control/cache-init-preflight-summary.csv`
- `docs/artifacts/20260706-01-flash-vqg-default-dropout-r16-mstate-control/batch-order-summary.csv`
- `docs/artifacts/20260706-01-flash-vqg-default-dropout-r16-mstate-control/source-manifest.csv`
- `docs/artifacts/20260706-01-flash-vqg-default-dropout-r16-mstate-control/metadata.json`
