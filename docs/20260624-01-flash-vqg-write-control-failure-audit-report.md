# 20260624-01 Flash-VQG write-control 失败机制审计报告

updated: 2026-06-24
experiment_id: `20260624-01-flash-vqg-write-control-failure-audit`

## 摘要

本轮没有启动新训练, 只把已有 `cb64-r16` write-control 历史 run 重新整理为可复查表格. 审计脚本读取本地 `history.csv` 和 manifest, 输出到:

```text
docs/artifacts/20260624-01-flash-vqg-write-control-failure-audit/
```

最重要结论:

1. `hard04` 仍是最可靠的稳定基准: 三 seed hard 为 `0.945039 / 0.963055 / 0.952605`, spread `0.018016`, 但有明确 ceiling tax.
2. `caprel0406late` 证明 release 思路有潜力: 三 seed spread `0.013633`, 但 s123 final `m_norm_max=14.487579`, validation 曲线最大 `m_norm_max=15.735760`, 超过 `>12` 红线.
3. `cap0405` 已经失败, 不应原样重跑: s123/s124 final hard 为 `0.811086 / 0.960211`, spread `0.149125`, 且 s123 `m_norm_max=5.896`, 说明失败不是简单 `m_norm` 爆炸.
4. `cap0406 + m_norm_cap=8` 不是有效 guard: s123/s124 final hard 为 `0.895215 / 0.965512`, spread `0.070297`, `m_norm` 没越线但结果仍不稳.
5. 下一步如果实现 guard, 不能只看 `m_norm`. 更应该做 pressure-aware release: 同时看 cap-hit, write strength/sum_zeta, lambda/inject ratio, best-final gap, 以及 read-side 指标.

## 审计范围

| setting | seeds | 控制项 | 证据状态 |
|---|---:|---|---|
| `default` | `123/124/125` | no write cap | official baseline |
| `hard04` | `123/124/125` | `write_strength_cap=0.04` | exploratory trust-region baseline |
| `caprel0406late` | `123/124/125` | `0.04 -> 0.06`, release `2820->8468` | exploratory release |
| `cap0405` | `123/124` | `0.04 -> 0.05`, release `2820->8468` | exploratory conservative release |
| `cap0405_beta0p16` | `123/124` | `0.04 -> 0.05`, `beta_init=0.16` | exploratory beta diagnostic |
| `cap0406_mcap8` | `123/124` | `0.04 -> 0.06`, `m_norm_cap=8` | exploratory static cap diagnostic |

## 结果表

主指标是 `valid/mqar_case/accuracy-1024x256`.

| setting | seeds | final hard | spread | max `m_norm_max` | 判定 |
|---|---|---|---:|---:|---|
| `default` | `123/124/125` | `0.968711 / 0.819797 / 0.987285` | `0.167488` | `5.445922` | unstable baseline |
| `hard04` | `123/124/125` | `0.945039 / 0.963055 / 0.952605` | `0.018016` | `7.609639` | stable ceiling tax |
| `caprel0406late` | `123/124/125` | `0.949371 / 0.963004 / 0.960484` | `0.013633` | `15.735760` | low spread, state overrun |
| `cap0405` | `123/124` | `0.811086 / 0.960211` | `0.149125` | `5.896309` | late drift without m_norm overrun |
| `cap0405_beta0p16` | `123/124` | `0.900777 / 0.912422` | `0.011645` | `6.822552` | partial rescue, low ceiling |
| `cap0406_mcap8` | `123/124` | `0.895215 / 0.965512` | `0.070297` | `5.369979` | ineffective static m_norm cap |

完整表见:

- `write_control_final_summary.csv`
- `write_control_setting_summary.csv`
- `failure_taxonomy.csv`

## 关键观察

### 0. 旧实验能看的已整理, 但缺失指标不能事后补出

本轮只读取已有 `history.csv` 和 manifest. 因此旧实验中已经记录的 scalar 已经被系统整理, 包括 hard accuracy, best/final gap, `m_norm`, write strength, cap-hit, lambda, inject ratio 和 beta 等.

但部分现在最想看的指标在不少旧 run 中没有记录, 不能事后从 checkpoint 或最终 CSV 里恢复, 例如:

```text
read margin / entropy / selected mass 的完整早期轨迹
read candidate churn
update_norm p95
update_norm cap-hit ratio
cap guard reason
release progress / hold / rollback 状态
```

所以本报告能确认的是: `cap0405` 失败不是单纯 `m_norm` 爆炸, `caprel0406late` 有明确 state overrun, 静态 `m_norm_cap=8` 不等于有效 guard. 本报告不能确认的是: read 侧是否就是最终直接原因, update norm 是否是关键触发器, 以及最优 guard 应该使用哪组指标.

缺失指标明细见 `missing_metrics.csv`.

### 1. `hard04` 的价值和代价都明确

`hard04` 把 `cb64-r16` 的 hard spread 从 default `0.167488` 压到 `0.018016`. 这说明 write trust-region 的因果干预证据仍然最强.

代价是 good seed 上限下降:

| seed | default | hard04 | delta |
|---:|---:|---:|---:|
| `123` | `0.968711` | `0.945039` | `-0.023672` |
| `125` | `0.987285` | `0.952605` | `-0.034680` |

因此 `hard04` 应继续作为稳定基准和对照, 但不应写成最终性能方案.

### 2. `caprel0406late` 不是安全替代

`caprel0406late` 的 final spread 最低, 但 s123 final `m_norm_max=14.487579`, validation 曲线最大 `15.735760`. 这超过 roadmap 里 `m_norm_max > 12` 不进 official 的原则性红线.

这说明 release 方向有价值, 但无条件 `0.04 -> 0.06` release 不能直接推荐. 它是“低 spread 但 state 过冲”的风险案例.

### 3. `cap0405` 的失败否定了“只看 m_norm”的解释

`cap0405` 的 s123 final hard 只有 `0.811086`, best hard `0.856094`, best-final gap `0.045008`. 但 s123 最大 `m_norm_max=5.896309`, 没有过 `8` 警戒线.

这很关键: 如果失败只是 `M_state` 范数爆炸, `cap0405` 应该相对健康. 但它仍然失败, 说明至少还有 write pressure, lambda/readout, routing/read-side 或 late trajectory 的问题.

所以不应继续原样重跑 `0.04 -> 0.05`; 它已经是负结果.

### 4. `m_norm_cap=8` 不能等同于 guard

`cap0406_mcap8` 的 s123/s124 spread 是 `0.070297`, 明显差于 `hard04` 和 `caprel0406late`. 同时最大 `m_norm_max=5.369979`, 没有真正触及 `m_norm_cap=8` 的限制.

所以这轮不是有效的 active guard 压力测试. 它只能说明:

```text
静态 m_norm_cap=8 没有解决这个 failure.
```

不能把它写成 “guarded release 已经试过”.

### 5. `beta0.16` 只能算诊断, 不是主线

`cap0405_beta0p16` 的两 seed spread 很小, 但 final hard 只有 `0.900777 / 0.912422`, 低于 `hard04`. 它更像把两个 seed 都压到中等盆地, 而不是恢复 ceiling.

因此 beta init 或 beta band 可以继续作为诊断轴, 但不应替代 write-control 主线.

## 对下一步的影响

本轮收紧后的建议是:

1. 不要再原样跑 `cap0405`.
2. 不要把 `m_norm_cap=8` 当成真正 guarded release.
3. 不要只用 `m_norm` 设计 guard.
4. 如果实现新的 guard, 应优先做 `pressure-aware cap release`, 而不是单指标 `m_norm guard`.

最小 guard 设计应该至少记录或使用:

```text
m_norm_max
write_strength_cap_hit_ratio
write_strength_mean/p95/max
uncapped_write_strength_mean/p95/max
sum_zeta_mean/p95/max
lambda_mean
inject_ratio
best-final gap
read margin / entropy / selected mass, if available
```

推荐下一步分两段:

1. 先实现更完整的 write/update pressure telemetry, 尤其是 update norm p95/hit ratio, cap guard reason, effective cap progress.
2. 再实现最小 `state/pressure-aware write cap release`: release 不是按时间必然推进, 而是在 state/write/readout 健康时推进, 风险出现时 hold 或 rollback.

## 产物

| 文件 | 用途 |
|---|---|
| `write_control_final_summary.csv` | 每个 run 的 final/best, state/write/readout 指标 |
| `write_control_setting_summary.csv` | 每个 setting 的 spread 和最大 state 风险 |
| `write_control_step_curves.csv` | validation step 曲线 |
| `failure_taxonomy.csv` | setting 级失败类型 |
| `missing_metrics.csv` | 历史 run 缺失指标 |
| `source_manifest.csv` | source path, size, sha256 |
| `metadata.json` | artifact 元数据 |

## 验证

已执行:

```bash
python -m py_compile zoology/experiments/flash_vqg/scripts/20260624-01-flash-vqg-write-control-failure-audit/collect_write_control_audit.py
python zoology/experiments/flash_vqg/scripts/20260624-01-flash-vqg-write-control-failure-audit/collect_write_control_audit.py --check
```

`--check` 复核了 default, hard04, caprel0406late, cap0405 的关键 hard accuracy, 以及 caprel0406late s123 的 final/max `m_norm`.
