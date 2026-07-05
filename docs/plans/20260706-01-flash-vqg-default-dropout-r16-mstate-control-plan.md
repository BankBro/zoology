# 20260706-01 Flash-VQG default-dropout r16 M_state control plan

## 目标

本轮继续推进 Flash-VQG / `gd_residual_v1` 在 default dropout 下的跨机器稳定性问题. 目标不是继续扫 `read_topk`, 也不是找最高分, 而是固定 `read_topk=16`, 验证 `M_state` residual memory 是否因为以下三类机制放大正常训练扰动:

1. 单次 residual update 太猛.
2. `M_state` 整体范数长太大.
3. residual correction 太早, 太强地影响最终输出.

网页 ChatGPT 的建议只作为候选假设. 本轮以当前代码实际支持的配置项为准. 已确认 Flash-VQG 当前支持:

- `fox_gd_residual_update_norm_softcap`
- `fox_gd_residual_update_norm_softcap_mode="smooth_p4"`
- `fox_gd_residual_m_norm_cap`
- `fox_gd_residual_injection_warmup_*`

## 固定条件

所有 formal runs 固定:

| item | value |
|---|---|
| seed | `124` |
| data seed | `123` |
| data/init/batch | canonical MQAR cache, canonical seed124 init, same batch order |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| model | `cb64-r16` |
| residual read | `fox_remote_read_topk=16` |
| residual write | `fox_gd_residual_write_topk=4` |
| train length | `max_epochs=1`, about `704` optimizer steps |
| grad accumulation | `4` |
| machines | `2080ti` + `3090`, paired |

本轮不做:

- `read_topk` sweep.
- `fixed-r64`.
- `softread`.
- `read-confidence injection`.
- `topk_mass_scaled`.
- hard `update_norm_cap=0.5`.
- 4ep.
- 多 seed.
- no-dropout 或 dropout 调整.
- heavy read trace / inline event trace.

## Variants

| variant | 目的 | 配置差异 |
|---|---|---|
| `fixed-r16-baseline` | 本轮同环境 r16 对照 | 无 M_state control |
| `r16-update-softcap0p5` | 验证单次 `M_state` update 幅度是否是放大器 | `fox_gd_residual_update_norm_softcap=0.5`, `fox_gd_residual_update_norm_softcap_mode="smooth_p4"` |
| `r16-mnorm-cap6` | 验证 `M_state` 整体状态范数是否是放大器 | `fox_gd_residual_m_norm_cap=6.0` |
| `r16-update-softcap0p5-injwarm512` | 验证写入控制 + residual 输出注入延迟是否需要联合 | update softcap 同上, `injection_warmup` 从 optimizer step `0` 线性升到 `512` |

注意: warmup 配置使用 train-forward step, 因为 `grad_accumulation_steps=4`, 所以 optimizer step `512` 对应 train-forward step `2048`.

## Preflight 和 smoke

formal 前必须两机分别检查:

- 容器内 `nvidia-smi` / NVML 可用.
- 容器内 `torch.cuda.is_available()` 为 true.
- zoology commit 一致.
- Flash-VQG commit 一致.
- MQAR cache 内容 hash 一致.
- init checkpoint tensor hash 一致.
- batch order hash 一致.
- heavy read trace 和 inline event trace 为关闭.

Smoke:

- 每台机器跑同一批 4 个 variant.
- `SMOKE_TRAIN_STEPS=8`.
- `SMOKE_VALIDATION_BATCHES=16`.
- 每个 variant 都执行 `cache-hash -> preflight -> batch_preflight -> train`.
- smoke 无 `Traceback`, `CUDA out of memory`, `NaN/Inf` 后才启动 formal.

Formal:

- 两机各一个 queue.
- 每台机器内部 4 个 formal run 串行自动接续.
- `CONTINUE_ON_FAIL=1`, 单个 variant 失败后继续后续 variant, 但状态写入 `queue-status.tsv`.

## 输出与判定

主要输出:

- `run-summary.csv`
- `cross-machine-comparison.csv`
- `mechanism-metrics-summary.csv`
- `variant-summary.csv`
- `source-manifest.csv`
- `metadata.json`
- `README.md`

主要指标:

- final/best valid accuracy.
- final/best `1024x256` accuracy.
- paired hard gap.
- `read_selected_mass_mean`.
- `read_entropy_mean`, 若日志中存在.
- `read_margin_top1_top2_mean`.
- `update_norm_mean/p95/max`.
- `update_softcap_scale_mean/min/p05`.
- `m_norm_mean/max`.
- `m_norm_cap_hit_ratio`.
- `lambda_mean`.
- `inject_ratio`.
- `injection_warmup_factor`.
- `write_strength_mean`.
- `raw_topk_mass_mean`.
- `write_top1_mass_mean`.
- VQ entropy / usage / write sharpness.
- NaN / OOM / Traceback 状态.

Screen pass 条件:

```text
两机 final 1024x256 accuracy 都 >= 0.85
paired gap <= 4pp
无 NaN / OOM / Traceback
cache / init / batch order 一致
```

低分但 gap 小不算成功. 单机高分不算成功. 若有 variant 过线, 下一步只做 same-seed paired rerun, 不直接推进 4ep 或多 seed.

## 报告必须回答

1. `fixed-r16-baseline` 在本轮是否复现.
2. `r16-update-softcap0p5` 是否比 baseline 更稳.
3. `r16-mnorm-cap6` 是否有效, 是高分稳定还是低分稳定.
4. `r16-update-softcap0p5-injwarm512` 是否优于单独 update softcap.
5. 如果某个 variant 过线, 是否值得 same-seed paired rerun.
6. 如果都不过线, 下一步是否应转向 support confidence guard, code/head-aware control, 或更深层 read/write/state 解耦.
7. 本轮结果是否支持继续从 `M_state` 写入, 状态, 注入控制入手.
