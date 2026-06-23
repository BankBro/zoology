# 20260624-02 Flash-VQG pressure telemetry and guard 规划

updated: 2026-06-24
status: stage-2 launch
experiment_id: `20260624-02-flash-vqg-pressure-telemetry-guard`

## 目标

本计划承接 `20260624-01-flash-vqg-write-control-failure-audit`. 该审计已经确认:

```text
1. hard04 稳定但有 ceiling tax.
2. caprel0406late 有 low spread, 但 m_norm 过冲.
3. cap0405 失败时 m_norm 没爆, 所以只看 m_norm 不够.
4. 静态 m_norm_cap=8 不是有效 guarded release.
5. 旧实验缺少 update_norm, guard 状态和完整 read-side early telemetry, 不能事后补.
```

本计划的目标是先补 telemetry, 再用最小复现实验判断 guard 应该防什么. 不直接进入大矩阵, 不直接把 `m_norm_cap` 当作 guard, 不直接实现复杂 read gate.

## 非目标

- 不重跑同一个 `cap0405` 作为性能候选.
- 不把 `caprel0406late` 推入 official.
- 不把 `m_norm_cap=8` 称为 state-aware guarded release.
- 不同时叠加 beta band, read gate, init transplant 或大范围 seed sweep.
- 不在 telemetry 结果出来前决定 guard 的最终指标组合.

## 阶段 1: 补 telemetry, 默认不改变训练行为

新增或确认输出以下 scalar:

```text
update_norm_mean / p95 / max
update_norm_cap_hit_ratio
write_strength_cap_hit_ratio
write_strength_effective_cap
write_strength_scheduled_cap, if schedule exists
write release progress, if schedule exists
uncapped_write_strength_mean / p95 / max
sum_zeta_mean / p95 / max
uncapped_sum_zeta_mean / p95 / max
lambda_mean
inject_ratio
read_margin_top1_top2_mean / p05, if available
read_entropy_mean, if available
read_selected_mass_mean / p05, if available
```

实现要求:

- 这些指标只作为 telemetry, 第一版不能改变训练结果.
- 如果某个指标只在 validation 阶段可得, 在报告中明确说明.
- train-step early telemetry 要覆盖 release 前后, 至少能观察 step `352/705/1408/2117/2823` 附近.
- manifest 或 stdout 中必须能看到 requested controls 和 effective controls.

第一阶段执行边界:

- 只修改 telemetry 与 metric whitelist, 不实现 `guarded release`.
- 只运行 config-to-runtime smoke, 不启动完整 MQAR 训练.
- smoke 覆盖 `hard04`, `caprel0406late` 风格的 scheduled release, 以及 `update_norm_cap`.
- smoke 成功标准是新增 scalar 能从 runtime metrics 传出, requested config 与 effective runtime metric 一致.
- 2080ti 作为主开发和首轮验证机器; 3090 只在代码 commit/push 后通过 git pull 同步, 再运行同一 smoke.

## 阶段 2: 最小 telemetry probe

只跑用于机制观测的小矩阵:

| layout | seed | setting | 目的 |
|---|---:|---|---|
| `cb64-r16` | `123` | `default` | 原始 good seed pressure 基线 |
| `cb64-r16` | `124` | `default` | 原始 weak seed pressure 基线 |
| `cb64-r16` | `123` | `hard04` | 稳定低税基准的 pressure 轨迹 |
| `cb64-r16` | `124` | `hard04` | bad seed rescue 的 pressure 轨迹 |
| `cb64-r16` | `123` | `caprel0406late` | m_norm overrun / release 风险轨迹 |
| `cb64-r16` | `124` | `caprel0406late` | release 对 bad seed 的健康轨迹 |
| `cb64-r16` | `123` | `cap0405` | m_norm 不爆但 final 失败的 pressure/readout 轨迹 |
| `cb64-r16` | `124` | `cap0405` | 同配置高分轨迹对照 |

说明:

- 这些 run 的目的不是重新证明 final 结果, 而是用新增 telemetry 比较 release window 前后发生了什么.
- release 配置统一使用 `write_strength_cap_eval_policy=scheduled`, 让 validation 按当前训练进度读取 cap, 避免中途 valid 提前看到 final cap.
- 3090 跑 seed123 四条, 单卡最多 3 条并发; 2080ti 跑 seed124 四条, 两张卡各 1 条 run, 不在单卡上叠两条 run.
- 每条 run 需要输出 final/best hard, best-final gap, telemetry 曲线和 source manifest.

阶段 2 统一训练口径:

```text
d_model=128
num_codebook_vectors=64
fox_gd_residual_rank=16
data_seed=123
read_topk=2
write_topk=4
mu_min_count=0.1
beta_init=0.5
lambda_init=0.05
vq_score_mode=codebook_dot
vq_weight_mode=dense_softmax
vq_update_mode=grad
vq_softmax_tau=0.25
train_batch_size=64
eval_batch_size=16
gradient_accumulation_steps=4
max_epochs=4
validations_per_epoch=4
disable_early_stopping=true
read_churn_probe_enabled=true
read_churn_probe_valid_batches=441
read_trace_enabled=false
```

启动后退出会话的条件:

```text
1. 3090 GPU0 有训练显存占用, active run 数不超过 3.
2. 2080ti GPU0 和 GPU1 都有训练显存占用, 且每卡只有 1 条 active run.
3. active run 日志已越过配置生成和数据加载阶段, 进入训练循环.
4. queue 主进程和 active 子进程仍在.
5. 日志没有 Traceback, CUDA out of memory, ValidationError, nan 或 inf.
6. 观察至少 10 分钟, 但 10 分钟不是唯一退出条件.
```

## 阶段 3: 判读规则

根据 telemetry probe 决定后续 guard 设计:

| 观察 | 解释 | 后续 |
|---|---|---|
| `m_norm` 先过线, 且与 final 失败同步 | state norm overrun 是主风险 | guard 看 `m_norm` 和 release hold |
| `m_norm` 不过线, 但 cap-hit, update_norm 或 uncapped write pressure 高 | write/update pressure 是主风险 | guard 看 write/update pressure |
| `lambda/inject_ratio` 高, 但 update/m_norm 不高 | residual readout 过强 | guard 需要限制 lambda/injection |
| read margin/selected mass 变差早于 hard collapse | read-side 不确定性参与失败 | 不能只做 write guard, 后续接 read gate |
| 所有 pressure 指标都不异常 | 当前 telemetry 仍不足 | 先补更细 per-code/per-head 或 fixed-sample trace |

## 阶段 4: 最小 pressure-aware cap release, 仅在阶段 2/3 后执行

第一版 guard 只控制 write cap release:

```text
base cap = 0.04
scheduled target cap = 0.06
健康: 按 schedule release
不健康: hold 当前 cap
第一版不做 rollback
```

第一版 guard 候选指标由阶段 2/3 的结果决定. 默认优先顺序:

```text
1. write/update pressure
2. m_norm health
3. lambda/inject ratio
4. read uncertainty, only if telemetry supports it
```

验收门槛:

```text
cb64-r16 seeds 123/124/125 spread <= 0.03
best-final gap <= 0.01
good-seed ceiling tax < hard04
m_norm_max < 8 优先, >12 不进 official
repeat gap <= 0.01 before official
```

## 产物位置

若执行本计划:

```text
script: zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/
artifact: docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/
report: docs/20260624-02-flash-vqg-pressure-telemetry-guard-report.md
```

## 当前决策

当前执行第二阶段: 最小 telemetry probe 启动和稳定性观察. 只有 telemetry probe 证明哪个 pressure 信号先出问题后, 才实现 pressure-aware guard.
