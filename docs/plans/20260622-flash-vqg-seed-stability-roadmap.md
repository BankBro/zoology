# Flash-VQG seed 稳定性研究路线草案

updated: 2026-06-22
status: roadmap draft
branch: `flash-vqg`

## 1. 背景与定位

本文档用于沉淀 Flash-VQG `gd_residual_v1` seed 效果波动问题的下一轮研究路线. 它基于已有 2080ti 和 3090 实验报告, artifact, 代码入口复核, 以及 2026-06-22 对网页版 ChatGPT 回复和 subagent 交叉审计的整理.

当前目标不是宣布单一根因, 也不是直接给出最终训练方法. 更合适的定位是:

```text
把已验证事实, 当前最强工作假设, 工程 readiness, 下一轮最小可执行实验和 official 准入门槛写成可复查路线图.
```

本路线只作为 plan. 生成本文档不代表开始新实验, 不创建新 artifact, 不写 canonical ledger.

## 2. 当前可支持结论

### 2.1 seed instability 是真实问题

固定 `data_seed=123` 时, 同一 MQAR hard slice 会因 model seed 或训练路径进入不同 basin.

| 配置 | s123 | s124 | s125 | 说明 |
|---|---:|---:|---:|---|
| `cb256-r4` strict official | `0.895023` | `0.675371` | `0.834781` | 同一容量轴下 hard accuracy spread 明显. |
| `cb64-r16` strict official | `0.968711` | `0.819797` | `0.987285` | s124 弱, s125 强, 不是单点异常. |

这些结果说明问题不是单个 bad run, 而是 seed/path 级别的不稳定.

### 2.2 单一 RNG/codebook bug 不是充分根因

`ScaleInitStrategy` 曾经存在 codebook 初始化依赖全局 RNG 顺序的问题. 这是真随机源, 会影响 codebook/head 的早期相对几何. 但 corrected runtime probe 显示, 单独 `codebook_init_seed` 不能稳定所有 seed:

```text
codebook_init_seed=seed 后 1ep step705:
s124=0.294
s125=0.626
```

因此, codebook RNG 修复应保留为复现和诊断工具, 但不能当作最终稳定方案.

### 2.3 fixed `read_topk=4` 是局部候选, 不是全局默认

`read_topk=4` 在 `cb256-r4` 和 `cb256-r8` 上有强局部正证据.

| 配置 | readk2 现象 | readk4 现象 | 当前定位 |
|---|---|---|---|
| `cb256-r4` | ordinary readk2 有明显 seed/path split. | formal readk4 completed runs `0.943/0.958/0.944`, spread `0.015`. | 局部 read-side 稳定候选. |
| `cb256-r8` | readk2 s124/s125=`0.988/0.804`, spread `0.184`. | readk4 completed runs `0.982/0.982/0.988/0.992`, spread `0.010`. | 当前 read-side 最强正证据. |
| `cb64-r16` | readk2 main s124/s125=`0.959/0.915`. | readk4 s124 repeat `0.831/0.849`, 伤 high path. | fixed readk4 反例. |
| `cb128-r8` | readk2 s124/s125=`0.956/0.956`. | readk4 main `0.973/0.972`, 但 s125 rerun=`0.609`. | fixed readk4 最大风险点. |

结论应写成: fixed readk4 在 cb256-like 配置上值得正式化, 但不能作为全局默认. 后续更可能需要 schedule/gate/margin-aware read-side 控制, 但这些机制目前还不是现成功能.

### 2.4 write trust-region 是强稳定基准, 但有 ceiling tax

`cb64-r16` 上的 write trust-region 证据最强.

| 配置 | s123 | s124 | s125 | spread |
|---|---:|---:|---:|---:|
| default | `0.968711` | `0.819797` | `0.987285` | `0.167488` |
| hard04 | `0.945039` | `0.963055` | `0.952605` | `0.018016` |
| caprel0406late | `0.949371` | `0.963004` | `0.960484` | `0.013633` |

`hard04` 把 bad seed 拉回稳定区间, 但把 good seed ceiling 压低:

```text
s123: 0.968711 -> 0.945039
s125: 0.987285 -> 0.952605
```

`caprel0406late` spread 更小, 但 s123 final `m_norm_max=14.487579`, 有明显 state 过冲风险. 因此 `hard04` 是 trust-region baseline 和诊断对照, 不是直接 official 主线. 下一步应测试更保守的 `0.04 -> 0.05` release 或 release + guard.

### 2.5 init transplant 是诊断工具, 不是训练方法

init transplant 显示 good flash-only donor 不能稳定救回 bad/boundary recipient.

| 实验 | hard |
|---|---:|
| normal `cb64-r16` s124 | `0.952305` |
| normal `cb64-r16` s125 | `0.981039` |
| flashdonor s125 -> s124 | `0.836082` |
| nonflashdonor s125 -> s124 | `0.661695` |
| normal `cb256-r4` s124 | `0.747195` |
| flashdonor `cb256-r4` s123 -> s124 | `0.679957` |

更精确的表述是: good flash-only init 或简单 step0 geometry 不是充分干预. 这不否定更复杂的联合初始化几何假设, 但说明 init transplant 和 `codebook_init_seed` 不应进入训练方法候选池.

## 3. 工作假设与证据等级

当前最强解释链应写成工作假设:

```text
codebook/address/projection 联合初始化几何
-> early routing/write 边界
-> zeta/M-state 早期差异
-> phase2 read_topk/lambda 注入放大
-> high/low basin selection
-> final 1024x256 分叉
```

这条链不是已证明因果定论. 不同环节的证据等级如下.

| 假设 | 当前证据等级 | 支持证据 | 主要不足 |
|---|---|---|---|
| write/state amplification 是主要 basin 分叉器 | 高 | hard04/cap 对 `cb64-r16` 的强干预; early `rho/zeta/M/lambda` 分叉. | 跨 `cb128-r8/cb256-r4/cb256-r8` transfer 证据不足. |
| read-side bad candidate lock-in | 中高 | `cb256-r4/r8` readk4 强正, `cb64-r16/cb128-r8` 明确反例. | 缺 read margin, entropy, candidate churn, coverage 直接 telemetry. |
| codebook/projection/address/beta/lambda 联合几何 | 中 | codebook seed 单独不够; init transplant 不支持单点救回. | simple init geometry scalar 不能区分 good/bad, 需要动态指标. |
| numerical path sensitivity 是 boundary trigger | 中 | pseudo-det 与 normal 在 step `130-203` 出现差异, step `353-448` 被放大. | 目前主要是 report-backed telemetry summary, 不是 final CSV 可直接复算. |
| capacity/layout 决定控制项边界 | 中高 | `cb64-r16`, `cb128-r8`, `cb256-r4`, `cb256-r8` 对 readk4 响应不同. | 还缺完整 matched seed/repeat 表. |
| RNG/init stream 是历史随机源 | 中 | `global/local_burn/local_noburn` 已实现, codebook RNG 明确曾有问题. | 单独固定 codebook seed 不能稳定, 不应作为主根因. |

## 4. failure taxonomy

后续 artifact 和报告不应只看 final hard spread, 必须标注 failure type.

| Failure type | 典型现象 | 需要记录 |
|---|---|---|
| `early_write_state_amplification` | first meaningful accuracy 前 `zeta/M/write_strength/lambda` 已分叉. | `write_strength`, uncapped zeta, beta, `M_norm`, update norm, cap hit ratio. |
| `read_side_lockin` | top2 过窄或 topk 过宽导致 early residual proposal 锁入坏路径. | read margin, read entropy, topk mass, candidate churn, selected code stability. |
| `late_drift` | best 高但 final 低, best-final gap 大. | per-validation best/final, late `m_norm` slope, lambda/inject ratio. |
| `rerun_instability` | 同 seed/config repeat 不一致. | repeat gap, GPU, dtype, branch commit, entrypoint, checkpoint hash. |
| `numerical_boundary` | 同 init 下 normal/pseudo/deterministic 早期微差被放大. | first divergence step, routing logits diff, write zeta diff, M diff. |
| `capacity_negative_transfer` | 同一控制项在不同 codebook/rank 上方向相反. | capacity/rank, active code count, per-code usage, final/best/repeat. |
| `config_runtime_drift` | 实验名或 manifest 写了某控制项, runtime 未真实生效. | resolved config, runtime effective metrics, requested vs clipped topk. |

## 5. 工程 readiness

### 5.1 已基本支持的静态控制

当前代码已经支持大部分静态控制实验:

- fixed `fox_remote_read_topk`: 支持正整数和 `dense/none/null`.
- GD residual 基础参数: rank, write_topk, builder, pack mode, chunk size, mu_min_count, addr/den/rho eps.
- write trust-region: `write_strength_cap`, cap mode, cap until, cap final, release start/end, eval policy.
- write budget/total cap: `write_budget`, `write_total_cap`, schedule, effective scale metrics.
- state cap: `m_norm_cap`, `update_norm_cap`.
- beta control: `hard_cap`, `bounded_sigmoid`, beta low/high/final, beta cap/final, release/eval policy.
- init RNG: `codebook_init_rng_mode`, `codebook_init_seed`, addr init RNG mode/seed.
- metrics collection: `FlashVQGMixer.get_scalar_metrics()` 经 `train.py` 收集并写入训练/验证日志.

### 5.2 尚未支持或需要补实现的功能

以下功能不应假定已经存在:

- `read_topk schedule`: 当前 fixed `int` 或 `None`, 没有 top4 -> top2 schedule.
- margin-aware read gate: 没有基于 margin/entropy/churn 动态改 topk 或 lambda 的逻辑.
- margin-aware lambda gate: 现有 lambda 是 learned sigmoid + floor/scale, 不结合 read-side uncertainty.
- guarded cap release: 现有 release 按 train forward count 无条件 schedule, `m_norm_cap` 是硬裁剪, 不是 release guard.
- runtime effective dump: manifest 记录 config kwargs, 不能证明 clipped topk, effective cap/beta, forward count.
- read-side 细粒度 telemetry: 当前 phase2 主要有 `gd_residual_lambda_mean` 和 `gd_residual_inject_ratio`, 不足以直接证明 read lock-in.
- failure tagging: 当前没有自动标注 `early_amp`, `read_lockin`, `late_drift`, `rerun_instability`.

### 5.3 工程风险

- `--fox-gd-residual-write-q-alpha` 和 `--fox-gd-residual-addr-proj-orthogonal-init` 在部分 generated config 路径可能存在漏传风险. 正式实验前必须用 config-to-runtime smoke 覆盖.
- schedule 计数基于 attention 内部 train forward count, 不是 optimizer step. gradient accumulation, validation frequency 和多层 attention 都会影响解释.
- `gd_residual_v1` 当前正式实验应显式使用 torch backend, 不要沿用不兼容的 flash/accel backend.
- metrics whitelist 可能影响新增 metrics 输出. 若启用窄 whitelist, 新 metric 需要同步注册.

## 6. 下一轮最小可执行实验清单

下一轮先做事实表和边界复核, 不先改 read gate 机制.

| 优先级 | 机器 | 实验 | 配置 | 目的 | 通过门槛 |
|---|---|---|---|---|---|
| P0 | 2080ti | config-to-runtime smoke | 短 smoke 覆盖 readk2/readk4, cap none/0.04, beta default/bounded, `write_q_alpha`, `addr_proj_orthogonal_init`. | 证明入口, generated config, manifest, runtime effective 一致. | 任一关键控制项不能闭环证明时, 暂停正式实验. |
| P0 | 3090 | `cb128-r8` readk4/readk2 rerun triage | `read_topk=2,4`, seeds `123/124/125`; `readk4 s125` 至少再 repeat 1 次, `readk2 worst` repeat 1 次. | 判断 cb128 是 readk4 真实不稳定, 偶发 rerun, 还是配置/环境问题. | readk4 repeat gap `>0.03` 时, fixed readk4 对 cb128 降级为 failure case. |
| P0 | 3090 | `cb256-r8` readk2 vs readk4 formalization | `read_topk=2,4`, seeds `123/124/125`; worst repeat 1 次. | 把 cb256-r8 readk4 正证据正式化. | spread `<=0.03`, repeat gap `<=0.01`, best-final gap `<=0.01`. |
| P1 | 3090 或空闲机器 | `cb256-r4` readk2 vs readk4 completion | `read_topk=2,4`, seeds `123/124/125`; worst repeat 1 次. | 补齐 cb256-r4 read-side 局部结论. | 同 cb256-r8 门槛. |
| P1 | 2080ti | `cb64-r16` conservative write trust-region | default, hard04, cap0405, guarded cap0405 release; seeds `123/124/125`; worst repeat 1 次. | 找比 hard04 ceiling tax 更低且不过冲的稳定写入控制. | spread `<=0.03`, ceiling tax 小于 hard04, `m_norm_max < 8` 优先. |

执行顺序:

1. 先完成 config-to-runtime smoke.
2. 同步跑 `cb128-r8` triage 和 `cb256-r8` formalization.
3. 根据 3090 空闲情况补 `cb256-r4` completion.
4. 2080ti 并行跑 conservative write trust-region.
5. 只有 P0/P1 通过门槛后, 再讨论 schedule/gate 实现或 official longer-MQAR.

## 7. 统一准入门槛

### 7.1 候选控制项门槛

控制项要从 diagnostic 升级为 candidate, 至少满足:

- seeds `123/124/125` 全覆盖.
- worst seed 或异常 seed 至少 repeat 1 次.
- final hard spread `<=0.03`.
- repeat gap `<=0.01`.
- best-final gap `<=0.01`.
- bad seed 有显著 rescue.
- good seed ceiling tax 小于 hard04, 或有明确机制收益.
- `m_norm_max > 8` 标红, `m_norm_max > 12` 原则上不进入 official.
- read-side 结果必须同时报告 cb256 正例和 cb64/cb128 反例, 不允许只收集正例.

### 7.2 official longer-MQAR 准入

official longer-MQAR 是验证层, 不是探索层. 进入前必须:

- 先通过 P0/P1 的 seed spread, repeat gap, best-final gap 和 state health 门槛.
- 记录 final checkpoint, source manifest, dtype policy, GPU, started/ended time, wall clock, status.
- 明确 `official`, `preliminary`, `exploratory`, `debug/smoke` 状态.
- 未完成到预期 final checkpoint 的 run 不写入 official ledger.

## 8. 暂缓或降级方向

下一轮暂缓:

- margin-aware read gate 实现.
- `read_topk` dense 或大范围 topk 扫描.
- large beta/BBSB sweep.
- full transplant 扩展.
- official longer-MQAR.
- 把 fixed readk4 写成全局默认.
- 把 hard04 写成最终主线.

下一轮只保留为 diagnostic 的方向:

- `codebook_init_seed`, preserve RNG, orthogonal addr_proj.
- static tau, static write_topk, topk_softmax.
- higher lambda 或更强 early injection.
- numerical normal/pseudo/deterministic replay.

## 9. 推荐产物位置

本文档:

```text
docs/plans/20260622-flash-vqg-seed-stability-roadmap.md
```

若后续启动实验, 再创建:

```text
zoology/experiments/flash_vqg/scripts/20260622-flash-vqg-seed-stability-next/
docs/artifacts/20260622-flash-vqg-seed-stability-next/
```

正式 artifact 至少包含:

- final CSV.
- source manifest CSV.
- metadata JSON.
- README.
- status JSON/CSV.
- 对失败, smoke, debug, 中断 run 的原因记录.

## 10. 引用材料

核心报告:

- `docs/20260528-flash-seed-stability-report.md`
- `docs/20260530-gd-seed-diag-report.md`
- `docs/20260603-gd-init-transplant-report.md`
- `docs/20260605-flash-vqg-stability-research-direction-report.md`
- `docs/20260605-flash-vqg-stability-direction-independent-review.md`

核心 artifact:

- `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv`
- `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-source-manifest.csv`
- `docs/artifacts/20260530-gd-seed-diag/final.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-key-metrics.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-spread-summary.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-final.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-spread-summary.csv`
- `docs/artifacts/20260603-gd-init-transplant/train-core-final.csv`
- `docs/artifacts/20260603-gd-init-transplant/train-core-matrix.csv`
- `docs/artifacts/20260603-gd-init-transplant/early-core-final.csv`
- `docs/artifacts/20260603-gd-init-transplant/init-geometry-audit.csv`
- `docs/artifacts/20260603-gd-init-transplant/init-geometry-probe.csv`

代码入口和实现:

- `zoology/experiments/flash_vqg/run_flash_vqg_suite.py`
- `zoology/experiments/flash_vqg/flash_vqg_suite.py`
- `zoology/experiments/flash_vqg/manifest.py`
- `zoology/mixers/flash_vqg.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/vq_init.py`
