# 20260624-02 Flash-VQG pressure telemetry guard 阶段报告

updated: 2026-06-24
experiment_id: `20260624-02-flash-vqg-pressure-telemetry-guard`
status: stage-2-completed

## 摘要

本报告覆盖两个阶段:

- 阶段 1 完成 telemetry 补齐和 config-to-runtime smoke, 确认 update norm, update cap hit, write cap effective/scheduled value, release progress 以及已有 write/read/state 指标能从 runtime 传出.
- 阶段 2 完成 `cb64-r16` 最小 telemetry probe, 在 `s123` 和 `s124` 上观察 `default`, `hard04`, `cap0405`, `caprel0406late` 的 pressure 曲线.

本实验没有实现 guarded release, 也不是 official MQAR 结果. 它的目标是先判断 guard 实现前应该重点防什么.

## 代码变更

Flash-VQG:

- `gd_residual.py` 新增 `update_norm_mean/p95/max` 和 `update_norm_cap_hit_ratio`.
- update norm 记录的是 cap 之前的 `abs(zeta) * ||err||`, 用于判断原始 update pressure 是否越界.
- token-step 和 grouped-chunk 两条 state build 路径都接入同一组 telemetry.
- `attn.py` 新增 write cap schedule telemetry: `write_strength_scheduled_cap` 和 `write_strength_cap_release_progress`.

zoology:

- metrics whitelist 增加新增 pressure telemetry.
- 新增 config-to-runtime smoke 脚本, 覆盖 `hard04`, cap release progress, `update_norm_cap`, update cap hit 四个 case.
- 沿用 plan: `docs/plans/20260624-02-flash-vqg-pressure-telemetry-guard-plan.md`.

## 双机 smoke 结果

| machine | status | device | torch |
|---|---|---|---|
| 2080ti | passed | NVIDIA GeForce RTX 2080 Ti | 2.6.0+cu118 |
| 3090 | passed | NVIDIA GeForce RTX 3090 | 2.6.0+cu118 |

核心 case:

| case | 检查内容 | 结果 |
|---|---|---|
| `hard04` | `effective_cap=0.04`, `scheduled_cap=0.04`, release progress `0` | passed |
| `caprel0406late-progress` | 3 次 forward 后 release progress `0.5`, cap `0.05` | passed |
| `update-norm-cap` | `update_norm_cap_active=1`, effective cap `0.02` | passed |
| `update-norm-cap-hit` | 低 cap `0.001` 触发 `update_norm_cap_hit_ratio > 0` | passed |

详细结果见 `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/`.

## 验证

已执行:

```bash
python -m py_compile src/flash_vqg/nn/fox/gd_residual.py src/flash_vqg/nn/attn.py
python -m py_compile zoology/experiments/flash_vqg/metrics_white_list.py
python -m py_compile zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/config_runtime_smoke.py
/home/lyj/miniconda3/envs/flash-vqg/bin/python -m pytest tests/test_fox_gd_residual_v1.py -q
/home/lyj/miniconda3/envs/flash-vqg/bin/python -m pytest tests/test_fox_phase2_metrics.py -q
bash zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/run_config_runtime_smoke.sh --device cuda
```

## 阶段 1 判定

第一阶段通过, 可以进入阶段 2: 最小 telemetry probe.

阶段 2 不应直接实现 guard. 应先在 `cb64-r16` 的 `hard04`, `caprel0406late`, `cap0405` 小矩阵上跑短/完整可比 telemetry, 看失败先出现在 update pressure, cap-hit, m_norm, lambda/inject, 还是 read-side 指标.

## 阶段 2 启动方案

本阶段沿用同一个 experiment_id, 不新建目录. 新增 launcher 和 metrics:

```text
zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/metrics.yaml
zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/run_stage2_probe_train.sh
zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/run_stage2_probe_queue.sh
zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/start_stage2_probe_queue.sh
```

矩阵:

| machine | targets | 并发 |
|---|---|---|
| 3090 | `default-s123`, `hard04-s123`, `cap0405-s123`, `caprel0406late-s123` | 单卡最多 3 条 |
| 2080ti | `default-s124`, `hard04-s124`, `cap0405-s124`, `caprel0406late-s124` | 两张卡各 1 条 |

release 配置统一使用 `write_strength_cap_eval_policy=scheduled`. 这和部分历史 caprel 口径不完全相同, 但更适合定位 release 前后 pressure 曲线.

观察退出条件: 至少观察 10 分钟, 且 3090 GPU0, 2080ti GPU0, 2080ti GPU1 都已经进入训练状态; 日志没有 `Traceback`, `CUDA out of memory`, `ValidationError`, `nan`, `inf`.

## 阶段 2 启动修正

首次启动尝试在配置生成后进入 checkpoint 目录创建阶段失败, 原因是 launcher 没有传入实验专用 `config-builder`, 导致通用 run id 生成路径过长. 该尝试没有形成有效训练结果, outputs 只保留为 ignored raw log.

已修正:

- `run_stage2_probe_train.sh` 现在显式使用 `20260425-gd-residual-v1-mqar/config_builder.py:build_gd_residual_v1_train_configs`.
- `launch_id_prefix` 和 `run_id` 改为短名, 避免 checkpoint 路径过长.
- queue status 的完成行记录实际子进程 pid, 方便后续审计.

## 阶段 2 收尾结果

完成 8 条有效训练:

| machine | seed | targets |
|---|---:|---|
| 3090 | 123 | `default`, `hard04`, `cap0405`, `caprel0406late` |
| 2080ti | 124 | `default`, `hard04`, `cap0405`, `caprel0406late` |

2080ti 上 `cap0405-s124` 首次尝试在 `2026-06-24T08:40:26+08:00` 启动后 OOM, 原因是 GPU0 上 `default-s124` 尚未结束, 两条 run 重叠占用同一张 2080Ti. 该失败 run 已标记为 excluded, 后续 `cap0405-s124-rerun` 单独运行完成.

关键结果:

| variant | s123 final / best | s124 final / best | two-seed final spread | max `m_norm_max` | max final `update_norm_p95` |
|---|---:|---:|---:|---:|---:|
| `default` | `0.776699 / 0.776699` | `0.963559 / 0.963559` | `0.186859` | `7.122381` | `0.706640` |
| `hard04` | `0.852586 / 0.874480` | `0.965418 / 0.965418` | `0.112832` | `4.108659` | `0.234283` |
| `cap0405` | `0.812492 / 0.821961` | `0.965113 / 0.965859` | `0.152621` | `4.157090` | `0.303398` |
| `caprel0406late` | `0.821625 / 0.839250` | `0.965742 / 0.965742` | `0.144117` | `4.284737` | `0.350590` |

详细 CSV:

- `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/stage2-key-metrics.csv`
- `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/stage2-run-summary.csv`
- `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/stage2-variant-summary.csv`
- `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/stage2-invalid-runs.csv`
- `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/stage2-source-manifest.csv`

## 阶段 2 判断

第一, pressure 控制确实改变了低 seed 的轨迹. 在 `s123`, `hard04` 把 final hard 从 `0.776699` 提到 `0.852586`, 同时把 `m_norm_max` 从 `7.122381` 压到 `3.451890`, 把 final `update_norm_p95` 从 `0.706640` 压到 `0.234283`.

第二, 这还不是一个足够的稳定方案. `hard04` 是 `s123` 本轮最好配置, 但 final 仍只有 `0.852586`, best-final gap 为 `0.021895`. `cap0405` 和 `caprel0406late` 没有把 `s123` 拉到高位, final 分别是 `0.812492` 和 `0.821625`.

第三, 当前失败不能简化成 “只防 `m_norm` 爆”. `s123 default` 是最差 run, 但 `m_norm_max=7.122381`, 没过 `m_norm_max > 12` 红线, 也低于 roadmap 中 `8` 的高风险阈值. 所以后续 guard 如果只看 `m_norm_max`, 很可能放过这类低盆地.

第四, release 本身在本轮没有触发 state 红线. `cap0405` 和 `caprel0406late` 的 two-seed 最大 `m_norm_max` 分别是 `4.157090` 和 `4.284737`; 这说明更保守的 release 在本轮是 state-safe 的. 但它没有解决 `s123` 的低盆地问题.

第五, read 指标不能单独解释本轮结果. 例如 `default-s123` final `read_margin_mean=1.218576`, 反而高于 `default-s124` 的 `0.700348`; capped runs 中 `s123` 的 read entropy 更高, selected mass 更低. 这说明 read-side 仍需要早期窗口和 per-step 对齐分析, 不能只靠 final aggregate 断言根因.

## 已定位的候选控制信号

本轮已经把下一步需要观察和控制的信号范围收窄了, 但还没有得到可以直接实现的最终 guard 规则.

已经确认有价值的信号:

| 信号 | 本轮证据 | 当前定位 |
|---|---|---|
| `update_norm_p95/max` | `s123 default` final `update_norm_p95=0.706640`, `hard04` 后降到 `0.234283`, 同时 hard acc 从 `0.776699` 提到 `0.852586`. | 这是本轮最有价值的新 pressure 指标之一, 应作为 guard 主观测量. |
| `m_norm_max` / `m_norm` slope | `hard04` 能把 `s123` 的 `m_norm_max` 从 `7.122381` 压到 `3.451890`. 但最差 run 的 `m_norm_max` 也没过 `8/12` 红线. | 适合防 state 过冲, 但不能单独识别所有低盆地. |
| `uncapped_write_strength`, `uncapped_sum_zeta`, write cap hit ratio | capped run 能显示 cap 是否持续压制原始 write pressure; `hard04` 和 release 配置都留下了 cap hit telemetry. | 用来判断 cap 是偶发保护, 还是长期在救火; 后者可能对应 ceiling tax 或早期写入边界. |
| `lambda_mean`, `inject_ratio` | 本轮没有证明它们单独决定坏 seed, 但它们描述 residual read 注入强度. | 作为 residual amplification 的配套监控项, 需要和 write/read 指标一起看. |
| read margin, entropy, selected mass | final aggregate 不能单独解释结果; `default-s123` 的 final read margin 反而高于 `default-s124`, 但 capped runs 中 `s123` entropy 更高, selected mass 更低. | 不能只看 final 平均值, 需要 early-window 和 per-step 对齐. |

尚未找到的东西是一个可直接落地的规则, 例如:

```text
if m_norm > X:
    hold release
```

当前证据反而说明这种单指标规则太粗. 坏 seed 不一定表现为 `m_norm` 爆炸; 它可能在早期 write/update pressure, residual injection 和 read-side confidence 的组合边界上进入低盆地.

因此下一步 guard 设计应先按组合信号分析:

```text
state health:
    m_norm_max, m_norm slope

write/update pressure:
    update_norm_p95/max, uncapped_sum_zeta, uncapped_write_strength, cap hit ratio

residual injection:
    lambda_mean, inject_ratio

read-side confidence:
    read_margin, read_entropy, read_selected_mass
```

只有 early-window trace 证明这些信号能提前区分低盆地, 才应该把它们写成 guarded cap release 或 read/write gate.

## 限制

本轮是最小 telemetry probe, 不是正式稳定性验证. 它只有 `s123` 和 `s124`, 且两个 seed 分别跑在不同机器上: `s123` 在 3090, `s124` 在 2080ti. 之前 smoke 已确认两边 torch/CUDA 链路一致, 但这仍然不是 cross-machine matched repeat. 因此本轮可以用来判断 pressure 指标是否有信号, 不能用来给出最终 seed spread 结论.

`cap0405-s124` 首次 OOM 是调度问题, 不是模型配置失败. 报告和 artifact 只使用单独 rerun 的完成结果.

## 下一步

不要直接实现复杂 guard. 先做两个收紧动作:

1. 对 `s123` 做 early-window pressure/read trace, 对齐 step `130`, `203`, `352`, `448` 附近的 `m_norm`, `update_norm`, write cap hit, lambda/inject, read margin/entropy/selected mass. 目标是找出低盆地是在 pressure 降下来之后仍然形成, 还是早期已被写入锁定.
2. 补一个最小 seed/machine 复核: 至少补 `s125` 的同矩阵, 或者把 `s123`/`s124` 做 cross-machine repeat. 目标是拆开 seed 效应和机器效应.

guard 设计上, 当前证据支持 “不能只看 `m_norm`”. 更合理的候选应至少同时观察:

- state health: `m_norm_max`, `m_norm` slope.
- write/update pressure: `update_norm_p95/max`, `uncapped_sum_zeta`, write cap hit ratio.
- residual injection: `lambda_mean`, `inject_ratio`.
- read-side aggregate: read margin, entropy, selected mass.

只有在 early-window trace 证明这些指标能提前区分低盆地后, 再实现 guarded cap release.
