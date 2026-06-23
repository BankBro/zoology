# 20260624-02 Flash-VQG pressure telemetry guard 第一阶段报告

updated: 2026-06-24
experiment_id: `20260624-02-flash-vqg-pressure-telemetry-guard`
status: stage-1-smoke-passed

## 摘要

本阶段完成 telemetry 补齐和 config-to-runtime smoke. 没有启动完整 MQAR 训练, 没有实现 guarded release.

目标是把后续判断 guard 应该防什么所需的观测链路先打通: update norm, update cap hit, write cap effective/scheduled value, release progress, 以及已有 write/read/state 指标.

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

## 判定

第一阶段通过. 现在可以进入阶段 2: 最小 telemetry probe.

阶段 2 不应直接实现 guard. 应先在 `cb64-r16` 的 `hard04`, `caprel0406late`, `cap0405` 小矩阵上跑短/完整可比 telemetry, 看失败先出现在 update pressure, cap-hit, m_norm, lambda/inject, 还是 read-side 指标.
