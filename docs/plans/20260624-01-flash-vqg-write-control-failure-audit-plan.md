# 20260624-01 Flash-VQG write-control 失败机制审计计划

updated: 2026-06-24
status: implemented plan
experiment_id: `20260624-01-flash-vqg-write-control-failure-audit`

## 目标

本轮只审计已有 `cb64-r16` write-control 历史实验, 不启动新训练, 不实现新的 guarded release 机制. 目标是回答:

```text
旧 write-control 方案失败时, 主要是 m_norm 过冲, write pressure 过强, lambda/readout 过强, 还是 read-side 轨迹问题?
```

## 范围

主审计对象:

- `default`: seeds `123/124/125`.
- `hard04`: `write_strength_cap=0.04`, seeds `123/124/125`.
- `caprel0406late`: `write_strength_cap=0.04 -> 0.06`, seeds `123/124/125`.
- `cap0405`: `write_strength_cap=0.04 -> 0.05`, seeds `123/124`.
- `cap0405_beta0p16`: `0.04 -> 0.05` with `beta_init=0.16`, seeds `123/124`.
- `cap0406_mcap8`: `0.04 -> 0.06` with `m_norm_cap=8`, seeds `123/124`.

本轮不把 `cb256-r8` 或 `cb128-r8` 加入新训练矩阵.

## 产物

- script: `zoology/experiments/flash_vqg/scripts/20260624-01-flash-vqg-write-control-failure-audit/`
- artifact: `docs/artifacts/20260624-01-flash-vqg-write-control-failure-audit/`
- report: `docs/20260624-01-flash-vqg-write-control-failure-audit-report.md`

artifact 至少包含:

- `write_control_final_summary.csv`
- `write_control_setting_summary.csv`
- `write_control_step_curves.csv`
- `failure_taxonomy.csv`
- `missing_metrics.csv`
- `source_manifest.csv`
- `metadata.json`
- `README.md`

## 判读规则

- `hard04` 若 spread 小但 good seed 下降, 标为 `stable_ceiling_tax`.
- `caprel0406late` 若 spread 小但 `m_norm_max > 12`, 标为 `state_overrun`.
- `cap0405` 若 `m_norm_max < 8` 但 seed123 final 低, 标为 `late_drift_without_mnorm_overrun`.
- `cap0406_mcap8` 若 `m_norm_cap_hit_ratio` 缺失或为 0 且 spread 仍大, 标为 `ineffective_mnorm_cap`.
- `m_norm_max > 8` 仅作为警戒线, `>12` 作为原则性红线.

## 验收

审计脚本的 `--check` 必须复核这些锚点:

- default hard: `0.968711 / 0.819797 / 0.987285`.
- hard04 hard: `0.945039 / 0.963055 / 0.952605`.
- caprel0406late hard: `0.949371 / 0.963004 / 0.960484`.
- caprel0406late s123 `m_norm_max ~= 14.487579`.
- cap0405 s123/s124 hard 约 `0.811 / 0.960`, 且 s123 `m_norm_max < 8`.
