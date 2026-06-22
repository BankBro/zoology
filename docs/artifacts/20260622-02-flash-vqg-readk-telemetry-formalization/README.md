# Flash-VQG readk telemetry formalization artifact

## 基本信息

- Artifact ID: `20260622-02-flash-vqg-readk-telemetry-formalization`
- 创建日期: 2026-06-22
- 类型: targeted fixed readk4 formalization / telemetry run
- 机器: `mclab-3090` 的 `Flash-VQG-tun` 容器
- 主指标: `valid/mqar_case/accuracy-1024x256`
- 代码版本:
  - zoology: `flash-vqg`, `5d06e5646fce1ac50cee43cc157f6356a5c194c9`
  - Flash-VQG: `20260428-gd-residual-v1-sync`, `4d02c71ee6d19228f8104cc9844042f398d44f86`

## 文件说明

| 文件 | 说明 |
|---|---|
| `final.csv` | 三条 targeted run 的 final/best 指标和 final telemetry. |
| `validation-history.csv` | 每条 run 去重后的 validation history, 包含 hard curve 和 read/write/state telemetry. |
| `spread-summary.csv` | 本轮三条 run 的 config-level 摘要. 注意每个 config 只有一个新 seed/repeat. |
| `source-manifest.csv` | raw log, generated manifest, checkpoint, local mirror 和 SwanLab URL 索引. |
| `readk4-context-summary.csv` | 本轮结果与 `20260622-01` 历史 readk4 审计的连接说明. |
| `metadata.json` | 机器, 配置, 结论和 caveat 元数据. |
| `README.md` | 本说明文件. |

## 本轮目标

这轮不是重跑完整 readk2/readk4 矩阵, 而是补 `20260622-01` 历史审计里缺的 targeted 证据:

1. `cb256-r8 readk4 s123`, 补齐 cb256-r8 fixed readk4 正例的 seed123 缺口.
2. `cb256-r4 readk4 s123`, 补齐 cb256-r4 fixed readk4 正例的 seed123 缺口.
3. `cb128-r8 readk4 s125 repeat`, 复核历史 `cb128-r8 readk4 s125` rerun collapse 风险, 并带上新增 read-side telemetry.

同时验证新增 telemetry 已经从模型侧传出:

- `valid/attn/gd_residual_read_margin_top1_top2_mean`
- `valid/attn/gd_residual_read_margin_top1_top2_p05`
- `valid/attn/gd_residual_read_entropy_mean`
- `valid/attn/gd_residual_read_selected_mass_mean`
- `valid/attn/gd_residual_read_selected_mass_p05`

## 训练配置

共同配置:

```text
data_seed=123
d_model=128
max_epochs=4
validations_per_epoch=2
train_batch_size=64
eval_batch_size=16
gradient_accumulation_steps=4
fox_remote_formula=gd_residual_v1
fox_remote_read_topk=4
fox_gd_residual_write_topk=4
write_cap=None
beta_control=hard_cap
logger_backend=swanlab
```

## 结果摘要

| target | final hard | best hard | best-final gap | final `m_norm_max` | final read margin | final read entropy | final selected mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cb256r8-readk4-s123` | 0.992 | 0.992 | 0.000 | 1.85 | 0.650 | 1.270 | 0.283 |
| `cb256r4-readk4-s123` | 0.965 | 0.965 | 0.000 | 0.376 | 0.552 | 1.150 | 0.304 |
| `cb128r8-readk4-s125-repeat` | 0.967 | 0.967 | 0.000 | 13.8 | 0.959 | 0.747 | 0.376 |

三条 run 都正常完成, `log_error_count=0`, 且 validation stdout 中 `remote_read_topk_effective=4`.

## 与历史审计的关系

- `cb256-r8`: 历史 readk4 completed rows 是 `0.982/0.982/0.988/0.992`, 但缺 seed123. 本轮 seed123=`0.992`, 进一步支持 cb256-r8 fixed readk4 是强局部正例.
- `cb256-r4`: 历史 readk4 completed rows 是 `0.943/0.958/0.944`, 但缺 seed123. 本轮 seed123=`0.965`, 进一步支持 cb256-r4 fixed readk4 是局部正例.
- `cb128-r8`: 历史 readk4 有 s125 rerun collapse=`0.609`. 本轮 s125 repeat=`0.967`, 没有复现 collapse, 但 final `m_norm_max=13.8` 超过之前 `m_norm>12` 的 no-official redline, 因此只能说 collapse 频率估计被削弱, 不能把 cb128-r8 升级为干净正例.

## 结论

1. 本轮补齐了 cb256-r4/r8 fixed readk4 seed123 缺口, 两者都走 high path.
2. 新增 read-side telemetry 已经可以在训练日志中稳定记录, 可用于后续 report 和机制诊断.
3. `cb128-r8 readk4` 的历史 collapse 这次没有复现, 但高 `m_norm_max` 保留了边界风险信号.
4. fixed readk4 仍应写成 cb256-like 的局部候选, 不是全局默认.

## 注意事项

- 本轮没有 same-wave readk2 control, 只能与历史 readk2/readk4 rows 做背景对比.
- 本轮每个 config 只补一个新 seed/repeat, 不是完整三 seed same-wave formal matrix.
- 历史 rows 没有本轮新增 telemetry, 因此 combined conclusion 只能混合 final accuracy 和本轮 telemetry.
- candidate churn 仍未实现.
- 3090 产生的 stdout log 和 generated manifest/config 已镜像回 2080ti 的 ignored workspace, 并完成 sha256 校验.
- checkpoint, SwanLab backup 和大型 raw history 仍保留在 3090 原位, 不进入 git artifact.
