# Flash-VQG readk telemetry formalization 报告

updated: 2026-06-22

## 摘要

本轮在 3090 上完成三条 targeted fixed `read_topk=4` 训练, 目的是补 `20260622-01` 历史 readk 边界审计中缺失的 seed123/formalization 证据, 并验证新增 read-side telemetry 是否能从模型侧写入训练日志.

三条 run 都 completed, 没有 traceback, OOM 或 runtime error. `cb256-r8 readk4 s123` final hard=`0.992`, `cb256-r4 readk4 s123` final hard=`0.965`, 都补上了 cb256-like fixed readk4 局部正例中的 seed123 缺口. `cb128-r8 readk4 s125 repeat` final hard=`0.967`, 这次没有复现历史 `0.609` collapse, 但 final `m_norm_max=13.8`, 超过之前 `m_norm>12` 的 no-official redline, 所以它仍应保留为边界/风险样例, 不能升级为 clean positive.

本轮最重要的工程结果是: 新增 read telemetry 已经进入 validation logs 和 artifact, 包括 read margin, read entropy 和 selected mass. 但 candidate churn 仍未实现.

## 代码与来源

- plan: `docs/plans/20260622-02-flash-vqg-readk-telemetry-formalization-plan.md`
- artifact: `docs/artifacts/20260622-02-flash-vqg-readk-telemetry-formalization/`
- scripts: `zoology/experiments/flash_vqg/scripts/20260622-02-flash-vqg-readk-telemetry-formalization/`
- raw logs: source on 3090 `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260622-02-flash-vqg-readk-telemetry-formalization/outputs/logs/20260622T041508Z/`; mirrored to 2080ti at the same container path and sha256 verified.
- generated manifest/config: source on 3090 under `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated/flash-vqg-20260622-02-readk-tel-*/`; mirrored to 2080ti at the same container paths and sha256 verified.
- zoology branch/head: `flash-vqg`, `5d06e5646fce1ac50cee43cc157f6356a5c194c9`
- Flash-VQG branch/head: `20260428-gd-residual-v1-sync`, `4d02c71ee6d19228f8104cc9844042f398d44f86`

本轮新增或验证的 telemetry:

```text
valid/attn/gd_residual_read_margin_top1_top2_mean
valid/attn/gd_residual_read_margin_top1_top2_p05
valid/attn/gd_residual_read_entropy_mean
valid/attn/gd_residual_read_selected_mass_mean
valid/attn/gd_residual_read_selected_mass_p05
valid/attn/gd_residual_remote_read_topk_effective
```

## 训练矩阵

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

| target | seed | codebook | rank | 目的 |
|---|---:|---:|---:|---|
| `cb256r8-readk4-s123` | 123 | 256 | 8 | 补齐 cb256-r8 fixed readk4 正例 seed123. |
| `cb256r4-readk4-s123` | 123 | 256 | 4 | 补齐 cb256-r4 fixed readk4 正例 seed123. |
| `cb128r8-readk4-s125-repeat` | 125 | 128 | 8 | 复核 cb128-r8 readk4 s125 rerun collapse 风险. |

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`.

| target | hard curve | best hard | final hard | valid acc | best-final gap |
|---|---|---:|---:|---:|---:|
| `cb256r8-readk4-s123` | `0.000734/0.964/0.982/0.983/0.988/0.989/0.992/0.992` | 0.992 | 0.992 | 0.997 | 0.000 |
| `cb256r4-readk4-s123` | `0.000191/0.892/0.938/0.953/0.959/0.963/0.965/0.965` | 0.965 | 0.965 | 0.993 | 0.000 |
| `cb128r8-readk4-s125-repeat` | `0.000266/0.913/0.932/0.946/0.950/0.963/0.966/0.967` | 0.967 | 0.967 | 0.993 | 0.000 |

Telemetry final:

| target | read margin mean | read entropy mean | selected mass mean | lambda mean | inject ratio | final `m_norm_max` | max `m_norm_max` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cb256r8-readk4-s123` | 0.650 | 1.270 | 0.283 | 0.122 | 0.164 | 1.85 | 1.86 |
| `cb256r4-readk4-s123` | 0.552 | 1.150 | 0.304 | 0.253 | 0.203 | 0.376 | 1.40 |
| `cb128r8-readk4-s125-repeat` | 0.959 | 0.747 | 0.376 | 0.297 | 0.0868 | 13.8 | 13.8 |

`remote_read_topk_effective=4` 在三条 final validation 中都被记录到. 这说明 fixed readk4 control 和新 telemetry 都在 runtime 生效.

## 与历史 readk 边界审计的关系

本轮应和 `docs/artifacts/20260622-01-flash-vqg-readk-boundary-audit/` 一起解释.

`cb256-r8` 的历史 fixed readk4 completed rows 是:

```text
s124-r1=0.982
s124-r2=0.982
s125-r1=0.988
s125-r2=0.992
```

历史表缺 seed123. 本轮 `s123=0.992`, 说明 `cb256-r8` fixed readk4 继续保持 high path, 也补齐了最强正例中缺失的 seed123. 这支持把 `cb256-r8 + fixed readk4` 作为局部 candidate, 但仍不是全局默认.

`cb256-r4` 的历史 fixed readk4 completed rows 是:

```text
s124-r1c=0.943
s124-r2c=0.958
s125-r1c=0.944
```

历史表也缺 seed123. 本轮 `s123=0.965`, 同样补齐 seed123 并走 high path. 它支持 `cb256-r4` 也是 fixed readk4 局部正例.

`cb128-r8` 的历史 fixed readk4 rows 是:

```text
s124-r1=0.973
s125-r1=0.972
s125-r2=0.609
```

本轮追加 `s125 repeat=0.967`, 没有复现 collapse. 这削弱了“s125 必然 collapse”的说法, 但不能推翻 “cb128-r8/readk4 有 rerun instability 风险” 这个边界判断. 原因有两个: 第一, 既有 `0.609` 仍是真实 completed run; 第二, 本轮高分 repeat 的 `m_norm_max=13.8`, 明显超过之前用于 official 过滤的 redline, 不适合当 clean stable case.

## 机制观察

这轮不是 late drift. 三条 run 的 best-final gap 都是 `0`, hard curve 从初始低点后整体上升并保持 final 高点.

这轮也不是 cb256 上的 state norm explosion. `cb256-r8` max `m_norm_max=1.86`, `cb256-r4` max `m_norm_max=1.40`, 都远低于 redline. 这和它们 high path 的 final 表现一致.

`cb128-r8` 比较特殊: final hard 高, read margin mean 也高, read entropy 更低, 但 `m_norm_max=13.8`. 这说明 high final accuracy 不足以证明它是健康稳定解. 对 cb128-r8, readk4 可能可以在某些 run 走高分路径, 但 state health 仍提示边界敏感或状态集中风险.

新增 read telemetry 的信息量初步可用, 但还不够支撑完整机制判别:

- `read_margin_top1_top2_mean`: 能显示 read candidate 的平均分离度.
- `read_entropy_mean`: 能显示 read 分布是否尖锐.
- `read_selected_mass_mean`: 能显示 selected topk 覆盖了多少概率质量.
- 仍缺 candidate churn, per-head/per-code read usage, early train-step snapshots.

## 决策

1. fixed readk4 仍然不能作为 global default. `cb64-r16` high-path damage 和 `cb128-r8` historical collapse 仍然有效.
2. `cb256-r8` 和 `cb256-r4` 的 fixed readk4 局部正例更强了, 因为本轮补上的 seed123 都是 high path.
3. `cb128-r8 readk4` 不能再只写成 “s125 rerun 必然 collapse”; 更准确的说法是 “存在已观察到的 rerun collapse, 本轮 repeat 未复现, 但 state health 仍不达 official redline.”
4. 后续如果要把 cb256-like fixed readk4 做成正式候选, 应整理 combined historical+new 三 seed 表, 但要明确不是 same-wave full matrix.
5. 后续机制方向仍应是 margin-aware read gate 或 read/write confidence gate, 而不是把 fixed readk4 写进全局默认.

## 下一步

建议下一步分两条线:

1. **整理 combined readk4 formal table.** 把 `20260622-01` 历史 rows 和本轮 seed123 rows 合并成 cb256-r4/r8 的三 seed candidate table, 但显式标记 historical rows 缺少 read telemetry.
2. **实现 candidate churn / early probe.** 在 step `0,130,203,352,353,448,705` 或 validation 前后记录 read margin, entropy, selected mass, candidate churn, lambda/inject ratio, zeta 和 M norm. 没有 churn 和 early step, 仍然很难判断 bad candidate lock-in 是否发生在 final 前.

不建议立刻做大规模新训练. 这轮已经说明 targeted seed123 缺口被补齐, 更高价值的下一步是把 telemetry 做完整, 再进入 margin-aware gate.
