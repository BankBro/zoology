# 20260709-01 Flash-VQG default dropout R16 support-confidence screen report

## 摘要

本轮在 default dropout 口径下, 固定 `cb64-r16`, `read_topk=16`, `write_topk=4`, `update_norm_softcap=0.5`, residual injection warmup `0->512` optimizer steps, 对 read-side support-confidence 控制做了 paired 1ep screen。

结论很直接: 新增 read confidence 控制没有超过当前 `baseline-r16-joint` 底座。`baseline-r16-joint` 是本轮唯一两个 seed paired run 都严格过线的配置, `s125` 为 `0.929/0.953`, gap `2.4pp`; `s124` 为 `0.902/0.939`, gap `3.7pp`。

`read-gate-r16` 两个 seed 都是高分, 但 gap 分别为 `4.1pp` 和 `4.3pp`, 略超 `4pp` 稳定线。`read-softmargin-r16` 和 `read-gate-softmargin-r16` 都出现了单机明显掉分, 不能作为当前默认候选。

## 固定条件

- 机器: `2080ti GPU1` + `3090 GPU0`.
- 训练: paired 1ep, `max_train_steps=704`, `grad_accumulation_steps=4`.
- 数据: same canonical MQAR cache, content hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- 初始化: same canonical init checkpoint, model state hash `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- batch order: hash `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.
- dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- 模型: `cb64-r16`, `read_topk=16`, `write_topk=4`.
- 稳定底座: `update_norm_softcap=0.5`, mode `smooth_p4`, residual injection warmup `0->512` optimizer steps.
- heavy read trace, hash probe, train inline event trace, D-geometry trace 均关闭。

所有 16 个 formal runs 都 completed, `error_count=0`, 无 NaN/OOM/Traceback。`best` 和 `final` 的 `1024x256` hard slice accuracy 在本轮所有 run 中一致。

## Variant 含义

| variant | 含义 |
|---|---|
| `baseline-r16-joint` | 当前强底座: `read_topk=16 + update_norm_softcap=0.5 + injection warmup 0->512` |
| `read-gate-r16` | 在底座上增加 read confidence gate. read support 低置信度时降低 residual 注入强度 |
| `read-softmargin-r16` | 在底座上增加 read softmargin. read top-k 候选分数接近时把 selected code 内部权重变平滑 |
| `read-gate-softmargin-r16` | 同时启用 read confidence gate 和 read softmargin |

read control 确实被启用: `read-gate` 的 final gate mean 约为 `0.537-0.649`; `read-softmargin` 的 final tau mean 约为 `2.37-2.52`; combined variant 的 gate mean 约为 `0.543-0.646`, tau mean 约为 `2.42-2.83`。

## 结果

Screen pass 标准: 两机 final `1024x256` accuracy 都 `>=0.85`, 且 paired gap `<=4pp`。

| seed | variant | 2080ti final 1024x256 | 3090 final 1024x256 | gap | pass |
|---:|---|---:|---:|---:|---|
| 125 | `baseline-r16-joint` | 0.929 | 0.953 | 2.4pp | yes |
| 125 | `read-gate-r16` | 0.916 | 0.957 | 4.1pp | no |
| 125 | `read-softmargin-r16` | 0.916 | 0.934 | 1.8pp | yes |
| 125 | `read-gate-softmargin-r16` | 0.939 | 0.745 | 19.4pp | no |
| 124 | `baseline-r16-joint` | 0.902 | 0.939 | 3.7pp | yes |
| 124 | `read-gate-r16` | 0.962 | 0.919 | 4.3pp | no |
| 124 | `read-softmargin-r16` | 0.546 | 0.943 | 39.7pp | no |
| 124 | `read-gate-softmargin-r16` | 0.951 | 0.949 | 0.2pp | yes |

Variant-level summary:

| variant | passed pairs | mean gap | max gap | min 1024x256 | 判断 |
|---|---:|---:|---:|---:|---|
| `baseline-r16-joint` | 2/2 | 3.05pp | 3.7pp | 0.902 | 本轮唯一 two-seed strict pass |
| `read-gate-r16` | 0/2 | 4.2pp | 4.3pp | 0.916 | 高分但 gap 略超线, 不算过线 |
| `read-softmargin-r16` | 1/2 | 20.75pp | 39.7pp | 0.546 | 有单机崩, 不稳 |
| `read-gate-softmargin-r16` | 1/2 | 9.8pp | 19.4pp | 0.745 | 有单机崩, 不稳 |

## 机制指标观察

`read-gate-r16` 虽然没有低分崩溃, 但没有把跨机器 gap 压进 `4pp`。同时它没有稳定降低 residual state 的尖峰风险: `s125` 的 2080ti run 出现 `update_norm_p95=2.78`, `update_norm_max=14.3`, `m_norm_max=14.7`; `s124` 的 3090 run 出现 `update_norm_max=3.8`, `m_norm_max=7.07`。

`read-softmargin-r16` 在 `s125` 过线, 但在 `s124` 的 2080ti 掉到 `0.546`, loss 为 `1.08`。这说明只把 selected top-k 内部权重变平滑并不可靠, 也可能在某些轨迹上把 residual read 的有效信号削弱或改坏。

`read-gate-softmargin-r16` 在 `s124` 表现很好, 但在 `s125` 的 3090 掉到 `0.745`。这说明 gate 和 softmargin 叠加存在交互风险, 至少当前参数不能推进为候选默认。

需要注意: 本轮 formal runs 关闭了 heavy trace, 因此这些 scalar metrics 只能说明训练末端状态和机制已启用, 不能单独证明具体分叉发生在哪一步。

## 结论

1. `baseline-r16-joint` 复现为当前最稳的 default-dropout paired 1ep 底座。它在两个 seed 上都达到高分且 gap 小于 `4pp`。

2. 新增 read confidence 控制没有提供明确增益。`read-gate` 是 borderline high-score variant, 但两个 seed 都略超 gap 线; `read-softmargin` 和 combined variant 都有单机明显掉分。

3. 当前稳定化主线仍应保留在 `update_norm_softcap + residual injection warmup + moderately wide read_topk=16`。这轮结果不支持把 read gate 或 read softmargin 加入默认训练配置。

4. read-side support-confidence 不是完全没有价值, 但当前实现更像是额外扰动源, 不是稳定器。如果后续继续研究 read-side 机制, 应先做轻量 trace 或 replay, 不要直接叠加到 formal training 候选。

## 下一步建议

短期内把 `baseline-r16-joint` 作为 default-dropout r16 对照底座。后续如果要继续验证稳定性, 优先做 same-seed rerun 或少量 alternate seed paired 1ep, 不要继续堆叠新的 read-side 控制。

如果要继续做机制改进, 更值得围绕已经有正信号的两点推进: M_state update 幅度控制和 residual injection schedule。read-side confidence guard 可以先退回 diagnostic/probe 角色。

在扩大实验前, 也需要继续优化显存和训练速度。当前实验成本仍偏高, 如果不先降低成本, 后续多 seed 或 4ep confirm 会比较低效。

## Artifact

- `docs/artifacts/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/run-summary.csv`
- `docs/artifacts/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/cross-machine-comparison.csv`
- `docs/artifacts/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/variant-summary.csv`
- `docs/artifacts/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/mechanism-metrics-summary.csv`
- `docs/artifacts/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/cache-init-preflight-summary.csv`
- `docs/artifacts/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/batch-order-summary.csv`
- `docs/artifacts/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen/source-manifest.csv`
