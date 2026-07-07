# 20260707-01 Flash-VQG R8/R16 Joint Control 复现实验报告

## 结论

本轮原计划是复跑 `update_softcap0p5 + injwarm512` 这个当前最强 joint-control 方案, 分别测试 `read_topk=8` 和 `read_topk=16`, 并在 3090 GPU0, 2080ti GPU0, 2080ti GPU1 上各跑一次。

实际执行中, `read_topk=8` 在 3090 GPU0 和 2080ti GPU1 完成, 但 2080ti GPU0 在 validation 后卡住且容器内 NVML/CUDA 失效。按照 AGENTS.md 的 GPU 硬门槛, 正式 paired 实验被中止, 后续只在 3090 上补跑了 `read_topk=16`。

因此本轮不能作为完整的跨机器稳定性 pass/fail 证明。可用结果说明:

- 3090 上 `read_topk=8` 和 `read_topk=16` 都仍然高分, 分别为 `0.931` 和 `0.936`.
- 2080ti GPU1 的 `read_topk=8` 完成, 1024x256 为 `0.853`, 但与 3090 的 `0.931` 相差 `7.8pp`, 超过 4pp 稳定线。
- 2080ti GPU0 的结果不计入模型结论, 因为该 run 没有写出 result JSON, 且随后确认 GPU/NVML 失效。

一句话: 这轮没有证明 R8/R16 joint control 已经跨机器稳定复现, 但继续支持 3090 上该方案是高分候选。下一步必须先恢复或替换 2080ti 这一路硬件环境, 再做干净 paired rerun。

## 实验口径

固定条件:

- `seed=124`
- canonical MQAR cache, 13 个文件内容 hash 一致
- canonical seed124 init checkpoint 一致
- batch order 一致
- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`
- model: `cb64-r16`
- `fox_gd_residual_write_topk=4`
- `fox_gd_residual_update_norm_softcap=0.5`
- `fox_gd_residual_update_norm_softcap_mode=smooth_p4`
- residual injection warmup: optimizer step `0 -> 512`
- `max_epochs=1`, `max_train_steps=704`, `gradient_accumulation_steps=4`
- heavy `read_trace`, `D-geometry trace`, train inline event trace 均关闭

关键一致性记录:

- zoology commit: `4a92405`
- Flash-VQG commit: `0eba390`
- MQAR cache combined content sha256: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`
- init model_state sha256: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`
- batch order sha256: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`

## 运行结果

本轮 1 epoch 日志中, `best` 与 `final` 一致。下表的 `1024x256` 是 final/best hard slice accuracy。

| variant | machine | GPU | status | 1024x256 | valid acc | valid loss | 说明 |
|---|---:|---:|---|---:|---:|---:|---|
| `r8-update-softcap0p5-injwarm512-rerun` | 3090 | 0 | completed | `0.931` | `0.987` | `0.200` | 原 paired attempt 中完成 |
| `r8-update-softcap0p5-injwarm512-rerun` | 2080ti | 1 | completed | `0.853` | `0.971` | `0.321` | 原 paired attempt 中完成 |
| `r8-update-softcap0p5-injwarm512-rerun` | 2080ti | 0 | failed | `0.000234` | `0.000289` | `8.410` | 进程卡住且无 result JSON, 不计入正式模型结果 |
| `r16-update-softcap0p5-injwarm512-rerun` | 3090 | 0 | completed | `0.936` | `0.988` | `0.191` | 2080ti 故障后补跑的 3090-only 结果 |

按可用 completed 结果汇总:

| variant | completed runs | 1024x256 min | 1024x256 max | available gap | 判定 |
|---|---:|---:|---:|---:|---|
| `r8-update-softcap0p5-injwarm512-rerun` | 2 | `0.853` | `0.931` | `7.8pp` | 不过线, 且 paired 证据不完整 |
| `r16-update-softcap0p5-injwarm512-rerun` | 1 | `0.936` | `0.936` | N/A | 3090-only, 不能判定跨机器稳定 |

## 机制指标

| run | read_topk | 1024x256 | update p95 | update max | softcap hit | M max | lambda | inject ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3090 R8 | 8 | `0.931` | `0.923` | `8.18` | `0.0692` | `14.2` | `0.890` | `0.245` |
| 2080ti GPU1 R8 | 8 | `0.853` | `0.710` | `3.73` | `0.0646` | `6.60` | `0.211` | `0.170` |
| 3090 R16 | 16 | `0.936` | `2.05` | `10.2` | `0.105` | `13.4` | `0.831` | `0.228` |

从这些指标只能得到谨慎结论:

- 3090 R16 比 3090 R8 的 update p95/max 和 softcap hit 更高, 但 hard slice 仍略高。
- 2080ti GPU1 R8 的 update/M/lambda/inject 都明显低于 3090 R8, 但 hard slice 也低。这个差异可能是训练轨迹差异的结果, 不能直接解释成某个单一指标导致低分。
- 由于 2080ti GPU0 硬件失败, 本轮不足以判断 R8/R16 joint control 的跨机器稳定性。

## 硬件失败记录

2080ti GPU0 在 R8 validation 后没有写出 result JSON, master 进程未能自动进入后续 R16。随后检查发现容器内:

- `nvidia-smi` 报 `Unable to determine the device handle for GPU0000:01:00.0: Unknown Error`
- `torch.cuda.is_available()` 返回 false
- NVML 初始化失败

因此本轮中止 2080ti 后续训练是正确的。这个失败属于硬件/驱动/NVML 层问题, 不能把 GPU0 的低分 partial log 当作模型结果解释。

## Artifact

本轮整理后的轻量 artifact 位于:

`docs/artifacts/20260707-01-flash-vqg-r8-r16-joint-repro/`

主要文件:

- `run-summary.csv`: 每个 run 的状态和最终指标。
- `variant-summary.csv`: 按 variant 汇总可用结果和 gap。
- `mechanism-metrics-summary.csv`: 完成 run 的机制指标。
- `source-manifest.csv`: 镜像的轻量证据和源路径。
- `metadata.json`: commit, 执行口径和硬件失败说明。
- `raw-evidence/`: queue status, config, result JSON 和精简 final metric log lines。

注意: 当前 wrapper 写出的 result JSON 中 `train_result=null`, 所以本报告的 final/best 指标来自训练日志解析, 对应精简证据已写入 `raw-evidence/*/final-metric-log-lines.txt`。

## 下一步建议

1. 先处理 2080ti 容器内 GPU/NVML 问题, 或暂时把 2080ti 从 paired 机器池中移除。
2. 硬件恢复后, 重新做 `R8 + R16` 的 same-seed paired rerun。不要把本轮 3090-only R16 当成跨机器证明。
3. 如果短期只能用 3090, 可以做 3090-only 同 seed 重复 run, 只用于估计单机训练轨迹波动, 不用于跨机器结论。
4. 修正 launcher 的结果记录问题, 避免 completed run 的 `train_result=null`, 让后续 report 不再依赖日志解析。
5. 继续优先降低显存和训练耗时, 否则这种复现实验的周转成本仍然偏高。
