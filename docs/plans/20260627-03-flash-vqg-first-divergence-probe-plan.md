# 20260627-03 Flash-VQG First-Divergence Probe Plan

updated: 2026-06-27
status: implemented
experiment_id: `20260627-03-flash-vqg-first-divergence-probe`

## 目标

本轮是定位实验, 不是最终方案实验, 不写 official ledger.

要回答的问题:

```text
在 cache 内容相同, init checkpoint 相同后, 2080ti 和 3090 的第一次有意义分叉出现在什么层级或路径.
```

优先判断三件事:

- `strict-fp32` 是否能明显推迟或缩小第一步的 logits/grad/param 分叉.
- `shadow-read` 中 top-k residual read 和 full dense residual read 的差异是否足够大, 是否可能解释后续 1024x256 准确率敏感.
- `ref-gd` 是否改变第一步分叉形态, 用来判断 grouped/chunk state build 是否是数值放大点.

## 前置硬门槛

两台机器都必须在对应宿主机的 `Flash-VQG-tun` 容器内执行:

- `nvidia-smi` 可用.
- `torch.cuda.is_available()` 为 true.
- `zoology` 和 `Flash-VQG` 同步到同一 commit.
- 本轮实际加载的 13 个 MQAR cache content hash 等于 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash 等于 `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.

任一项失败, 停止启动 probe.

## 实验矩阵

第一轮只跑短 probe, 每个 run 默认 1 个 optimizer step, 即 4 个 micro-batch.

| variant | 机器 | 目的 |
|---|---|---|
| `baseline` | 2080ti + 3090 | 复现 init/cache 已锁定后的第一分叉 |
| `strict-fp32` | 2080ti + 3090 | 检查 TF32/deterministic policy 是否影响分叉 |
| `shadow-read` | 2080ti + 3090 | 不改变训练输出, 只记录 full dense residual read shadow 指标 |
| `ref-gd` | 2080ti + 3090, 可选补充 | 用慢速 `loop_ref` event pack 路径检查 semivec/chunk pack 数值路径 |

首轮默认先跑 `baseline`, `strict-fp32`, `shadow-read`. `ref-gd` 本机 1 step 约 2 分多, 明显慢于其他路径; 只有前三个 variant 指向 state pack 路径时, 再补跨机 `ref-gd`.

如果 1 step 看不出有用差别, 再追加 warmup probe:

```text
--max-optimizer-steps 8
--capture-optimizer-steps 0,1,8
```

必要时再到 `32,64,130`, 但仍不直接跑完整 1 epoch.

## 关键口径

`shadow-read` 不改变训练语义:

- 实际输出仍然使用当前 `read_topk=2`.
- 额外计算 full dense residual read 只写入 metrics.
- 这个 variant 用来判断 top-k residual read 和 full residual read 的偏差有多大, 不是候选训练方案.

`ref-gd` 改变执行路径, 数学语义目标与 baseline 一致:

- builder: `grouped_chunk_torch_ref`.
- pack: `loop_ref`.
- `token_step_ref` 是更朴素的 forward reference, 但当前完整训练 backward 会触发 in-place autograd 错误, 所以本轮不把它作为跨机训练 probe.
- 如果同机 smoke 都不稳, 不做跨机结论.

## 判读

第一优先看:

- cache/init/batch_order/input/target 是否 match.
- 第一批 logits 是否 match.
- 每层 forward hash 的第一个 mismatch 模块.
- backward grad hash 的第一次 mismatch.
- step1 model param hash 是否 match.

第二优先看:

- `valid` 不是本轮目标.
- 1024x256 准确率问题只作为后续解释对象, 不在本轮直接做完整 eval.

可能结论:

- 如果 `strict-fp32` 推迟分叉或明显缩小 loss/logit/margin 差异, 下一步优先试 dtype policy 或 kernel policy.
- 如果 `shadow-read` 指标显示 top-k residual read 与 dense residual read 偏差很大, 下一步优先做 read-side gate/read mass guard, 而不是继续纠缠 init.
- 如果 `ref-gd` 改善分叉形态, 下一步检查 grouped/chunk state build 的数值稳定性.
- 如果四个 variant 都很早分叉且形态相似, 则说明跨 GPU 数值路径漂移不可避免, 下一步应做稳定性机制, 不是继续追 bitwise 一致.

## 产物

脚本:

```text
zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/probe_first_divergence.py
```

本地输出:

```text
zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/outputs/
```

收尾 artifact:

```text
docs/artifacts/20260627-03-flash-vqg-first-divergence-probe/
```

报告:

```text
docs/20260627-03-flash-vqg-first-divergence-probe-report.md
```
