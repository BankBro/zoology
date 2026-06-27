# 20260627-03 Flash-VQG First-Divergence Probe Report

status: completed_debug_probe
ledger: not written

## 目标

本轮只定位 2080ti 和 3090 在 cache/init 都锁定后的第一处分叉, 不跑完整 1 epoch.

## 前置一致性

- zoology: `flash-vqg`, commit `d8ead20`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `1e7ed33`.
- MQAR cache: 13/13 content hash match, combined hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash: `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.
- batch order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.

## 结果

| variant | 首个分叉点 | 说明 |
|---|---|---|
| `baseline` | `backbone.layers.0.dropout1` | embeddings 相同, 第一层 dropout 输出不同. |
| `strict-fp32` | `backbone.layers.0.dropout1` | 禁 TF32 和 deterministic policy 没有消除 dropout 分叉. |
| `shadow-read` | `backbone.layers.0.dropout1` | shadow metrics 不改变训练输出. |
| `no-dropout` | `backbone.layers.1.sequence_mixer.mixer` | 第一层完全一致, 分叉推迟到第二层 Flash-VQG mixer. |

关键数字:

- baseline first forward loss: 2080ti `10.864863`, 3090 `10.880036`.
- no-dropout first forward loss: 两边均为 `10.871896`.
- baseline first preds hash match, logits hash mismatch.
- no-dropout first preds hash match, logits hash mismatch.

## 判断

这轮把问题拆成两层:

1. 当前正式训练配置里, 第一处实际分叉不是 cache, init, input, embedding, 也不是 GD residual read, 而是 `embed_dropout=0.1` 导致的 CUDA dropout RNG 跨 GPU 差异.
2. 如果为了定位关掉 dropout, 第一层可以做到完全一致; 下一处跨 GPU 分叉出现在第 1 层 Flash-VQG mixer. 这说明仍有 Flash-VQG 数值路径差异, 但它不是 baseline 的第一处扰动源.

`shadow-read` 的第一批 dense/top-k residual read shadow 指标全为 0, 这是因为第一块远程 residual state 尚为零, 所以它不能解释第一步分叉. 这个指标要在更晚 train step 才有判读价值.

## 下一步

不要直接再跑完整 1 epoch. 更低成本的下一步是:

- 加一个正式可配置开关或实验 variant: `embed_dropout=0.0`, 其余机制不动, 跑 `1024x256` 重点 slice 的短 screen.
- 若 no-dropout 明显缩小 1024x256 gap, 再考虑把 dropout 作为稳定性因素处理.
- 若 no-dropout 仍有 1024x256 gap, 再针对第 1 层 Flash-VQG mixer 做更细粒度 probe, 例如 q/k/v, state build, phase2 read 各自 hash.
