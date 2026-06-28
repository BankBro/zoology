# 20260628-03 Flash-VQG mixer divergence probe plan

status: planned
experiment_id: `20260628-03-flash-vqg-mixer-divergence-probe`

## 背景

`20260628-02` no-dropout 4 epoch confirm 没有通过 4pp 稳定线:

| run | final 1024x256 |
|---|---:|
| 2080ti r1 | 0.840 |
| 3090 r1 | 0.790 |
| 3090 r2 | 0.762 |

3090 两条相对 2080ti final gap 是 5.0pp 和 7.8pp, 超过用户可接受的 4pp. cache/init 都已锁住且验证 match, 所以继续重复 cache/init 或继续追加 no-dropout 长训练收益低.

`20260627-03` first-divergence probe 已定位到:

- baseline 首个分叉: `backbone.layers.0.dropout1`.
- no-dropout 首个分叉: `backbone.layers.1.sequence_mixer.mixer`.

本计划只定位 no-dropout 后剩余跨机器分叉在 Flash-VQG mixer 内部的哪条路径上出现或被放大.

## 目标

回答一个窄问题:

```text
在相同 cache, 相同 batch order, 相同 canonical init, no-dropout 条件下,
2080ti 和 3090 从 layer 1 Flash-VQG mixer 的哪个子路径开始出现可解释的数值分叉?
```

本轮不是效果实验, 不跑 4 epoch, 不写 official MQAR ledger.

## 矩阵

共同配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`, `read_topk=2`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- canonical cache hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.

| machine | run |
|---|---|
| 2080ti | `mixer-probe-s123-r1` |
| 3090 | `mixer-probe-s123-r1` |

不跑 3090 r2. 当前 3090 repeat gap 在 4pp 内, 主要问题是跨机器带差异.

## 建议执行方式

优先短窗口, 不做长训:

1. 复用 first-divergence probe 的 batch order 和 input/target hash 检查.
2. 只跑前若干 optimizer step 或固定 trace step, 例如 `0, 1, 4, 16, 64, 130, 203, 352, 448, 704`.
3. 在每个 trace point 保存 CPU tensor hash 和必要摘要, 不保存大 tensor 全量除非首次 mismatch 需要局部 dump.
4. 对 2080ti 与 3090 的 trace 按 module path 和 step 做 join, 输出 first mismatch timeline.

## Trace 范围

最小 trace 面:

- layer 1 mixer input/output.
- q/k/v 或 mixer 内等价投影输出.
- VQ routing: nearest code indices, code assignment mass, code usage summary.
- GD residual write: write q/top1, write strength, sum zeta, beta, update norm.
- GD residual state: `M`/`m` 的 norm/hash, state build 前后 hash.
- GD residual read: read candidates, top-k indices, top1/top2 margin, selected mass, read output.
- mixer output 和下一层输入.
- loss/logits hash.

优先记录 hash, shape, dtype, mean/std/max/norm/top-k index digest. 只有 first mismatch 附近再 dump 少量 selected tensor slice.

## 判读

期望产物不是 final accuracy, 而是:

- `first_mismatch_step`.
- `first_mismatch_module`.
- `first_mismatch_tensor_or_metric`.
- 分叉是否从 continuous tensor hash 开始, 还是先体现在 discrete VQ/read index 选择.
- 分叉是否随后进入 GD residual state 累积并扩大.

如果 first mismatch 出现在 q/k/v 连续值但 VQ/read discrete selection 尚一致, 下一步查 GPU kernel / matmul / normalization 数值路径.

如果 first mismatch 直接体现在 VQ code index 或 residual read top-k, 下一步查 routing margin, tie/near-tie 和候选稳定性.

如果 continuous 差异很小但 state norm/read candidate churn 很快放大, 下一步查 GD residual state update 和 read/write guard.

## 硬门槛

启动前必须在两机 `Flash-VQG-tun` 容器内确认:

- `nvidia-smi` 可用.
- `torch.cuda.is_available()` 为 true.
- zoology 和 Flash-VQG 同步到同一 commit.
- cache content hash match.
- init state hash match.
- batch order hash match.

任一项失败, 不启动 probe.

## 收尾

Artifact 目录:

```text
docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/
```

报告:

```text
docs/20260628-03-flash-vqg-mixer-divergence-probe-report.md
```

收尾时记录:

- source machine 和 mirror path.
- trace manifest 和 sha256.
- first mismatch table.
- 未镜像的大型 raw tensor dump 路径.
