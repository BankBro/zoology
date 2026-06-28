# 20260628-03 Flash-VQG mixer divergence probe 报告

status: completed_debug_probe
ledger: not written

## 目标

本轮不是效果实验, 不跑正式 4 epoch, 不写 official MQAR ledger.

目标是在相同 cache, 相同 canonical init, 相同 batch order, no-dropout 条件下, 定位 2080ti 和 3090 在 `backbone.layers.1.sequence_mixer.mixer` 内部的第一处分叉.

## 口径

代码版本:

- zoology: `flash-vqg`, commit `99cc2f8`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `474b763`.

共同配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`, `read_topk=2`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- canonical MQAR cache hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.

前置检查全部通过:

| item | 2080ti | 3090 |
|---|---|---|
| cache hash | match | match |
| init hash | match | match |
| batch order hash | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

运行完成后两边 GPU 均释放. 3090 raw JSON 已镜像回 2080ti, `preflight.json` 和 `probe.json` 的远端/本地 sha256 均一致.

## 结果

两边都完成 `703` 个 optimizer step. 本轮计划 trace `0,1,4,16,64,130,203,352,448,704`, 但 dataloader 一轮实际只产生到 step `703`, 因此实际 trace step 是:

```text
0, 1, 4, 16, 64, 130, 203, 352, 448
```

每个 trace step 都有 25 个 layer-1 mixer 内部 trace record. 跨机器 join 后:

- trace rows: `504`.
- comparison rows: `252`.
- mismatch rows: `200`.
- first mismatch: step `0`, micro `0`, layer `1`, `state_build/logf_all`.

step 0 明细:

| trace | match |
|---|---|
| `phase1/q_all` | true |
| `phase1/k_all` | true |
| `phase1/v_all` | true |
| `phase1/g_raw_all` | true |
| `phase1/K_q_all` | true |
| `phase1/Delta_all` | true |
| `phase1/W_all` | true |
| `state_build/logf_all` | false |
| `state_build/beta_all` | true |
| `state_build/G_state` | false |
| `state_build/L_state` | false |
| `state_build/M_state` | false |
| `phase2_read/top_idx` | true |
| `phase2_read/u_res` | true |
| `forward/preds` | true |
| `forward/loss` | true |

后续传播:

| step | first mismatch | mismatch count |
|---:|---|---:|
| 0 | `state_build/logf_all` | 10/28 |
| 1 | `phase1/q_all` | 24/28 |
| 4 | `phase1/q_all` | 17/28 |
| 16 | `phase1/q_all` | 25/28 |
| 64 | `phase1/q_all` | 24/28 |
| 130 | `phase1/q_all` | 25/28 |
| 203 | `phase1/q_all` | 27/28 |
| 352 | `phase1/q_all` | 28/28 |
| 448 | `phase1/q_all` | 20/28 |

离散路径开始分叉较晚:

- `phase2_read/top_idx` mismatch 出现在 step `16, 130, 203, 352`.
- `phase1/Delta_all` mismatch 出现在 step `352, 448`.
- `forward/preds` 和 `forward/loss` mismatch 出现在 step `203, 352, 448`.

## 判读

这轮把 no-dropout 后的剩余分叉位置进一步缩小了.

step 0 时, layer 1 的 `q/k/v`, `g_raw`, `K_q`, VQ assignment `Delta_all` 和 write weight `W_all` 都是 bitwise match. 因此第一处分叉不是 VQ routing, 不是 read top-k, 也不是输入, cache, init 或 batch order.

第一处分叉是 `state_build/logf_all`. 代码路径是:

```text
logf_all = fox_gate_logf(x, self.fox_gate_proj, self.config, attention_mask)
```

其中 `fox_gate_logf` 先做 `fox_gate_proj(x)`, 再做 `F.logsigmoid(logits.float())`. 在 step 0 的 `x` 已经由 phase1 侧间接证明一致, 所以当前最可能的起点是 gate/state-build 连续值路径上的 CUDA linear/logsigmoid 数值差异. 这个差异非常小, mean/l2 summary 仍几乎相同, 但 hash 已经不同.

随后, 这个很小的 `logf_all` 差异进入 `G_state/L_state/M_state`, 再进入 phase2 output. 到 step 16 开始影响 read top-k index, 到 step 203 以后影响 preds/loss. 这说明后续离散 read/top-k 是放大器之一, 但不是本轮观察到的第一起点.

## 对下一步的含义

不建议继续补 4 epoch 或继续只做 dropout ablation. 当前更有价值的下一步是围绕 `fox_gate_logf` 做最小对照:

- `gate-fp64-shadow`: 不改变训练输出, 额外用 CPU 或 fp64 shadow 计算 `fox_gate_proj/logsigmoid` 摘要, 判断两机差异是否来自 GPU linear/logsigmoid.
- `gate-fp32-ref-path`: 只把 `fox_gate_logf` 切到更朴素的 reference path, 看第一处分叉是否推迟.
- `logf-rounding/guard` 小实验: 对 `logf_all` 做轻量量化或稳定化, 看 read top-k/preds 分叉是否明显推迟.

这不是最终解决方案, 但已经把定位从“Flash-VQG mixer 内部”缩小到“layer 1 state-build gate/logf 连续值路径先分叉, 后续 GD residual state/read/top-k 放大”.

## 产物

- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/trace-summary.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/cross-machine-trace-comparison.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/preflight-summary.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/source-manifest.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/metadata.json`
