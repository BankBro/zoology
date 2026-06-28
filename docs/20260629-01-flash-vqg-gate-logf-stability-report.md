# 20260629-01 Flash-VQG gate/logf stability 报告

status: p0_controls_done
ledger: not written

## 当前状态

本轮已完成 P0 跨机器 probe 和 3 个低成本 control. 这些结果是 diagnostic / exploratory, 不写 official MQAR ledger.

已实现:

- `fox_gate_logf` 可选 trace, 默认关闭, 普通训练语义不变.
- P0 trace 字段:
  - `fox_gate/input_x`
  - `fox_gate/logits_cuda`
  - `fox_gate/logf_cuda`
  - `fox_gate/logits_ref_fp64_cpu`
  - `fox_gate/logf_ref_fp64_cpu`
- P1 最小干预开关:
  - `fox_gate_logf_compute_mode=fp32_linear`
  - `fox_gate_logf_round_quantum`
  - `fox_gate_logit_normalizer`
- P0 probe 脚本:
  - `zoology/experiments/flash_vqg/scripts/20260629-01-flash-vqg-gate-logf-stability/gate_logf_stability_probe.py`

## 待执行

下一步不要直接补更多定位 probe. 当前最有价值的候选是 `fox_gate_logf_round_quantum=1e-5`, 应进入 1 epoch screen:

1. 仍使用 canonical cache 和 canonical init.
2. 跑 `2080ti x1 + 3090 x1`.
3. 指标重点看 `valid/mqar_case/accuracy-1024x256`.
4. 若 final gap <= 4pp 且没有明显掉分, 再补 `3090 r2`.
5. 若 1 epoch 不稳, 停止 rounding 方向, 不进入 4 epoch confirm.

## 判读口径

本轮只看定位证据, 不写 official MQAR ledger.

核心问题:

```text
fox_gate_logf 的分叉到底来自 input, gate logits, logsigmoid/logf, 还是后续 state/read 放大?
```

P0 已定位到 gate projection/logf 边界, 但还没有证明这会导致 1 epoch / 4 epoch 指标 gap. 需要用短训练 screen 验证候选是否真的改善主指标稳定性.

## 本地 smoke 验证

已在 2080ti 容器内完成最小 smoke:

- Flash-VQG 变更文件 `py_compile` 通过.
- `gate_logf_stability_probe.py` `py_compile` 和 `--help` 通过.
- 2080ti 容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 通过.
- 2080ti `preflight --max-optimizer-steps 1` 通过.
- 2080ti `probe --max-optimizer-steps 1 --trace-optimizer-steps 0` 通过.

smoke trace 已确认包含新增 gate 路径:

| trace_name | shape | dtype |
|---|---:|---|
| `fox_gate/input_x` | `[64,64,128]` | `torch.float32` |
| `fox_gate/logits_cuda` | `[64,2,64]` | `torch.float32` |
| `fox_gate/logf_cuda` | `[64,2,64]` | `torch.float32` |
| `fox_gate/logits_ref_fp64_cpu` | `[64,2,64]` | `torch.float64` |
| `fox_gate/logf_ref_fp64_cpu` | `[64,2,64]` | `torch.float64` |
| `state_build/logf_all` | `[64,2,64]` | `torch.float32` |

smoke JSON 已从正式 artifact manifest 中移除, 只保留 P0 和 control 的正式 JSON summary.

## P0 跨机器结果

执行口径:

- 机器: `2080ti x1 + 3090 x1`.
- 步数: `max_optimizer_steps=17`.
- trace step: `0,1,4,16`.
- 条件: no-dropout, `seed=123`, canonical cache, canonical init, 同一 batch order.
- zoology commit: `c8dd698`.
- Flash-VQG commit: `b5f8fee`.
- torch: 两边均为 `2.6.0+cu118`.
- dtype policy: 两边 `torch.backends.cuda.matmul.allow_tf32=False`, `torch.get_float32_matmul_precision()=highest`; `cudnn.allow_tf32=True`, 但本轮首分叉在 linear/logf 路径, 不是 cudnn 路径.

preflight 证据:

| item | 2080ti | 3090 |
|---|---|---|
| cache content hash | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` | same |
| init state hash | `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf` | same |
| batch order hash | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` | same |
| zoology commit | `c8dd698` | `c8dd698` |
| Flash-VQG commit | `b5f8fee` | `b5f8fee` |

首个跨机器 mismatch:

```text
optimizer_step=0, micro_step=0, layer_idx=1, trace_name=fox_gate/logits_cuda
```

关键传播链:

- `fox_gate/input_x` 在 step 0 match, 说明进入 layer 1 gate 的输入不是首因.
- `fox_gate/logits_cuda` 在 step 0 首先 mismatch, mean gap 约 `2.33e-10`.
- `fox_gate/logf_cuda` 和 `state_build/logf_all` 紧随其后 mismatch.
- `G_state/L_state/M_state` 在 step 0 已有 bitwise mismatch, 但统计差异仍极小.
- `phase2_read/top_idx` 到 step 16 才第一次 mismatch.
- `forward/preds` 到 step 16 仍 match, `forward/loss` 到 step 16 才出现 `9.54e-7` 差异.
- backward hash 从 micro step 1 起已经不同, optimizer step 1 后 model/optimizer state hash 也不同. 所以后续 `input_x` 分叉是训练状态分叉的结果, 不是 batch 或 cache 变了.

这说明: 当前跨机器漂移的最早可观测位置在 `fox_gate_proj(x)` / `fox_gate_logf` 边界, read top-k 是后续放大点, 不是第一分叉点.

不能说明的是: P0 只跑了 17 个 optimizer step, 每台机器 1 次, 只看 `s123/cb64-r16/read_topk=2/layer1`. 它还不能证明该早期低位分叉就是 1 epoch 或 4 epoch accuracy gap 的充分原因.

## Control 结果

低成本 control 均为 `2080ti x1 + 3090 x1`, `max_optimizer_steps=17`, trace step `0,1,4,16`.

| variant | first mismatch | mismatch rows | step16 top_idx match | step16 loss match | step17 param match |
|---|---|---:|---|---|---|
| baseline | `0:0:1:fox_gate/logits_cuda` | 96 | False | False | False |
| `fp32-linear` | `0:0:1:fox_gate/logits_cuda` | 96 | False | False | False |
| `round1e-6` | `0:0:1:fox_gate/logits_cuda` | 94 | True | True | False |
| `round1e-5` | `0:0:1:fox_gate/logits_cuda` | 89 | True | True | False |

判读:

- `fox_gate_logf_compute_mode=fp32_linear` 基本没有帮助. 它没有改变首个 mismatch, 也没有阻止 step16 的 `top_idx/loss` 分叉.
- `fox_gate_logf_round_quantum=1e-6` 和 `1e-5` 都不能让训练 bitwise 一致, 梯度和参数 hash 仍从第一轮 backward / optimizer step 起分叉.
- 但 `round1e-6` 和 `round1e-5` 都把 step16 的 `phase2_read/top_idx` 和 `forward/loss` 拉回 match.
- `round1e-5` 的 mismatch rows 更少, 是当前最值得进入 1 epoch screen 的候选.

## 产物

正式 artifact:

```text
docs/artifacts/20260629-01-flash-vqg-gate-logf-stability/
```

核心文件:

- `preflight-summary.csv`: P0 cache/init/batch/code 证据.
- `metadata.json`: P0 首个 mismatch.
- `gate-comparison-summary.csv`: P0 gate/logf/state/read/loss 关键对比.
- `control-summary.csv`: baseline 与 3 个 control 的压缩总结.
- `controls/fp32-linear/`
- `controls/round1e-6/`
- `controls/round1e-5/`

原始 JSON 仍在本地脚本输出目录和 3090 source path 原位保留. `source-manifest.csv` 记录了本机 mirror 后的轻量 JSON hash.
