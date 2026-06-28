# 20260629-01 Flash-VQG gate/logf stability 报告

status: p0_smoke_ready
ledger: not written

## 当前状态

本轮已完成 P0/P1 所需的最小代码准备, 尚未启动跨机器正式 probe.

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

下一步按 plan 先跑 P0:

1. 两机容器内 preflight: GPU, commit, canonical cache, canonical init, batch order.
2. 2080ti x1 + 3090 x1 执行 P0 `probe`.
3. mirror 3090 轻量 JSON 回 2080ti.
4. 运行 `collect`, 生成 artifact.
5. 根据 `gate-comparison-summary.csv` 决定是否进入 P1.

## 判读口径

本轮只看定位证据, 不写 official MQAR ledger.

核心问题:

```text
fox_gate_logf 的分叉到底来自 input, gate logits, logsigmoid/logf, 还是后续 state/read 放大?
```

如果 P0 不能定位来源, 不进入 1 epoch / 4 epoch 长训练.

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

当前 artifact 只包含单机 smoke payload, `gate-comparison-summary.csv` 尚无跨机器 join 行. 等 3090 probe JSON mirror 回来后, 需要重新 `collect`.
