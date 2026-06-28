# 20260629-02 Flash-VQG gate/top-k/M-state causal probe plan

## 目的

本轮实验不继续调参, 也不把 `fox_gate_logf_round_quantum=1e-5` 当成最终方案. 目标是用最小 17-step cross-machine probe 拆清当前 `gd_residual_v1` 分叉链路中谁是触发入口, 谁是放大器.

当前已知条件:

- 同一份 MQAR canonical cache.
- 同一份 canonical init checkpoint.
- 同一 batch order.
- no-dropout.
- seed=123.
- cb64-r16.
- 2080ti x1 vs 3090 x1.
- trace optimizer steps: `0,1,4,16`.

已有证据显示首个可观测分叉在 `optimizer_step=0`, `micro_step=0`, `layer_idx=1`, `fox_gate/logits_cuda`, 同一位置 `fox_gate/input_x` match. 后续分叉链路经过 `fox_gate/logf_cuda`, `state_build/logf_all`, `G_state/L_state/M_state`, `phase2_read/top_idx`, `forward/loss`, `grad`, `optimizer/model state`.

## 第一轮 variants

第一轮只跑 17-step probe:

| variant | 配置差异 | 目的 |
|---|---|---|
| `baseline` | 无额外改动 | 复现当前 no-dropout 分叉链路 |
| `constant-logf-f0.95` | `fox_gate_logf_constant_f=0.95` | 验证 learned/dynamic gate/logf 是否是必要触发入口 |
| `dense-read` | `fox_remote_read_topk=64`, `fox_gd_residual_dense_read_chunked=True` | cb64 下等价 full-code read, 验证 read top-k 是否是主要离散放大器 |
| `residual-off` | `fox_gd_residual_residual_norm_mode=zero` | 验证 residual contribution 是否是主要输出放大器 |
| `round1e-5-control` | `fox_gate_logf_round_quantum=1e-5` | 仅作为 diagnostic control, 不作为部署方案 |

注意:

- `residual-off` 关闭的是 residual branch 对输出的贡献, 不是完全跳过 `M_state` 构建.
- `dense-read=64` 只在当前 cb64 配置下等价 no-topk. 若换 cb128/cb256, 需要改为对应 codebook size.
- `fox_gd_residual_dense_read_chunked=True` 仅用于本轮 dense-read 诊断, 目的是避免 full-code residual read 的展开张量 OOM. 默认关闭, 不改变普通训练语义.
- `constant-logf-f0.95` 是诊断干预, 不是候选主方法.
- `GDN-style gate` 暂不纳入第一轮. 它会引入新参数, 新 init, 新语义, 需要单独设计.

## 执行流程

1. 在 Flash-VQG 增加默认关闭的 `fox_gate_logf_constant_f`.
2. 在 zoology 新增 `20260629-02` probe 脚本, 暴露 variant 和配置 override.
3. 在 2080ti 跑 smoke:
   - `baseline`, 1-step.
   - `constant-logf-f0.95`, 1-step.
   - `dense-read`, 1-step.
   - `residual-off`, 1-step.
4. commit/push Flash-VQG 和 zoology.
5. 在 3090 pull 到相同 commit.
6. 在两边容器内检查 GPU/NVML, `torch.cuda.is_available()`, cache hash, init hash, batch order hash.
7. 两边并行跑 17-step variants.
8. collect artifact, 生成 report.
9. 如果 probe 指向明确且时间充足, 最多追加 1-2 个 1ep screen, 不跑 4ep.

长训练或评估任务进入稳定期后, 显式 `sleep 1200` 轮询, 不进行高频轮询.

## 判定口径

- 如果 `constant-logf-f0.95` 明显推迟或消除 state/top-k/loss 分叉, 说明 learned/dynamic gate/logf 是关键触发入口.
- 如果 `dense-read` 在 state 仍 mismatch 的情况下推迟或消除 loss/grad 分叉, 说明 read top-k candidate flip 是主要离散放大器.
- 如果 `residual-off` 在 M-state 仍 mismatch 的情况下稳定 output/loss/grad, 说明 residual branch 是主要输出放大器.
- 如果所有 probe 都很快分叉, 下一步应检查 state build, phase2 reduce, backward 数值路径.
- `round1e-5` 若有效, 只能说明低位 logf 扰动参与触发放大, 不能说明 rounding 修好了根因.

## 产物

- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/preflight-summary.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/trace-summary.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/cross-machine-trace-comparison.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/gate-comparison-summary.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/variant-summary.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/source-manifest.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/metadata.json`
- `docs/20260629-02-flash-vqg-gate-topk-mstate-causal-probe-report.md`
