# 20260629-02 Flash-VQG gate/top-k/M-state causal probe 报告

status: completed_diagnostic
ledger: not written

## 目标

本轮不是继续调参, 也不是把 `fox_gate_logf_round_quantum=1e-5` 推成解决方案. 目标是用短 probe 拆清 `gd_residual_v1` 跨机器分叉链路中谁更像触发入口, 谁更像放大器。

共同口径:

- same MQAR canonical cache.
- same canonical init checkpoint.
- same batch order.
- no-dropout.
- `seed=123`, `cb64-r16`, `read_topk=2` baseline.
- 2080ti x1 vs 3090 x1.
- 17-step trace steps: `0,1,4,16`.

本轮是 diagnostic / exploratory, 不写 official MQAR ledger.

## 代码改动

Flash-VQG 新增两个默认关闭的诊断能力:

- `fox_gate_logf_constant_f`: 显式开启时绕过 learned `fox_gate_proj(x)` 动态 gate, 直接使用常数 `f`, 本轮使用 `f=0.95`. 默认 `None`, 普通训练不受影响.
- `fox_gd_residual_dense_read_chunked`: 显式开启时, full-code GD residual read 用 chunked 方式计算 `u_res`, 避免 dense-read probe 展开全码张量 OOM. 默认 `False`, 默认 full-code read 仍走原 index-select 路径.

注意: `constant-logf` 和 `dense-read` 都是诊断干预, 不是候选最终方案. `dense-read` 的 chunked 计算只解决诊断运行成本, 不改变本轮要验证的语义: cb64 下读取全部 code, 去掉 read top-k 离散 candidate flip.

## 17-step 结果

preflight 全部通过. `preflight-summary.csv` 显示 4 个 variant 在 2080ti/3090 上均满足:

| item | hash |
|---|---|
| cache content | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| canonical init | `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf` |
| batch order | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

17-step variant summary:

| variant | first mismatch | mismatch rows | step16 logf | step16 M_state | step16 top_idx | step16 loss |
|---|---|---:|---|---|---|---|
| baseline | `0:0:1:fox_gate/logits_cuda` | 96/132 | mismatch | mismatch | mismatch | mismatch |
| constant-logf-f0.95 | `1:4:1:fox_gate/input_x` | 65/124 | match | mismatch | match | match |
| dense-read | `0:0:1:fox_gate/logits_cuda` | 95/132 | mismatch | mismatch | match | match |
| residual-off | `0:0:1:fox_gate/logits_cuda` | 91/132 | mismatch | mismatch | match | mismatch |

判读:

- baseline 复现了已知链路: step0/micro0/layer1 的首个 mismatch 是 `fox_gate/logits_cuda`, 同一位置 `fox_gate/input_x` match. 这继续排除 cache/init/batch order 作为首因.
- `constant-logf-f0.95` 把 step16 的 `logf`, `top_idx`, `loss` 都压回 match, first mismatch 推迟到 step1/micro4 的 `fox_gate/input_x`. 这说明 learned/dynamic gate/logf 是关键入口之一. 但 `M_state` 仍 mismatch, 所以它不是 bitwise 确定性修复.
- `dense-read` 在 `logf` 和 `M_state` 仍 mismatch 的情况下, 让 step16 的 `top_idx` 和 `loss` match. 这说明 read top-k candidate flip 是重要离散放大器.
- `residual-off` 让 step16 `top_idx` match, 但 `loss` 仍 mismatch. 这说明 residual 输出贡献不是唯一放大器, 只关 residual contribution 不能解释全部 loss 分叉.

限制:

- 这是 layer1, seed/run 固定的 17-step probe, 不是完整训练结论.
- exact sha256 mismatch 很敏感, 很多 gap 是低位数值差异; mismatch 不等于指标显著差.
- 各 variant 的 `forward/preds` 在 step16 仍可 match, 但 scalar loss 可能不 match; report 中不能混用这两个口径.
- 17-step 只能说明局部因果链路, 不能单独证明 1ep/4ep accuracy gap 的充分原因.

## 1ep effect screen

17-step probe 后追加了两个 1 epoch effect screen:

| variant | machine | valid acc | 1024x256 | gap vs 2080ti | within 4pp |
|---|---|---:|---:|---:|---|
| constant-logf-f0.95 | 2080ti | 0.570 | 0.000434 | - | - |
| constant-logf-f0.95 | 3090 | 0.552 | 0.000367 | -0.000067 | true |
| dense-read | 2080ti | 0.979 | 0.892 | - | - |
| dense-read | 3090 | 0.979 | 0.894 | +0.002 | true |

判读:

- `constant-logf-f0.95` 的跨机器 gap 很小, 但 `1024x256` 几乎没有学起来. 所以它只能作为诊断 control: 固定 logf 可以切断 learned gate/logf 入口的一部分扰动, 但常数遗忘不是可用训练方案.
- `dense-read` 是本轮最强信号. 在 cb64 下把 remote read 改成 full-code read 后, 两机 `1024x256` 分别是 `0.892` 和 `0.894`, gap 只有 0.2pp, 且整体 valid acc/loss 也基本一致.
- 这支持一个更具体的判断: 当前不稳定的主要放大点很可能不是 “state 有低位 mismatch” 这个事实本身, 而是 state 分数进入 read top-k 后发生 candidate flip, 把连续低位差异变成离散读路径差异.
- 但 dense-read 还不是最终方案. 它改变了 read support, 计算/显存成本也更高. 更合理的后续方案是 read-candidate 稳定化, 例如 early dense/read_topk warmup, larger read_topk early then anneal, top-k margin 监控, 或 soft/dense screen 后收紧.

代码与审计 caveat:

- 1ep screen 的 `result.json` 记录版本是 zoology `ccabb7c`, Flash-VQG `2743149`.
- dense-read 1ep run 启动时, Flash-VQG `2743149` 对 `read_topk == num_codes` 自动使用 chunked dense residual read, 因此 `result.json` 的 `config_overrides` 只显示 `fox_remote_read_topk=64`.
- 后续 Flash-VQG `bc391c0` 已把 chunked dense residual read 改成显式 opt-in `fox_gd_residual_dense_read_chunked=True`, zoology 脚本也已补充记录该 override. 这个修补是为了恢复普通默认语义, 不改变已完成 run 的记录版本.

## round1e-5 定位

`fox_gate_logf_round_quantum=1e-5` 只作为此前 `20260629-01` 的 diagnostic control 引用. 它能让 step16 的 `phase2_read/top_idx` 和 `forward/loss` 回到 match, 但 `logf/state/grad/param/optimizer state` 仍 mismatch. 因此必须保持如下口径:

```text
round1e-5 is a diagnostic intervention, not a deployment fix.
```

本轮没有把 rounding 作为最终方案推进.

## 当前机制判断

`gd_residual_v1` 是合理的 V1 假设: VQ-indexed residual fast-weight memory 有潜力, 并且 dense-read 的 1ep screen 能帮助判断 read candidate 是否是可用方向.

但当前 V1 把 GDN 的全局 gated-delta 机制迁移到 VQ 分桶 memory 时, forget gate, per-code memory decay, write/read support, residual target 与 top-k candidate selection 的耦合仍偏粗. 本轮证据支持更谨慎的说法:

```text
当前 Flash-VQG/gd_residual_v1 的跨机器不稳定不是单一 cache/init/dropout 问题.
在 no-dropout 和相同输入/初始化条件下, learned gate/logf 提供早期低位扰动入口,
read top-k candidate selection 会把 state 中的低位差异转成离散读路径差异,
M_state/residual branch 参与传播但不是唯一输出放大器.
```

## 下一步

下一步不应该继续押 `round1e-5`, 也不应该把 `constant-logf` 当方案. 当前最值得利用的线索是 dense-read 的 1ep 正信号。

建议下一轮新增独立实验:

```text
20260629-03-flash-vqg-dense-read-confirm
```

第一步跑 `dense-read` 的 4 epoch confirm, 至少 `2080ti x1 + 3090 x1`, 条件保持 canonical cache/init/no-dropout/seed123/cb64-r16, 主指标仍看 `valid/mqar_case/accuracy-1024x256`. 如果 4 epoch 也在 4pp 内且没有明显性能税, 再设计低成本版本:

- read_topk warmup: early full/dense 或较大 read_topk, 后期收紧到 2.
- read_topk schedule: 64 -> 16 -> 8 -> 2 或 16 -> 8 -> 2.
- top-k margin telemetry: 监控 top1/top2 margin, 判断 candidate flip 是否集中在低 margin query.
- soft/dense auxiliary screen: 用 dense read 做 teacher/regularizer, 但最终 inference 保持 sparse top-k.

如果 dense-read 4 epoch 失败, 则说明 1ep 正信号只稳定了早期训练, 后续还需要回到 forget/state build/M_state/write support 的机制设计.

## Artifact

- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/preflight-summary.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/variant-summary.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/gate-comparison-summary.csv`
- `docs/artifacts/20260629-02-flash-vqg-gate-topk-mstate-causal-probe/effect-screen-summary.csv`
