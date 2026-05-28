# FLA K-blocked GDN kernel 简报

## 背景

`docs/20260526-gdn-flash-fairness-phase5-phase6-report.md` 提出, 如果论文需要 true per-head `K=1024,V=64` GDN baseline, 应新开独立 FLA kernel research goal. 原因是官方 FLA `chunk_gated_delta_rule` 训练路径的 hidden-state update kernel 限制 per-head `K<=256`, 使 `ek8-ev2` 和 `ek16-ev1` 不能作为 true expanded-K GDN endpoint 直接训练.

本次实验实现并验证了 research-stage K-blocked state-update path, 目标是在不改变 Gated Delta Rule 数学语义的前提下, 让 large-K hidden state 按 K block 分段计算.

## 实现摘要

K-blocked kernel 的核心做法是按 K tile 循环更新 state. 一个 Triton program 负责一个 sequence/value-head/V tile stream, program 内按 chunk 前进, 再按 K block 循环. Forward 跨 K block 累积 `w_k @ state_k`, 得到 delta 后更新 state; backward 对应做跨 K block 的梯度累积和 state 梯度传播.

工程上保留 `K<=256` 原路径不变, 只给 large-K 走新路径. 3090 上 shared-memory repair 后将 large-K backward launch 的 `num_stages` 调整为 2. 为降低 fp32 下默认 TF32 风格 `tl.dot` 带来的 `dk/dbeta` 偏差, large-K kernel 的 `tl.dot` 暂时使用 `input_precision="ieee"`.

## Correctness 状态

已有 CUDA correctness 和 wrapper smoke 通过. `K=512,V=128` 与 `K=1024,V=64` 的 large-K case 按 FLA-style tolerance 通过. 但 `K=1024,V=64,fp32` 在 strict `abs_tol=0.005` 下仍有轻微超限:

| gradient | max_abs | over strict 0.005 |
|---|---:|---:|
| `dk` | 0.007466 | 0.002466 |
| `dbeta` | 0.005418 | 0.000418 |

因此当前结论是: research training/eval evidence 可用, upstream-ready correctness 仍为 no-go.

## Longer-MQAR 结果

| model | active cap | 1024x256 | 2048x512 | 4096x1024 | 8190x512 | 8190x2047 |
|---|---:|---:|---:|---:|---:|---:|
| `ek16-ev1-kblocked` | 131,072 | 0.791492 | 0.092754 | 0.002947 | 0.062453 | 0.000530 |
| `ek8-ev2-kblocked` | 131,072 | 0.899836 | 0.203586 | 0.007020 | 0.121328 | 0.000472 |
| Flash `cb64-r16` | 131,072 | 0.969625 | 0.822645 | 0.468924 | 0.717262 | 0.162288 |
| Flash `cb256-r4` | 131,072 | 0.894156 | 0.666691 | 0.380678 | 0.566734 | 0.171702 |
| GDNXK `h4-ek8-ev4` | 131,072 | 0.956945 | 0.410500 | 0.053760 | 0.254805 | 0.002526 |

`ek8-ev2-kblocked` 明显好于 `ek16-ev1-kblocked`, 说明 `K=1024,V=64` 这个容量形状在当前训练设置下可能更差. 但两者都没有接近 Flash OOD 表现, 也没有稳定超过之前的 GDN/GDNXK 变体.

## 结论

这次实验回答了两个问题:

1. true expanded-K GDN 能不能通过 K-blocked kernel 跑起来: 能, 但 kernel 仍是 research-stage.
2. true expanded-K GDN 是否接近 Flash: 当前 evidence 显示没有. `ek8-ev2` 比 `ek16-ev1` 好, 但仍远低于 Flash `cb64-r16` 在 longer-MQAR 上的表现.

后续如果要考虑合入, 应先进入 patch review/export gate, 并把 `input_precision="ieee"` 是否写死, `K=1024,V=64` strict fp32 gradient error, API/dispatch 边界和旧路径 regression 作为重点 review 项.

## Artifact

- 主 artifact: `docs/artifacts/20260528-fla-kblocked-gdn-kernel/`
- expanded-K 总表: `docs/artifacts/gdn-expanded-k/gdn-expanded-k-summary.csv`
- longer-MQAR 索引: `docs/artifacts/longer-mqar/kblocked-gdn-20260528/`
- 可复用脚本入口: `zoology/experiments/flash_vqg/scripts/20260528-fla-kblocked-gdn-kernel/`
- 标准 generated 配置: `zoology/experiments/flash_vqg/generated/flash-vqg-20260528-kblocked-probe-*/`
