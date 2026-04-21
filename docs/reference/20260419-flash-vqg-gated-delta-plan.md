# Flash-VQG Gated-Delta Remote Memory v1 落地说明

## 1. 背景与目标

当前主线 baseline 已经收口到:

- `dense-t025`
- `top2 read`
- `den_aware`
- `shared_den`

结合 `20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md`, `20260413-e5a-top2-audit.md` 和 `20260416-flash-vqg-vnext-architecture-directions.md`, 当前更合理的判断是:

- selector 本身不是最像主瓶颈的部分
- 当前更值得优先升级的是 remote memory mechanics
- 本轮先只推进 write/state update
- 不正式落地新 reader

本轮目标是把 **Flash-VQG 的 Gated-Delta Remote Memory v1** 推到可以开始真实试验的程度, 并完成:

- reference 文档
- 配置改造
- builder / dispatch 接线
- 最小实验矩阵
- 可直接起跑的 shell 脚本

## 2. 本地代码差异

和最初假设相比, 本地真实代码有 4 个关键差异:

1. `zoology/src/flash_vqg/...` 这套路径在本地不存在.
真正核心实现位于旁边仓库 `Flash-VQG`, Zoology 这边主要是 wrapper, suite, manifest 和 analysis.

2. 当前 weighted routing write 只支持 `fox_remote_formula='clr_v1'`.
已有 `clr_delta_v1`, 但它是另一条更窄的硬指派 delta 路径, 不能直接承担当前 `dense-t025` 主线.

3. 当前 `phase2_fox_clr_torch` 的 reader 接口已经稳定.
如果本轮改 reader, 风险会显著上升. 因此本轮保持 reader 接口不变, 只保证 state builder 产出的 `(g, R, L, a)` 仍能被现有 phase2 消费.

4. `beta` 参数原本只在 `fox_remote_formula='clr_delta_v1'` 下初始化.
但这轮 delta 写入挂在 `clr_v1` 下, 所以 attention 初始化处也必须一起扩到 `clr_v1 + residual_update_mode='delta'`.

## 3. 本轮实际落点

### 3.1 总体策略

- `coarse state` 继续走 weighted additive
- 只升级 `CLR residual state`
- forget v1 采用 `code_aware decay`
- 外层保持 `chunkwise/blockwise`
- `coarse` 分支 block 内并行
- `residual` 分支 block 内按 token 顺序更新
- reader 先保持原状

### 3.2 写入模式矩阵

首波正式矩阵固定为 4 组:

1. `additive`
2. `gated-only`
3. `delta-only`
4. `gated-delta`

对应本地配置映射:

| mode | `fox_clr_residual_update_mode` | `fox_clr_residual_forget_mode` |
| --- | --- | --- |
| `additive` | `additive` | `global` |
| `gated-only` | `additive` | `code_aware` |
| `delta-only` | `delta` | `global` |
| `gated-delta` | `delta` | `code_aware` |

## 4. 配置改造

本轮在 `FlashVQGConfig` 上新增 4 个局部开关:

- `fox_clr_residual_update_mode = additive | delta`
- `fox_clr_residual_forget_mode = global | code_aware`
- `fox_clr_residual_write_topk = 4 | 8 | ...`
- `fox_clr_delta_target_mode = residual_to_coarse`

设计原则:

- 不新增新的 `fox_remote_formula`
- `gated-delta` 仍挂在 `clr_v1`
- `clr_delta_v1` 继续保留为旧的硬指派 delta 分支
- `fox_clr_delta_target_mode` v1 只支持 `residual_to_coarse`

## 5. 数学定义

对每个 code `s`, 维护:

- `g_s ∈ R^V`
- `L_s ∈ R`
- `M_s ∈ R^{V×r}`
- `b_s ∈ R^r`

语义:

- `g/L` 是 coarse numerator / denominator
- `M/b` 是 residual value / denominator corrector

现有 reader 继续读取:

```text
Num_s(q) = g_s + M_s α_s(q)
Den_s(q) = L_s + b_s^T α_s(q)
```

### 5.1 coarse 分支

对 block `n` 内的 token `l`, 写入权重为 `W_(n,l,s)`, suffix decay 为 `τ_(n,l)`.

```text
dL_s^(n) = Σ_l τ_(n,l) W_(n,l,s)
dG_s^(n) = Σ_l τ_(n,l) W_(n,l,s) v_(n,l)
L_s^(n+1) = Γ_n L_s^(n) + dL_s^(n)
g_s^(n+1) = Γ_n g_s^(n) + dG_s^(n)
```

这里 `Γ_n` 是 block-level decay. 这部分继续沿用当前 weighted CLR 的 block 内并行聚合.

### 5.2 residual 分支

写入 support 来自写入侧 `W_blk`, 不是读出侧 top2:

```text
S_(n,l) = TopK_s(W_(n,l,s), k_write)
ρ_(n,l,s) = W_(n,l,s) / Σ_{j∈S_(n,l)} W_(n,l,j)
```

局部 residual 坐标:

```text
h_(n,l,s) = B_s^T r_(n,l)
```

其中:

- `r_(n,l)` 对应 `residual_blk`
- `B_s` 对应 `clr_basis[s]`

delta 路径不直接拟合完整 `v`, 而是拟合相对 coarse baseline 的残差:

```text
μ_s^(n) = stopgrad(g_s^(n) / (L_s^(n) + eps))
u_(n,l,s) = v_(n,l) - μ_s^(n)
```

forget:

```text
f_(n,l) = exp(logf_(n,l))
ξ_(n,l,s) = f_(n,l)^(ρ_(n,l,s))
```

delta 步长:

```text
η_(n,l,s) = β_h ρ_(n,l,s) / (||h_(n,l,s)||^2 + eps)
```

delta 更新:

```text
û = M_pre h
e_v = u - û
M_new = M_pre + η e_v h^T

d̂ = b_pre^T h
e_d = y_den - d̂
b_new = b_pre + η e_d h
```

第一版固定:

- `y_den = 1`
- `κ = 1`

## 6. 并行化设计

本轮落地的是:

```text
chunkwise outer + sequential inner
```

也就是:

- 外层继续 blockwise/chunkwise pipeline
- `coarse` 分支 block 内并行
- `residual` 分支 block 内 token loop

这不是全并行方案, 但它是当前最稳的实现点:

- 和现有 `clr_v1` state build 接口兼容
- 能直接做 paired validation
- 容易和旧 `weighted additive` 路径逐步对比

## 7. 接近实现的伪代码

### 7.1 dispatch

```python
if W_blk is not None:
    if update_mode == "additive" and forget_mode == "global":
        return build_states_fox_clr_weighted_torch(...)
    return build_states_fox_clr_gated_delta_weighted_torch(...)

if Delta_blk is not None and fox_remote_formula == "clr_delta_v1":
    return build_states_fox_clr_delta_torch(...)

return build_states_fox_clr_torch(...)
```

### 7.2 new weighted gated-delta builder

```python
for each block n:
    snapshot current (g, M, L, b)
    mu_blk = stopgrad(g_cur / (L_cur + eps)) if update_mode == "delta" else None

    # coarse additive, block-parallel
    dL_n = sum_l suffix_decay * W_blk
    dG_n = einsum(W_blk, V_blk)
    g_cur = block_decay * g_cur + dG_n
    L_cur = block_decay * L_cur + dL_n

    # residual sequential, token loop
    for each token l in block:
        cd *= f_l
        support = topk(W_l, k_write)
        rho = normalize_on_support(W_l[support])

        if forget_mode == "code_aware":
            lazy_support *= exp(logf_l * rho)

        h = project(residual_l, basis[support])
        actual_state = lazy_support * cd_support

        if update_mode == "additive":
            M_support += rho * outer(v_l, h)
            b_support += rho * h
        else:
            u = v_l - mu_blk[support]
            eta = beta * rho / (||h||^2 + eps)
            M_support += eta * outer(u - M_support @ h, h)
            b_support += eta * (1 - b_support·h) * h

        scatter updated support back
        reset cd[support] = 1
```

## 8. 本地代码改造点

### 8.1 Flash-VQG

核心改动位于:

- `Flash-VQG/src/flash_vqg/nn/configuration_flash_vqg.py`
- `Flash-VQG/src/flash_vqg/nn/attn.py`
- `Flash-VQG/src/flash_vqg/nn/attn_fox.py`

具体包括:

- 新增 4 个 config 开关
- `clr_v1` 下新增 weighted gated-delta builder
- `_state_build_execute_pipeline_clr()` 新增 dispatch
- `beta` 参数初始化扩到 `clr_v1 + residual_update_mode='delta'`
- `additive/global` 继续走旧 `build_states_fox_clr_weighted_torch`

### 8.2 Zoology

配套改动位于:

- `zoology/experiments/models_repo.py`
- `zoology/experiments/flash_vqg/flash_vqg_suite.py`
- `zoology/experiments/flash_vqg/run_flash_vqg_suite.py`
- `zoology/experiments/flash_vqg/manifest.py`
- `zoology/analysis/flash_vqg/flash_vqg_analysis_suite.py`

目的:

- 透传新 4 个配置字段
- 让 manifest 记录写入模式
- 让本地 analysis summary 直接展开写入模式字段

## 9. 最小实验矩阵

### 9.1 固定超参

首波统一固定:

- `d_model = 128`
- `num_codebook_vectors = 128`
- `block_len = 32`
- `local_num_blocks = 2`
- `fox_clr_rank = 4`
- `fox_clr_remat_mode = off`
- `vq_score_mode = codebook_dot`
- `vq_weight_mode = dense_softmax`
- `vq_update_mode = grad`
- `vq_softmax_tau = 0.25`
- `vq_topk = 4`
- `fox_remote_read_topk = 2`
- `fox_clr_selector_mode = den_aware`
- `fox_clr_merge_mode = shared_den`
- `fox_clr_residual_write_topk = 4`
- `fox_clr_delta_target_mode = residual_to_coarse`
- `seed = 123`
- `data_seed = 123`
- `max_epochs = 32`

### 9.2 正式训练矩阵

| run_id | update | forget |
| --- | --- | --- |
| `dense-t025-additive-s123-d123` | `additive` | `global` |
| `dense-t025-gated-only-s123-d123` | `additive` | `code_aware` |
| `dense-t025-delta-only-s123-d123` | `delta` | `global` |
| `dense-t025-gated-delta-s123-d123` | `delta` | `code_aware` |

### 9.3 smoke 矩阵

只跑两组:

- `dense-t025-additive-s123-d123`
- `dense-t025-gated-delta-s123-d123`

目的只是先找稳定的:

- `TRAIN_BATCH_SIZE`
- `EVAL_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`

## 10. 实验结果

### 10.1 结果来源

以下表格统一使用本地 `history.csv` 中 epoch 32 的最后一个有效验证值:

- Flash-VQG baseline anchor:
  - `flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12`
  - `dense-t025-s123-d123`
- 写入矩阵前半程:
  - `flash-vqg-20260419-gated-delta-write-2026-04-19-20-14-16`
  - `dense-t025-additive-s123-d123`
  - `dense-t025-gated-only-s123-d123`
- 写入矩阵后半程:
  - `flash-vqg-20260420-gated-delta-write-tail-2026-04-20-09-43-23`
  - `dense-t025-delta-only-s123-d123`
  - `dense-t025-gated-delta-s123-d123`
- GatedDeltaNet 参考:
  - `flash-vqg-20260420-gdn-default-baseline-2026-04-20-08-35-50`
  - `gated_delta_net-default-s123-d123`

### 10.2 主指标结果表

| 组别 | run_id | valid/accuracy | valid/loss | 512x128 | 1024x256 | 相对 baseline acc | 相对 baseline 1024 | 相对 baseline loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline anchor | `dense-t025-s123-d123` | 0.981208 | 0.081622 | 0.987250 | 0.874887 | 0.000000 | 0.000000 | 0.000000 |
| additive | `dense-t025-additive-s123-d123` | 0.976447 | 0.107207 | 0.976234 | 0.854781 | -0.004761 | -0.020105 | +0.025586 |
| gated-only | `dense-t025-gated-only-s123-d123` | 0.965594 | 0.158976 | 0.959719 | 0.793410 | -0.015614 | -0.081477 | +0.077355 |
| delta-only | `dense-t025-delta-only-s123-d123` | 0.963992 | 0.175120 | 0.949719 | 0.810453 | -0.017216 | -0.064434 | +0.093498 |
| gated-delta | `dense-t025-gated-delta-s123-d123` | 0.936662 | 0.476012 | 0.905023 | 0.732363 | -0.044547 | -0.142523 | +0.394391 |
| GatedDeltaNet 参考 | `gated_delta_net-default-s123-d123` | 0.986256 | 0.072575 | 0.999453 | 0.891031 | +0.005047 | +0.016145 | -0.009046 |

### 10.3 稳定性与机制指标表

| 组别 | den_min | nan_inf_count | o_remote_energy_ratio | vq relative_err | vq c_entropy | vq write_entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline anchor | 1.000000 | 0.0 | 0.312489 | 0.015469 | 3.568453 | 2.969296 |
| additive | 1.000000 | 0.0 | 0.385918 | 0.020505 | 3.075862 | 2.816708 |
| gated-only | 1.000000 | 0.0 | 0.366224 | 0.023681 | 2.486758 | 2.240704 |
| delta-only | 1.000000 | 0.0 | 0.327788 | 0.022702 | 2.496436 | 2.214466 |
| gated-delta | 1.000000 | 0.0 | 0.359010 | 0.027510 | 2.322370 | 2.057677 |
| GatedDeltaNet 参考 | - | - | - | - | - | - |

### 10.4 判读摘要

| 问题 | 观察 | 结论 |
| --- | --- | --- |
| `code_aware forget` 单独是否带来收益 | `gated-only` 相对 `additive` 再降 `-0.010853 acc`, `-0.061371 @1024`, `+0.051769 loss` | 第一版 `code_aware forget` 没有给出正收益 |
| `delta update` 单独是否带来收益 | `delta-only` 相对 `additive` 再降 `-0.012455 acc`, `-0.044328 @1024`, `+0.067913 loss` | 第一版 `delta residual update` 没有给出正收益 |
| 两者组合是否互补 | `gated-delta` 是四组里最差, 明显低于 `gated-only` 和 `delta-only` | 当前实现下两者没有形成可用互补, 反而叠加退化 |
| 写入侧改造是否已经超过 baseline | 四组都低于 `dense-t025-s123-d123` | 本轮不能把 writer-only 改造晋升为新主线 |
| Flash-VQG 是否还有提升空间 | GatedDeltaNet 参考高于 baseline `+0.005047 acc`, `+0.016145 @1024` | 说明现有 Flash-VQG 仍有明确提升空间, 问题不在任务上限 |

## 11. 最小运行建议

建议顺序:

1. 先跑 `write-mainline/run_write_smoke.sh`
2. 用 smoke 产出的 batch/GA 跑 `write-mainline/run_write_train.sh`
3. 正式矩阵完成后, 优先读本地产物:
   - `zoology/experiments/flash_vqg/generated/<launch_id>/manifest.json`
   - `zoology/analysis/flash_vqg/results/<launch_id>/launch_analysis/run_summary.csv`
4. 只有 `B1-B4` 不给正信号时, 再起 `diagnostics-read/run_read_diagnostics.sh`

## 12. 当前结论

### 12.1 工程落地结论

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| `clr_v1` 下新写入模式可配置 | 已完成 | `additive / delta` 与 `global / code_aware` 均已接入 |
| weighted gated-delta builder 接入 dispatch | 已完成 | `attn.py` 已按配置分派到新 builder |
| Zoology 可发起 B1-B4 | 已完成 | 四组写入矩阵都已跑通并产出本地结果 |
| manifest / local analysis 可按写入模式读结果 | 已完成 | `run_id` 和本地结果目录均可直接对齐写入模式 |
| read diagnostics 最小入口 | 已完成但未正式起用 | 本轮结果已足够说明 writer-only 不成立, 暂未继续正式扩 reader |

### 12.2 实验结论表

| 结论主题 | 结论 | 依据 |
| --- | --- | --- |
| 本轮 target winner `gated-delta` | 未达预期, 且是四组中最差 | `valid/accuracy = 0.936662`, `1024x256 = 0.732363` |
| `gated-only` | 不支持“memory hygiene 就能显著修复 baseline” | 相对 baseline `acc -0.015614`, `1024 -0.081477` |
| `delta-only` | 不支持“delta 写入单独即可收敛为正收益” | 相对 baseline `acc -0.017216`, `1024 -0.064434` |
| 四组 writer 消融 | 全部失败于 baseline anchor | 四组无一在主指标上转正 |
| 对 Flash-VQG 方向的判断 | 方向未被否定, 但当前瓶颈不只在 writer | GatedDeltaNet 参考仍高于 Flash-VQG baseline |
| 下一步优先级 | 不应继续只堆 writer, 应转向 reader mismatch 或 remote/local integration | writer-only 证据已经足够弱 |

### 12.3 下一步建议表

| 优先级 | 建议 | 理由 |
| --- | --- | --- |
| P0 | 暂停继续扩 `gated-delta writer` 超参 | 主实验已经显示方向不成立 |
| P1 | 进入 reader mismatch 诊断 | 当前更像 read/write coupling 问题, 而不是 selector 单点问题 |
| P1 | 用 `score_only + shared_local_den + top2` 与 `score_only + residual_add + top2` 做最小 read diagnostics | 这是本轮文档已预留的最小诊断入口 |
| P2 | 以 `GatedDeltaNet` 单点为外部参考线 | 说明当前任务仍有提升空间, 可以作为后续 gate |

### 12.4 深入分析: 为什么当前方案打不过 baseline

| 分析维度 | 观察 | 判断 |
| --- | --- | --- |
| 主任务结果 | 四组 writer 变体都低于 baseline, 且 `gated-delta` 最差 | 当前方案不是“略输”, 而是整体未形成可用收益 |
| 数值稳定性 | 四组的 `valid/attn/den_min` 都是 `1.0`, `nan_inf_count` 都是 `0.0` | 问题不是 NaN, Inf 或 denominator collapse |
| remote 参与度 | `o_remote_energy_ratio` 没有塌掉, `additive / gated-only / gated-delta` 甚至高于 baseline | 问题不是 remote path 没被用到, 而是“用了更多 remote 但质量更差” |
| memory 拟合质量 | `vq/relative_err_mean` 四组都高于 baseline, 其中 `gated-delta` 最差 | residual memory 被写得更差, 不是更好 |
| code usage 多样性 | `vq/c_entropy` 与 `vq/write_entropy_mean` 四组都低于 baseline | code usage 更集中, 覆盖更差, memory 更容易过尖 |
| 写入尖锐度 | `vq/write_top1_mass_mean`: baseline `0.281898`, additive `0.284336`, gated-only `0.529247`, delta-only `0.534106`, gated-delta `0.552621` | 新 writer 尤其是 `code_aware` 和 `delta` 组合后, 写入显著变尖 |

| 关键判断 | 证据 | 结论 |
| --- | --- | --- |
| 不是单独 `delta` 的问题 | 连 `additive` 组都低于 baseline `acc -0.004761`, `1024 -0.020105` | 更基础的问题出在“新 residual topk-write 框架”本身 |
| 不是单独 `forget` 的问题 | `gated-only` 比 `additive` 继续退化 | 第一版 `code_aware forget` 没有改善 memory hygiene |
| 不是单独 `reader` 完全失效 | remote energy 仍在, 但任务结果下降 | 更像 writer 与现有 reader 的语义耦合不对 |
| 不是任务已接近 ceiling | GatedDeltaNet 参考高于 baseline `+0.005047 acc`, `+0.016145 @1024` | Flash-VQG 仍有提升空间, 问题在当前工作点 |

| 当前最可能机制解释 | 说明 | 对后续动作的含义 |
| --- | --- | --- |
| residual 写入从 dense 变成 topk, 先天降低覆盖 | baseline 的历史 residual write 是 dense weighted, 当前新线从 `additive` 起就变成了 residual topk-write | 应先解释“为什么 topk residual write 本身已经输 baseline”, 再谈更复杂的 gated-delta |
| writer support 与 reader support mismatch | 写入侧按 `W_blk` 的 support 更新 state, 读出侧仍按 `den_aware + shared_den + top2` 读 | 下一步优先做 read diagnostics, 而不是继续只调 writer |
| `code_aware forget + delta` 把写入推得过尖 | `write_top1_mass_mean` 从 baseline 的 `0.28` 量级升到 `0.53 ~ 0.55` | 当前 state dynamics 可能过于局部化, 导致远程记忆泛化和校准同时变差 |
| remote 质量而不是 remote 强度在拖后腿 | `o_remote_energy_ratio` 未塌, 但 `relative_err` 和任务指标同步变坏 | 后续应优先关注 read/write coupling 和 state semantics, 而不是盲目提升 remote 占比 |
