# gd_residual_v1 reference path debug plan

日期: 2026-04-26

## 1. 结论

当前不应该直接启动 `gd_residual_v1` MQAR 4 epoch 正式训练.

阻断原因不是 baseline 或候选超参定义不清楚, 而是 `rank=16, write_topk=4, cb256, read_topk=2, dense_softmax, tau=0.25` 主配置在 2080 Ti 11GB 上的 PyTorch reference path 不满足可训练性:

- `64x4` 已实测 backward OOM, 尝试分配 `32.00 GiB`.
- `32x8` 已实测 backward OOM, 尝试分配 `16.00 GiB`.
- `16x16` 已实测能完成 smoke, 但 24 个 micro-batch 约 `9.7 min`, 线性估算约 `76 h/epoch/run`, `304 h/4epoch/run`.
- 当前没有 gd 4 epoch 正式训练结果, 不能把 smoke 的 `valid/accuracy=0` 或 `valid/loss≈9.04` 当成 4 epoch 质量结论.

下一轮优先级:

1. 先定位并修复 `rank=16, write_topk=4` 主配置的 reference path 显存与吞吐问题.
2. `rank=8` 或 `write_topk=2` 只能作为 debug-only attribution, 不能作为 official candidate, 不能写成主结果.
3. 第一优先可疑点是 phase2 residual read 的 expanded `gather` backward materialization, 第二优先是 `event_pack` + `grouped_chunk_torch_ref` 的逐 event 小算子与 Python/CPU sync.

## 2. 读取口径和实际读过的文件

本地工作树状态:

- `/home/lyj/mnt/project/zoology`: 当前分支 `flash-vqg`.
- `/home/lyj/mnt/project/Flash-VQG`: 当前分支 `20260425-gd-residual-v1-codex`.
- `origin/main` 只包含本轮所需文件的一部分: Flash-VQG 的 `attn.py`, `attn_fox.py`, 以及 zoology baseline 结果文件. 本轮候选文档, 测试和脚本不在 `origin/main` 可读范围内.
- 因此本报告基于当前工作树中的指定文件. 若需要严格 main 分支口径, 应先把这些候选文件合入 main 或提供 main 上对应 ref.

实际读过的仓库文件:

- `/home/lyj/mnt/project/Flash-VQG/docs/20260425-flash-vqg-gated-delta-v1-math-plan-final.md`
- `/home/lyj/mnt/project/Flash-VQG/docs/20260425-flash-vqg-gated-delta-v1-codex-blueprint.md`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn_fox.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py`
- `/home/lyj/mnt/project/Flash-VQG/tests/test_fox_gd_residual_v1.py`
- `/home/lyj/mnt/project/zoology/docs/20260425-gd-residual-v1-4epoch-t025-cb256-report.md`
- `/home/lyj/mnt/project/zoology/docs/20260425-gd-residual-v1-mqar-performance-report.md`
- `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py`
- `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh`
- `/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh`
- `/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45/dense-t025-cb256-s123-d123/data/metadata.json`
- `/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45/dense-t025-cb256-s123-d123/data/history.csv`

未读取额外项目源码文件. `history.csv` 有 359195 行, 本轮只读取表头和 baseline 前 4 epoch 相关 metric 行.

## 3. Baseline 配置和前 4 epoch 指标

baseline 只读取, 没有重跑.

baseline 路径:

- metadata: `zoology/analysis/flash_vqg/results/flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45/dense-t025-cb256-s123-d123/data/metadata.json`
- history: `zoology/analysis/flash_vqg/results/flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45/dense-t025-cb256-s123-d123/data/history.csv`

baseline 配置摘要:

| 项 | 值 |
| --- | --- |
| run_id | `dense-t025-cb256-s123-d123` |
| formula | `clr_v1` |
| d_model / n_layers | `128 / 2` |
| heads / key_dim / value_dim | `2 / 64 / 64` |
| block_len / local_num_blocks | `32 / 2` |
| num_codebook_vectors | `256` |
| VQ | `codebook_dot`, `dense_softmax`, `grad`, `tau=0.25`, `vq_topk=4` |
| fox_remote_read_topk | `2` |
| train_batch_size / eval_batch_size / gradient_accumulation_steps | `64 / 16 / 4` |
| effective train batch | `256` |
| lr / weight_decay | `1e-3 / 0.1` |
| seed / data_seed | `123 / 123` |
| train_batch_order | `global_shuffle` |
| max_epochs | `32` |

baseline 前 4 epoch 指标:

| epoch | step | valid/loss | valid/accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 704 | 1.492455 | 0.776821 |
| 2 | 1409 | 0.519401 | 0.914123 |
| 3 | 2114 | 0.299164 | 0.951626 |
| 4 | 2819 | 0.237862 | 0.961423 |

epoch 4 的长序列 slice 仍是主要难点:

| slice | valid accuracy |
| --- | ---: |
| `input_seq_len=1024` | 0.774844 |
| `mqar_case=1024x256` | 0.774844 |
| `input_seq_len=512` | 0.967660 |
| `mqar_case=512x128` | 0.952477 |

## 4. gd candidates 配置

正式候选:

| run_id | mu_min_count |
| --- | ---: |
| `gd-r16-wk4-mu015-t025-cb256-s123-d123` | `0.15` |
| `gd-r16-wk4-mu01-t025-cb256-s123-d123` | `0.1` |

必须保持的 official candidate 语义:

| 项 | 值 |
| --- | --- |
| formula | `gd_residual_v1` |
| fox_gd_residual_rank | `16` |
| fox_gd_residual_write_topk | `4` |
| fox_remote_read_topk | `2` |
| num_codebook_vectors | `256` |
| vq_weight_mode | `dense_softmax` |
| vq_softmax_tau | `0.25` |
| vq_score_mode / vq_update_mode | `codebook_dot / grad` |
| builder / pack / chunk | `grouped_chunk_torch_ref / semivec_ref / 64` |
| lambda_init / beta_init | `0.05 / 0.5` |
| norm_with_gain | `false` |
| use_separate_addr_codebook | `false` |
| addr_eps / den_eps / rho_eps | `1e-6 / 1e-6 / 1e-12` |
| d_model / n_layers | `128 / 2` |
| heads / key_dim / value_dim | `2 / 64 / 64` |
| block_len / local_num_blocks | `32 / 2` |
| lr / weight_decay | `1e-3 / 0.1` |
| seed / data_seed | `123 / 123` |
| train_batch_order | `global_shuffle` |
| intended train_batch_size / eval_batch_size / ga | `64 / 16 / 4` |

本轮可运行 smoke 曾临时使用 `16x16`, 只是为了保持 effective batch `256` 并通过 smoke gate. 这不改变 official candidate 的正式目标, 也不是正式主训练配置.

## 5. 已实测事实 vs 线性估算

已实测事实:

| 配置 | 结果 | 备注 |
| --- | --- | --- |
| `64x4`, `rank=16`, `write_topk=4`, `read_topk=2`, `cb256` | backward OOM | 需要分配 `32.00 GiB`, GPU 可用约 `10.74 GiB` |
| `32x8`, `rank=16`, `write_topk=4`, `read_topk=2`, `cb256` | backward OOM | 需要分配 `16.00 GiB`, GPU 可用约 `10.74 GiB` |
| `16x16`, `mu=0.15` | smoke completed | 24 micro-batch train wall `9m50s`, avg `24.59s`, sampled peak `~10363 MiB` |
| `16x16`, `mu=0.1` | smoke completed | 24 micro-batch train wall `9m40s`, avg `24.19s`, sampled peak `~10351 MiB` |
| `16x16` smoke final valid | `valid/loss≈9.043`, `valid/accuracy=0.0` | 只跑 24 个 micro-batch, 不能作为 4 epoch 质量结论 |

线性估算:

- baseline 每 epoch 约 `704` optimizer steps.
- `16x16` 每 optimizer step 有 `16` 个 micro-batch, 所以每 epoch 约 `704 * 16 = 11264` 个 micro-batch.
- 用两个 smoke 的平均 micro-batch 时间约 `24.4s` 估算:
  - 每 epoch/run: `11264 * 24.4s ≈ 76 h`.
  - 4 epoch/run: `≈ 304 h`, 即 `≈ 12.7 days`.
  - 两个候选若各占一张 GPU 并行跑, wall-clock 仍约 `12.7 days`.
  - 若串行跑两个候选, wall-clock 约 `25.4 days`.

必须强调:

- `64x4` 和 `32x8` 是实测 OOM, 不是估算.
- `16x16` 的 `76 h/epoch/run` 和 `304 h/4epoch/run` 是线性估算, 不是正式训练实测.
- 本轮没有启动正式 4 epoch, 因此没有 gd 4 epoch 曲线, 也没有可和 baseline epoch 1-4 对齐比较的 gd result.

## 6. OOM 和吞吐结论

显存结论:

- official intended `64x4` 不能训练, backward 已 OOM.
- 降到 `32x8` 仍不能训练, backward 已 OOM.
- `16x16` 虽能 smoke, 但 sampled peak 已约 `10.35 GiB`, 接近 2080 Ti 11GB 上限, 留给 allocator 碎片, eval, metrics 和不同 seq bucket 的余量很小.

吞吐结论:

- `16x16` smoke 的 micro-batch 约 `24.4s`, 线性估算 4 epoch/run 约 `304h`.
- 这会把一次双候选筛选推到约两周 wall-clock, 还没有解决 OOM 的 official intended batch.
- 因此直接跑正式 4 epoch 的机会成本过高, 且失败概率高.

为什么不启动正式 4 epoch:

1. `64x4` 是对齐 baseline 的正式 batch 口径, 但已在 backward OOM.
2. `32x8` 是保留 effective batch 的直接缩小 micro-batch 尝试, 仍 OOM, 说明不是简单把 batch 减半就能进入主训练.
3. `16x16` 只是 smoke 可完成, 但吞吐不可接受, 且显存接近上限. 启动 4 epoch 会消耗约 `12.7 days` 双卡 wall-clock, 但产出仍可能只是中途失败或无法比较的低效曲线.
4. smoke 结果不是正式训练结果, 不能证明候选质量失败, 也不能证明候选质量可行. 当前唯一稳健结论是 reference path 不可训练.

## 7. 代码侧瓶颈排序

### 7.1 第一优先: phase2 residual read 的 expanded gather backward

`phase2_fox_gd_residual_correction_torch` 中 residual read 关键路径:

```python
codebook_exp = codebook.view(1, H, 1, 1, num_codes, -1).expand(B, H, N, L, num_codes, -1)
C_sel = torch.gather(codebook_exp, dim=4, index=...)

M_sel = torch.gather(
    M_remote.unsqueeze(3).expand(B, H, N, L, num_codes, d_v, r),
    dim=4,
    index=...
)
proposal = torch.einsum("bhnlkvr,bhnlkr->bhnlkv", M_sel.float(), d_read)
```

风险点:

- forward 只选 `read_topk=2`, 但 source 被 expand 成含完整 `num_codes=256` 和 `L=32` 的视图.
- `torch.gather` 的 backward 很可能为 expanded source 形状建立 dense grad buffer, 再沿 expand 维度归并.
- performance report 已观察到 B8/T128 profiler 中 `aten::gather_backward` CUDA memory usage 约 `1.06GB`.
- 按 batch 和 sequence bucket 线性放大, 该路径可以解释 `32x8` 需要 `16GiB`, `64x4` 需要 `32GiB` 这类大块 backward allocation.
- `M_state` 形状为 `[B,H,N,S,d_v,r]`. 在 B64,T256,H2,N8,S256,d_v64,r16 下, 单层 `M_state` 就约 `268M` elements, fp32 约 `1.0GiB`, backward 中 expanded gather 源会再乘上 `L` 和相关中间项.

判断:

- 当前 OOM 最可能先来自 phase2 residual read 的 `M_sel` gather backward materialization.
- `read_topk=2` 降低了 forward selected proposal 的规模, 但当前实现仍可能在 backward 为 expanded dense source 付出接近 dense read 的显存代价.

### 7.2 第二优先: `grouped_chunk_torch_ref` 实际是逐 event loop

数学蓝图中的 grouped chunk solver 应使用 chunk-local triangular solve. 但当前 `grouped_chunk_torch_ref` 实现中:

```python
for g in range(M_ent_pack.size(0)):
    state_f32 = M_ent_pack[g].float()
    for e in range(start, end):
        state_f32 = torch.exp(logabar_pack[e].float()) * state_f32
        d = D_pack[e].float()
        u = U_pack[e].float()
        zeta = zeta_pack[e].float()
        pred = torch.matmul(state_f32, d)
        state_f32 = state_f32 + zeta * torch.outer(u - pred, d)
```

风险点:

- `chunk_size` 参数目前没有发挥 chunk 矩阵公式的作用.
- 每个 event 都产生 `exp`, `matmul`, `outer`, elementwise update 和 assignment.
- dense_softmax 下每个有效 token 对每个 head 发出 `write_topk=4` 个 positive events. 对 B16,H2,L32,每 block event 数约 `4096`; B64 时每 block约 `16384`.
- backward 会保留大量小 op 的 autograd graph, 吞吐差, GPU 利用率低.
- performance report 中 `gd_residual/grouped_chunk` 和 `gd_residual/event_pack` 合计约占 B8/T128 profiler self CUDA time 的 `97%`.

判断:

- 该路径是吞吐第一嫌疑.
- 它也会增加 backward graph 规模和 allocator 压力, 但 OOM 中 `16GiB/32GiB` 大块 allocation 更像 phase2 gather backward.

### 7.3 第三优先: `event_pack` 的 metadata 和 pack materialization

`_build_gd_event_pack_semivec_ref` 当前路径:

```python
top_vals, top_idx = torch.topk(W_blk.float(), k=min(write_topk, S), dim=-1)
rho = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(eps_rho)
event_mask = rho > 0.0
event_pos = event_mask.nonzero(as_tuple=False)
sort_key = group_id_e * (L + 1) + ell_e
order = torch.argsort(sort_key)
...
K_e = K_blk[b_e, h_e, ell_e]
V_e = V_blk[b_e, h_e, ell_e]
C_e = codebook[h_e, s_e]
R_e = addr_proj[h_e]
z = torch.einsum("ed,edr->er", K_e - C_e, R_e)
D_pack = _gd_l2_normalize_eps(z, eps_addr)
U_pack = V_e - mu_pre[b_e, h_e, s_e]
zeta_pack = beta_blk[b_e, h_e, ell_e] * rho_e
```

风险点:

- `W_blk.float()` materializes `[B,H,L,S]` per block.
- `D_pack [E,r]`, `U_pack [E,d_v]`, `zeta_pack [E]`, `M_ent_pack [G,d_v,r]` 都会进入后续 autograd.
- group loop 中 `int(cu_seqlens[g].item())`, `int(group_ids[g].item())`, `int(ell_e[e].item())` 会造成 CPU sync 和大量 small copy.
- profiler 已观察到 `aten::copy_` 约 `163537` calls, `aten::item/_local_scalar_dense` 约 `31368` calls, `aten::outer` 约 `14336` calls, `aten::matmul` 约 `8197` calls.

autograd 判断:

- `top_idx`, `event_pos`, `group_id`, `order`, `cu_seqlens`, `event_ell` 是 routing metadata, 不需要梯度.
- `rho_e` 来自 selected `top_vals`, 需要保留对 `W_blk` 的梯度, 因为 write responsibility 是训练路径的一部分.
- `K_e`, `V_e`, `C_e`, `R_e`, `beta_blk` 和 `M_ent_pack` 需要梯度.
- `_gd_mu_pre(...).detach()` 已符合 block-entry frozen baseline 语义, 不应去掉.

判断:

- metadata 应尽量 `no_grad` 或显式 detach, 但不能 detach `rho_e` 本身.
- 进一步应减少 group loop 的 `.item()` 和 per-element `copy_`.

### 7.4 第四优先: 大状态和重复 dense materialization

state build 维护:

```python
G_state = [B,H,N,S,d_v]
L_state = [B,H,N,S]
M_state = [B,H,N,S,d_v,r]
M_default = full_block_alpha[:, :, None, None, None] * M_cur
flat = M_default.reshape(B * H * S, d_v, r)
flat[pack.group_ids] = M_bd_pack.float()
```

风险点:

- `M_state` 是主配置下最大的常驻激活之一, 且 rank 线性放大.
- `M_default` 每个 block 都 materialize `[B,H,S,d_v,r]`.
- `flat[pack.group_ids] = ...` 触发 indexed assignment/copy.
- `G_state/L_state/M_state` 后续 phase2 还会 `index_select` 成 remote states.

判断:

- 这部分是显存基线压力, 与 phase2 expanded gather backward 叠加后触发 OOM.
- `rank=8` debug-only 对照可帮助确认 rank 维度贡献, 但不能作为 official candidate.

### 7.5 metrics 和 diagnostics

当前 `attn.py` 中:

```python
gd_collect_metrics = (
    self.enable_layer_metrics
    and _get_fox_phase2_metrics_mode(self.config) != "off"
)
```

测试覆盖:

- `enable_layer_metrics=False` 时 gd residual metrics 不出现.
- `fox_phase2_metrics_mode="off"` 时 gd residual metrics 不出现.
- `FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS=1` 时 debug metrics 才出现.
- 默认不会输出 `gd_residual_debug_*`.

判断:

- metrics/diagnostics 不是当前 OOM 的第一嫌疑.
- 下一轮 profile 必须默认 `FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS=0`, 只在 debug-only 单点打开.

## 8. 下一轮最小 debug 实验矩阵

所有 official-candidate gate 必须保持:

```text
rank=16
write_topk=4
cb256
read_topk=2
vq_weight_mode=dense_softmax
vq_softmax_tau=0.25
```

### 8.1 official-candidate gate

| ID | 配置 | batch / seq | 目的 | 通过条件 |
| --- | --- | --- | --- | --- |
| G0 | official main config, metrics off, diagnostics off | `B=16,T=256`, 1-2 micro-batch | 拆 forward/backward/optimizer peak memory 和 wall time | 不 OOM, 记录 phase peak, 不作为质量结论 |
| G1 | official main config, metrics off, diagnostics off | `B=16,T=128`, torch profiler 1 micro-batch | 复核 `gather_backward`, `event_pack`, `grouped_chunk`, `copy_`, `outer`, `matmul` 排序 | profiler 能完成, 不用 full batch profiler |
| G2 | official main config, diagnostics on | `B=8,T=128`, 1 micro-batch | 记录 event_count/group_count/phase wall time | 只用于定位, 诊断默认仍关闭 |
| G3 | official main config, apply phase2 flat-gather patch 后 | `B=16,T=256` | 验证 `M_sel/C_sel` gather backward 显存是否下降 | peak reserved 显著低于当前, backward 不出现 16GiB/32GiB allocation |
| G4 | official main config, apply phase2 flat-gather patch 后 | `32x8` smoke | official candidate memory gate | backward 不 OOM, 24 micro-batch wall time可用于新线性估算 |
| G5 | official main config, apply phase2 flat-gather patch 后 | `64x4` smoke | baseline batch 对齐 gate | 若通过, 才考虑正式 4 epoch |
| G6 | official main config, metrics on, diagnostics off | 已通过的最大 batch | 检查训练日志 metrics overhead | metrics overhead < 5-10%, 无 NaN/Inf |

说明:

- G0-G6 都不改变 official candidate 的数学语义.
- G4/G5 是是否恢复正式候选训练资格的关键 gate.
- 如果 G4/G5 仍 OOM, 不启动 4 epoch.

### 8.2 debug-only attribution

以下实验只能标注为:

```text
debug-only
bottleneck attribution only
not official candidate
not main result
```

| ID | 临时变更 | 固定项 | 目的 | 禁止用途 |
| --- | --- | --- | --- | --- |
| D1 | `rank=8`, `write_topk=4` | `cb256, read_topk=2, dense_softmax, tau=0.25` | 测 rank 对 `M_state`, `M_sel`, `gather_backward` 显存的线性贡献 | 不能作为候选质量结果 |
| D2 | `rank=16`, `write_topk=2` | `cb256, read_topk=2, dense_softmax, tau=0.25` | 测 event_count 对 `event_pack/grouped_chunk` 吞吐的贡献 | 不能替代 `write_topk=4` |
| D3 | `rank=8`, `write_topk=2` | `cb256, read_topk=2, dense_softmax, tau=0.25` | 获得 memory/time lower bound, 验证瓶颈是否按 rank/event 缩放 | 不能进入报告主曲线 |
| D4 | `builder=token_step_ref`, tiny shape | 小 batch, 小 T | 只做 correctness 和 profiler sanity, 对比 grouped path 小算子结构 | 不能做训练性能外推 |
| D5 | official main config + diagnostics on | `B=8,T=128` | 只打开 `FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS=1` 复核 event/group 计数 | 不能作为常规训练设置 |

## 9. 继续和停止规则

停止规则:

- `64x4` 和 `32x8` 任一仍在 backward OOM 时, 不启动正式 4 epoch.
- 如果 only `16x16` 可跑, 且线性估算仍超过 `72h/4epoch/run`, 不启动正式 4 epoch.
- 如果修复后 `B16,T256` 仍出现接近整卡的 peak reserved, 先继续 profile, 不进入 smoke 扩展.
- 如果 debug-only `rank=8` 或 `write_topk=2` 通过但 official main config 不通过, 只能说明瓶颈归因成立, 不能推进缩水配置为候选.
- 如果 metrics on 相比 metrics off 开销超过 `10%` 或引入明显显存峰值, official training 前必须先关闭 diagnostics 并重新评估 metrics 策略.

继续规则:

- phase2 flat-gather patch 后, official `B16,T256` backward peak 必须显著下降, 且无数值差异.
- `32x8` smoke 必须通过, 才允许重新估算 4 epoch 成本.
- 优先争取 `64x4` 通过, 因为它对齐 baseline micro-batch 口径.
- 正式 4 epoch 启动前, 至少需要一个 official-candidate gate 给出可接受的 wall-clock 估算. 建议阈值是 `<=72h/4epoch/run`; 超过该阈值继续 debug, 不消耗双卡跑长训.
- 通过 gate 后, 仍只启动两个正式候选: `mu=0.15` 和 `mu=0.1`, 且保留 `rank=16, write_topk=4`.

## 10. reference path 低风险优化方向

以下方向只允许做 PyTorch 侧修复, 不改数学, 不改 block-entry frozen baseline, 不改 event compression/grouped-chunk 与 token-step reference 等价目标, 不写 Triton/CUDA/custom backward.

### 10.1 phase2 residual read flat gather

目标:

- 替换 `codebook_exp.expand(...).gather(...)` 和 `M_remote.unsqueeze(3).expand(...).gather(...)`.
- 使用 flatten index 从 compact source 选择:
  - `codebook` source 形状保持 `[H,S,d_k]` 或 `[H*S,d_k]`.
  - `M_remote` source 形状保持 `[B,H,N,S,d_v,r]` 或 `[B*H*N*S,d_v,r]`.
  - selected output 仍为 `[B,H,N,L,K_read,...]`.
- backward 只对 compact source 建 grad, 避免 expanded source 的 `[B,H,N,L,S,d_v,r]` dense grad buffer.

验证:

- forward output 与旧实现 `assert_close`.
- codebook, addr_proj, lambda, beta, M_state 相关梯度存在且 finite.
- profiler 中 `aten::gather_backward` memory usage 明显下降.
- `tests/test_fox_gd_residual_v1.py` 增加或复用 backward smoke.

### 10.2 routing metadata no_grad / detach

目标:

- `top_idx`, `event_pos`, `sort_key`, `order`, `group_ids`, `cu_seqlens`, `event_ell` 放入 `torch.no_grad()` 或显式 detach.
- 不 detach selected `rho_e`, 因为 `zeta=beta*rho` 应保留 write routing 梯度.
- 不改变 `_gd_mu_pre(...).detach()`, 它是 frozen baseline 语义.

收益:

- 减少 autograd graph 噪声.
- 降低 CPU sync 和 metadata 生命周期风险.

### 10.3 vectorize event logabar / alpha_tail

目标:

- 当前 semivec pack 仍按 group 和 event 使用 `.item()` 计算 `logabar_pack` 和 `alpha_tail_pack`.
- 可用 sorted event 的 previous ell tensor 计算:
  - group first event 的 prev ell 为 `0`.
  - 同 group 后续 event 的 prev ell 为上一 event ell + 1.
  - `logabar = prefix[..., ell+1] - prefix[..., prev]`.
- tail 使用每组最后 event ell + 1 一次性 gather.

收益:

- 降低 `aten::item/_local_scalar_dense`, `copy_`, 小 kernel 和 Python loop.
- 不改变 pack 输出语义.

### 10.4 grouped_chunk_torch_ref 恢复 chunk-local 矩阵公式

当前实现虽然函数名是 `grouped_chunk_torch_ref`, 但实际逐 event loop 更新. 可按数学方案恢复 chunk-local formula:

```text
L_c = I + tril(Z_c K_c K_c^T, -1)
W_c = solve_triangular(L_c, Z_c K_c)
Vtilde_c = solve_triangular(L_c, Xi_c Z_c V_c)
H_bd = Gamma * (H_ent - (H_ent @ W_c.T) @ K_c) + Vtilde_c.T @ K_c
```

约束:

- 仍用 PyTorch.
- 不自定义 backward.
- 与 token-step reference `assert_close`.
- 先在 small shape 上开启, 再跑 B8/T128 profiler.

收益:

- 让 `chunk_size` 真正生效.
- 减少 per-event `outer/matmul/copy_` 小算子数量.
- 可能改善 backward graph 规模.

### 10.5 activation checkpoint / recompute

只作为第二阶段 memory lever:

- 对 residual state build 或 phase2 residual read 做 checkpoint 可降低 activation 保存, 但会在 backward 重算 event_pack/grouped_chunk, 可能进一步拖慢.
- 应先修 phase2 expanded gather, 再评估 checkpoint.
- checkpoint run 只能作为 memory debug 或 gate, 不应掩盖吞吐不可接受的问题.

### 10.6 profile 拆分

下一轮 profile 必须拆:

- state build forward peak.
- phase2 residual read forward peak.
- loss backward peak.
- optimizer step peak.
- metrics collect peak.

建议策略:

- full batch 不开 torch profiler, 只记 CUDA memory snapshots 和 wall time.
- torch profiler 只跑 B8/T128 单点, 避免 profiler overhead 让实验不可完成.
- diagnostics 默认关闭, 只在 D5 单点打开.

## 11. 代码改动摘要和回退方式

本轮实际改动:

- 新增文档: `docs/20260425-gd-residual-v1-reference-path-debug-plan.md`.
- 未修改 Flash-VQG 源码.
- 未修改 zoology 训练代码.
- 未重跑 baseline.
- 未启动正式 4 epoch.

本轮回退方式:

```bash
rm docs/20260425-gd-residual-v1-reference-path-debug-plan.md
```

下一轮若实施代码修复, 建议 diff 范围:

- Flash-VQG `src/flash_vqg/nn/attn_fox.py`:
  - 增加 phase2 flat gather helper.
  - 对 event routing metadata 加 no_grad/detach.
  - 可选增加 vectorized logabar/alpha_tail.
  - 可选增加真正 chunk-local grouped chunk solver.
- Flash-VQG `tests/test_fox_gd_residual_v1.py`:
  - 增加 flat gather forward/backward 等价测试.
  - 增加 metadata no_grad 不破坏 selected routing grad 的测试.
  - 增加 chunk solver 与 token-step reference 等价测试.
- zoology profile runner 或脚本:
  - 如需 phase memory split, 只加 opt-in debug instrumentation, 默认关闭.

下一轮代码回退方式:

- 每个优化应独立 commit 或独立 patch.
- 若 flat gather 引入数值或梯度差异, revert 对应 helper 和调用点.
- 若 vectorized event pack 与 loop/semivec reference 不一致, revert pack 改动.
- 若 chunk solver 性能更差或不稳定, 保留旧逐 event path 作为 reference fallback, 但不能 silent fallback 到 legacy/CLR.

## 12. 核心原则

- 正式训练不要通过缩小 `rank` 或 `write_topk` 绕过问题.
- 当前要解决的是 `rank=16, write_topk=4` 主配置的 PyTorch reference path 显存和吞吐.
- `rank=8` 和 `write_topk=2` 只服务瓶颈归因, 不服务候选结论.
- 不重新设计 `gd_residual_v1` 数学.
- 不改变 block-entry frozen baseline 语义.
- 不改变 event compression/grouped-chunk 与 token-step reference 等价目标.
- 不改变 residual correction branch 公式.
- 不写 Triton/CUDA/custom backward.
- 不改 legacy/CLR 语义.
- 不允许 silent fallback.
- 不重跑 baseline.
