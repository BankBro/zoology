# gd_residual_v1 grouped_chunk_torch_ref bucketed PyTorch reference 优化实施文档

日期: 2026-05-14

适用仓库:

- `/home/lyj/mnt/project/Flash-VQG`
- `/home/lyj/mnt/project/zoology`

建议放入 Flash-VQG 仓库:

- `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-implementation-plan.md`

---

## 1. 当前目标

本阶段只做一件事:

> 将 `gd_residual_v1` 中的 `grouped_chunk_torch_ref` 从逐 group / 逐 event 的 PyTorch 小循环, 改成按 event count 分桶的 batched PyTorch reference 实现。

本阶段不是正式训练阶段, 不是新数学阶段, 也不是 Triton/CUDA 阶段。

本阶段的目标是降低当前 `grouped_chunk_torch_ref` 的小算子和 autograd 开销, 为后续 official 4 epoch 降低成本。

---

## 2. 当前事实依据

请先阅读并核对以下文件。

Flash-VQG:

- `docs/20260425-flash-vqg-gated-delta-v1-math-plan-final.md`
- `docs/20260425-flash-vqg-gated-delta-v1-codex-blueprint.md`
- `src/flash_vqg/nn/fox/gd_residual.py`
- `src/flash_vqg/nn/attn_fox.py`
- `tests/test_fox_gd_residual_v1.py`
- `tests/test_attn_fox_compat.py`

zoology:

- `docs/20260427-gd-residual-v1-phase2-flat-gather-report.md`
- `docs/20260427-gd-residual-v1-eventpack-v1-report.md`
- `docs/20260428-gd-residual-v1-profile-tau-alignment-report.md`
- `docs/20260428-gd-residual-v1-current-t025-profile-gate-report.md`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/profile_gd_residual_v1.py`

当前已经完成的事实:

1. phase2 residual read 的 expanded gather 显存热点已通过 flat gather 解决。
2. event_pack v1 已完成, metadata/autograd 和 scalarization 开销已明显下降。
3. profile tau 口径已修正, 当前 profile 已严格记录 `vq_softmax_tau=0.25`。
4. strict tau=0.25 B64,T256,8 microbatches profile 已完成。
5. strict tau=0.25 B8,T128 profiler 已完成。
6. current 64x4 smoke 在本轮 strict tau=0.25 gate 中被跳过, 不能声明 current 64x4 smoke 已通过。
7. official 4 epoch 尚未启动, 当前没有正式质量结论。

当前 strict tau=0.25 B64,T256,8 microbatches profile 关键数值:

| 指标 | 当前值 |
|---|---:|
| `vq_softmax_tau` | `0.25` |
| `peak_reserved_GB` | `8.501953` |
| `peak_allocated_GB` | `6.749899` |
| `avg_microbatch_sec` | `80.005957` |
| `avg_forward_sec` | `22.903922` |
| `avg_backward_sec` | `57.025518` |
| losses finite | `true` |

当前 strict tau=0.25 B8,T128 profiler 关键热点:

| 项 | 当前值 |
|---|---:|
| `gd_residual/grouped_chunk` CUDA total | `2.783s` |
| `gd_residual/event_pack` CUDA total | `84.894ms` |
| `gd_residual/phase2_residual_read` CUDA total | `1.220ms` |
| `aten::copy_` | `51314` calls |
| `aten::item` | `11681` calls |
| `aten::_local_scalar_dense` | `11681` calls |
| `aten::outer` | `14336` calls |
| `aten::mv` | `14336` calls |
| `aten::matmul` | `8197` calls |
| `aten::addmv_` | `14336` calls |
| `aten::select_backward` | `31863` calls, `20.09 GB` CUDA memory usage |
| `torch::autograd::CopySlices` | `4360` calls, `24.25 GB` CUDA memory usage |
| `aten::gather_backward` | `2.00 MB`, `2` calls |
| `aten::index_select_backward` | `132.19 MB`, `5` calls |

解释:

- `phase2_residual_read` 已不是主瓶颈。
- `event_pack` 已不是主瓶颈。
- 当前主瓶颈是 `grouped_chunk_torch_ref` 内的逐 group / 逐 event 小算子和长 autograd graph。

---

## 3. 禁止事项

本阶段必须遵守以下硬约束。

禁止:

- 不要重跑 baseline。
- 不要启动 official full 4 epoch。
- 不要把 smoke/profile 当正式质量结论。
- 不要把 profile loss 下降当模型质量结论。
- 不要修改 official candidate 超参。
- 不要把 `rank=8` 或 `write_topk=2` 写成 official result。
- 不要重新设计 `gd_residual_v1` 数学。
- 不要写 Triton / CUDA / custom backward。
- 不要改 `legacy` / `clr_v1` / `clr_delta_v1` 语义。
- 不要继续优先优化 phase2 read 或 event_pack。
- 不要引入新的 public API, 除非测试或兼容层确有必要。

Official 配置必须保持:

- `fox_gd_residual_rank=16`
- `fox_gd_residual_write_topk=4`
- `fox_remote_read_topk=2`
- `num_codebook_vectors=256`
- `vq_weight_mode=dense_softmax`
- `vq_softmax_tau=0.25`
- `vq_score_mode=codebook_dot`
- `vq_update_mode=grad`
- `fox_gd_residual_builder=grouped_chunk_torch_ref`
- `fox_gd_residual_pack_mode=semivec_ref`

---

## 4. 当前实现的问题

当前 `grouped_chunk_torch_ref` 的核心逻辑是:

- 逐 group 循环。
- 每个 group 内逐 event 循环。
- 每个 event 执行 decay, matmul, outer, state update。
- 每个 group 最后执行 tail decay, 并写回 `out[g]`。

当前等价更新为:

\[
M \leftarrow \exp(\text{logabar}) M
\]

\[
\hat{u} = M d
\]

\[
M \leftarrow M + \zeta (u - \hat{u}) d^\top
\]

问题不是数学复杂, 而是实现形态低效:

- `cu_seqlens[g].item()` 造成 CPU scalarization。
- `D_pack[e]`, `U_pack[e]`, `zeta_pack[e]`, `logabar_pack[e]` 逐 event select 导致大量 `SelectBackward0`。
- `torch.outer` 和 `torch.matmul` 每个 event 各自发小算子。
- `out[g] = ...` 引入大量 `CopySlices`。
- autograd graph 以 event 为粒度增长。

---

## 5. 推荐优化方案

采用按 event count 分桶的 batched PyTorch recurrence。

不要直接做真正 WY / UT chunkwise delta solve。本阶段只改变执行形态, 不改变 recurrence 语义。

### 5.1 新增 loop oracle

将当前 `grouped_chunk_torch_ref` 的旧逻辑抽成内部函数:

`_grouped_chunk_torch_loop_oracle`

要求:

- 完整保留旧逻辑。
- 只作为 correctness oracle 和紧急回退路径。
- 不作为新的 public API。

### 5.2 新增 bucketed 实现

新增内部函数:

`_grouped_chunk_torch_bucketed_ref`

输入和输出与 `grouped_chunk_torch_ref` 保持一致。

总体逻辑:

1. 计算每个 group 的 event count。
2. 按 event count 分桶。
3. 同一桶内的 group event 数相同, 可构造 `[G_bucket, count]` 的 event index 矩阵。
4. 一次性 gather 出该 bucket 的 `D/U/zeta/logabar`。
5. 对该 bucket 只在 event step 上循环, 每步对全部 group batched 更新。
6. 最后乘 tail decay。
7. 用 bucket 级 `index_copy` 或等价方式写回输出。

伪代码:

    def _grouped_chunk_torch_bucketed_ref(...):
        if M_ent_pack.numel() == 0:
            return M_ent_pack.clone()
        if chunk_size <= 0:
            raise ValueError(...)

        counts = cu_seqlens[1:] - cu_seqlens[:-1]
        out = torch.empty_like(M_ent_pack)

        with torch.no_grad():
            unique_counts = torch.unique(counts)

        for c_tensor in unique_counts:
            c = int(c_tensor.item())
            with torch.no_grad():
                bucket_groups = (counts == c_tensor).nonzero(as_tuple=False).flatten()

            if c == 0:
                state = M_ent_pack.index_select(0, bucket_groups).float()
                state = alpha_tail_pack.index_select(0, bucket_groups).float()[:, None, None] * state
                out.index_copy_(0, bucket_groups, state.to(M_ent_pack.dtype))
                continue

            with torch.no_grad():
                starts = cu_seqlens.index_select(0, bucket_groups)
                offsets = torch.arange(c, device=cu_seqlens.device, dtype=torch.long)
                event_idx = starts[:, None] + offsets[None, :]
                flat_event_idx = event_idx.reshape(-1)

            D = D_pack.index_select(0, flat_event_idx).view(bucket_groups.numel(), c, -1).float()
            U = U_pack.index_select(0, flat_event_idx).view(bucket_groups.numel(), c, -1).float()
            zeta = zeta_pack.index_select(0, flat_event_idx).view(bucket_groups.numel(), c).float()
            logabar = logabar_pack.index_select(0, flat_event_idx).view(bucket_groups.numel(), c).float()
            state = M_ent_pack.index_select(0, bucket_groups).float()

            for j in range(c):
                alpha = torch.exp(logabar[:, j])
                state = alpha[:, None, None] * state
                d = D[:, j]
                u = U[:, j]
                z = zeta[:, j]
                pred = torch.bmm(state, d.unsqueeze(-1)).squeeze(-1)
                update = z[:, None, None] * (u - pred).unsqueeze(-1) * d.unsqueeze(1)
                state = state + update

            tail = alpha_tail_pack.index_select(0, bucket_groups).float()
            state = tail[:, None, None] * state
            out.index_copy_(0, bucket_groups, state.to(M_ent_pack.dtype))

        return out

注意:

- `with torch.no_grad()` 只包 metadata, 例如 `bucket_groups`, `starts`, `offsets`, `event_idx`。
- 不要 detach `M_ent_pack`, `D_pack`, `U_pack`, `zeta_pack`, `logabar_pack`, `alpha_tail_pack`。
- `D/U/zeta/logabar/state/tail` 这类数值路径必须保留梯度。
- 内部计算继续用 float32, 输出 dtype 对齐 `M_ent_pack.dtype`。

### 5.3 更新 grouped_chunk_torch_ref

`grouped_chunk_torch_ref` 应调用新的 bucketed 实现。

建议结构:

    def grouped_chunk_torch_ref(...):
        return _grouped_chunk_torch_bucketed_ref(...)

旧 loop oracle 不应默认使用, 只用于测试和回退。

如需临时 debug 开关, 只能使用内部环境变量, 不能改变 official builder 名称。

---

## 6. 正确性测试要求

修改或新增 `tests/test_fox_gd_residual_v1.py`。

建议新增测试函数:

1. `test_gd_grouped_chunk_bucketed_matches_loop_oracle_forward`
2. `test_gd_grouped_chunk_bucketed_matches_loop_oracle_backward`
3. `test_gd_grouped_chunk_bucketed_handles_empty_pack`
4. `test_gd_grouped_chunk_bucketed_chunk_size_semantics`
5. 继续保留现有 `test_gd_grouped_chunk_matches_token_step`

### 6.1 forward 对齐

构造一个手工 event pack, 至少覆盖:

- 多个 group。
- event count 不均匀, 例如 `[1, 2, 4, 3]`。
- event 顺序按 group 内时间排列。
- 非平凡 `logabar_pack`。
- 非平凡 `alpha_tail_pack`。
- `chunk_size=1`, `chunk_size=2`, `chunk_size=64`。

比较:

- `_grouped_chunk_torch_bucketed_ref(...)`
- `_grouped_chunk_torch_loop_oracle(...)`

使用 `torch.testing.assert_close`, 建议 `atol=1e-6, rtol=1e-6` 或按 dtype 适当放宽。

### 6.2 backward 对齐

对以下输入分别设置 `requires_grad=True`:

- `M_ent_pack`
- `D_pack`
- `U_pack`
- `zeta_pack`
- `logabar_pack`
- `alpha_tail_pack`

分别跑 loop oracle 和 bucketed:

- `loss = out.float().square().sum()`
- `loss.backward()`

比较每个输入的 grad:

- finite。
- non-None。
- 与 oracle close。

如果某个 grad 理论上可能为 0, 可以只检查 finite + close, 不强制 non-zero。

### 6.3 空 pack

验证空 event pack 行为不变:

- 输入 `M_ent_pack` shape 为 `[0, d_v, r]`。
- 输出应为 clone, shape 和 dtype 不变。
- 不报错。

### 6.4 集成对齐

现有 `test_gd_grouped_chunk_matches_token_step` 必须继续通过。

这条测试是高层语义 gate, 用于确认:

- token_step reference 和 grouped_chunk reference 仍等价。
- block-entry frozen baseline 语义未被破坏。
- event_pack 与 grouped_chunk 的接口未被破坏。

---

## 7. 性能实验 gate

所有 profile 必须保持 strict `tau=0.25` 和 official 配置。

### 7.1 Flash-VQG correctness gate

在 Flash-VQG 仓库运行:

    cd /home/lyj/mnt/project/Flash-VQG
    pytest tests/test_fox_gd_residual_v1.py -q
    pytest tests/test_attn_fox_compat.py -q

如果测试失败, 停止, 不跑 profile。

### 7.2 B8,T128 profiler gate

在 zoology 仓库运行:

    cd /home/lyj/mnt/project/zoology
    FOX_REMOTE_READ_TOPK=2 \
    FOX_GD_RESIDUAL_RANK=16 \
    FOX_GD_RESIDUAL_WRITE_TOPK=4 \
    NUM_CODEBOOK_VECTORS=256 \
    VQ_WEIGHT_MODE=dense_softmax \
    VQ_SOFTMAX_TAU=0.25 \
    TRAIN_BATCH_SIZE=8 \
    PROFILE_SEQ_LEN=128 \
    PROFILE_MICROBATCHES=1 \
    PROFILE_ENABLE_TORCH_PROFILER=1 \
    PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
    PROFILE_OUTPUT_DIR=tmp/20260514-gd-groupedchunk-bucketed-t025-prof-b8-t128 \
    bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh

记录并对比:

| 指标 | 当前 baseline | 目标 |
|---|---:|---:|
| `gd_residual/grouped_chunk` CUDA total | `2.783s` | 明显下降, 理想降幅 >= 20% |
| `aten::copy_` calls | `51314` | 明显下降 |
| `aten::item` calls | `11681` | 明显下降 |
| `aten::_local_scalar_dense` calls | `11681` | 明显下降 |
| `aten::outer` calls | `14336` | 明显下降 |
| `aten::mv` calls | `14336` | 明显下降 |
| `aten::addmv_` calls | `14336` | 明显下降 |
| `aten::select_backward` calls | `31863` | 明显下降 |
| `CopySlices` calls | `4360` | 明显下降 |
| `gather_backward` memory | `2.00 MB` | 不回归到大显存 |
| `index_select_backward` memory | `132.19 MB` | 不明显膨胀 |

硬性通过条件:

- `summary["profile"]["vq_softmax_tau"] == 0.25`
- loss finite。
- 不 OOM。
- `phase2_residual_read` 不回归。
- `event_pack` 不回归为主瓶颈。

### 7.3 B64,T256,8 microbatches profile gate

在 zoology 仓库运行:

    cd /home/lyj/mnt/project/zoology
    FOX_REMOTE_READ_TOPK=2 \
    FOX_GD_RESIDUAL_RANK=16 \
    FOX_GD_RESIDUAL_WRITE_TOPK=4 \
    NUM_CODEBOOK_VECTORS=256 \
    VQ_WEIGHT_MODE=dense_softmax \
    VQ_SOFTMAX_TAU=0.25 \
    TRAIN_BATCH_SIZE=64 \
    PROFILE_SEQ_LEN=256 \
    PROFILE_MICROBATCHES=8 \
    PROFILE_ENABLE_TORCH_PROFILER=0 \
    PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
    PROFILE_OUTPUT_DIR=tmp/20260514-gd-groupedchunk-bucketed-t025-b64-t256-mb8 \
    bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh

记录并对比:

| 指标 | 当前 baseline | 目标 |
|---|---:|---:|
| `avg_microbatch_sec` | `80.005957s` | 明显下降, 理想低于 `70s` |
| `avg_backward_sec` | `57.025518s` | 明显下降, 理想低于 `50s` |
| `avg_forward_sec` | `22.903922s` | 不明显升高 |
| `peak_reserved_GB` | `8.501953` | 不明显高于 `8.5`, 硬上限建议 `9.0` |
| `peak_allocated_GB` | `6.749899` | 不明显升高 |
| losses finite | `true` | 必须保持 |

说明:

- 如果速度下降明显但没达到理想阈值, 仍应报告为 partial success。
- 如果显存明显升高或产生 NaN/Inf, 应视为失败或需要回退。

### 7.4 64x4 smoke

本阶段可选。

如果 B8/B64 profile 明显改善且时间允许, 再跑 current 64x4 smoke 回归。

如果没有跑, 报告中必须明确写:

- current 64x4 smoke 未执行。
- 不能声明 current 64x4 smoke 已通过。

---

## 8. 产物要求

### 8.1 Flash-VQG 代码产物

预期修改:

- `src/flash_vqg/nn/fox/gd_residual.py`
- `tests/test_fox_gd_residual_v1.py`

尽量不要修改其他文件。

### 8.2 zoology 实验产物

建议生成报告:

- `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-report.md`

建议提交小体积 artifact:

- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/b64-t256-mb8-summary.json`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-summary.json`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-profiler_cpu_time_total.txt`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-profiler_cuda_time_total.txt`
- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/prof-b8-t128-profiler_cuda_memory_usage.txt`

不要提交:

- profiler trace 大文件。
- `tmp/` 全目录。
- SwanLab 本地日志。
- checkpoint。
- generated launch config。
- `__pycache__`。

---

## 9. 报告必须包含的内容

报告至少包括:

1. 仓库 branch / commit / git status。
2. 修改文件列表。
3. old loop oracle 与 bucketed 实现的设计说明。
4. 正确性测试命令和结果。
5. B8,T128 profiler 对比表。
6. B64,T256,8 microbatches profile 对比表。
7. 是否执行 64x4 smoke, 若未执行必须明确说明。
8. 是否建议进入下一阶段。
9. 明确声明:
   - 没有重跑 baseline。
   - 没有启动 official full 4 epoch。
   - 没有修改 official 超参。
   - 没有重新设计 `gd_residual_v1` 数学。
   - 没有写 Triton/CUDA/custom backward。
   - 没有改 legacy / `clr_v1` / `clr_delta_v1`。
   - smoke/profile 不是正式训练质量结论。

---

## 10. 成功标准

本阶段成功分为 hard success 和 soft success。

### 10.1 hard success

必须全部满足:

- 所有 correctness tests 通过。
- old loop oracle 和 bucketed forward 对齐。
- old loop oracle 和 bucketed backward gradients 对齐。
- token_step reference 与 grouped_chunk reference 仍对齐。
- strict `tau=0.25` profile summary 正确记录 `vq_softmax_tau=0.25`。
- B64/T256 profile loss finite。
- 不 OOM。
- peak reserved 不明显高于当前 `8.5 GiB`, 硬上限建议 `9.0 GiB`。

### 10.2 soft success

满足 hard success 后, 至少满足一项:

- B8 profiler 中 `gd_residual/grouped_chunk` CUDA total 明显下降。
- B64 profile 中 `avg_microbatch_sec` 明显低于当前 `80s`。
- B64 profile 中 `avg_backward_sec` 明显低于当前 `57s`。
- `copy_`, `item`, `outer`, `mv`, `addmv_`, `select_backward`, `CopySlices` calls/time 明显下降。

### 10.3 失败和回退标准

以下情况应回退或暂停:

- 任何 correctness test 失败。
- forward close 通过但 backward grad 不对齐。
- B64 profile 出现 NaN/Inf。
- B64 profile OOM。
- peak reserved 明显超过 `9.0 GiB`。
- `avg_microbatch_sec` 没降反升且 profiler 显示 grouped_chunk 没改善。
- phase2 read 或 event_pack 回归为主瓶颈。

---

## 11. 给 Codex 的执行顺序

请按下面顺序执行, 不要跳步。

1. 检查两个仓库的 branch / commit / status。
2. 阅读本文件和相关报告。
3. 在 Flash-VQG 中阅读 `gd_residual.py` 和 `test_fox_gd_residual_v1.py`。
4. 抽出 `_grouped_chunk_torch_loop_oracle`。
5. 实现 `_grouped_chunk_torch_bucketed_ref`。
6. 让 `grouped_chunk_torch_ref` 调用 bucketed 实现。
7. 增加 forward/backward oracle 对齐测试。
8. 跑 Flash-VQG correctness tests。
9. 如果 tests 通过, 在 zoology 跑 B8 profiler gate。
10. 如果 B8 gate 无 OOM/NaN, 跑 B64 profile gate。
11. 整理报告和小体积 artifacts。
12. 不要启动 official full 4 epoch。
13. 不要重跑 baseline。

---

## 12. 一句话总结

本阶段要把 `grouped_chunk_torch_ref` 从“每个 event 一个小算子”的 reference loop, 改成“按 event count 分桶的一批 group batched 更新”的 PyTorch reference 实现。正确性靠旧 loop oracle 和 token-step oracle 双重保证, 性能靠 strict `tau=0.25` B8 profiler 和 B64 profile 做 gate。
