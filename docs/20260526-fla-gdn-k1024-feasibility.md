# FLA 官方 GDN K=1024 可行性分析, 2026-05-26

## 结论

官方 `fla-org/flash-linear-attention` 最新 `main` 仍不能直接用 `chunk_gated_delta_rule` 训练 per-head `K=1024` 的 GDN。

原因不是 zoology 包装层, 而是官方 FLA 的 GDN chunk state-update kernel 本身有硬限制: `K <= 256`。因此我们当前想跑的 `gdnxk-h2-ek16-ev1`, per-head `K=1024,V=64`, 在官方高效训练路径下仍不可执行。

但并不是完全没有机会。较可行的路线是 fork FLA, 新增一个 K-blocked 的 GDN state-update kernel, 用官方 `naive_recurrent_gated_delta_rule` 和 `fused_recurrent_gated_delta_rule` 做 correctness oracle。这个是 kernel 工程任务, 不是改一两个 Python 参数能解决。

## 官方仓库状态

已下载到:

`/home/lyj/mnt/project/flash-linear-attention`

版本:

| 字段 | 值 |
|---|---|
| remote | `https://github.com/fla-org/flash-linear-attention.git` |
| branch | `main` |
| commit | `19b5a3f4` |
| commit 时间 | `2026-05-25 14:23:04 +0800` |
| commit 标题 | `[GDN] Support use_beta_sigmoid_in_kernel & allow_neg_eigval (#919)` |

## 限制位置

官方源码:

`/home/lyj/mnt/project/flash-linear-attention/fla/ops/common/chunk_delta_h.py`

关键断言:

- line 625: `assert K <= 256, "current kernel does not support head dimension larger than 256."`
- line 678: `assert K <= 256, "current kernel does not support head dimension being larger than 256."`

直接用官方源码 probe 的结果:

| K | V | 结果 |
|---:|---:|---|
| 256 | 256 | precheck 通过 |
| 512 | 128 | `AssertionError: current kernel does not support head dimension larger than 256.` |
| 1024 | 64 | `AssertionError: current kernel does not support head dimension larger than 256.` |

## 为什么不能只删断言

`chunk_gated_delta_rule_fwd_h` 的 Triton kernel 叫 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`。它把 K 维写死成最多 4 个 64-dim block:

- `b_h1`: K `[0,64)`
- `b_h2`: K `[64,128)`
- `b_h3`: K `[128,192)`
- `b_h4`: K `[192,256)`

forward 和 backward 都是这种手工展开结构。`K=1024` 需要 16 个 64-dim block。如果粗暴复制到 `b_h16`, 单个 Triton program 要同时持有更多 `K x V` state fragment。以 `K=1024,V=64` 为例, active state 是 `65536` float elements per head, 远超当前 kernel 的寄存器/shared-memory 设计边界, 在 RTX 2080 Ti/sm75 上尤其不现实。

所以正确方向不是删断言, 而是把 state update kernel 改成 K-blocked 多 program 设计。

## 哪些路径已经支持大 K

官方代码里有些子路径对 K 是 tile 循环, 理论上不构成主要限制:

- `fla/ops/gated_delta_rule/chunk_fwd.py`: intra-chunk KKT/solve 使用 `for i_k in range(tl.cdiv(K, BK))`。
- `fla/ops/common/chunk_o.py`: `chunk_fwd_o` 使用 K tile 循环计算 output。

真正卡住的是:

- `chunk_gated_delta_rule_fwd_h`
- `chunk_gated_delta_rule_bwd_dhu`
- CP pre-process 相关的 `chunk_gated_delta_rule_fwd_h_pre_process` 和 `chunk_gated_delta_rule_bwd_dhu_pre_process`

这些路径负责跨 chunk 的 recurrent state update。

## 替代路径验证

### fused_recurrent

官方 `fused_recurrent_gated_delta_rule` 可以 forward `K=1024,V=64` 的极小输入:

| shape | 结果 |
|---|---|
| `B=1,T=4,H=2,HV=2,K=1024,V=64` | forward 通过, 约 `1.383s` |

但是 backward 明确未实现:

`NotImplementedError: Backward pass is not implemented yet ...`

因此它可以作为 inference/probe 或 correctness 参考, 不能直接用于正式 MQAR 训练。

### naive recurrent

官方 `naive_recurrent_gated_delta_rule` 可以在极小输入上跑 `K=1024,V=64` 的 backward:

| shape | 结果 |
|---|---|
| `B=1,T=4,H=2,K=1024,V=64` | forward/backward 通过, 约 `0.131s` |

但这是 PyTorch reference loop, 序列维逐 token 执行, 只适合作为 correctness oracle 或极小规模单元测试, 不适合作为正式 MQAR 训练 kernel。

## 可行方案

### 方案 A: 短期受限实验, 只跑 K<=256

继续只跑 `ek4-ev4`, 即 per-head `K=256,V=256`。

优点:

- 不需要 fork FLA。
- 可以回答“从 GDN 原始 K=64 增到 K=256 是否有帮助”。

缺点:

- 不能对齐 Flash 的抽象 `K=1024,V=64`。
- 不能回答原始问题的核心 endpoint。

这个方案适合作为临时 probe, 不适合作为最终公平对照。

### 方案 B: fork FLA, 实现 K-blocked GDN state-update kernel

这是最接近目标的方案。

核心任务:

1. 在 FLA fork 中新增 `chunk_gated_delta_rule_fwd_h_kblocked` 和 `chunk_gated_delta_rule_bwd_dhu_kblocked`。
2. 把 state tensor `h` 按 K block 分片, 例如 `BK=64` 或 `BK=128`, grid 维度增加 `i_k`。
3. 对每个 K block 独立计算 `w_block @ h_block`, 再把不同 K block 的贡献 reduce 成完整 `v_new = u - sum_k(w_k @ h_k)`。
4. 更新 `h_k += k_k^T @ v_new`。
5. backward 同步实现同样的 K-block reduce 和梯度传播。
6. 用 `naive_recurrent_gated_delta_rule` 在小 shape 上验证 forward/backward correctness。
7. 用 `fused_recurrent_gated_delta_rule` 验证 forward 数值一致性。
8. 最后接回 zoology 的 `GatedDeltaNetExpandedK`。

主要难点:

- `v_new` 依赖所有 K block 的 `w_k @ h_k` 求和, 所以 K-blocked 并不是 embarrassingly parallel, 需要额外中间 buffer 或两阶段 kernel。
- backward 也需要对应的跨 K-block reduce。
- 2080 Ti/sm75 上 fp32 训练会更慢, 需要先做 microbench, 不应直接上 MQAR 正式训练。

推荐实现结构:

1. `fwd_h_partial`: grid `(i_k, i_v, batch_head)`, 输出 `v_decay_partial[NK, B, T, HV, V_block]` 或 chunk 内 partial。
2. `reduce_v_decay`: 沿 `NK` reduce, 得到 `v_new`。
3. `fwd_h_update`: 用 `v_new` 更新每个 K block 的 state, 写出分块 `h`。
4. backward 按同样思路拆成 partial, reduce, update-grad 三段。

这会增加 HBM 读写, 速度可能显著慢于当前 K<=256 kernel, 但至少工程上比“单 program 持有 K=1024 全 state”更现实。

### 方案 C: 只做 naive 极小规模 proof-of-concept

把 zoology 的 `GatedDeltaNetExpandedK` 加一个 debug-only mode, 在很小 `T` 和很小 batch 下调用官方 naive recurrent 实现。

优点:

- 最快验证模型包装和梯度链路。
- 可作为未来 kernel 的 correctness oracle。

缺点:

- 无法用于正式 MQAR 训练。
- 不能产生可比 experimental result。

## 推荐下一步

如果目标仍是严格对齐 Flash 的抽象 `K=1024,V=64`, 我建议 fork 官方 FLA, 做方案 B。第一轮不要直接改完整训练, 而是立一个 kernel research goal:

1. fork `/home/lyj/mnt/project/flash-linear-attention`。
2. 新建分支 `codex/gdn-k1024-kblocked`.
3. 先实现 forward-only K-blocked `chunk_gated_delta_rule_fwd_h`。
4. 在 `B=1,T=64,H=2,HV=2,K=512,V=128` 和 `K=1024,V=64` 上对齐 `naive_recurrent_gated_delta_rule`。
5. forward 对齐后再做 backward。
6. backward 对齐后再回到 zoology 跑 `GatedDeltaNetExpandedK` smoke。

如果近期主要目标是实验推进而不是 kernel 开发, 则先跑 `ek4-ev4` 受限 probe, 并在报告中明确“该实验只覆盖 K<=256, 不能对齐 Flash K=1024 endpoint”。

## Artifact

结构化记录:

`docs/artifacts/gdn-expanded-k/fla-k1024-feasibility-20260526.json`
