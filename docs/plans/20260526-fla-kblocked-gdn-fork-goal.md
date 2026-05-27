# FLA K-blocked GDN fork 独立目标

updated: 2026-05-26
status: proposed independent goal

## 背景

`docs/plans/20260526-gdn-flash-fairness-experiment-plan.md` 的 Phase 6 结论是 `go_as_separate_goal`. 当前 kernel-compatible GDN 已完成 Phase 0-5, 但没有消除公平性质疑:

- Banked-K GDN 是 K-sharded approximation, 不是 true single continuous per-head `K=1024,V=64`.
- Banked-K seed123/126 上限高, 但多 seed 不稳定.
- 同等 active state capacity 的 Flash `cb64-r16` 在 longer-MQAR OOD 上明显强于当前 GDN 对照.
- FLA chunk state-update kernel 当前限制 per-head `K<=256`, 阻止 true `K=1024,V=64` GDN 正式训练.

因此, 如果论文需要最干净的 endpoint, 应把 FLA fork 作为独立 kernel research goal 推进.

## 建议 /goal 目标

```text
目标:
在 /home/lyj/mnt/project 下基于 fla-org/flash-linear-attention 的目标 commit 建立独立 fork/worktree, 为 Gated Delta Rule chunk training 新增 K-blocked state-update 路径, 使 chunk_gated_delta_rule 训练路径支持 per-head K=512,V=128 和 K=1024,V=64, 同时保持 K<=256 原路径行为不变。该目标只解决 true expanded-K GDN 训练所需的 FLA kernel 限制, 不改变 GDN 数学公式, 不把 naive recurrent 作为正式训练路径。

执行约束:
1. 先确认目标 FLA commit, 记录 commit hash, 分支名, GPU, CUDA/Torch/Triton 版本.
2. 不直接删除 K<=256 assert 后宣称完成; 必须实现实际 K-blocked state-update 支持.
3. 旧 K<=256 路径必须保留并默认继续走旧 kernel.
4. 新路径只在 K>256 且目标 shape 命中时分发, 第一版可以限制到 K=512,V=128 和 K=1024,V=64.
5. 使用 naive_recurrent_gated_delta_rule 作为 correctness oracle; fused_recurrent forward 可作为补充 forward oracle, 但不能替代 backward 训练验证.
6. 所有 correctness, smoke, performance 结果必须写入 fork 内 artifact/report, 包括 started_at_utc, ended_at_utc, wall_clock_sec, gpu, gpu_name, status.
7. 通过 FLA smoke 后, 再回 /home/lyj/mnt/project/zoology 的 GatedDeltaNetExpandedK 跑 ek8-ev2 和 ek16-ev1 forward/backward smoke. 不提前启动正式 MQAR 训练.

验收标准:
1. Forward correctness:
   - B=1,T=4,H=2,HV=2,K=512,V=128.
   - B=1,T=4,H=2,HV=2,K=1024,V=64.
   - B=2,T=65,H=2,HV=2,K=512,V=128.
   - B=2,T=65,H=2,HV=2,K=1024,V=64.
   - initial_state=None/provided, output_final_state=False/True.
   - fp32 max_abs_err <= 5e-4, max_rel_err <= 1e-3; fp16/bf16 如测试, max_abs_err <= 2e-2, max_rel_err <= 2e-2.
2. Backward correctness:
   - 对 dq, dk, dv, dg, dbeta, dinitial_state 与 naive oracle 对齐.
   - fp32 grad max_abs_err <= 1e-3, max_rel_err <= 1e-2; fp16/bf16 如测试, max_abs_err <= 5e-2, max_rel_err <= 5e-2.
3. 旧路径不回归:
   - K=64,V=64; K=128,V=128; K=256,V=256 仍走旧 kernel.
   - correctness 与原路径一致, 性能无明显回退.
4. Zoology 集成 smoke:
   - ek8-ev2: num_heads=2, head_k_dim=512, head_v_dim=128, batch=1, seq_len=8, d_model=128, forward/backward pass.
   - ek16-ev1: num_heads=2, head_k_dim=1024, head_v_dim=64, batch=1, seq_len=8, d_model=128, forward/backward pass.
   - 稍大 smoke: batch=2 或 4, seq_len=128 或 256, no OOM, loss.backward 正常, optimizer.step 正常.
5. 性能底线:
   - RTX 2080 Ti / sm75 不 OOM.
   - K=1024,V=64 batch=1,seq=8 smoke 可在可接受时间内完成.
   - batch=4,seq=256 forward/backward 可完成.
   - 输出 compile time, step wall-clock, peak memory.

停止条件:
1. 以上 forward/backward correctness, 旧路径不回归, zoology smoke, 性能底线均完成并落盘.
2. 如果 kernel 实现不可行或性能不可接受, 写明具体失败点, shape, error, wall-clock, peak memory, 并给出 no-go 结论.
3. 不把该 fork 的实验结果混入当前 GDN/Flash fairness 训练 ledger; 只有 ek8/ek16 正式 MQAR 训练真正启动并跑满后, 才按 zoology MQAR official ledger 规则记录.
```
