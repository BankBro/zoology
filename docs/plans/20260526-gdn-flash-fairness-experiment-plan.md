# GDN 与 Flash-VQG 公平对照实验方案

updated: 2026-05-26
status: active plan
branch: `flash-vqg`

## 1. 核心判断

当前不要先 fork FLA, 也不要先做 VQ-routed GDN. 最稳路线是:

```text
先固定 GDN expanded-K blocked 结论,
再跑不改 FLA kernel 的 GDN 强公平对照,
同时补 Flash-VQG 主线稳定性,
根据结果决定是否实现 Banked-K GDN,
最后才考虑 FLA fork 做 true K=1024,V=64.
```

这个方案的核心目标是回答:

```text
在现有 FLA kernel 能支持的范围内把 GDN 做强后,
Flash-VQG 是否仍然有稳定优势?
```

如果答案是肯定的, 再讨论 Banked-K GDN 或 FLA fork 才有意义.

## 2. 全局执行约束

所有 Phase 都遵守以下约束:

```text
交互和报告使用中文, 专业词汇和代码标识符可用英文.
正式 MQAR 训练默认 seed=123, data_seed=123 作为第一轮方向判断.
train_batch_size=64.
gradient_accumulation_steps=4.
effective_train_batch_size=256.
RTX 2080 Ti / sm75 上默认 float32 训练口径.
GDN official 可比实验显式设置 GDN_KERNEL_DTYPE=float32.
完整跑到预期 final checkpoint 后才写 canonical ledger.
smoke/debug/失败/中断/未跑满预期 epoch 的实验只写 status/report/artifact, 不写 official ledger.
不同 dtype policy 的结果不能混入同一 official 直接质量对比.
```

所有完整正式 MQAR 训练和正式 longer-MQAR eval 必须记录时间字段:

```text
started_at_utc
ended_at_utc
wall_clock_sec
gpu
gpu_name
status
```

`/goal` 执行时必须额外遵守:

```text
每完成一个 Phase 后, 重新阅读本文档, 检查下一步是否仍符合方案.
每发生一次上下文压缩后, 恢复执行的第一步必须重新阅读本文档.
如果发现实际结果和方案门控冲突, 先更新 status/report, 再决定是否进入下一阶段.
不得跳过 Phase 0 到 Phase 6 的顺序和门控.
```

## 3. 推荐产物位置

方案文档:

```text
docs/plans/20260526-gdn-flash-fairness-experiment-plan.md
```

已有相关报告:

```text
docs/20260526-gdn-expanded-k-longer-mqar-report.md
docs/20260526-fla-gdn-k1024-feasibility.md
docs/20260526-longer-mqar-official-core-report.md
```

已有或建议使用的 artifact 目录:

```text
docs/artifacts/gdn-expanded-k/
docs/artifacts/longer-mqar/
docs/artifacts/20260526-gdn-flash-fairness/
```

如果需要新增 summary 表, 优先放到:

```text
docs/artifacts/20260526-gdn-flash-fairness/
```

完整正式 GDN hparam/baseline 结果如符合既有 canonical ledger 口径, 再追加到:

```text
docs/artifacts/gdn/gdn-hparam-effect-summary.csv
```

但 blocked, smoke, debug, failed, interrupted run 不写入 canonical ledger.

## 4. Phase 0: 固定 blocked 结论

这一阶段不训练新模型, 只整理已有结论和 artifact.

需要确认并写清:

```text
ek4-ev4:
  H=2,K=256,V=256.
  active_state_capacity=131072.
  forward/backward 通过.

ek8-ev2:
  H=2,K=512,V=128.
  active_state_capacity=131072.
  FLA chunk kernel 因 K>256 断言失败.

ek16-ev1:
  H=2,K=1024,V=64.
  active_state_capacity=131072.
  因同一 K>256 限制不启动正式训练.
```

需要输出一张 capacity/accounting 表, 至少包含:

```text
run_id
model_family
num_heads
per-head K
per-head V
active_state_capacity
trainable_params
kernel path
kernel dtype
GPU
是否可训练
失败原因
smoke started_at_utc
smoke ended_at_utc
smoke wall_clock_sec
```

Phase 0 完成标准:

```text
blocked 结论已写入 report 或 summary.
ek8/ek16 不可训练原因明确归因到 FLA chunk state-update kernel K<=256 限制.
没有把 ek4-ev4 误写成 true K=1024,V=64 对照.
```

## 5. Phase 1: GDN kernel-compatible fairness probe

目标:

```text
在不改 FLA kernel 的情况下, 把 GDN baseline 做强,
判断 GDN 在 K<=256 边界内是否仍明显落后 Flash-VQG.
```

优先跑 3 个配置, 都先 smoke, 通过后正式 4 epoch:

```text
1. ek4-ev4
   H=2,K=256,V=256
   active_state_capacity=131072
   作用: 当前 expanded-K 唯一能直接运行的配置.

2. mh-h4-k256-v128
   H=4,K=256,V=128
   active_state_capacity=131072
   作用: 同容量下增加 head/address partition.

3. mh-h8-k256-v64
   H=8,K=256,V=64
   active_state_capacity=131072
   作用: 更接近多个 address bank 的 GDN.
```

正式运行约束:

```text
seed=123.
data_seed=123.
train_batch_size=64.
gradient_accumulation_steps=4.
effective_train_batch_size=256.
GDN_KERNEL_DTYPE=float32.
configured_max_epochs=4.
不使用 early stop 作为主结果.
```

Phase 1 完成标准:

```text
三个配置都有 smoke 结论.
能跑的配置完成正式 4 epoch.
完整跑满的正式实验写入对应 canonical ledger.
失败/OOM/NaN/中断配置写入 status/report, 不写 official ledger.
每个正式实验记录 wall-clock, GPU, dtype, peak memory, batch/GA 配置.
```

进入 Phase 3 的初步判断:

```text
如果 mh-h4 或 mh-h8 有明显提升,
或者没有明显否定 address-capacity 路线,
且训练成本可接受,
则 Banked-K GDN 值得进入实现评估.
```

## 6. Phase 2: Flash-VQG 主线稳定性

目标:

```text
补 Flash-VQG 关键 seed,
确认 mid-capacity 和 high-capacity anchor 是否跨 seed 稳定.
```

优先补:

```text
cb256-r8-s126.
cb256-r16-s126.
```

资源允许后再补:

```text
cb256-r6-s125.
cb256-r10-s125.
```

重点指标:

```text
overall accuracy.
1024x256 hard accuracy.
rank neighborhood trend.
cross-seed stability.
wall-clock.
peak memory.
```

Phase 2 完成标准:

```text
cb256-r8-s126 和 cb256-r16-s126 有完整结果或明确失败原因.
能判断 Flash 主结果候选是 best mid-capacity, best high-capacity, 还是二者都保留.
没有把 Phase 2 扩展成无边界调参.
```

## 7. Phase 3: Banked-K GDN 决策与实现

Phase 3 不是默认立即执行, 必须根据 Phase 1 结果触发.

触发条件:

```text
如果 mh-h4 或 mh-h8 有明显提升, 或者 GDN probe 没有明显否定 address-capacity 路线, 且训练成本可接受, 则实现 Banked-K GDN.
如果 mh-h8 完全崩掉, OOM, NaN/Inf, 或明显无收益, 则 Banked-K 延后.
```

推荐第一版:

```text
class: GatedDeltaNetBankedK.
projection: shared-V.
logical_heads=2.
banks_per_head=4.
bank_k_dim=256.
bank_v_dim=64.
merge=dense softmax bank gate.
active_state_capacity=2*4*256*64=131072.
```

语义边界:

```text
这是 kernel-compatible K-sharded approximation.
不是 true single continuous K=1024 head.
每个 FLA kernel 调用仍然满足 K=256,V=64.
论文或报告中不能把它写成等价于 true K=1024,V=64.
```

Phase 3 完成标准:

```text
若不实现 Banked-K, 必须写清 no-go 原因.
若实现 Banked-K, 必须新增最小模块和配置, 不改变原 GatedDeltaNet 默认语义.
Banked-K 先通过 forward/backward smoke.
smoke 通过后跑 seed=123 正式 4 epoch.
只有单 seed 明显有希望时才进入 Phase 4 补多 seed.
```

## 8. Phase 4: 只对胜出的点补多 seed

不要所有配置都补多 seed.

只考虑这些对象:

```text
best Flash mid-capacity.
best Flash high-capacity.
best kernel-compatible GDN.
Banked-K GDN, 仅当 seed=123 明显有希望.
Full Attention baseline, 仅当已有结果不完整或不可比.
```

主表至少记录:

```text
mean ± std.
overall accuracy.
1024x256 hard accuracy.
active_state_capacity.
trainable_params.
train_batch_size.
eval_batch_size.
gradient_accumulation_steps.
effective_train_batch_size.
dtype policy.
GPU.
GPU name.
started_at_utc.
ended_at_utc.
wall_clock_sec.
peak memory.
```

Phase 4 完成标准:

```text
胜出点有 mean ± std.
Flash 和 GDN 的 official direct comparison 口径一致.
不同 dtype 或不完整 run 已明确排除在 official direct comparison 外.
```

## 9. Phase 5: OOD / 泛化任务

这一阶段放在 fairness 主结果稳定之后, 不提前做.

最小顺序:

```text
1. longer-MQAR OOD.
2. passkey 或 needle retrieval.
3. 小规模 long-context language modeling slice.
```

定位:

```text
OOD 不用于继续调参.
OOD 用于验证 MQAR 主线结论是否能迁移到更真实 retrieval 场景.
```

Phase 5 完成标准:

```text
至少完成 longer-MQAR OOD.
如果 passkey/needle 或 language modeling slice 未做, 需要写明原因.
OOD 结果和主表结果分开记录, 不混入 MQAR official training ledger.
```

## 10. Phase 6: FLA fork 门控

FLA fork 是最后选项, 不和主实验混在一起.

只有满足下面任一条件才启动:

```text
Flash 仍明显赢过所有 kernel-compatible GDN.
Banked-K 结果不足以消除公平性质疑.
论文主张必须包含 true K=1024,V=64 GDN baseline.
```

如果启动, 作为单独 kernel research goal:

```text
基于 fla-org/flash-linear-attention commit 19b5a3f4 或当时确认的目标 commit.
新增 Gated Delta Rule chunk training 的 K-blocked state-update 路径.
支持 per-head K=512,V=128 和 K=1024,V=64.
保持 K<=256 原路径行为不变.
用 naive_recurrent_gated_delta_rule 做 correctness oracle.
不把 naive recurrent 作为正式训练路径.
```

Phase 6 完成标准:

```text
给出 go/no-go 结论.
如果 go, 新开独立方案文档或独立 /goal.
如果 no-go, 写清为什么当前主实验已经足够或 fork 成本暂时不值得.
```

## 11. 当前立即执行顺序

```text
1. 整理 gdn-expanded-k blocked 表和报告.
2. 跑 ek4-ev4 seed123 正式 4 epoch.
3. 跑 mh-h4-k256-v128 smoke, 通过后跑 seed123 正式 4 epoch.
4. 跑 mh-h8-k256-v64 smoke, 通过后跑 seed123 正式 4 epoch.
5. 补 Flash cb256-r8-s126 和 cb256-r16-s126.
6. 根据 GDN probe 结果决定是否实现 Banked-K shared-V.
```

## 12. /goal 使用提示词

可以直接把下面这段交给 `/goal`:

```text
目标:
在 /home/lyj/mnt/project/zoology 的 flash-vqg 分支上, 按照 docs/plans/20260526-gdn-flash-fairness-experiment-plan.md 执行完整的 GDN 与 Flash-VQG 公平对照实验方案, 从 Phase 0 推进到 Phase 6。总体原则是先固定 GDN expanded-K blocked 结论, 再跑不改 FLA kernel 的 GDN kernel-compatible fairness probe, 同时补 Flash-VQG 主线稳定性, 根据结果决定是否实现 Banked-K GDN, 最后才考虑 FLA fork。执行过程中不得跳过阶段顺序和门控条件。

执行约束:
1. 每完成一个 Phase 后, 必须重新阅读 docs/plans/20260526-gdn-flash-fairness-experiment-plan.md, 检查下一步是否仍符合方案。
2. 每发生一次上下文压缩后, 恢复执行的第一步必须重新阅读该方案文档, 并用一句话说明当前处于哪个 Phase, 下一步是否偏离方案。
3. 正式 MQAR 训练默认使用 seed=123, data_seed=123, train_batch_size=64, gradient_accumulation_steps=4, effective_train_batch_size=256。
4. RTX 2080 Ti / sm75 上 GDN official 可比实验必须显式设置 GDN_KERNEL_DTYPE=float32。
5. 所有完整正式 MQAR 训练和正式 longer-MQAR eval 必须记录 started_at_utc, ended_at_utc, wall_clock_sec, gpu, gpu_name, status。
6. smoke/debug/失败/中断/未跑满预期 epoch 的实验只写 status/report/artifact, 不写 official canonical ledger。
7. 完整跑到预期 final checkpoint 的正式实验才写入对应 canonical ledger, 且必须保留 batch, gradient accumulation, dtype, GPU, run_id, seed, data_seed 等字段。
8. 不先 fork FLA, 不先做 VQ-routed GDN, 不对所有配置盲目补多 seed。
9. 如果实际结果触发 no-go 或偏离原计划, 先更新 status/report/artifact 说明原因, 再继续下一步。

阶段任务:
Phase 0: 固定 ek4-ev4, ek8-ev2, ek16-ev1 的 blocked 结论, 输出 capacity/accounting 表和失败原因。
Phase 1: 依次 smoke 并正式运行 ek4-ev4, mh-h4-k256-v128, mh-h8-k256-v64 的 seed123 4epoch GDN probe。
Phase 2: 补 Flash-VQG cb256-r8-s126 和 cb256-r16-s126, 资源允许再补 cb256-r6-s125 和 cb256-r10-s125。
Phase 3: 根据 Phase 1 结果决定是否实现 GatedDeltaNetBankedK shared-V。如果实现, 先 smoke, 再跑 seed123 4epoch。
Phase 4: 只对胜出的 Flash/GDN 点补多 seed, 输出 mean ± std 和主表字段。
Phase 5: 在主线公平结果稳定后做 longer-MQAR OOD, 再视资源做 passkey/needle 和小规模 long-context language modeling slice。
Phase 6: 只在必要时给出 FLA fork go/no-go 结论。若 go, 新开独立 kernel research goal, 不混入本目标。

停止条件:
1. Phase 0 到 Phase 6 均完成, 且每个阶段都有明确 done/no-go/blocked 记录。
2. 所有成功完成的正式实验已经写入对应报告, artifact 和 canonical ledger。
3. 所有失败, OOM, NaN/Inf, 中断或未跑满实验已经写入 status/report/artifact, 并说明没有写入 official ledger 的原因。
4. 已给出 Flash-VQG 与 best GDN baseline 的当前 official 对比结论, 包括 overall accuracy, hard slice accuracy, active state capacity, trainable params, dtype, GPU, wall-clock。
5. 已明确 Banked-K GDN 是 done, no-go, 还是 deferred, 并说明依据。
6. 已明确 FLA fork 是 go 还是 no-go。如果 go, 已给出下一步独立 /goal 的目标和验收标准。
7. 工作区最终状态已检查, 并说明是否有未提交变更。
```
