# GDN expanded-K 与 longer-MQAR 进展报告, 2026-05-26

## 目标

本轮目标是在不改变原始 `GatedDeltaNet` 语义的前提下, 新增 `GatedDeltaNetExpandedK`, 固定 `num_heads=2`, `train_batch_size=64`, `gradient_accumulation_steps=4`, `GDN_KERNEL_DTYPE=float32`, 总 active state capacity 为 131072, 测试扩大 GDN key/address 维度是否能解释 Flash-VQG 在 longer-MQAR 上的优势。

计划中的第一阶段配置为:

| run_id | per-head K | per-head V | state capacity |
|---|---:|---:|---:|
| gdnxk-h2-ek4-ev4-s123-d123-b64-ga4-fp32-noearly4ep | 256 | 256 | 131072 |
| gdnxk-h2-ek8-ev2-s123-d123-b64-ga4-fp32-noearly4ep | 512 | 128 | 131072 |
| gdnxk-h2-ek16-ev1-s123-d123-b64-ga4-fp32-noearly4ep | 1024 | 64 | 131072 |

其中 `ek16-ev1` 是关键 endpoint, 它在抽象数量上对齐 Flash `cb64-r16` 和 `cb256-r4` 的 `K=1024, V=64`, 但机制上仍不是同构。

## 已完成的工程适配

- 新增 `GatedDeltaNetExpandedK`, 支持显式 `expand_k`, 默认 `expand_k=1` 时保持原始 GDN 维度行为不变。
- 新增 `20260526-gdn-expanded-k` 训练脚本配置, 固定 `num_heads=2`, `64x4`, `GDN_KERNEL_DTYPE=float32`, 目标配置为 `ek4-ev4`, `ek8-ev2`, `ek16-ev1`。
- 训练 manifest 增加 `started_at_utc`, `ended_at_utc`, `wall_clock_sec` 字段。
- 新增单元测试覆盖默认 GDN 维度不变, expanded-K 维度计算, 以及三组配置的 active GDN state capacity。

## Kernel smoke 结果

环境:

| 字段 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti |
| compute capability | sm75 |
| torch | 2.6.0+cu118 |
| dtype policy | float32 |
| GDN_KERNEL_DTYPE | float32 |
| smoke input | batch=1, seq_len=8, d_model=128 |
| started_at_utc | 2026-05-26T05:37:36Z |
| ended_at_utc | 2026-05-26T05:45:41Z |
| wall_clock_sec | 485 |

结果:

| 配置 | head_k_dim | head_v_dim | 结果 |
|---|---:|---:|---|
| ek4-ev4 | 256 | 256 | 最小 forward/backward 通过, 首次 forward 182.250s, backward 250.422s |
| ek8-ev2 | 512 | 128 | forward 失败, `AssertionError: current kernel does not support head dimension larger than 256.` |
| ek16-ev1 | 1024 | 64 | 未启动, 因同一 FLA kernel 限制必然超过 `K <= 256` |

FLA 源码位置:

`/home/lyj/miniconda3/envs/flash-vqg/lib/python3.12/site-packages/fla/ops/common/chunk_delta_h.py`

相关断言:

- line 454: `assert K <= 256, "current kernel does not support head dimension larger than 256."`
- line 499: `assert K <= 256, "current kernel does not support head dimension being larger than 256."`

## 当前结论

这次不是简单“卡住”, 而是两件事叠加:

1. 第一次 Triton/FLA kernel 编译和执行确实很耗时, `ek4-ev4` 的最小 forward/backward 用了约 433 秒。
2. 更关键的是当前 FLA chunk kernel 对 `head_k_dim` 有硬限制, 只支持 `K <= 256`。因此 `ek8-ev2` 和关键的 Flash-like endpoint `ek16-ev1` 在当前实现和硬件口径下不可执行。

所以本轮没有启动正式 MQAR 训练, 没有产生 final checkpoint, 没有写入正式训练结果行, 也没有启动 longer-MQAR formal eval。该结果满足本 goal 的停止条件 4: 记录 shape, dtype, GPU, batch, traceback/日志和时间后停止, 报告该设计当前不可执行或需要缩小配置。

## Phase 0 capacity/accounting 固化, 2026-05-26

后续 GDN 与 Flash-VQG 公平对照方案将本节作为 Phase 0 固化结论。三组 expanded-K 配置的 active state capacity 均为 131072, 但可训练性不同:

| run_id | per-head K | per-head V | trainable params | 可训练性 | ledger 处理 |
|---|---:|---:|---:|---|---|
| `gdnxk-h2-ek4-ev4-s123-d123-b64-ga4-fp32-noearly4ep` | 256 | 256 | 1335942 | 最小 forward/backward smoke 通过, 不是正式训练 | 不写 official ledger |
| `gdnxk-h2-ek8-ev2-s123-d123-b64-ga4-fp32-noearly4ep` | 512 | 128 | 1404422 | FLA chunk state-update kernel `K<=256` 断言失败 | 不写 official ledger |
| `gdnxk-h2-ek16-ev1-s123-d123-b64-ga4-fp32-noearly4ep` | 1024 | 64 | 1641414 | 因同一 kernel 限制未启动 | 不写 official ledger |

补充 accounting 表见:

`docs/artifacts/20260526-gdn-flash-fairness/phase0-gdn-expanded-k-accounting.csv`

该表按实际 `LanguageModel(config.model)` 口径统计整模型 trainable params, 并保留 smoke 时间, GPU, dtype, batch 和 ledger policy 字段。`ek4-ev4` 只能作为 current-kernel-compatible probe, 不能写成 true `K=1024,V=64` 对照。

## 对实验方案的影响

当前结果不能回答“只扩大 GDN K/address capacity 是否足以追上 Flash”, 因为最关键的 `K=1024, V=64` GDN 对齐配置无法跑起来。

可继续推进的方向有三个:

1. 缩小问题, 只跑 kernel 可支持的 `ek4-ev4`, 用它回答“从 K=64 提升到 K=256 是否有收益”。这不能直接对齐 Flash 的 `K=1024`。
2. 改 FLA kernel 或引入替代实现, 让 `head_k_dim > 256` 可训练。这个工作量和风险高于本轮 goal 的“最小适配”边界。
3. 换机制对照, 不强行把 GDN 连续 key 维度拉到 1024, 而是把实验问题改成“在相同 state capacity 下, GDN 的 K/V 分配曲线是否存在最佳点”。

## Artifact

- 训练族 ledger: `docs/artifacts/gdn-expanded-k/gdn-expanded-k-summary.csv`
- kernel smoke: `docs/artifacts/gdn-expanded-k/kernel-smoke-20260526.json`
- longer-MQAR 状态: `docs/artifacts/longer-mqar/gdn-expanded-k-probe-20260526/status.json`
