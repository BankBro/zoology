# GDN expanded-K longer-MQAR probe, 2026-05-26

本目录记录 GDN expanded-K 方案没有进入正式 longer-MQAR eval 的原因。

当前 `GatedDeltaNetExpandedK` 已完成最小代码适配和单元测试, 但在 RTX 2080 Ti/sm75, `GDN_KERNEL_DTYPE=float32` 下, FLA `chunk_gated_delta_rule` 路径存在 `head_k_dim <= 256` 的 kernel 限制。`ek4-ev4` 的最小 forward/backward 可以通过, 但首次 JIT/执行耗时约 433 秒。`ek8-ev2` 在 forward 时报错, `ek16-ev1` 因同一限制没有启动。

因此本轮未产生正式 checkpoint, 未写入训练结果行, 也没有启动 longer-MQAR formal eval。
