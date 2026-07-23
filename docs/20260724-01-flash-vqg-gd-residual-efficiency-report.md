# Flash-VQG GD residual 显存与运行时间优化报告

实验 ID: `20260724-01-flash-vqg-gd-residual-efficiency`.

## 1. 结论

本轮已经在不降低任何模型超参数, 不更改 `gd_residual_v1` 数学机制, 不启用 TF32/FP16/BF16, 不使用 gradient checkpointing 的前提下, 完成并启用了两项核心内核优化和若干训练循环清理:

1. 将 Python grouped recurrence 替换为按 active group 执行的 FP32 Triton recurrence, 原样保留 decay, prediction, error, `smooth_p4` update softcap, rank-1 update 和 tail decay 顺序.
2. 将 residual selected read 融合为直接从 `M_state + idx_remote + top_idx` 读取的 Triton custom autograd, forward 不再向全局 HBM 写出 `M_remote`, `M_sel`, `C_sel`, `z`, `d_read` 和 `proposal`, backward 重算中间量并使用确定性 segmented accumulation.
3. Core 模式不构建未请求的诊断中间量. 训练循环不计算未使用的 `argmax`, 在 GPU 上累计 loss 后每个 optimizer boundary 只同步一次, scalar metrics 批量 D2H, `zero_grad(set_to_none=True)`, schedule counter 保留在 host.

最终 Flash-VQG 相对自身 reference 实现取得:

| 机器 | 阶段 | reference p50 | optimized p50 | 加速 | reference allocated | optimized allocated | 降幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2080 Ti GPU1 | eval `B16/T1024` | 648.006 ms | 77.818 ms | 8.327x | 3.205 GiB | 1.020 GiB | 68.16% |
| 2080 Ti GPU1 | train `B64/T256/GA4` | 4423.033 ms | 900.448 ms | 4.912x | 6.711 GiB | 3.052 GiB | 54.53% |
| 3090 GPU0 | eval `B16/T1024` | 453.241 ms | 54.055 ms | 8.385x | 3.205 GiB | 1.020 GiB | 68.16% |
| 3090 GPU0 | train `B64/T256/GA4` | 3189.973 ms | 690.209 ms | 4.622x | 6.709 GiB | 3.051 GiB | 54.52% |

在可执行冻结 GDN 的 2080 Ti 上, Flash/GDN 的 core p50 与 allocated peak 四项均满足 `<=2.0x`:

| 阶段 | Flash p50 | GDN p50 | 时间比 | Flash allocated | GDN allocated | 显存比 |
|---|---:|---:|---:|---:|---:|---:|
| eval | 77.818 ms | 41.297 ms | 1.884x, PASS | 1.020 GiB | 1.021 GiB | 0.999x, PASS |
| train | 900.448 ms | 597.183 ms | 1.508x, PASS | 3.052 GiB | 1.875 GiB | 1.627x, PASS |

补充 formal诊断口径中, train仍为 1.623x, 但 eval因每个 Flash batch额外计算并回传 206项 layer diagnostics而为 2.510x. 该值在报告中单列为诊断成本, 没有用降低采样频率掩盖; 对称 metrics-off模型 core仍以 1.884x通过硬线.

双机 seed124/125 正式 1ep 质量回归全部通过: 所有 `1024x256` accuracy 均高于 0.85, 同 seed 跨机器 gap 分别为 0.289pp 和 2.984pp, 所有 overall accuracy 相对历史对应 baseline 的退化均小于 1pp, 且无 OOM, NaN, Inf 或 Traceback.

本实验唯一无法关闭的硬条件是 3090 上的冻结 GDN 对照. 该 GDN FP32 kernel 在 sm86 编译时需要 147456 bytes shared memory, 而硬件上限为 101376 bytes, train 和 eval 均在正式计时前抛出 `OutOfResources`. 按任务约束不能改 GDN source/config, 不能改精度, 也不能 monkeypatch launch 参数, 因此 3090 Flash/GDN 时间, 显存和 full-epoch ratio 必须记为 unavailable, 不能伪造为通过或失败. 这不影响 Flash-VQG 自身在 3090 上的绝对优化结果, 但意味着字面上的“双机全部 GDN ratio”目标受外部对照不可执行所阻塞.

## 2. 固定实验口径

### 2.1. 源码

| 仓库 | base | 优化分支 | 本报告源码 commit |
|---|---|---|---|
| Flash-VQG | `0eba390` | `20260724-gd-residual-efficiency` | `ec770f33676036432c6514acd1ac05bd2d01f3e8` |
| zoology | `8ba9618` | `20260724-gd-residual-efficiency` | `0dfea8e07b003c3b4dbe35af0f72f63c07b8919c` 为最终实验执行代码, artifact/report 由后续收尾 commit 固化 |

两仓均只推送到新分支, 未 merge 回原分支. 多机源码通过 Git 同步, 没有用 `scp` 覆盖源码.

### 2.2. 模型和数值策略

- Flash: `d_model=128`, `num_layers=2`, `num_heads=2`, `key_dim=value_dim=64`, `codebook_size=64`, `block_len=32`, `local_num_blocks=2`, `gd_rank=16`, `read_topk=16`, `write_topk=4`.
- `update_norm_softcap=0.5`, mode `smooth_p4`; residual injection warmup optimizer step `0->512`; read/write schedule 不变.
- Train 为 `B64/T256/GA4`, eval 为 `B16/T1024`, default dropout.
- 全流程 FP32, PyTorch matmul TF32 off, cuDNN TF32 off, Triton IEEE.
- GDN 固定为 `gdnxk-h2-ek4-ev4-usegate0`, 不修改源码和 config.
- Flash 与 GDN active state capacity 均为 131072. Flash 参数量 1160390, GDN 参数量 1335942.

### 2.3. 数据和初始化

两机预检均验证通过:

- Cache tensor content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- Flash init model-state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- GDN init model-state hash: `bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6`.
- Epoch-0 batch-order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.

2080 Ti 容器使用物理 GPU1, capability sm75. 3090 宿主机实际只有一张 GPU, 宿主机和容器索引均为 GPU0, capability sm86, 因此不能按最初文字要求选择不存在的 GPU1. 两机均为 PyTorch 2.6.0+cu118, CUDA runtime 11.8.

## 3. 对网页建议的验证与纠正

网页分析的主方向可以应用, 但不能直接当成当前代码事实. 本轮实测后的判断如下:

| 网页建议或推断 | 当前代码与实测结论 |
|---|---|
| `M_sel` 是主要显存瓶颈 | 证实. 消除 selected-read materialization 后 eval allocated 从 3.205 GiB 降至 1.020 GiB, train 从 6.711 GiB 降至约 3.052 GiB. |
| 两层各有一份 2 GiB `M_sel`, 共 4 GiB | 纠正. 当前两层布局实际是一个 BaseConv 和一个 FlashVQG mixer, 只有一个活跃 GD residual 层, 因而全模型 reference `M_sel` 是 2 GiB. |
| `M_remote` 是可消除的 256 MiB copy | 证实. Fused selected read 直接使用 `M_state + idx_remote`, 不再形成完整 `M_remote`. |
| grouped recurrence/Python 小 kernel 可能是时间热点 | 证实. 单独 fused grouped recurrence 将 2080 Ti eval smoke 从 648.006 ms 降到 110.568 ms, train 从 4423.033 ms 降到 918.787 ms. |
| fused linear cross entropy 应优先 | 未证实为首要项. 最终 2080 Ti train 中 LM head 约 12.0 ms, CE 约 9.1 ms, 远小于 backbone/backward, 因此本轮不承担额外语义和 backward 风险. |
| 普通 query chunking 可作为最终训练解法 | 不采用. 普通 autograd chunking 可能仍累计 saved activations, 最终采用 custom backward rematerialization. |
| 直接切旧 Triton backend | 不适用. 本轮为 GD residual 单独实现并验证了受限 shape/dtype 的新 backend, unsupported 输入继续 fallback. |
| exact recurrent cache 是推理主方向 | 对自回归 decode 成立, 但明确不在本轮范围. 本报告的 inference 指 MQAR full-sequence eval, 没有声称完成 decode cache. |

静态 shape 账本也得到修正. 对单个活跃 GD residual 层, reference FP32 大项为:

| 张量 | shape | 大小 | 最终状态 |
|---|---|---:|---|
| `M_state` | `[64,2,8,64,64,16]` | 256 MiB | 保留 |
| `M_remote` | 同上 | 256 MiB | 消除完整 copy |
| `M_sel` | `[64,2,8,32,16,64,16]` | 2048 MiB | 消除全局 materialization |
| `C_sel` | `[64,2,8,32,16,64]` | 128 MiB | kernel 内加载 |
| `z` / `d_read` | `[64,2,8,32,16,16]` | 各 32 MiB | kernel 内计算, backward 重算 |
| `proposal` | `[64,2,8,32,16,64]` | 128 MiB | 在写回前归约 |
| `logits` | `[64,256,8192]` | 512 MiB | 保留, 非本轮第一热点 |

## 4. 优化实现

### 4.1. Fused grouped recurrence

Reference 路径按 group bucket 后在 Python 中循环事件, 产生大量小 kernel 和 launch/CPU 开销. 新 backend 让一个 Triton program 处理一个 active group, state tile 固定为 `[64,16]`, 并在 program 内顺序执行该组事件. Custom backward 按相同依赖关系计算梯度, 保留 FP32 accumulation 和 detached softcap scale 的原语义.

该优化是最大的时间收益来源. 它没有改变 event 支持集, group 内事件顺序, decay 或更新公式.

### 4.2. Fused selected residual read

Reference 路径先 `index_select` 得到 `M_remote`, 再按每个 query/top-k 复制成 2 GiB `M_sel`, 之后物化 code, address coordinate 和 proposal. 新 kernel 接收原始 `M_state`, `idx_remote`, `top_idx`, `omega`, query, codebook 和 address projection, 在 kernel 内完成:

```text
selected M/code load
  -> z=(q-code)@addr_proj
  -> d=z/max(||z||, eps)
  -> proposal=M@d
  -> omega weighted top-k reduction
  -> u_res
```

Backward 不保存上述大中间量, 而是重算. 多个 query 对同一个 state/code 的梯度采用稳定排序后的 deterministic segmented accumulation, 避免无序 FP32 atomic 对训练轨迹引入额外不确定性. `top_idx` 仍由原 reference 计算, 因而没有重写 top-k 或 tie 语义.

### 4.3. Core event pack 和 host/training-loop 清理

- Metrics off 时不再构建只供诊断使用的 event statistics.
- Schedule forward count 从每次 GPU `.item()` 改为 host counter, 保留 `fill_`, `item`, resume 和 warmup count 接口语义.
- Train 不再扫描完整 logits 做未使用的 `argmax`.
- Loss 在 GPU 上累计, optimizer boundary 单次 D2H.
- Scalar metrics 先 stack 再一次 D2H.
- `optimizer.zero_grad(set_to_none=True)`.

2080 Ti formal train 单次 A/B 中, loop policy 从 941.008 ms 降至 909.681 ms, 约 3.33%. 这部分不是主加速来源, 但避免公共训练循环掩盖 mixer 优化收益.

### 4.4. 明确拒绝或延期的候选

- Triton stable event-order/counting pack: eval smoke 从 76.438 ms 回退到 80.694 ms, 已删除该候选, 保留现有 semivec pack.
- Fused LM projection + CE: LM head + CE 占比不足以优先承担 custom backward 风险.
- Gradient checkpointing: 可降显存但通常增加时间, 且任务明确不接受为最终方案.
- 降 batch, rank, top-k, codebook, 序列长度或精度: 均未使用.
- TF32/BF16/FP16: 均未使用.
- State streaming: `M_state` 当前仅 256 MiB, 在 selected read 消除后 ROI 低于实现和反向风险, 本轮不做.
- Decode cache: 不在本轮 full-sequence eval 范围.

## 5. 性能与显存结果

### 5.1. 稳态测量方法

每个 hard timing 点使用独立 fresh process, warmup 5, active 10, 共 3 次 repeat. 表中是“三次各自 p50 的中位数”; train 单位为一次包含 4 个 microbatch 的 optimizer step, eval 单位为一个 `B16/T1024` batch. Timing 和 memory snapshot 分开运行, profiler 数字不进入硬性能表.

2080 Ti 最终 p90 中位数为 eval 78.326 ms, train 909.843 ms. GDN 对应 p90 为 eval 41.516 ms, train 600.007 ms. 结论不依赖单个幸运样本.

3090 最终 Flash p90 中位数为 eval 64.140 ms, train 720.273 ms. 由于冻结 GDN 不可执行, 只报告 Flash 绝对值与相对自身 reference 的收益.

按 timed unit处理的 token数计算, 2080 Ti Flash throughput为 eval 210.5k token/s, train 72.8k token/s; GDN分别为 396.7k和 109.7k token/s. 3090 Flash分别为 303.1k和 95.0k token/s. Train p50扣除 optimizer step后除以 GA4得到的 microbatch-equivalent wall为: 2080 Ti Flash 224.959 ms, GDN 149.151 ms; 3090 Flash 172.429 ms. 这是由对称 optimizer-step测量派生的等价值, 不是另一次异步 microbenchmark.

Allocator reserved不是 hard threshold, 但也保留报告. 2080 Ti eval为 Flash 1.914 GiB, GDN 1.895 GiB, ratio 1.010x; train为 Flash 3.422 GiB, GDN 2.438 GiB, ratio 1.404x. Allocated和 reserved均未通过把显存压力转移到 allocator cache来制造表面收益.

分段 p50也显示剩余瓶颈在 backward. 2080 Ti Flash每个 GA4 optimizer step的 backbone和 backward分别约 124.463 ms和 750.752 ms; GDN分别约 147.323 ms和 424.135 ms. Flash forward已经不慢于 GDN, 当前 1.508x train差距主要来自 fused custom backward, 这是后续继续优化时最有价值的方向.

### 5.2. Formal diagnostics口径

Formal模式保留 Flash当前 206项 layer diagnostics, eval还与 GDN对称执行完整 logits `argmax`. 三次 fresh-process中位数为:

| 机器 | 阶段 | Flash formal | GDN formal | Flash/GDN | 对应 core ratio |
|---|---:|---:|---:|---:|---:|
| 2080 Ti | eval | 108.584 ms | 43.252 ms | **2.510x** | 1.884x |
| 2080 Ti | train | 984.986 ms | 606.836 ms | 1.623x | 1.508x |
| 3090 | eval | 74.780 ms | unavailable | unavailable | unavailable |
| 3090 | train | 730.778 ms | unavailable | unavailable | unavailable |

Formal eval超过 2x必须保留, 不能与 core结果混写成通过. 归因也很明确: 2080 Ti Flash core/formal backbone为 72.526/102.093 ms, formal额外计算诊断统计约 29.6 ms; 批量 D2H仍需 5.787 ms. GDN没有对应的 Flash-specific diagnostics, formal只从 core 41.297 ms增加到 43.252 ms. `argmax`本身只有约 0.94 ms, 不是主要差距.

本轮已消除逐 scalar同步, 但没有通过降低 formal diagnostics采样频率来把 2.510x“做成”通过, 因为这会改变正式观察口径. 如果实际训练允许每 N个 validation batch采样一次 layer diagnostics, 模型输出和梯度不会改变, formal wall可接近 core; 该策略应作为明确的观测策略变更另行启用, 不能冒充内核加速. Hard模型性能判断使用 metrics/trace/logging均关闭的对称 core口径, formal比值作为诊断系统实际成本单列.

### 5.3. Warmed full epoch

同一 runner 先预编译 train `T64/128/256` 和 eval `T64/128/256/512/1024`, 再重新构建 canonical init, 完整执行 2815 个 train microbatch, GA4 共 704 optimizer steps, 并在 step 176/352/528 及结尾做 validation.

2080 Ti GPU1 正式对称运行得到:

| 模型 | Train wall, 不含 validation | Validation wall | Total wall | Epoch peak allocated |
|---|---:|---:|---:|---:|
| Flash-VQG | 329.363 s | 58.024 s | 386.916 s | 3.560 GiB |
| GDN | 241.211 s | 32.145 s | 273.356 s | 2.375 GiB |
| Flash/GDN | 1.365x | 1.805x | **1.415x, PASS** | 1.499x |

表中是每个模型 3个独立 fresh-process repeat的中位数. Flash有效样本为 r1/r2/r3的 389.542/386.034/386.916 s; GDN有效样本为 r1/r3/r4的 273.356/271.045/274.651 s. GDN r2与一次本机 CPU-heavy pytest有部分重叠, 因此即使其 273.941 s没有明显异常, 仍从 hard median中排除并用干净 r4替代; raw结果和 `EXCLUDED.json`均保留. GPU0 preliminary 对称运行还得到 GDN 350.822 s, Flash 422.189 s, ratio 1.203x, 只作为辅助证据.

### 5.4. Cold-start和编译成本

使用全新的空 `TRITON_CACHE_DIR`, `warmup=0`, `active=1`, 只计首次 timed iteration, 不含 Python process/model/data setup:

| 机器 | 模型 | eval首次迭代 | train首次 GA4 |
|---|---|---:|---:|
| 2080 Ti | Flash-VQG | 6.728 s | 12.777 s |
| 2080 Ti | GDN | 276.757 s | 553.830 s |
| 3090 | Flash-VQG | 5.476 s | 10.749 s |
| 3090 | GDN | 编译失败 | 编译失败 |

这些数字主要是 Triton/FLA JIT和 autotune成本, 不代表 steady-state. Warmed epoch runner会在 throwaway model上预编译全部使用 shape, 再重建 canonical init并开始计时. 实际部署和一次性作业必须采用相同预热或持久化 Triton cache, 否则首步延迟可高出数个数量级.

### 5.5. 3090 GDN 硬件阻塞

冻结 GDN 在 train 和 eval 均得到相同错误:

```text
OutOfResources: shared memory, Required: 147456, Hardware limit: 101376
```

失败发生在 FLA `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 的 Triton 编译阶段, 不是 Flash OOM 或输入错误. 当前 `K=256`, FP32 tile 的共享内存需求超过 sm86 每 block 上限. 通过改 block size, stages, GDN source, dtype 或 kernel dispatch 可能使其运行, 但都会违反“冻结 GDN”和“FP32”口径. 因此本实验只保留失败证据, 不修改对照模型.

## 6. 等价性和质量

### 6.1. Full-model one-step

固定输入, initialization 和 RNG 后, 两机 reference 与 fused 组合均通过门槛:

| 机器 | eval hidden max abs | eval relative L2 | loss abs | gradient max abs | gradient max relative L2 | one-step parameter max abs |
|---|---:|---:|---:|---:|---:|---:|
| 2080 Ti | 7.15e-7 | 2.23e-8 | 0 | 4.66e-10 | 1.45e-6 | 1.48e-7 |
| 3090 | 7.15e-7 | 2.22e-8 | 0 | 4.66e-10 | 1.54e-6 | 1.60e-7 |

Selected-read repeated forward/backward 还通过了 deterministic exact-repeat 测试. 收尾 targeted rerun 覆盖 GD residual, attention facade/forward/config/state-dict compatibility, guards和 phase2 metrics, Flash-VQG共 103 passed; zoology benchmark, metrics和 train logging合计 15 passed.

### 6.2. 32/128/512-step 轨迹

3090 使用实际 canonical batch order 对 reference 和 fused 组合分别训练到 512 optimizer steps. 该测试必须如实报告为自动严格门槛未通过:

| Step | 当前 loss abs | loss trajectory relative L2 | parameter max abs | parameter relative L2 |
|---:|---:|---:|---:|---:|
| 32 | 0 | 4.61e-8 | 1.66e-5 | 1.55e-3 |
| 128 | 0 | 4.36e-8 | 3.39e-5 | 4.77e-4 |
| 512 | 1.19e-2 | 4.52e-3 | 4.85e-1 | 3.93e-1 |

首次 loss difference `>1e-5` 出现在 step 242, `>0.01` 出现在 step 300, 与 residual injection warmup 开始放大 residual 影响的阶段一致. Reference 512-step 用时 1324.043 s, fused 用时 188.764 s, 约 7.01x.

这说明“数学公式和支持集不变”不等于“经过数百次非凸更新仍 bitwise 同轨迹”. One-step 梯度误差很小且确定, 但 FP32 reduction association 的差异会被优化过程放大. 因此本轮没有用口头保证覆盖该风险, 而是继续以双 seed, 双 GPU 的正式最终效果作为效果验收. 如果未来项目要求 bitwise trajectory, 必须保留 reference reduction order, 代价将显著损害当前性能收益.

### 6.3. 双机正式 1ep 质量

| Seed | 2080 Ti overall | 2080 Ti 1024x256 | 3090 overall | 3090 1024x256 | 跨机 hard gap | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 124 | 0.987586 | 0.933207 | 0.986721 | 0.930316 | 0.289pp | PASS |
| 125 | 0.985244 | 0.921488 | 0.990554 | 0.951328 | 2.984pp | PASS |

对应历史 overall baseline 为 2080 Ti seed124/125 的 0.982/0.986, 3090 seed124/125 的 0.988/0.991. 本轮 delta 分别为 +0.005586, -0.000756, -0.001279, -0.000446, 均满足退化不超过 0.01.

最初 formal v2 因 `cudnn_allow_tf32=true` 且 seed124 被中断而整体作废, 已在 `INVALID.json` 记录, 不纳入正式结果. 有效 v3/v1 明确记录 PyTorch matmul TF32 off, cuDNN TF32 off 和 Triton IEEE. 四个有效 run 均写入 formal ledger; checkpoint 留在来源机器, lightweight summary 保存 checkpoint 和 model-state SHA256.

有效 formal运行的 Flash-VQG源码均为 `ec770f33676036432c6514acd1ac05bd2d01f3e8`; 2080 Ti zoology为 `e2e2deae0df240269bf41c954ea1d8d2961dd240`, 3090 zoology为 `4c171d64a260e4daaf0a2a0ef70756ed9871a02b`. 后续 zoology commit只增加兼容性证据, epoch runner, checkpoint摘要和 artifact汇总, 未改变该正式训练路径.

## 7. Artifact 和复现入口

正式 artifact 目录为 `docs/artifacts/20260724-01-flash-vqg-gd-residual-efficiency/`, 主要文件包括:

- `baseline-timing.csv`, `baseline-memory.csv`.
- `final-timing.csv`, `final-memory.csv`.
- `performance-ratios.csv`, `memory-ratios.csv`, `epoch-ratios.csv`.
- `formal-timing.csv`, `formal-performance-ratios.csv`, `cold-start.csv`.
- `candidate-waterfall.csv`, `tensor-lifetime.csv`, `metrics-on-off-comparison.csv`.
- `equivalence-summary.csv`, `trajectory-equivalence.csv`.
- `formal-quality.csv`, `formal-paired-quality.csv`, `formal-ledger.csv`.
- `gdn-3090-compatibility.csv`.
- `source-manifest.csv`, `metadata.json`, `README.md`.

Runner 和完整命令说明位于 `zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/`. Raw profiler trace, allocator snapshot, logs, resolved config 和 source/env evidence 的来源路径, mirror 路径及 SHA256 均由 `source-manifest.csv` 追踪. 大型 checkpoint 和 raw swanlog 按规范留在来源机器, 不提交 Git.

## 8. 最终判定和后续边界

已完成并通过:

- Flash 自身双机 train/eval 大幅加速和 allocated peak 降低.
- 2080 Ti 上同 capacity GDN 的 core time/memory 四项 `<=2x`.
- 2080 Ti warmed full-epoch `<=2x`.
- 双机 one-step 等价性和 deterministic repeat.
- 双 seed, 双机正式质量门槛.
- 新分支, fallback, tests, artifact, report 和远端推送.

无法关闭:

- 3090 frozen GDN train/eval/full-epoch 本身无法在 FP32 sm86 上编译, 因而所有 3090 Flash/GDN ratio unavailable.
- 保留 206项 Flash layer diagnostics的 2080 Ti formal eval为 2.510x GDN, 虽然对称 core eval为 1.884x. 该差异作为诊断成本单列, 未靠降低采样频率掩盖.
- 512-step strict trajectory 不保持同一数值轨迹, 虽然数学语义, one-step 误差和正式最终效果验收通过.

因此本轮优化代码可以作为默认的 `baseline-r16-joint` 高效 backend 使用, 并保留 reference fallback. 若要求继续关闭 3090 对照, 需要用户单独授权改变 GDN 对照边界, 例如允许修复其 launch configuration 或选择可在 sm86 FP32 执行的等容量 GDN kernel; 这属于新的对照实验, 不能偷偷并入本轮结果.
