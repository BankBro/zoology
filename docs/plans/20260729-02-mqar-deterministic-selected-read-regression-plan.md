# MQAR 确定性 Selected-Read 回归实验计划

## 1. 实验登记

- Experiment ID: `20260729-02-mqar-deterministic-selected-read-regression`.
- 状态: `completed`, 终态为 `quality_recovered_but_not_deterministic`.
- 执行机器: `mclab-3090` 的 `Flash-VQG-tun` 容器.
- zoology base: `flash-vqg@3a7511a5bc0e8fc9950dc8d0f3759042a38683d5`.
- Flash-VQG base: `20260729-014613-300m-steady-state-memory-optimization@79fef6a8e9d3f41dfcbf40bf668ec83286dd5d62`.
- 默认环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`, PyTorch `2.6.0+cu118`, Triton `3.2.0`, FLA `0.4.2`.

## 2. 目标与假设

目标是在 A0 和 A1 共同使用的 selected-read custom backward 中, 将 `addr_proj` 的 CUDA 非确定性 `index_select` 反向归约替换为固定顺序的 deterministic segment accumulation, 然后重新验证 A1 `post_phase1` remat 的标准 MQAR 与 Longer-MQAR 质量.

核心假设是: 历史 A1 质量退化主要来自共享 selected-read backward 的非确定性梯度归约, 而不是 remat 改变模型数学语义. 修复后, fixed A0 与 fixed A1 应在相同 seed 下形成逐位一致的训练状态和评估结果.

本轮不复用历史 A0 作为严格基线. fixed A0 和 fixed A1 必须在相同修复 commit、数据、初始化、训练预算和评估协议下从头训练. 历史实验只用于条件性判断退化是否缓解.

## 3. 实现与固定口径

Flash-VQG 只修改 selected-read backward 的 `addr_proj` 梯度回填:

```text
per-query grad_addr_selected
-> stable head grouping
-> deterministic segment accumulation
-> grad_addr_proj
```

不改变 forward、模型公式、配置面、FP32 内部计算或梯度 dtype 契约. 修复同时作用于 A0 和 A1, 两组唯一变量仍是 `fox_gd_residual_remat_mode`.

正式训练固定为:

| 项目 | 固定值 |
|---|---|
| GPU 与精度 | RTX 3090, AMP BF16, FP32 master weights 与 optimizer state |
| 模型 | `baseline-r16-joint`, 1,160,390 参数 |
| GD 配置 | codebook 64, rank 16, read top-k 16, write top-k 4 |
| Block | `block_len=32`, `local_num_blocks=2` |
| Backend | grouped Triton, selected-read Triton remat, `fp32_boundary` |
| 数据 | canonical MQAR cache, data seed 123 |
| 训练 | B64, validation B16, GA4, 4 epochs, early stopping 关闭 |
| Seeds | 123, 124, 125 |
| Checkpoint | `last.pt` 主结果, `best.pt` 敏感性分析 |

正式矩阵:

| Variant | Remat mode | Seeds | Formal runs |
|---|---|---|---:|
| `a0-fixed-off` | `off` | 123, 124, 125 | 3 |
| `a1-fixed-post-phase1` | `post_phase1` | 123, 124, 125 | 3 |

## 4. 执行流程与停止规则

**(1)** Preflight 锁定两个仓库的干净 commit、canonical 环境、GPU、cache、init、resolved config、参数量和单变量差异.

**(2)** 低层门禁在默认非 deterministic 模式下重复 selected-read backward 8 次, 要求输出和全部梯度逐位一致; 同时与确定性参考比较, 要求 `atol=1e-5, rtol=1e-4`. 覆盖重复 head、跨 query chunk、FP32 boundary 和 BF16.

**(3)** seed124 执行 fixed A0/A1 的 128 optimizer-update 锁步轨迹, GA4, 共 512 microbatches. 每个 microbatch 比较 hidden、logits、loss 和全部梯度, 每个 update 比较参数与 Adam 状态, 要求全程逐位一致.

**(4)** fixed A0 和 fixed A1 分别执行两次 fresh-process 32-update 轨迹, 同 variant 的最终 model-state hash 必须一致.

**(5)** 执行 3-update integration smoke、optimizer-boundary resume、checkpoint save/load、validation 和两个 Longer-MQAR slice smoke.

**(6)** 任一前置门禁失败立即停止正式训练并保留现场. 门禁通过后, 按 `A0-s123, A1-s123, A0-s124, A1-s124, A0-s125, A1-s125` 串行执行 6 条正式训练. 运行失败时停止后续任务, 不自动补 seed 或覆盖输出.

**(7)** 对 last 和 best checkpoint 分别在 `1024x256`, `2048x512`, `4096x1024`, `8190x512`, `8190x2047` 上执行 500-example matching-BF16 评估, batch size 依次为 `128/64/32/16/16`. 相同 model-state hash 允许物理去重, 但保留全部逻辑评估事件.

## 5. 裁决与分析

A1 只有同时满足以下门槛才恢复为 300M 自然语言质量 pilot 候选:

```text
每个 seed 的 fixed A0/A1 final model-state hash 完全一致
mean(A1 - A0) standard MQAR 1024x256 >= -0.01
mean(A1 - A0) four-slice extrapolation macro >= -0.02
```

结果分类:

- `fully_recovered`: 三 seed hash 全部一致, 标准与外推 delta 均为 0.
- `quality_recovered_but_not_deterministic`: 质量门槛通过但存在 hash 分叉, A1 不晋升.
- `not_alleviated`: 任一质量门槛失败, 继续保留 A0 canonical.
- `correctness_failed`: 前置确定性门禁失败, 不启动正式训练.

分析使用三 seed 配对 delta、均值和 population SD. 训练 loss 与 validation accuracy 是轨迹代理指标; 固定 `1024x256` 和四个 Longer-MQAR 外推任务是目标指标. 历史 `-0.04020` 与 `-0.10562` 只作为条件性改善参考.

性能只记录 wall time、optimizer-step p50、peak allocated/reserved 和 runtime audit, 不作为本轮硬门槛.

## 6. 预算与产物

- 预计 6 个正式训练 run, 总预算约 3 至 4 GPU 小时.
- 预计 raw、checkpoint 和日志低于 1 GiB.
- 原始输出保留在 3090 实验目录; 轻量证据镜像回 2080 Ti 并逐文件校验 SHA256.
- Artifact 至少包含 determinism summary、trajectory summary、training ledger、Longer-MQAR 明细、paired quality、historical comparison、system summary、metadata 和 source manifest.
- 实验无论 completed、failed 或 aborted 都生成终态 report 并追加 `docs/EXPERIMENT_LOG.md`; 仅在当前结论或下一步变化时更新两仓 `STATUS.md`.
- Flash 修复只有在本实验通过后才合入当前 300M 显存优化分支. zoology 实验分支无论成功或失败都合入 `flash-vqg`, 不合入 `main`.
