# Flash-VQG GD residual 显存与性能优化计划

实验 ID: `20260724-01-flash-vqg-gd-residual-efficiency`.

## 1. 目标与完成条件

本实验优化 `Flash-VQG` 的 `gd_residual_v1 baseline-r16-joint`, 覆盖 FP32 训练和 MQAR 全序列 eval. 不实现 `use_cache` 或自回归 decode cache, 不修改 GDN 实现.

优化必须同时满足以下条件:

- 不改变模型数学语义, 训练目标, top-k 支持集, 状态更新顺序, `smooth_p4` softcap, injection warmup 和 read/write schedule.
- 不降低 rank, read/write top-k, codebook, batch, 序列长度, 层数或精度, 不启用 TF32, FP16 或 BF16.
- 保留 reference fallback, 旧 config/checkpoint 兼容和未启用优化时的原行为.
- 在 2080ti GPU1 和 3090 GPU1 上分别达到 Flash/GDN `<=2.0` 的 train optimizer-step p50, eval batch p50, warmed 1ep wall-clock, train allocated peak 和 eval allocated peak.
- 通过算子, full-model, one-step, 短轨迹及双机正式质量回归.
- 完成可审计 artifacts, 报告和两仓目标分支推送.

硬判定使用同卡独立 3 次 fresh-process 重复的中位数. 接近 2x 边界时同时检查原始样本, p90 和 ABBA 交替顺序, 不用单个幸运样本判定通过.

## 2. 固定版本与配置

### 2.1. 代码版本

| 仓库 | base | 目标分支 |
|---|---|---|
| `/home/lyj/mnt/project/Flash-VQG` | `0eba390` | `20260724-gd-residual-efficiency` |
| `/home/lyj/mnt/project/zoology` | `8ba9618` | `20260724-gd-residual-efficiency` |

两仓分阶段 commit/push, 本实验内不 merge 回原分支. 多机只通过 Git 同步源码.

### 2.2. Flash 固定配置

- `d_model=128`, `n_layers=2`, 实际布局为 `BaseConv + FlashVQGMixer`, 因此只有 1 个活跃 GD residual 层.
- `num_heads=2`, `key_dim=value_dim=64`, `codebook_size=64`.
- `block_len=32`, `local_num_blocks=2`, `gd_rank=16`.
- `read_topk=16`, `write_topk=4`.
- `update_norm_softcap=0.5`, mode `smooth_p4`.
- injection warmup 为 optimizer step `0->512`, 当前等价 train-forward count 为 `0->2048`.
- train `B64/T256/GA4`, eval `B16/T1024`, default dropout.
- FP32, float32 matmul precision `highest`, TF32 off.

### 2.3. GDN 冻结对照

固定 `gdnxk-h2-ek4-ev4-usegate0`: `d_model=128`, `n_layers=2`, heads 2, per-head `K/V=256/256`, active capacity `131072`, 历史参数量 `1335942`.

Flash `cb64-r16` active capacity 同为 `131072`, 历史参数量 `1160390`. 该对照只表示 same-scale engineering comparator, 不表示两种结构数学同构.

GDN 复用相同 cache, 输入 batch 和 batch order, 使用并记录自身确定性 init hash, 不加载 Flash init. GDN kernel 输入 dtype 必须实测确认为 FP32.

### 2.4. 固定数据证据

- Cache tensor hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- Flash init model-state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- Batch-order hash: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.

训练前在两机对实际加载对象重新计算稳定内容 hash. 文件名和历史记录不能替代本轮检查.

## 3. 已确认事实与待验证假设

Train `B64/T256` 与 eval `B16/T1024` 都有 `B*T=16384` 和 `B*N=512`. 当前单个活跃 Flash 层的主要 FP32 显式张量为:

| 张量 | 静态大小 | 当前判断 |
|---|---:|---|
| `M_sel` | 2 GiB | 最大单项, 第一显存嫌疑 |
| `M_state` | 256 MiB | 保存全部 block-entry state |
| `M_remote` | 256 MiB | 规则 `idx_remote` 的完整复制 |
| `C_sel` | 128 MiB | selected code materialization |
| `Q-C` 临时量 | 128 MiB | residual address 临时量 |
| `z`, `d_read` | 各 32 MiB | rank coordinate |
| `proposal` | 128 MiB | selected residual proposal |
| 完整 logits | 512 MiB | 公共 LM head 输出 |

关键 GD 显式账本约 2.9–3.0 GiB, 不是历史网页分析中的双层 5.6 GiB. 该静态合计不能直接当作 allocator peak, 必须用 memory snapshot 和 autograd saved-tensor 证据验证生命周期.

待验证时间热点包括 LM head/CE/validation, selected residual read, event pack, grouped recurrence, metrics GPU 计算/D2H 和 schedule `.item()` 同步. 不在 baseline 前预设第一热点.

## 4. 方案比较与推荐路线

### 4.1. 路线 A: 只清理训练循环和公共开销

候选包括训练不做 `argmax`, GPU 累计 loss, `zero_grad(set_to_none=True)`, 批量 metrics D2H, target-only projection/CE 和现有 FLCE.

优点是风险较低, 易做单变量 A/B. 缺点是无法单独消除 2 GiB `M_sel`, 很可能不足以达到全部 2x 硬线. 因此该路线作为 P0, 不作为提前结束条件.

### 4.2. 路线 B: Exact selected-read oracle 与 fused custom autograd

先建立 exact top-k query-chunked oracle, 再实现直接读取 `M_state + idx_remote + top_idx` 的 FP32 fused selected read. Kernel 内完成 selected state/code load, address projection, reference L2 normalize, `M@d` 和 omega reduction, 不写回 `M_sel/C_sel/Q-C/z/d_read/proposal`, backward 不保存这些大张量.

该路线显存 ROI 最高, 并可能显著减少 HBM 流量. 风险是 `M_state`, codebook 和 omega 的多对一 gradient reduction, top-k tie, normalization 边界及跨 GPU 数值轨迹. 推荐作为主线, 但必须先由 profiler 证明时间收益需求, 并使用 deterministic two-pass/segmented reduction, 禁止无序 atomic backward.

### 4.3. 路线 C: Event packing, grouped recurrence 和 state streaming

可用稳定 counting/segmented packing 替代通用 argsort, 并融合按 group 的 recurrence. 更深层可研究 blockwise state streaming.

该路线可能解决 CPU gap 和小 kernel, 但 grouped recurrence 已在 2026-05-14 做过数量级优化, 当前是否仍是热点未知. Recurrence 还包含 detached softcap scale 等易错 gradient 语义. 因此只在 P0/P2 后仍未达到 2x, 且 profiler 证明其为剩余主因时实施. State streaming 最后考虑.

推荐顺序为 `baseline -> P0 单变量 -> selected-read oracle/fusion -> 条件式 event/state kernel -> formal`. 每阶段继续以剩余最大热点决定下一项, 直到两机全部硬线满足.

## 5. 基线与 profiler 设计

### 5.1. 运行前硬门槛

在 2080ti 和 3090 宿主机分别通过 `docker exec -u lyj Flash-VQG-tun` 进入容器, 然后检查:

- `nvidia-smi`/NVML 和 `torch.cuda.is_available()`.
- GPU1 无其他计算任务, 温度, 功耗和时钟处于可比较状态.
- 两仓 branch/commit 完全一致且工作区干净.
- PyTorch, CUDA, Triton, FLA 和驱动版本.
- Canonical cache/init/batch-order hash.

若任一目标容器 NVML/CUDA 不可用, 停止 GPU 实验并报告, 不绕到宿主机运行.

### 5.2. 公平测量矩阵

Flash baseline, GDN baseline 和每个接受候选至少测:

- Core metrics/trace/logger off.
- Formal metrics on.
- Mixer-only forward/backward.
- Full-model forward/loss/backward.
- GA4 optimizer-step.
- Eval `B16/T1024` full batch.
- Warmed 1ep end-to-end.

公共 LM head/CE/logger 成本采用三种口径共同报告: 实际 Flash 端到端, 对称公共 runner, 剥离公共成本后的 mixer-core. 不用 Flash 单侧公共优化虚构 mixer 对齐.

### 5.3. Timing 与 memory 分离

- 稳态 timing 使用 CUDA Event/NVTX, warmup `>=5`, active `>=10`, fresh-process repeat `>=3`, 报 raw/p50/p90/mean/std.
- Memory run 在 optimizer lazy state 初始化后开始, fresh process 中记录 persistent baseline, peak allocated/reserved 和 `nvidia-smi` 辅助值.
- Torch profiler 的 `profile_memory=True` 运行只做归因, 不用于最终速度数字.
- Memory history/snapshot 和 tensor lifetime 表记录 producer, consumer, shape, dtype, bytes, view/copy/materialized, requires-grad 和 saved-tensor 状态.

Phase 标签至少覆盖 H2D, embeddings/backbone, phase1 projection, event pack, grouped state update, phase2 local/coarse/selected read, output projection, LM head, CE, auxiliary loss, backward, optimizer 和 metrics.

## 6. 实施阶段与门控

### 6.1. Phase 0: 基础设施与新鲜 baseline

建立本实验专用 runner, 不把旧 `read_topk=2/cb256` 随机 full-target profiler 当当前 baseline. Runner 读取真实固定 MQAR batch, 固化 resolved config, 参数量, capacity, 输入 hash 和环境信息, 输出 JSON/CSV 与 raw trace 路径.

完成两机 Flash/GDN core/formal timing, memory 和 tensor lifetime 后, 形成热点排序和首轮 Flash/GDN ratio. 只有这一步完成后才选择首个源码优化.

### 6.2. Phase 1: 低风险单变量 waterfall

依次独立评估:

1. 训练 `need_predictions=false`, 避免无用 `argmax`.
2. GPU loss 累计, optimizer boundary 单次 D2H.
3. `zero_grad(set_to_none=True)`.
4. 有效 target hidden gather 后的 linear+CE/preds.
5. 在有效 target 上比较标准 CE 与现有 FLCE.
6. Metrics 批量 D2H及仅在已请求/采样边界构建重指标.
7. 消除 warmup/schedule counter 每 forward D2H, 保持现有 train-forward count 和 resume 语义.

每项都有独立开关, fallback, correctness, 两机 A/B, commit 和 accept/reject 记录. 收益未超过噪声或非目标回退超过 2% 时拒绝.

### 6.3. Phase 2: Exact selected-read

先增加 exact top-k query-chunked PyTorch oracle. 它必须复用 reference `top_idx/omega`, 保持 FP32 和 `z / z.norm().clamp_min(eps)`. Eval/no-grad 验证不再构建全局 `M_sel`. 训练侧审计 autograd saved tensors, 不把普通 Python chunking当最终训练显存解法.

若 selected read 是主要显存项或重要时间项, 实现 fused custom autograd:

- 首版由 reference 提供 `top_idx`, 不重写 top-k.
- 直接按 `idx_remote` 从 `M_state` 读取, 消除 `M_remote` copy.
- Forward 不 materialize selected-read 大中间量.
- Backward 覆盖 Q, codebook, addr projection, M-state 和 omega gradient.
- Backward 重算复用保存的 `top_idx`, 不消费 RNG.
- 多对一 gradient 使用确定性两阶段或 segmented reduction.
- sm75/sm86 分别选择 launch config, 未支持形状/dtype明确 fallback.

### 6.4. Phase 3: 条件式 event/state 优化

若 profiler 证明 CPU gap, event pack 或 grouped recurrence仍是剩余主因:

- 使用固定 group-id 范围上的 counting, prefix sum 和 stable scatter, 保持 group 内 `ell` 递增.
- Fused recurrence严格保留 decay, pred, err, detached softcap scale, rank-1 update, optional cap 和 tail decay顺序.
- 未支持 config 走 reference fallback.

若 P0–P3 后显存仍未达标, 才评估 state streaming. Checkpoint/rematerialization可作为诊断, 不作为最终双目标成功方案.

## 7. 等价性与质量验证

### 7.1. 算子与 full-model

- `top_idx` exact.
- Forward max-abs和 relative-L2均 `<=1e-5`.
- Loss abs diff `<=1e-6`.
- Backward逐输入 relative-L2 `<=1e-4`, 并检查 max-abs `<=1e-5`, finite, 支持集合, 符号翻转和最差位置.
- 固定 CPU/CUDA/dropout RNG 做 one-step 参数与 AdamW state 对照.
- 按风险执行 32, 128, 512 optimizer-step短轨迹.

测试矩阵覆盖 production/small shape, partial block, empty event, repeated `idx_remote`, top-k tie/near-tie, `read_topk=1/16/S`, normalize eps边界, softcap未命中/边界/强命中和多个 query选择同一 state/code.

### 7.2. 正式质量回归

最终组合用 canonical cache/init/order 在两机 GPU1 跑 s124/s125 paired 1ep. 必须满足:

- Flash `1024x256 >=0.85`.
- Paired machine gap `<=4pp`.
- 对应 seed/机器 overall accuracy 相对未优化 baseline 退化 `<=1pp`.
- 无 OOM, NaN, Inf 或 Traceback.

正式完成的 run 写 ledger, 包含 started/ended UTC, wall-clock, GPU, dtype policy和状态. Smoke/debug/失败不写正式 ledger, 但进入 artifact/status/report.

## 8. 异常处理与长任务

- GPU 容器预检失败时停止实验, 保存命令和错误, 不采用旁路 runner.
- Cache 内容 hash 不一致时停止严格跨机运行, 不自行接受独立数据口径.
- Correctness 失败时保留最小复现, 关闭候选开关并回到 reference, 不放宽门槛掩盖差异.
- 性能候选只在一张卡受益时允许硬件 dispatch, 但两卡最终都必须过 2x.
- 长任务启动后监控到真实 train/eval loop 稳定推进, 记录 PID/队列, run count, log/result路径, ETA和恢复检查命令.

## 9. 交付物

实验脚本位于 `zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/`, 正式 artifacts 位于 `docs/artifacts/20260724-01-flash-vqg-gd-residual-efficiency/`.

最终至少交付:

- Flash/GDN baseline与final timing-memory ratio CSV.
- Tensor lifetime, metrics-on/off和candidate waterfall CSV.
- Equivalence/trajectory结果.
- Formal ledger, source manifest, metadata JSON和README.
- Raw profiler trace和memory snapshot的来源机器, 原路径, mirror路径和hash.
- `docs/20260724-01-flash-vqg-gd-residual-efficiency-report.md`.
- 两仓 clean status, 目标分支最终 commit和远端同步证据.

报告必须逐机列出绝对值及 Flash/GDN ratio, 说明接受/拒绝候选, 网页建议的证实或纠正, 限制条件和完整复现命令. 只有审计, 实现, 双机全部 2x 硬线, 正式质量回归, artifacts/report及远端推送全部完成后, 本实验才可标记完成.
