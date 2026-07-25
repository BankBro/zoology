# MQAR 低精度与长度泛化实验计划

## 1. 实验定义

- `experiment_id`: `20260726-01-mqar-precision-profile`.
- 目标: 在 RTX 2080 Ti 和 RTX 3090 上重新训练当前 Flash-VQG baseline 与 `h2-ek4-ev4` GDN, 系统比较 FP32, AMP-FP16 和 AMP-BF16 的训练与评估精度, 并复测标准 MQAR 与 longer-MQAR 长度泛化.
- Flash-VQG baseline: `baseline-r16-joint`.
- GDN baseline: `gdnxk-h2-ek4-ev4-usegate0`.
- 默认环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`.
- 代码分支: Zoology `flash-vqg`, Flash-VQG `20260428-gd-residual-v1-sync`.
- 结果定位: 本实验是独立 precision profile, 不覆盖现有 FP32 baseline 和历史 canonical ledger.

## 2. 实现方案

### 2.1. Flash-VQG 低精度边界

增加显式配置 `fox_gd_residual_triton_input_policy`, 取值为 `input` 或 `fp32_boundary`, 默认 `input` 以保持历史行为. 本实验固定使用 `fp32_boundary`.

`fp32_boundary` 只包围两个实际 Triton 调用点:

- grouped state-update: 在 kernel 外将实际输入转换为 FP32, 关闭 autocast 后调用 kernel.
- selected-read: 在 kernel 外将 Q, codebook, addr, state 和 omega 转换为 FP32, 关闭 autocast 后调用 kernel.

Triton kernel 始终接收 FP32 pointer, 不在 kernel 内实现低精度输入 ABI. 投影, 局部注意力和外围 reduction 继续使用 AMP 目标 dtype. 实验模式下任何 fallback 都视为硬错误. 记录 boundary 前 dtype, kernel 实际 dtype, 调用次数和 fallback 次数.

Flash 注入 schedule 计数器必须支持显式导出与恢复. `resume.pt` 保存每层 `_fox_gd_residual_train_forward_count`, 恢复后不得重新经历 warmup. 正式训练沿用当前 `0 -> 2048` train-forward microbatch 的注入 warmup, eval 固定 `final`.

### 2.2. GDN 低精度策略

外层模型采用目标 AMP dtype, FLA kernel 通过 `GDN_KERNEL_DTYPE` 显式设为 `float32`, `float16` 或 `bfloat16`. 禁止正式实验使用 `auto`. 2080 Ti 不运行 BF16, 3090 运行 BF16. 每个事件记录目标 dtype 与实际 kernel dtype.

### 2.3. AMP 与精确恢复

为 `TrainConfig` 增加向后兼容的 precision 与 resume 配置. 默认仍为 FP32 且不自动恢复. AMP-FP16 使用 `GradScaler`, AMP-BF16 不使用 scaler, master weights 与 Adam 状态保持 FP32.

`last.pt` 和 `best.pt` 保持现有 model checkpoint 格式. 另建原子覆盖的 `resume.pt`, 至少保存:

- model, optimizer, scheduler 和 GradScaler 状态.
- Python, NumPy, Torch CPU/CUDA RNG 状态.
- epoch, 下一个 train microbatch cursor, optimizer step, logger step 和已完成 validation.
- best checkpoint 跟踪状态和 epoch-end scheduler 边界状态.
- source commit, normalized config, cache, init 与数据身份.
- Flash-VQG 每层 schedule runtime state.

只在 optimizer boundary 且 gradient 已清空时保存恢复点. 正式训练在每次 validation 后写 `resume.pt`, epoch-end 的恢复点必须表示 scheduler 已完成或明确记录下一动作, 避免重复 `scheduler.step()`. 恢复时重建同一 epoch 的 sampler 顺序并跳过已完成 batch, 所有身份不一致均硬停止.

保留历史 `valid/loss` 语义, 新增 sample-weighted loss 作为补充指标. checkpoint 选择仍按 `valid/accuracy`.

## 3. 正式训练矩阵

固定训练参数:

- seeds: `123, 124, 125`.
- train batch: `64`.
- gradient accumulation: `4`.
- effective batch: `256`.
- validation batch: `16`.
- epochs: `4`.
- validations per epoch: `4`.
- early stopping: disabled.
- `TORCH_DETERMINISTIC=0`.
- TF32 disabled.
- `TRITON_F32_DEFAULT=ieee`.

矩阵总计 30 个训练 run:

| 机器 | 模型 | 训练 dtype | seeds | run 数 |
|---|---|---|---|---:|
| 2080 Ti | Flash, GDN | FP32, AMP-FP16 | 123,124,125 | 12 |
| 3090 | Flash, GDN | FP32, AMP-FP16, AMP-BF16 | 123,124,125 | 18 |

每台机器串行执行. dtype 顺序为 FP32, FP16, BF16. 每个 dtype 内按 seed 递增, 同一 seed 先 GDN 后 Flash. 每个训练完成后立即评估其 last 和 best.

## 4. 评估矩阵

每个新 checkpoint 在其 source machine 上运行该机器支持的所有 eval dtype:

- 2080 Ti: FP32, FP16.
- 3090: FP32, FP16, BF16.

标准 MQAR 共 8 个 shape, 每个 1000 examples:

`64x4`, `64x8`, `64x16`, `128x32`, `256x64`, `512x64`, `512x128`, `1024x256`.

longer-MQAR 共 5 个 shape, 每个 500 examples:

`1024x256`, `2048x512`, `4096x1024`, `8190x512`, `8190x2047`.

last 是主结果, best 是敏感性分析. 若两者 state hash 相同, 物理评估可去重但逻辑角色必须完整. matching train/eval dtype 对角线为主结论, off-diagonal 全网格用于机制分析.

保留 4 个历史 FP32 regression canary, 即每台机器各一个 Flash 与 GDN seed123 last. canary 只桥接旧 evaluator, 不计入正式统计.

### 4.1. Eval batch 容量与恢复

候选 batch 固定为 `128,64,32,16,8,4,2,1`. 每个 `machine x model x eval dtype x shape` 在全量事件上使用全新子进程做容量试验. 候选 OOM 可以审计后继续下降, 非 OOM 错误或 batch1 仍失败均硬停止. 选定 batch 后在 smoke 和 formal 中保持一致.

对选定 batch 和下一个更小的安全 batch 使用同一 checkpoint, dtype 和 samples 做不变性检查. prediction 与 accuracy 必须完全一致, per-sample loss 只允许预注册的浮点容差. 任何不一致都禁止正式启动.

评估事件每完成一个 batch 原子更新 progress, 保存 next batch cursor, sample accuracy sum/count, query correct/count, loss accumulator, peak memory, wall time 和全部身份. 事件 ID 包含 machine, model, seed, role, train/eval dtype, shape, data hash, batch 和 checkpoint state hash.

## 5. Smoke 与正式启动硬门槛

正式任务开始前, 30 个正式 descriptor 必须逐个完成实际 train, validation 和 eval smoke. 任何环节不得只做 dry-run 或静态检查.

### 5.1. 普通训练与恢复 smoke

- 每个 descriptor 完成 3 个成功 optimizer update, 即 12 个 microbatch.
- 训练 microbatch 跨 5 个训练 shape 分层抽取, 每个 shape 至少 2 batches, 最后一个 accumulation group 偏重 `256x64`.
- 第 1 个 update 后运行 validation 并保存 resume, 受控退出, supervisor 自动重启后完成 update 2 和 3.
- validation 覆盖 8 个正式 validation shape, 每个 shape 实跑 3 个 B16 batch, 共 24 batches.
- 恢复后检查 batch cursor, optimizer/logger step, RNG, scaler, scheduler 与 Flash schedule counter 连续.
- FP16 以成功 optimizer update 计数. 最多容许 2 次经过记录的 scaler calibration skip, 必须最终取得 3 次成功 update; 连续 nonfinite 或持续 skip 硬停止.

### 5.2. Flash 满注入 stress smoke

15 个 Flash descriptor 额外分别执行一次 3-update stress smoke. 仅把 smoke schedule counter 初始化为 2048, 强制 injection factor 为 1, 其余机器, seed, dtype, B64, GA4 和 cache 均与正式配置一致.

### 5.3. Eval smoke

每个训练 descriptor 的 smoke checkpoint 在所有支持 eval dtype 和全部 13 个 shape 上各执行至少 3 个选定 batch. 每个 `run x eval dtype` 的 `8190x2047` 在第一个 batch 后受控中断, 再自动恢复完成 smoke. Eval 使用正式 `final` 注入策略.

### 5.4. 全局 gate

双机全部普通 smoke, Flash stress smoke, 4 个 canary, 容量搜索和 batch invariance 都成功后, coordinator 才生成 global gate. Gate 绑定两个仓库 commit, normalized configs, Python/torch/CUDA/Triton/FLA 环境, cache/init/data/batch hashes. Gate 后任何绑定项变化都会自动失效并禁止 formal.

## 6. 自动队列与失败策略

- 每台机器使用一个容器内 tmux supervisor 和串行 GPU worker.
- worker 使用 GPU/file lock, atomic state 和 heartbeat, 启动命令幂等, 禁止重复队列.
- transient child/process/CUDA failure 最多分类重试 2 次.
- 正式 OOM, NaN/Inf, config/hash/dtype/Triton assertion 均为硬失败并停止该机器整个队列. 容量搜索中的预期 OOM 除外.
- 程序自动完成所有 train, validation, checkpoint evaluation 和汇总, 无需保持当前会话在线.

首个正式顺序固定为两机同时启动 GDN FP32 seed123. 当前会话退出前必须确认:

- 双机 GPU/NVML 与 `torch.cuda.is_available()` 通过.
- 双机 env, code, config, cache/init/data hash gate 通过.
- 两机已进入实际 formal train loop 并持续推进.
- 2080 Ti 首个正式 checkpoint 的 `8190x512` 与 `8190x2047`, 在 FP32 和 FP16 下均完成全 500 examples 且审计通过.
- 上述评估完成后, 2080 Ti formal 队列已进入下一节点并继续推进.

## 7. 产物与报告

- 脚本: `zoology/experiments/flash_vqg/scripts/20260726-01-mqar-precision-profile/`.
- generated: `zoology/experiments/flash_vqg/generated/<launch_id>/`.
- raw analysis: `zoology/analysis/flash_vqg/results/<launch_id>/`.
- artifact: `docs/artifacts/20260726-01-mqar-precision-profile/`, 按 `machines/2080ti`, `machines/3090`, `combined`, `figures` 分开.
- 报告: `docs/20260726-01-mqar-precision-profile-report.md`.
- 实验日志: 追加 `docs/EXPERIMENT_LOG.md`.

正式统计在每台机器内对 3 seeds 计算 mean 和 population std, 并报告 seed-paired delta. 两张 GPU 不合并成 `n=6`. 主图按 matching dtype 分 last 与 best 两张, 每张包含 2080 Ti 和 3090 panel. 补充材料给出完整 train x eval dtype 网格表或 heatmap. Flash 标注为 hybrid precision, GDN 标注为 native kernel precision.

## 8. 验收标准

- 30/30 正式训练完成 epoch 4.
- 所有逻辑 last/best checkpoint 在 source machine 的所有支持 eval dtype 与 13 个 shape 上完成.
- cache, init, checkpoint, data 和 code hash 一致且可追溯.
- 无未处理 OOM, NaN/Inf, Traceback 或 Flash fallback.
- ledger 包含开始/结束时间, wall time, GPU, dtype policy, batch, scaler, kernel dtype, peak memory 与状态.
- 双机轻量 evidence 镜像到主工作区并通过 sha256 校验.
- 生成 final CSV, source manifest CSV, metadata JSON, README, last/best 图和正式报告.

预计完整 wall time 为 14-18 小时, 若 best 与 last 经常不同或发生可恢复重试, 上限约 20 小时.
