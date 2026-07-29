# MQAR seed124 remat 数值分叉因果诊断实验计划

## 1. 实验登记

- Experiment ID: `20260729-03-mqar-seed124-remat-causal-diagnosis`.
- 状态: `planned`.
- 执行机器: `mclab-3090` 的 `Flash-VQG-tun` 容器.
- Zoology base: `flash-vqg@3a84f718c1d52f5f86929941ffe64ee985110f92`.
- Flash-VQG base: `20260729-172300-fix-selected-read-addr-grad-determinism@d7dbb1282d20ad860634ee4b8f0a74b948fe6c61`.
- 环境: `/home/lyj/miniconda3/envs/flash-vqg-fla042`.
- 预算: 最多 6 RTX 3090 GPU 小时, replay capsule 不超过 2 GiB.

## 2. 目标与假设

目标是定位 seed124 的 A0 `off` 与 A1 `post_phase1` remat 首个数值分叉, 精确到 optimizer window, microbatch, tensor 和算子, 并用单变量干预完成因果验证.

历史正式 telemetry 已显示, seed123 和 seed125 的 2816 个训练 loss 全部一致, seed124 的首个可见 loss 差异出现在第 10 个 optimizer window, 即 `log_step=9`. 初始 16-step probe 因而足以覆盖已知边界. 该观测尚不能判断参数是在第 9 步更新后已经分叉, 还是第 10 个 window 的 forward/backward 首次分叉.

核心假设依次为:

1. A0 和 A1 各自 fresh-process 可重复, 差异来自 remat 路径.
2. 如果某个 variant 自身不可重复, 则优先定位该 variant 的内在非确定性.
3. 首个分叉更可能位于 RNG 生命周期、GD forward/recompute 或共享梯度归约, 而不是最终 AdamW 更新.

## 3. 固定口径与实验矩阵

全部 probe 使用正式 seed124 配置: RTX 3090, AMP BF16, FP32 master weights 和 optimizer state, `baseline-r16-joint`, B64, validation B16, GA4, 4 epochs, data seed123, canonical cache/init, `validations_per_epoch=4`.

| 阶段 | Variant | Fresh processes | 最大 optimizer steps | 记录粒度 |
|---|---|---:|---:|---|
| 初始定位 | A0, A1 | 各 1 | 16 | step/microbatch 聚合 hash |
| 自重复 | A0, A1 | 各 1 | 首个分叉点 + 1 | 同 variant 重复性 |
| 详细定位 | A0, A1 | 各 1 | 首个分叉点 + 1 | 首个窗口逐 tensor/module hash |
| 因果干预 | 原路径与单变量候选 | 按需 | 分叉点 + 128 | 正对照与干预对照 |

初始 16 步未复现时, 依次扩展到 32, 128, 704 和 2816 步. 每次只增加观察长度, 不改变训练配置.

## 4. 实现与执行

实验代码只在本实验目录增加 `DiagnosticTrainer`. 正式 `zoology.train.train()` 仍负责模型、数据、logger、optimizer、scheduler、validation 和 checkpoint 生命周期, 诊断进程仅临时绑定 Trainer 子类.

第一层记录:

- 每个 microbatch 的 input/target hash、shape、loss bit hash及 forward 前后 RNG hash.
- 每个 optimizer step 的参数、累计梯度和 optimizer state 的 pre/post hash.
- zero-grad 后状态、validation 前后状态和运行时计数.

第二层只对首个分叉窗口记录逐 tensor hash, 并在 embedding、layer、sequence mixer、state mixer、final norm 和 logits 边界记录输入输出. 如果差异缩小到 Flash GD 内部但普通 module hook 不足, 才在 Flash 实验分支增加默认关闭、只对指定 step/microbatch 生效的 debug tap.

候选干预按首个分叉位置选择:

- RNG: 显式 checkpoint RNG 保存/恢复.
- selected-read: Triton/custom backward 与确定性 Torch reference.
- grouped recurrence: Triton 与 Torch reference.
- top-k/event pack: 固定稳定排序.
- optimizer: 参数顺序、AdamW state 与 reference optimizer.
- 其他 CUDA op: strict deterministic mode 或 reference 实现.

每次干预只改变一个变量, 并同时保留原路径正对照.

## 5. 裁决与产物

根因成立必须同时满足:

1. 原始配置可重复地在同一阶段分叉.
2. 定位到具体函数、算子和首个不同 tensor.
3. 单变量干预消除首个差异.
4. 干预路径精确通过原分叉点后至少 128 步并跨过最近 validation 或 epoch 边界.
5. replay capsule 或小形状测试验证数值一致性.

超过预算仍未闭环时, 终态记为 `inconclusive_budget_exhausted`, 只报告已排除假设和剩余候选, 不修改生产实现.

原始输出保留在 3090 的实验 `outputs/`. 收尾生成实验 report 和最小 artifact, 包含 first-divergence summary、repeatability classification、intervention matrix、环境和 source manifest. 本轮不进入正式 MQAR training ledger, 不重跑三 seed 完整质量回归.
