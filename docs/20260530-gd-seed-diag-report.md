# Flash-VQG gd_residual_v1 Seed 稳定性诊断报告

**日期**: 2026-06-03  
**环境**: 3090 `Flash-VQG-tun`  
**主指标**: `valid/mqar_case/accuracy-1024x256`  
**Artifact**: `docs/artifacts/20260530-gd-seed-diag/`

## 范围

本报告记录 Flash-VQG `gd_residual_v1` 在 MQAR hard slice `1024x256` 上的 seed/path stability 诊断. 固定训练口径为 `data_seed=123`, `b64_ga4`, fp32, `MAX_EPOCHS=4`, `validations_per_epoch=2`, early stopping disabled. 全程未使用 `TORCH_DETERMINISTIC=1`.

本轮最终目标不是只解释 codebook 初始化 RNG, 而是闭环 ordinary normal training 下 high/low basin 为什么分叉, 并明确当前稳定化候选的适用边界与下一步设计方向.

## 最终结论

1. `ScaleInitStrategy` 无局部 `torch.Generator` 是明确随机源, 但不是最终充分根因. 它会让 VQ codebook 初始化依赖全局 RNG 顺序, 影响 codebook/head 的早期相对几何.
2. 单独 `codebook_init_seed` 不是稳定解. corrected runtime probe 中, 真正注入 `codebook_init_seed=seed` 后 1ep step705 为 s124=`0.294`, s125=`0.626`; preserve-RNG/fixed-codebook variants 也不能跨 seed 稳定.
3. ordinary normal high/low 的首个可解释分叉窗口在 step 353-448, 即 first meaningful accuracy 之前的 residual write/read feedback 启动期. 最早可观测的机制差异不是 final accuracy, 而是 `rho/zeta/M` 的 head-level 转向; `lambda` 更像后续放大/锁定器.
4. 默认 phase2 read-side `fox_remote_read_topk=2` 是关键放大器: 它让早期 residual read 过窄, 容易在 routing margin 尚不可靠时锁到坏候选. `read_topk=4` 提供受控候选覆盖, 能把 s124 从旧 low path 拉回 high path, 同时不伤 s125.
5. `read_topk=4` 是 `cb256-r4` 和 `cb256-r8` 上有效的 read-side 稳定候选, 但跨 codebook/rank 复核后不能作为全局默认. `cb64-r16` 会伤 s124 high path, `cb128-r8` 的 s125 rerun 从 `0.972` 掉到 `0.609`. 因此最终方案定位应转向 schedule/gate/margin-aware read-side 控制.

## 关键证据

| 对照 | 1024x256 结果 | spread | 口径 |
|---|---:|---:|---|
| old ordinary readk2 r1 | s124=0.772465, s125=0.956375 | 0.183910 | 旧 normal 分叉 |
| old ordinary readk2 r2 | s124=0.891, s125=0.953 | 0.062 | 仍有路径摆动 |
| pseudo-det readk2 | s124=0.951145, s125=0.951586 | 0.000441 | 特殊数值路径, 非 ordinary baseline |
| runtime readk4 4ep r1 | s124=0.918258, s125=0.942277 | 0.024020 | runtime probe |
| runtime readk4 4ep r2 | s124=0.973395, s125=0.948074 | 0.025320 | runtime rerun |
| formal readk4 | s124-r1=0.943, s124-r2=0.958, s125-r1=0.944 | 0.015 | ordinary `run_train.sh` |

## 2026-06-03 跨 codebook/rank 复核

新增复核覆盖 `cb64-r16`, `cb128-r8`, `cb256-r8` 的 readk2/readk4 × s124/s125, 并对 readk4 最弱或异常 seed 做 rerun. 结果如下:

| 配置 | readk2 main | readk2 spread | readk4 main | readk4 rerun | 结论 |
|---|---|---:|---|---|---|
| `cb64-r16` | s124=`0.959`, s125=`0.915` | `0.044` | s124=`0.831`, s125=`0.965` | s124 r1/r2=`0.831/0.849`, spread=`0.018` | readk4 伤 s124 high path, 不可默认 |
| `cb128-r8` | s124=`0.956`, s125=`0.956` | `0.000` | s124=`0.973`, s125=`0.972` | s125 r1/r2=`0.972/0.609`, spread=`0.363` | readk4 主结果不复现, 需 gate/schedule |
| `cb256-r8` | s124=`0.988`, s125=`0.804` | `0.184` | s124=`0.982`, s125=`0.988` | s124 r1/r2=`0.982/0.982`, s125 r1/r2=`0.988/0.992` | readk4 稳定救回弱 seed |

这个复核改变了方案定位: 固定 `read_topk=4` 不是跨配置稳定默认值. 它对 `cb256-r4/cb256-r8` 的分叉有强修复作用, 但在 `cb64-r16` 表现为 high-path damage, 在 `cb128-r8` 表现为 rerun instability. 因此更可靠的工程方向是 early read_topk schedule, margin-aware residual gate, 或按 capacity/rank 条件启用 readk4.

早期曲线也支持这个边界判断: `cb64-r16/readk4/s124` 在 v2/v4 已是 `0.667/0.772`, rerun 为 `0.697/0.794`, 明显低于 `readk2/s124` 的 `0.882/0.938`, 因此 high-path damage 不是 final 才出现; `cb128-r8/readk4/s125-r2` 从 v2/v4=`0.271/0.480` 开始偏离 main high path; `cb256-r8/readk4/s125` 则在 v2/v4=`0.954/0.985` 附近直接救回 readk2 s125 的弱早期曲线 `0.396/0.700`. 完整 validation curve, overall accuracy 和 loss 见 `gd-seed-diag-cross-config-final.csv`.

readk4 反例边界也很重要:

| 干预 | 结果 | 说明 |
|---|---:|---|
| read_topk=8, s124 1ep | 0.079113 | 候选不是越宽越好 |
| read_topk=4 + lambda_init=0.15, s124 1ep | 0.428566 | 更强 early injection 不是充分解 |
| cbseed124 + preserve RNG + readk4, s125 1ep | 0.527 | 好 codebook seed 不能替代联合几何稳定化 |

## 根因链条

最终链条应写成:

`codebook/address/projection 联合初始化几何` -> `early routing/write rho 边界` -> `zeta/M state 早期差异` -> `phase2 read_topk/lambda 注入放大` -> `high/low basin selection` -> `final 1024x256 分叉`.

旧说法“VQ codebook 初始化没有 Generator 导致两个 VQ head 竞争”只解释了第一层随机源. 它说明为什么系统敏感, 但不能解释为什么 corrected `codebook_init_seed` 仍不稳定. 后续 probe 证明, 单独固定 codebook 甚至可能改变 addr_proj 后续 RNG stream, 破坏 codebook 和 address/projection 的联合几何.

在 normal s124/s125 1ep head-level probe 中, step 352 两者 accuracy 都接近 0. 到 step 353-448:

- s124: `rho_diff(h0-h1)=-0.028`, `zeta_h0/h1=0.407`, `mmax_h0/h1=0.700`, `lambda_h0/h1=0.470`.
- s125: `rho_diff(h0-h1)=+0.011`, `zeta_h0/h1=1.799`, `mmax_h0/h1=1.340`, `lambda_h0/h1=1.626`.

这说明分叉不是 final 阶段才发生, 而是在 residual feedback 刚启动时就形成了 head-level 方向差异. `lambda` 在之后进一步放大, 但更早的 `rho/zeta/M` 转向更接近触发条件.

## pseudo-det vs normal

pseudo-det s124 能进入 high basin, 不是因为初始权重本身不同. 已有审计显示同 seed normal 与 pseudo-det 的初始 codebook/addr_proj/beta/lambda 一致. 差异也不能只写成“数值路径不同”: 更具体地说, s124 的联合初始化几何靠近 basin 边界, 底层算子路径引入的微小 roundoff/order 差异先在 step 130-203 变成可测训练轨迹差异, 随后在 step 353-448 的 residual feedback 启动窗口改变写入状态强度和 M state 规模.

聚合指标也支持这一点: pseudo s124 high 在 step 352 还没有 meaningful accuracy, 但 `write_strength=0.0107`, `mmax=1.80`, `lambda=0.0030`; normal s124 r1/r2 同点只有 `write_strength=0.0046/0.0055`, `mmax=0.689/0.692`, `lambda=0.0010/0.0009`. 到 step 448, pseudo high 的 `write_strength=0.0255`, `mmax=2.54`, 已经明显强于 normal s124 r1/r2 的 `write_strength=0.0136/0.0105`, `mmax=0.928/0.851`. 这说明 pseudo-det 进入 high basin 的直接机制是 early residual write/M state 先被抬起来, 让后续 read/lambda 注入拥有可用状态; normal s124 则在同一窗口没有建立足够强且有效的状态, 后续 top2 read 更容易锁入 low/medium path. 因此 pseudo-det 的意义是证明 high basin 存在和分叉窗口可观测, 但它不是可部署修复手段.

## 稳定方案定位

跨配置复核后, 固定 `fox_remote_read_topk=4` 不应再写成最终全局稳定方案. 更准确的定位是:

- 在 `cb256-r4` 和 `cb256-r8` 上, readk4 能缓解 top2 过窄导致的 basin lock-in, 并显著降低 seed spread.
- 在 `cb64-r16` 上, readk4 会伤害原本 high 的 s124 path, 且 s124 rerun 复现弱路径.
- 在 `cb128-r8` 上, readk4 主矩阵看似改善, 但 s125 rerun final=`0.609`, 说明固定 readk4 自身也可能落入不稳 basin.

机制解释仍成立: top2 过窄, top8 过宽, top4 在部分配置提供受控候选覆盖. 但工程解法不能停在固定 top4, 而应转向 schedule/gate: 早期避免 top2 锁死, 同时用 margin/置信度/质量门控防止错误 proposal 注入.

## 已验证结论

- `codebook_init_seed` 去掉 codebook 初始化全局 RNG 依赖, 但不是最终稳定解.
- ordinary normal 分叉发生在 first meaningful accuracy 之前, 与 early routing/write/read/M/lambda feedback 相关.
- static tau, static write_topk, topk_softmax, 正交 addr_proj, 固定 codebook seed, preserve-RNG, 高 lambda 都不是稳定充分解.
- `read_topk=4` 在 `cb256-r4/cb256-r8` 中有效, 但在 `cb64-r16/cb128-r8` 中暴露副作用或复现风险; 因此不是全局默认解.

## 合理推断

- 好的 codebook 初始化不是静态 codebook 分布好看, 而是 codebook 与 `k_proj/v_proj/addr_proj/beta/lambda` 的联合几何让 early routing/write/read feedback 不容易落入坏 basin.
- `read_topk=4` 暴露并部分缓解的是放大器/注入侧的稳定性问题, 不是替代所有初始化设计问题. 它能在 cb256-like 配置中避免早期 residual read 过早锁死, 但跨配置仍需要 schedule/gate 约束.

## 未解问题和风险

- 还没有把 read-side schedule/gate 写成正式源码或配置规范. 源码改动需另行确认.
- 固定 readk4 的适用范围已被限制: `cb256-r4/cb256-r8` 有效, `cb64-r16` 和 `cb128-r8` 不应默认启用.
- `cb128-r8` 的 s125-r2 异常说明, 即使主矩阵两 seed 都 high, rerun 仍可能暴露 basin instability. 后续需要 early/head-level probe 聚焦 step 353-448 的 rho/zeta/M/lambda/read 轨迹, 而不是继续大扫参.
- 最优工程方案更可能是 schedule/gate/margin-aware 控制, 或 codebook/address/projection 联合初始化约束, 不是固定 `read_topk=4`.

## 设计启发

1. 初始化要看联合几何, 不是只看 codebook 的 pairwise cos, RMS 或 effective rank.
2. 对带状态的 residual memory, early read/write 候选选择是稳定性核心控制面.
3. 候选覆盖要受控: top2 过窄会锁死, top8 过宽会注入坏 proposal.
4. `lambda` 是放大器, 不是万能增益. 过早/过强注入可能把模型推到另一个不稳 basin.
5. 未来稳定化应优先考虑 margin-aware gate, early schedule, 或 codebook/address/projection 联合初始化, 而不是 `TORCH_DETERMINISTIC=1`.

## Artifact 文件

- `docs/artifacts/20260530-gd-seed-diag/final.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-key-metrics.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-spread-summary.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-source-manifest.csv`
- `docs/artifacts/20260530-gd-seed-diag/metadata.json`
- `docs/artifacts/20260530-gd-seed-diag/README.md`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-final.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-spread-summary.csv`
- `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-source-manifest.csv`
