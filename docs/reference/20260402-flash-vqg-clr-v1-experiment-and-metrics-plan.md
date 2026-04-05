# Flash-VQG `clr_v1` 主线实验规划与指标指南

## 1. 目标与范围

当前主线只围绕 3 件事:

- `k` 的归类与还原更精细
- `v` 写入 remote memory 更准确
- `v` 从 remote memory 读出更准确

本规划只针对 `clr_v1`, 不把 `clr_delta_v1` 作为主线.

主线设计与实现主体当前围绕实验 1 到实验 6 展开. 仓库里现有的 `e7` eval-only 能力继续保留.

这份文档后续同时承担两类职责:

- 作为实验 1 到实验 6 的主线规划文档
- 作为实验 1 到实验 7 的统一结果记录文档

因此, 当前文档口径固定为:

- 规划主体默认覆盖实验 1 到实验 6
- 结果记录章节预留实验 1 到实验 7
- 本轮只实际落实验 1 的实现, 执行与结果记录

统一原则:

- 每个实验先跑 pilot, 有明确信号再追加预算
- 最终判断以验证集任务指标为准
- `attn/*` 和 `vq/*` 指标只负责解释机理, 不负责裁决结果
- 正式训练统一写 SwanLab, 分析统一优先读取本地产物

### 1.1 统一脚本入口

- 文档主线脚本统一放在 `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/`
- 该目录内部继续按 `e1` 到 `e6` 独立子目录组织, 每个实验只负责自己的 builder, smoke 和 train wrapper
- 正式训练与 analysis 统一通过 `zoology.experiments.flash_vqg.run_flash_vqg_suite` 发起
- 统一入口固定为 `run_flash_vqg_suite.py`, 各实验 wrapper 通过 `--config-builder module_or_file:callable` 预组装本实验矩阵
- SwanLab 项目统一使用 `flash_vqg_clr_v1_mainline`
- analysis 固定使用 `--analysis local`
- 旧 `run_flash_vqg_e1.sh` 到 `run_flash_vqg_e7.sh` 只作为 legacy 参考保留, 不再作为主线入口

### 1.2 本轮交付范围

- 实验 1 已完成并晋升 `clr_v1 + top2 read` 为当前主线 baseline
- 本轮继续实现并执行实验 2, 且明确拆成 `E2-main` 与 `E2b`
- `E2-main` 是主实验, `E2b` 是补充实验
- 预算按不收紧版本执行时, `E2-main` 与 `E2b` 使用两张显卡并行训练
- 实验 3 到实验 6 继续保留为主线规划
- 实验 7 当前只保留结果记录占位, 不在本轮实现
- `20260402-clr-v1-mainline/` 目录继续作为实验 1 到实验 6 的公共根目录

## 2. 公共基线对齐

除实验变量外, 主线统一对齐 `zoology/experiments/flash_vqg/scripts/run_flash_vqg_e0.sh` 的显式和隐式基线.

### 2.1 训练侧公共基线

| 项目 | 固定值 |
| --- | --- |
| `d_model` | `128` |
| `learning_rate` | `1e-3` |
| `max_epochs` | `32` |
| `seed` | `123` |
| `data_seed` | `123` |
| `train_batch_order` | `global_shuffle` |
| `weight_decay` | `0.1` |
| `early_stopping_metric` | `valid/accuracy` |
| `early_stopping_threshold` | `0.99` |
| `slice_keys` | `[num_kv_pairs, input_seq_len, mqar_case]` |
| 默认 batch | `train_batch_size=256`, `eval_batch_size=32`, `gradient_accumulation_steps=1` |

`_build_data_config()` 的公共数据段固定为:

- train: `(64,4,100k)`, `(128,8,20k)`, `(256,16,20k)`, `(256,32,20k)`, `(256,64,20k)`
- test: `(64,4,1k)`, `(64,8,1k)`, `(64,16,1k)`, `(128,32,1k)`, `(256,64,1k)`, `(512,64,1k)`, `(512,128,1k)`, `(1024,256,1k)`

正式实验前必须先通过 smoke 选择稳定的 `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`.

### 2.2 模型侧公共基线

| 项目 | 固定值 |
| --- | --- |
| `num_layers` | `2` |
| `num_heads` | `2` |
| `key_dim` | `64` |
| `value_dim` | `64` |
| `num_codebook_vectors` | `128` |
| `block_len` | `32` |
| `local_num_blocks` | `2` |
| `if_remote_enabled` | `true` |
| 主体结构 | `Hybrid(BaseConv + FlashVQG)` |
| `BaseConv` | `kernel_size=3`, `implicit_long_conv=true` |
| `state_mixer` | `Identity` |
| `use_time_mixing` | `kv_shift` |
| `vq_score_mode` | `l2` |
| `vq_weight_mode` | `one-hot` |
| `vq_update_mode` | `ema` |
| `codebook_beta` | `0.25` |
| `if_value_silu` | `true` |
| `if_output_gate_use_rmsnorm` | `true` |
| `output_gate_activation` | `swish` |
| `fox_if_local_use_vq_k` | `false` |

## 3. 主线实验总览

### 3.1 基线晋升总规则

实验 1 到实验 6 统一采用同一条裁决规则. 每个实验都拿当前 baseline 做对照, 并只在满足以下条件时晋升为下一轮 baseline:

- `best_valid_accuracy` 明确更好
- 至少两个 hard buckets 同时上涨
- 没有明显训练不稳, 机制塌缩, 或实现异常

如果以上条件同时满足, 则当前实验的最佳变体晋升为新 baseline, 再进入下一个实验. 如果不满足, 则不晋升, 只把结果保留为机理判断, 后续实验继续沿用旧 baseline.

裁决优先级固定为:

- `valid/*` 主指标决定是否晋升
- `attn/*` 与 `vq/*` 只用于解释机理, 不单独决定晋升

### 3.2 执行顺序与主线分工

- 实验 1 已完成, 当前主线 baseline 已固定为 `clr_v1 + top2 read`
- 实验 2 拆成 `E2-main` 与 `E2b`, 其中主线判断以 `E2-main` 为准
- `E2-main` 先回答 "remote denominator 是否是 remote 接口的必要组成部分"
- `E2b` 再回答 "在固定当前 denominator-aware selector 不变时, 哪种 merge 更好"
- 预算不收紧时, 一张显卡跑 `E2-main`, 一张显卡跑 `E2b`
- 两边正式训练前都必须先各自完成 smoke, 并分别确定 `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`
- 实验 2 之后再跑实验 3, 优先判断 remote denominator 与码本学习谁是当前主瓶颈
- 再跑实验 4, 实验 5, 实验 6, 在主方向基本明确后继续做高精度检索与写入侧优化
- 一句话概括: 前 3 个实验用于找主矛盾, 后 3 个实验用于沿着主矛盾做精细化优化

### 实验 1: soft top-k read

- 变量: `fox_remote_read_topk ∈ {dense,1,2,4}`
- 当前 baseline: `dense read`
- 目标: 验证 remote read 是否太 dense, 太糊
- 简述: 把 remote 从对所有 code 的 dense 聚合改成只读 top-k 候选 code, 重点判断当前瓶颈是否主要来自 remote 读得太糊, 检索不够准. 理想现象是 `top2` 先拉升长长度和大 `num_kv_pairs` 的 hard buckets
- 候选判断:
  - 如果 `top2` 或 `top4` 明显优于 `dense`, 且 hard buckets 也上涨, 则说明 read-side blur 是关键问题, 最优 top-k 版本晋升为新 baseline
  - 如果 overall 只小涨, 但 hard buckets 基本不动, 则说明 read-side blur 不是唯一主瓶颈, 不晋升
  - 如果 top-k 比 `dense` 更差, 或出现训练不稳, 则不晋升, 后续继续沿用 `dense baseline`
- 本轮状态: 实现并执行, 当前结果支持 `top2` 作为后续优先候选

### 实验 2: 去 remote denominator

- 变量: `residual expert` 与 `shared local denominator`
- 当前 baseline: `clr_v1 + top2 read`
- 目标: 验证 remote denominator 是否必要
- 简述: 实验 2 拆成 `E2-main` 与 `E2b`. `E2-main` 测试把 denominator 从 remote 接口整体拿掉后是否还能成立, `E2b` 测试固定当前 denominator-aware selector 时哪种 merge 更好
- 候选判断:
  - 如果 `2A residual expert` 明显最好, 则说明 remote 更适合作 residual value expert, `2A` 晋升为新 baseline
  - 如果 `2B shared local denominator` 明显最好, 则说明 remote 不需要独立 denominator, 但仍需要 local denominator 锚定, `2B` 晋升为新 baseline
  - 如果 `2A` 和 `2B` 都不如原版, 则说明当前 remote 仍离不开显式 denominator, 不晋升
- 本轮状态: 已进入实现与执行, 其中 `E2-main` 为主实验, `E2b` 为补充实验

### 实验 3: learnable routingVQ / codebook

- 变量: codebook 慢更新, 例如 EMA 或小学习率
- 当前 baseline: 来自实验 2 的 baseline
- 目标: 验证问题是否来自码本不会分桶
- 简述: 把码本改成可学习, 但用 EMA, 小学习率, 或 warmup 控制更新速度, 重点判断瓶颈是否来自码本本身不会分桶, 而不是读法有问题. 理想现象是 usage 更健康, 量化误差下降, 且 hard buckets 同时上涨
- 候选判断:
  - 如果 `valid/*` 主指标上涨, hard buckets 也上涨, 且 codebook usage 更健康, 则说明主瓶颈偏 codebook 学习与分桶能力, learnable codebook 晋升为新 baseline
  - 如果 `vq/*` 指标更好看, 但 MQAR 主指标基本不动, 则说明码本更会用, 但 retrieval 仍未更准, 不晋升
  - 如果训练不稳, 或 code usage 明显塌缩, 则回退并保持旧 baseline
- 本轮状态: 只保留文档规划

### 实验 4: 两级 key quantization

- 变量: coarse + residual 两级量化
- 当前 baseline: 来自实验 3 的 baseline
- 目标: 验证单级量化是否过粗
- 简述: 把单级量化改成 `coarse code + residual code` 的两级 key quantization, 重点判断单级量化是否过粗, 导致近邻 key 在 code 空间里分不开
- 候选判断:
  - 如果 recon 更好, routing 更稳, 且 hard buckets 明显上涨, 则说明单级 quantization 过粗是主因, 两级 quantization 晋升为新 baseline
  - 如果 recon 更好, 但任务表现基本不动, 则说明问题不主要在量化级数不够, 不晋升
  - 如果成本明显上升而收益很小, 则说明当前两级 quantization 性价比不足, 不晋升
- 本轮状态: 只保留文档规划

### 实验 5: retrieval-aware VQ loss

- 变量: `q·k_hat ≈ q·k` 的 retrieval-aware 辅助损失
- 当前 baseline: 来自实验 4 的 baseline
- 目标: 验证 query 看见量化 key 后是否仍然排序错误
- 简述: 在 VQ loss 上增加 retrieval 对齐目标, 例如让 `q·k_hat` 更接近 `q·k`, 重点判断即使重建误差不大, query 面对量化 key 时是否仍然存在排序失真. 理想现象是 routing 更稳, hard buckets 提升更明显
- 候选判断:
  - 如果 routing 更稳, `top1-top2 margin` 更健康, 且 hard buckets 明显上涨, 则说明主瓶颈是 retrieval ranking distortion, retrieval-aware loss 晋升为新 baseline
  - 如果 routing 指标更好, 但任务基本不动, 则说明 read-side ranking 已不是主矛盾, 不晋升
  - 如果主训练被明显拖坏, 则说明当前 retrieval-aware loss 权重或形式不合适, 回退并保持旧 baseline
- 本轮状态: 只保留文档规划

### 实验 6: remote write hygiene

- 变量: 轻量写入 gate
- 当前 baseline: 来自实验 5 的 baseline
- 目标: 验证 remote memory 是否被低价值 token 污染
- 简述: 给 remote 写入增加轻量 gate, 只让更值得记忆的 token 强写入, 重点判断 remote 是否被低价值 token 污染. 理想现象是 remote 总能量下降或持平, 但 hard buckets 反而更好
- 候选判断:
  - 如果 `o_remote_energy_ratio` 下降或持平, 但 hard buckets 上涨, 则说明 remote 更干净了, 不是更强了, write hygiene 晋升为新 baseline
  - 如果 remote 能量明显塌缩, 且任务一起变差, 则说明 gate 太强, 把 remote 写废了, 回退并保持旧 baseline
  - 如果 easy buckets 上涨, 但 hard buckets 不动, 则说明 write pollution 不是主瓶颈, 不晋升
- 本轮状态: 只保留文档规划

## 4. 实验 1: soft top-k read

### 4.1 目标与矩阵

实验 1 固定比较 4 个 read mode:

- `dense`
- `top1`
- `top2`
- `top4`

对应配置为:

- `fox_remote_read_topk = dense`
- `fox_remote_read_topk = 1`
- `fox_remote_read_topk = 2`
- `fox_remote_read_topk = 4`

目标是验证当前主瓶颈是否来自 remote read 太 dense, 太糊, 而不是 remote branch 本身没有价值.

### 4.2 实验 1 专有覆盖项

- `backend=torch`
- `fox_state_build_backend=torch`
- `fox_remote_path_backend=torch`
- `vq_use_triton_shortcodes=false`
- `fox_remote_formula=clr_v1`
- `fox_clr_rank=4`
- `fox_clr_use_den_residual=true`
- `fox_clr_remat_mode=off`
- `fox_remote_read_topk ∈ {dense,1,2,4}`

### 4.3 `clr_v1` soft top-k 最终口径

只为 `clr_v1` 增加 soft top-k remote read. `clr_delta_v1` 继续不支持 `fox_remote_read_topk`.

top-k 的排序口径固定为截断前 dense remote logits:

$$
z_s = q \cdot c_s + p_{bias} + \log(\mathrm{den\_eff}_s)
$$

无效 code 先 mask 为 $-\infty$.

数值稳定口径保持 dense 方案:

$$
m_{shared} = \max(\max_s z_s,\ \mathrm{local\_max})
$$

top-k 只影响第二遍 remote `Num_far` 和 `Den_far` 的保留集合, 不修改 shared shift.

### 4.4 新增分析型指标

以下 3 个指标只在 `fox_phase2_metrics_mode == "full"` 时产出:

#### `attn/remote_routing_entropy`

$$
p_s = \mathrm{softmax}(z)_s
$$

$$
H = -\sum_s p_s \log(p_s + \varepsilon)
$$

说明:

- 基于截断前 dense remote logits
- 只统计至少有 1 个有效 remote code 的 row
- 不可定义时不产出 key

#### `attn/remote_top1_top2_margin`

$$
\Delta_{12} = z_{(1)} - z_{(2)}
$$

说明:

- 基于截断前 dense remote logits
- 只统计至少有 2 个有效 remote code 的 row
- 不可定义时不产出 key

#### `attn/remote_topk_den_capture_ratio`

$$
\mathrm{capture}
=
\frac{\mathrm{Den}_{remote}^{topk}}{\mathrm{Den}_{remote}^{dense}}
$$

其中两者都在同一个 `m_shared` 下计算:

$$
\mathrm{Den}_{remote}^{dense}
=
\sum_s \exp(z_s - m_{shared})
$$

$$
\mathrm{Den}_{remote}^{topk}
=
\sum_{s \in \mathrm{topk}(z)} \exp(z_s - m_{shared})
$$

说明:

- `dense` 模式下该值应接近 `1`
- 若 `top2` 已接近 `1`, 说明收益主要来自去 dense blur, 不是大量丢掉有效质量
- 不可定义时不产出 key

### 4.5 `attn/*` 指标模式

`off`:

- 不产出任何 `attn/*`

`lite`:

- `attn/nan_inf_count`
- `attn/den_min`
- `attn/o_remote_energy_ratio`
- `attn/clr_alpha_norm_mean`
- `attn/clr_den_neg_ratio`, 仅当 denominator residual 存在时

`full`:

- 在 `lite` 基础上追加 `attn/remote_win_rate`
- 在 `lite` 基础上追加 `attn/den_cache_ratio`
- 在 `lite` 基础上追加 `attn/remote_routing_entropy`
- 在 `lite` 基础上追加 `attn/remote_top1_top2_margin`
- 在 `lite` 基础上追加 `attn/remote_topk_den_capture_ratio`
- 在 `lite` 基础上追加 `attn/q_rms_mean`
- 在 `lite` 基础上追加 `attn/k_rms_mean`
- 在 `lite` 基础上追加 `attn/k_hat_rms_mean`
- 在 `lite` 基础上追加 `attn/clr_h_norm_mean`

分析侧统一按 "缺失即 NA" 处理, 不新增额外 mode tag.

### 4.6 正式脚本与公共接口

实验 1 使用以下脚本:

- `README.md`: 说明目录承载主线实验 1 到 6, 本轮先落地实验 1
- `common_env.sh`: 统一项目名, entity, analysis 方式, 路径和默认 read-mode 矩阵
- `metrics_e1_soft_topk.yaml`: 实验 1 专用 metrics white list
- `smoke_e1_batch_accum.py`: 低成本端到端 smoke
- `run_e1_smoke.sh`: smoke 包装脚本
- `run_e1_train.sh`: 正式实验入口

正式训练脚本固定参数:

- `--logger-backend swanlab`
- `--analysis local`
- `--project flash_vqg_clr_v1_mainline`
- `--backend torch`
- `--fox-remote-formula clr_v1`
- `--fox-remote-read-topk-values dense,1,2,4`

保持现有配置接口名不变:

- `fox_remote_read_topk`
- `fox_phase2_metrics_mode`

新增对外指标名:

- `attn/remote_routing_entropy`
- `attn/remote_top1_top2_margin`
- `attn/remote_topk_den_capture_ratio`

### 4.7 主要观察指标

任务指标:

- `valid/accuracy`
- `valid/loss`
- `valid/input_seq_len/accuracy-{256,512,1024}`
- `valid/num_kv_pairs/accuracy-{64,128,256}`

模型指标:

- `attn/o_remote_energy_ratio`
- `attn/clr_alpha_norm_mean`
- `attn/remote_routing_entropy`
- `attn/remote_top1_top2_margin`
- `attn/remote_topk_den_capture_ratio`
- `vq/relative_err_mean`

预期:

- `top2` 通常优于 `dense`
- `top4` 若与 `top2` 接近, 说明主要收益来自去 dense blur
- 提升应优先出现在 hard buckets, 而不是只涨 easy case

### 4.8 验证与执行摘要

实验 1 的验证与执行口径固定为:

- 先完成 `Flash-VQG` 低层测试, 覆盖 top-k guard, dense 等价性, 边界 case 和新指标公式
- 再完成 `zoology` 侧测试, 覆盖 metrics white list, suite 构造, 分析候选指标和脚本参数渲染
- 端到端 smoke 固定先跑 `clr_v1 + top4 + full metrics`, 候选 batch/GA 组合按从大到小探测, 选出第一个稳定组合后再补跑 dense sanity
- 正式运行前必须同时满足代码完成, 文档同步, 测试通过, smoke 选出稳定 batch/GA, 本地 SwanLab 产物与 `analysis local` 路径验证无误
- 正式运行使用 `run_e1_train.sh`, 训练完成后执行本地 analysis, 并检查 `run_dir`, `backup.swanlab`, `launch_analysis/`, `run_summary.csv` 以及四种 read mode 产物

## 5. 实验 2: 去 remote denominator

### 5.1 目标与拆分

实验 2 固定建立在实验 1 已晋升的 baseline 之上:

- baseline = `clr_v1 + top2 read`
- `fox_remote_read_topk=2`
- `fox_clr_rank=4`
- `fox_clr_use_den_residual=true`
- `fox_clr_remat_mode=off`
- `seed=123`, `data_seed=123`
- `block_len=32`, `local_num_blocks=2`

实验 2 明确拆成两部分:

- `E2-main`: 主实验, 测 "remote denominator 是否是 remote 接口的必要组成部分"
- `E2b`: 补充实验, 测 "固定当前 denominator-aware selector 不变时, 哪种 merge 更好"

两部分都维持同一套 10 个配置:

- `baseline`
- `2A residual expert`, `λ ∈ {0.25, 0.5, 1.0}`
- `2B shared local denominator`, `λ ∈ {0.25, 0.5, 1.0}`
- `2C gated residual expert`, `λ_max ∈ {0.25, 0.5, 1.0}`

smoke 都固定跑 4 个代表配置:

- `baseline`
- `2A λ=0.5`
- `2B λ=0.5`
- `2C λ_max=0.5`

### 5.2 E2-main: denominator-free interface

`E2-main` 的原则是: baseline 保持当前实现, 2A / 2B / 2C 则把 denominator 从 remote selector 中整体拿掉.

统一中间量:

$$
\mathrm{score}_s(q)=q \cdot c_s + p_{bias,s}
$$

$$
d_s(q)=\max(L_s+a_s^\top \alpha_s(q), \varepsilon)
$$

$$
n_s(q)=g_s+R_s \alpha_s(q)
$$

$$
\mu_s(q)=n_s(q) / d_s(q)
$$

其中 `score_only` 模式下, `d_s(q)` 只承担两件事:

- 恢复 code-level mean, 即 `\mu = n / d`
- 判定 code 是否有效, 即 `d > eps`

selector 口径:

- baseline: `TopK(score + log d, 2)`
- 2A / 2B / 2C: `TopK(score, 2)`

merge 口径:

- baseline: 继续走 `shared_den`
- 2A: `o = o_local + λ r_remote`
- 2B: `o = (Num_local + λ Num_remote) / (Den_local + eps)`
- 2C: `o_{t,h} = o^{local}_{t,h} + λ_{t,h} r^{remote}_{t,h}`

其中 2C 的 gate 固定为:

$$
λ_{t,h}=λ_{max}\cdot \sigma(w^\top q_{t,h}+b)
$$

并且:

- `w` 全 head 共享
- `w` 零初始化
- `b=-2.0`

结论口径固定为:

- 若 2A 最好, 说明 remote 更适合作 residual expert
- 若 2B 最好, 说明 remote 自己的 denominator 不是必要的, 但 merge 仍需要 local denominator 锚定
- 若 2C 最好, 说明 denominator-free residual expert + 轻量 gate 成立
- 若三者都不如 baseline, 说明当前 remote 接口仍需要 denominator-aware 设计

### 5.3 E2b: merge-only ablation

`E2b` 的原则是: selector 固定沿用当前 `clr_v1` denominator-aware routing, 只比较 merge 方式.

selector 口径对所有分支都相同:

$$
\mathrm{TopK}(\mathrm{score} + \log d, 2)
$$

因此:

- baseline 与 `E2-main baseline` 完全一致
- `E2b-2A` 只把 merge 改成 `residual_add`
- `E2b-2B` 只把 merge 改成 `shared_local_den`
- `E2b-2C` 只把 merge 改成 `residual_add + shared_query_linear gate`

`E2b` 能回答的结论固定为:

- 在保留当前 denominator-aware selector 的前提下, 哪种 merge 更好

`E2b` 不能直接回答的结论也固定写清:

- 不能直接据此推出 "remote denominator 不是必要接口组成"

### 5.4 统一公共约束

Flash-VQG 侧新增显式 selector 配置:

- `fox_clr_selector_mode ∈ {den_aware, score_only}`
- `fox_clr_merge_mode`
- `fox_clr_gate_mode`
- `fox_clr_lambda_remote`
- `fox_clr_gate_init_bias`

数学定义固定为:

- `den_aware`: `z_s(q) = score_s(q) + log d_s(q)`
- `score_only`: `z_s(q) = score_s(q)`
- `I_k(q) = TopK(z_s(q), k)`

所有 mode 都必须满足:

- remote 求和只在 `s ∈ I_k(q)` 上进行
- 当 `k >= num_codes` 时, top-k 版本严格退化为该 mode 的 dense 版本
- rows without valid codes 时, 行为保持有限且稳定

指标约束:

- `remote_routing_entropy` 与 `remote_top1_top2_margin` 基于当前生效的 selector logits 计算
- `attn/remote_topk_den_capture_ratio` 只对 `den_aware` selector 有原生意义
- `E2-main` 的 local analysis 不依赖 `attn/remote_topk_den_capture_ratio`

### 5.5 smoke, batch/GA 与双卡执行方式

目录编排固定为:

- `e2-remote-interface/config_builder.py`
- `run_e2_main_smoke.sh`
- `run_e2_main_train.sh`
- `run_e2b_smoke.sh`
- `run_e2b_train.sh`
- `run_e2_dual_train.sh`

所有训练都统一通过:

- `zoology.experiments.flash_vqg.run_flash_vqg_suite`
- `--config-builder <module_or_file>:<callable>`

smoke 口径固定为:

- `E2-main` 和 `E2b` 各自独立做 smoke
- 每个 smoke 同时确定 `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`
- smoke 结果写出 summary JSON, 并生成可直接 source 的 env 文件
- full train 前必须先完成两边 smoke

双卡执行口径固定为:

- GPU-A 跑 `E2-main`
- GPU-B 跑 `E2b`
- 推荐入口为 `run_e2_dual_train.sh`
- 两个 launch 各自产出 `backup.swanlab`, `launch_analysis/`, `run_summary.csv`

### 5.6 主要观察指标与结论解释

主指标:

- `valid/accuracy`
- `valid/loss`
- `valid/input_seq_len/accuracy-*`
- `valid/num_kv_pairs/accuracy-*`
- `valid/mqar_case/*`

分析指标:

- `attn/o_remote_energy_ratio`
- `attn/clr_alpha_norm_mean`
- `attn/remote_routing_entropy`
- `attn/remote_top1_top2_margin`
- `vq/relative_err_mean`

`E2-main` 的结论优先级高于 `E2b`.

解释顺序固定为:

- 先看 `valid/*` 是否支持某个分支晋升
- 再看 `attn/*` 与 `vq/*` 是否解释该分支为什么有效
- 若 `E2-main` 与 `E2b` 结论不同, 主线判断仍以 `E2-main` 为准

## 6. 实验 3: learnable routingVQ / codebook

### 6.1 目标与变量

- 目标: 验证问题是否来自码本不会分桶
- 变量: codebook 慢更新, 例如 EMA 或小学习率
- 本轮状态: 待补充详细方案

### 6.2 待补充项

后续在本章补充:

- 实验 3 专有覆盖项
- 更新规则与训练稳定性约束
- 指标与脚本方案
- 主要观察指标
- 与实验 1, 实验 2 的衔接关系

## 7. 实验 4: 两级 key quantization

### 7.1 目标与变量

- 目标: 验证单级量化是否过粗
- 变量: coarse + residual 两级量化
- 本轮状态: 待补充详细方案

### 7.2 待补充项

后续在本章补充:

- 实验 4 专有覆盖项
- 两级量化结构与数学口径
- 指标与脚本方案
- 主要观察指标
- 与前序实验的衔接关系

## 8. 实验 5: retrieval-aware VQ loss

### 8.1 目标与变量

- 目标: 验证 query 看见量化 key 后是否仍然排序错误
- 变量: `q·k_hat ≈ q·k` 的 retrieval-aware 辅助损失
- 本轮状态: 待补充详细方案

### 8.2 待补充项

后续在本章补充:

- 实验 5 专有覆盖项
- loss 定义与权重口径
- 指标与脚本方案
- 主要观察指标
- 与前序实验的衔接关系

## 9. 实验 6: remote write hygiene

### 9.1 目标与变量

- 目标: 验证 remote memory 是否被低价值 token 污染
- 变量: 轻量写入 gate
- 本轮状态: 待补充详细方案

### 9.2 待补充项

后续在本章补充:

- 实验 6 专有覆盖项
- write gate 设计与约束
- 指标与脚本方案
- 主要观察指标
- 与前序实验的衔接关系

## 10. 实验结果记录

### 10.1 记录约定

本章用于持续记录实验 1 到实验 7 的正式结果.

记录约定固定为:

- 每个实验使用单独小节
- 每节至少记录配置范围, 核心结果表, 关键指标联动, 当前结论
- 当前只补齐实验 1 结果
- 其余实验先保留占位, 后续按正式产物续写

### 10.2 实验 1 结果

#### 10.2.1 结果摘要

本节记录 2026-04-03 完成的实验 1 正式结果. 当前结论基于 `seed=123` 单次正式运行, 用于主线决策和后续实验排序, 不等同于多 seed 统计显著性结论.

4 个 read mode 的核心结果如下:

| read mode | best `valid/accuracy` | final `valid/accuracy` | best `valid/loss` | `1024x256` |
| --- | --- | --- | --- | --- |
| `dense` | `0.900193` | `0.900193` | `0.460637` | `0.479523` |
| `top1` | `0.680621` | `0.456676` | `2.447646` | `0.003523` |
| `top2` | `0.913348` | `0.913348` | `0.431028` | `0.608180` |
| `top4` | `0.888548` | `0.888548` | `0.527936` | `0.507922` |

当前排序为:

- `top2` 最优
- `dense` 次优
- `top4` 低于 `dense`
- `top1` 明显失败

按难度分层看:

- easy case, 即 `64x{4,8,16}`, 4 组结果几乎一致
- medium case, 即 `128x32`, `256x64`, `512x64`, `top2` 只比 `dense` 小幅回落, `top4` 回落更明显, `top1` 大幅崩塌
- hard case, 即 `512x128`, `1024x256`, `top2` 明显优于 `dense`, `top4` 只有轻微改善或基本持平, `top1` 近乎失效

#### 10.2.2 指标联动分析

##### `top2` 为什么最优

`top2` 的最终机制指标为:

- `valid/attn/remote_routing_entropy = 2.083449`
- `valid/attn/remote_top1_top2_margin = 0.541730`
- `valid/attn/remote_topk_den_capture_ratio = 0.466305`
- `valid/attn/o_remote_energy_ratio = 0.194553`
- `valid/vq/relative_err_mean = 0.047937`

综合现象:

- `top2` 的 `routing_entropy` 与 `dense` 接近, 说明它仍保留了足够的软竞争, 没有过早塌成硬路由
- `top2` 的 `margin` 只比 `dense` 略高, 说明排序更干净, 但没有走到过度自信
- `capture_ratio` 只有约 `0.47`, 但精度最好, 说明收益并不来自保留更多 remote mass, 而是来自保留更有用的 remote mass
- `o_remote_energy_ratio` 反而是四组里最低, 说明最优方案不是更依赖 remote, 而是更克制地使用 remote
- `vq/relative_err_mean` 也是四组里最低, 说明 `top2` 不只是在读路径上做了稀疏化, 还改善了 query 与量化 key 的对齐

当前解释是: `top2` 落在一个有效稀疏区间, 既减少 dense read 带来的 blur, 又保留了必要的候选竞争.

##### `top1` 为什么失败

`top1` 的最终机制指标为:

- `valid/attn/remote_routing_entropy = 1.131581`
- `valid/attn/remote_top1_top2_margin = 1.162572`
- `valid/attn/remote_topk_den_capture_ratio = 0.604929`
- `valid/attn/o_remote_energy_ratio = 0.372560`
- `valid/vq/relative_err_mean = 0.096957`

训练过程中的关键信号是:

- `top1` 在 epoch 3 达到最佳 `valid/accuracy = 0.680621`
- 到 epoch 4 立刻跌到 `0.448702`
- 随后长期停留在 `0.44` 到 `0.46` 区间

这组信号说明:

- `top1` 不是数值稳定性问题. 本轮 4 组 run 都满足 `valid/attn/nan_inf_count = 0`, `valid/attn/den_min = 1.0`
- 真正的问题是 routing 过硬. `entropy` 明显偏低, `margin` 明显偏高, 表示模型过早把 remote 选择锁死到单一 code
- easy case 几乎不受影响, 但 medium 和 hard buckets 大幅退化, 说明单 code read 对简单检索足够, 但对需要多个候选共同参与的复杂检索明显不够

当前解释是: `k=1` 过度离散化, 把本来应该由 soft mixture 解决的歧义, 变成了不可恢复的硬路由错误.

##### `top4` 为什么没有超过 `dense`

`top4` 的最终机制指标为:

- `valid/attn/remote_routing_entropy = 0.940996`
- `valid/attn/remote_top1_top2_margin = 0.853456`
- `valid/attn/remote_topk_den_capture_ratio = 0.880213`
- `valid/attn/o_remote_energy_ratio = 0.247944`
- `valid/vq/relative_err_mean = 0.055234`

这组信号说明:

- `top4` 保留了较高的 remote denominator mass, `capture_ratio` 接近 `0.88`
- 但它的 `entropy` 反而最低, `margin` 明显高于 `dense` 和 `top2`, 说明虽然保留的 code 更多, routing 仍然更尖锐
- hardest bucket `1024x256` 比 `dense` 略好, 但中高难 buckets 普遍不如 `dense`, 最终总精度反而回落

当前解释是: `top4` 没有像 `top2` 那样形成足够强的去噪约束, 又已经偏离 dense 的完整 remote 汇聚, 处于一个两边收益都不充分的中间区间.

##### 数值稳定与 CLR 指标补充

4 组 run 的共性:

- `valid/attn/nan_inf_count = 0`
- `valid/attn/den_min = 1.0`
- `valid/attn/clr_den_neg_ratio` 都稳定在约 `0.003` 左右

因此, 本轮主要现象应归因于 remote routing 行为和 remote read 稀疏度, 而不是 shared shift, denominator 或数值稳定性故障.

#### 10.2.3 当前主线结论

基于本轮实验 1, 当前主线结论可以简化为:

- `clr_v1` 的 soft top-k read 是有效方向, 当前最优点落在 `top2`
- `top1` 过于激进, 退出主线
- `top4` 没有超过 `dense`, 只保留为次级补充对照

baseline 的变化固定为:

- 实验 1 之前, 主线 baseline 是 `clr_v1 + dense read`
- 实验 1 之后, 因为 `top2` 同时提升了整体 `valid/accuracy` 和 hard buckets, 所以 `clr_v1 + top2 read` 晋升为新的主线 baseline
- 从实验 2 开始, 后续主线实验默认都先与 `clr_v1 + top2 read` 比较
- `dense` 不再是主线 baseline, 但保留为稳健强对照

后续 baseline 的替换规则不变:

- 实验 2 到实验 6 只有在新方案明确优于当前 baseline, 且至少两个 hard buckets 同时上涨时, 才允许继续晋升
- 如果后续实验不满足晋升条件, 则主线 baseline 保持为当前版本, 不回退为 `dense`

### 10.3 实验 2 结果

#### 10.3.1 E2-main 结果

- 配置范围:
  - `baseline`
  - `2a-l025`, `2a-l050`, `2a-l100`
  - `2b-l025`, `2b-l050`, `2b-l100`
  - `2c-lmax025`, `2c-lmax050`, `2c-lmax100`
- 记录模板:
  - `valid/accuracy`, `valid/loss`
  - `valid/input_seq_len/accuracy-*`
  - `valid/num_kv_pairs/accuracy-*`
  - `attn/remote_routing_entropy`
  - `attn/remote_top1_top2_margin`
  - `attn/o_remote_energy_ratio`
- 结论模板:
  - 若 2A 最优, 记录为 `residual expert` 胜出
  - 若 2B 最优, 记录为 `shared local denominator` 胜出
  - 若 2C 最优, 记录为 `gated residual expert` 胜出
  - 若 baseline 最优, 记录为当前 remote denominator-aware 接口仍需保留

#### 10.3.2 E2b 结果

- 配置范围:
  - `baseline`
  - `e2b-2a-l025`, `e2b-2a-l050`, `e2b-2a-l100`
  - `e2b-2b-l025`, `e2b-2b-l050`, `e2b-2b-l100`
  - `e2b-2c-lmax025`, `e2b-2c-lmax050`, `e2b-2c-lmax100`
- 记录模板:
  - 与 `E2-main` 相同的主指标和分析指标
- 结论模板:
  - 只解释固定 denominator-aware selector 时哪种 merge 更好
  - 不直接用于裁决 "remote denominator 是否是必要接口组成"

### 10.4 实验 3 结果

- 状态: 待补充
- 记录项: 配置范围, 核心结果表, 关键指标联动, 当前结论

### 10.5 实验 4 结果

- 状态: 待补充
- 记录项: 配置范围, 核心结果表, 关键指标联动, 当前结论

### 10.6 实验 5 结果

- 状态: 待补充
- 记录项: 配置范围, 核心结果表, 关键指标联动, 当前结论

### 10.7 实验 6 结果

- 状态: 待补充
- 记录项: 配置范围, 核心结果表, 关键指标联动, 当前结论

### 10.8 实验 7 结果

- 状态: 待补充
- 记录项: 配置范围, 核心结果表, 关键指标联动, 当前结论

## 11. 假设与默认值

- 实验 1 已完成并把 `clr_v1 + top2 read` 晋升为当前主线 baseline
- 实验 2 当前正式拆成 `E2-main` 与 `E2b`, 其中主线判断以 `E2-main` 为准
- 预算不收紧时, `E2-main` 与 `E2b` 使用双卡并行训练
- 两边 full train 前都必须先完成 smoke, 并各自确定 batch/GA
- 只在 `clr_v1` 主线内推进实验 2, 不引入 `clr_delta_v1`
- SwanLab 继续作为 logger, 但分析统一优先读取本地产物
