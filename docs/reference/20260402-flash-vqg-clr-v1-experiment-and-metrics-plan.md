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
- 实验 2 已完成, 当前不支持 denominator-free 方案晋升, `den_aware selector + residual_add` 只保留为后续候选
- 实验 3 已完成 `baseline + dense-t050/t100/t200` 的单 seed 正式训练, 并对 `dense-t050` 补了 `seed=124/125` 稳定性 run
- 当前结果已完成 `tau` 局部精调, matched validation, `top-k write` probe 与 `dense read` probe, **`dense-t025 + top2 read + den_aware selector + shared_den` 已正式成为实验 3 默认工作点**
- 实验 4 到实验 6 继续保留为主线规划
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
- 本轮状态: 已完成并把 `top2` 晋升为当前主线 baseline

### 实验 2: 去 remote denominator

- 变量: `residual expert` 与 `shared local denominator`
- 当前 baseline: `clr_v1 + top2 read`
- 目标: 验证 remote denominator 是否必要
- 简述: 实验 2 拆成 `E2-main` 与 `E2b`. `E2-main` 测试把 denominator 从 remote 接口整体拿掉后是否还能成立, `E2b` 测试固定当前 denominator-aware selector 时哪种 merge 更好
- 候选判断:
  - 如果 `2A residual expert` 明显最好, 则说明 remote 更适合作 residual value expert, `2A` 晋升为新 baseline
  - 如果 `2B shared local denominator` 明显最好, 则说明 remote 不需要独立 denominator, 但仍需要 local denominator 锚定, `2B` 晋升为新 baseline
  - 如果 `2A` 和 `2B` 都不如原版, 则说明当前 remote 仍离不开显式 denominator, 不晋升
- 本轮状态: 已完成结果记录, 当前不支持 denominator-free 方案晋升, `2A residual_add` 只保留为候选

### 实验 3: dense write routingVQ / codebook

- 变量: write-side 从硬指派 `Δ` 改成 dense 分配 `Π`
- 当前 baseline: `clr_v1 + top2 read + den_aware selector + shared_den merge + legacy one-hot write`
- 目标: 验证当前瓶颈是否来自 write-side 分桶过硬, 导致 remote cache 写入过糙
- 简述: 实验 3 本轮不改 read-side, 只改 write-side. 具体做法是把 remote memory 的写入从 hard one-hot `Δ` 推广为 dense assignment `Π`, 并保持 FoX forgetting, local 精确分支, remote cache 递推, 以及 `shared_den` 合并语义不变. 理想现象是 write entropy, usage 与重建误差更健康, 同时 hard buckets 与总 `valid/accuracy` 上涨
- 候选判断:
  - 如果 dense write 的 `valid/*` 主指标上涨, hard buckets 同时上涨, 且 `vq/write_entropy_mean` 与 `vq/c_entropy` 没有塌缩, 则说明 write-side 离散化过硬是主瓶颈, dense write 晋升为下一轮 baseline
  - 如果 `vq/*` 指标更好看, 但 MQAR 主指标基本不动, 则说明码本使用更平滑, 但 retrieval 仍未更准, 不晋升
  - 如果训练不稳, `nan_inf_count` 升高, `den_min` 恶化, 或 usage 明显塌缩, 则回退并保持旧 baseline
- 本轮状态: 已完成单 seed 正式矩阵, `tau` 局部精调, `dense-t025` 相对 `dense-t050` 的 matched validation, `top-k write` probe 与 `dense read` probe; **当前最优点已收口为 `dense-t025 + top2 read + den_aware selector + shared_den`**

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

## 6. 实验 3: dense write routingVQ / codebook

### 6.1 目标与范围

- 当前 baseline 固定为 `clr_v1 + top2 read + den_aware selector + shared_den merge + legacy one-hot write`
- 实验 3 本轮只实现 `dense write + top2 read`
- read-side 继续固定 `fox_remote_read_topk=2`, 不回退到 dense read
- local 分支不改
- FoX forgetting 不改
- `shared_den` 归一化语义不改
- 当前不引入 codebook 慢更新, 统一使用 `vq_update_mode=grad`
- 当前只做 dense 版, 但接口必须预留后续 `topk write`

一句话概括:

- 实验 3 不是改 remote read, 而是把 write-side 的 `Δ -> Π`

### 6.2 Dense 数学口径

本轮严格采用 dense write 公式:

- write-side 打分:

$$
B^{(n)} = \frac{K^{(n)} E^\top}{\sqrt{d} \cdot \tau_w}
$$

- dense 分配:

$$
\Pi^{(n)} = \mathrm{rowSoftmax}(B^{(n)})
$$

- dense block summary:

$$
\beta_n^{U,dense} = (\Pi^{(n)})^\top (s^{(n)} \odot V^{(n)})
$$

$$
\beta_n^{Z,dense} = (\Pi^{(n)})^\top s^{(n)}
$$

- 块间递推保持 FoX 原样:

$$
U_{e_n}^{dense} = \alpha_n U_{e_{n-1}}^{dense} + \beta_n^{U,dense}
$$

$$
Z_{e_n}^{dense} = \alpha_n Z_{e_{n-1}}^{dense} + \beta_n^{Z,dense}
$$

- remote read 仍保持当前 `clr_v1 + top2 read` 结构, 本轮不做 dense read
- 最终输出继续使用 shared denominator:

$$
O = \frac{Num_{loc} + Num_{far}}{Den_{loc} + Den_{far}}
$$

对 `clr_v1` 额外约束:

- dense write 不只要写 `U/Z`, 还要把 CLR rank/basis 相关状态同步推广为 weighted 写入
- 这样 readout 和 merge 逻辑才能保持主线不变

### 6.3 配置映射与实现约束

实验 3 的配置映射固定为:

- baseline:
  - `vq_score_mode=l2`
  - `vq_weight_mode=one-hot`
  - `vq_update_mode=ema`
- dense write:
  - `vq_score_mode=codebook_dot`
  - `vq_weight_mode=dense_softmax`
  - `vq_update_mode=grad`
  - `vq_softmax_tau ∈ {0.5, 1.0, 2.0}`

为了预留 top-k write, 代码层需要一次性透传:

- `vq_score_mode`
- `vq_weight_mode`
- `vq_update_mode`
- `vq_softmax_tau`
- `vq_topk`

但本轮默认正式主矩阵只启用 `dense_softmax`. 截至 2026-04-09, `topk_softmax` 已进入独立 probe launch, 但尚未进入默认正式主矩阵.

当前实现约束固定为:

- `fox_remote_formula=clr_v1`
- `fox_remote_path_backend=torch`
- `fox_state_build_backend=torch`
- `fox_clr_remat_mode=off`

也就是说, 实验 3 dense write 当前只在 `torch + clr_v1 + eager materialize` 路径下成立.

### 6.4 实验 3 专有覆盖项

- `backend=torch`
- `fox_state_build_backend=torch`
- `fox_remote_path_backend=torch`
- `vq_use_triton_shortcodes=false`
- `fox_remote_formula=clr_v1`
- `fox_remote_read_topk=2`
- `fox_clr_rank=4`
- `fox_clr_use_den_residual=true`
- `fox_clr_remat_mode=off`
- `fox_clr_selector_mode=den_aware`
- `fox_clr_merge_mode=shared_den`
- `block_len=32`
- `local_num_blocks=2`
- `num_codebook_vectors=128`

### 6.5 正式实验矩阵

本轮先做单 seed `123`, 正式训练矩阵固定为 4 组:

- `baseline`
- `dense-t050`
- `dense-t100`
- `dense-t200`

对应配置:

- `baseline`: `l2 + one-hot + ema`
- `dense-t050`: `codebook_dot + dense_softmax + grad`, `vq_softmax_tau=0.5`
- `dense-t100`: `codebook_dot + dense_softmax + grad`, `vq_softmax_tau=1.0`
- `dense-t200`: `codebook_dot + dense_softmax + grad`, `vq_softmax_tau=2.0`

smoke 只跑 2 组:

- `baseline`
- `dense-t100`

原因很直接:

- smoke 只负责确认实现和 batch/GA 稳定性
- 正式区分不同温度值的比较留到 full train

### 6.6 指标与脚本方案

实验 3 除保留主线指标外, 额外要求记录 dense write 相关指标.

主指标:

- `valid/accuracy`
- `valid/loss`
- `valid/input_seq_len/*`
- `valid/num_kv_pairs/*`
- `valid/mqar_case/*`

稳定性与 remote 指标:

- `attn/nan_inf_count`
- `attn/den_min`
- `attn/o_remote_energy_ratio`
- `attn/clr_alpha_norm_mean`
- `attn/clr_den_neg_ratio`
- `attn/remote_routing_entropy`
- `attn/remote_top1_top2_margin`

VQ / write-side 指标:

- `vq/relative_err_mean`
- `vq/k_norm_mean`
- `vq/k_hat_norm_mean`
- `vq/c_rms_mean`
- `vq/c_entropy`
- `vq/c_usage_mean`
- `vq/c_usage_max`
- `vq/write_entropy_mean`
- `vq/write_top1_mass_mean`

上述指标都要求有 `valid/*` 镜像.

脚本目录固定为:

- `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e3-dense-routing/`

其中:

- `config_builder.py`: 组装 `baseline + dense` 矩阵
- `smoke_e3_batch_accum.py`: 选择 `train_batch_size / eval_batch_size / gradient_accumulation_steps`
- `run_e3_smoke.sh`: 先跑 smoke
- `run_e3_train.sh`: 再跑正式训练
- `metrics.yaml`: SwanLab 与 analysis 统一指标白名单

### 6.7 执行顺序与验收

执行顺序固定为:

1. 先完成代码修改与文档更新
2. 跑低层单测和脚本层回归
3. 跑 `E3 smoke`, 确定 `train_batch_size / eval_batch_size / gradient_accumulation_steps`
4. 确认 `dense-t100` smoke 无 OOM, 无 NaN/Inf
5. 在空闲 GPU 上启动 `baseline + dense-t050/t100/t200` 的单 seed 正式训练
6. 正式训练统一记录到 SwanLab

晋升标准:

- `best valid/accuracy` 明确高于 baseline
- 至少两个 hard buckets 同时上涨
- `vq/write_entropy_mean` 与 `vq/write_top1_mass_mean` 没有塌缩
- `attn/nan_inf_count` 不升高, `attn/den_min` 不恶化

若只看到 usage/entropy 更好看, 但主任务指标没有变好, 则不晋升.

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

单 seed 排序为:

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

`E2-main` 的目标是判断: 当 remote selector 不再使用 denominator-aware routing, 而改成 `score_only` 时, 当前 remote interface 是否仍然成立.

本轮 `E2-main` 覆盖了 10 个配置:

- `baseline`
- `2a-l025`, `2a-l050`, `2a-l100`
- `2b-l025`, `2b-l050`, `2b-l100`
- `2c-lmax025`, `2c-lmax050`, `2c-lmax100`

其中:

- `baseline` = `den_aware selector + shared_den merge`
- `2A` = `score_only selector + residual_add`
- `2B` = `score_only selector + shared_local_den`
- `2C` = `score_only selector + residual_add + shared_query_linear gate`

关键结果表如下:

| config | best valid_acc | final valid_acc | final delta vs baseline | final `o_remote_energy_ratio` | final routing entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.9490 | 0.9324 | 0.0000 | 0.2694 | 2.3222 |
| `2a-l025` | 0.9205 | 0.9111 | -0.0213 | 0.0917 | 2.4888 |
| `2a-l050` | 0.9261 | 0.9249 | -0.0075 | 0.2970 | 2.8600 |
| `2a-l100` | 0.9219 | 0.9149 | -0.0175 | 0.3892 | 3.0017 |
| `2b-l025` | 0.4132 | 0.4049 | -0.5275 | 0.0127 | 1.7919 |
| `2b-l050` | 0.4207 | 0.4055 | -0.5269 | 0.0982 | 1.5927 |
| `2b-l100` | 0.4405 | 0.3961 | -0.5363 | 0.5946 | 0.6251 |
| `2c-lmax025` | 0.9231 | 0.9210 | -0.0114 | 0.0465 | 3.4800 |
| `2c-lmax050` | 0.9349 | 0.9326 | +0.0002 | 0.2030 | 3.4511 |
| `2c-lmax100` | 0.9331 | 0.9300 | -0.0024 | 0.3568 | 2.5944 |

当前结果的主结论是:

- `E2-main` 中没有任何一个 denominator-free 配置形成 "稳定超过 baseline" 的证据
- 因此, 本轮主线判断仍然支持: **当前 remote interface 仍需要 denominator-aware 设计**
- 换句话说, 在 `clr_v1 + top2 read` 这条主线上, `score_only selector` 目前还不足以替代 `score + log d` 的 selector 口径

从本轮结果看, 三类 denominator-free 变体的相对结论可以概括为:

- `2A` 说明: remote 在拿掉 denominator-aware selector 之后, 仍然保留了一定 usable signal, 但整体还不足以稳定打过 baseline
- `2B` 说明: 单纯改成 `shared_local_den` 并没有把 denominator-free interface 救回来, 这一路径当前没有显示出主线价值
- `2C` 说明: gate 能在一定程度上约束 residual remote 的注入强度, 但目前更像 "缓和波动的辅助项", 而不是能够改写主结论的主胜方案

因此, `E2-main` 的结论不是 "2A / 2C 完全无效", 而是:

- **当前 remote path 仍离不开 denominator-aware selector**
- 如果后续还要继续优化 remote interface, 应优先在保留 `den_aware selector` 的前提下继续做 merge / routing / codebook 优化, 而不是直接把 remote denominator 从主接口中拿掉

#### 10.3.2 E2b 结果

`E2b` 的目标是判断: **在固定当前 denominator-aware selector 不变时, 哪种 merge 更有潜力.**

本轮 `E2b` 也覆盖了 10 个配置:

- `baseline`
- `e2b-2a-l025`, `e2b-2a-l050`, `e2b-2a-l100`
- `e2b-2b-l025`, `e2b-2b-l050`, `e2b-2b-l100`
- `e2b-2c-lmax025`, `e2b-2c-lmax050`, `e2b-2c-lmax100`

其解释口径与 `E2-main` 不同:

- `E2-main` 回答的是 "remote denominator 是否仍是必要接口组成"
- `E2b` 回答的是 "当 selector 继续保持 denominator-aware 时, 哪种 merge 更好"

关键结果表如下:

| config | best valid_acc | final valid_acc | final delta vs baseline | final `o_remote_energy_ratio` | final routing entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.9414 | 0.9388 | 0.0000 | 0.2682 | 2.6089 |
| `e2b-2a-l025` | 0.5969 | 0.5209 | -0.4179 | 0.0501 | 2.5238 |
| `e2b-2a-l050` | 0.9288 | 0.9266 | -0.0122 | 0.1748 | 2.7580 |
| `e2b-2a-l100` | 0.9588 | 0.9551 | +0.0163 | 0.4088 | 1.8823 |
| `e2b-2b-l025` | 0.4155 | 0.4051 | -0.5337 | 0.0002 | 1.7319 |
| `e2b-2b-l050` | 0.4220 | 0.4043 | -0.5345 | 0.1112 | 1.0236 |
| `e2b-2b-l100` | 0.4303 | 0.4097 | -0.5291 | 0.1056 | 0.9850 |
| `e2b-2c-lmax025` | 0.9384 | 0.9361 | -0.0027 | 0.0182 | 3.0867 |
| `e2b-2c-lmax050` | 0.9177 | 0.5895 | -0.3493 | 0.0549 | 2.7714 |
| `e2b-2c-lmax100` | 0.9298 | 0.9205 | -0.0183 | 0.2508 | 2.0518 |

当前 `E2b` 的主要结论是:

- 在保留 denominator-aware selector 的前提下, **`2A residual_add` 是本轮最有潜力的 merge 方向**
- 其中, `e2b-2a-l100` 给出了本轮 `E2b` 中最强的单次结果, 说明:
  - `den_aware selector + residual_add`
  - 是一个值得继续追踪的组合
  - remote 更可能适合作为一个 **residual value expert**, 而不是重新组织一套独立的 denominator 归一化接口

与此同时, `E2b` 也给出了两个重要限制:

- `2B` 在 `E2-main` 和 `E2b` 两条线上都没有显示出竞争力, 说明 `shared_local_den` 当前不是值得继续投入主预算的方向
- `2C` 虽然体现出一定稳定性价值, 但当前更像是在 `2A` 基础上加了一个轻量调节器; 它还没有形成 "比 `2A` 更强" 的清晰证据

因此, `E2b` 的结论可以总结为:

- **merge 侧最值得继续追的不是 `2B`, 而是 `2A residual_add`**
- `2C` 可保留为稳定性备选
- `2B` 当前可以降级处理

#### 10.3.3 `E2b` 的对称补跑与稳定性判断

为了把实验 2 的结论做稳, 本轮在已有 `e2b-2a-l100-s124` 和 `e2b-2a-l100-s125` probe 的基础上, 又补了 3 条 matched-seed run:

- `baseline-s124`
- `baseline-s125`
- `e2b-2a-l100-s123-rerun`

这一步的目标不再只是看 `e2b-2a-l100` 有没有单次高点, 而是同时回答 3 个问题:

- `E2b baseline` 自己是否稳定
- `e2b-2a-l100` 的 `seed=123` 强结果能否自复现
- `l100` 和 baseline 的差异, 究竟来自结构性增益, 还是来自 baseline 自身的高方差

先看 baseline 的对称补跑结果:

| run | best valid_acc | final valid_acc | final `512x128` | final `1024x256` | final `o_remote_energy_ratio` | final routing entropy | final top1-top2 margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` (`seed=123`) | 0.9414 | 0.9388 | 0.8957 | 0.7086 | 0.2682 | 2.6089 | 0.4767 |
| `baseline-s124` | 0.9124 | 0.9123 | 0.8243 | 0.6334 | 0.2930 | 1.9598 | 0.6092 |
| `baseline-s125` | 0.5725 | 0.5713 | 0.1179 | 0.0310 | 0.3744 | 0.5908 | 0.8841 |

这个结果直接说明:

- `E2b baseline` 不是稳定控制组
- 3 个 seed 的 `final valid_acc` 极差达到 `0.3676`
- `1024x256` hardest bucket 的极差达到 `0.6776`
- `baseline-s125` 并不是 "中后期掉点" 的假峰值, 而是从头到尾都收敛到明显更差的区域

也就是说, 到这一步为止, 实验 2 的关键不确定性已经从 "某个候选点是不是偶然跑高" 扩展成了 "baseline 本身存在显著 seed 敏感性".

再看 `e2b-2a-l100` 的复现情况:

| run | best valid_acc | final valid_acc | final `512x128` | final `1024x256` | final `o_remote_energy_ratio` | final routing entropy | final top1-top2 margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2b-2a-l100` (`seed=123`) | 0.9588 | 0.9551 | 0.9414 | 0.8047 | 0.4088 | 1.8823 | 0.9814 |
| `e2b-2a-l100-s123-rerun` | 0.9386 | 0.9376 | 0.9055 | 0.7917 | 0.3413 | 2.5359 | 0.6030 |
| `e2b-2a-l100-s124` | 0.9078 | 0.9073 | 0.8461 | 0.6549 | 0.3519 | 2.5274 | 0.6549 |
| `e2b-2a-l100-s125` | 0.9288 | 0.9284 | 0.8832 | 0.8089 | 0.4333 | 2.0580 | 0.7185 |

这里有两个需要同时成立的事实:

- 一方面, `seed=123` 的最强结果没有被 rerun 复现
  - `e2b-2a-l100-s123-rerun` 相比原 `seed=123`, `final valid_acc` 下降了 `0.0175`
  - `1024x256` 也下降了 `0.0130`
- 另一方面, `l100` 的跨 seed 波动显著小于 baseline
  - `l100` 三个 seed 的 `final valid_acc` 极差只有 `0.0303`
  - `1024x256` 的极差是 `0.1540`

因此, 这一步不能简单总结成 "`l100` 只是偶然高点". 更准确的判断是:

- `l100` 还没有证明自己是 **稳定优于 baseline** 的新主胜配置
- 但它显示出比 baseline 更强的 **抗 seed 崩溃能力**, 尤其在 hardest bucket 上更稳

把 matched-seed 对照写开后, 结论会更清楚:

| seed | baseline final valid_acc | `l100` final valid_acc | final delta | baseline `1024x256` | `l100` `1024x256` | `1024x256` delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `123` | 0.9388 | 0.9376 | -0.0012 | 0.7086 | 0.7917 | +0.0831 |
| `124` | 0.9123 | 0.9073 | -0.0050 | 0.6334 | 0.6549 | +0.0215 |
| `125` | 0.5713 | 0.9284 | +0.3571 | 0.0310 | 0.8089 | +0.7779 |

这个 matched-seed 比较说明:

- 对健康 seed 来说, `l100` 并没有稳定赢下 overall `valid_acc`
- 但在 `1024x256` hardest bucket 上, `l100` 3 个 seed 全部不差于 matched baseline, 而且 `seed=123/125` 的优势很明显
- `seed=125` 的巨大差距更多说明 baseline 本身会崩, 而不能直接当成 "`l100` 已经稳定更强" 的证据

因此, 当前更合理的稳定性裁决是:

- `baseline`: **unstable**
- `e2b-2a-l100`: **not yet reproducible as a new winner, but more robust than baseline**
- `e2b-2a-l050`: 仍然保持 `no clear gain`

#### 10.3.4 `E2b baseline` 的 decoupled 诊断结果

为了进一步拆开 baseline 的不稳定性来源, 本轮又补了 4 条 decoupled baseline run:

- `baseline-s123-d124`
- `baseline-s123-d125`
- `baseline-s124-d123`
- `baseline-s125-d123`

这里的目标不再是比较新候选点, 而是专门回答:

- baseline 是更敏感于 `seed`, 还是更敏感于 `data_seed`
- baseline 的坏收敛是 "从头就学坏", 还是 "先冲高后回落"
- baseline 的问题是否主要来自数值稳定性

先给出 anchor + decoupled 的关键结果表:

| run | best valid_acc | final valid_acc | best-last | final `1024x256` | final routing entropy | final top1-top2 margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s123-d123` | 0.9414 | 0.9388 | 0.0026 | 0.7086 | 2.6089 | 0.4767 |
| `s123-d124` | 0.7548 | 0.7463 | 0.0085 | 0.2067 | 1.3415 | 1.7410 |
| `s123-d125` | 0.7215 | 0.6185 | 0.1030 | 0.0196 | 1.4985 | 0.3962 |
| `s124-d123` | 0.9021 | 0.6470 | 0.2552 | 0.0550 | 1.9215 | 0.4097 |
| `s125-d123` | 0.8215 | 0.7735 | 0.0481 | 0.2923 | 1.3685 | 1.4556 |

把两条轴分开后, 结论会更清楚:

- `data axis`: `s123-d123 -> s123-d124 -> s123-d125`
  - `final valid_acc` 极差是 `0.3203`
  - `1024x256` 极差是 `0.6890`
- `seed axis`: `s123-d123 -> s124-d123 -> s125-d123`
  - `final valid_acc` 极差是 `0.2918`
  - `1024x256` 极差是 `0.6536`

这说明:

- `data_seed` 对 terminal quality 和 hardest bucket 的破坏略更重
- 但 `seed` 轴本身也足够单独把 baseline 弄崩
- 因此 baseline 的不稳不能被简化成 "只是模型初始化问题" 或 "只是数据实例问题"

更关键的是, decoupled 结果暴露出很强的交互项:

- `s124-d124` 本来是健康 run, `final valid_acc = 0.9123`
- 但 `s123-d124` 和 `s124-d123` 都明显更差, 说明不能把 `124` 简单归因成单侧坏值
- `s125-d125` 也比 `s125-d123` 和 `s123-d125` 更差, 说明 `125/125` 这组组合还存在额外的负交互放大

坏收敛的形态也并不相同:

- `s123-d124` 更像 "从一开始就偏弱, 后期相对平稳"
- `s123-d125` 更像 "很早冲到局部高点, 然后一路掉回去"
- `s124-d123` 是最典型的 late collapse, `best-last = 0.2552`
- `s125-d123` 能学起来, 但整体明显偏弱

同时, 这 4 条 decoupled run 都满足:

- `valid/attn/nan_inf_count = 0`
- `valid/attn/den_min = 1.0`

因此, decoupled 诊断进一步支持:

- baseline 的问题主要来自训练动力学 / routing 收敛
- 不是数值爆炸或 denominator 直接失稳

这一步之后, baseline 的稳定性裁决可以进一步收紧为:

- `baseline` 的不稳同时来自 `model seed`, `data_seed`, 以及二者交互
- 如果硬要排优先级, `data_seed` 的破坏略强, 但不足以把问题归因到单一来源

#### 10.3.5 实验 2 的总体结论

把 `E2-main`, `E2b`, baseline 对称补跑, 以及 `e2b-2a-l100` 的 probe / rerun 结果合在一起, 本轮实验 2 的总体结论可以整理为:

1. **主线仍应保持 baseline**

- `E2-main` 没有给出 denominator-free interface 稳定优于 baseline 的证据
- 因而当前 `clr_v1` 主线仍应保持:
  - `den_aware selector`
  - `shared_den` baseline interface
  - `top2 read`

2. **`den_aware selector + residual_add` 仍然是实验 2 中最值得继续追的方向**

- `E2b` 说明在保留 denominator-aware selector 时, `2A residual_add` 最有潜力
- 这支持把 remote 进一步理解为一个 residual remote expert, 而不是强行改写成独立归一化接口

3. **当前实验 2 的首要问题已经变成 baseline 稳定性**

- 对称补跑和 decoupled 诊断都表明 `E2b baseline` 本身存在显著稳定性问题
- `seed` 和 `data_seed` 两边都能单独触发坏收敛, 而且还存在明显交互放大
- 因而接下来不能再把单条 baseline 曲线当成足够可靠的裁决器
- 在 baseline 稳定性没有解释清楚之前, 后续实验的结构比较都容易被训练方差污染

4. **`2B` 可以基本排除**

- `2B` 在 `E2-main` 和 `E2b` 两条线上都没有形成主线竞争力
- 后续若继续保留, 也只需要作为轻量对照, 不应再占主实验预算

5. **`2C` 当前更适合作为稳定性备选, 而不是主攻方向**

- 它有一定 "抑制注入过强 / 缓和波动" 的价值
- 但目前还没有形成 "比 `2A` 更强" 的清晰证据

6. **`e2b-2a-l100` 值得继续追, 但暂时不应晋升主线**

- 它是本轮最值得继续保留的 robustness 候选点
- 但在完成更系统的稳定性诊断前, 还不能把它视为已被验证的新 baseline
- 特别是 `seed=123` 的原最强结果没有被 rerun 复现, 所以当前还不能把它定义为 "已证明稳定更强"

#### 10.3.6 对后续实验排序的影响

实验 2 结束后, 后续实验排序应按下面的逻辑推进:

- 第一优先级: 先稳定化 `E2b baseline`, 重点拆解 `model seed`, `data_seed`, 以及二者交互
- 第二优先级: 在 baseline 稳定化方案更清楚后, 再围绕 `E2b-2A(l100)` 做更系统的多 seed / 邻域 λ 复现
- 第三优先级: 在 baseline 稳定性和 `l100` 的真实收益边界都更清楚之后, 再进入实验 3

也就是说, 实验 2 的更新结果仍然不支持 "直接跳到实验 3", 也不支持 "直接把 `e2b-2a-l100` 升成新 baseline".

更合理的动作是:

- **先把 baseline 的 `model seed / data_seed / interaction` 不稳定性解释清楚**
- **再判断 `2A` 到底是在提升性能, 还是只是在提升鲁棒性**
- **最后再进入实验 3**

后续实际仍执行了实验 3 的单 seed 正式矩阵, 并对 `dense-t050` 追加了 3-seed 稳定性补跑, 结果见 10.4. 因而本节应理解为实验 2 收官时点的排序意见, 而不是对后续工作的禁止约束.

### 10.4 实验 3 结果

#### 10.4.1 结果摘要

本节记录 2026-04-06 到 2026-04-08 完成的实验 3 结果. 当前主结论由两部分组成:

- 2026-04-06 完成的 `baseline + dense-t050/t100/t200` 单 seed 正式矩阵
- 2026-04-07 到 2026-04-08 对 `dense-t050` 追加的 `seed=124/125` 稳定性补跑

本轮 `E3 smoke` 选出的训练配置为:

- `train_batch_size=64`
- `eval_batch_size=16`
- `gradient_accumulation_steps=4`
- `effective_train_batch_size=256`

4 个单 seed 配置的核心结果如下:

| config | final `valid/accuracy` | final `valid/loss` | `acc@512` | `acc@1024` | note |
| --- | --- | --- | --- | --- | --- |
| `baseline` | `0.930982` | `0.378482` | `0.915094` | `0.673875` | `legacy one-hot write` |
| `dense-t050` | `0.966792` | `0.141224` | `0.985051` | `0.776309` | `单 seed 最优` |
| `dense-t100` | `0.943368` | `0.255391` | `0.961426` | `0.641938` | `overall 仍高于 baseline, 但 hardest bucket 已回落` |
| `dense-t200` | `0.892124` | `0.525412` | `0.894504` | `0.380340` | `明显退化; 本地 backup checksum 异常, 最终指标取 remote SwanLab` |

单 seed 排序为:

- `dense-t050` 明确最优
- `dense-t100` 仍优于 `baseline`, 但已经明显低于 `dense-t050`
- `dense-t200` 明显失败

这个排序说明:

- dense write 确实是有效方向
- 但有效区间不是 "越 dense 越好"
- 当前 sweet spot 落在 `vq_softmax_tau=0.5`, 继续增大到 `1.0` 和 `2.0` 都会回落

#### 10.4.2 指标联动分析

先看与 write-side 最相关的几组指标:

| config | `valid/attn/o_remote_energy_ratio` | `valid/attn/remote_routing_entropy` | `valid/vq/relative_err_mean` | `valid/vq/write_entropy_mean` | `valid/vq/write_top1_mass_mean` |
| --- | --- | --- | --- | --- | --- |
| `baseline` | `0.132093` | `2.348460` | `0.041568` | `-` | `-` |
| `dense-t050` | `0.271016` | `3.404022` | `0.020689` | `3.790525` | `0.143738` |
| `dense-t100` | `0.277255` | `3.788672` | `0.023802` | `4.588185` | `0.036676` |
| `dense-t200` | `0.266110` | `3.742404` | `0.036605` | `4.731038` | `0.019364` |

这里有 3 个清晰信号.

1. `dense-t050` 同时做对了两件事:

- remote 参与度明显高于 `baseline`
- `vq/relative_err_mean` 明显降低
- 同时任务精度和长序列 bucket 一起上涨

这说明 `dense-t050` 不是单纯 "让 remote 更强", 而是让 write-side 表示更可用.

2. `dense-t100` 和 `dense-t200` 继续把 `write_entropy` 推高, 把 `top1_mass` 压低, 但任务指标反而回落.

- 这说明更平滑的 dense write 并不自动等于更好的 retrieval
- 当写入过于分散时, hardest bucket 会最先受伤

3. hardest bucket 对过强 dense write 最敏感.

- `dense-t050` 的 `acc@1024 = 0.776309`, 明显高于 `baseline = 0.673875`
- `dense-t100` 的 `acc@1024 = 0.641938`, 已经低于 `baseline`
- `dense-t200` 的 `acc@1024 = 0.380340`, 可以视为明显失效

因此, 当前更合理的机理解释是:

- 实验 3 的收益来自适度缓和 hard one-hot write 带来的分桶过硬
- 但一旦 write 分布过平, remote cache 会先失去 hardest case 所需的判别性
- 所以实验 3 的关键不是 "把 write 尽量做 dense", 而是找到一个适度 dense 的工作点

#### 10.4.3 `dense-t050` 的 3-seed 稳定性

为判断 `dense-t050` 的收益是否只是 `seed=123` 的偶然高点, 本轮又追加了 2 个 seed. 其中, 最初的 `seed=125` 运行因 CLI 会话退出被中断, 最终统计采用从头 rerun 的结果, 不纳入半截 run.

3 个 seed 的最终结果如下:

| seed | run_id | final `valid/accuracy` | final `valid/loss` | `acc@512` | `acc@1024` |
| --- | --- | --- | --- | --- | --- |
| `123` | `dense-t050` | `0.966792` | `0.141224` | `0.985051` | `0.776309` |
| `124` | `dense-t050-s124` | `0.966789` | `0.140598` | `0.984707` | `0.777801` |
| `125` | `dense-t050-s125-rerun` | `0.962067` | `0.164392` | `0.974266` | `0.764488` |

汇总后:

- `valid/accuracy = 0.965216 ± 0.002727`
- `valid/loss = 0.148738 ± 0.013561`
- `acc@512 = 0.981341 ± 0.006130`
- `acc@1024 = 0.772866 ± 0.007293`

稳定性判断可以写得更具体一些:

- `seed=123` 和 `seed=124` 基本重合, 说明 `dense-t050` 至少存在一条很稳的高性能收敛轨道
- `seed=125` 明显偏弱, 但没有翻车
- 最弱的 `seed=125` 仍然比 `baseline` 高 `+3.11` 个百分点, 比 `dense-t100` 高 `+1.87` 个百分点

如果继续看 telemetry, `seed=125` 的 `valid/attn/o_remote_energy_ratio` 降到 `0.238132`, 同时 `valid/attn/remote_routing_entropy` 升到 `3.951421`. 这更像 remote 使用变得更分散, 而不是 dense write 完全失效.

因此, `dense-t050` 不是 "完全无 seed 波动", 但它的 seed 波动远小于它相对 `baseline` 和 `dense-t100` 的收益.

#### 10.4.4 `tau` 局部精调决策分析

本小节只基于同一 `seed=123, data_seed=123` 的 `tau` 邻域局部扫描做工作点裁决. 相关数据见 10.4.6 中第 3 张表. 其目标不是直接晋升新 baseline, 而是决定实验 3 下一轮验证预算应压在哪个 `tau` 点上.

先看排序:

| tau | final `valid/accuracy` | final `valid/loss` | `acc@512` | `acc@1024` | `valid/vq/relative_err_mean` |
| --- | --- | --- | --- | --- | --- |
| `0.25` | `0.981208` | `0.081622` | `0.991031` | `0.874887` | `0.015469` |
| `0.50` | `0.966792` | `0.141224` | `0.985051` | `0.776309` | `0.020689` |
| `0.625` | `0.962646` | `0.165927` | `0.974984` | `0.767289` | `0.022640` |
| `0.375` | `0.959515` | `0.184253` | `0.970082` | `0.752281` | `0.021957` |
| `0.75` | `0.946263` | `0.239818` | `0.964105` | `0.657563` | `0.021967` |

可以看到 3 个明确信号.

1. `tau=0.25` 不是边缘改善, 而是显著优于当前 anchor `tau=0.5`.

- `valid/accuracy` 提高 `+0.014416`
- `valid/loss` 下降 `-0.059602`
- `acc@512` 提高 `+0.005980`
- `acc@1024` 提高 `+0.098578`
- `valid/vq/relative_err_mean` 下降 `-0.005220`

这说明收益不是只体现在平均准确率上, hardest bucket 和 VQ 重建质量也一起改善.

2. `tau > 0.5` 的右侧区间已经可以收住.

- `tau=0.625` 相对 `0.5` 已经整体回落
- `tau=0.75` 进一步明显回落

因此, 当前数据不支持继续往更软的 write 分布上加预算.

3. 左侧局部地形并不平滑, `tau=0.375` 是一个明显异常点.

- 按直觉, `0.375` 应位于 `0.25` 和 `0.5` 之间
- 但它同时低于 `0.25` 和 `0.5`

这说明目前不能简单归纳成 "`tau` 越小越好". 更稳的说法是: `0.25` 是当前扫描中最强候选, 但左侧局部区域存在非单调性, 因而下一步应该做验证而不是继续盲扫更密网格.

从 telemetry 看, `tau=0.25` 的组合也更一致:

- `write_entropy_mean` 从 `3.790525` 降到 `2.969296`
- `write_top1_mass_mean` 从 `0.143738` 升到 `0.287130`
- `o_remote_energy_ratio` 从 `0.271016` 升到 `0.312489`
- `remote_routing_entropy` 从 `3.404022` 降到 `2.905342`

这更像是更尖锐, 更有判别性的 write routing 带来了更好的 remote 读出, 而不是简单依赖更分散的 dense mixing.

因此, 本轮 `tau` 扫描的裁决应当写成:

- 停止继续扩 `tau` 网格
- 将 `tau=0.25` 晋升为实验 3 的下一轮主候选
- 下一步先补 `tau=0.25` 的稳定性验证, 而不是直接进入新的 write/read 交互矩阵

更具体的后续顺序建议是:

1. `dense-t025`, `seed=124`, `data_seed=123`
2. `dense-t025`, `seed=123`, `data_seed=124`
3. 如预算允许, 再补 `dense-t025`, `seed=123`, `data_seed=125`

只有当 `tau=0.25` 在 seed 轴和危险 `data_seed` 轴上都站稳后, 才应正式替代 `dense-t050` 成为实验 3 的默认工作点.

#### 10.4.5 当前结论

基于实验 3 当前完整数据, 本节结论可以归纳为 3 点.

1. `dense write` 已经被证明是有效主方向, 但最优区间不是 "越 dense 越好" 或 "越软越好".

- `dense-t100` 和 `dense-t200` 已可排除
- 真正值得保留的工作区间在 `tau=0.25` 到 `tau=0.5` 左侧邻域
- 从 telemetry 看, 更优点更像是来自更尖锐, 更有判别性的 write routing 和更强的 remote 参与, 而不是单纯依赖更低的 `relative_err_mean`

因此, `tau` 局部扫描的最终裁决不是继续扩网格, 而是停止 `tau` 搜索, 将 `tau=0.25` 作为最终收口候选, 再用 matched validation 完成默认工作点裁决.

2. `dense-t025` 已经收口, 应正式晋升为实验 3 默认工作点.

- `dense-t025` 在 `reference`, `seed=124`, `seed=125`, `data_seed=124`, `data_seed=125` 这 5 个 matched 口径里, 全部优于对应的 `dense-t050`
- 在固定 `data_seed=123` 的 3-seed 口径上, `dense-t025` 平均 `valid/accuracy = 0.978818`, 高于 `dense-t050` 的 `0.965216`
- 同一 3-seed 口径下, `dense-t025` 的 `valid/loss` 更低, `acc@512` 更高, `acc@1024` 平均高出 `+0.091566`
- 在危险 `data_seed` 轴上, `dense-t025` 对 `dense-t050` 的 `acc@1024` 平均仍高出 `+0.050734`

这里也需要把 `dense-t050` 的角色写清楚: 它不是当前最优点, 但仍是证据最完整的保守回退点. 因而后续文档和实验口径都应把 `dense-t050` 视为 fallback / control, 而不是继续作为实验 3 的默认主线.

因此, 当前工程口径应明确写成:

- **`dense-t025 + top2 read + den_aware selector + shared_den` 正式晋升为实验 3 默认工作点**
- **`dense-t050` 降级为保守回退点和对照点, 不再占主验证预算**

3. `write/read` 交互与 `dense -> top-k write` 的当前证据, 都继续支持 `dense-t025 + top2 read` 作为实验 3 默认主线.

- 基于实验 3 当前已完成的 `tau` 局部精调, `dense-t025` 相对 `dense-t050` 的 matched validation, 以及 `top-k write` probe 数据, 目前可以得到较明确的结构性结论: 实验 3 的最优工作点不是单独追求更 dense 或更 sparse, 而是 **`dense write + top2 read`** 这一配套组合
- `dense-t025` 在 write 侧表现最佳, 说明写入阶段需要保留一定的软分配能力, 以兼顾更低的重建误差, 更健康的 codebook 使用, 以及更强的长序列表现
- 相比之下, `topk2`, `topk4`, `topk8`, `topk16` 虽然逐步改善 slot 纯度, 但会过早截断写入信息, 导致重建误差上升, codebook 使用收缩, 最终整体指标和 `1024` 长度表现均落后于 `dense-t025`
- 其中 `topk16` 是当前 top-k 家族最强点, 但仍低于 `dense-t025-s123-d123`: `valid/accuracy -0.010779`, `valid/loss +0.045061`, `acc@512 -0.009723`, `acc@1024 -0.062738`
- read 侧的现有证据也继续支持 **`top2 read` 优于 dense read**: 历史实验中 `one-hot + top2 read` 已优于 `one-hot + dense read`, 而当前 `dense-t025 + dense read` probe 最终结果仍明显落后于 `dense-t025 + top2 read`: `valid/accuracy -0.015795`, `valid/loss +0.076486`, `acc@512 -0.003609`, `acc@1024 -0.109379`, 表明对 MQAR 这类更依赖精确远程检索的任务而言, remote read 更适合做稀疏选择而不是全量混合

因此, 当前最合理的机制判断是:

- **write 端应采用尖锐但不硬截断的 dense write**
- **read 端应采用稀疏的 top-k read**
- **`dense-t025 + top2 read + den_aware selector + shared_den` 应继续作为实验 3 的默认主线配置**
- **`topk2` 可以排除**
- **`topk16` 可保留为 sparse-write 参考点, 但仍不足以替代 dense write**
- **dense-read 分支当前不具竞争力, 不再继续扩成 2x2 matched validation 矩阵**

#### 10.4.6 实验数据记录表

本小节只作为实验 3 的数据台账入口. 原始数据以 `CSV / history.csv / summary.json / manifest.json` 为准. Markdown 表格只承担人工回填, 复核和后续决策入口作用.

1. 主矩阵记录表. 记录实验 3 正式 4-run 矩阵.

| run_id | tau / mode | seed | data_seed | launch_id | status | manifest | history.csv | summary.json | final `valid/accuracy` | final `valid/loss` | `acc@512` | `acc@1024` | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | `one-hot` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-2026-04-06-20-01-22` | `completed` | `generated/.../manifest.json` | `results/.../baseline/data/history.csv` | `results/.../baseline/data/summary.json` | `0.930982` | `0.378482` | `0.915094` | `0.673875` | `legacy one-hot write` |
| `dense-t050` | `tau=0.5` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-2026-04-06-20-01-22` | `completed` | `generated/.../manifest.json` | `results/.../dense-t050/data/history.csv` | `results/.../dense-t050/data/summary.json` | `0.966792` | `0.141224` | `0.985051` | `0.776309` | `single-seed winner` |
| `dense-t100` | `tau=1.0` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-2026-04-06-20-01-22` | `completed` | `generated/.../manifest.json` | `results/.../dense-t100/data/history.csv` | `results/.../dense-t100/data/summary.json` | `0.943368` | `0.255391` | `0.961426` | `0.641938` | `overall > baseline, hard bucket fallback` |
| `dense-t200` | `tau=2.0` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-2026-04-06-20-01-22` | `completed` | `generated/.../manifest.json` | `results/.../dense-t200/data/history.csv` | `results/.../dense-t200/data/summary.json` | `0.892124` | `0.525412` | `0.894504` | `0.380340` | `remote SwanLab backfill used for final metrics` |

2. `dense-t050` 稳定性记录表. 记录 seed 轴和危险 `data_seed` 轴.

| run_id | seed | data_seed | launch_id | status | final `valid/accuracy` | final `valid/loss` | `acc@512` | `acc@1024` | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense-t050` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-2026-04-06-20-01-22` | `completed` | `0.966792` | `0.141224` | `0.985051` | `0.776309` | `anchor` |
| `dense-t050-s124` | `124` | `123` | `flash-vqg-20260402-clr-v1-e3-dense-t050-seeds124-125-2026-04-07-10-24-16` | `completed` | `0.966789` | `0.140598` | `0.984707` | `0.777801` | `seed axis` |
| `dense-t050-s125-rerun` | `125` | `123` | `flash-vqg-20260402-clr-v1-e3-dense-t050-s125-rerun-2026-04-07-16-00-32` | `completed` | `0.962067` | `0.164392` | `0.974266` | `0.764488` | `rerun result; interrupted half-run excluded` |
| `dense-t050-s123-d124` | `123` | `124` | `flash-vqg-20260402-clr-v1-e3-dense-t050-dseed124-125-rerun-2026-04-08-05-51-07` | `completed` | `0.970286` | `0.126707` | `0.979867` | `0.817445` | `danger data axis` |
| `dense-t050-s123-d125` | `123` | `125` | `flash-vqg-20260402-clr-v1-e3-dense-t050-dseed124-125-rerun-2026-04-08-05-51-07` | `completed` | `0.969170` | `0.131283` | `0.978938` | `0.811410` | `danger data axis` |

3. `tau` 局部精调记录表. 记录 `dense-t050` 邻域工作点扫描.

| run_id | tau | seed | data_seed | launch_id | status | final `valid/accuracy` | final `valid/loss` | `acc@512` | `acc@1024` | `valid/vq/relative_err_mean` | `valid/vq/c_entropy` | `valid/vq/write_entropy_mean` | `valid/vq/write_top1_mass_mean` | `valid/attn/o_remote_energy_ratio` | `valid/attn/remote_routing_entropy` | decision_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense-t025-s123-d123` | `0.25` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12` | `completed` | `0.981208` | `0.081622` | `0.991031` | `0.874887` | `0.015469` | `3.568453` | `2.969296` | `0.287130` | `0.312489` | `2.905342` | `final promoted default after matched validation closed` |
| `dense-t0375-s123-d123` | `0.375` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-tau-local-t0375-2026-04-08-11-47-12` | `completed` | `0.959515` | `0.184253` | `0.970082` | `0.752281` | `0.021957` | `3.475924` | `3.276300` | `0.321505` | `0.321408` | `3.543396` | `non-monotonic dip; lower than 0.25 and 0.5; no extra budget` |
| `dense-t050` | `0.5` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-2026-04-06-20-01-22` | `completed` | `0.966792` | `0.141224` | `0.985051` | `0.776309` | `0.020689` | `4.075660` | `3.790525` | `0.143738` | `0.271016` | `3.404022` | `current conservative default; most fully validated point` |
| `dense-t0625-s123-d123` | `0.625` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-tau-local-t0625-2026-04-08-14-44-02` | `completed` | `0.962646` | `0.165927` | `0.974984` | `0.767289` | `0.022640` | `4.213327` | `4.080534` | `0.188378` | `0.379031` | `2.325209` | `right-side fallback; stop expanding softer tau` |
| `dense-t075-s123-d123` | `0.75` | `123` | `123` | `flash-vqg-20260402-clr-v1-e3-tau-local-t075-2026-04-08-15-00-38` | `completed` | `0.946263` | `0.239818` | `0.964105` | `0.657563` | `0.021967` | `4.375891` | `4.272873` | `0.154602` | `0.355248` | `3.892298` | `clear fallback; exclude from next budget` |

4. `dense-t025 vs dense-t050` 对应验证口径主对照表. 用于直接判断 `dense-t025` 能否稳定替代 `dense-t050`.

| axis | `dense-t025` run_id | `dense-t050` run_id | seed | data_seed | `t025 valid/accuracy` | `t050 valid/accuracy` | delta | `t025 valid/loss` | `t050 valid/loss` | delta | `t025 acc@512` | `t050 acc@512` | delta | `t025 acc@1024` | `t050 acc@1024` | delta | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reference` | `dense-t025-s123-d123` | `dense-t050` | `123` | `123` | `0.981208` | `0.966792` | `+0.014416` | `0.081622` | `0.141224` | `-0.059602` | `0.991031` | `0.985051` | `+0.005980` | `0.874887` | `0.776309` | `+0.098578` | `t025 anchor > t050 anchor` |
| `seed axis` | `dense-t025-s124-d123` | `dense-t050-s124` | `124` | `123` | `0.980097` | `0.966789` | `+0.013308` | `0.090468` | `0.140598` | `-0.050130` | `0.992430` | `0.984707` | `+0.007723` | `0.860320` | `0.777801` | `+0.082519` | `t025 > t050; seed-axis pass` |
| `seed axis` | `dense-t025-s125-d123` | `dense-t050-s125-rerun` | `125` | `123` | `0.975150` | `0.962067` | `+0.013083` | `0.107810` | `0.164392` | `-0.056582` | `0.979109` | `0.974266` | `+0.004844` | `0.858090` | `0.764488` | `+0.093602` | `t025 > t050; seed-axis pass` |
| `danger data axis` | `dense-t025-s123-d124` | `dense-t050-s123-d124` | `123` | `124` | `0.977402` | `0.970286` | `+0.007116` | `0.098177` | `0.126707` | `-0.028530` | `0.980609` | `0.979867` | `+0.000742` | `0.874203` | `0.817445` | `+0.056758` | `t025 > t050; danger-data pass` |
| `danger data axis` | `dense-t025-s123-d125` | `dense-t050-s123-d125` | `123` | `125` | `0.975364` | `0.969170` | `+0.006194` | `0.106890` | `0.131283` | `-0.024393` | `0.980098` | `0.978938` | `+0.001160` | `0.856121` | `0.811410` | `+0.044711` | `t025 > t050; danger-data pass` |

5. `dense-t025 vs dense-t050` telemetry 辅助对照表. 用于判断收益是否伴随表示质量和 remote 路由模式的同步变化.

| axis | `dense-t025` run_id | `dense-t050` run_id | `t025 relative_err` | `t050 relative_err` | delta | `t025 remote_energy` | `t050 remote_energy` | delta | `t025 routing_entropy` | `t050 routing_entropy` | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reference` | `dense-t025-s123-d123` | `dense-t050` | `0.015469` | `0.020689` | `-0.005220` | `0.312489` | `0.271016` | `+0.041473` | `2.905342` | `3.404022` | `-0.498680` |
| `seed axis` | `dense-t025-s124-d123` | `dense-t050-s124` | `0.011814` | `0.021286` | `-0.009472` | `0.305752` | `0.275976` | `+0.029775` | `2.621012` | `3.445248` | `-0.824236` |
| `seed axis` | `dense-t025-s125-d123` | `dense-t050-s125-rerun` | `0.023259` | `0.020406` | `+0.002852` | `0.286617` | `0.238132` | `+0.048485` | `3.531729` | `3.951421` | `-0.419692` |
| `danger data axis` | `dense-t025-s123-d124` | `dense-t050-s123-d124` | `0.023502` | `0.022500` | `+0.001002` | `0.325135` | `0.226479` | `+0.098656` | `2.262894` | `3.889661` | `-1.626767` |
| `danger data axis` | `dense-t025-s123-d125` | `dense-t050-s123-d125` | `0.027154` | `0.023563` | `+0.003591` | `0.301903` | `0.234914` | `+0.066989` | `3.497218` | `3.828746` | `-0.331528` |

6. `dense-t025` 验证运行跟踪表. 记录当前已启动和排队中的验证任务, 便于后续回填 4, 5 两张对照表.

| run_id | seed | data_seed | current_status | queue / session | launch_id | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `dense-t025-s124-d123` | `124` | `123` | `completed` | `GPU0 queue / e3_t025_val_182802` | `flash-vqg-20260402-clr-v1-e3-t025-s124-d123-2026-04-08-18-28-05` | `seed axis validation; matched pass vs dense-t050-s124` |
| `dense-t025-s123-d124` | `123` | `124` | `completed` | `GPU0 queue / e3_t025_val_182802` | `flash-vqg-20260402-clr-v1-e3-t025-s123-d124-2026-04-08-21-34-52` | `danger data axis validation; matched pass vs dense-t050-s123-d124` |
| `dense-t025-s123-d125` | `123` | `125` | `completed` | `GPU0 queue / e3_t025_val_182802` | `flash-vqg-20260402-clr-v1-e3-t025-s123-d125-2026-04-09-00-43-34` | `danger data axis validation; matched pass vs dense-t050-s123-d125` |
| `dense-t025-s125-d123` | `125` | `123` | `completed` | `GPU0 follow-up queue / e3_t025_wait_183741` | `flash-vqg-20260402-clr-v1-e3-t025-s125-d123-2026-04-09-03-45-46` | `seed axis validation; matched pass vs dense-t050-s125-rerun` |

7. `dense -> top-k write` probe 对照表. 固定 `tau=0.25`, `top2 read`, `den_aware selector`, `shared_den`, 只改 `vq_weight_mode=topk_softmax` 和 `vq_topk`.

| run_id | write_mode | `vq_topk` | launch_id | status | final `valid/accuracy` | delta vs `dense-t025` | final `valid/loss` | delta vs `dense-t025` | `acc@512` | delta vs `dense-t025` | `acc@1024` | delta vs `dense-t025` | `valid/vq/relative_err_mean` | `valid/vq/c_entropy` | `valid/vq/write_entropy_mean` | `valid/vq/write_top1_mass_mean` | queue / session | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense-t025-s123-d123` | `dense_softmax` | `128` | `flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12` | `completed` | `0.981208` | `reference` | `0.081622` | `reference` | `0.991031` | `reference` | `0.874887` | `reference` | `0.015469` | `3.568453` | `2.969296` | `0.287130` | `-` | `dense anchor` |
| `topk2-t025-s123-d123` | `topk_softmax` | `2` | `flash-vqg-20260402-clr-v1-e3-topkwrite-probe-t025-2026-04-08-19-08-17` | `completed` | `0.893867` | `-0.087341` | `0.506182` | `+0.424560` | `0.886621` | `-0.104410` | `0.419883` | `-0.455004` | `0.059813` | `1.335524` | `0.403248` | `0.845876` | `-` | `clear fallback vs dense anchor` |
| `topk4-t025-s123-d123` | `topk_softmax` | `4` | `flash-vqg-20260402-clr-v1-e3-topkwrite-probe-t025-2026-04-08-19-08-17` | `completed` | `0.956567` | `-0.024642` | `0.187159` | `+0.105537` | `0.969309` | `-0.021723` | `0.729371` | `-0.145516` | `0.035858` | `2.126657` | `0.951813` | `0.674146` | `-` | `clear recovery vs topk2, but still below dense anchor` |
| `topk8-t025-s123-d123` | `topk_softmax` | `8` | `flash-vqg-20260402-clr-v1-e3-topkwrite-probe-t025-k8k16-2026-04-09-03-14-58` | `completed` | `0.964524` | `-0.016684` | `0.152148` | `+0.070527` | `0.977973` | `-0.013059` | `0.772375` | `-0.102512` | `0.023085` | `2.732556` | `1.569098` | `0.530218` | `GPU1 detached queue / e3_topkwrite_probe_k8k16_031456` | `continued recovery; still below dense anchor` |
| `topk16-t025-s123-d123` | `topk_softmax` | `16` | `flash-vqg-20260402-clr-v1-e3-topkwrite-probe-t025-k8k16-2026-04-09-03-14-58` | `completed` | `0.970430` | `-0.010779` | `0.126683` | `+0.045061` | `0.981309` | `-0.009723` | `0.812148` | `-0.062738` | `0.021223` | `3.296324` | `2.166535` | `0.434942` | `GPU1 detached queue / e3_topkwrite_probe_k8k16_031456` | `best sparse point; trends toward dense but does not beat dense anchor` |

8. `best-write x read` interaction 2x2 记录表. 固定 `tau=0.25`, `den_aware selector`, `shared_den`, 只比较当前最优 dense write 候选 `dense-t025` 与当前最优 sparse write 候选 `topk16-t025` 在 `top2 read` 和 `dense read` 下的表现. 本表只记录实验 3 当前机制族内部的 interaction/probe 数据, 不引入 `one-hot + ema` 历史结果.

| run_id | write_mode | `vq_topk` | read_mode | `fox_remote_read_topk` | launch_id | status | final `valid/accuracy` | delta vs `dense-t025 + top2` | final `valid/loss` | delta vs `dense-t025 + top2` | `acc@512` | delta vs `dense-t025 + top2` | `acc@1024` | delta vs `dense-t025 + top2` | queue / session | decision_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense-t025-s123-d123` | `dense_softmax` | `128` | `topk` | `2` | `flash-vqg-20260402-clr-v1-e3-tau-local-t025-2026-04-08-11-45-12` | `completed` | `0.981208` | `reference` | `0.081622` | `reference` | `0.991031` | `reference` | `0.874887` | `reference` | `-` | `current write/read anchor` |
| `dense-t025-dread-s123-d123` | `dense_softmax` | `128` | `dense` | `None` | `flash-vqg-20260402-clr-v1-e3-dread-t025-2026-04-09-08-32-54` | `completed` | `0.965413` | `-0.015795` | `0.158107` | `+0.076486` | `0.987422` | `-0.003609` | `0.765508` | `-0.109379` | `GPU0 detached queue / e3_t025_dread_083252` | `completed probe; dense read is not competitive under dense write` |
| `topk16-t025-s123-d123` | `topk_softmax` | `16` | `topk` | `2` | `flash-vqg-20260402-clr-v1-e3-topkwrite-probe-t025-k8k16-2026-04-09-03-14-58` | `completed` | `0.970430` | `-0.010779` | `0.126683` | `+0.045061` | `0.981309` | `-0.009723` | `0.812148` | `-0.062738` | `GPU1 detached queue / e3_topkwrite_probe_k8k16_031456` | `best sparse-write point under top2 read; still below dense anchor` |
| `topk16-t025-dread-s123-d123` | `topk_softmax` | `16` | `dense` | `None` | `-` | `dropped` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `not scheduled because dense-read branch is already non-competitive` |

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
- 实验 2 已完成, 当前不支持 denominator-free 方案晋升, `2A residual_add` 只保留为候选
- 预算不收紧时, `E2-main` 与 `E2b` 使用双卡并行训练
- 两边 full train 前都必须先完成 smoke, 并各自确定 batch/GA
- 实验 3 当前默认工作点是 `dense-t025 + top2 read + den_aware selector + shared_den`, `dense-t050` 退回保守回退点
- 实验 3 full train 前也必须先完成自己的 smoke, 并单独确定 batch/GA
- 只在 `clr_v1` 主线内推进实验 2, 不引入 `clr_delta_v1`
- SwanLab 继续作为 logger, 但分析统一优先读取本地产物
