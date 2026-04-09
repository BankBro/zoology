# Flash-VQG 实验 2 baseline 稳定性处理方案

> status: validating  
> doc_id: flash-vqg-e2-baseline-stability-plan  
> version: v1  
> owner: lyj  
> reviewers: pending  
> created: 2026-04-07  
> updated: 2026-04-07  
> parent: none  
> supersedes: none

## 1. 背景现状与目标

### 1.1. 背景

当前 `clr_v1` 主线在实验 1 之后已经切到 `top2 read` baseline. 实验 2 的职责边界已经明确: `E2-main` 判断 remote denominator 是否仍然是接口必要组成, `E2b` 判断在固定 `den_aware selector` 时哪种 merge 更好. 最近补跑的 `baseline-s124`, `baseline-s125`, `e2b-2a-l100-s123-rerun` 进一步暴露出一个更基础的问题: `E2b baseline` 自己明显不稳. 这意味着后续无论是继续追 `E2b-2A`, 还是进入实验 3, 都会被 baseline 漂移污染.

### 1.2. 现状

- 已确认 `E2b baseline` 的 probe 配置与正式配置一致, 不是脚本口径漂移.
- 代码分析显示 baseline 走的是 `den_aware + shared_den + legacy VQ(one-hot + ema)` 路径, 这条路径会把早期随机 routing 偏差通过 `L_state -> log(den_eff) -> selector -> L_state` 持续放大.
- 当前补跑中 `baseline-s125` 是典型坏 seed, 但 `l100` 在相同 seed 下没有同步崩坏, 说明问题更像 baseline 路径结构敏感, 而不是单纯数据坏 seed.
- 现有脚本还不支持本轮最关键的诊断动作: `seed` 与 `data_seed` 解耦后的 baseline 单 run probe.
- 现有 CLI 也还没有暴露 `codebook_init_method` 和 `codebook_init_seed`, 因而无法直接做 baseline 初始化稳定化对照.

### 1.3. 目标

- 用最小实验集先分清 baseline 不稳主要来自模型初始化, 还是数据实例 / batch 顺序.
- 在不改变 `E2b baseline` 接口定义的前提下, 优先验证"只稳住码本初始化"是否足以显著降低 baseline 方差.
- 给出进入实验 3 之前的明确闸门, 避免在 baseline 未稳住时继续扩搜索面.

## 2. 推荐方案概览

### 2.1. 一句话方案

先补一个可解耦 `seed` / `data_seed` 的 baseline probe 入口, 用 4 条最小诊断 run 分清方差来源; 若模型初始化主导或混合主导, 再补 3 条固定码本初始化 run 做 baseline 稳定化验证; 只有 baseline 稳定性达标后, 才继续 `E2b-2A` 或进入实验 3.

### 2.2. 核心流程

1. 先补最小脚本入口, 允许 baseline probe 在 `seed != data_seed` 时单 run 启动.
2. 跑 4 条 `seed/data_seed` 解耦诊断 run, 用最小成本判断 baseline 漂移源头.
3. 如果诊断显示模型初始化主导或模型/数据共同主导, 再补"固定码本初始化"这一最小稳定化干预.
4. 若固定初始化仍不够稳, 再进入 `bootstrap_farthest` 这一第二层干预.
5. 只有当 baseline 稳定性满足闸门, 才恢复 `E2b-2A` 比较或推进实验 3.

### 2.3. 关键接口与数据

- 相关脚本目录:
  - `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/`
- 现有可复用脚本:
  - `run_e2b_baseline_probe.sh`
  - `run_e2b_2a_l100_probe.sh`
  - `run_e2b_train.sh`
- 需要新增的最小入口:
  - `config_builder.py:build_e2b_baseline_probe_decoupled_configs`
  - `run_e2b_baseline_probe_decoupled.sh`
  - `run_e2b_baseline_probe_initfix.sh`
- 需要新增的最小 CLI 参数:
  - `--codebook-init-method`
  - `--codebook-init-seed`
- 固定运行口径:
  - `dmodel=128`
  - `num_codebook_vectors=128`
  - `train_batch_order=global_shuffle`
  - `max_epochs=32`
  - `fox_remote_formula=clr_v1`
  - `fox_clr_selector_mode=den_aware`
  - `fox_clr_merge_mode=shared_den`
  - `fox_clr_gate_mode=off`
  - `fox_remote_read_topk=2`
  - `TRAIN_BATCH_SIZE=64`
  - `EVAL_BATCH_SIZE=16`
  - `GRADIENT_ACCUMULATION_STEPS=4`

### 2.4. 边界条件与不变式

- 本文所有新实验都属于 `performance baseline1: E2b canonical baseline stability`.
- 除非专题内显式说明, 不允许改 `E2b baseline` 的接口定义, 只允许改"启动方式"和"初始化稳定化手段".
- `E2b-2A(l100)` 在本文中只作为后续受益方, 不是当前主诊断对象.
- baseline 没稳住前, 不新增 `lambda` 搜索, 不新增 `2B/2C` 搜索, 不进入实验 3.

## 3. 专题设计

### 3.1. 专题清单与边界

| topic | goal | current_version | status | note |
|---|---|---|---|---|
| topic-a | 解耦 `seed` 与 `data_seed`, 判断 baseline 方差来源 | v1 | validating | 这是当前最高优先级, 也是后续所有处理的前置 |
| topic-b | 在不改接口定义前提下, 用最小初始化干预稳定 baseline | v1 | validating | 只做 fixed init 和 `bootstrap_farthest` 两层, 不直接改 merge mode |
| topic-c | 定义 baseline 稳定性闸门, 决定何时恢复 `E2b-2A` 或进入实验 3 | v1 | validating | 不靠单次 best 点做决策 |

### 3.2. 专题 A: `seed/data_seed` 解耦诊断

**(1)** 当前状态与采纳版本

- status: validating
- current_version: v1
- summary: 当前 baseline seed 实验同时改变了模型初始化, 数据实例和 batch 顺序, 还不能判断 baseline 不稳究竟由哪一侧主导. 第一优先级是先做解耦诊断.

**(2)** 当前设计

先新增一个只产出单条 canonical baseline config 的 builder, 但去掉 `seed == data_seed` 限制:

- builder 名称: `build_e2b_baseline_probe_decoupled_configs`
- 脚本名称: `run_e2b_baseline_probe_decoupled.sh`
- 配置语义保持不变:
  - `experiment_part='e2b_probe'`
  - `experiment_mode='baseline'`
  - `selector_mode='den_aware'`
  - `merge_mode='shared_den'`
  - `gate_mode='off'`
  - `lambda_remote=1.0`
  - `run_id='baseline-s{seed}-d{data_seed}'`

本轮最小诊断集固定为 4 条:

| launch_id_prefix | run_id | seed | data_seed | goal |
|---|---|---:|---:|---|
| `flash-vqg-20260402-clr-v1-e2b-baseline-s124-d123` | `baseline-s124-d123` | 124 | 123 | 固定数据, 只换模型 seed |
| `flash-vqg-20260402-clr-v1-e2b-baseline-s125-d123` | `baseline-s125-d123` | 125 | 123 | 固定数据, 只换模型 seed |
| `flash-vqg-20260402-clr-v1-e2b-baseline-s123-d124` | `baseline-s123-d124` | 123 | 124 | 固定模型 seed, 只换数据 seed |
| `flash-vqg-20260402-clr-v1-e2b-baseline-s123-d125` | `baseline-s123-d125` | 123 | 125 | 固定模型 seed, 只换数据 seed |

推荐执行顺序:

1. 第一轮双卡:
   - GPU0: `baseline-s124-d123`
   - GPU1: `baseline-s123-d124`
2. 第二轮双卡:
   - GPU0: `baseline-s125-d123`
   - GPU1: `baseline-s123-d125`

命令口径固定如下:

```bash
GPU_ID=0 \
SEED_VALUES=124 \
DATA_SEED=123 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-s124-d123 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_decoupled.sh
```

```bash
GPU_ID=1 \
SEED_VALUES=123 \
DATA_SEED=124 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-s123-d124 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_decoupled.sh
```

```bash
GPU_ID=0 \
SEED_VALUES=125 \
DATA_SEED=123 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-s125-d123 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_decoupled.sh
```

```bash
GPU_ID=1 \
SEED_VALUES=123 \
DATA_SEED=125 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-s123-d125 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_decoupled.sh
```

本专题的判读规则:

- 若 `s124-d123` 和 `s125-d123` 仍大幅漂移, 而 `s123-d124` / `s123-d125` 漂移较小, 判为"模型初始化主导".
- 若 `s123-d124` 和 `s123-d125` 漂移更大, 判为"数据实例 / batch 顺序主导".
- 若两边都大, 判为"模型与数据共同放大", 后续仍优先从初始化稳定化入手.

**(3)** 历史版本演进

| version | date | status | summary | change_reason |
|---|---|---|---|---|
| v1 | 2026-04-07 | validating | 先用 4 条最小 run 解耦 `seed` 与 `data_seed`, 判断 baseline 方差来源 | baseline 漂移已成为是否继续实验 2/3 的主阻塞 |

**(4)** 实验记录

| version | date | experiment | metric | baseline | current | result | note |
|---|---|---|---|---|---|---|---|
| v1 | 2026-04-06 20:30:00 | `baseline-s124`, `baseline-s125` 对称补跑 | `baseline-s124 last=0.9123`, `baseline-s125 last=0.5713`, `1024x256` 末值极差巨大 | `performance baseline1: E2b canonical baseline stability` | 原始 matched-seed baseline 补齐 | fail | 已证明 baseline 明显不稳, 但仍未分清模型 seed 与数据 seed 的相对贡献 |

**(5)** 风险与兼容性

- 现有 baseline probe 脚本强制 `seed == data_seed`, 不补入口就无法执行本专题.
- 若沿用 `run_e2b_train.sh`, 会把 `2A/2B/2C` 全矩阵一起带上, 成本过高且会污染结论边界.
- 该专题只适合回答"波动来源", 不能直接回答"如何修复".

**(6)** 未决问题与下一步

- 先补 decoupled baseline probe 入口.
- 跑完 4 条后, 立刻出一版 `seed axis` 与 `data axis` 的 paired delta 表.
- 若模型初始化主导或混合主导, 直接进入专题 B.
- 若数据实例主导, 先单独审查 data segment seed 和 sampler 再决定是否做初始化干预.

### 3.3. 专题 B: baseline 初始化稳定化

**(1)** 当前状态与采纳版本

- status: validating
- current_version: v1
- summary: 若专题 A 显示模型初始化参与主导 baseline 漂移, 则优先尝试不改接口定义的最小干预: 固定码本初始化. 只有 fixed init 仍不够稳时, 才进入 `bootstrap_farthest`.

**(2)** 当前设计

本专题需要先补最小 CLI 透传, 让实验脚本可以传入:

- `--codebook-init-method`
- `--codebook-init-seed`

同时新增单 run 入口:

- 脚本名称: `run_e2b_baseline_probe_initfix.sh`
- builder 名称: `build_e2b_baseline_probe_initfix_configs`
- 运行 ID 规则:
  - `baseline-initfix-scale-cb20260407-s123-d123`
  - `baseline-initfix-scale-cb20260407-s124-d124`
  - `baseline-initfix-scale-cb20260407-s125-d125`
  - 若进入第二层干预, 则使用 `baseline-initfix-bootstrap-s{seed}-d{data_seed}`

分两层执行:

第一层, fixed init:

| launch_id_prefix | run_id | codebook_init_method | codebook_init_seed | seed | data_seed |
|---|---|---|---:|---:|---:|
| `flash-vqg-20260402-clr-v1-e2b-baseline-initfix-scale-cb20260407-s123-d123` | `baseline-initfix-scale-cb20260407-s123-d123` | `scale` | 20260407 | 123 | 123 |
| `flash-vqg-20260402-clr-v1-e2b-baseline-initfix-scale-cb20260407-s124-d124` | `baseline-initfix-scale-cb20260407-s124-d124` | `scale` | 20260407 | 124 | 124 |
| `flash-vqg-20260402-clr-v1-e2b-baseline-initfix-scale-cb20260407-s125-d125` | `baseline-initfix-scale-cb20260407-s125-d125` | `scale` | 20260407 | 125 | 125 |

第二层, 仅当第一层不够稳时才执行:

| launch_id_prefix | run_id | codebook_init_method | codebook_init_seed | seed | data_seed |
|---|---|---|---:|---:|---:|
| `flash-vqg-20260402-clr-v1-e2b-baseline-initfix-bootstrap-s123-d123` | `baseline-initfix-bootstrap-s123-d123` | `bootstrap_farthest` | -1 | 123 | 123 |
| `flash-vqg-20260402-clr-v1-e2b-baseline-initfix-bootstrap-s124-d124` | `baseline-initfix-bootstrap-s124-d124` | `bootstrap_farthest` | -1 | 124 | 124 |
| `flash-vqg-20260402-clr-v1-e2b-baseline-initfix-bootstrap-s125-d125` | `baseline-initfix-bootstrap-s125-d125` | `bootstrap_farthest` | -1 | 125 | 125 |

第一层命令模板:

```bash
GPU_ID=0 \
SEED_VALUES=123 \
DATA_SEED=123 \
CODEBOOK_INIT_METHOD=scale \
CODEBOOK_INIT_SEED=20260407 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-initfix-scale-cb20260407-s123-d123 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_initfix.sh
```

```bash
GPU_ID=1 \
SEED_VALUES=124 \
DATA_SEED=124 \
CODEBOOK_INIT_METHOD=scale \
CODEBOOK_INIT_SEED=20260407 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-initfix-scale-cb20260407-s124-d124 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_initfix.sh
```

```bash
GPU_ID=0 \
SEED_VALUES=125 \
DATA_SEED=125 \
CODEBOOK_INIT_METHOD=scale \
CODEBOOK_INIT_SEED=20260407 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-initfix-scale-cb20260407-s125-d125 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_initfix.sh
```

第二层命令模板:

```bash
GPU_ID=0 \
SEED_VALUES=123 \
DATA_SEED=123 \
CODEBOOK_INIT_METHOD=bootstrap_farthest \
CODEBOOK_INIT_SEED=-1 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-initfix-bootstrap-s123-d123 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_initfix.sh
```

本专题的执行规则:

1. 先只跑 fixed init 的 3 条.
2. 如果 fixed init 已显著缩小 `last valid/accuracy` 极差和 `1024x256` 极差, 则停止, 不再进入第二层.
3. 只有 fixed init 仍无法压住 `s125` 类坏收敛时, 才跑 `bootstrap_farthest`.

**(3)** 历史版本演进

| version | date | status | summary | change_reason |
|---|---|---|---|---|
| v1 | 2026-04-07 | validating | 优先尝试 fixed init, 失败后再尝试 `bootstrap_farthest` | 码本随机初始化是当前最可疑的结构性放大器 |

**(4)** 实验记录

| version | date | experiment | metric | baseline | current | result | note |
|---|---|---|---|---|---|---|---|
| v1 | 2026-04-07 00:00:00 | 代码审查 baseline 路径 | 发现默认 `codebook_init_method=scale`, `codebook_init_seed=None`, 且 baseline 使用 `one-hot + ema + shared_den` | `performance baseline1: E2b canonical baseline stability` | baseline 初始化稳定化设计 | pass | 该证据支持"先修初始化, 再谈结构升级"的处理顺序 |

**(5)** 风险与兼容性

- 当前 CLI 还不支持传 `codebook_init_method` 和 `codebook_init_seed`, 需先补入口.
- fixed init 若显著改善稳定性, 说明 baseline 不稳主要是训练工艺问题, 不能把此前 `l100` 的稳健性简单解释成更优接口.
- `bootstrap_farthest` 可能改变早期几轮训练动力学, 因而其结果只能用于"baseline 稳定化", 不能直接回写为历史 baseline.

**(6)** 未决问题与下一步

- 若 fixed init 就足够稳, 直接拿稳定化 baseline 去复核 `E2b-2A(l100)` 的真实收益.
- 若 `bootstrap_farthest` 也不稳, 则需要把调查范围扩大到 data segment seed 与 batch sampler.
- baseline 未稳住前, 不恢复 `l050`, `2B`, `2C`, 也不进入实验 3.

### 3.4. 专题 C: 稳定性闸门与恢复条件

**(1)** 当前状态与采纳版本

- status: validating
- current_version: v1
- summary: 当前不接受基于单次 `best_valid_accuracy` 的升级. 需要先定义 baseline 稳定性闸门, 再决定何时恢复 `E2b-2A` 或进入实验 3.

**(2)** 当前设计

baseline 稳定性闸门固定为:

- 三个 canonical seed 的 `last valid/accuracy` 极差 `<= 0.015`
- 三个 canonical seed 的 `valid/mqar_case/accuracy-1024x256` 末值极差 `<= 0.030`
- 任一 seed 的 `best-last` 回落 `<= 0.010`
- 不允许出现类似 `baseline-s125` 这种"从 epoch 早期开始就长期低 entropy, 高 margin"的坏收敛

只有 baseline 达标后, 才执行下一步:

1. 用稳定化 baseline 重做 `E2b-2A(l100)` 的 matched-seed 比较.
2. 若 `l100` 仍只表现为 `robust-but-not-winner`, 则继续 baseline 作为主线.
3. 若 `l100` 在稳定化 baseline 下仍稳定赢 overall 和 hardest bucket, 才有资格进入主线候选讨论.
4. 在这之前, 不进入实验 3.

可选排除项只保留 1 条:

```bash
TORCH_DETERMINISTIC=1 \
GPU_ID=0 \
SEED_VALUES=123 \
DATA_SEED=123 \
LAUNCH_ID_PREFIX=flash-vqg-20260402-clr-v1-e2b-baseline-s123-d123-det1 \
bash zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe_decoupled.sh
```

这条 run 只用于排除底层 nondeterminism 是否额外放大了噪声, 不是主修复路径.

**(3)** 历史版本演进

| version | date | status | summary | change_reason |
|---|---|---|---|---|
| v1 | 2026-04-07 | validating | 明确 baseline 稳定性闸门, baseline 未达标前不进实验 3 | 当前最大风险是把训练方差误判成结构结论 |

**(4)** 实验记录

| version | date | experiment | metric | baseline | current | result | note |
|---|---|---|---|---|---|---|---|
| v1 | 2026-04-06 20:30:00 | `baseline-s124`, `baseline-s125`, `l100-s123-rerun` 综合复盘 | baseline `last` 与 hardest bucket 极差都远超可接受范围, `l100` 更稳但尚未形成稳定新赢家 | `performance baseline1: E2b canonical baseline stability` | 实验 2 收尾裁决 | fail | 这条证据直接支持"先稳 baseline, 再恢复 E2b 或进入实验 3" |

**(5)** 风险与兼容性

- 若跳过闸门直接做实验 3, 后续 codebook / routing 结论会高度可疑.
- 若把 `l100` 的稳健性误当成绝对性能优势, 会过早替换 baseline, 反而丢失定位 baseline 问题的机会.
- `TORCH_DETERMINISTIC=1` 即使正向, 也不能替代专题 A 和专题 B.

**(6)** 未决问题与下一步

- 专题 A 和专题 B 完成后, 立刻用本文闸门做一次统一裁决.
- baseline 达标后再决定是否补 `l100-s124/s125` 新对照.
- 未达标则继续 baseline 稳定性诊断, 暂停实验 3.

## 4. 集成验证与落地

### 4.1. 集成验证与回归

文档落地前的最小工程动作如下:

1. 新增 `build_e2b_baseline_probe_decoupled_configs`.
2. 新增 `run_e2b_baseline_probe_decoupled.sh`.
3. 在 `run_flash_vqg_suite.py` 和 `flash_vqg_suite.py` 中透传 `codebook_init_method` 与 `codebook_init_seed`.
4. 新增 `build_e2b_baseline_probe_initfix_configs` 与 `run_e2b_baseline_probe_initfix.sh`.
5. 用 `bash -n` 和 `python -m py_compile` 做最小静态校验.

实验执行后的最小分析输出如下:

- 一张 paired delta 表:
  - `seed axis delta`
  - `data axis delta`
- 一张 baseline 稳定性表:
  - `best`
  - `last`
  - `best-last`
  - `512x128`
  - `1024x256`
  - `remote_routing_entropy`
  - `remote_top1_top2_margin`
  - `o_remote_energy_ratio`
- 一份统一裁决:
  - `baseline stable / unstable`
  - `方差来源: init / data / mixed`
  - `是否恢复 E2b-2A: yes / no`
  - `是否进入实验 3: no`

### 4.2. 发布与回退

- 本文只定义计划, 不自动替换任何 baseline.
- 若专题 A 失败, 仅回退到"继续查数据 seed 与 sampler", 不扩矩阵.
- 若专题 B 的 fixed init 成功, 也先把它视为"稳定化 baseline", 不立即改写历史 baseline 定义.
- 只有在稳定性闸门满足后, 才能把新 baseline 写回实验文档主结论.

## 5. 参考

### 5.1. 参考资料

- `docs/reference/20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md`
- `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/README.md`
- `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/config_builder.py`
- `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_baseline_probe.sh`
- `zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e2-remote-interface/run_e2b_2a_l100_probe.sh`
- `zoology/data/utils.py`
- `zoology/train.py`
- `../Flash-VQG/src/flash_vqg/nn/attn_fox.py`
- `../Flash-VQG/src/flash_vqg/nn/vq.py`
- `../Flash-VQG/src/flash_vqg/nn/vq_init.py`
- `../Flash-VQG/src/flash_vqg/nn/configuration_flash_vqg.py`

### 5.2. 相关文档与实验附件

- 文档: `docs/plans/20260407-flash-vqg-e2-baseline-stability-plan.md`
- 文档: `docs/reference/20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md`
- 附件目录: `docs/plans/artifacts/flash-vqg-e2-baseline-stability-plan/topic-a/v1/`
- 附件目录: `docs/plans/artifacts/flash-vqg-e2-baseline-stability-plan/topic-b/v1/`
- 附件目录: `docs/plans/artifacts/flash-vqg-e2-baseline-stability-plan/topic-c/v1/`
