# Flash-VQG 稳定性探索初步结论与后续方向

日期: 2026-06-05

## 背景与本轮口径

本轮按三个前提重新整理已有实验:

- codebook size 不作为稳定性控制手段, 只作为容量/布局超参记录.
- 优先把 `128` 和 `256` codebook size 的效果做稳定, 特别是 `cb128-r8`, `cb256-r4` 这组 matched active capacity, 以及 `cb256-r8` 这组扩容边界.
- 不再孤立评价 `hard04`, `BBSB`, `readk4` 等名字, 而是按控制面分析它们是否可能互补. 组合有效性目前只是研究假设, 必须由矩阵实验验证.

本轮使用子代理集群做只读复核. `128/256 codebook 稳定性` 和 `源码可实施性` 两个代理完成. `组合机制` 和 `统计路线` 代理在远端 compact 时断流, 其分工由主线程用已有 artifact 和源码行号补齐. 因此本文结论以 CSV, history, report, manifest 和源码为依据, 不依赖 Codex 线程日志.

## 一句话结论

已有结果更像是两个放大链同时存在:

1. write/state 放大链: early write strength, beta, zeta, M-state 会把 seed 差异放大成 high/low basin. `hard04` 证明写入 trust-region 能显著稳定 cb64-r16, 但有 ceiling tax.
2. read-side basin lock-in: residual read 的 top-k 候选过窄时, 部分 seed 早期锁到坏候选. `read_topk=4` 在 cb256-r4/cb256-r8 上救回 weak seed, 但在 cb128-r8 和 cb64-r16 有明确反例, 所以方向不是固定 readk4, 而是 schedule/gate/margin-aware read-side 控制.

codebook size 本身不应被解释为控制手段. 更合理的研究路线是: 固定容量轴, 在每个容量轴上测试 write trust-region, beta trajectory, read-side coverage/gate 的组合.

## 关键实验事实

### 128/256 codebook 不是控制按钮

| 容量轴 | 已有事实 | 初步解释 |
|---|---|---|
| `cb128-r8` | 3090 readk2 s124/s125=`0.956/0.956`, spread `0.000`; readk4 主跑 `0.973/0.972`, 但 s125 rerun=`0.609`, all spread `0.364` | `cb128-r8` 可稳定, 但固定 readk4 不可靠. 不能把 readk4 当默认控制. |
| `cb256-r4` | 2080ti default/readk2 s124/s125=`0.6754/0.8348`; 3090 readk2 r1=`0.7725/0.9564`, r2=`0.891/0.953`; 3090 formal readk4 s124-r1c/r2c/s125=`0.943/0.958/0.944`, spread `0.015` | 同一 codebook/rank 下 default path-sensitive, read-side 控制能显著改善. |
| `cb256-r8` | 3090 readk2 s124/s125=`0.988/0.804`, spread `0.184`; readk4 s124-r1/r2=`0.982/0.982`, s125-r1/r2=`0.988/0.992`, spread `0.010` | readk4 在 cb256-r8 上是强正证据, 但这是 cb256-like 的 read-side 现象, 不是 codebook size 本身的控制效果. |
| `cb256-r10` | official longer-MQAR 4 seeds: `1024x256=0.9142±0.0824`, `2048x512=0.7080±0.2360`, `4096x1024=0.4589±0.2716`, `8190x512=0.6232±0.2605` | 上限高, 但 longer-MQAR 方差大. 扩容不能自动带来稳定. |

因此后续不应写成 "`cb256` 稳定" 或 "`cb128` 不稳定". 更准确的说法是: 在给定容量/布局下, read/write/beta/初始化轨迹决定是否进入稳定 basin.

### 写入 trust-region 是稳定化实锤, 但会降 ceiling

cb64-r16 default 三 seed:

| 配置 | s123 | s124 | s125 | spread |
|---|---:|---:|---:|---:|
| default | `0.968711` | `0.819797` | `0.987285` | `0.167488` |
| hard04 | `0.945039` | `0.963055` | `0.952605` | `0.018016` |
| caprel0406late | `0.949371` | `0.963004` | `0.960484` | `0.013633` |

`hard04` 的 same-seed 证据也强: s124 4ep repeat 为 `0.961621/0.960176`, gap `0.001445`. 350-step repeat 中 `write_gap=0`, `m_norm_max_gap=0.028772`.

但 `hard04` 明显压低 good seed ceiling: s123 从 `0.968711` 降到 `0.945039`, s125 从 `0.987285` 降到 `0.952605`. `caprel0406late` 能恢复部分 ceiling, 但 s123 `m_norm_max=14.487579`, 有 state 过冲风险. 这说明 release 思路有价值, 但 `0.04 -> 0.06` 释放过激, 不能直接成为主线.

### BBSB/ bounded beta 是 ceiling 恢复信号, 不是稳定结论

| 配置 | 数据 | 问题 |
|---|---|---|
| `BBSB t2` | s123 final `0.965113`, s124 final `0.914547`, s124 best `0.944094`, spread `0.050566` | s123 ceiling 好, s124 late drift. 缺 seed125. |
| `bounded beta fixed + cap0405` | s123 `0.963758`, s124 `0.900934`, s124 best `0.948012`, spread `0.062824` | s124 best-final gap `0.047078`. |
| `wqa0.75` | s123 `0.909148`, s124 `0.956859`, spread `0.047711` | seed123 ceiling 被压低. |
| `btb/budgeted` 系列 | 多数在 `0.85~0.87`, 或出现强 seed split | 当前设置下不适合作为主线. |

beta band 的机制仍值得保留: 它可能在 write cap 降低不稳定性的同时恢复部分 ceiling. 但当前 BBSB 不是稳定方案, 只能作为组合候选.

### 初始化实验更像因果诊断, 不是训练方法

`init transplant` 显示 good flash-only donor 不能稳定救回 bad/boundary recipient:

| 实验 | hard |
|---|---:|
| normal cb64-r16 s124 | `0.952305` |
| normal cb64-r16 s125 | `0.981039` |
| flashdonor s125 -> s124 | `0.836082` |
| nonflashdonor s125 -> s124 | `0.661695` |
| normal cb256-r4 s124 | `0.747195` |
| flashdonor cb256-r4 s123 -> s124 | `0.679957` |

结论: 初始化几何是放大链的一部分, 但不是充分条件. `codebook_init_seed` 或 init transplant 更适合作为诊断和可复现实验工具, 不应作为效果控制手段.

## 控制面重新归纳

### 容量/布局轴, 固定而非控制

优先固定三组:

| 轴 | 目的 |
|---|---|
| `cb128-r8` | 128 codebook, active capacity 约 `131k`, 与 cb256-r4 做 matched-capacity 对照. |
| `cb256-r4` | 256 codebook, active capacity 约 `131k`, 当前 default 不稳, readk4 有正证据. |
| `cb256-r8` | 256 codebook 扩容边界, readk4 当前最强, 但需要正式化和 longer-MQAR. |

`cb64-r16` 仍保留为机制对照, 因为 hard04 证据最完整, 但不应继续作为优先优化目标.

### 写入侧 trust-region

已有实现支持:

- `write_strength_cap`, `write_strength_cap_mode`, `cap_final`, release start/end.
- `write_budget`, `write_total_cap`, `m_norm_cap`, `update_norm_cap`.

当前结论:

- hard cap `0.04` 是稳定基准.
- `0.04 -> 0.06` 有 ceiling 恢复, 但可能导致 M-state 过冲.
- 下一步更像是 `0.04 -> 0.05` conservative release, 或 release + m_norm/update guard.

### beta trajectory

已有实现支持:

- `hard_cap` beta.
- `bounded_sigmoid`: `beta = low + (high - low) * sigmoid(logits / temp)`.
- beta band low/high 到 final band 的 schedule.

当前结论:

- beta band 可能负责恢复表达能力, 但单独不稳.
- beta band 应与 write trust-region 联合测试, 不应再孤立评价 BBSB.

### read-side coverage/gate

已有实现支持固定 `fox_remote_read_topk`.

当前结论:

- 固定 readk4 在 cb256-r4/cb256-r8 上强正.
- 固定 readk4 在 cb128-r8 rerun 和 cb64-r16 上有强反例.
- 更合理方向是 schedule/gate/margin-aware read control:
  - early 阶段允许更多 residual candidates.
  - routing margin 足够大时回到 top2.
  - margin 低或 residual uncertainty 高时临时扩到 top4.

注意: 这个 gate/schedule 目前不是完整现成功能, 需要新增机制或先用固定 readk2/readk4 做边界验证.

## 组合假设

下面不是结论, 是下一步需要证伪的组合假设.

| 组合 | 作用分工 | 预期收益 | 主要风险 |
|---|---|---|---|
| `write hard04 + read gate` | hard04 抑制 write/state 爆, read gate 避免 top2 early lock-in | 同时解决 state 放大和 bad basin lock | 双重保守可能压低 ceiling. |
| `cap0405 conservative release + bounded beta band` | cap 保持 early 稳定, beta band 恢复中后期表达能力 | 可能比 hard04 少 ceiling tax | release 仍可能 late drift, 需看 best-final gap. |
| `cb256 readk4 + mild write cap` | readk4 已能救 weak seed, mild cap 防止 m_norm 过冲 | 对 cb256-r4/r8 可能进一步降低 repeat 方差 | cb256-r8 readk4 已高, 加 cap 可能反而降 ceiling. |
| `cb128 readk2 + write/beta 控制` | 避开 cb128-r8 fixed readk4 的 rerun 崩盘, 先从 write/beta 稳定轨迹入手 | 稳住 128 codebook 的 matched-capacity 轴 | 如果主要问题是 read candidate coverage, readk2 可能 ceiling 不够. |
| `read schedule/gate + beta band` | read gate 控制候选覆盖, beta band 控制 residual 注入强度 | 解决 readk4 fixed 过宽和 top2 过窄的两端问题 | 机制较复杂, 需要先做小矩阵验证. |

## 后续研究路线

### Phase A: 建立 128/256 容量轴基线

固定:

- `data_seed=123`, `b64_ga4`, effective batch `256`, `fp32`.
- `max_epochs=4`, no early stopping, validations per epoch `2`.
- `gd_residual_v1`, `write_topk=4`, `vq_topk=4`, `codebook_dot`, `dense_softmax`, `grad`, tau `0.25`.
- codebook size 和 rank 只作为容量轴: `cb128-r8`, `cb256-r4`, `cb256-r8`.

最小矩阵:

| 容量轴 | baseline | seeds | repeat |
|---|---|---|---|
| `cb128-r8` | readk2 | s123/s124/s125 | 最差 seed rerun |
| `cb256-r4` | readk2 | s123/s124/s125 | 最差 seed rerun |
| `cb256-r8` | readk2 | s123/s124/s125 | 最差 seed rerun |

目标是先确认每个容量轴的 default 弱 seed, 不急于调参.

### Phase B: read-side 边界矩阵

| 容量轴 | 条件 | 目的 |
|---|---|---|
| `cb128-r8` | readk2 vs readk4 | 复核 fixed readk4 崩盘是否稳定复现. |
| `cb256-r4` | readk2 vs readk4 | 正式化 3090 readk4 结论. |
| `cb256-r8` | readk2 vs readk4 | 加 seed123, 复核最强候选. |

如果 fixed readk4 的反例继续存在, 再新增 `read_topk schedule/gate` 机制. 不建议直接把 fixed readk4 写入默认配置.

### Phase C: write/beta 组合矩阵

先在 `cb128-r8` 和 `cb256-r4` 做 matched-capacity, 再扩到 `cb256-r8`.

| 编号 | write 控制 | beta 控制 | 目的 |
|---|---|---|---|
| C0 | none | default | baseline. |
| C1 | hard04 | default | 复核 trust-region 是否跨 128/256 泛化. |
| C2 | cap `0.04 -> 0.05` late release | default | 测 ceiling 恢复, 降低 `0.04 -> 0.06` 过冲风险. |
| C3 | hard04 | bounded beta fixed band | 测 beta band 是否恢复 hard04 ceiling. |
| C4 | cap `0.04 -> 0.05` late release | bounded beta scheduled band | BBSB 的保守版本. |
| C5 | cap `0.04 -> 0.05` + m_norm/update guard | bounded beta scheduled band | 针对 caprel0406late 的 state 过冲风险. |

### Phase D: 组合主候选

只在 Phase B/C 中通过门槛的容量轴上跑:

| 候选 | 适用轴 | 备注 |
|---|---|---|
| `read gate + cap0405` | 优先 cb256-r4/cb256-r8 | read gate 负责候选覆盖, cap0405 负责 trust-region. |
| `read gate + cap0405 + bounded beta band` | 仅在上一个候选稳定后 | 增加 ceiling 恢复, 但复杂度更高. |
| `readk2 + cap0405 + bounded beta band` | 优先 cb128-r8 | 避免 cb128 fixed readk4 的 rerun 风险. |

### Phase E: longer-MQAR 正式化

进入 longer-MQAR 前的硬门槛:

- `1024x256` 三 seed final spread `<=0.03`.
- 最差 seed final hard `>=0.94`, 对 `cb128-r8` 可先用 `>=0.92` 作为探索门槛.
- same-seed repeat gap `<=0.01`.
- best-final gap `<=0.01`, 超过 `0.02` 视为 late drift 风险.
- `m_norm_max` 超过 `8` 标红, 超过 `12` 原则上不进入正式主线, 除非有额外 guard 和 repeat 证明.
- 需要记录 final checkpoint, dtype policy, GPU, seed, data_seed, codebook/rank, read/write/beta 控制项, launch_id/run_id.

longer-MQAR 指标至少覆盖:

- `1024x256`.
- `2048x512`.
- `4096x1024`.
- `8190x512`.
- `8190x2047`.

## 源码和执行边界

已有入口与实现:

- `FlashVQGMixer` 已透传 write cap, budget, beta band, lambda, codebook init 等配置: `zoology/mixers/flash_vqg.py`.
- phase2 `run_train.sh` 透传新增控制项: `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh`.
- 旧 `20260425-gd-residual-v1-mqar/run_train.sh` 只传基础 GD/VQ 参数, 不透传 phase2 控制项, 不应直接用于复现实验中的 cap/beta/codebook-init 控制.
- write cap 实际作用在 `zeta_uncapped = beta * write_strength` 后, 不是 raw routing weight.
- 当前 release 基于每个 attention module 的 train forward count, 不是 optimizer step.
- `read_topk` 当前是固定值, margin-aware gate/read schedule 需要新增机制.
- `codebook_init_rng_mode=local_burn` 可作为诊断/可复现工具, 但不应被报告成效果控制手段.

## 给后续评估 agent 的读取清单

为了让后续 agent 复核本方案, 建议按下面顺序读取文件. 不建议先全仓库搜索, 因为同名实验和历史 preliminary artifact 较多, 容易混入口径不同的数据.

### 读取顺序

1. 先读本文, 明确本轮口径: codebook size/rank 是容量轴, 不是控制手段.
2. 读 2080ti 写入控制线, 确认 `hard04`, `caprel0406late`, `BBSB` 的 seed, final, best-final gap 和 `m_norm_max`.
3. 读 3090 read-side 诊断线, 确认 `cb128-r8`, `cb256-r4`, `cb256-r8` 的 readk2/readk4 反例和正例.
4. 读 official/longer-MQAR artifact, 防止把探索性结果误写成正式结论.
5. 最后读源码入口和实现, 确认可组合控制项是否真的能由当前脚本透传.

### 本机必读文件

| 用途 | 文件 | 重点读取内容 |
|---|---|---|
| 2080ti seed instability 总结 | `tmp/20260529-seed-instability-full-cap/seed-instability-current-report-draft.md` | `hard04`, `caprel0406late`, ceiling tax, `m_norm_max` 风险. |
| 2080ti hard04/default 汇总 | `tmp/20260529-seed-instability-full-cap/seed-instability-final-summary.csv` | `run`, `valid/mqar_case/accuracy-1024x256`, `valid/attn/gd_residual_m_norm_max`, `valid/attn/gd_residual_write_strength_mean`, `valid/attn/gd_residual_write_strength_cap_hit_ratio`. |
| 2080ti 后续探索交接 | `tmp/20260529-seed-instability-full-cap/research-handoff.md` | BBSB, bounded beta, WQA, BTB 系列的负面和部分正面结果. |
| official seed stability | `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv` | `cb256-r4`, `cb64-r16` strict official seed stability rows. |
| official longer-MQAR | `docs/artifacts/longer-mqar/official-core-20260526/longer-mqar-official-core-summary.csv` | `cb256-r10`, `cb256-r4`, `cb64-r16`, GDN baselines 的 longer-MQAR 方差和均值. |
| init transplant | `docs/artifacts/20260603-gd-init-transplant/train-core-final.csv` | 证明 init transplant 是诊断工具, 不是稳定方法. |
| report 对应源码入口 | `zoology/mixers/flash_vqg.py` | mixer 是否透传 cap, beta band, init seed 等配置. |
| 旧入口边界 | `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh` | 只透传基础 GD/VQ 参数, 不适合直接复现 phase2 控制项. |
| phase2 入口 | `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh` | cap, budget, beta band, init rng/seed, lambda_floor, write_strength_mode 的透传. |

### 3090 必读文件

3090 默认指宿主机 `192.168.2.114` 的 `Flash-VQG-tun` 容器, 项目路径为 `/home/lyj/mnt/project/zoology`. 推荐读取方式:

```bash
ssh lyj@192.168.2.114 "docker exec -u lyj Flash-VQG-tun bash -lc 'cd /home/lyj/mnt/project/zoology && <cmd>'"
```

| 用途 | 3090 文件 | 重点读取内容 |
|---|---|---|
| 3090 总报告 | `docs/20260530-gd-seed-diag-report.md` | readk4 的适用边界, read-side basin lock-in 机制. |
| 跨配置 run-level 指标 | `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-final.csv` | `cb128-r8`, `cb256-r8`, `cb64-r16` 的 readk2/readk4 final hard, acc, loss, role, notes. |
| 跨配置 spread 汇总 | `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-spread-summary.csv` | readk4 all-completed spread, rerun gap, 反例判断. |
| cb256-r4 run-level 指标 | `docs/artifacts/20260530-gd-seed-diag/final.csv` | `ordinary_normal_readk2`, `ordinary_formal_readk4`, `pseudo_det_readk2` 的区别. |
| cb256-r4 spread 汇总 | `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-spread-summary.csv` | formal readk4 cross-seed 和 same-seed spread. |
| source manifest | `docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-source-manifest.csv` 和 `gd-seed-diag-cross-config-source-manifest.csv` | 对应 launch_id, run_id, log, manifest, history 路径. |

### 建议使用的快速核验命令

本机核验 hard04/default:

```bash
python3 - <<'PY'
import csv
p = 'tmp/20260529-seed-instability-full-cap/seed-instability-final-summary.csv'
for r in csv.DictReader(open(p)):
    print(r['run'], r['valid/mqar_case/accuracy-1024x256'], r['valid/attn/gd_residual_m_norm_max'])
PY
```

3090 核验 read-side 关键结果:

```bash
ssh lyj@192.168.2.114 "docker exec -u lyj Flash-VQG-tun bash -lc 'cd /home/lyj/mnt/project/zoology && python3 - <<\"PY\"
import csv
p = \"docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-final.csv\"
for r in csv.DictReader(open(p)):
    if r.get(\"config_family\") in {\"cb128-r8\", \"cb256-r8\", \"cb64-r16\"}:
        print(r.get(\"config_family\"), r.get(\"condition\"), r.get(\"seed\"), r.get(\"replicate\"), r.get(\"read_topk\"), r.get(\"final_valid_mqar_case_accuracy_1024x256\"), r.get(\"role\"), r.get(\"notes\"))
PY'"
```

### 源码复核行索引

| 结论 | 文件和行号 |
|---|---|
| mixer 已有 write cap, budget, beta band, lambda, init seed 参数 | `zoology/mixers/flash_vqg.py:25` 到 `zoology/mixers/flash_vqg.py:84` |
| 20260425 旧入口只传基础 GD/VQ 参数 | `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh:13` 到 `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh:60` |
| 20260526 phase2 入口透传 cap/budget/total cap | `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh:55` 到 `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh:130` |
| 20260526 phase2 入口透传 beta band/init seed | `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh:138` 到 `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh:219` |
| write cap release 使用 train forward count, 不是 optimizer step | `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py:498` 到 `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py:535` |
| bounded sigmoid beta band 计算 | `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py:718` 到 `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py:736` |
| write cap 作用在 `zeta=beta*write_strength` 之后 | `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:525` 到 `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:534` |
| residual read 的 `read_topk` 只作用于 residual correction top-k | `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:2186` 到 `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:2220` |
| codebook init RNG mode | `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/vq_init.py:15` 到 `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/vq_init.py:38` |

### 评估时必须避免的混淆

- 不要把 3090 readk4 的 cb256 正证据泛化为全局 fixed readk4.
- 不要把 `cb256-r8` 的高分解释成 codebook size 本身的稳定化效果. 同配置 readk2 s125=`0.804`, readk4 才救回到 `0.988/0.992`.
- 不要把 `caprel0406late` 直接推荐为正式主线. 它的 spread 很好, 但 s123 `m_norm_max=14.487579`.
- 不要把 BBSB 写成已验证稳定. 它目前是 ceiling 恢复信号, 但 s124 final drift 明显.
- 不要把 `codebook_init_seed` 或 init transplant 写成训练方法. 它们主要是诊断和复现工具.
- 不要混用 preliminary longer-MQAR 和 official longer-MQAR. cb128 目前没有 official core longer-MQAR 多 seed 结论.

## 报告表述规范

后续报告建议统一使用这些表述:

- 使用 "`codebook size/rank 是容量/布局轴`", 不使用 "`codebook 是稳定化控制手段`".
- 使用 "`readk4 在 cb256-like 配置上有强正证据, 但不是全局默认`".
- 使用 "`hard04 是 write/state trust-region 稳定基准, 但有 ceiling tax`".
- 使用 "`BBSB/ bounded beta 是 ceiling 恢复候选, 当前还不是稳定方案`".
- 使用 "`组合方向需要矩阵验证`", 不直接说组合已经有效.

## 主要来源

本机:

- `tmp/20260529-seed-instability-full-cap/seed-instability-current-report-draft.md`
- `tmp/20260529-seed-instability-full-cap/seed-instability-final-summary.csv`
- `tmp/20260529-seed-instability-full-cap/research-handoff.md`
- `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv`
- `docs/artifacts/longer-mqar/official-core-20260526/longer-mqar-official-core-summary.csv`
- `docs/artifacts/20260603-gd-init-transplant/train-core-final.csv`
- `zoology/mixers/flash_vqg.py`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh`
- `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py`
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/vq_init.py`

3090:

- `3090:/home/lyj/mnt/project/zoology/docs/20260530-gd-seed-diag-report.md`
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-final.csv`
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-spread-summary.csv`
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/final.csv`
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-spread-summary.csv`
