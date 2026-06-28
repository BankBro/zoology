# 20260628-03 Flash-VQG mixer divergence probe 报告

status: completed_debug_probe
ledger: not written

## 目标

本轮不是效果实验, 不跑正式 4 epoch, 不写 official MQAR ledger.

目标是在相同 cache, 相同 canonical init, 相同 batch order, no-dropout 条件下, 定位 2080ti 和 3090 在 `backbone.layers.1.sequence_mixer.mixer` 内部的第一处分叉.

## 口径

代码版本:

- zoology: `flash-vqg`, commit `99cc2f8`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `474b763`.

共同配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`, `read_topk=2`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- `fox_gate_logit_normalizer` 未在本轮实验配置中显式覆盖, zoology wrapper 构造 `FlashVQGConfig` 时也没有传入该字段, 因此实际走 Flash-VQG 默认值 `16`.
- canonical MQAR cache hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.

前置检查全部通过:

| item | 2080ti | 3090 |
|---|---|---|
| cache hash | match | match |
| init hash | match | match |
| batch order hash | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

运行完成后两边 GPU 均释放. 3090 raw JSON 已镜像回 2080ti, `preflight.json` 和 `probe.json` 的远端/本地 sha256 均一致.

## 结果

两边都完成 `703` 个 optimizer step. 本轮计划 trace `0,1,4,16,64,130,203,352,448,704`, 但 dataloader 一轮实际只产生到 step `703`, 因此实际 trace step 是:

```text
0, 1, 4, 16, 64, 130, 203, 352, 448
```

每个 trace step 都有 25 个 layer-1 mixer 内部 trace record. 跨机器 join 后:

- trace rows: `504`.
- comparison rows: `252`.
- mismatch rows: `200`.
- first mismatch: step `0`, micro `0`, layer `1`, `state_build/logf_all`.

step 0 明细:

| trace | match |
|---|---|
| `phase1/q_all` | true |
| `phase1/k_all` | true |
| `phase1/v_all` | true |
| `phase1/g_raw_all` | true |
| `phase1/K_q_all` | true |
| `phase1/Delta_all` | true |
| `phase1/W_all` | true |
| `state_build/logf_all` | false |
| `state_build/beta_all` | true |
| `state_build/G_state` | false |
| `state_build/L_state` | false |
| `state_build/M_state` | false |
| `phase2_read/top_idx` | true |
| `phase2_read/u_res` | true |
| `forward/preds` | true |
| `forward/loss` | true |

后续传播:

| step | first mismatch | mismatch count |
|---:|---|---:|
| 0 | `state_build/logf_all` | 10/28 |
| 1 | `phase1/q_all` | 24/28 |
| 4 | `phase1/q_all` | 17/28 |
| 16 | `phase1/q_all` | 25/28 |
| 64 | `phase1/q_all` | 24/28 |
| 130 | `phase1/q_all` | 25/28 |
| 203 | `phase1/q_all` | 27/28 |
| 352 | `phase1/q_all` | 28/28 |
| 448 | `phase1/q_all` | 20/28 |

离散路径开始分叉较晚:

- `phase2_read/top_idx` mismatch 出现在 step `16, 130, 203, 352`.
- `phase1/Delta_all` mismatch 出现在 step `352, 448`.
- `forward/preds` 和 `forward/loss` mismatch 出现在 step `203, 352, 448`.

## 判读

这轮把 no-dropout 后的剩余分叉位置进一步缩小了.

step 0 时, layer 1 的 `q/k/v`, `g_raw`, `K_q`, VQ assignment `Delta_all` 和 write weight `W_all` 都是 bitwise match. 因此第一处分叉不是 VQ routing, 不是 read top-k, 也不是输入, cache, init 或 batch order.

第一处分叉是 `state_build/logf_all`. 代码路径是:

```text
logf_all = fox_gate_logf(x, self.fox_gate_proj, self.config, attention_mask)
```

其中 `fox_gate_logf` 先做 `fox_gate_proj(x)`, 再做 `F.logsigmoid(logits.float())`. 在 step 0 的 `x` 已经由 phase1 侧间接证明一致, 所以当前最可能的起点是 gate/state-build 连续值路径上的 CUDA linear/logsigmoid 数值差异. 这个差异非常小, mean/l2 summary 仍几乎相同, 但 hash 已经不同.

随后, 这个很小的 `logf_all` 差异进入 `G_state/L_state/M_state`, 再进入 phase2 output. 到 step 16 开始影响 read top-k index, 到 step 203 以后影响 preds/loss. 这说明后续离散 read/top-k 是放大器之一, 但不是本轮观察到的第一起点.

## `fox_gate_logf` 数学流程

这条路径的作用不是做 VQ routing, 也不是直接做 read top-k. 它给 FoX/GD residual state 生成 per-token, per-head 的遗忘率, 决定历史写入在后续位置还保留多少.

### 1. 从输入生成 log-domain 遗忘门控

输入为当前层的 hidden state:

$$
x \in \mathbb{R}^{B \times T \times D}
$$

`fox_gate_proj` 给每个 token 和每个 attention head 生成一个 gate logit:

$$
z = \operatorname{permute}_{B,H,T}
\left(
  \operatorname{view}_{B,T,H}
  \left(
    \operatorname{fox\_gate\_proj}(x)
  \right)
\right),
\quad
z \in \mathbb{R}^{B \times H \times T}
$$

然后在 float32 中计算:

$$
\ell_{b,h,t}
=
\frac{\log \sigma(z_{b,h,t})}{\eta},
\quad
f_{b,h,t}
=
\exp(\ell_{b,h,t})
$$

其中 $\ell$ 对应代码里的 `logf`, $\eta$ 对应 `fox_gate_logit_normalizer`, $\sigma(\cdot)$ 是 sigmoid.

本轮实验中, `serialized_config` 的 Flash-VQG 子配置没有显式写入 `fox_gate_logit_normalizer`, `zoology.mixers.flash_vqg.FlashVQGMixer` 构造 `FlashVQGConfig` 时也没有传入该字段. 因此实际使用的是 Flash-VQG 默认值:

$$
\eta
=
\max(1, \texttt{fox\_gate\_logit\_normalizer})
=
\max(1, 16)
=
16
$$

因为 `log(sigmoid(z)) <= 0`, 所以:

$$
\ell_{b,h,t} \le 0,
\quad
0 < f_{b,h,t} \le 1
$$

直观含义:

```text
f 接近 1: 历史 state 衰减慢, 记得久
f 接近 0: 历史 state 衰减快, 忘得快
logf 接近 0: 记得久
logf 更负: 忘得快
```

代码对应:

```python
logits = fox_gate_proj(x).view(B, T, H).permute(0, 2, 1)
logf = F.logsigmoid(logits.float()) / normalizer
```

### 2. 用 prefix sum 表示长距离指数衰减

如果从位置 `i` 到位置 `t` 的历史保留量直接写, 是很多个 `f` 的连乘:

$$
\operatorname{decay}(i \rightarrow t)
=
\prod_{j=i+1}^{t} f_j
$$

为了避免长序列连乘带来的数值问题, 代码在 log 空间里做加法:

$$
c_t
=
\sum_{j \le t} \ell_j
$$

$$
\operatorname{decay}(i \rightarrow t)
=
\exp(c_t - c_i)
$$

代码对应:

```python
c_all = torch.cumsum(logf_all, dim=-1)
```

所以 `c_all` 可以理解成一条累计衰减时间轴. 后面 phase2 remote read 会用它给远处 state 加上位置衰减 bias.

### 3. block 内和 block 间的 state build

序列按 block 切分:

$$
T = N \cdot L
$$

其中 $N$ 是 block 数, $L$ 是 block 长度.

每个 token 会把自己的 value 写入若干 codebook bucket. 简化记号:

$$
W_{n,l,s}
=
\text{第 } n \text{ 个 block 内第 } l \text{ 个 token 写入 code } s \text{ 的权重}
$$

$$
V_{n,l}
=
\text{第 } n \text{ 个 block 内第 } l \text{ 个 token 的 value}
$$

`logf` 先产生两个关键衰减量:

$$
\alpha_n
=
\exp\left(
  \sum_{l=1}^{L} \ell_{n,l}
\right)
$$

$$
\gamma_{n,l}
=
\exp\left(
  \sum_{j>l} \ell_{n,j}
\right)
$$

含义:

```text
alpha_n: 整个 block 对进入该 block 之前旧 state 的衰减
gamma_{n,l}: block 内第 l 个 token 写入后, 到 block 结束时还剩多少
```

FoX coarse state 可以简化成两份累计量:

```text
L_state[s]: code s 的累计写入质量
G_state[s]: code s 的累计 value 总和
```

每个 block 的增量近似为:

$$
\Delta L_n[s]
=
\sum_{l=1}^{L}
\gamma_{n,l} W_{n,l,s}
$$

$$
\Delta G_n[s]
=
\sum_{l=1}^{L}
\gamma_{n,l} W_{n,l,s} V_{n,l}
$$

然后 block 级更新:

$$
L_{\mathrm{cur}}
\leftarrow
\alpha_n L_{\mathrm{cur}} + \Delta L_n
$$

$$
G_{\mathrm{cur}}
\leftarrow
\alpha_n G_{\mathrm{cur}} + \Delta G_n
$$

GD residual 额外维护 `M_state`, 用于保存 code 内部的 residual correction state. 它也受同一条 `logf` 衰减路径控制, 简化地看:

$$
M_{\mathrm{cur}}
\leftarrow
f_l M_{\mathrm{cur}} + \Delta M_l
$$

因此 `fox_gate_logf` 的极小差异不会只影响一个普通标量, 而是会影响 `G_state/L_state/M_state` 这整套远程 memory.

### 4. phase2 remote read 和离散 top-k 放大

读取时, local path 负责近邻窗口, remote path 负责更远历史. remote score 简化为:

$$
S_{\mathrm{far}}(q, s)
=
q^\top C_s
+
b_{\mathrm{decay}}
+
\log Z_s
$$

其中 `decay_bias` 来自 `c_all`, 直观上是:

$$
b_{\mathrm{decay}}
\approx
c_{\mathrm{query}} - c_{\mathrm{state\_boundary}}
$$

因为 `logf <= 0`, 距离越远, 衰减越大, 对应 score bias 越负.

远程 coarse output 简化为:

$$
\operatorname{Num}_{\mathrm{far}}
=
\sum_s
\exp(S_{\mathrm{far},s}) G_{\mathrm{state}}[s]
$$

$$
\operatorname{Den}_{\mathrm{far}}
=
\sum_s
\exp(S_{\mathrm{far},s}) L_{\mathrm{state}}[s]
$$

$$
O_{\mathrm{base}}
=
\frac{
  \operatorname{Num}_{\mathrm{far}}
}{
  \operatorname{Den}_{\mathrm{far}}
}
$$

GD residual 再从 `S_far` 里选若干 code 读取 `M_state` 做残差修正:

```python
top_idx = torch.topk(S_far.float(), k=read_topk, dim=-1).indices
```

这就是为什么 `logf_all` 的连续小差异会被放大: 它先改变远程 state 和 score, 后面 `topk` 又把连续 score 差异变成离散候选差异.

### 5. 本轮 probe 对这条链路的定位

本轮 step 0 观察到:

```text
phase1/q_all, k_all, v_all, K_q_all, Delta_all, W_all: match
state_build/logf_all: mismatch
state_build/G_state, L_state, M_state: mismatch
phase2_read/top_idx, forward/preds, forward/loss: still match at step 0
```

所以当前结论是:

```text
两边一开始写入 memory 的主要内容相同, 但是“记多久, 忘多快”的 decay gate 先出现极小数值差异.
这个差异进入 G/L/M state 后逐步传播, 再被 phase2 score 和 read top-k 放大.
```

## 对下一步的含义

不建议继续补 4 epoch 或继续只做 dropout ablation. 当前更有价值的下一步是围绕 `fox_gate_logf` 做最小对照:

- `gate-fp64-shadow`: 不改变训练输出, 额外用 CPU 或 fp64 shadow 计算 `fox_gate_proj/logsigmoid` 摘要, 判断两机差异是否来自 GPU linear/logsigmoid.
- `gate-fp32-ref-path`: 只把 `fox_gate_logf` 切到更朴素的 reference path, 看第一处分叉是否推迟.
- `logf-rounding/guard` 小实验: 对 `logf_all` 做轻量量化或稳定化, 看 read top-k/preds 分叉是否明显推迟.

这不是最终解决方案, 但已经把定位从“Flash-VQG mixer 内部”缩小到“layer 1 state-build gate/logf 连续值路径先分叉, 后续 GD residual state/read/top-k 放大”.

## 产物

- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/trace-summary.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/cross-machine-trace-comparison.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/preflight-summary.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/source-manifest.csv`
- `docs/artifacts/20260628-03-flash-vqg-mixer-divergence-probe/metadata.json`
