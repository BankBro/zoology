# `clr_delta_v1` 完整算法整理

下面是一版**完整、连贯、符号先定义后使用**的 `clr_delta_v1` 算法整理稿。

这版按“**目标 → 符号 → 状态 → 训练流程 → 推理流程 → 伪代码**”的顺序组织，并吸收前面讨论中最关键的修正：

- **保留**当前 `clr_v1` 的 VQ / centroid / FoX / local+remote 合并骨架；
- **保留** remote 读接口
  \[
  g+M\alpha,\qquad L+b^\top\alpha
  \]
  不改；
- **保留** block 训练时“query 只读 block 开始前 snapshot，block 内更新只供下一个 block 用”的语义；
- **把 residual 分子状态改成 Delta memory**，其目标直接取真实 value，这和 DeltaNet / Gated DeltaNet 的 key\(\to\)value 在线回归是一致的；
- **把 residual 分母状态也写成 Delta 模板，但不把 target 伪装成已经严格推出的闭式**。所以这里显式保留一个待定义的标量目标 \(y^{den}\)，这是“严格不越界”的版本。

---

## 0. 算法目标

我们要实现的不是从零开始的新模型，而是当前 `clr_v1` 的一个**结构性升级版**：

\[
\boxed{\texttt{clr\_delta\_v1}}
\]

它的目标是：

1. **保留当前 `clr_v1` 的 coarse 路由骨架**：  
   VQ codebook、centroid coarse logit、FoX forget 边界项、dense remote read 都保留。
2. **保留 local / remote 分工**：  
   local 负责窗口内 exact mixing，remote 负责 local window 之前的压缩历史。
3. **把 residual 状态从 additive 变成 Delta memory**：  
   当前 `clr_v1` 的 residual 状态是 additive 的 `R_state` 与 `a_state`；新版本把它们替换成 Delta 风格的 `M_state` 与 `b_state`，以提高同一 code 桶内的精细记忆能力。
4. **保留 block 训练语义**：  
   当前 block 的 query 只读取该 block 开始前的 remote snapshot；block 内顺序更新 working state，只用来生成下一个 block 的入口状态。

---

## 1. 符号与索引

先一次性把后面会用到的符号定义完。

### 1.1 维度

- \(B\)：batch size
- \(H\)：attention head 数
- \(T\)：pad 之后的序列长度
- \(C\)：每个 block 的长度，也就是 `block_len`
- \(N\)：block 数，定义为
  \[
  N = T / C
  \]
- \(S\)：codebook 中的 code 数，也就是 `num_codebook_vectors`
- \(d_k\)：每个 head 的 query / key 维度
- \(d_v\)：每个 head 的 value 维度
- \(r\)：每个 code 的 residual basis rank，也就是 `fox_clr_rank`

### 1.2 索引方式

为了避免混乱，整个算法同时使用两套索引。

#### 全局 token 索引
第 \(t\) 个 token，记作
\[
t \in \{1,\dots,T\}
\]

#### 分块索引
第 \(n\) 个 block 中第 \(i\) 个 token，记作
\[
(n,i),\qquad n\in\{1,\dots,N\},\ i\in\{1,\dots,C\}
\]

两者关系是
\[
t=(n-1)C+i
\]

后面如果写 \(q_t\) 与 \(q_{n,i}\)，表示的是同一个 query，只是索引方式不同。

### 1.3 输入与投影

对每个 token \(t\)，输入 hidden state 为
\[
x_t\in\mathbb{R}^{d_{\text{model}}}
\]

经过当前 attention 模块的投影后，得到

\[
q_t\in\mathbb{R}^{d_k},\qquad
k_t\in\mathbb{R}^{d_k},\qquad
v_t\in\mathbb{R}^{d_v}
\]

分别表示 query、key、value。

---

## 2. VQ 与 residual 几何结构

这部分沿用当前 `clr_v1` 的几何定义，不改。

### 2.1 codebook centroid

共有 \(S\) 个 code，第 \(s\) 个 code 的 centroid 记为

\[
c_s\in\mathbb{R}^{d_k},\qquad s\in\{1,\dots,S\}
\]

### 2.2 token 的 code id 与量化 key

token \(t\) 的 code id 记为

\[
z_t\in\{1,\dots,S\}
\]

于是它对应的量化 key 是

\[
\hat{k}_t = c_{z_t}
\]

### 2.3 key residual

token \(t\) 的 residual 定义为

\[
r_t = k_t - c_{z_t}
\]

### 2.4 每个 code 的 residual basis

对每个 code \(s\)，维护一个 residual basis

\[
B_s\in\mathbb{R}^{d_k\times r}
\]

它把该 code 桶内的 residual 子空间压到 \(r\) 维。

### 2.5 token 的 residual 坐标

token \(t\) 在自己所属 code 的 basis 上的 residual 坐标定义为

\[
h_t = B_{z_t}^\top r_t
      = B_{z_t}^\top (k_t-c_{z_t})
\]

其中

\[
h_t\in\mathbb{R}^r
\]

### 2.6 query 对某个 code 的 residual 查询向量

当 query \(q_t\) 去读第 \(s\) 个 code 的 remote state 时，需要把 query 投到该 code 的 basis 上：

\[
\alpha_{t,s} = B_s^\top q_t
\]

其中

\[
\alpha_{t,s}\in\mathbb{R}^r
\]

这就是 remote residual read 时的查询向量。

---

## 3. FoX forget 与 remote 边界量

remote 读仍然沿用 FoX 风格的 forget 边界偏置。

### 3.1 token 级 forget gate

对每个 token \(t\)，先得到 forget 对数值

\[
\log f_t
\]

并定义

\[
f_t = \exp(\log f_t),\qquad 0<f_t\le 1
\]

### 3.2 累积 forget 前缀

定义累积量

\[
\phi_t = \sum_{\tau=1}^{t}\log f_\tau
\]

于是

\[
\exp(\phi_t-\phi_u)
=
\prod_{\tau=u+1}^{t}f_\tau
\]

它表示从位置 \(u\) 衰减到位置 \(t\) 的累计 forget 因子。这里统一记作 \(\phi_t\)，避免和 codebook centroid \(c_s\) 混淆。

### 3.3 local / remote 分工与 remote 历史边界

设 local 覆盖最近 \(w\) 个 block，其中

\[
w=\texttt{local\_num\_blocks}
\]

那么对当前 block \(n\)，remote 只负责 **local window 之前** 的历史。定义

\[
e(n)
\]

为 block \(n\) 的 remote 历史边界位置，也就是“local window 之前最后一个 token 的全局时间位置”。对应的 forget 边界累计量是

\[
\phi_{e(n)}
\]

如果 block 太靠前、remote 历史为空，就把 \(\phi_{e(n)}\) 视为 0。

---

## 4. 当前 `clr_v1` 与新 `clr_delta_v1` 的关系

当前 `clr_v1` 在每个 block、每个 code 上维护四组 remote state：

- coarse 分子状态 \(g_{n,s}\in\mathbb{R}^{d_v}\)
- residual 分子状态 \(R_{n,s}\in\mathbb{R}^{d_v\times r}\)
- coarse 分母状态 \(L_{n,s}\in\mathbb{R}\)
- residual 分母状态 \(a_{n,s}\in\mathbb{R}^{r}\)

它们进入 remote 读时的形式是

\[
g_{n,s}+R_{n,s}\alpha_{t,s},
\qquad
L_{n,s}+a_{n,s}^\top\alpha_{t,s}
\]

并对所有 code 做 dense 聚合。

`clr_delta_v1` 的改动只有一条：

\[
R_{n,s},a_{n,s}
\quad\Longrightarrow\quad
M_{n,s},b_{n,s}
\]

即把两条 residual additive state 改成 Delta state；coarse 的 \(g,L\) 保持 additive 不变。

---

## 5. `clr_delta_v1` 的状态定义

对每个 block \(n\)、每个 code \(s\)，定义以下四组 remote snapshot state：

### 5.1 coarse 状态

#### coarse 分子状态
\[
g_{n,s}\in\mathbb{R}^{d_v}
\]

#### coarse 分母状态
\[
L_{n,s}\in\mathbb{R}
\]

这两组状态仍然按 additive 方式递推。

### 5.2 residual Delta 状态

#### residual 分子 Delta 状态
\[
M_{n,s}\in\mathbb{R}^{d_v\times r}
\]

含义：给定 residual 查询向量 \(\alpha\)，输出 residual value 修正 \(M_{n,s}\alpha\)。

#### residual 分母 Delta 状态
\[
b_{n,s}\in\mathbb{R}^{r}
\]

含义：给定 residual 查询向量 \(\alpha\)，输出 residual denominator 修正 \(b_{n,s}^\top\alpha\)。

---

## 6. remote 读公式

这部分沿用当前 `clr_v1` 的读接口，只是把 residual 状态换成 \(M,b\)。

### 6.1 coarse base logit

对 block \(n\) 中第 \(i\) 个 query \(q_{n,i}\) 和 code \(s\)，定义

\[
\ell^{base}_{n,i,s}
=
q_{n,i}^\top c_s + (\phi_{n,i}-\phi_{e(n)})
\]

### 6.2 residual 查询向量

\[
\alpha_{n,i,s}=B_s^\top q_{n,i}
\]

### 6.3 每个 code 的 remote 分子 / 分母贡献

#### remote 分子贡献
\[
\mathrm{Num}^{remote}_{n,i,s}
=
\exp(\ell^{base}_{n,i,s})
\bigl(g_{n,s}+M_{n,s}\alpha_{n,i,s}\bigr)
\]

#### remote 分母贡献
\[
\mathrm{Den}^{remote}_{n,i,s}
=
\exp(\ell^{base}_{n,i,s})
\bigl(L_{n,s}+b_{n,s}^\top\alpha_{n,i,s}\bigr)
\]

### 6.4 dense code 聚合

当前 CLR 路径是 dense remote read，不做 top-k code 筛选，因此直接对全部 \(S\) 个 code 求和：

\[
\boxed{
\mathrm{Num}^{remote}_{n,i}
=
\sum_{s=1}^{S}
\exp(\ell^{base}_{n,i,s}-m_{n,i})
\bigl(g_{n,s}+M_{n,s}\alpha_{n,i,s}\bigr)
}
\]

\[
\boxed{
\mathrm{Den}^{remote}_{n,i}
=
\sum_{s=1}^{S}
\exp(\ell^{base}_{n,i,s}-m_{n,i})
\bigl(L_{n,s}+b_{n,s}^\top\alpha_{n,i,s}\bigr)
}
\]

其中 \(m_{n,i}\) 是和 local 路径共享的 row-wise 数值稳定化 shift。

---

## 7. block 训练语义

这是整个算法最容易写乱的地方。

### 7.1 block 开始前 snapshot

进入 block \(n\) 之前，每个 code \(s\) 都有一份 remote 历史快照：

\[
g_{n,s},\quad L_{n,s},\quad M_{n,s},\quad b_{n,s}
\]

它们只包含 **当前 block 的 local window 之外** 的更早历史。

### 7.2 当前 block 的 query 读什么

对 block \(n\) 内所有 query，remote path **都只读 block 开始前的 snapshot**：

\[
g_{n,s},\;L_{n,s},\;M_{n,s},\;b_{n,s}
\]

block 内更近的依赖继续由 local path 负责。因此，block 内顺序更新出来的新状态不会反过来参与同一个 block 的 remote read；它们只用于生成下一个 block 的入口状态。

---

## 8. block 内 working state

为了从 block \(n\) 递推出 block \(n+1\) 的入口状态，我们在 block \(n\) 内维护 working state：

\[
\tilde g_s^{(0)} = g_{n,s},\qquad
\tilde L_s^{(0)} = L_{n,s},\qquad
\tilde M_s^{(0)} = M_{n,s},\qquad
\tilde b_s^{(0)} = b_{n,s}
\]

这里：

- 上标 \((i)\) 表示“处理完 block 内前 \(i\) 个 token 之后”的状态；
- 波浪号 \(\tilde{\cdot}\) 表示“block 内递推用的 working state”。

---

## 9. block 内第 \(i\) 个 token 的递推

现在看 block \(n\) 中第 \(i\) 个 token。

### 9.1 当前 token 的 code 与 residual 坐标

定义当前 token 的 code 为

\[
s_{n,i} = z_{n,i}
\]

定义它的 residual 坐标为

\[
h_{n,i}
=
B_{s_{n,i}}^\top\bigl(k_{n,i}-c_{s_{n,i}}\bigr)
\]

### 9.2 当前 token 的 forget 因子

定义

\[
f_{n,i}=\exp(\log f_{n,i})
\]

### 9.3 先做 forget

对所有 code \(s\)，先做一次 forget：

\[
\tilde g_s^{(i-\frac12)} = f_{n,i}\tilde g_s^{(i-1)}
\]
\[
\tilde L_s^{(i-\frac12)} = f_{n,i}\tilde L_s^{(i-1)}
\]
\[
\tilde M_s^{(i-\frac12)} = f_{n,i}\tilde M_s^{(i-1)}
\]
\[
\tilde b_s^{(i-\frac12)} = f_{n,i}\tilde b_s^{(i-1)}
\]

这里 \(i-\frac12\) 表示“忘掉旧状态、但还没写当前 token”。

### 9.4 再写 coarse additive 状态

当前 token 只写自己的 code 桶 \(s_{n,i}\)。

#### coarse 分子写入
\[
\tilde g_{s_{n,i}}^{(i)}
=
\tilde g_{s_{n,i}}^{(i-\frac12)} + v_{n,i}
\]

#### coarse 分母写入
\[
\tilde L_{s_{n,i}}^{(i)}
=
\tilde L_{s_{n,i}}^{(i-\frac12)} + 1
\]

对其他 code \(s\neq s_{n,i}\)，保持：

\[
\tilde g_s^{(i)}=\tilde g_s^{(i-\frac12)},\qquad
\tilde L_s^{(i)}=\tilde L_s^{(i-\frac12)}
\]

---

## 10. residual Delta 写入

这是新算法的核心。

### 10.1 residual 分子的旧预测

在当前 code 上，先用 working state 读出 residual 分子的旧预测：

\[
v^{old}_{n,i}
=
\tilde M_{s_{n,i}}^{(i-\frac12)}h_{n,i}
\]

其中

\[
v^{old}_{n,i}\in\mathbb{R}^{d_v}
\]

### 10.2 residual 分母的旧预测

同理，定义 residual 分母的旧预测：

\[
d^{old}_{n,i}
=
\bigl(\tilde b_{s_{n,i}}^{(i-\frac12)}\bigr)^\top h_{n,i}
\]

其中

\[
d^{old}_{n,i}\in\mathbb{R}
\]

### 10.3 写入 gate 与归一化步长

定义当前 token 的 residual 写入强度：

\[
\beta_{n,i}\in(0,1)
\]

并定义归一化步长：

\[
\eta_{n,i}
=
\frac{\beta_{n,i}}{\|h_{n,i}\|_2^2+\varepsilon}
\]

这样可以把更新强度和 \(h_{n,i}\) 的尺度解耦。

### 10.4 residual 分子的目标与误差

这部分可以直接沿用 DeltaNet / Gated DeltaNet 的在线回归语义：状态在输入当前 key 时，应该输出目标 value。DeltaNet 的核心就是最小化 \(\frac12\|Sk-v\|^2\) 并做一步 delta 更新。

因此这里令 residual 分子目标为

\[
\tilde M_{s_{n,i}}^{(i-\frac12)} h_{n,i} \approx v_{n,i}
\]

对应误差定义为

\[
e^{num}_{n,i}=v_{n,i}-v^{old}_{n,i}
\]

### 10.5 residual 分母的目标与误差

这里必须**谨慎处理**。

当前旧版 `clr_v1` 的 residual 分母项是 \(a^\top\alpha\)，它的职责是作为 coarse 分母 \(L\) 之外的一个 residual denominator correction。  
但是，Gated DeltaNet / DeltaNet 本体直接给出的，是“key \(\to\) value”在线回归更新；它并**没有**直接告诉我们 softmax / FoX 风格 residual denominator correction 的唯一正确 target 是什么。

因此，为了写出**严格不越界**的版本，这里不把 target 伪装成已经被严格推出的闭式，而是显式引入一个待定义的标量目标：

\[
y^{den}_{n,i}\in\mathbb{R}
\]

它表示：

> 当前 residual key \(h_{n,i}\) 在 denominator 通道上希望对齐的目标值。

于是定义 residual 分母误差为

\[
e^{den}_{n,i}
=
y^{den}_{n,i}-d^{old}_{n,i}
\]

这里要明确：

- \(y^{den}_{n,i}\) 是一个**设计接口**；
- 它需要由原始 `clr_v1` 分母校正项的统计语义进一步具体化；
- 在这一版算法整理里，不把它硬写成 \(\|h_{n,i}\|^2\) 或 \(1\)，以避免把启发式 surrogate 冒充成严格推导结论。

### 10.6 residual 分子 Delta 更新

\[
\boxed{
\tilde M_{s_{n,i}}^{(i)}
=
\tilde M_{s_{n,i}}^{(i-\frac12)}
+
\eta_{n,i}\,e^{num}_{n,i}\,h_{n,i}^\top
}
\]

展开后：

\[
\boxed{
\tilde M_{s_{n,i}}^{(i)}
=
\tilde M_{s_{n,i}}^{(i-\frac12)}
+
\frac{\beta_{n,i}}{\|h_{n,i}\|_2^2+\varepsilon}
\Bigl(
v_{n,i}-\tilde M_{s_{n,i}}^{(i-\frac12)}h_{n,i}
\Bigr)h_{n,i}^\top
}
\]

### 10.7 residual 分母 Delta 更新

\[
\boxed{
\tilde b_{s_{n,i}}^{(i)}
=
\tilde b_{s_{n,i}}^{(i-\frac12)}
+
\eta_{n,i}\,e^{den}_{n,i}\,h_{n,i}
}
\]

展开后：

\[
\boxed{
\tilde b_{s_{n,i}}^{(i)}
=
\tilde b_{s_{n,i}}^{(i-\frac12)}
+
\frac{\beta_{n,i}}{\|h_{n,i}\|_2^2+\varepsilon}
\Bigl(
y^{den}_{n,i}
-
\bigl(\tilde b_{s_{n,i}}^{(i-\frac12)}\bigr)^\top h_{n,i}
\Bigr)h_{n,i}
}
\]

这就是“paired Delta residual memory”的严格版本：分子有明确 target，分母有明确 Delta 模板，但其 target 保留为接口。

---

## 11. block 结束后的状态交接

当 block \(n\) 的 \(C\) 个 token 都处理完后，把 working state 作为 block \(n+1\) 的入口状态：

\[
g_{n+1,s} = \tilde g_s^{(C)}
\]
\[
L_{n+1,s} = \tilde L_s^{(C)}
\]
\[
M_{n+1,s} = \tilde M_s^{(C)}
\]
\[
b_{n+1,s} = \tilde b_s^{(C)}
\]

这样就完成了按 block 的远程状态递推。

---

## 12. local + remote 合并输出

设 local path 为 block \(n\) 中第 \(i\) 个 query 给出

\[
\mathrm{Num}^{local}_{n,i},\qquad \mathrm{Den}^{local}_{n,i}
\]

remote path 则按第 6 节给出

\[
\mathrm{Num}^{remote}_{n,i},\qquad \mathrm{Den}^{remote}_{n,i}
\]

最终输出为

\[
\boxed{
o_{n,i}
=
\frac{
\mathrm{Num}^{local}_{n,i}+\mathrm{Num}^{remote}_{n,i}
}{
\mathrm{Den}^{local}_{n,i}+\mathrm{Den}^{remote}_{n,i}+\varepsilon
}
}
\]

这与当前 `clr_v1` 的“local 与 remote 的分子分母统一相加后归一化”的外部接口保持一致。

---

## 13. 训练流程

把上面所有定义串起来，得到完整训练流程。

### 输入
一段长度为 \(T\) 的序列 hidden states \(\{x_t\}_{t=1}^T\)。

### 步骤 1：pad 与分块
把序列 pad 到 \(C\) 的整数倍，并 reshape 成 \(N\) 个 block。

### 步骤 2：投影
对所有 token 计算 \(q_t,k_t,v_t\)。

### 步骤 3：VQ 与 residual 几何
计算 \(z_t,\hat{k}_t,r_t,h_t\)，并为每个 code \(s\) 准备 basis \(B_s\)。

### 步骤 4：FoX forget 前缀
计算 \(f_t,\phi_t\)，以及每个 block 的 remote 边界量 \(\phi_{e(n)}\)。

### 步骤 5：local path
在每个 block 内，计算 local window 对应的
\[
\mathrm{Num}^{local}_{n,i},\ \mathrm{Den}^{local}_{n,i}
\]

### 步骤 6：remote snapshot 读
对 block \(n\) 内所有 query，只使用
\[
g_{n,s},L_{n,s},M_{n,s},b_{n,s}
\]
计算 remote 读出。

### 步骤 7：block 内顺序更新 working state
按 \(i=1,\dots,C\) 顺序，对当前 token：
1. 做 forget；
2. 写入 coarse additive 状态；
3. 按 Delta 公式写入 residual 分子状态 \(M\)；
4. 按 Delta 模板写入 residual 分母状态 \(b\)。

### 步骤 8：状态交接
把 block 内 working state 的末态变成下一个 block 的入口 snapshot。

### 步骤 9：合并输出
对每个 query，使用
\[
o_{n,i}
=
\frac{
\mathrm{Num}^{local}_{n,i}+\mathrm{Num}^{remote}_{n,i}
}{
\mathrm{Den}^{local}_{n,i}+\mathrm{Den}^{remote}_{n,i}+\varepsilon
}
\]

---

## 14. 推理流程

推理与训练的外部接口一致，但使用方式更简单：

1. 维护当前 remote snapshot  
   \[
   g_s,\ L_s,\ M_s,\ b_s
   \]
2. 对新 token 计算 \(q,k,v,z,h\)；
3. 用当前 snapshot 做 remote 读；
4. 用 local cache / local window 做 local 读；
5. 合并得到输出；
6. 再用当前 token 更新 working state，并在 block 边界处提交为新的 snapshot。

也就是说：

- **读**始终读“当前时刻之前已经完成的 snapshot”；
- **写**始终只推进未来状态。

---

## 15. 伪代码

```text
Algorithm clr_delta_v1

Input:
  x_1:T
Hyper-parameters:
  block length C
  local_num_blocks = w
  codebook size S
  residual rank r

State per code s at block boundary:
  g_s in R^{d_v}
  L_s in R
  M_s in R^{d_v x r}
  b_s in R^{r}

1. Pad sequence to multiple of C, split into N blocks.
2. Compute q_t, k_t, v_t for all tokens.
3. Quantize each key:
     z_t = code_id(k_t)
     c_{z_t} = centroid
     h_t = B_{z_t}^T (k_t - c_{z_t})
4. Compute forget gates f_t and cumulative prefixes phi_t.
5. For each block n = 1..N:
     5.1 Read-only snapshot for this block:
         {g_{n,s}, L_{n,s}, M_{n,s}, b_{n,s}}_{s=1}^S
     5.2 For each query (n,i):
         alpha_{n,i,s} = B_s^T q_{n,i}
         l_base_{n,i,s} = q_{n,i}^T c_s + (phi_{n,i} - phi_{e(n)})
         Num_remote_{n,i} = sum_s exp(l_base_{n,i,s} - m_{n,i}) (g_{n,s} + M_{n,s} alpha_{n,i,s})
         Den_remote_{n,i} = sum_s exp(l_base_{n,i,s} - m_{n,i}) (L_{n,s} + b_{n,s}^T alpha_{n,i,s})
         Combine with local path to get output.
     5.3 Initialize working states:
         g~_s^(0)=g_{n,s}, L~_s^(0)=L_{n,s}, M~_s^(0)=M_{n,s}, b~_s^(0)=b_{n,s}
     5.4 For i = 1..C:
         s = z_{n,i}, h = h_{n,i}, f = f_{n,i}
         For all codes u:
             g~_u <- f g~_u
             L~_u <- f L~_u
             M~_u <- f M~_u
             b~_u <- f b~_u
         Coarse write on code s:
             g~_s <- g~_s + v_{n,i}
             L~_s <- L~_s + 1
         Residual Delta write:
             v_old = M~_s h
             d_old = b~_s^T h
             eta = beta_{n,i} / (||h||^2 + eps)
             M~_s <- M~_s + eta (v_{n,i} - v_old) h^T
             b~_s <- b~_s + eta (y^{den}_{n,i} - d_old) h
     5.5 Commit next block snapshot:
         g_{n+1,s}=g~_s^(C), L_{n+1,s}=L~_s^(C), M_{n+1,s}=M~_s^(C), b_{n+1,s}=b~_s^(C)
```

---

## 16. 一句话总结

\[
\boxed{
\texttt{clr\_delta\_v1}
=
\text{“保留 clr\_v1 的 coarse VQ/FoX/local-remote 骨架，}
\text{把 residual additive state 改成按 block 递推的 paired Delta memory。”}
}
\]

更具体地说，它就是：

- 用 VQ codebook 提供 coarse 路由；
- 用每个 code 的 residual basis 提供桶内细粒度坐标；
- remote 读时仍然走
  \[
  g+M\alpha,\qquad L+b^\top\alpha
  \]
  的接口；
- coarse 状态 \(g,L\) 继续 additive；
- residual 状态 \(M,b\) 在 block 内按 token 顺序做 forget + Delta 更新；
- 当前 block 的 query 只读 block 开始前的 snapshot；
- 最后 local 与 remote 的分子分母相加并归一化输出。
