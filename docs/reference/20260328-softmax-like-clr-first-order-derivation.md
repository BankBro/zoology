# softmax-like CLR 一阶近似推导

先说明边界:

> 下面的最终公式, 是把文档里的 `Code-local Residual Correction` 分解, 与 GLA 的 gated/chunkwise 递推形式拼起来得到的一个可实现版本.
>
> 其中:
>
> - $k_j = c_{z_j} + r_j$, $r_j \approx B_s h_j$, 以及 $S_t^\star = \sum u_j k_j^\top$ 这部分, 来自前面的设计文档.
> - "带遗忘的 recurrence 可以做成 chunkwise, 并用 $(\Lambda, \Gamma, \gamma)$ 传播 chunk 内/块间状态" 这部分, 来自 GLA 的推导框架.

## 1. 目标: 我们到底想近似什么

先只看 remote memory. 历史第 $j$ 个 token 产生:

- key: $k_j \in \mathbb{R}^{d_k}$
- value: $v_j \in \mathbb{R}^{d_v}$

在当前第一版里, 先取:

$$
u_j = v_j
$$

也就是不先引入 Delta-style pseudo-value. 前面的设计文档把更一般的目标写成:

$$
S_t^\star = \sum_{j \le t} u_j k_j^\top
$$

这只是为了以后兼容 DeltaNet 风格. 当前 v1 里可以直接令 $u = v$.

## 2. Code-local residual decomposition

VQ 把每个 key 分到某个 code:

$$
z_j = \mathrm{VQ}(k_j)
$$

记第 $s$ 个 code 的 centroid 为 $c_s$. 当 $z_j = s$ 时, 把 key 分解成:

$$
k_j = c_s + r_j
$$

其中:

- $c_s$: 桶 $s$ 的粗中心
- $r_j$: 第 $j$ 个 key 在桶 $s$ 内的 residual

这是 `Code-local Residual Correction` 的起点. 前面的文档里已经把这个分解和由此得到的 centroid / residual 两项写清楚了.

## 3. 加入 forgetting 后的精确 softmax-like remote

设当前 query 为 $q_t$. 设从历史位置 $j$ 到当前位置 $t$ 的累计遗忘因子为:

$$
F_{t,j} = \prod_{\ell = j + 1}^{t} f_\ell,
\qquad
0 < f_\ell \le 1
$$

这里先写成最简单的标量 gate. 如果以后想扩成 per-head / per-channel gate, 只要把下面的标量乘法改成逐元素广播即可. GLA 的关键点正是: 带 gate 的 recurrence 仍然可以写成 chunkwise 形式, 并用 cumulative decay 做并行训练.

### 3.1 全局精确式

精确的 remote 分子 / 分母是:

$$
\mathrm{Num}_t^{\mathrm{exact}}
=
\sum_{j < t}
F_{t,j} \exp(q_t^\top k_j) v_j
$$

$$
\mathrm{Den}_t^{\mathrm{exact}}
=
\sum_{j < t}
F_{t,j} \exp(q_t^\top k_j)
$$

### 3.2 按桶 $s$ 分组

对固定桶 $s$, 把 $k_j = c_s + r_j$ 代入:

$$
q_t^\top k_j = q_t^\top c_s + q_t^\top r_j
$$

于是桶 $s$ 的精确贡献变成:

$$
\mathrm{Num}_{t,s}^{\mathrm{exact}}
=
\exp(q_t^\top c_s)
\sum_{j : z_j = s}
F_{t,j} \exp(q_t^\top r_j) v_j
$$

$$
\mathrm{Den}_{t,s}^{\mathrm{exact}}
=
\exp(q_t^\top c_s)
\sum_{j : z_j = s}
F_{t,j} \exp(q_t^\top r_j)
$$

如果想加 code-level bias / position bias, 也只需定义:

$$
\ell_{t,s}^{\mathrm{base}} = q_t^\top c_s + b_{t,s}
$$

然后把外面的 $\exp(q_t^\top c_s)$ 换成 $\exp(\ell_{t,s}^{\mathrm{base}})$ 即可.

## 4. 低维 residual 假设

为了不保留桶内所有历史 residual, 做 code-local 低维近似:

$$
r_j \approx B_s h_j,
\qquad
B_s \in \mathbb{R}^{d_k \times r},
\quad
h_j \in \mathbb{R}^r,
\quad
r \ll d_k
$$

这里:

- $B_s$: 桶 $s$ 的局部 basis
- $h_j$: 第 $j$ 个历史项在该 basis 下的 residual coordinate

于是:

$$
q_t^\top r_j
\approx
q_t^\top B_s h_j
=
(B_s^\top q_t)^\top h_j
$$

定义:

$$
\alpha_{t,s} := B_s^\top q_t \in \mathbb{R}^r
$$

那么桶 $s$ 的精确式可以改写为:

$$
\mathrm{Num}_{t,s}^{\mathrm{exact}}
=
\exp(\ell_{t,s}^{\mathrm{base}})
\sum_{j : z_j = s}
F_{t,j} \exp(\alpha_{t,s}^\top h_j) v_j
$$

$$
\mathrm{Den}_{t,s}^{\mathrm{exact}}
=
\exp(\ell_{t,s}^{\mathrm{base}})
\sum_{j : z_j = s}
F_{t,j} \exp(\alpha_{t,s}^\top h_j)
$$

## 5. 一阶近似: 真正近似发生在哪里

近似发生在桶内这一项:

$$
\exp(\alpha_{t,s}^\top h_j)
$$

做一阶展开:

$$
\exp(x) \approx 1 + x
$$

令 $x = \alpha_{t,s}^\top h_j$, 得到:

$$
\exp(\alpha_{t,s}^\top h_j)
\approx
1 + \alpha_{t,s}^\top h_j
$$

这一步把"桶内每个历史项都要根据当前 query 重新做指数重加权"的精确读法, 压成了低阶统计量.

## 6. 定义一阶统计量

把一阶展开代入桶内精确式, 先看分子:

$$
\sum_{j : z_j = s}
F_{t,j}
\left(1 + \alpha_{t,s}^\top h_j\right)
v_j
=
\sum_{j : z_j = s} F_{t,j} v_j
+
\sum_{j : z_j = s} F_{t,j} v_j h_j^\top \alpha_{t,s}
$$

于是定义:

$$
g_{s,t}
:=
\sum_{j : z_j = s} F_{t,j} v_j
\in
\mathbb{R}^{d_v}
$$

$$
R_{s,t}
:=
\sum_{j : z_j = s} F_{t,j} v_j h_j^\top
\in
\mathbb{R}^{d_v \times r}
$$

含义是:

- $g_{s,t}$: 桶 $s$ 的 decayed coarse value sum
- $R_{s,t}$: 桶 $s$ 的 decayed residual-correction memory

再看分母:

$$
\sum_{j : z_j = s}
F_{t,j}
\left(1 + \alpha_{t,s}^\top h_j\right)
=
\sum_{j : z_j = s} F_{t,j}
+
\sum_{j : z_j = s} F_{t,j} h_j^\top \alpha_{t,s}
$$

定义:

$$
L_{s,t}
:=
\sum_{j : z_j = s} F_{t,j}
\in
\mathbb{R}
$$

$$
a_{s,t}
:=
\sum_{j : z_j = s} F_{t,j} h_j
\in
\mathbb{R}^r
$$

含义是:

- $L_{s,t}$: 桶 $s$ 的 decayed mass
- $a_{s,t}$: 桶 $s$ 的 decayed first residual moment

于是, 桶 $s$ 的一阶近似分子 / 分母就是:

$$
\boxed{
\mathrm{Num}_{t,s}
\approx
\exp(\ell_{t,s}^{\mathrm{base}})
\left[g_{s,t} + R_{s,t} \alpha_{t,s}\right]
}
$$

$$
\boxed{
\mathrm{Den}_{t,s}
\approx
\exp(\ell_{t,s}^{\mathrm{base}})
\left[L_{s,t} + a_{s,t}^\top \alpha_{t,s}\right]
}
$$

其中 $\alpha_{t,s} = B_s^\top q_t$.

## 7. 全部 code 聚合后的 remote 分支

如果做 dense 聚合, 就对全部 code 求和. 如果做 E7 / top-k, 则只对候选集合 $\mathcal{S}_t$ 求和.

为了数值稳定, 设:

$$
m_t = \max_{s \in \mathcal{S}_t} \ell_{t,s}^{\mathrm{base}}
$$

则 remote 分子 / 分母写成:

$$
\boxed{
\mathrm{Num}_t^{\mathrm{remote}}
=
\sum_{s \in \mathcal{S}_t}
\exp(\ell_{t,s}^{\mathrm{base}} - m_t)
\left[g_{s,t} + R_{s,t}(B_s^\top q_t)\right]
}
$$

$$
\boxed{
\mathrm{Den}_t^{\mathrm{remote}}
=
\sum_{s \in \mathcal{S}_t}
\exp(\ell_{t,s}^{\mathrm{base}} - m_t)
\left[L_{s,t} + a_{s,t}^\top(B_s^\top q_t)\right]
}
$$

这是 `softmax-like CLR v1` 的核心公式.

## 8. 和 local 分支的统一合并

现在 local 分支本来就是 Num / Den 形式, 所以最终输出最自然的合并方式是:

$$
\boxed{
o_t
=
\frac{
\mathrm{Num}_t^{\mathrm{local}} + \mathrm{Num}_t^{\mathrm{remote}}
}{
\mathrm{Den}_t^{\mathrm{local}} + \mathrm{Den}_t^{\mathrm{remote}} + \varepsilon
}
}
$$

这一步和前面 insist 的"remote 也必须写成分子 / 分母再和 local 统一合并"是一致的.

## 9. 递推形式: 带遗忘的一阶状态更新

因为:

$$
F_{t,j} = f_t F_{t-1,j}
$$

所以 $g$, $R$, $L$, $a$ 都满足固定状态递推. 若当前 token $t$ 落在桶 $s_t = z_t$, 并定义:

$$
h_t := B_{s_t}^\top (k_t - c_{s_t}),
\qquad
v_t \in \mathbb{R}^{d_v}
$$

则对所有桶 $s$ 有:

$$
\boxed{
g_{s,t}
=
f_t g_{s,t-1}
+
\mathbf{1}[s = s_t] \, v_t
}
$$

$$
\boxed{
R_{s,t}
=
f_t R_{s,t-1}
+
\mathbf{1}[s = s_t] \, v_t h_t^\top
}
$$

$$
\boxed{
L_{s,t}
=
f_t L_{s,t-1}
+
\mathbf{1}[s = s_t]
}
$$

$$
\boxed{
a_{s,t}
=
f_t a_{s,t-1}
+
\mathbf{1}[s = s_t] \, h_t
}
$$

这就是带 forgetting 的一阶状态版 CLR recurrence. GLA 的 recurrent form 也是"先对旧状态乘 gate, 再加新输入", 并且正因为这个结构, 才能推导 chunkwise parallel form.

## 10. 分块训练: chunkwise 版本

下面给出最终要用的 chunkwise 训练公式.

设序列被切成 $N = L / C$ 个 chunk, 每个 chunk 长度为 $C$. 第 $b$ 个 chunk 中第 $i$ 个 token 记为 $(b,i)$, 其中 $i = 1, \dots, C$.

### 10.1 chunk 内 decay 量

定义 chunk 内前缀衰减:

$$
\lambda_{b,i}
:=
\prod_{u = 1}^{i} f_{b,u}
$$

chunk 总衰减:

$$
\gamma_b := \lambda_{b,C}
$$

以及 suffix-style 权重:

$$
\Gamma_{b,i}
:=
\frac{\gamma_b}{\lambda_{b,i}}
$$

这和 GLA 里 $(\Lambda, \Gamma, \gamma)$ 的角色一致:

- $\Lambda$: 从 chunk 起点传播到当前位置
- $\Gamma$: 从当前位置传播到 chunk 末尾
- $\gamma$: 整个 chunk 的总衰减

### 10.2 chunk summary

对每个桶 $s$, 定义第 $b$ 个 chunk 的 summary:

$$
\bar g_b[s]
:=
\sum_{i = 1}^{C}
\mathbf{1}[z_{b,i} = s] \,
\Gamma_{b,i} \,
v_{b,i}
$$

$$
\bar R_b[s]
:=
\sum_{i = 1}^{C}
\mathbf{1}[z_{b,i} = s] \,
\Gamma_{b,i} \,
v_{b,i} h_{b,i}^\top
$$

$$
\bar L_b[s]
:=
\sum_{i = 1}^{C}
\mathbf{1}[z_{b,i} = s] \,
\Gamma_{b,i}
$$

$$
\bar a_b[s]
:=
\sum_{i = 1}^{C}
\mathbf{1}[z_{b,i} = s] \,
\Gamma_{b,i} \,
h_{b,i}
$$

这些就是第 $b$ 个 chunk 对"下一个 chunk 入口状态"的贡献.

### 10.3 inter-chunk recurrence

设第 $b$ 个 chunk 入口状态为:

$$
g_b^{\mathrm{in}}[s], \quad
R_b^{\mathrm{in}}[s], \quad
L_b^{\mathrm{in}}[s], \quad
a_b^{\mathrm{in}}[s]
$$

则第 $b$ 个 chunk 结束后, 传给下一 chunk 的状态为:

$$
\boxed{
g_{b+1}^{\mathrm{in}}[s]
=
\gamma_b g_b^{\mathrm{in}}[s] + \bar g_b[s]
}
$$

$$
\boxed{
R_{b+1}^{\mathrm{in}}[s]
=
\gamma_b R_b^{\mathrm{in}}[s] + \bar R_b[s]
}
$$

$$
\boxed{
L_{b+1}^{\mathrm{in}}[s]
=
\gamma_b L_b^{\mathrm{in}}[s] + \bar L_b[s]
}
$$

$$
\boxed{
a_{b+1}^{\mathrm{in}}[s]
=
\gamma_b a_b^{\mathrm{in}}[s] + \bar a_b[s]
}
$$

这就是 chunk 间递推. 结构上和 GLA 的"上一 chunk hidden state 乘总衰减, 再加本 chunk 累计贡献"完全同型.

### 10.4 intra-chunk states

对 chunk $b$ 中第 $r$ 个位置, 定义:

$$
\Lambda_{b,r} := \lambda_{b,r}
$$

那么桶 $s$ 在该位置的实时状态是:

$$
\boxed{
g_{b,r}[s]
=
\Lambda_{b,r} \, g_b^{\mathrm{in}}[s]
+
\sum_{i = 1}^{r}
\mathbf{1}[z_{b,i} = s] \,
\frac{\Lambda_{b,r}}{\Lambda_{b,i}} \,
v_{b,i}
}
$$

$$
\boxed{
R_{b,r}[s]
=
\Lambda_{b,r} \, R_b^{\mathrm{in}}[s]
+
\sum_{i = 1}^{r}
\mathbf{1}[z_{b,i} = s] \,
\frac{\Lambda_{b,r}}{\Lambda_{b,i}} \,
v_{b,i} h_{b,i}^\top
}
$$

$$
\boxed{
L_{b,r}[s]
=
\Lambda_{b,r} \, L_b^{\mathrm{in}}[s]
+
\sum_{i = 1}^{r}
\mathbf{1}[z_{b,i} = s] \,
\frac{\Lambda_{b,r}}{\Lambda_{b,i}}
}
$$

$$
\boxed{
a_{b,r}[s]
=
\Lambda_{b,r} \, a_b^{\mathrm{in}}[s]
+
\sum_{i = 1}^{r}
\mathbf{1}[z_{b,i} = s] \,
\frac{\Lambda_{b,r}}{\Lambda_{b,i}} \,
h_{b,i}
}
$$

这就是 chunk 内任意位置 $r$ 的 decayed 状态.

### 10.5 分块训练时该位置的 remote Num / Den

对位置 $(b,r)$, 先算:

$$
\alpha_{b,r,s} = B_s^\top q_{b,r}
$$

$$
\ell_{b,r,s}^{\mathrm{base}} = q_{b,r}^\top c_s + b_{b,r,s}
$$

设候选集合为 $\mathcal{S}_{b,r}$, 可以是 dense 或 top-k, 并定义:

$$
m_{b,r}
=
\max_{s \in \mathcal{S}_{b,r}}
\ell_{b,r,s}^{\mathrm{base}}
$$

则:

$$
\boxed{
\mathrm{Num}_{b,r}^{\mathrm{remote}}
=
\sum_{s \in \mathcal{S}_{b,r}}
\exp(\ell_{b,r,s}^{\mathrm{base}} - m_{b,r})
\left[
g_{b,r}[s] + R_{b,r}[s] \alpha_{b,r,s}
\right]
}
$$

$$
\boxed{
\mathrm{Den}_{b,r}^{\mathrm{remote}}
=
\sum_{s \in \mathcal{S}_{b,r}}
\exp(\ell_{b,r,s}^{\mathrm{base}} - m_{b,r})
\left[
L_{b,r}[s] + a_{b,r}[s]^\top \alpha_{b,r,s}
\right]
}
$$

最后输出:

$$
\boxed{
o_{b,r}
=
\frac{
\mathrm{Num}_{b,r}^{\mathrm{local}} + \mathrm{Num}_{b,r}^{\mathrm{remote}}
}{
\mathrm{Den}_{b,r}^{\mathrm{local}} + \mathrm{Den}_{b,r}^{\mathrm{remote}} + \varepsilon
}
}
$$

这就是最终应用于分块训练的数学公式.

## 11. 最小实现建议

如果现在要做 `CLR v1`, 建议第一版固定成:

- $u = v$
- `softmax-like CLR`
- 一阶近似
- 先保留分母修正 $a_{s,t}$ 的完整式

若工程复杂度太高, 可以先 ablate 一个简化版:

$$
a_{s,t} \approx 0
$$

那么 remote 分母退化为:

$$
\mathrm{Den}_t^{\mathrm{remote}}
\approx
\sum_{s \in \mathcal{S}_t}
\exp(\ell_{t,s}^{\mathrm{base}} - m_t) L_{s,t}
$$

这个版本和当前 `U / L` 结构更像, 也更容易先跑通.

## 12. 一句话总结

这套一阶近似 `softmax-like CLR` 的核心就是:

1. 先把 key 拆成 `centroid + residual`.
2. 把桶内精确的 $\exp(\alpha^\top h)$ 重加权做一阶展开.
3. 把结果压成每个桶的 4 个固定状态: $g$, $R$, $L$, $a$.
4. 再按 GLA 的 $(\Lambda, \Gamma, \gamma)$ 思路做 chunkwise 训练.

所以它既保持了 `softmax-like remote` 语义, 又能写成固定状态、可分块训练的形式. `Code-local Residual Correction` 的几何直觉和 GLA 的 chunkwise 框架, 正好在这里拼上.
