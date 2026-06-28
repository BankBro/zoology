# Flash-VQG GD residual 里的 M_state 是什么

## 结论

`M_state` 不是随便加的工程缓存. 它是当前 `gd_residual_v1` 数学方案里的核心 residual memory.

它的作用是: 在 coarse memory 的 code-level 平均值之外, 学一个 code 内部的低秩线性残差修正器. 它也受同一条遗忘门控路径控制.

## 设计来源

最核心数学来源:

- `/home/lyj/mnt/project/Flash-VQG/docs/20260425-flash-vqg-gated-delta-v1-math-plan-final.md`

实现蓝图:

- `/home/lyj/mnt/project/Flash-VQG/docs/20260425-flash-vqg-gated-delta-v1-codex-blueprint.md`

关键位置:

- math plan line 29: Flash-VQG V1 维护两级 memory.
- math plan line 74: 每个 code 的状态定义.
- math plan line 283: `Residual address`, 说明 `M_s` 是从地址到 correction 的线性 map.
- math plan line 424: `Residual writer`, 给出 gated-delta 写入公式.
- math plan line 485: `Residual read`, 给出读取 residual correction 的公式.
- blueprint line 239: 数学符号 `M` 对应代码名 `M_state`, 形状 `[B,H,N,S,d_v,r]`.

## 两级 memory

Flash-VQG V1 维护两级 memory:

| 层级 | 记什么 | 数学对象 | 作用 |
| --- | --- | --- | --- |
| coarse memory | 每个 code 的 value 平均 | $(\mathcal G,\mathcal L)$ | 给出长期历史的粗略压缩 |
| residual memory | coarse 平均值的误差 | $M_s$ | 修正 coarse mean 表达不了的细节 |

coarse mean 是:

$$
\mu_s
=
\frac{\mathcal G_s}{\mathcal L_s+\varepsilon_{den}}
$$

这里 $\mathcal G_s$ 是 value 加权和, $\mathcal L_s$ 是写入质量.

## M_s 的数学定义

每个 code 维护一个 residual correction memory:

$$
  M_s \in \mathbb R^{d_v \times r}
$$

它可以看成一个从 $r$ 维地址到 $d_v$ 维 correction 的线性 map:

$$
  M_s d \in \mathbb R^{d_v}
$$

直观理解:

```text
G/L 只知道 code s 的平均 value.
M_s 负责在 code s 内部, 根据更细的地址 d, 预测还需要补多少 value residual.
```

## residual target

对于写入 code $s$ 的 token, 先用 coarse memory 得到这个 code 的平均 value:

$$
\mu^{pre}_{t,s}
$$

真实 value 是:

$$
  v_t
$$

所以 residual target 是:

$$
  u_{t,s}
=
  v_t
-
\operatorname{sg}(\mu^{pre}_{t,s})
$$

其中 $\operatorname{sg}$ 是 stop-gradient. 直白说, $u_{t,s}$ 就是这个 token 的真实 value 相对 code 平均值还多出来的部分.

## residual address

写入地址由 key 和 code 的差得到:

$$
  z^{write}_{t,s}
=
(k_t-c_s)R
$$
$$
  d^{write}_{t,s}
=
\frac{
    z^{write}_{t,s}
  }{
\max(\|z^{write}_{t,s}\|_2,\varepsilon_{addr})
  }
$$

读取地址由 query 和 code 的差得到:

$$
  z^{read}_{t,s}
=
(q_t-c_s)R
$$
$$
  d^{read}_{t,s}
=
\frac{
    z^{read}_{t,s}
  }{
\max(\|z^{read}_{t,s}\|_2,\varepsilon_{addr})
  }
$$

其中 $R$ 对应实现里的 `addr_proj`.

## M 的写入公式

原型文档中的 token-level residual writer 是:

$$
  M_{t,s}
=
\alpha_t M_{t-1,s}
+
\zeta_t
\left(
  u_t-\alpha_t M_{t-1,s}d_t
\right)d_t^\top
$$

这里的 $\zeta_t$ 是当前 token 对这个 code 的实际 residual 写入强度. 更完整的符号是:

$$
\zeta_{t,s}
=
\beta_t \rho_{t,s}
$$

这段公式固定了 code $s$, 所以文档里简写为:

$$
\zeta_t
=
\zeta_{t,s}
$$

- $\beta_t$: token 级 residual write gate, 表示当前 token 总体上愿不愿意写 residual correction.
- $\rho_{t,s}$: code 级 residual write responsibility, 表示当前 token 的 residual correction 应该分多少给 code $s$.

所以 $\zeta_{t,s}$ 可以理解为这一步 gated-delta update 的有效学习率. 它控制预测误差写回 $M_s$ 的力度:

- $\zeta_{t,s}=0$: 不写入当前 token 的 residual update, 只保留遗忘后的旧 memory.
- $\zeta_{t,s}$ 较小: 轻微修正 $M_s$.
- $\zeta_{t,s}$ 较大: 更强地用当前 token 修正 $M_s$.

含义:

- $\alpha_t M_{t-1,s}$: 先遗忘旧 memory.
- $\alpha_t M_{t-1,s}d_t$: 用遗忘后的 memory 预测 residual target.
- $u_t-\alpha_t M_{t-1,s}d_t$: 预测误差.
- $\zeta_t (u_t-\alpha_t M_{t-1,s}d_t)d_t^\top$: 用误差做 outer-product delta update.

如果当前 token 不写入 code $s$, 那么 $\zeta_t=0$, 公式退化为:

$$
  M_{t,s}
=
\alpha_t M_{t-1,s}
$$

也就是 no-write code 仍然 pure decay.

## 和当前实现里的 f_l 对应关系

当前实现中:

```text
f_l = torch.exp(logf_blk_eff[:, :, n, ell])
M_cur = f_l[:, :, None, None, None] * M_cur
```

这对应数学原型里的:

$$
\alpha_t M_{t-1,s}
$$

所以是的, `M_state` 也通过同一条 `logf -> f` 路径做衰减控制.

然后当前 token 的 residual update 对应:

```text
pred = M_cur[b, h, s] @ d
err = u - pred
M_cur[b, h, s] = M_cur[b, h, s] + zeta * torch.outer(err, d)
```

对应数学:

$$
\hat u = M_s d
$$
$$
  e = u - \hat u
$$
$$
  M_s \leftarrow M_s + \zeta e d^\top
$$

## M 的读取公式

原型文档里的 residual read 是:

$$
  u_t^{res}
=
\sum_{s\in S_t^{coarse}}
\omega^{coarse}_{t,s}
  M^{pre}_{t,s}
  d^{read}_{t,s}
$$

最终输出:

$$
  o_t
=
  o_t^{base}
+
\lambda_t \operatorname{RMSNorm}(u_t^{res})
$$

当前实现里对应:

```text
d_read = normalize((Q - C_sel) @ addr_proj)
proposal = M_sel @ d_read
u_res = sum(omega_sel * proposal)
Out = O_base + lambda * RMSNorm(u_res)
```

## 和 GDN / Gated DeltaNet 公式的关系

这个 `M` 的写入公式和 GDN / gated-delta 的核心更新形式很像. 这不是巧合, `gd_residual_v1` 里的 `GD` 就是 gated-delta 这一类思想.

GDN / gated-delta 可以抽象成:

$$
  S_t
=
\alpha_t S_{t-1}
+
\gamma_t
\left(
    v_t-\alpha_t S_{t-1}k_t
\right)k_t^\top
$$

其中 $S_t$ 是序列状态矩阵, $k_t$ 是写入地址, $v_t$ 是目标 value, $\alpha_t$ 是遗忘门, $\gamma_t$ 是写入门或更新强度.

当前 GD residual 的 code 内部更新是:

$$
  M_{t,s}
=
\alpha_t M_{t-1,s}
+
\zeta_{t,s}
\left(
    u_{t,s}-\alpha_t M_{t-1,s}d_{t,s}
\right)d_{t,s}^\top
$$

| GDN / gated-delta 概念 | 当前 GD residual 里的对应物 |
| --- | --- |
| state matrix $S_t$ | 每个 code 的 residual memory $M_{t,s}$ |
| forget gate $\alpha_t$ | 同一类遗忘门, 当前实现里对应 $f_l=\exp(\operatorname{logf})$ |
| write address $k_t$ | code 内部地址 $d_{t,s}$ |
| target value $v_t$ | residual target $u_{t,s}=v_t-\mu^{pre}_{t,s}$ |
| prediction $\alpha_t S_{t-1}k_t$ | $\alpha_t M_{t-1,s}d_{t,s}$ |
| prediction error | $u_{t,s}-\alpha_t M_{t-1,s}d_{t,s}$ |
| update strength $\gamma_t$ | $\zeta_{t,s}=\beta_t\rho_{t,s}$ |
| outer-product update | $\zeta_{t,s}(\cdots)d_{t,s}^\top$ |

主要相同点是: 都先用旧 state 预测目标, 再把预测误差通过 outer-product 写回 state.

主要差异是: GDN 通常用一个全局序列状态直接建模记忆; 当前 GD residual 是在 Flash-VQG 的 VQ/coarse remote memory 后面, 给每个 code $s$ 单独维护一个小的 residual state $M_s$, 只用来补 code-level 平均值表达不了的细节.

### GDN 源码里 alpha 和 beta 的来源

上面抽象公式里的 $\gamma_t$ 在 GDN 源码和 FLA kernel 参数里叫 `beta`. 所以可以把它们理解成同一个角色:

$$
\gamma_t \equiv \beta_t
$$

在 `zoology/mixers/gated_delta_net.py` 里, GDN 定义了两条门控投影:

```text
self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
```

更新强度来自 `b_proj` 后接 sigmoid:

```text
beta = self.b_proj(hidden_states).sigmoid()
```

$$
\beta_t
=
\sigma(W_b x_t)
$$

这里 $\beta_t$ 控制这一步 gated-delta correction 写入 state 的力度. 它对应抽象公式里的 $\gamma_t$.

遗忘门不是直接输出 $\alpha_t$, 而是先输出 log-space gate `g`:

```text
g = -self.A_log.float().exp() * F.softplus(
    self.a_proj(hidden_states).float() + self.dt_bias
)
```

$$
  g_t
=
-\exp(A_{\log})
\operatorname{softplus}(W_a x_t + b_\Delta)
$$

FLA 的 gated-delta kernel 把 `g` 当作 log-space forget gate, 也就是:

$$
  g_t
=
\log \alpha_t
$$
$$
\alpha_t
=
\exp(g_t)
$$

展开后就是:

$$
\alpha_t
=
\exp\left(
-\exp(A_{\log})
\operatorname{softplus}(W_a x_t + b_\Delta)
\right)
$$

这里的 $A_{\log}$ 不是 $\log \alpha_t$, 而是 GDN 源码里存的一个可学习 per-head 衰减速率参数. 源码初始化逻辑是:

```text
A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
A_log = torch.log(A)
self.A_log = nn.Parameter(A_log)
```

所以它的含义是:

$$
  A_{\log}
=
\log A
$$

真正进入遗忘门计算时, 源码会通过 `self.A_log.float().exp()` 把它还原为正数:

$$
\exp(A_{\log})
=
  A
$$

把这条门控拆开看, 可以写成:

$$
  d_t
=
\operatorname{softplus}(W_a x_t + b_\Delta)
$$
$$
  g_t
=
-A d_t
$$
$$
\alpha_t
=
\exp(-A d_t)
$$

这里 $A$ 的作用是每个 head 的基础遗忘速度, 也可以理解成 intrinsic decay rate. $d_t$ 是当前 token 产生的动态衰减量, $A$ 是这个 head 自身学到的全局衰减尺度.

- $A$ 越大, $-A d_t$ 越负, $\alpha_t$ 越小, 旧 state 忘得越快.
- $A$ 越小, $\alpha_t$ 越接近 1, 旧 state 保留越多, 记忆更长.

所以 $A$ 不是普通的 value 权重, 而是控制某个 head 天生偏短记忆还是偏长记忆的可学习衰减速率.

普通 GDN 里 $A$ 的形状是每个 head 一个标量:

$$
  A \in \mathbb{R}^{H}
$$

其中 $H=\texttt{num\_heads}$. 对第 $h$ 个 head, 可以写成:

$$
  g_{t,h}
=
-A_h
\operatorname{softplus}(W_a^{(h)}x_t+b_{\Delta,h})
$$
$$
\alpha_{t,h}
=
\exp(g_{t,h})
$$

对应源码里, `self.A_log.float().exp()` 的形状是 `[H]`, `self.a_proj(hidden_states)` 的形状是 `[B,T,H]`, `self.dt_bias` 的形状也是 `[H]`. 通过 broadcast 后, 最终 `g` 的形状是 `[B,T,H]`.

如果是源码里的 banked GDN 版本, $A$ 的形状是 `[effective_heads]`, 语义仍然是每个有效 head 一个基础遗忘速率.

这样做的好处是: 模型实际使用的衰减速率 $A$ 必然为正, 不需要额外 clamp. 它控制某个 head 整体上忘得快还是慢; `a_proj(hidden_states)` 和 `dt_bias` 则控制当前 token 对遗忘强度的动态调节.

因此三者关系是: $A_{\log}$ 是源码存储的 log 参数, $g_t$ 才是 log-space forget gate, $\alpha_t$ 才是真正乘到旧 recurrent state 上的遗忘系数.

因为 $\exp(A_{\log})>0$, $\operatorname{softplus}(\cdot)>0$, 所以 $g_t \le 0$, 进而 $0 < \alpha_t \le 1$. 这保证它是遗忘衰减门: 越接近 1, 旧 state 保留越多; 越接近 0, 旧 state 被衰减得越强.

对应到当前 Flash-VQG GD residual, GDN 的 $\alpha_t$ 对应当前的 $f_l=\exp(\operatorname{logf})$, 也就是 `fox_gate_proj -> logf -> f_l` 这条路径. GDN 的 $\beta_t$ 对应当前 residual write gate 的 token 级部分; 当前实现还多了 code 级责任分配 $\rho_{t,s}$, 所以实际写入强度是:

$$
\zeta_{t,s}
=
\beta_t \rho_{t,s}
$$

把 GDN 的 $\alpha_t$, $\gamma_t$ 和当前 Flash-VQG 的对应量放在一起看, 来源大致是:

| 模型 | 量 | 作用 | 代码来源 | 计算式 |
| --- | --- | --- | --- | --- |
| GDN / Gated DeltaNet | $\alpha_t$ | forget gate, 控制旧 recurrent state 留多少 | `a_proj(hidden_states)`, `A_log`, `dt_bias`, 再传给 FLA kernel 的 `g` | $g_t=-\exp(A_{\log})\operatorname{softplus}(W_a x_t+b_\Delta)$<br> $\alpha_t=\exp(g_t)$ |
| GDN / Gated DeltaNet | $\gamma_t$ 或 $\beta_t$ | write/update gate, 控制 delta correction 写多强 | `beta = self.b_proj(hidden_states).sigmoid()` | $\gamma_t\equiv\beta_t$<br> $\beta_t=\sigma(W_b x_t)$ |
| Flash-VQG GD residual | $\alpha_t$ 或 $f_t$ | forget gate, 控制每个 code 的 `G/L/M` state 衰减 | `fox_gate_proj(x)` -> `fox_gate_logf` -> `f_l = exp(logf)` | $\ell_t=\log\sigma(W_f x_t+b_f)/\eta$<br> $f_t=\exp(\ell_t)$ |
| Flash-VQG GD residual | $\zeta_{t,s}$ | code-specific residual write strength, 控制 token 对 code $s$ 的 $M_s$ 写入力度 | `fox_gd_residual_beta_proj(x)` 生成 token 级 `beta_all`; top-k code 权重生成 `write_strength`; 二者相乘后再过可选 cap | $\beta_t=\sigma(W_\beta x_t+b_\beta)$, 默认 hard-cap 模式下可能再 clamp<br> $\rho_{t,s}\approx\operatorname{renorm\_topk}(\omega_{t,s})$<br> $\zeta^{raw}_{t,s}=\beta_t\rho_{t,s}$<br> $\zeta_{t,s}=\operatorname{cap}(\zeta^{raw}_{t,s})$ |

注意: 当前 Flash-VQG 没有 GDN 那个显式的 $A_{\log}$ 参数. 它的整体遗忘尺度主要由 `fox_gate_logit_normalizer` 和 `fox_gate_proj` 的 bias/weight 共同决定.

一句话说: GDN 里 $\alpha_t$ 来自 `a_proj + A_log + dt_bias`, 控制旧 state 留多少; $\gamma_t$ 或 $\beta_t$ 来自 `b_proj.sigmoid()`, 控制 delta update 写多强. 当前 GD residual 保留了这套 gated-delta 语义, 但把全局 recurrent state 换成了按 code 分桶的 residual memory $M_s$.

## 与当前代码的对应关系

- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:1617`: 分配 `M_state = [B,H,N,S,d_v,r]`.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:1645`: 每个 block 开头保存 `M_state[:, :, n] = M_cur`.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:1670`: `M_cur = f_l * M_cur`, 即先遗忘.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:1728`: `pred = M_cur @ d`.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:1733`: `M_cur += zeta * outer(err, d)`.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:2790`: 读取时 `proposal = M_sel @ d_read`.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py:2791`: 用 `omega_sel` 加权成 `u_res`.

## 一句话总结

`G_state/L_state` 存的是每个 code 的平均 value. `M_state` 存的是每个 code 内部的 residual correction map. 它根据 code 内部地址 $d$ 预测 value residual, 并且像 `G/L` 一样受遗忘门控衰减.
