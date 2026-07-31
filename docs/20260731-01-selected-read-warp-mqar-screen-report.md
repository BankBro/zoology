# Selected-read Warp MQAR 筛选报告

## 1. 结果概览

- Experiment ID: `20260731-01-selected-read-warp-mqar-screen`.
- 状态: `completed`, 终态为`quality_mixed`.
- 执行机器: RTX 3090.
- Zoology source: `b13f961d34608986fce0e57a8077708836923804`.
- Flash-VQG source: `efc75ad5539b636b026c76bedb70878bfe2390cf`.
- Plan: [实验计划](plans/20260731-01-selected-read-warp-mqar-screen-plan.md).
- Artifact: [精简证据](artifacts/20260731-01-selected-read-warp-mqar-screen/README.md).

W2 direct通过seed123 AMP BF16一轮MQAR筛选, 标准任务delta为`-0.005613`, 四外推slice宏平均delta为`-0.019200`. Preproject在标准任务上通过, 但外推宏平均delta为`-0.036765`, 因而拒绝. 三组FLA fused-gate backward均为`BT64/warps8`, runtime和fallback门禁全部通过.

W2 direct仍不能替换S1 exact质量canonical. 原因是它在production-shape低层对照中的`grad_addr_proj`最大绝对差约为`2e-4`, 超过预注册`2e-5`门槛, 本轮也只有一个training seed. 它应保留为通过低成本MQAR筛选的fast resource candidate.

## 2. 实验合同

三组共同使用canonical cache/init、seed123、AMP BF16、block64、local2、rank16、write4/read16、`post_phase1` remat、grouped builder、`fp32_boundary`和chunk8192. Preflight确认三组参数量均为1,160,390, 初始state hash相同, candidate相对S1只改变selected backward backend.

| Variant | Selected backward | 作用 |
|---|---|---|
| S1 | `triton_deterministic_s1_head` | Exact质量对照 |
| W2 direct | `triton_state_owner_r1a_s1_w2` | State-owner加warps2 |
| Preproject W2 | `triton_state_owner_r1b_preproject_w2_fast` | 追加query/code预投影 |

三组smoke均完成3个optimizer updates. 三组Q0均从canonical init独立训练1 epoch, 完成704个optimizer updates、last/best/resume checkpoint及5个locked MQAR评估任务. 全部loss和checkpoint finite, GradScaler skip为0, Triton fallback为0.

## 3. 质量结果

### 3.1. 标准与外推任务

| 指标 | S1 | W2 direct | Delta | Preproject W2 | Delta |
|---|---:|---:|---:|---:|---:|
| Validation `1024x256` | 0.962172 | 0.956559 | **-0.005613** | 0.955762 | **-0.006410** |
| `2048x512` delta | - | - | -0.022691 | - | -0.022332 |
| `4096x1024` delta | - | - | -0.029805 | - | -0.051338 |
| `8190x512` delta | - | - | -0.010512 | - | -0.028871 |
| `8190x2047` delta | - | - | -0.013792 | - | -0.044520 |
| 四外推宏平均delta | - | - | **-0.019200** | - | **-0.036765** |

预注册门槛为标准delta不低于`-0.01`, 外推宏平均delta不低于`-0.02`. W2 direct两项均通过, 但外推门槛余量只有约`0.000800`. Preproject标准通过、外推失败, 且4个外推slice全部负向, 不能用单个边界指标解释.

### 3.2. 数值与证据边界

三组最终model和optimizer hash均不同, 说明W2改变的FP32归约树在704步训练中已经形成不同轨迹. FLA config完全一致, 排除了本轮差异来自历史warps4/warps8 autotune混杂的解释.

W2 direct的结果支持:

> 在seed123、block64、AMP BF16和一轮训练下, 没有观察到超过注册非劣门槛的质量下降.

它不支持:

> W2 direct已经证明与S1质量等价, 或可以直接进入300M自然语言正式训练.

要提高证据等级, 至少需要更多training seeds或300M BF16短自然语言paired pilot. 在此之前,S1 exact继续作为质量canonical.

## 4. 小模型资源信号

| Variant | Wall | Step p50 | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| S1 | 210.84 s | 0.22392 s | 1456.64 MiB | 2192 MiB |
| W2 direct | 203.83 s | 0.21200 s | 1456.64 MiB | 2194 MiB |
| Preproject W2 | 203.75 s | 0.21329 s | 1456.64 MiB | 2194 MiB |

小模型中W2 direct的step p50改善约5.3%, peak reserved只增加2 MiB. Preproject没有比direct产生更好的step p50, 同时外推质量失败. 因此它的300M额外约0.72%吞吐收益不足以抵消实现复杂度和质量风险.

300M正式吞吐不使用本表裁决, 仍以Flash-VQG项目仓的3×5 fresh-process结果为准.

## 5. 决策

1. 保留S1 exact为质量canonical.
2. 保留W2 direct为fast resource candidate, 继续做K2组合资源测量和生命周期验证.
3. 拒绝preproject进入质量路径; 它只保留为资源上限证据, 最终生产源码应删除该prototype.
4. 不启动三seed四epochMQAR矩阵. 当前首要目标是完成本轮效率闭环, 随后由300M BF16短自然语言paired pilot决定是否进一步投资W2质量验证.

## 6. 原始证据

3090原始目录:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260731-01-selected-read-warp-mqar-screen/outputs/3090/
20260731-selected-warp-mqar-01/
```

排除18个checkpoint后的89个轻量文件已镜像回2080 Ti同相对路径, aggregate SHA256为`2181c777ab52b32f089c50db5141f7e1c4f51e2ca360df8cd3b8454ffafca2c0`. 18个checkpoint共189,993,688 bytes, 继续保留在3090.
