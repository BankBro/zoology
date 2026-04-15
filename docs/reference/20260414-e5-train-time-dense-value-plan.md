# 实验 5 Train-Time Dense-Value Teacher 实施方案

## 1. 编号说明与实验定位

本文中的"实验 5"沿用 [20260413-e5a-top2-audit.md](/home/lyj/mnt/project/zoology/docs/reference/20260413-e5a-top2-audit.md) 第 19 节之后的分流语义, 指的是 **teacher-guided selection 的训练期主线**. 它和 [20260325-flash-vqg-mqar-localization-plan.md](/home/lyj/mnt/project/zoology/docs/reference/20260325-flash-vqg-mqar-localization-plan.md) 里原始 `E5 = Codebook size sweep` 不是同一个编号对象. 为避免继续混淆, 本文后续统一记为 `E5-train`.

`E5-train` 的目标不是再证明 `eval-time override` 是否有效, 而是验证下面这个更具体的问题:

- `dense_value teacher` 里已经暴露出来的 value-aware selection signal, 能否在 **训练期** 被当前模型吸收, 并转化为稳定的 end-to-end 提升.

因此, 本文默认排除以下路线:

- 不继续扩展 `eval-time partial override`.
- 不把 `teacher top2` 直接硬塞进真实推理路径.
- 不把问题重新定义成"证明 selector 本身一定是主瓶颈".

本文只讨论一个最小而明确的训练假设:

- 在当前 `dense-t025 + top2 read + den_aware selector + shared_den` 主线上,
- 把 `dense_value teacher` 变成 **训练期辅助监督**,
- 让 selector 与 downstream 一起重新适应,
- 看它是否能超过当前 `student` baseline.

## 2. 当前证据与立项理由

来自阶段 A/B 的现有证据可以归纳为三点:

1. `dense_value teacher` 明显强于 `query/ref`, 说明 selection 中确实存在真实的 value-aware signal.
2. 但 `post-hoc full override` 和各种 `partial override` 都没有超过 `student`, 说明这个信号 **不能通过事后硬替换直接兑现**.
3. `conf_q90` 已经非常接近 `student`, 说明 teacher 并不是在乱选; 真正的问题更像是 **信号如何被系统吸收**, 而不是信号是否存在.

这组结果对 `E5-train` 的意义是:

- 它不再支持继续做"推理时替换 selector".
- 但它开始支持一个更合理的训练期命题:
  - `teacher signal` 可能是真的;
  - 只是它需要在 train-time 与 downstream 一起完成 joint adaptation, 才能变成收益.

因此, `E5-train` 的定位应是:

- **验证 `dense_value teacher` 能否被训练期联合吸收**,
- 而不是重复验证 `override` 能否工作.

## 3. 推荐方案概览

推荐方案固定为:

- **训练期辅助监督, 不做训练期 teacher forcing**.
- **teacher 固定使用 `dense_value`**, 不再引入 `ref/query/dense_mass` 对照.
- **第一轮只挂在当前唯一的 FlashVQG 主线层上**, 不做多层扩展.
- **第一轮优先从现有 `dense-t025` checkpoint 短程 fine-tune 开始**, 不直接大规模从零训练.

核心思想是:

1. 正常跑当前模型 forward.
2. 在同一次 forward 里, 复用已经存在的 `Q/K/V/write_weights/c_all` 等张量.
3. 用这些张量, 以 `no_grad` 方式计算 `dense_value teacher`.
4. 把 teacher 变成 selector 的辅助损失.
5. 主任务 CE 不变, downstream 路径不手工替换.

因此, 训练时发生的改变只有一件事:

- selector 额外被要求"更像 dense_value teacher", 但模型仍然自己前向生成最终输出.

这条路线与阶段 B 的根本区别是:

- 阶段 B 是 **直接改推理时读哪两个 code**;
- `E5-train` 是 **用 teacher 作为监督, 让模型自己学会更接近 teacher 的读法**.

## 4. 训练期 Teacher 的构造方式

### 4.1 总体原则

训练期 teacher 不来自外部模型, 也不需要第二次完整模型 forward.  
它来自 **当前这一次 forward 中已经算出来的层内张量**.

推荐沿用 E5A 的 `dense_value` 定义:

- 先假设当前 row 可以看完整个 remote-visible dense history.
- 得到一个 dense remote 的理想输出方向.
- 再统计每个 code 对这个理想输出方向贡献了多少有效 value.
- 贡献最高的 code, 就是 teacher 认为更该被读的 code.

这意味着训练期 teacher 依赖的主要张量就是:

- `Q_blk`
- `K_blk`
- `V_blk`
- `write_weights`
- `c_all`

这些张量在当前主路径里已经存在, 对应代码位置可从 [Flash-VQG/src/flash_vqg/nn/attn.py](/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py:983) 起确认.

### 4.2 训练期有效行集合

第一轮训练不建议对所有 row 都上 teacher loss.  
推荐只在下面这个集合上生效:

- `task row`: 标签不是 `-100`
- 且 `remote-visible history` 非空

也就是与 E5A / E5B 一致的 `audit_rows` 语义.

这样做有两个好处:

1. 训练信号直接对齐真正影响任务 loss 的位置.
2. 额外计算成本更可控.

### 4.3 teacher 必须 stop-gradient

训练期 `dense_value teacher` 只能作为监督目标, 不能反向参与主图更新.  
实现上应固定满足:

- teacher 分数用 `detach` 或 `torch.no_grad()` 计算
- teacher 目标不回传梯度
- 梯度只流向当前 selector / query / codebook / downstream 主路径

否则会把"teacher 目标本身也跟着一起动"混进训练图, 使监督语义失真.

## 5. 辅助损失设计

## 5.1 推荐版本: Sparse Soft CE

第一轮最推荐的损失不是 full dense KL, 而是一个更便宜的 **Sparse Soft CE**.

它的定义是:

- 先用 `dense_value teacher` 找出当前 head-row 上的 teacher top2 codes.
- 再用 teacher 自己在这两个 code 上的相对分数, 形成一个 2 维 soft target.
- student 侧仍然面对全部 code 的 selector logits.
- 损失要求 student 在全部 code 中, 把更大的概率质量放到 teacher 选出的两个 code 上.

写成等价形式, 可以理解为:

```text
L_sparse = 全部 code 的 logsumexp
           - teacher_top2 上按 teacher 权重加权的 student logit 平均
```

这个版本的优点是:

- **保留了 teacher 的强弱信息**, 不是纯 hard label.
- **不需要保存完整 teacher dense 分布**, 只需要 teacher top2 和其相对权重.
- **student 侧仍然对全部 code 做归一化竞争**, 不会退化成二分类小问题.
- 很适合当前 `num_codebook_vectors=128` 的主线实现.

### 5.2 为什么不把 full dense KL 作为首版

full dense KL 在理论上更完整, 但首轮不推荐直接上, 原因是:

- 需要保留更完整的 teacher 分布.
- 实现和显存压力都更大.
- 当前我们首先要验证的是"这个信号能不能被吸收", 不是一步到位做最强蒸馏器.

因此更稳妥的顺序应是:

1. 先做 `Sparse Soft CE`
2. 若它已经出现稳定正收益, 再考虑升级到 full dense KL

### 5.3 行级权重

首轮 teacher loss 仍保留行级权重接口, 但正式执行口径固定为:

1. 默认版本: `uniform`
2. `adv_relu` 只作为后续 ablation 备选, 不进入首轮正式筛选

其中 `adv_relu` 的直觉仍然成立:

- teacher 明显优于当前 student 的 row, 权重大
- teacher 和 student 差不多的 row, 权重小

实现上可写成:

```text
w_row = relu(teacher_adv_row)
L_dense_teacher = weighted_mean(L_sparse, w_row)
```

但本轮最关键的问题不是"权重函数怎么调", 而是先用最小预算确认 `teacher signal` 本身是否值得继续投. 因此 v1 / v2 首轮默认固定 `row_weight_mode = uniform`.

### 5.4 loss 总式

第一轮总损失固定建议写成:

```text
L_total = L_task + lambda_teacher * L_dense_teacher + beta_commit * L_commit
```

其中:

- `L_task` = 原始 MQAR 交叉熵
- `L_dense_teacher` = 新增的 teacher-guided selector loss
- `L_commit` = 现有 codebook commit loss

并建议对 `lambda_teacher` 加 warmup:

- 前 `5% - 10%` 训练步, 从 `0` 线性升到目标值

这样可以避免训练一开始 teacher loss 直接压过主任务.

## 6. 实现边界与代码落点

### 6.1 Flash-VQG 仓库

核心改动固定落在 `Flash-VQG` 仓库, 因为 teacher 和 selector logits 都在 attention 内部最容易拿到.

推荐落点:

- [configuration_flash_vqg.py](/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/configuration_flash_vqg.py)
  - 新增训练期开关和超参
- [attn.py](/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py)
  - 在当前 forward 中挂接 teacher loss
  - 把 `l_dense_teacher` 放入 `aux`
- [attn_fox.py](/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn_fox.py)
  - 新增 `dense_value teacher` 计算 helper
  - 新增 `Sparse Soft CE` 的 streaming 计算 helper

推荐新增配置项:

- `fox_dense_teacher_mode = off | dense_value`
- `fox_dense_teacher_loss_mode = off | sparse_top2_ce`
- `fox_dense_teacher_lambda`
- `fox_dense_teacher_tau_teacher`
- `fox_dense_teacher_row_weight_mode = uniform | adv_relu`
- `fox_dense_teacher_warmup_steps`

第一轮明确不做:

- 不做多层 teacher
- 不做 train-time hard override
- 不做 full dense KL
- 不做额外 teacher model checkpoint

### 6.2 Zoology 仓库

`Zoology` 侧只承担接线和实验脚本职责.

推荐落点:

- [zoology/mixers/flash_vqg.py](/home/lyj/mnt/project/zoology/zoology/mixers/flash_vqg.py)
  - 在 `get_auxiliary_loss()` 中把 `l_dense_teacher` 一并加进去
- `zoology/experiments/flash_vqg/...`
  - 新增 `E5-train` 对应的 run script / suite 接线
  - 固定默认配置与结果目录

训练器本身已经支持模块返回辅助损失并自动加总, 可直接复用 [train.py](/home/lyj/mnt/project/zoology/zoology/train.py:265) 的现成通路.

## 7. v2 执行阶段

当前正式执行口径不再直接从零训练大 sweep, 而是固定拆成 `P0 -> P1 -> P2 -> P3` 四段, 并且统一采用 paired control.

结果记录:

- P1 screening 结果见 [20260415-e5-train-screening-results.md](/home/lyj/mnt/project/zoology/docs/reference/20260415-e5-train-screening-results.md)

### 7.1 P0: 1 epoch smoke

目标:

- 只验证训练链路稳定性
- 只确认 dense teacher 指标能正常记出

设置:

- source checkpoint 固定为 `dense-t025-s123-d123`
- `teacher_mode = dense_value`
- `loss_mode = sparse_top2_ce`
- `row_weight_mode = uniform`
- `lambda_teacher = 0.05`
- `max_epochs = 1`

通过条件:

- `attn/dense_teacher_loss` 有限且非零
- 无 NaN / OOM / shape mismatch
- 无明显吞吐崩塌

### 7.2 P1: s123 上的 4-epoch screening

目标:

- 用最小预算判断 `teacher loss` 是否存在正信号
- 区分收益是否真的来自 teacher, 而不是"单纯继续训练 4 epoch"

设置:

- source checkpoint 固定为 `dense-t025-s123-d123`
- `row_weight_mode = uniform`
- `max_epochs = 4`
- teacher 组扫描:
  - `lambda_teacher = 0.02`
  - `lambda_teacher = 0.05`
  - `lambda_teacher = 0.10`
- paired control:
  - 同 source checkpoint
  - 同 `max_epochs = 4`
  - `lambda_teacher = 0.0`

### 7.3 P2: s124 上的 4-epoch repro

目标:

- 判断 P1 最优 `lambda_teacher` 的方向是否能跨 seed 复现

设置:

- source checkpoint 固定为 `dense-t025-s124-d123`
- `row_weight_mode = uniform`
- `max_epochs = 4`
- teacher 组:
  - `lambda_teacher = best_lambda_from_s123`
- paired control:
  - 同 source checkpoint
  - 同 `max_epochs = 4`
  - `lambda_teacher = 0.0`

### 7.4 P3: 32-epoch confirm

只有当 `P1 + P2` 都满足推进 gate 时, 才允许进入 `32 epoch confirm`.

确认组固定为 4 个 run:

- `s123`, `best lambda_teacher`, `max_epochs = 32`
- `s123`, control `lambda_teacher = 0.0`, `max_epochs = 32`
- `s124`, `best lambda_teacher`, `max_epochs = 32`
- `s124`, control `lambda_teacher = 0.0`, `max_epochs = 32`

这里的对照基线固定是:

- 从同一个 checkpoint 出发
- 同样继续训练同样的 epoch 数
- 唯一差别只是 `teacher loss` 关掉

因此, `P3` 结果不能直接只和历史上"从零训练 32 epoch 的原始 baseline"硬比, 否则会混入继续训练预算这个混杂因素.

## 8. 记录指标与成功判据

### 8.1 必看指标

除现有主指标外, 首轮需要新增下面这些训练期监控:

- `attn/dense_teacher_loss`
- `attn/dense_teacher_valid_row_ratio`
- `attn/dense_teacher_top2_overlap`
- `attn/dense_teacher_teacher_adv_mean`
- `attn/dense_teacher_row_weight_mean`

同时继续保留:

- `train/loss`
- `valid/loss`
- `valid/accuracy`
- `valid/mqar_case/*`
- 尤其是 `512x128` 与 `1024x256`

### 8.2 推进与成功判据

`P0` 通过条件:

1. `attn/dense_teacher_loss` 有限
2. 无 NaN / OOM
3. dense teacher 指标正常记录

`P1 / P2` 进入下一阶段的条件:

1. 至少 1 个 `lambda_teacher > 0` 的 run 同时优于 paired control 的 `acc_total`
2. 且 `acc_1024x256` 也优于 paired control
3. 不能只是 `dense_teacher_top2_overlap` 上升, 但 end-to-end 不涨

`P3` 成功条件:

1. 两个 seed 都优于各自的 `32-epoch paired control`
2. `acc_1024x256` 在两个 seed 上都同向为正

### 8.3 失败判据

满足任一条即可视为 fail-fast:

- 所有 `lambda_teacher > 0` 都不如对应 control
- `dense_teacher_top2_overlap` 上升, 但 `acc_total` / `acc_1024x256` 不涨
- 两个 seed 方向相反
- 训练明显不稳定或频繁 NaN

## 9. 风险, 成本与退路

### 9.1 主要风险

- `dense_value teacher` 计算会增加训练开销
- teacher loss 过强时, 可能直接破坏现有已收敛的 selector/downstream 协同
- 即使 teacher overlap 提高, 也可能无法兑现成最终 accuracy

### 9.2 控成本策略

若首轮成本偏高, 推荐按下面顺序降成本:

1. 只在 `task rows ∩ valid remote rows` 上算 teacher loss
2. 只保留 `Sparse Soft CE`, 不升级 full dense KL
3. 只做短程 fine-tune, 不从零训练
4. 若仍过重, 再考虑"每隔若干 step 才计算一次 teacher loss"

### 9.3 路线退路

若 `E5-train` 首轮失败, 当前最稳妥的回退结论是:

- `dense_value teacher` 可能有信号, 但以当前代价和实现方式不值得优先推进
- 后续主线应回到实验 4 所代表的表示层 / quantization 路线

也就是说:

- `E5-train` 的价值在于做一次 **最小而可判定的训练期验证**
- 若 `P1 / P2` 都没有正信号, 就应直接止损, 不进入 `P3`

## 10. 最小实施顺序

推荐按以下顺序执行:

1. 先在 `Flash-VQG` 与 `Zoology` 各自固化一版实验快照 commit
2. 跑 `P0 smoke`
3. `P0` 通过后, 跑 `P1 screening`
4. 产出 screening 表:
   - `variant`
   - `lambda_teacher`
   - `max_epochs`
   - `acc_total`
   - `acc_512x128`
   - `acc_1024x256`
   - `delta_vs_control`
5. 若 `P1` 有正信号, 用最优 `lambda_teacher` 跑 `P2 repro`
6. 产出双 seed 对照表:
   - `seed`
   - `best_lambda_teacher`
   - `teacher_run_acc_total`
   - `control_acc_total`
   - `teacher_run_acc_1024x256`
   - `control_acc_1024x256`
7. 只有当两张表都支持继续推进时, 才排 `P3 confirm`

本文的最终建议是:

- **可以推进 `E5-train`, 但必须把它当作一次低风险, 可失败, 可快速止损的最小训练验证**.
- 第一轮不要追求"一口气证明实验 5 正确", 而应优先回答:
  - `dense_value teacher` 能否在训练期被当前系统稳定吸收?
  - 它是否相对 paired control 带来 `1024x256` 上的稳定收益?
