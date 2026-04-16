# Flash-VQG vNext 架构方向整理

## 摘要

基于当前 repo 里的主线实验结论, 再结合你补充的那份外部回答, 这份文档把下一轮架构探索收束成一个更明确的原则:

- 下一轮不该再把主要赌注押在 selector 小修小补上.
- 更值得优先修改的是 remote memory 的 `写法, 存法, 遗忘法, 以及它和 local exact path 的协同方式`.
- 这不否定 quantization 路线, 相反, 更合理的理解是: 下一轮主线应同时覆盖 `memory mechanics` 和 `representation / quantization capacity`.

因此, 当前更推荐的 `vNext` 方向是:

`FlashVQG-vNext = Gated-Delta Remote Memory + Residual Grouped Quantization`

其中 `Dual-Code Memory + Adaptive Merge` 是最值得叠加的第二层升级, hybrid refresh layer, multi-timescale memory 和 retrieval-aware training 适合放在后续增强阶段.

## 为什么要这样收束

当前 baseline 已经稳定收口到 `dense-t025 + top2 read + den_aware selector + shared_den`. 这一点来自 [20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md](./20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md).

后续 `E5A / E5B / E5-train` 提供了几个比较清楚的信号:

- `ref` 这类 attention-mass teacher 不能稳定超过 `student`, 说明问题不是简单的 "Top2 选错了".
- `dense_value teacher` 显著强于 `query / ref`, 说明 selection 里确实存在真实的 value-aware signal.
- 但不管是 eval-time partial override, 还是 train-time teacher auxiliary loss, 截至目前都还没有给出足够稳定的 end-to-end 正收益.

所以更合理的判断不是 "selector 完全不重要", 而是:

- selector 不是当前最该优先动的入口.
- 真实瓶颈更像 remote memory 的状态更新机制, 表示近似能力, 以及它和 downstream 的联合适配.

你贴的那份外部回答里, 真正值得吸收的部分也和这个判断一致:

- `Gated Delta Networks` 提示 gating 和 delta update 是互补的, 适合处理 memory update.
- `Forgetting Transformer` 提示 data-dependent forgetting 不是只能放在纯 attention 或纯 RNN 里, 它可以作为 memory hygiene 的显式机制.
- `GLA` 和 `SSD / Mamba-2` 这条线说明, 只替换 remote state update, 但保留 chunkwise / hardware-friendly pipeline, 在形式上是可行的.
- `Titans` 则更像后续高风险方向的灵感, 适合启发 multi-timescale memory, 但不该立即上升为当前主线.

因此, 这份文档对外部回答的整合方式不是 "整包照搬", 而是:

- 吸收其中关于 `memory update / forgetting / hybrid schedule / training recipe` 的启发.
- 但把更激进, 更像新模型的部分降级为后续候选.

## 第一优先级: Gated-Delta Remote Memory

这是最值得先做的一条线.

核心想法是: 当前 remote memory 更像 "把很多 key-value 对往有限 code state 里叠加", 而不是 "对已有记忆做精确修改和清理". 在这种机制下, 即使 selector 有真实信号, remote state 也可能因为 collision, stale memory 或累计误差而无法兑现.

所以第一步不该推翻当前 `dense routing write + top2 read`, 而应优先升级 `state update`.

我建议的最小版本是:

- 保留当前 RoutingVQ dense assignment, 不推翻 code-level write 路径.
- 把 remote state 更新从简单加总累计, 升级成 `per-code gated delta update`.
- 首轮可以只对 `CLR residual state` 做 delta update, coarse state 先不动.
- 在 state transition 中加入 FoX 风格的 data-dependent forget gate.
- forget gate 第一版优先做成 code-aware decay, 不一开始就把 merge-time gate 和 logit bias 全并进来.

这条线的价值在于, 它不是去证明 "teacher 对不对", 而是直接让 `student` 自己的 remote retrieval 变得更会记, 更会改, 也更会忘.

我对它的预期也比较明确:

- 如果它有效, 首先改善的应该是 retrieval-heavy 的 `1024x256`.
- 收益应该体现为 `student` 本身变强, 而不是 teacher 对齐指标单独变好.

最小实验矩阵建议是:

1. `additive` baseline
2. `gated-only`
3. `delta-only`
4. `gated-delta`

在 `gated-delta` 稳定后, 再补一轮:

1. `row-wise forget`
2. `code-aware forget`

## 第二优先级: Residual Grouped Quantization

这条线和 topic A 是并列的重要主线, 不是替代关系.

当前最强的本地证据之一, 就是问题更像落在 `representation / quantization capacity`, 而不是 selector 规则本身. 所以两级 / 分组 / 残差量化, 仍然是非常值得押注的方向.

推荐的收束方式不是泛泛地说 "把 codebook 变大", 而是明确做成:

- `coarse_codebook` 负责大粒度召回
- `residual_codebook` 负责 coarse bucket 内的细粒度 disambiguation
- 如果成本敏感, 第二层进一步做成 `grouped residual quantization`

一个我认为值得保留的升级点是:

**不要只做 "两级 key quantization", 而是尽量朝 "coarse recall + residual state disambiguation" 的联合结构走.**

也就是说, coarse 层更像召回器, residual 层更像判别器. 这比单纯再套一层 codebook 更像真正的系统升级.

这条线的 gate 也应该收得比较严:

- 不是只看 recon 变好.
- 必须同时看 `acc_total`, `acc_1024x256` 和 harder buckets.
- 如果只是重建误差更好看, 但 retrieval 没变准, 就不应该晋升 baseline.

## 第三优先级: Dual-Code Memory + Adaptive Merge

这条线最适合在前两条出现正信号后再叠加.

当前 remote memory 很可能把三件事情绑得太死了:

- 找谁
- 读出什么
- 何时该信 remote

而 `dense_value teacher` 的结果说明, 真正有用的信号更接近 "对最终 remote output 的 value contribution", 而不只是 "query-key matching". 这意味着当前单一 code 表示可能同时承担了过多职责.

所以我认为最值得尝试的改法是:

- `routing_codebook`: 只负责寻址与排序
- `value_basis`: 只负责 remote value 重建
- write-side 同时维护 routing prototype 和 value prototype
- read-side 先按 routing code 做 Top2 寻址, 再由 value basis 恢复输出

这条线还有一个可以吸收的启发, 就是不要把 `shared_den` 视为永远不可动的固定语义.

更稳的做法不是立刻把 merge 全部推翻, 而是分两步:

1. 先只做 `routing / value` 解耦, merge 暂时尽量不动
2. 如果这一步成立, 再尝试:
   - `remote residual correction`
   - `query-conditioned local/remote fusion`

也就是说, "解耦 memory" 和 "放松 merge" 应该是同一方向里的两阶段, 而不是两个并发大改造.

## 后续增强: Hybrid Refresh, Multi-Timescale Memory And Training

外部回答里还有几类启发是可以吸收的, 但我不建议现在就把它们上升为第一主线.

### Periodic exact/FoX refresh layer

这个方向我认为是合理的.

它的核心不是 "把 Flash-VQG 换成别的层", 而是承认长层链中误差会积累, 所以需要少量 exact 或 FoX-style refresh 层充当校准锚点.

如果后续 topic A/B 成立, 这个方向很值得接着做:

- 比如每 4 层或每 6 层插一层 exact/sliding-window/FoX refresh
- 用来周期性刷新 local geometry 和 remote memory 的偏移

### Multi-timescale memory

这条线更有下一代架构味道, 但也更高风险.

它对应的直觉是:

- `local exact window`
- `VQ associative bank`
- `persistent bank`

这三种记忆职责不一定适合同一个 memory dynamics. 这个启发可以保留, 但当前不建议直接把 Titans 风格 persistent memory 当成正式主线.

### Retrieval-aware joint training

这一条我认为不是可有可无, 而是很可能要和结构升级绑在一起.

比较值得保留的训练侧想法有:

- `dense-value distill`
- `anti-collision objective`
- retrieval-heavy rows 的额外训练压力
- long-context curriculum
- chunkwise / TBPTT-compatible 训练

但这里的前提是: 训练目标应该服务于新的 memory update 或 quantization 结构, 而不是在当前旧结构上继续大量扫 teacher loss 超参.

## 暂不建议优先做的事

基于现有证据, 我会明确降级这几类方向:

- 继续深挖 eval-time selector 花活
- 只把 read 从 `top2` 改得更 dense
- 单纯把 codebook 做得更大, 但不改 memory update / forgetting / merge

这些方向不是永远没价值, 但从当前 repo 结论看, 它们不应该是下一轮主赌注.

## 推荐实验路线

如果要把这份文档变成真正的研发路线图, 我建议按下面顺序推进:

### 阶段 1

`baseline` vs `Gated-Delta Remote Memory`

目标:

- 先验证 remote state 的 `写法 + 遗忘法` 是否是主入口

### 阶段 2

`best Stage 1` vs `Stage 1 + Residual Grouped Quantization`

目标:

- 在 memory mechanics 稳定后, 验证表示层是否仍然是主要瓶颈

### 阶段 3

`best Stage 2` vs `Stage 2 + Dual-Code Memory`

目标:

- 验证路由和值表达的解耦, 是否能把 value-aware signal 更稳定地兑现出来

### 阶段 4

只有前 3 阶段给出正信号后, 才评估:

- hybrid refresh layer
- retrieval-aware joint training
- multi-timescale memory

统一 gate 建议为:

- `acc_total` 相对当前 baseline 稳定转正
- `acc_1024x256` 必须同向为正
- 不只看 selector 或 teacher 指标, 必须同时看 end-to-end 任务结果
- 若收益只出现在单 seed 或只出现在中间指标, 不晋升 baseline

## 参考资料

### 本地文档

- [20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md](./20260402-flash-vqg-clr-v1-experiment-and-metrics-plan.md)
- [20260413-e5a-top2-audit.md](./20260413-e5a-top2-audit.md)
- [20260414-e5-train-time-dense-value-plan.md](./20260414-e5-train-time-dense-value-plan.md)
- [20260415-e5-train-screening-results.md](./20260415-e5-train-screening-results.md)
- [20260415-e5-train-continuation-calibration-results.md](./20260415-e5-train-continuation-calibration-results.md)
- [20260415-e5-train-rescreening-lr1e4-results.md](./20260415-e5-train-rescreening-lr1e4-results.md)

### 外部参考

- Gated Delta Networks: Improving Mamba2 with Delta Rule  
  <https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule>
- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality  
  <https://huggingface.co/papers/2405.21060>
- Forgetting Transformer: Softmax Attention with a Forget Gate  
  <https://huggingface.co/papers/2503.02130>
- Gated Linear Attention Transformers with Hardware-Efficient Training  
  <https://research.ibm.com/publications/gated-linear-attention-transformers-with-hardware-efficient-training>
- Titans: Learning to Memorize at Test Time  
  <https://research.google/pubs/titans-learning-to-memorize-at-test-time/>
- QINCo: Residual Quantization with Implicit Neural Codebooks  
  <https://arxiv.org/abs/2401.14732>
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads  
  <https://arxiv.org/abs/2410.10819>
- DeepSeek-V2  
  <https://arxiv.org/abs/2405.04434>
- Fast-weight Product Key Memory  
  <https://arxiv.org/abs/2601.00671>
- Attention Residuals  
  <https://arxiv.org/abs/2603.15031>
- DeepCrossAttention: Supercharging Transformer Residual Connections  
  <https://arxiv.org/abs/2502.06785>
