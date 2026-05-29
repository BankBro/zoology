# 20260529 Flash local window fairness report

## 摘要

本轮检查的公平性假设从代码上成立: Flash-VQG/GD residual 配置有 exact local attention branch, 而 GatedDeltaNet baseline 没有同等 exact local attention branch. 因此 Flash-VQG vs GDN 的严格公平比较需要承认这个结构差异.

但需要修正一个关键表述: 当前 `block_len=32`, `local_num_blocks=2` 不是“每个 query guaranteed exact attend 最近 64 token”. 它更准确是 **2-block causal exact local path**: local window tensor width 为 64, 由前一个 block 和当前 block 组成, 再经过 causal mask. 所以每个 query 稳定 guaranteed 的 exact local history 约为 `<=32` token, `33-64` 是 block alignment 边界区.

阶段 1-3 和 eval-only 补强结果共同说明: local exact branch 会带来 very-near distance 的优势, 主要影响 `<=32` 和部分 `33-64`; 但 Flash-VQG 在 longer-MQAR 长距离桶上的优势不能主要由 local branch 解释. local-only 在 official longer-MQAR 上接近随机, 而 remote-on 的 Flash 仍在 `1025+` 和 `2049+` 距离桶明显强.

## Artifact

- 3090 source-of-truth 分支: `codex/20260529-flash-local-window-fairness`.
- 阶段 1/2 longer-MQAR eval artifact: `docs/artifacts/longer-mqar/local-window-fairness-20260529/`.
- 阶段 3 training artifact: `docs/artifacts/20260529-flash-local-window-fairness/`.
- 阶段 3 训练 summary: `docs/artifacts/20260529-flash-local-window-fairness/stage3_train_summary.csv`.
- eval-only 补强 artifact: `docs/artifacts/20260529-flash-local-window-fairness-eval-only/`.
- eval-only 补强报告: `docs/20260529-flash-local-window-fairness-eval-only-report.md`.

## 代码事实

Flash-VQG wrapper 暴露并传递 `block_len` 和 `local_num_blocks`. 现有 gd residual v1 builder 固定 `block_len=32`, `local_num_blocks=2`. Flash-VQG backend 的 phase2 local path 使用 `W = local_num_blocks * L`, 即窗口张量宽度为 `2 * 32 = 64`.

这个 64 是窗口张量宽度, 不是每个 query 的 guaranteed 向左历史长度. local path 以 block 为单位构造 window. 对某个 query 所在 block 来说, `local_num_blocks=2` 覆盖前一个 32-token block 和当前 32-token block, 然后 causal mask 去掉当前 block 中 query 右侧的 token.

若 query 在当前 block 内 offset 为 `r`, `r in [0, 31]`, 它可见的历史长度大约是:

```text
previous block 32 tokens + current block prefix r tokens = 32 + r
```

因此:

```text
<=32:   稳定落在 exact local path 可见范围内.
33-64:  是否可见取决于 query 在 block 内的位置, 是边界区.
65+:    超出 local_num_blocks=2 的 local path 能力.
```

GatedDeltaNet 代码只有 short convolution 和 gated delta recurrent state: 默认 `use_short_conv=True`, `conv_size=4`, 然后调用 gated delta rule kernel, 没有同等 exact local attention branch. 因此原始 Flash-VQG vs GDN 比较在机制上确实可能混入 very-near local exact attention 优势.

## 阶段 1: 现有 checkpoint longer-MQAR distance bucket eval

阶段 1 对已有 Flash/GDN checkpoint 跑 full variant 的 longer-MQAR bucket eval. 覆盖 slices: `1024x256`, `2048x512`, `4096x512`, `4096x1024`. Flash sanity 通过或 no_ref; GDN 有部分 3090 vs 2080Ti kernel 数值差异导致 strict sanity invalid, 因此 GDN 数字需要带 caveat 看, 但通过 sanity 的 `gdn-h2-ev16` 仍明显低于 Flash.

`4096x1024` 表示 `input_seq_len=4096`, `num_kv_pairs=1024`. 这个 slice 的 `1025-2048` 和 `2049-4096` bucket 远超 2-block local path 可见范围.

| model | sanity | overall | 1025-2048 | 2049-4096 |
|---|---:|---:|---:|---:|
| `cb64-r16-s123` | passed | 0.468924 | 0.454906 | 0.488529 |
| `cb256-r4-s123` | passed | 0.380678 | 0.390927 | 0.368109 |
| `gdn-h2-ev16-s123` | passed | 0.084990 | 0.125882 | 0.024325 |
| `gdn-h2-ev10-s123` | invalid | 0.076709 | 0.114299 | 0.014631 |

结论: Flash 在 `1025-2048` 和 `2049-4096` 仍显著强于 GDN, 这不是 local exact path 能直接解释的距离范围, 支持 remote/VQ/GD residual 路径提供了长距离能力.

## 阶段 2: Flash eval-time local ablation

阶段 2 只对 Flash checkpoint 做 eval-time override: `full`, `local_only`, `local1`, `local4`. 这里没有重新训练, 只是改变 eval-time local/remote 配置.

`cb64-r16-s123` slice accuracy:

| slice | full | local_only | local1 | local4 |
|---|---:|---:|---:|---:|
| `1024x256` | 0.969625 | 0.000227 | 0.968711 | 0.972219 |
| `2048x512` | 0.822645 | 0.000242 | 0.820266 | 0.827703 |
| `4096x512` | 0.780719 | 0.000238 | 0.777793 | 0.785773 |
| `4096x1024` | 0.468924 | 0.000227 | 0.466941 | 0.472484 |

`cb64-r16-s123`, `4096x1024`, distance buckets:

| variant | 1025-2048 | 2049-4096 |
|---|---:|---:|
| full | 0.454906 | 0.488529 |
| local_only | 0.000221 | 0.000225 |
| local1 | 0.453101 | 0.486288 |
| local4 | 0.458563 | 0.492071 |

结论: 只保留 local exact path 几乎完全失败. 把 local tensor width 改为 1 block (`local1`) 仍基本等于 full, 改为 4 blocks (`local4`) 只有小幅提升. 这说明已训练 cb64-r16 的 longer-MQAR 长距离结果不是靠 local exact path 单独完成的.

## 阶段 3: cb64-r16 从头训练 ablation

阶段 3 在 3090 上顺序完成三个正式训练, 均为 cb64-r16 seed123/data_seed123, 4 epochs, no early stopping. summary 表记录 final `last.pt` 指标; `best.pt` 路径也在 artifact 中保留.

| variant | local blocks | remote | overall | 1024x256 | 512x128 | 512x64 | 256x64 | 128x32 | 64x16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `local-only` | 2 | false | 0.404470 | 0.000262 | 0.000219 | 0.025156 | 0.006781 | 0.203969 | 1.000000 |
| `local1` | 1 | true | 0.917821 | 0.508379 | 0.877281 | 0.977297 | 0.984453 | 0.997844 | 0.999563 |
| `local4` | 4 | true | 0.967501 | 0.758496 | 0.986875 | 0.996828 | 0.998094 | 0.999969 | 1.000000 |

`local-only` 可以解决很短的 local case, 但长配置基本失败. `local1` 只有 1-block local path, 但保留 remote path, 仍显著强于 `local-only`. `local4` final 更高, 但有训练后段回落: best validation accuracy 是 `0.991211`, 高于 final `0.967501`.

阶段 3 的结论是: local exact path 会影响训练动态和近距离表现, 放大到 4 blocks 有帮助; 但没有 remote path 的 local-only 无法学到长 MQAR, 所以 local exact path 不是长距离能力的充分原因.

## Eval-only 补强: stage3 best/last longer-MQAR bucket eval

补强实验对阶段 3 的 `local-only`, `local1`, `local4` 同时跑 `last.pt` 和 `best.pt`, 覆盖 `1024x256`, `2048x512`, `4096x512`, `4096x1024`. 这是 eval-only, 没有新训练.

| checkpoint | 1024x256 | 2048x512 | 4096x512 | 4096x1024 |
|---|---:|---:|---:|---:|
| local-only last | 0.000273 | 0.000215 | 0.000223 | 0.000244 |
| local-only best | 0.000250 | 0.000266 | 0.000203 | 0.000221 |
| local1 last | 0.510586 | 0.149480 | 0.123422 | 0.030406 |
| local1 best | 0.510586 | 0.149480 | 0.123422 | 0.030406 |
| local4 last | 0.758836 | 0.261277 | 0.181711 | 0.039805 |
| local4 best | 0.934250 | 0.755320 | 0.703703 | 0.424316 |

`4096x1024` long-distance buckets:

| checkpoint | 1025-2048 | 2049-4096 |
|---|---:|---:|
| local-only last | 0.000266 | 0.000212 |
| local1 last | 0.041846 | 0.012668 |
| local4 last | 0.058855 | 0.002350 |
| local4 best | 0.449291 | 0.391810 |

结论: local4 best 在 `1025-2048` 和 `2049-4096` 仍强, 这进一步说明长距离能力不能主要归因于 local exact path.

## Eval-only 补强: near-distance enriched eval

near-distance enriched 使用 position metadata 计算 `query_pos - value_pos`, 不通过 token value 反查位置. 它显式补足 `<=32`, `33-64`, `65-128` 三个近距离桶.

`4096x1024` near buckets:

| checkpoint | <=32 | 33-64 | 65-128 |
|---|---:|---:|---:|
| local-only last | 1.000000 | 0.504797 | 0.000668 |
| local1 last | 0.363024 | 0.381181 | 0.385232 |
| local4 last | 0.889970 | 0.867528 | 0.864350 |
| local4 best | 0.001497 | 0.002214 | 0.171066 |

解释: local-only 在 `<=32` 为 1.0, 证明 exact local path 的 very-near 优势真实存在. `33-64` 只有约 0.50, 与 block-aligned causal local path 预期一致: query 在 block 后半段时可见, 在 block 前半段时不可见. `65-128` 基本不可见.

local4 best 与 local4 last 的 near/long-distance 行为相反. `local4 last` near-local 很强但 official 4096x1024 很弱; `local4 best` long-distance/overall 强但 near-local 弱. 这说明 checkpoint selection 会改变能力侧重, 后续必须明确使用 `best.pt` 还是 `last.pt`.

## 最终回答

1. 这个假设从代码上是否成立?

成立. Flash-VQG 当前配置有 exact local attention branch; GDN baseline 没有同等 branch. 但 Flash 的当前 local path 应表述为 `2-block causal exact local path`, 不是每个 query guaranteed 的 64-token sliding window.

2. 当前 Flash-VQG 与 GDN 的比较是否可能混入 local exact attention 优势?

可能. 这种优势主要体现在 very-near distance, 尤其是 `<=32`, 以及部分受 block alignment 影响的 `33-64`.

3. local exact path 是否解释了 Flash 的长距离优势?

不能主要解释. 证据是 `local-only` 在 official longer-MQAR 上接近随机; Flash 在 `1025+` 和 `2049+` 远距离桶仍明显强于 GDN; local4 best 的 long-distance 强项也不对应 near-local 强项.

4. 哪些结果需要保留 caveat?

GDN 的部分 strict sanity 为 invalid, 需要带 3090 vs 2080Ti kernel 数值差异 caveat. local4 必须区分 `best.pt` 和 `last.pt`. 对 local window 的表述必须避免写成稳定 64-token sliding window.

5. 是否还需要 GDN+exact-local 阶段 4?

如果要最严格回应 model-class fairness reviewer, 仍建议保留阶段 4: 给 GDN 增加同等 exact-local branch 后重跑关键 eval/训练. 但当前 evidence 已足够支持: local exact path 是混杂因素, 不是 Flash longer-MQAR 长距离优势的主因.

## 推荐表述

```text
Flash-VQG 的 exact local path 会带来 very-near distance 的额外优势, 因此 Flash vs GDN 的严格公平比较应承认这个结构差异. 但当前实现是 block-aligned causal local path, local_num_blocks=2 时 guaranteed exact local history 约为 <=32 token, 33-64 是边界区. 实验证据显示, Flash 在 longer-MQAR 长距离桶上的主要优势不能由 local exact path 解释, 更支持 remote/VQ/GD residual 是关键贡献来源.
```
