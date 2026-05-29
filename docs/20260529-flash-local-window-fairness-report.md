# 20260529 Flash local window fairness report

## 摘要

本轮检查的公平性假设从代码上成立: 当前 Flash-VQG/GD residual 配置使用 `block_len=32`, `local_num_blocks=2`, 每个 query 有 64 token 的 exact local attention 窗口, 而 GatedDeltaNet baseline 没有同等 exact local attention branch.

但阶段 1-3 的结果不支持“Flash-VQG 的长距离优势主要来自 64-token local exact window”这个解释. 更准确的结论是: local exact attention 可能影响训练动态和一部分中短距离表现, 但 long-MQAR 能力需要 remote/VQ/GD residual 路径. 关键证据是 `local-only` 在 eval-time 和从头训练时都无法解决长 MQAR, 而 `local1` 和 `local4` 只要保留 remote path 就显著强于 `local-only`.

## Artifact

- 3090 source-of-truth 分支: `codex/20260529-flash-local-window-fairness`.
- 阶段 1/2 longer-MQAR eval artifact: `docs/artifacts/longer-mqar/local-window-fairness-20260529/`.
- 阶段 3 training artifact: `docs/artifacts/20260529-flash-local-window-fairness/`.
- 阶段 3 训练 summary: `docs/artifacts/20260529-flash-local-window-fairness/stage3_train_summary.csv`.

## 代码事实

Flash-VQG wrapper 暴露并传递 `block_len` 和 `local_num_blocks`: `zoology/mixers/flash_vqg.py` 中 `local_window_len = self.local_num_blocks * self.block_len`. 现有 gd residual v1 builder 固定 `block_len=32`, `local_num_blocks=2`, 因此当前 exact local window 是 64 token.

Flash-VQG backend 的 phase2 local path 用 `W = local_num_blocks * L`, 并从最近 blocks 构造 `K_windows/V_windows`, 即 exact local attention window. GatedDeltaNet 代码只有 short convolution 和 gated delta recurrent state: 默认 `use_short_conv=True`, `conv_size=4`, 然后调用 gated delta rule kernel, 没有同等 exact local attention branch.

因此, 原始 Flash-VQG vs GDN 比较在机制上确实可能混入 local exact attention 优势. 后续判断必须看 local-only, local window ablation, 以及 distance bucket.

## 阶段 1: 现有 checkpoint longer-MQAR distance bucket eval

阶段 1 对已有 Flash/GDN checkpoint 跑 full variant 的 longer-MQAR bucket eval. 覆盖 slices: `1024x256`, `2048x512`, `4096x512`, `4096x1024`. Flash sanity 通过或 no_ref; GDN 有部分 3090 vs 2080Ti kernel 数值差异导致 strict sanity invalid, 因此 GDN 数字需要带 caveat 看, 但通过 sanity 的 `gdn-h2-ev16` 仍明显低于 Flash.

`4096x1024` full accuracy:

| model | sanity | overall | 1025-2048 | 2049-4096 |
|---|---:|---:|---:|---:|
| `cb64-r16-s123` | passed | 0.468924 | 0.454906 | 0.488529 |
| `cb256-r4-s123` | passed | 0.380678 | 0.390927 | 0.368109 |
| `gdn-h2-ev16-s123` | passed | 0.084990 | 0.125882 | 0.024325 |
| `gdn-h2-ev10-s123` | invalid | 0.076709 | 0.114299 | 0.014631 |

这个 slice 的有效样本主要在远超 64-token local window 的 buckets. Flash 在 `1025-2048` 和 `2049-4096` 仍显著强于 GDN, 支持 remote/VQ/GD residual 路径确实提供了长距离能力.

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

结论: 只保留 exact local window 几乎完全失败; 只把 local window 从 64 token 改为 32 token (`local1`) 仍基本等于 full; 改为 128 token (`local4`) 有小幅提升. 这说明当前 longer-MQAR 长距离结果不是靠 64-token local exact attention 单独完成的.

## 阶段 3: cb64-r16 从头训练 ablation

阶段 3 在 3090 上顺序完成三个正式训练, 均为 cb64-r16 seed123/data_seed123, 4 epochs, no early stopping. summary 表记录 final `last.pt` 指标; `best.pt` 路径也在 artifact 中保留.

| variant | local blocks | remote | overall | 1024x256 | 512x128 | 512x64 | 256x64 | 128x32 | 64x16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `local-only` | 2 | false | 0.404470 | 0.000262 | 0.000219 | 0.025156 | 0.006781 | 0.203969 | 1.000000 |
| `local1` | 1 | true | 0.917821 | 0.508379 | 0.877281 | 0.977297 | 0.984453 | 0.997844 | 0.999563 |
| `local4` | 4 | true | 0.967501 | 0.758496 | 0.986875 | 0.996828 | 0.998094 | 0.999969 | 1.000000 |

`local-only` 可以解决很短的 local case, 但长配置基本失败. `local1` 只有 32-token local exact window, 仍显著强于 `local-only`. `local4` final 更高, 但有训练后段回落: best validation accuracy 是 `0.991211`, 高于 final `0.967501`, 因此报告和 summary 都以 final 为正式口径, 同时保留 `best.pt`.

阶段 3 的结论是: local exact window 会影响训练速度和最终表现, 放大到 128 token 有明显帮助; 但没有 remote path 的 local-only 无法学到长 MQAR, 所以 local exact window 不是长距离能力的充分原因.

## 对五个问题的回答

1. 这个假设从代码上是否成立?

成立. Flash-VQG 当前配置有 exact local attention window, 长度为 `32 * 2 = 64` token. GDN baseline 只有 short convolution 和 gated delta recurrent state, 没有同等 exact local attention branch.

2. 当前 Flash-VQG 与 GDN 的比较是否可能混入 local exact attention 优势?

原则上可能. 原始比较并不是完全机制对称. 但现有证据显示, 长距离优势不能主要归因于这个 64-token local window: Flash 在远距离 buckets 仍强, `local_only` 在 eval 和训练中都失败, remote-on 的 `local1/local4` 才能解决长配置.

3. 最小验证实验应该怎么设计?

已完成的最小闭环是三段式: existing checkpoint distance bucket eval, Flash eval-time `local_only/local1/local4` ablation, cb64-r16 seed123 从头训练 ablation. 这个组合已经能区分“local window 单独有效”与“remote path 必要”.

如果继续补强, 下一步应优先做两个 eval-only 检查: 对阶段 3 训练出的 `local-only/local1/local4` 的 `best.pt` 和 `last.pt` 再跑同一套 longer-MQAR distance bucket eval; 以及构造 near-distance enriched eval, 显式覆盖 `<=32` 和 `33-64` buckets. 当前第一轮 longer-MQAR 主 slice 对 `<=64` 样本覆盖不足, 不能直接回答 very-near bucket 的局部优势曲线.

4. 哪些实验需要训练, 哪些只需要 eval 或已有 checkpoint?

阶段 1 和阶段 2 只需要已有 checkpoint eval, 已完成. 阶段 3 需要训练, 已完成 `local-only`, `local1`, `local4`. 后续若只是对阶段 3 checkpoint 做 longer-MQAR bucket eval, 不需要训练. 若要做 GDN+exact-local fairness baseline, 需要改模型并重新训练, 因为当前 GDN 没有可 eval-time 打开的 exact local branch.

5. 推荐顺序和归档方式是什么?

当前阶段 0-3 已完成, 已按 artifact 目录归档. 若继续, 推荐顺序是: 先跑阶段 3 checkpoint 的 longer-MQAR bucket eval, 只做 eval-only; 再按结果决定是否进入 GDN+exact-local 模型改造. 新结果仍按 `docs/artifacts/20260529-flash-local-window-fairness/` 或独立 subdir 归档, 包含 `source_checkpoints.csv`, `distance_bucket.csv`, `slice_summary.csv`, `metadata.json`, README 和 status. 如果启动 GDN+exact-local, 应在同一分支上另建清晰 artifact/report section, 并把它标为阶段 4, 不与阶段 0-3 的正式结果混表.

## 当前结论边界

- 阶段 1 的 GDN 对比有部分 strict sanity invalid, 需要读者注意 3090 vs 2080Ti FLA/GDN kernel 数值差异. 通过 sanity 的 GDN 结果仍支持同方向结论.
- 阶段 2 是 eval-time override, 不是重新训练, 因此主要说明已训练 Flash checkpoint 是否依赖 local branch.
- 阶段 3 是从头训练, 但 summary 是 official validation case, 不是 longer-MQAR distance bucket. 如需把训练 ablation 与阶段 1/2 完全同口径比较, 还需要对阶段 3 checkpoint 追加 eval-only bucket run.
- `local4` 的 best 和 final 差距说明训练后段存在回落. 后续比较 `local4` 时必须明确使用 `best.pt` 还是 `last.pt`.
