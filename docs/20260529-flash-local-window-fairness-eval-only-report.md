# Flash local window fairness eval-only 补强报告

日期: 2026-05-29.

本轮只做 eval-only 补强, 没有启动新训练. 所有结果来自 3090 机器 `mclab-3090` 的 zoology 仓库分支 `codex/20260529-flash-local-window-fairness`.

## 输入与归档

- source checkpoint: 由 `docs/artifacts/20260529-flash-local-window-fairness/stage3_train_summary.csv` 自动生成, 覆盖 `local-only`, `local1`, `local4` 的 `last.pt` 和 `best.pt`.
- longer-MQAR bucket artifact: `docs/artifacts/20260529-flash-local-window-fairness-eval-only/stage3-longer-mqar-bucket/`.
- near-distance enriched artifact: `docs/artifacts/20260529-flash-local-window-fairness-eval-only/near-distance-enriched/`.
- 每段均输出 `source_checkpoints.csv`, `slice_summary.csv`, `distance_bucket.csv`, `eval_runs.csv`, `metadata.json`, `status.md`.
- 两段正式 eval 均为 `completed`, batch size 均为 8, OOM fallback 为 0. stage3 checkpoint 没有 official longer-MQAR ref, 所以 sanity 为 `no_ref`.

## Local path 语义修正

当前配置是:

```text
block_len = 32
local_num_blocks = 2
```

它不等价于“每个 query guaranteed exact attend 最近 64 token”. 更准确说, 这是 **2-block causal exact local path**. 窗口张量宽度是 64, 包含前一个 block 和当前 block; 但 causal mask 会去掉当前 block 中 query 右侧的 token.

若 query 在当前 block 内 offset 为 `r`, `r in [0, 31]`, 它能看到的历史长度大约是 `32 + r`. 因此 `<=32` 是稳定 exact local 区间, `33-64` 是 block alignment 边界区, `65-128` 超出 `local_num_blocks=2` 的 local path.

## Stage3 checkpoint longer-MQAR overall

| checkpoint | 1024x256 | 2048x512 | 4096x512 | 4096x1024 |
|---|---:|---:|---:|---:|
| local-only last | 0.000273 | 0.000215 | 0.000223 | 0.000244 |
| local-only best | 0.000250 | 0.000266 | 0.000203 | 0.000221 |
| local1 last | 0.510586 | 0.149480 | 0.123422 | 0.030406 |
| local1 best | 0.510586 | 0.149480 | 0.123422 | 0.030406 |
| local4 last | 0.758836 | 0.261277 | 0.181711 | 0.039805 |
| local4 best | 0.934250 | 0.755320 | 0.703703 | 0.424316 |

结论: stage3 local-only checkpoint 在 official longer-MQAR 分布上接近随机, 说明只有 exact local path 不能解决远距离任务. local1 能覆盖一部分 1024x256, 但长度扩展到 4096x1024 后明显崩掉. local4 的 best/last 差异很大, `best.pt` 在所有 longer slices 上显著强于 `last.pt`, 因此 local4 需要同时报告 best 和 last, 不能只看 final.

## 4096x1024 long-distance buckets

| checkpoint | 1025-2048 | 2049-4096 |
|---|---:|---:|
| local-only last | 0.000266 | 0.000212 |
| local1 last | 0.041846 | 0.012668 |
| local4 last | 0.058855 | 0.002350 |
| local4 best | 0.449291 | 0.391810 |

结论: local4 best 的优势在 `1025-2048` 和 `2049-4096` 仍然存在. 这些距离远超 2-block local path 的可见范围, 因此不能用 local exact path 直接解释. 这支持阶段 1/2 的判断: Flash 的长距离优势不只是 local exact attention.

## Near-distance enriched eval

near-distance enriched 使用 position metadata 计算 `query_pos - value_pos`. 每个样本轮换强化一个近距离桶, `<=32` 固定代表距离 31, `33-64` 在 odd distances 33-63 内均匀采样, `65-128` 在 odd distances 65-127 内均匀采样. 统计禁止通过 token value 反查 target 位置.

### 1024x256

| checkpoint | <=32 | 33-64 | 65-128 |
|---|---:|---:|---:|
| local-only last | 1.000000 | 0.475468 | 0.000972 |
| local1 last | 0.719686 | 0.748676 | 0.753098 |
| local4 last | 0.939371 | 0.914578 | 0.908627 |
| local4 best | 0.129491 | 0.119661 | 0.408748 |

### 4096x1024

| checkpoint | <=32 | 33-64 | 65-128 |
|---|---:|---:|---:|
| local-only last | 1.000000 | 0.504797 | 0.000668 |
| local1 last | 0.363024 | 0.381181 | 0.385232 |
| local4 last | 0.889970 | 0.867528 | 0.864350 |
| local4 best | 0.001497 | 0.002214 | 0.171066 |

每个 near bucket 的样本量为数千级. 例如 4096x1024 中 `<=32` n=2672, `33-64` n=2710, `65-128` n=2993.

## 解释

1. local-only 在 `<=32` 为 1.0, 但在 `65-128` 接近 0, 说明 eval 确实捕捉到了 exact local path 的边界收益.
2. local-only 在 `33-64` 只有约 0.48-0.50, 这不是 bug. 这是因为 `local_num_blocks=2` 是 block-aligned causal local path, 不是 guaranteed 64-token sliding window. query 在 block 后半段时可见, 在 block 前半段时不可见, 平均后接近一半.
3. local4 last 在 near buckets 上很强, 但 official 4096x1024 很弱. 这说明 last checkpoint 更偏近距离/局部能力, 长距离能力退化.
4. local4 best 在 official longer-MQAR 上强, 但 near `<=64` 很弱, 尤其 4096x1024 的 `<=32` 和 `33-64` 近乎 0. 这说明 local4 best 的 longer-MQAR 优势主要不是靠 exact local path, 更像 remote/VQ/GD residual 路径贡献.
5. local1 best 与 last 数值相同, local4 best 与 last 差异巨大. 因此后续报告必须对 local4 明确 checkpoint 选择口径.

## 对公平性假设的回答

代码层面的假设仍成立: Flash-VQG 有 exact local attention branch, GDN baseline 没有同等分支. 当前 Flash-VQG vs GDN 比较确实可能混入 local exact attention 优势, 尤其在 `<=32` 和一部分 `33-64` 查询上.

但这次补强后, 更强的结论是: Flash 在长距离上的优势不能主要归因于 local exact path. local-only 只能解决 very-near distance, local4 best 在 `1025+` 和 `2049+` bucket 仍强, 而 near `<=64` 并不强. 这说明 remote/VQ/GD residual 机制仍是长距离收益的必要解释.

## 是否还需要 GDN+exact-local 阶段 4

建议仍保留阶段 4, 但优先级可以低于当前报告收尾. 原因是当前 evidence 已经足够说明 Flash 长距离优势不只是 local path, 但如果要做最严格的 model-class fairness, 仍应给 GDN 增加同等 exact-local branch 后重跑关键 eval, 以消除 reviewer 对 baseline local capacity 的质疑.

推荐阶段 4 最小集合: 只实现 GDN+exact-local 的 eval/训练分支, 先跑 smoke 和 near-distance enriched eval, 再决定是否做正式训练. 不建议把阶段 4 混入本 artifact, 应另建清晰子目录或后续 artifact.
