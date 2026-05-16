# gd_residual_v1 dynamic capacity sweep 报告

日期: 2026-05-16

## 1. 摘要

本报告补充 `gd_residual_v1` 的 dynamic residual matrix capacity ablation. 目标是回答前一轮 `gd_residual_v1 mu01 noearly4ep` 明显优于 GDN baseline 时, 这个优势是否主要来自更大的动态记忆容量.

本轮没有改代码, 没有改 dataset, 没有改 `grouped_chunk_torch_ref` 实现. 除 `NUM_CODEBOOK_VECTORS` 和 `FOX_GD_RESIDUAL_RANK` 外, 训练配置保持 official 4 epoch noearly 口径一致.

阶段结论:

- 7 个新 capacity run 全部 completed, 加上已完成的 `cb256-r16` anchor, 形成 8 点容量曲线.
- 本轮 capacity sweep 支持 high-capacity `gd_residual_v1` 很强. `cb256-r16` 仍是当前最强 high-capacity practical configuration: final `valid/accuracy=0.996`, `1024x256=0.980`.
- `cb256-r4` 是本轮新跑配置中最强: final `valid/accuracy=0.986`, `1024x256=0.938`. 但它仍只是当前单 seed 下本轮新跑配置最强, 不能直接称为稳定最优 rank.
- 等 GDN dynamic capacity 的两个配置都没有超过新 GDN `use_gate=False` baseline. `cb64-r2` final `valid/accuracy=0.869`, `1024x256=0.464`; `cb128-r1` final `valid/accuracy=0.787`, `1024x256=0.300`.
- 2x GDN capacity 下, `cb128-r2` 的 hard case `1024x256=0.710` 接近 GDN `use_gate=False` 的 `0.711`, 但 overall `valid/accuracy=0.925` 低于 GDN 的 `0.962`.
- 本轮结果将前一轮 `gd_residual_v1` 优于 GDN 的结论收紧为: high-capacity `gd_residual_v1` 是当前 MQAR 4 epoch 中最强 practical configuration; 但在与 GDN `use_gate=False` 相同的 1x dynamic capacity 下, `gd_residual_v1` 当前配置尚未超过 GDN. 因此, 当前结果不能表述为等动态容量下 `gd_residual_v1` 优于 GDN.
- 当前优势应解释为 VQ-indexed high-capacity residual memory 的实用效果, 不能单独归因于 recurrence rule 本身.

## 2. 仓库状态

Flash-VQG:

- repo: `BankBro/Flash-VQG`
- branch: `20260428-gd-residual-v1-sync`
- commit: `811e1ce5f140e97d93ad6f1adae07b95b4219143`
- relevant files:
  - `src/flash_vqg/nn/fox/gd_residual.py`
  - `tests/test_fox_gd_residual_v1.py`
  - `tests/test_attn_fox_compat.py`

zoology:

- repo: `BankBro/zoology`
- branch: `flash-vqg`
- base commit before this report: `ab7a97370c0d38f47f4b176d83443abbae74c1f2`
- this report commit: see Git history after push
- new report:
  - `docs/20260516-gd-residual-v1-capacity-sweep-report.md`
- new artifacts:
  - `docs/artifacts/20260516-gd-residual-v1-capacity-sweep/`

## 3. 共同配置

所有本轮 `gd_residual_v1` capacity run 保持:

| item | value |
|---|---|
| project | `flash_vqg_gd_residual_v1_mqar` |
| max epochs | `4` |
| train batch size | `64` |
| eval batch size | `16` |
| gradient accumulation steps | `4` |
| seed / data seed | `123 / 123` |
| d_model | `128` |
| learning rate | `1e-3` |
| validations per epoch | `2` |
| early stopping | disabled |
| formula | `gd_residual_v1` |
| read_topk / write_topk | `2 / 4` |
| builder / pack | `grouped_chunk_torch_ref / semivec_ref` |
| chunk size | `64` |
| mu_min_count | `0.1` |
| VQ score / weight / update | `codebook_dot / dense_softmax / grad` |
| VQ softmax tau | `0.25` |
| VQ topk | `4` |

只变化:

- `NUM_CODEBOOK_VECTORS`
- `FOX_GD_RESIDUAL_RANK`

其中 `cb256-r16` 是 2026-05-15 已完成的 noearly4ep anchor, 本轮没有重跑.

## 4. 容量与参数量

dynamic capacity 使用本轮讨论口径:

`2 layers * codebook slots * 64 value dim * rank`

| config | dynamic capacity | relative to GDN | trainable params |
|---|---:|---:|---:|
| `cb256-r16` | `524,288` | `32x` | `1,184,966` |
| `cb256-r8` | `262,144` | `16x` | `1,183,942` |
| `cb256-r4` | `131,072` | `8x` | `1,183,430` |
| `cb256-r2` | `65,536` | `4x` | `1,183,174` |
| `cb256-r1` | `32,768` | `2x` | `1,183,046` |
| `cb128-r2` | `32,768` | `2x` | `1,166,790` |
| `cb64-r2` | `16,384` | `1x` | `1,158,598` |
| `cb128-r1` | `16,384` | `1x` | `1,166,662` |
| `GDN use_gate=False` | `16,384` | `1x` | `1,167,878` |
| `GDN use_gate=True` | `16,384` | `1x` | `1,200,646` |

注意: 改 `rank` 更接近纯 residual matrix rank/capacity ablation. 改 `cb` 同时改变 dynamic capacity 和 VQ codebook/routing capacity, 不是纯容量变量.

## 5. Final 结果

| config | dynamic capacity | relative | valid/loss | valid/accuracy | 1024x256 acc |
|---|---:|---:|---:|---:|---:|
| `cb256-r16` | `524,288` | `32x` | `0.0672` | `0.996` | `0.980` |
| `cb256-r8` | `262,144` | `16x` | `0.280` | `0.957` | `0.737` |
| `cb256-r4` | `131,072` | `8x` | `0.0961` | `0.986` | `0.938` |
| `cb256-r2` | `65,536` | `4x` | `0.265` | `0.946` | `0.776` |
| `cb256-r1` | `32,768` | `2x` | `0.637` | `0.864` | `0.492` |
| `cb128-r2` | `32,768` | `2x` | `0.370` | `0.925` | `0.710` |
| `cb64-r2` | `16,384` | `1x` | `0.681` | `0.869` | `0.464` |
| `cb128-r1` | `16,384` | `1x` | `1.000` | `0.787` | `0.300` |
| `GDN use_gate=False` | `16,384` | `1x` | `0.345` | `0.962` | `0.711` |
| `GDN use_gate=True` | `16,384` | `1x` | `0.729` | `0.884` | `0.334` |

说明:

- 表内 `GDN` 两行来自 2026-05-15 已完成的同 project noearly4ep baseline.
- `cb256-r16` 来自已完成 noearly4ep anchor, 未在本轮重跑.
- CSV artifacts 中保留了 8 次 validation history 和 final gd/vq metrics.

## 6. 关键观察

### 6.1 等动态容量下, gd_residual_v1 没有超过 GDN use_gate=False

1x GDN capacity:

| model/config | valid/accuracy | 1024x256 acc |
|---|---:|---:|
| `GDN use_gate=False` | `0.962` | `0.711` |
| `cb64-r2` | `0.869` | `0.464` |
| `cb128-r1` | `0.787` | `0.300` |

这说明之前高容量 `gd_residual_v1` 的优势不能简单归因为结构本身在等动态容量下优于 GDN.

### 6.2 2x capacity 已接近 GDN hard case, 但 overall 仍落后

2x GDN capacity:

| config | valid/accuracy | 1024x256 acc |
|---|---:|---:|
| `cb256-r1` | `0.864` | `0.492` |
| `cb128-r2` | `0.925` | `0.710` |
| `GDN use_gate=False` | `0.962` | `0.711` |

`cb128-r2` 是 2x GDN dynamic capacity. 它的 hard case `1024x256=0.710`, 几乎追平 GDN `use_gate=False` 的 `0.711`. 但它的 overall `valid/accuracy=0.925`, 仍低于 GDN `use_gate=False` 的 `0.962`.

这说明低容量 `gd_residual_v1` 已经有一定 hard-case long-context recall 潜力, 但整体 slice-level 稳定性仍不足. Hard-case 接近不等于 overall 追平. 后续需要补完整 slice-level analysis, 尤其是 `input_seq_len/*`, `num_kv_pairs/*`, `mqar_case/*`, 判断到底是哪类 slice 拉低 overall accuracy.

### 6.3 8x capacity 的 cb256-r4 已明显强于新 GDN baseline

`cb256-r4` final `valid/accuracy=0.986`, `1024x256=0.938`, 已明显强于新 GDN `use_gate=False` 的 `0.962 / 0.711`. 这说明 `gd_residual_v1` 不一定需要 32x capacity 才能取得强结果, 但至少本轮 1x/2x 还不足以稳定超过 GDN. 当前应将 `cb256-r4` 称为本轮单 seed 下最强新跑配置, 不能称为稳定最优 rank.

### 6.4 cb256 rank sweep 非单调

固定 `cb=256` 下:

| config | rank | valid/accuracy | 1024x256 acc |
|---|---:|---:|---:|
| `cb256-r16` | `16` | `0.996` | `0.980` |
| `cb256-r8` | `8` | `0.957` | `0.737` |
| `cb256-r4` | `4` | `0.986` | `0.938` |
| `cb256-r2` | `2` | `0.946` | `0.776` |
| `cb256-r1` | `1` | `0.864` | `0.492` |

单 seed 下 `cb256-r4` 明显强于 `cb256-r8`. 但在复跑或多 seed 前, 不能直接得出 `rank=4` 稳定优于 `rank=8`.

该非单调可能来自 seed 波动, 优化动态, rank 与 routing/codebook 交互, residual branch 使用强度差异, 或某些 gd metrics 的训练动态差异. 在正式结论前建议复查 SwanLab 曲线, 并至少复跑 `cb256-r4` 和 `cb256-r8` 的一个额外 seed 或复现实验.

### 6.5 两类 ablation 要分开解释

A. 固定 `cb=256` 的 rank sweep:

- 主要看 residual matrix rank/capacity 对质量的影响.
- 但当前 `r4 > r8` 非单调, 需要复跑或多 seed 后才能判断是否稳定.

B. 固定 dynamic capacity 的 `cb/rank` 分解对照:

- 这不是纯容量实验.
- `cb` 改变 VQ codebook/routing 粒度.
- `rank` 改变每个 code slot 的 residual correction 表达能力.
- 当前低容量下 `rank=2` 明显强于 `rank=1`.

| dynamic capacity | rank=2 config | result | rank=1 config | result |
|---:|---|---|---|---|
| `32,768` | `cb128-r2` | `0.925 / 0.710` | `cb256-r1` | `0.864 / 0.492` |
| `16,384` | `cb64-r2` | `0.869 / 0.464` | `cb128-r1` | `0.787 / 0.300` |

这提示低容量 `gd_residual_v1` 的瓶颈不只是 codebook slots, rank 表达能力也很关键.

## 7. Fairness 结论更新

### 7.1 Practical result

- `cb256-r16` / `cb256-r4` 等较高容量 `gd_residual_v1` 配置在 MQAR 上显著强于 GDN baseline.
- `cb256-r16` 是当前最强结果: `valid/accuracy=0.996`, `1024x256=0.980`.
- `cb256-r4` 在 8x GDN capacity 下已经明显超过 GDN `use_gate=False`: `valid/accuracy=0.986` vs `0.962`, `1024x256=0.938` vs `0.711`.

因此, 前一轮可以收紧为:

> 在相近 trainable parameter 数量, 相同训练预算和同一任务下, high-capacity `gd_residual_v1` practical configurations 明显优于新补跑 GDN baselines.

### 7.2 Equal-capacity result

- 在 1x GDN dynamic capacity 下, `cb64-r2` 和 `cb128-r1` 均没有超过 GDN `use_gate=False`.
- `cb64-r2`: `valid/accuracy=0.869`, `1024x256=0.464`.
- `cb128-r1`: `valid/accuracy=0.787`, `1024x256=0.300`.
- GDN `use_gate=False`: `valid/accuracy=0.962`, `1024x256=0.711`.

因此, 本轮不支持把结果解释成:

> 在相同 1x dynamic capacity budget 下, `gd_residual_v1` 已经胜过 GDN.

更严谨的 caveat 是:

> 当前优势依赖更大的 dynamic residual matrix capacity. 当 dynamic capacity 降到 GDN 1x 时, `gd_residual_v1` 没有超过 GDN `use_gate=False`; 当 capacity 提升到 8x 时, `cb256-r4` 已明显超过新 GDN baseline.

不应表述为:

- 在 equal dynamic capacity budget 下, `gd_residual_v1` 已经胜过 GDN.
- 优势可以单独归因到 recurrence update rule.
- dynamic capacity caveat 不影响当前结论.

更严谨的当前结论:

> `gd_residual_v1` 的 high-capacity variant 是当前 MQAR official 4 epoch 中最强模型; 但结构优势与动态容量优势尚未完全解耦. 本轮 capacity sweep 强化了 dynamic capacity caveat, 同时显示 `rank` 和 `codebook slots` 的分解方式本身也会影响质量.

### 7.3 cb128-r2 的 fairness 含义

`cb128-r2` 是 2x GDN dynamic capacity. 它在 hard case 上几乎追平 GDN `use_gate=False`, 但 overall accuracy 仍低:

| model/config | dynamic capacity | valid/accuracy | 1024x256 acc |
|---|---:|---:|---:|
| `cb128-r2` | `32,768` | `0.925` | `0.710` |
| `GDN use_gate=False` | `16,384` | `0.962` | `0.711` |

这说明低容量 `gd_residual_v1` 可能已经具备 long-context hard-case recall 潜力, 但 slice-level 稳定性仍不足. 不能把 hard-case 接近解释成 overall 追平, 更不能解释成等容量下已经优于 GDN.

## 8. Artifacts

本次提交到远端 GitHub 的小体积 artifacts:

- `docs/artifacts/20260516-gd-residual-v1-capacity-sweep/capacity-sweep-final.csv`
- `docs/artifacts/20260516-gd-residual-v1-capacity-sweep/capacity-sweep-validation-history.csv`
- `docs/artifacts/20260516-gd-residual-v1-capacity-sweep/capacity-sweep-epoch-end-valid.csv`
- `docs/artifacts/20260516-gd-residual-v1-capacity-sweep/capacity-and-params.csv`
- `docs/artifacts/20260516-gd-residual-v1-capacity-sweep/capacity-sweep-with-baselines.csv`
- `docs/artifacts/20260516-gd-residual-v1-capacity-sweep/run-manifest.json`

未提交:

- `tmp/` 下完整日志.
- checkpoints: `best.pt`, `last.pt`.
- SwanLab 本地运行目录.
- data cache.
- 本地临时 worker script.

## 9. 下一步建议

建议优先级:

1. 第一优先: 复跑 `cb256-r4` 和 `cb256-r8`, 至少加一个新 seed, 判断 `r4 > r8` 是否稳定. 推荐先用 `seed/data_seed=124/124`; 如果时间允许, 再加 `125/125`.
2. 第二优先: 补完整 slice-level artifact / table:
   - `input_seq_len/*`
   - `num_kv_pairs/*`
   - `mqar_case/*`
   重点解释 `cb128-r2` 为什么 hard case 接近 GDN, 但 overall accuracy 仍低.
3. 第三优先: 如果 `r4` 稳定强, 再围绕 `cb256` 做 `rank=3/4/5/6` 局部搜索.
4. 第四优先: 如果要继续 fairness, 再做 GDN capacity-up ablation. `event_pack` runtime 优化暂不优先, 因为当前主要问题是质量归因和公平性, 不是 runtime.

## 10. 可引用最终结论

本轮 capacity sweep 表明, high-capacity `gd_residual_v1` 是当前最强 practical configuration: `cb256-r16` 在 official 4 epoch noearly MQAR 口径下达到 `valid/accuracy=0.996`, hard case `1024x256=0.980`. 与此同时, sweep 也强化了 dynamic capacity caveat: 当 `gd_residual_v1` 的 dynamic residual matrix capacity 降到与 GDN `use_gate=False` 相同的 1x 水平时, `cb64-r2` 和 `cb128-r1` 均未超过 GDN. 因此, 当前结果应解释为 VQ-indexed high-capacity residual memory 的实用优势, 而不能表述为 `gd_residual_v1` 在等动态容量下已经优于 GDN. 2x capacity 的 `cb128-r2` 在 hard case 上几乎追平 GDN, 但 overall accuracy 仍低, 说明低容量 `gd_residual_v1` 具有 hard-case recall 潜力但整体 slice-level 稳定性不足. 后续应优先复跑 `cb256-r4/r8` 并补充 slice-level 分析, 再考虑 rank 局部搜索和 GDN capacity-up fairness 实验.
