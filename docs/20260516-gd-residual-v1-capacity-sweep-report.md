# gd_residual_v1 dynamic capacity sweep 报告

日期: 2026-05-16

## 1. 摘要

本报告补充 `gd_residual_v1` 的 dynamic residual matrix capacity ablation. 目标是回答前一轮 `gd_residual_v1 mu01 noearly4ep` 明显优于 GDN baseline 时, 这个优势是否主要来自更大的动态记忆容量.

本轮没有改代码, 没有改 dataset, 没有改 `grouped_chunk_torch_ref` 实现. 除 `NUM_CODEBOOK_VECTORS` 和 `FOX_GD_RESIDUAL_RANK` 外, 训练配置保持 official 4 epoch noearly 口径一致.

阶段结论:

- 7 个新 capacity run 全部 completed, 加上已完成的 `cb256-r16` anchor, 形成 8 点容量曲线.
- `cb256-r16` 仍是最强结果: final `valid/accuracy=0.996`, `1024x256=0.980`.
- `cb256-r4` 是本轮新跑配置中最强: final `valid/accuracy=0.986`, `1024x256=0.938`.
- 等 GDN dynamic capacity 的两个配置都没有超过新 GDN `use_gate=False` baseline. `cb64-r2` final `valid/accuracy=0.869`, `1024x256=0.464`; `cb128-r1` final `valid/accuracy=0.787`, `1024x256=0.300`.
- 2x GDN capacity 下, `cb128-r2` 的 hard case `1024x256=0.710` 接近 GDN `use_gate=False` 的 `0.711`, 但 overall `valid/accuracy=0.925` 低于 GDN 的 `0.962`.
- 因此, 当前更严谨的解释是: 高容量 `gd_residual_v1` 的质量优势成立, 但等动态容量下还不能说 `gd_residual_v1` 优于 GDN.

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

`cb128-r2` 的 hard case 几乎追平 GDN `use_gate=False`, 但 overall accuracy 仍低. 这提示 `rank=2` 的 residual update 比 `rank=1` 更关键, 同时 codebook/routing 分解方式会显著影响结果.

### 6.3 8x capacity 的 cb256-r4 已明显强于新 GDN baseline

`cb256-r4` final `valid/accuracy=0.986`, `1024x256=0.938`, 已明显强于新 GDN `use_gate=False` 的 `0.962 / 0.711`. 这说明 `gd_residual_v1` 不一定需要 32x capacity 才能取得强结果, 但至少本轮 1x/2x 还不足以稳定超过 GDN.

### 6.4 cb256 rank sweep 非单调

固定 `cb=256` 下:

| config | rank | valid/accuracy | 1024x256 acc |
|---|---:|---:|---:|
| `cb256-r16` | `16` | `0.996` | `0.980` |
| `cb256-r8` | `8` | `0.957` | `0.737` |
| `cb256-r4` | `4` | `0.986` | `0.938` |
| `cb256-r2` | `2` | `0.946` | `0.776` |
| `cb256-r1` | `1` | `0.864` | `0.492` |

`r4 > r8` 不符合简单容量单调预期. 这可能来自单 seed 波动, 优化稳定性, rank 与 routing/codebook 交互, 或某些 gd metrics 的训练动态差异. 在正式结论前建议复查 SwanLab 曲线, 并至少复跑 `cb256-r4` 和 `cb256-r8` 的一个额外 seed 或复现实验.

## 7. Fairness 结论更新

前一轮可以说:

> 在相近 trainable parameter 数量, 相同训练预算和同一任务下, 高容量 `gd_residual_v1 cb256-r16` 明显优于新补跑 GDN baselines.

本轮之后应补充:

> 该优势依赖更大的 dynamic residual matrix capacity. 当 dynamic capacity 降到 GDN 1x 时, `gd_residual_v1` 没有超过 GDN `use_gate=False`; 当 capacity 提升到 8x 时, `cb256-r4` 已明显超过新 GDN baseline.

不应表述为:

> `gd_residual_v1` 在等动态容量下已经优于 GDN.

更严谨的当前结论:

> `gd_residual_v1` 的 high-capacity variant 是当前 MQAR official 4 epoch 中最强模型; 但结构优势与动态容量优势尚未完全解耦. 本轮 capacity sweep 强化了 dynamic capacity caveat, 同时显示 `rank` 和 `codebook slots` 的分解方式本身也会影响质量.

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

1. 复查 `cb256-r8` 为什么明显弱于 `cb256-r4`. 最小动作是复跑 `cb256-r4` 与 `cb256-r8` 各一个新 seed, 或复跑同 seed 确认稳定性.
2. 若目标是公平对比 GDN, 重点比较 `GDN use_gate=False` vs `cb64-r2`, `cb128-r1`, `cb128-r2`, `cb256-r2`. 当前看 1x GDN 更强, 2x gd 接近 hard case, 4x gd hard case 超过 GDN.
3. 若目标是寻找高质量低容量 gd_residual_v1, 可以围绕 `cb256-r4` 做局部搜索: `rank=3/4/5/6`, 或固定 `rank=4` 改 `cb=128/192/256`.
4. 若目标是机制归因, 建议区分两条线:
   - 固定 `cb=256`, 改 `rank`, 观察 residual matrix rank/capacity.
   - 固定 dynamic capacity, 对比不同 `cb/rank` 分解, 观察 VQ routing/codebook slots 与 rank 的交互.

