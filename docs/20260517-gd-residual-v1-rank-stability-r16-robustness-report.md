# gd_residual_v1 rank stability / r16 robustness noearly4ep report

Date: 2026-05-17

Repositories:

- zoology: `flash-vqg`, commit `c735bce2db464b126e96b278223336bfd52ce140`
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `811e1ce5f140e97d93ad6f1adae07b95b4219143`

Artifacts:

- `docs/artifacts/20260517-gd-rank-stability-r16-robustness/rank-stability-r16-robustness-final.csv`
- `docs/artifacts/20260517-gd-rank-stability-r16-robustness/rank-stability-r16-robustness-epoch-end-valid.csv`
- `docs/artifacts/20260517-gd-rank-stability-r16-robustness/rank-stability-r16-robustness-validation-history.csv`
- `docs/artifacts/20260517-gd-rank-stability-r16-robustness/rank-stability-r16-robustness-slice-level.csv`
- `docs/artifacts/20260517-gd-rank-stability-r16-robustness/rank-stability-r16-robustness-run-manifest.json`

## 1. Scope

本轮只执行 4 个 noearly4ep run, 不改 tracked code, 不重跑 baseline, 不做 rank=3/4/5/6 搜索, 不做 GDN capacity-up, 不优化 event_pack.

两组实验:

- A. `cb256-r4-s125-d123` vs `cb256-r8-s125-d123` paired recheck.
- B. `cb256-r16-s124-d123` 和 `cb256-r16-s125-d123` robustness check.

共同训练配置保持:

- `MAX_EPOCHS=4`, `VALIDATIONS_PER_EPOCH=2`, early stopping disabled.
- `TRAIN_BATCH_SIZE=64`, `EVAL_BATCH_SIZE=16`, `GRADIENT_ACCUMULATION_STEPS=4`.
- `DATA_SEED=123`, `DMODEL=128`, `LR=1e-3`.
- `FOX_REMOTE_FORMULA=gd_residual_v1`, `FOX_REMOTE_READ_TOPK=2`.
- `FOX_GD_RESIDUAL_WRITE_TOPK=4`, `FOX_GD_RESIDUAL_BUILDER=grouped_chunk_torch_ref`, `FOX_GD_RESIDUAL_PACK_MODE=semivec_ref`, `FOX_GD_RESIDUAL_CHUNK_SIZE=64`, `FOX_GD_RESIDUAL_MU_MIN_COUNT=0.1`.
- `NUM_CODEBOOK_VECTORS=256`, `VQ_SCORE_MODE=codebook_dot`, `VQ_WEIGHT_MODE=dense_softmax`, `VQ_UPDATE_MODE=grad`, `VQ_SOFTMAX_TAU=0.25`, `VQ_TOPK=4`.

Correctness before training:

- `pytest tests/test_fox_gd_residual_v1.py -q`: 17 passed.
- `pytest tests/test_attn_fox_compat.py -q`: 5 passed.

## 2. Executive conclusion

本轮结果将 `cb256` rank 主线从 `r4` 转向 `r8/r16`:

- `cb256-r16` 仍是 high-capacity practical anchor. seed123 和 seed125 均达到约 `0.996` overall accuracy 和 `0.98+` hard-case 1024x256 accuracy.
- `cb256-r8` 在 seed124 和 seed125 中均显著强于 `cb256-r4`, 尤其在 1024x256 hard slice 上优势明显.
- `cb256-r4` 不再适合作为稳定低容量高性价比主线. 它应被降级为 seed123 下出现过强表现、但 seed124/125 未复现的非稳定候选.
- 由于 seed123 曾出现 `r4 > r8`, 当前仍不能说 `rank=8` 是稳定最优 rank. 更严谨的说法是: `r8` 是当前更值得继续跟进的中容量 follow-up candidate.

需要同时保留三个 caveat:

- GPU/runtime: 前置 cross-GPU check 已基本排除 “r8 强只是因为跑在 GPU1” 的解释, 但同 config 跨 GPU 仍有 hard-slice 差异. 本报告中的 seed/rank 结论应理解为 optimization trajectory 层面的经验结果, 不应解释为严格确定性训练规律.
- Fairness: `cb256-r8` 和 `cb256-r16` 分别是高于 GDN use_gate=False 的 16x 和 32x dynamic capacity 配置, 不能用于证明 gd_residual_v1 在等动态容量下优于 GDN.
- Seed count: 目前仍只有少数 seed. 本报告可以调整主线优先级, 但不能给出稳定 rank law.

## 3. Run status

| config | run_id | GPU | status | wall-clock |
|---|---|---:|---|---:|
| `cb256-r4-s125` | `gd-r4-wk4-mu01-t025-cb256-s125-d123-noearly4ep-r4r8-seed125` | 0 | completed | 04:42:37 |
| `cb256-r8-s125` | `gd-r8-wk4-mu01-t025-cb256-s125-d123-noearly4ep-r4r8-seed125` | 1 | completed | 04:15:27 |
| `cb256-r16-s124` | `gd-r16-wk4-mu01-t025-cb256-s124-d123-noearly4ep-r16-robustness` | 0 | completed | 05:18:23 |
| `cb256-r16-s125` | `gd-r16-wk4-mu01-t025-cb256-s125-d123-noearly4ep-r16-robustness` | 1 | completed | 04:26:06 |

## 4. Final metrics

| config | valid/loss | valid/accuracy | 1024x256 | input_len=512 | input_len=1024 | kv=128 | kv=256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cb256-r4-s125` | 0.285444 | 0.950878 | 0.739074 | 0.948602 | 0.739074 | 0.925109 | 0.739074 |
| `cb256-r8-s125` | 0.046001 | 0.996053 | 0.982844 | 0.995102 | 0.982844 | 0.996141 | 0.982844 |
| `cb256-r16-s124` | 0.167113 | 0.980533 | 0.876508 | 0.989223 | 0.876508 | 0.986383 | 0.876508 |
| `cb256-r16-s125` | 0.049594 | 0.996914 | 0.985176 | 0.996379 | 0.985176 | 0.998539 | 0.985176 |

## 5. Cross-seed rank summary

每个单元是 `valid/accuracy / 1024x256`.

| rank | seed123 | seed124 best/cross | seed125 | current positioning |
|---|---:|---:|---:|---|
| r4 | 0.986 / 0.938 | 0.956 / 0.731 | 0.951 / 0.739 | 不稳定, 不作为主线 |
| r8 | 0.957 / 0.737 | 0.993 / 0.965 | 0.996 / 0.983 | 中容量 follow-up candidate |
| r16 | 0.996 / 0.980 | 0.981 / 0.877 | 0.997 / 0.985 | high-capacity practical anchor, with hard-slice caveat |

这个总表是当前最简洁的主线判断:

- r4 的 seed123 强表现没有在 seed124/125 复现.
- r8 在 seed124/125 连续强于 r4, 因此更值得继续跟进.
- r16 在 seed123/125 是最强 high-capacity anchor, 但 seed124 的 hard slice 降幅必须保留为 caveat.

## 6. Epoch-end validation

| config | epoch | valid/loss | valid/accuracy | 1024x256 |
|---|---:|---:|---:|---:|
| `cb256-r4-s125` | 1 | 1.863558 | 0.719774 | 0.102555 |
| `cb256-r4-s125` | 2 | 0.517782 | 0.911380 | 0.567910 |
| `cb256-r4-s125` | 3 | 0.335893 | 0.942016 | 0.698004 |
| `cb256-r4-s125` | 4 | 0.285444 | 0.950878 | 0.739074 |
| `cb256-r8-s125` | 1 | 0.286019 | 0.980134 | 0.921781 |
| `cb256-r8-s125` | 2 | 0.095340 | 0.990471 | 0.960277 |
| `cb256-r8-s125` | 3 | 0.056601 | 0.994679 | 0.977418 |
| `cb256-r8-s125` | 4 | 0.046001 | 0.996053 | 0.982844 |
| `cb256-r16-s124` | 1 | 0.508355 | 0.943681 | 0.678473 |
| `cb256-r16-s124` | 2 | 0.278245 | 0.966478 | 0.788219 |
| `cb256-r16-s124` | 3 | 0.196022 | 0.977079 | 0.852695 |
| `cb256-r16-s124` | 4 | 0.167113 | 0.980533 | 0.876508 |
| `cb256-r16-s125` | 1 | 0.197054 | 0.987303 | 0.931656 |
| `cb256-r16-s125` | 2 | 0.087483 | 0.993954 | 0.968320 |
| `cb256-r16-s125` | 3 | 0.058065 | 0.996292 | 0.981137 |
| `cb256-r16-s125` | 4 | 0.049594 | 0.996914 | 0.985176 |

## 7. A. r4/r8 seed125 paired recheck

Seed125 下方向非常明确:

- `cb256-r8-s125` final `valid/accuracy=0.996053`, `1024x256=0.982844`.
- `cb256-r4-s125` final `valid/accuracy=0.950878`, `1024x256=0.739074`.
- r8-r4 差值: `+0.045174` overall accuracy, `+0.243770` 1024x256.
- r8 从 epoch1 开始就明显领先: epoch1 1024x256 `0.921781` vs r4 `0.102555`.

与已有结果对齐:

| comparison | r4 valid/accuracy | r4 1024x256 | r8 valid/accuracy | r8 1024x256 | direction |
|---|---:|---:|---:|---:|---|
| seed123 capacity sweep | 0.986 | 0.938 | 0.957 | 0.737 | r4 > r8 |
| seed124 paired, original GPU assignment | 0.949452 | 0.687289 | 0.992765 | 0.965027 | r8 > r4 |
| seed124 cross-GPU | 0.955799 | 0.730594 | 0.982824 | 0.898813 | r8 > r4 |
| seed125 paired | 0.950878 | 0.739074 | 0.996053 | 0.982844 | r8 > r4 |

本轮 seed125 支持: seed124 的 `r8 > r4` 不是孤立现象, 并且 r8 的优势主要来自 hard long-context slices.

Final slice-level:

| slice | r4-s125 | r8-s125 | r8-r4 |
|---|---:|---:|---:|
| 64x4 | 0.999500 | 1.000000 | +0.000500 |
| 64x8 | 0.999750 | 1.000000 | +0.000250 |
| 64x16 | 0.999563 | 0.999938 | +0.000375 |
| 128x32 | 0.990969 | 0.998000 | +0.007031 |
| 256x64 | 0.980969 | 0.997438 | +0.016469 |
| 512x64 | 0.972094 | 0.994063 | +0.021969 |
| 512x128 | 0.925109 | 0.996141 | +0.071031 |
| 1024x256 | 0.739074 | 0.982844 | +0.243770 |

判断:

- 不能再把 `cb256-r4` 作为稳定低容量高性价比主线. seed123 很强, 但 seed124 和 seed125 都没有复现.
- `cb256-r8` 现在是更强的 follow-up candidate: seed124 和 seed125 都强, seed125 已接近 r16 anchor.
- 但仍不应写成 “r8 稳定最优 rank”. 现有结果显示 rank/capacity 与 seed/optimization dynamic 仍有交互, 且 seed123 曾反向.
- 因此, 后续不再建议围绕 r4 做 `r3/r4/r5/r6` 搜索. 原始 r4 低容量主线已被 seed124/125 削弱.
- 若继续 rank search, 应优先围绕 `r8` 做 `r6/r8/r10` 或补 r8 多 seed, 而不是立刻回到 `r3/r4/r5/r6`.

## 8. B. r16 robustness check

已有 seed123 anchor:

- `cb256-r16-s123`: `valid/loss=0.067216`, `valid/accuracy=0.996206`, `1024x256=0.980313`.

本轮新增:

- `cb256-r16-s124`: `valid/loss=0.167113`, `valid/accuracy=0.980533`, `1024x256=0.876508`.
- `cb256-r16-s125`: `valid/loss=0.049594`, `valid/accuracy=0.996914`, `1024x256=0.985176`.

结论需要分两层:

1. Practical anchor 仍成立. `cb256-r16` 在 seed123 和 seed125 都达到约 `0.996` overall, hard 1024x256 约 `0.98+`. seed124 虽然明显弱于 seed123/125, 但仍有 `valid/accuracy=0.980533`, `1024x256=0.876508`, 仍高于此前 GDN use_gate=False baseline 的 overall `0.962`, 1024x256 `0.711`.
2. 但不能说 r16 每个 seed 都稳定在 0.996 水平. seed124 的 1024x256 从 seed123/125 的约 `0.98` 降到 `0.876508`, 表明 high-capacity 配置也存在 hard-slice optimization sensitivity.

后续报告应写 “`r16` is a strong practical anchor with hard-slice seed sensitivity”, 而不是 “`r16` is fully robust”.

Final slice-level:

| slice | r16-s124 | r16-s125 |
|---|---:|---:|
| 64x4 | 0.999750 | 0.999500 |
| 64x8 | 0.999750 | 1.000000 |
| 64x16 | 0.999938 | 1.000000 |
| 128x32 | 0.992313 | 0.999125 |
| 256x64 | 0.997563 | 0.998750 |
| 512x64 | 0.992063 | 0.994219 |
| 512x128 | 0.986383 | 0.998539 |
| 1024x256 | 0.876508 | 0.985176 |

The r16 variance is concentrated on the hardest slice. Easy and medium slices are near saturated in both seeds.

## 9. GD residual and VQ metrics

Final GD residual metrics:

| config | write_strength | m_norm_mean | m_norm_max | mu_valid_ratio | lambda_mean | inject_ratio |
|---|---:|---:|---:|---:|---:|---:|
| `cb256-r4-s125` | 0.000665 | 0.001831 | 0.172691 | 0.347335 | 0.377151 | 0.231275 |
| `cb256-r8-s125` | 0.043133 | 0.011866 | 6.544880 | 0.426779 | 0.157119 | 0.064247 |
| `cb256-r16-s124` | 0.046836 | 0.070042 | 5.696550 | 0.305833 | 0.103331 | 0.175021 |
| `cb256-r16-s125` | 0.007349 | 0.004824 | 2.103744 | 0.329901 | 0.059424 | 0.054381 |

Final VQ metrics:

| config | relative_err | c_entropy | c_usage_mean | write_entropy | write_top1_mass |
|---|---:|---:|---:|---:|---:|
| `cb256-r4-s125` | 0.080195 | 3.312728 | 20.337302 | 3.036958 | 0.292364 |
| `cb256-r8-s125` | 0.070856 | 3.443470 | 20.337302 | 3.170576 | 0.380827 |
| `cb256-r16-s124` | 0.066837 | 3.058785 | 20.337302 | 2.907896 | 0.288278 |
| `cb256-r16-s125` | 0.055126 | 3.318777 | 20.337302 | 3.131379 | 0.342931 |

Interpretation:

- r8-s125 的 residual write/state 明显强于 r4-s125, 与质量差距同向.
- r4-s125 的 `lambda_mean` 和 `inject_ratio` 更高, 但 residual state norm/write strength 极低, 说明更强注入不等于更好质量.
- r16-s124 的 residual state 很强, 但 hard slice 仍弱于 r16-s125, 说明单看 state norm 不能完成机制归因.
- r16-s125 的 VQ relative error 最低, 并且整体/hard accuracy 最强, 是本轮最健康的 high-capacity run.

## 10. Caveats

### GPU/runtime nondeterminism

前置 cross-GPU check 已基本排除 “seed124 下 r8 强只是因为跑在 GPU1, r4 弱只是因为跑在 GPU0” 的解释: r8 换到 GPU0 仍强, r4 换到 GPU1 仍弱.

但同 config 跨 GPU 仍有可见差异, 尤其在 hard slices 上. 因此本报告中的 seed/rank 结论应理解为 optimization trajectory 层面的经验结果, 不应解释为严格确定性训练规律.

### Dynamic capacity fairness

本报告不改变 dynamic capacity fairness caveat:

- `cb256-r8` 是 16x GDN use_gate=False dynamic capacity.
- `cb256-r16` 是 32x GDN use_gate=False dynamic capacity.
- 因此本报告不能用于证明 gd_residual_v1 在等动态容量下优于 GDN.

等容量结论仍以 capacity sweep 中 `cb64-r2` / `cb128-r1` 未超过 GDN use_gate=False 为准.

### Seed count

目前 r4/r8/r16 仍只有少数 seeds. 本轮可以调整实验主线优先级, 但不能给出 “rank=8 稳定最优” 或 “rank=16 完全稳定” 这类强结论.

## 11. Updated conclusion

本轮新增 4 个 noearly4ep run 均 completed.

For r4/r8:

- seed125 继续支持 `cb256-r8 > cb256-r4`.
- 结合 seed124 paired recheck 和 cross-GPU check, `r8 > r4` 已经不太像 GPU assignment artifact.
- 但 seed123 曾出现 `r4 > r8`, 因此当前严谨说法不是 “r8 稳定最优”, 而是: `cb256-r4/r8` 的相对优劣存在 seed/optimization dynamic 敏感性, 但后续证据明显转向 `r8` 是更值得跟进的 candidate.
- `rank=3/4/5/6` 局部搜索不再是优先项. 若继续 rank search, 应重心转到 `r8`-centered search, 如 `r6/r8/r10`, 或先补 r8/r16 多 seed.

For r16:

- `cb256-r16` 仍是 practical high-capacity anchor, 因为 seed123 和 seed125 都达到约 `0.996` overall 和约 `0.98+` hard 1024x256, seed124 虽弱但仍显著强于 GDN use_gate=False baseline.
- 但 r16 也不是 fully robust 或完全 seed-invariant. seed124 的 1024x256 降到 `0.876508`, 说明 hard-case performance 仍受 optimization path 影响.
- 本轮结果支持 high-capacity gd_residual_v1 很强, 但不改变 dynamic capacity fairness caveat: 这些 cb256-r8/r16 都是 16x/32x GDN dynamic capacity, 不能用来证明等动态容量下 gd_residual_v1 优于 GDN.

One-sentence conclusion:

`cb256-r8-s125` and `cb256-r16-s125` both reached near-anchor quality, while `cb256-r4-s125` did not; this makes r8 the stronger follow-up candidate than r4, but seed123/124/125 together still indicate rank/seed optimization sensitivity rather than a fully settled stable-rank law. `cb256-r16` remains the high-capacity practical anchor, with a real seed124 hard-slice robustness caveat.

## 12. Recommended next steps

1. Do not run `rank=3/4/5/6` as the immediate next search. The original motivation, stable r4 superiority, is no longer supported.
2. Prioritize r8-centered follow-up:
   - run `r6/r8/r10` under the same noearly4ep setting, or
   - add r8 multi-seed confirmation if compute is limited.
3. For r16, keep `cb256-r16` as the practical high-capacity anchor, but report seed124 hard-slice variance explicitly.
4. Do not prioritize event_pack runtime optimization yet. The current uncertainty is quality/stability/fairness, not grouped_chunk runtime.
5. Baseline was not rerun in this task. Any baseline comparison here references earlier committed reports/artifacts.
