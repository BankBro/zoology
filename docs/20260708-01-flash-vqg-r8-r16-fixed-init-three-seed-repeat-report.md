# 20260708-01 Flash-VQG R8/R16 fixed-init three-seed repeat report

## 实验目的

本轮实验用于检查上一阶段较强的联合控制配置在更多 seed / repeat 下是否稳定:

- `update_norm_softcap=0.5`, `smooth_p4`.
- `residual injection warmup`, 从 train forward step 0 线性升到 2048, 对应 512 个 optimizer step.
- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- 固定 canonical seed124 init checkpoint, 所以 `s123/s124/s125` 只表示训练 RNG seed, 不表示重新初始化模型.

重点不是继续找最高分, 而是判断 `read_topk=8` 和 `read_topk=16` 在同一 init / same cache / same batch order 下, 是否能在 2080ti 和 3090 上稳定复现。

## 固定条件

| item | value |
|---|---|
| model | `cb64-r16` |
| data seed | 123 |
| init checkpoint | canonical seed124 init |
| training seeds | 123, 124, 125 |
| repeats | 1, 2 |
| machines | 2080ti GPU1, 3090 GPU0 |
| read_topk | 8, 16 |
| write_topk | 4 |
| max epochs | 1 |
| optimizer steps | 704 |
| grad accumulation | 4 |
| heavy trace | disabled |

Formal launch timestamp: `20260707T172212Z`.

Formal run status: 24/24 completed.

MQAR cache, canonical init checkpoint, and batch order were verified before formal training. The canonical MQAR cache content hash was:

```text
d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8
```

The expected canonical init `model_state_dict` hash was:

```text
2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0
```

## 判定标准

单个 paired run 过线需要同时满足:

- 2080ti 和 3090 的 final 1024x256 accuracy 都不低于 0.85.
- 两机 gap 不超过 4 percentage points.
- 无 NaN, OOM, Traceback.

更严格的 seed/read_topk 稳定规则是: 同一个 seed/read_topk 下, 两台机器乘以两次 repeat 的 4 个 final 1024x256 结果都不低于 0.85, 且最大值和最小值差距不超过 4 percentage points。

## Cross-machine paired results

| training seed | read_topk | repeat | 2080ti final 1024x256 | 3090 final 1024x256 | gap | pass |
|---:|---:|---:|---:|---:|---:|---|
| 123 | 8 | 1 | 0.118 | 0.918 | 80.0pp | no |
| 123 | 8 | 2 | 0.713 | 0.948 | 23.5pp | no |
| 123 | 16 | 1 | 0.958 | 0.924 | 3.4pp | yes |
| 123 | 16 | 2 | 0.960 | 0.833 | 12.7pp | no |
| 124 | 8 | 1 | 0.899 | 0.948 | 4.9pp | no |
| 124 | 8 | 2 | 0.875 | 0.941 | 6.6pp | no |
| 124 | 16 | 1 | 0.886 | 0.938 | 5.2pp | no |
| 124 | 16 | 2 | 0.896 | 0.933 | 3.7pp | yes |
| 125 | 8 | 1 | 0.960 | 0.668 | 29.2pp | no |
| 125 | 8 | 2 | 0.943 | 0.710 | 23.3pp | no |
| 125 | 16 | 1 | 0.842 | 0.949 | 10.7pp | no |
| 125 | 16 | 2 | 0.843 | 0.962 | 11.9pp | no |

## Aggregate summary

| read_topk | pairs | passed pairs | mean gap | max gap | 2080ti mean | 3090 mean |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 0 | 27.9pp | 80.0pp | 0.751 | 0.855 |
| 16 | 6 | 2 | 7.9pp | 12.7pp | 0.898 | 0.923 |

`read_topk=16` is clearly better than `read_topk=8` in this screen, but it is still not stable enough. Only 2 of 6 paired runs pass, and no seed/read_topk group passes the stricter 4-run stability rule.

## Same-machine repeat stability

This round also shows that the instability is not only a simple 2080ti-vs-3090 offset. Some same-machine repeats diverge strongly:

| training seed | read_topk | machine | repeat values | spread |
|---:|---:|---|---|---:|
| 123 | 8 | 2080ti | 0.118, 0.713 | 59.5pp |
| 123 | 8 | 3090 | 0.918, 0.948 | 3.0pp |
| 123 | 16 | 2080ti | 0.958, 0.960 | 0.2pp |
| 123 | 16 | 3090 | 0.924, 0.833 | 9.1pp |
| 124 | 8 | 2080ti | 0.899, 0.875 | 2.4pp |
| 124 | 8 | 3090 | 0.948, 0.941 | 0.7pp |
| 124 | 16 | 2080ti | 0.886, 0.896 | 1.0pp |
| 124 | 16 | 3090 | 0.938, 0.933 | 0.5pp |
| 125 | 8 | 2080ti | 0.960, 0.943 | 1.7pp |
| 125 | 8 | 3090 | 0.668, 0.710 | 4.2pp |
| 125 | 16 | 2080ti | 0.842, 0.843 | 0.1pp |
| 125 | 16 | 3090 | 0.949, 0.962 | 1.3pp |

This suggests that the remaining instability is a training-trajectory sensitivity problem under default dropout and sparse residual memory, not a simple data/cache/init mismatch.

## Main conclusions

1. The joint control is useful but not sufficient.

   The previous `r16-update-softcap0p5-injwarm512` positive signal does not generalize into a stable multi-seed, repeated result. It can still produce high absolute accuracy, but it does not reliably keep the 2080ti and 3090 trajectories within the 4pp tolerance.

2. `read_topk=8` should not be promoted.

   `read_topk=8` has 0/6 paired passes and very large gaps in seeds 123 and 125. It can score high on one machine while failing on the other, so it is not a stable candidate under this protocol.

3. `read_topk=16` is the better current read support width, but still not enough.

   `read_topk=16` has higher average final 1024x256 accuracy and smaller mean gap than `read_topk=8`, but only 2/6 paired runs pass. This points to partial mitigation, not a complete solution.

4. Controlling update magnitude and delaying residual injection only addresses part of the failure chain.

   The current controls reduce some early residual-memory damage, but they do not directly stabilize which code support is read/written. The remaining failures are consistent with sparse read/write support and M_state trajectory changes still being able to amplify dropout-induced perturbations.

5. The next mechanism work should move from pure amplitude/timing control to support confidence control.

   The strongest next direction is to test mechanisms such as margin-aware read/write, read/write support guard, early support smoothing, or code/head-aware damping. These target the discrete support switch itself, rather than only shrinking the M_state update after a support choice has already happened.

## Recommended next experiments

Do not immediately run 4ep or larger seed grids from this configuration. The 1ep repeat screen already shows the candidate is not stable enough.

Recommended next step:

1. Use `read_topk=16` as the main baseline because it is stronger than `read_topk=8`.
2. Add a minimal support-confidence intervention:
   - margin-aware read: when the top-k boundary margin is too small, widen support or reduce residual injection.
   - write support guard: when write support confidence is low, reduce M_state update strength.
   - early support smoothing: during early training, avoid overly sharp read/write support.
3. Compare against the current `r16-update-softcap0p5-injwarm512` with the same fixed init, same cache, same batch order, and paired 1ep protocol.
4. Keep heavy traces disabled for formal training; if needed, use separate diagnostic runs for read/write margin and support churn.

## Artifacts

Primary artifact directory:

```text
docs/artifacts/20260708-01-flash-vqg-r8-r16-fixed-init-three-seed-repeat/
```

Key files:

- `run-summary.csv`
- `cross-machine-comparison.csv`
- `variant-seed-repeat-summary.csv`
- `within-machine-repeat-summary.csv`
- `variant-summary.csv`
- `mechanism-metrics-summary.csv`
- `cache-init-preflight-summary.csv`
- `batch-order-summary.csv`
- `formal-ledger.csv`
- `source-manifest.csv`
- `metadata.json`

