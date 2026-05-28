# 2026-05-28 Flash seed stability report

## Scope

- 本报告记录 Flash-VQG `gd_residual_v1` 131k capacity seed stability follow-up.
- 训练口径固定为 `data_seed=123`, `b64_ga4`, fp32 official/default, `MAX_EPOCHS=4`, early stopping disabled.
- 本轮新增 strict official run 为 `cb256-r4-s124/s125` 和 `cb64-r16-s124/s125`. seed123 参考来自 2026-05-20 capacity decomposition official artifact.
- 不混入 historical inferred, b128_ga2, auto-fp16, probe 或失败行.

## Artifacts

- `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv`
- `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-source-manifest.csv`
- `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`

## Run Status

| config | run_id | GPU | status | wall-clock |
|---|---|---:|---|---:|
| `cb256-r4-s124` | `gd-cb256-r4-s124-d123-b64-ga4-fp32-noearly4ep` | 0 | completed | 05:13:19 |
| `cb256-r4-s125` | `gd-cb256-r4-s125-d123-b64-ga4-fp32-noearly4ep` | 1 | completed | 05:04:25 |
| `cb64-r16-s124` | `gd-cb64-r16-s124-d123-b64-ga4-fp32-noearly4ep` | 0 | completed | 04:55:36 |
| `cb64-r16-s125` | `gd-cb64-r16-s125-d123-b64-ga4-fp32-noearly4ep` | 1 | completed | 04:08:28 |

## Final Metrics

| config | valid/loss | valid/accuracy | 1024x256 | 512x128 | 512x64 | 256x64 |
|---|---:|---:|---:|---:|---:|---:|
| `cb256-r4-s124` | 0.343301 | 0.943530 | 0.675371 | 0.920258 | 0.973641 | 0.985938 |
| `cb256-r4-s125` | 0.203413 | 0.970299 | 0.834781 | 0.965297 | 0.982375 | 0.989969 |
| `cb64-r16-s124` | 0.223895 | 0.972140 | 0.819797 | 0.980398 | 0.989266 | 0.995781 |
| `cb64-r16-s125` | 0.047991 | 0.996954 | 0.987285 | 0.997797 | 0.994609 | 0.998406 |

## Three-seed Summary

Each cell is `valid/accuracy / 1024x256`.

| config family | seed123 | seed124 | seed125 | mean acc | mean 1024x256 |
|---|---:|---:|---:|---:|---:|
| `cb256-r4` | 0.980 / 0.895 | 0.944 / 0.675 | 0.970 / 0.835 | 0.965 | 0.802 |
| `cb64-r16` | 0.994 / 0.969 | 0.972 / 0.820 | 0.997 / 0.987 | 0.988 | 0.925 |

## Interpretation

- Matched seed124: `cb64-r16` beats `cb256-r4` by +0.028610 overall accuracy and +0.144426 on 1024x256.
- Matched seed125: `cb64-r16` beats `cb256-r4` by +0.026655 overall accuracy and +0.152504 on 1024x256.
- `cb256-r4` 的 seed123 强表现没有在 seed124/s125 上稳定复现, 尤其 seed124 hard slice 明显下降.
- `cb64-r16` 在 seed123/s125 很强, seed124 明显较弱但仍强于同 seed 的 `cb256-r4`. 因此它是当前 131k Flash decomposition 中更强的 practical anchor, 但不能描述为 seed-invariant.

## Caveats

- 这里只比较 Flash 131k decomposition family 内的 seed stability, 不是 GDN 等容量公平结论.
- seed 数仍为 3, 且 seed124 对 `cb64-r16` 的 hard-slice caveat 必须保留.
- 旧 ledger 中存在 historical inferred 的 `cb256-r4` s124/s125 行; 本报告新增的是 strict official `b64_ga4_fp32_official` 行, 不覆盖旧记录.
