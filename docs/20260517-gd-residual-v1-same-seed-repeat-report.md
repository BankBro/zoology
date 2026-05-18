# gd_residual_v1 same-seed same-GPU repeat report

Date: 2026-05-18

Repositories:

- zoology: `flash-vqg`, commit `7f6c611aa97c5acac48ba07df53ed86fa60e4c74`
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `811e1ce5f140e97d93ad6f1adae07b95b4219143`

Artifacts:

- Canonical run-level ledger: `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`
- Ledger schema: `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.schema.md`
- `docs/artifacts/20260517-gd-same-seed-repeat/same-seed-repeat-final.csv`
- `docs/artifacts/20260517-gd-same-seed-repeat/same-seed-repeat-epoch-end-valid.csv`
- `docs/artifacts/20260517-gd-same-seed-repeat/same-seed-repeat-validation-history.csv`
- `docs/artifacts/20260517-gd-same-seed-repeat/same-seed-repeat-slice-level.csv`
- `docs/artifacts/20260517-gd-same-seed-repeat/same-seed-repeat-run-manifest.json`
- `docs/artifacts/20260517-gd-same-seed-repeat/same-seed-repeat-seed-summary.csv`
- Local snapshot: `docs/artifacts/20260517-gd-same-seed-repeat/rank-seed-effect-summary.csv`

## 1. Scope and gate

本轮只执行两个 same-seed same-GPU repeat run, 不重跑 baseline, 不做 rank=3/4/5/6 搜索, 不做 GDN capacity-up, 不优化 event_pack.

Correctness:

- zoology wrapper/scripts targeted tests: 69 passed.
- Flash-VQG `tests/test_fox_gd_residual_v1.py`: 17 passed.
- Flash-VQG `tests/test_attn_fox_compat.py`: 5 passed.

Early stopping disabled through the tracked `--disable-early-stopping true` path. Run config metadata records `early_stopping_metric=null` and `early_stopping_threshold=null`.

## 2. Run status

| config | run_id | GPU | status | wall-clock | validations |
| --- | --- | --- | --- | --- | --- |
| cb256-r8-s125-d123-repeat | gd-r8-wk4-mu01-t025-cb256-s125-d123-noearly4ep-repeat-gpu1 | 1 | completed | 03:57:31 | 8 |
| cb256-r16-s124-d123-repeat | gd-r16-wk4-mu01-t025-cb256-s124-d123-noearly4ep-repeat-gpu0 | 0 | completed | 05:09:50 | 8 |

## 3. Epoch-end validation

| config | epoch | valid/loss | valid/accuracy | 1024x256 |
| --- | --- | --- | --- | --- |
| cb256-r8-s125-d123-repeat | 1 | 0.410950 | 0.972018 | 0.887926 |
| cb256-r8-s125-d123-repeat | 2 | 0.100380 | 0.989804 | 0.956824 |
| cb256-r8-s125-d123-repeat | 3 | 0.055763 | 0.994721 | 0.976199 |
| cb256-r8-s125-d123-repeat | 4 | 0.047221 | 0.995818 | 0.981223 |
| cb256-r16-s124-d123-repeat | 1 | 0.573359 | 0.934322 | 0.618078 |
| cb256-r16-s124-d123-repeat | 2 | 0.288893 | 0.964495 | 0.777012 |
| cb256-r16-s124-d123-repeat | 3 | 0.217590 | 0.973755 | 0.830063 |
| cb256-r16-s124-d123-repeat | 4 | 0.196372 | 0.975897 | 0.846437 |

## 4. Final repeat vs original

| config | delta loss | delta accuracy | delta 1024x256 | delta 512x128 | delta VQ relative_err | delta m_norm_max | delta lambda_mean | delta inject_ratio | drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cb256-r8-s125-d123-repeat | 0.001219 | -0.000235 | -0.001621 | -0.001055 | 0.000820 | 0.748902 | 0.008935 | -0.003706 | stable |
| cb256-r16-s124-d123-repeat | 0.029259 | -0.004636 | -0.030070 | -0.003953 | -0.001539 | -0.205647 | 0.006565 | -0.053887 | moderate |

Drift rule:

- `overall difference` uses `abs(delta valid/accuracy)`. `valid/loss` is kept as a diagnostic delta, but is not used directly for the stable/moderate/high bucket because the thresholds are accuracy-scale thresholds.
- stable: overall difference < 0.005 and 1024x256 difference < 0.03.
- moderate: overall difference in 0.005-0.015 or 1024x256 difference in 0.03-0.08.
- high: overall difference > 0.015 or 1024x256 difference > 0.08.

## 5. Seed 123/124/125 summary

Each cell is `valid/loss / valid/accuracy / 1024x256`.

| seed | r4 original | r8 original | r8 repeat | r16 original | r16 repeat | summary |
| --- | --- | --- | --- | --- | --- | --- |
| 123 | 0.0961 / 0.986 / 0.938 | 0.280 / 0.957 / 0.737 | - | 0.067216 / 0.996206 / 0.980313 | - | capacity sweep 中 r4 > r8, r16 是强 anchor |
| 124 | 0.341568 / 0.949452 / 0.687289 | 0.0773385 / 0.992765 / 0.965027 | - | 0.167113 / 0.980533 / 0.876508 | 0.196372 / 0.975897 / 0.846438 | r8 > r4, r16 hard slice 弱且 repeat 仍弱 |
| 125 | 0.285444 / 0.950878 / 0.739074 | 0.046001 / 0.996053 / 0.982844 | 0.047221 / 0.995818 / 0.981223 | 0.049594 / 0.996914 / 0.985176 | - | r8 > r4, r8 repeat 稳定, r16 强 |

Notes:

- seed123 r4/r8 values come from the capacity sweep artifact, where these values are recorded at report precision.
- seed124 main row uses the original paired GPU assignment. The separate cross-GPU check also kept `r8 > r4`: r4 GPU1 `0.955799 / 0.730594`, r8 GPU0 `0.982824 / 0.898813`.
- r8 is supported by seed124/125 and the seed125 same-GPU repeat, but seed123 still prevents claiming a fully seed-invariant rank law.
- r16 remains the high-capacity practical anchor, with an explicit seed124 hard-slice caveat.
- The canonical run-level record table is `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`; the same file under this dated artifact directory is the local snapshot for this report. It is epoch4-final-only for the current rows, one complete training run per row, and keeps `num_codebook_vectors`, `replicate_id`, `run_type`, `gpu`, `run_id`, `configured_max_epochs`, and `checkpoint_label` so future repeats or epoch32 runs can be appended without overwriting earlier rows.

## 6. Final key slices

| config | slice | accuracy |
| --- | --- | --- |
| cb256-r8-s125-d123-repeat | 1024x256 | 0.981223 |
| cb256-r8-s125-d123-repeat | 512x128 | 0.995086 |
| cb256-r8-s125-d123-repeat | 512x64 | 0.994578 |
| cb256-r8-s125-d123-repeat | 256x64 | 0.998062 |
| cb256-r8-s125-d123-repeat | 128x32 | 0.997781 |
| cb256-r8-s125-d123-repeat | 64x16 | 0.999938 |
| cb256-r8-s125-d123-repeat | 64x8 | 0.999875 |
| cb256-r8-s125-d123-repeat | 64x4 | 1.000000 |
| cb256-r16-s124-d123-repeat | 1024x256 | 0.846437 |
| cb256-r16-s124-d123-repeat | 512x128 | 0.982430 |
| cb256-r16-s124-d123-repeat | 512x64 | 0.991469 |
| cb256-r16-s124-d123-repeat | 256x64 | 0.997031 |
| cb256-r16-s124-d123-repeat | 128x32 | 0.990250 |
| cb256-r16-s124-d123-repeat | 64x16 | 0.999938 |
| cb256-r16-s124-d123-repeat | 64x8 | 0.999875 |
| cb256-r16-s124-d123-repeat | 64x4 | 0.999750 |

## 7. Interpretation

- `cb256-r8-s125` original final: `valid/loss=0.046001`, `valid/accuracy=0.996053`, `1024x256=0.982844`.
- `cb256-r8-s125` repeat final: `valid/loss=0.047221`, `valid/accuracy=0.995818`, `1024x256=0.981223`.
- `cb256-r16-s124` original final: `valid/loss=0.167113`, `valid/accuracy=0.980533`, `1024x256=0.876508`.
- `cb256-r16-s124` repeat final: `valid/loss=0.196372`, `valid/accuracy=0.975897`, `1024x256=0.846437`.
- `cb256-r8-s125` repeat verdict: stable.
- `cb256-r16-s124` repeat verdict: moderate. r16-s124 repeat 未明显变强, seed124 的 hard-slice caveat 更可信.
- r8-centered rank search recommendation: 可以继续 r8-centered rank search, 优先 r6/r8/r10 或 r8 多 seed.
