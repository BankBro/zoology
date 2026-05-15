# gd_residual_v1 official 4 epoch mu01/mu015 report

Date: 2026-05-15.

## Summary

Two requested `gd_residual_v1` official runs were launched in parallel on two RTX 2080 Ti GPUs with the bucketed `grouped_chunk_torch_ref` implementation.

Both runs finished with `status=completed`. `mu015` ran through all 4 epochs. `mu01` finished normally, but it triggered the existing official early stopping rule after the first epoch-end validation:

`Early stopping triggered at epoch 0 with valid/accuracy 0.99161279296875 > 0.99`

This means `mu01` is an official completed run with `MAX_EPOCHS=4`, but it does not have epoch 2-4 validation points because the unmodified training script stopped it early.

No baseline was rerun. The baseline run id for comparison is `dense-t025-cb256-s123-d123`. Official hyperparameters were not changed. Code was not modified.

## Repo and checks

Flash-VQG:

- branch: `20260428-gd-residual-v1-sync`
- commit: `811e1ce5f140e97d93ad6f1adae07b95b4219143`
- relevant files:
  - `src/flash_vqg/nn/fox/gd_residual.py`
  - `tests/test_fox_gd_residual_v1.py`

zoology:

- branch: `flash-vqg`
- commit: `d9b60e9cd079d74a2325e22f09b8b1c7c0448d12`
- relevant files:
  - `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-implementation-plan.md`
  - `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-report.md`
  - `docs/20260514-gd-residual-v1-bucketed-smoke-pilot-report.md`
  - `docs/20260514-gd-residual-v1-official-4epoch-mu01-mu015-report.md`
  - `docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/`

Pre-run correctness:

- `pytest tests/test_fox_gd_residual_v1.py -q`: `17 passed`
- `pytest tests/test_attn_fox_compat.py -q`: `5 passed`

GPU pre-check:

- GPU0 and GPU1 were idle before launch.
- After completion, both GPUs were idle again.

## Run configuration

Common official configuration:

- `MAX_EPOCHS=4`
- `TRAIN_BATCH_SIZE=64`
- `EVAL_BATCH_SIZE=16`
- `GRADIENT_ACCUMULATION_STEPS=4`
- `SEED_VALUES=123`
- `DATA_SEED=123`
- `DMODEL=128`
- `LR=1e-3`
- `NUM_CODEBOOK_VECTORS=256`
- `FOX_REMOTE_FORMULA=gd_residual_v1`
- `FOX_REMOTE_READ_TOPK=2`
- `FOX_GD_RESIDUAL_RANK=16`
- `FOX_GD_RESIDUAL_WRITE_TOPK=4`
- `FOX_GD_RESIDUAL_BUILDER=grouped_chunk_torch_ref`
- `FOX_GD_RESIDUAL_PACK_MODE=semivec_ref`
- `FOX_GD_RESIDUAL_CHUNK_SIZE=64`
- `VQ_SCORE_MODE=codebook_dot`
- `VQ_WEIGHT_MODE=dense_softmax`
- `VQ_UPDATE_MODE=grad`
- `VQ_SOFTMAX_TAU=0.25`
- `VQ_TOPK=4`

The run config also kept the existing script defaults:

- `validations_per_epoch=2`
- `early_stopping_metric=valid/accuracy`
- `early_stopping_threshold=0.99`

## Run status

| Candidate | GPU | status | wall-clock from history | valid checkpoints | epoch-end checkpoints | early stop |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `mu01` | 0 | completed | 01:15:05 | 2 | 1 | yes |
| `mu015` | 1 | completed | 05:21:39 | 8 | 4 | no |

Baseline:

- run id: `dense-t025-cb256-s123-d123`
- status in this report: not rerun

Run URLs:

- `mu01`: https://swanlab.cn/@scu-mclab/flash_vqg_gd_residual_v1_mqar/runs/p86rgd8uc5mug3t5sazku
- `mu015`: https://swanlab.cn/@scu-mclab/flash_vqg_gd_residual_v1_mqar/runs/sa9nuwu76b3pzckiek89v

## Epoch-end validation

The training script used two validations per epoch. The table below reports epoch-end validation checkpoints only. Mid-epoch checkpoints are preserved in `valid-checkpoints.csv`.

| Candidate | Epoch 1 valid/loss | Epoch 1 valid/accuracy | Epoch 2 valid/loss | Epoch 2 valid/accuracy | Epoch 3 valid/loss | Epoch 3 valid/accuracy | Epoch 4 valid/loss | Epoch 4 valid/accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mu01` | 0.165210 | 0.991613 | n/a | n/a | n/a | n/a | n/a | n/a |
| `mu015` | 0.442068 | 0.960777 | 0.241778 | 0.977436 | 0.187398 | 0.983247 | 0.163073 | 0.984996 |

`mu01` stopped after epoch 1 because `valid/accuracy` exceeded the default threshold. `mu015` continued through epoch 4 and improved smoothly, but did not reach 0.99.

## Final gd_residual metrics

| Metric | `mu01` final | `mu015` final |
| --- | ---: | ---: |
| `valid/attn/gd_residual_write_strength_mean` | 0.050733 | 0.066201 |
| `valid/attn/gd_residual_m_norm_mean` | 0.021095 | 0.088899 |
| `valid/attn/gd_residual_m_norm_max` | 3.997316 | 5.270032 |
| `valid/attn/gd_residual_mu_valid_ratio` | 0.410892 | 0.215953 |
| `valid/attn/gd_residual_lambda_mean` | 0.189539 | 0.064790 |
| `valid/attn/gd_residual_inject_ratio` | 0.385343 | 0.295928 |

The `mu01` final point is its epoch 1 end checkpoint. The `mu015` final point is its epoch 4 end checkpoint.

## Final VQ metrics

These values are from the final validation terminal line. The exact structured-history subset is also in `official-4epoch-key-metrics.json`.

| Metric | `mu01` final | `mu015` final |
| --- | ---: | ---: |
| `valid/vq/k_norm_mean` | 2.83 | 2.83 |
| `valid/vq/k_hat_norm_mean` | 2.65 | 2.79 |
| `valid/vq/relative_err_mean` | 0.143 | 0.0818 |
| `valid/vq/c_rms_mean` | 0.401 | 0.378 |
| `valid/vq/c_usage_min` | 1.58 | 0.569 |
| `valid/vq/c_usage_mean` | 20.3 | 20.3 |
| `valid/vq/c_usage_max` | 1420 | 647 |
| `valid/vq/c_entropy` | 3.67 | 3.33 |
| `valid/vq/c_usage_small_ratio` | 0.215 | 0.494 |
| `valid/vq/c_usage_large_ratio` | 0.0184 | 0.0341 |
| `valid/vq/write_entropy_mean` | 3.45 | 3.14 |
| `valid/vq/write_top1_mass_mean` | 0.286 | 0.202 |

## Interpretation

`mu01` is the stronger continuation candidate. It reached `valid/accuracy=0.991613` at the first epoch end and crossed the official early stopping threshold. It also achieved nearly the same final loss as `mu015` in much less wall-clock time.

`mu015` is stable and completed the full 4 epoch schedule, but its final `valid/accuracy=0.984996` stayed below the `0.99` threshold. It did improve monotonically across epoch-end checkpoints, so it is not a failed run. It is simply weaker than `mu01` under this setup.

The comparison is slightly asymmetric because `mu01` stopped early. If a strict four-epoch curve for `mu01` is required, it would need a separate run with early stopping disabled or threshold changed, which would be a deliberate change to the current official behavior.

## Baseline comparison

Baseline run ids:

- dense baseline: `dense-t025-cb256-s123-d123`
- gated delta net baseline: `gated_delta_net-default-s123-d123`

Dense baseline source:

- launch id: `flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45`
- run URL: https://swanlab.cn/@scu-mclab/flash_vqg_clr_v1_mainline/runs/41zwee3056ltjkegkyi28
- local history: `zoology/analysis/flash_vqg/results/flash-vqg-20260402-clr-v1-e35-codebook-sweep-t025-2026-04-09-16-15-45/dense-t025-cb256-s123-d123/data/history.csv`
- baseline status: completed
- baseline was not rerun for this report

Gated delta net baseline source:

- launch id: `flash-vqg-20260420-gdn-default-baseline-2026-04-20-08-35-50`
- run URL: https://swanlab.cn/@scu-mclab/flash_vqg_vs_gdn/runs/obaucgzwabipbxl3vz8of
- local history: `zoology/analysis/flash_vqg/results/flash-vqg-20260420-gdn-default-baseline-2026-04-20-08-35-50/gated_delta_net-default-s123-d123/data/history.csv`
- baseline status: completed
- baseline was not rerun for this report

Both baselines are 32-epoch completed runs, so there are two useful comparisons:

1. Compare against baseline epoch 4, to match the requested official 4 epoch budget.
2. Compare against baseline final epoch 32, to compare against the best available completed baseline result.

| Run/checkpoint | valid/loss | valid/accuracy | 64x4 acc | 1024x256 acc | VQ relative err |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense baseline epoch 4 | 0.237862 | 0.961423 | 0.999750 | 0.774844 | 0.079277 |
| dense baseline final epoch 32 | 0.084111 | 0.981071 | 0.999750 | 0.871535 | 0.015897 |
| GDN baseline epoch 4 | 0.268798 | 0.972832 | 1.000000 | 0.788387 | n/a |
| GDN baseline final epoch 32 | 0.072575 | 0.986256 | 1.000000 | 0.891031 | n/a |
| `mu01` epoch 1 end | 0.165210 | 0.991613 | 0.999750 | 0.961332 | 0.142552 |
| `mu015` epoch 4 end | 0.163073 | 0.984996 | 0.999750 | 0.898398 | 0.081776 |

By official 4 epoch quality metrics, the bucketed gd residual candidates beat both baselines:

- `mu01` beats dense final epoch 32 and GDN final epoch 32 on `valid/accuracy`: `0.991613` vs `0.981071` and `0.986256`.
- `mu01` beats dense final epoch 32 and GDN final epoch 32 on long-case `1024x256` accuracy: `0.961332` vs `0.871535` and `0.891031`.
- `mu015` beats dense final epoch 32 on `valid/accuracy`: `0.984996` vs `0.981071`.
- `mu015` is slightly below GDN final epoch 32 on `valid/accuracy`: `0.984996` vs `0.986256`.
- `mu015` beats dense final epoch 32 and GDN final epoch 32 on long-case `1024x256` accuracy: `0.898398` vs `0.871535` and `0.891031`.
- `mu015` beats both baselines at epoch 4 on `valid/loss`, `valid/accuracy`, and `1024x256` accuracy.

The caveat is loss and VQ reconstruction:

- dense final epoch 32 and GDN final epoch 32 have lower `valid/loss`: `0.084111` and `0.072575` vs `0.165210` for `mu01` and `0.163073` for `mu015`.
- dense final epoch 32 has much lower `valid/vq/relative_err_mean`: `0.015897` vs `0.142552` for `mu01` and `0.081776` for `mu015`.
- GDN baseline does not report VQ metrics, so VQ reconstruction cannot be compared for GDN.

Overall, for MQAR task accuracy under the requested 4 epoch official budget, `mu01` is better than both baselines and `mu015` is better than both epoch-4 baselines. Against final epoch-32 baselines, `mu01` is still better on accuracy and long-context accuracy, while `mu015` is better on long-context accuracy but slightly below GDN on overall `valid/accuracy`. For final loss after a much longer 32-epoch baseline run, both baselines remain stronger.

## Artifacts

Small artifacts were saved under:

`docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/`

Key files:

- `official-4epoch-key-metrics.json`: parsed run status, config, wall-clock, final metrics.
- `valid-checkpoints.csv`: all validation checkpoints, including mid-epoch and epoch-end.
- `epoch-end-valid.csv`: epoch-end validation checkpoints only.
- `terminal-final-validation-metrics.json`: final validation metrics parsed from terminal logs, including VQ fields not present in the structured history subset.
- `baseline-comparison.csv`: dense baseline, GDN baseline, `mu01`, and `mu015` comparison table.
- `mu01-history.csv`, `mu015-history.csv`: structured SwanLab histories.
- `mu01-summary.json`, `mu015-summary.json`: run summaries.
- `mu01-metadata.json`, `mu015-metadata.json`: config and manifest metadata.
- `mu01-run_summary.csv`, `mu015-run_summary.csv`: launch analysis summaries.
- `plots/`: selected `valid/loss` and `valid/accuracy` plots.

Raw tmux logs remain in:

- `tmp/20260514-gd-official-4epoch-logs/mu01-gpu0.log`
- `tmp/20260514-gd-official-4epoch-logs/mu015-gpu1.log`

Checkpoints, full SwanLab local run directories, generated config directories, and cached data were not copied into docs artifacts.

## Web ChatGPT reading guide

For remote-only review, read the Flash-VQG implementation and tests first, then read the zoology reports in chronological order:

1. `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-implementation-plan.md`
2. `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-report.md`
3. `docs/20260514-gd-residual-v1-bucketed-smoke-pilot-report.md`
4. `docs/20260514-gd-residual-v1-official-4epoch-mu01-mu015-report.md`

For data, the most useful compact files are:

- `docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/baseline-comparison.csv`
- `docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/epoch-end-valid.csv`
- `docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/valid-checkpoints.csv`
- `docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/official-4epoch-key-metrics.json`
- `docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/terminal-final-validation-metrics.json`

The full `mu01-history.csv` and `mu015-history.csv` are included for trace-level validation, but most analysis should start from the compact CSV/JSON files above.
