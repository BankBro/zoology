# gd_residual_v1 cb256 r4/r8 seed124 cross-GPU check

日期: 2026-05-17

## 1. 目的

本次只补 seed124 的 cross-GPU check, 用来判断前一轮 `cb256-r4` 和 `cb256-r8` 结果反转是否可能由 GPU assignment / runtime nondeterminism / 环境差异造成.

已有 seed124 paired recheck:

- `cb256-r4-s124` 跑在 GPU0, final 较弱.
- `cb256-r8-s124` 跑在 GPU1, final 较强.

本次只补两条交叉 GPU run:

- `cb256-r4-s124-d123` 跑 GPU1.
- `cb256-r8-s124-d123` 跑 GPU0.

本次没有跑 seed125, 没有做 rank=3/4/5/6 搜索, 没有重跑 baseline, 没有优化 event_pack, 没有修改模型代码.

## 2. 运行 gate

仓库状态:

| repo | branch | commit |
|---|---|---|
| `Flash-VQG` | `20260428-gd-residual-v1-sync` | `811e1ce5f140e97d93ad6f1adae07b95b4219143` |
| `zoology` | `flash-vqg` | `8c9495e6f02dee50f8a5e96df760c77bf6741504` |

correctness:

| test | result |
|---|---:|
| `pytest tests/test_fox_gd_residual_v1.py -q` | `17 passed` |
| `pytest tests/test_attn_fox_compat.py -q` | `5 passed` |

两条新 run 均使用 noearly4ep 配置, `early_stopping_metric=None`, `early_stopping_threshold=None`, `MAX_EPOCHS=4`, `VALIDATIONS_PER_EPOCH=2`.

## 3. 共同配置

| item | value |
|---|---|
| `MAX_EPOCHS` | `4` |
| `TRAIN_BATCH_SIZE` | `64` |
| `EVAL_BATCH_SIZE` | `16` |
| `GRADIENT_ACCUMULATION_STEPS` | `4` |
| `SEED_VALUES` | `124` |
| `DATA_SEED` | `123` |
| `DMODEL` | `128` |
| `LR` | `1e-3` |
| `FOX_REMOTE_FORMULA` | `gd_residual_v1` |
| `FOX_REMOTE_READ_TOPK` | `2` |
| `FOX_GD_RESIDUAL_WRITE_TOPK` | `4` |
| `FOX_GD_RESIDUAL_BUILDER` | `grouped_chunk_torch_ref` |
| `FOX_GD_RESIDUAL_PACK_MODE` | `semivec_ref` |
| `FOX_GD_RESIDUAL_CHUNK_SIZE` | `64` |
| `FOX_GD_RESIDUAL_MU_MIN_COUNT` | `0.1` |
| `NUM_CODEBOOK_VECTORS` | `256` |
| `VQ_SCORE_MODE` | `codebook_dot` |
| `VQ_WEIGHT_MODE` | `dense_softmax` |
| `VQ_UPDATE_MODE` | `grad` |
| `VQ_SOFTMAX_TAU` | `0.25` |
| `VQ_TOPK` | `4` |

只变化 rank 和 GPU:

| run | rank | GPU |
|---|---:|---:|
| `gd-r4-wk4-mu01-t025-cb256-s124-d123-noearly4ep-crossgpu-gpu1` | 4 | 1 |
| `gd-r8-wk4-mu01-t025-cb256-s124-d123-noearly4ep-crossgpu-gpu0` | 8 | 0 |

## 4. Run 状态

| config | run id | status | wall-clock |
|---|---|---:|---:|
| `cb256-r4-s124-gpu1` | `gd-r4-wk4-mu01-t025-cb256-s124-d123-noearly4ep-crossgpu-gpu1` | completed | `05:29:22` |
| `cb256-r8-s124-gpu0` | `gd-r8-wk4-mu01-t025-cb256-s124-d123-noearly4ep-crossgpu-gpu0` | completed | `05:13:15` |

对应 SwanLab run:

| config | SwanLab run |
|---|---|
| `cb256-r4-s124-gpu1` | `https://swanlab.cn/@scu-mclab/flash_vqg_gd_residual_v1_mqar/runs/ud352fm68ie04ghbquv2t` |
| `cb256-r8-s124-gpu0` | `https://swanlab.cn/@scu-mclab/flash_vqg_gd_residual_v1_mqar/runs/n49l4cmxwdpgnjkxvu438` |

## 5. Final 2x2 对照

| config | GPU0 valid/acc | GPU0 1024x256 | GPU1 valid/acc | GPU1 1024x256 |
|---|---:|---:|---:|---:|
| `cb256-r4-s124` | 0.949452 | 0.687289 | 0.955799 | 0.730594 |
| `cb256-r8-s124` | 0.982824 | 0.898813 | 0.992765 | 0.965027 |

同 GPU 内 rank 对照:

| GPU | r4 valid/acc | r8 valid/acc | r8 - r4 | r4 1024x256 | r8 1024x256 | r8 - r4 |
|---:|---:|---:|---:|---:|---:|---:|
| GPU0 | 0.949452 | 0.982824 | +0.033372 | 0.687289 | 0.898813 | +0.211523 |
| GPU1 | 0.955799 | 0.992765 | +0.036966 | 0.730594 | 0.965027 | +0.234434 |

跨 GPU 同 config 对照:

| config | GPU0 valid/acc | GPU1 valid/acc | GPU1 - GPU0 | GPU0 1024x256 | GPU1 1024x256 | GPU1 - GPU0 |
|---|---:|---:|---:|---:|---:|---:|
| `cb256-r4-s124` | 0.949452 | 0.955799 | +0.006347 | 0.687289 | 0.730594 | +0.043305 |
| `cb256-r8-s124` | 0.982824 | 0.992765 | +0.009941 | 0.898813 | 0.965027 | +0.066215 |

## 6. Epoch-end validation

| config | GPU | epoch | valid/loss | valid/accuracy | 1024x256 |
|---|---:|---:|---:|---:|---:|
| `cb256-r4-s124` | 0 | 1 | 1.582149 | 0.777299 | 0.140594 |
| `cb256-r4-s124` | 0 | 2 | 0.541403 | 0.918368 | 0.542684 |
| `cb256-r4-s124` | 0 | 3 | 0.389365 | 0.941946 | 0.647109 |
| `cb256-r4-s124` | 0 | 4 | 0.341568 | 0.949452 | 0.687289 |
| `cb256-r4-s124` | 1 | 1 | 1.641325 | 0.769345 | 0.142359 |
| `cb256-r4-s124` | 1 | 2 | 0.474671 | 0.931111 | 0.612570 |
| `cb256-r4-s124` | 1 | 3 | 0.352801 | 0.949050 | 0.695770 |
| `cb256-r4-s124` | 1 | 4 | 0.310667 | 0.955799 | 0.730594 |
| `cb256-r8-s124` | 0 | 1 | 0.945762 | 0.873206 | 0.420879 |
| `cb256-r8-s124` | 0 | 2 | 0.258402 | 0.966557 | 0.817121 |
| `cb256-r8-s124` | 0 | 3 | 0.171947 | 0.979097 | 0.878477 |
| `cb256-r8-s124` | 0 | 4 | 0.146457 | 0.982824 | 0.898813 |
| `cb256-r8-s124` | 1 | 1 | 0.720488 | 0.911697 | 0.564254 |
| `cb256-r8-s124` | 1 | 2 | 0.154914 | 0.983642 | 0.920363 |
| `cb256-r8-s124` | 1 | 3 | 0.092339 | 0.990852 | 0.954508 |
| `cb256-r8-s124` | 1 | 4 | 0.077338 | 0.992765 | 0.965027 |

## 7. Final hard-slice 对照

| slice | r4 GPU0 | r4 GPU1 | r8 GPU0 | r8 GPU1 |
|---|---:|---:|---:|---:|
| `64x4` | 0.999500 | 0.999750 | 0.999750 | 0.999750 |
| `64x8` | 0.999875 | 1.000000 | 0.999875 | 0.999875 |
| `64x16` | 0.999563 | 0.999688 | 0.999875 | 1.000000 |
| `128x32` | 0.994781 | 0.995313 | 0.997188 | 0.996031 |
| `256x64` | 0.991797 | 0.991031 | 0.994844 | 0.996766 |
| `512x64` | 0.978938 | 0.981125 | 0.990266 | 0.991609 |
| `512x128` | 0.943875 | 0.948891 | 0.981984 | 0.993063 |
| `1024x256` | 0.687289 | 0.730594 | 0.898813 | 0.965027 |

easy slices 基本都接近饱和, 差异主要集中在 `512x128` 和 `1024x256`. 在两个 GPU 上, `cb256-r8` 的 hard-slice accuracy 都明显高于 `cb256-r4`.

## 8. Final gd residual metrics

| config | GPU | write_strength | m_norm_mean | m_norm_max | mu_valid_ratio | lambda_mean | inject_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cb256-r4-s124` | 0 | 0.000510 | 0.001370 | 0.104011 | 0.323889 | 0.154536 | 0.330866 |
| `cb256-r4-s124` | 1 | 0.000598 | 0.001548 | 0.131454 | 0.334021 | 0.163780 | 0.322957 |
| `cb256-r8-s124` | 0 | 0.018446 | 0.026560 | 13.396392 | 0.364149 | 0.289273 | 0.195691 |
| `cb256-r8-s124` | 1 | 0.012735 | 0.007332 | 0.703070 | 0.362198 | 0.067582 | 0.186522 |

seed124 下 `r8` 在两个 GPU 上都更积极地写入和使用 residual state. 但 `r8` 的 `m_norm_max` 在 GPU0 上显著高于 GPU1, 说明 cross-GPU / runtime nondeterminism 确实会影响训练轨迹的幅度. 这不改变本次 rank 方向判断, 但需要作为机制解释 caveat 保留.

## 9. Final VQ metrics

| config | GPU | relative_err | c_entropy | c_usage_mean | write_entropy | write_top1_mass |
|---|---:|---:|---:|---:|---:|---:|
| `cb256-r4-s124` | 0 | 0.092305 | 3.441058 | 20.337302 | 3.027183 | 0.257043 |
| `cb256-r4-s124` | 1 | 0.086252 | 3.406604 | 20.337302 | 3.012588 | 0.303202 |
| `cb256-r8-s124` | 0 | 0.066534 | 3.427690 | 20.337302 | 3.173215 | 0.255615 |
| `cb256-r8-s124` | 1 | 0.063932 | 3.627937 | 20.337302 | 3.317474 | 0.192074 |

`r8` 在两个 GPU 上的 VQ relative error 都低于 `r4`, 与质量更强同向. 但 `c_entropy` 和 `write_top1_mass` 在 GPU0/GPU1 间有可见差异, 仍提示运行轨迹不是逐值确定的.

## 10. 问题回答

### 10.1 cross-GPU 后, r8 在 GPU0 是否仍强于 r4?

是. 在 GPU0 上, `cb256-r8-s124` final `valid/accuracy=0.982824`, `1024x256=0.898813`; `cb256-r4-s124` final `valid/accuracy=0.949452`, `1024x256=0.687289`.

差值为:

- overall accuracy: `+0.033372`.
- `1024x256`: `+0.211523`.

### 10.2 r4 换到 GPU1 后是否明显变强?

有小幅变强, 但不是足以解释 r8/r4 差距的量级.

`r4` 从 GPU0 换到 GPU1:

- overall accuracy: `0.949452 -> 0.955799`, 提升 `+0.006347`.
- `1024x256`: `0.687289 -> 0.730594`, 提升 `+0.043305`.

这个提升存在, 说明 GPU / runtime nondeterminism 会影响绝对数值. 但在 GPU1 上, `r8` 仍比 `r4` 高 `+0.036966` overall accuracy 和 `+0.234434` hard-case accuracy.

### 10.3 是否能排除 GPU assignment confound?

对 seed124 下的 rank 方向, 基本可以排除“r8 强只是因为 r8 跑在 GPU1, r4 弱只是因为 r4 跑在 GPU0”这个解释. 交换后:

- `r8` 在 GPU0 仍明显强于 `r4`.
- `r4` 在 GPU1 仍明显弱于 `r8`.

但不能说 GPU / runtime nondeterminism 完全没有影响. 同 config 跨 GPU 的 final 差异仍存在:

- `r4` GPU1 比 GPU0 高 `+0.006347` overall, `+0.043305` hard.
- `r8` GPU1 比 GPU0 高 `+0.009941` overall, `+0.066215` hard.

因此, 更严谨的结论是: GPU assignment 不能解释 seed124 下 `r8 > r4` 的 rank 方向, 但 GPU / runtime nondeterminism 可能影响绝对值和 hard-slice 幅度.

### 10.4 是否能继续执行 seed125 paired recheck?

可以. 本次 cross-GPU check 支持继续执行 seed125 paired recheck. 但 seed125 的目的仍是解决 seed / optimization dynamic 敏感性, 不是重复排查 GPU assignment.

建议 seed125 仍保持:

- `cb=256`
- `rank=4` vs `rank=8`
- `SEED_VALUES=125`
- `DATA_SEED=123`
- noearly4ep
- official common config unchanged

### 10.5 是否需要暂停 rank 搜索并排查 GPU/环境/非确定性?

不需要因为本次结果暂停全部后续实验. 但仍不建议直接进入 rank=3/4/5/6 搜索. 当前最小下一步仍应是 seed125 paired recheck, 因为 seed123 和 seed124 的 r4/r8 方向相反, 核心问题是 seed / optimization dynamic 稳定性.

只有在 seed125 也出现同 rank 跨 GPU 大幅改变方向, 或同配置复跑出现不可接受的离散跳变时, 才需要优先做 deterministic 诊断或同 GPU 顺序复跑.

## 11. 结论

本次 seed124 cross-GPU check 两条新增 run 均 completed. 结果显示, `cb256-r8-s124` 换到 GPU0 后仍明显强于 `cb256-r4-s124`; `cb256-r4-s124` 换到 GPU1 后虽然小幅变强, 但仍明显弱于 `r8`. 因此, seed124 下 `r8 > r4` 的方向基本不是 GPU assignment 造成的.

同时, 同 config 跨 GPU 的数值差异说明 runtime nondeterminism / GPU assignment 仍会影响绝对指标, 特别是 hard slice `1024x256`. 因此本次不能升级为 “r8 稳定优于 r4” 的最终结论. 结合 seed123 中 `r4 > r8`, 当前仍应判断为 `cb256` rank sweep 对 seed / optimization dynamic 敏感.

下一步应继续做最小 seed125 paired recheck, 而不是直接做 rank=3/4/5/6 搜索, GDN capacity-up, 或 event_pack runtime 优化.

## 12. Artifacts

新 artifacts:

- `docs/artifacts/20260517-gd-r4-r8-crossgpu/crossgpu-final.csv`
- `docs/artifacts/20260517-gd-r4-r8-crossgpu/crossgpu-epoch-end-valid.csv`
- `docs/artifacts/20260517-gd-r4-r8-crossgpu/crossgpu-validation-history.csv`
- `docs/artifacts/20260517-gd-r4-r8-crossgpu/crossgpu-slice-level.csv`
- `docs/artifacts/20260517-gd-r4-r8-crossgpu/crossgpu-run-manifest.json`

旧 artifacts 没有修改. 新 artifacts 里为了完整 2x2 对照, 复用了前一轮 seed124 paired recheck 的小体积指标行.
