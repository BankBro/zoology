# gd_residual_v1 current tau=0.25 profile gate report

日期: 2026-04-28

## 结论

本次完成 `gd_residual_v1` current checkout 的 strict `tau=0.25` 最小 runtime/profile gate.

已完成:

- Flash-VQG 最小正确性测试通过.
- zoology 脚本和 wrapper 测试通过.
- B64,T256,8 microbatches profile 在 strict `vq_softmax_tau=0.25` 下完成.
- B8,T128 torch profiler 在 strict `vq_softmax_tau=0.25` 下完成.
- profile summary 明确记录 `summary["profile"]["vq_softmax_tau"] == 0.25`.
- B64,T256 所有 8 个 microbatch loss 均 finite.

未执行:

- 未重跑 baseline.
- 未启动 official full 4 epoch.
- 未改 official 超参.
- 未优化 `grouped_chunk_torch_ref`.
- 未改模型数学.
- 未跑 64x4 smoke, 因为本次 B64,T256,8 microbatches profile 已约 `80s/microbatch`, 完整 smoke 预期明显更长. 本次 smoke 是可选 gate, 因耗时跳过.

结论判断:

当前 checkout 已具备进入 `grouped_chunk_torch_ref` PyTorch reference 侧优化的条件. 原因是 strict `tau=0.25` profile 已闭环, phase2 read 和 event_pack 不再是主要 gate, profiler 热点清楚落在 `grouped_chunk_torch_ref` 及其小 op/autograd loop 上.

## 本次提交的数据文件选择

网页 ChatGPT 只能读取 GitHub 代码仓, 所以本次提交小体积 profile 证据, 不提交大体积 trace 或临时目录.

提交:

- `docs/artifacts/20260428-gd-current-t025-profile-gate/b64-t256-mb8-summary.json`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-summary.json`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-profiler_cpu_time_total.txt`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-profiler_cuda_time_total.txt`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-profiler_cuda_memory_usage.txt`

不提交:

- `tmp/20260428-gd-current-t025-prof-b8-t128/trace/`, 约 `937MB`.
- profiler self tables, 当前问题定位用 CPU total, CUDA total 和 CUDA memory tables 已足够.
- SwanLab 本地日志.
- checkpoint.
- generated launch config.
- full `tmp/` 目录.

## 仓库状态

Flash-VQG:

- branch: `20260428-gd-residual-v1-sync`
- commit: `06d4b804977c0360bb3f28676357a15b13cc5935`
- status: clean

zoology:

- branch: `flash-vqg`
- commit at run time: `9c7680be7fd193d87a9615e42f893953c0c3b7db`
- status before docs/artifact commit: clean

## 最小正确性测试

Flash-VQG:

```bash
cd /home/lyj/mnt/project/Flash-VQG
pytest tests/test_fox_gd_residual_v1.py -q
pytest tests/test_attn_fox_compat.py -q
```

结果:

- `tests/test_fox_gd_residual_v1.py`: `11 passed`.
- `tests/test_attn_fox_compat.py`: `5 passed`.

zoology:

```bash
cd /home/lyj/mnt/project/zoology
pytest tests/test_flash_vqg_scripts.py tests/test_flash_vqg_wrapper.py tests/test_flash_vqg_metrics_white_list.py -q
```

结果:

- `71 passed, 1 warning`.

## B64,T256,8 microbatches profile

命令:

```bash
cd /home/lyj/mnt/project/zoology

FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
VQ_WEIGHT_MODE=dense_softmax \
VQ_SOFTMAX_TAU=0.25 \
TRAIN_BATCH_SIZE=64 \
PROFILE_SEQ_LEN=256 \
PROFILE_MICROBATCHES=8 \
PROFILE_ENABLE_TORCH_PROFILER=0 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/20260428-gd-current-t025-b64-t256-mb8 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

结果文件:

- `docs/artifacts/20260428-gd-current-t025-profile-gate/b64-t256-mb8-summary.json`

Profile:

| 指标 | 值 |
| --- | ---: |
| `summary["profile"]["vq_softmax_tau"]` | `0.25` |
| `peak_reserved_GB` | `8.501953` |
| `peak_allocated_GB` | `6.749899` |
| `avg_microbatch_sec` | `80.005957` |
| `avg_forward_sec` | `22.903922` |
| `avg_backward_sec` | `57.025518` |
| losses finite | `true` |

Microbatch losses:

| microbatch | loss |
| ---: | ---: |
| 0 | `9.039224` |
| 1 | `8.949969` |
| 2 | `8.863255` |
| 3 | `8.774924` |
| 4 | `8.688933` |
| 5 | `8.603722` |
| 6 | `8.519915` |
| 7 | `8.437822` |

最后一条 metrics:

| metric | value |
| --- | ---: |
| `attn/gd_residual_lambda_mean` | `0.0502246469` |
| `attn/gd_residual_inject_ratio` | `0.0148134222` |
| `attn/gd_residual_write_strength_mean` | `0.1250986606` |
| `attn/gd_residual_m_norm_mean` | `0.0197490640` |
| `attn/gd_residual_m_norm_max` | `0.2396711260` |
| `attn/gd_residual_mu_valid_ratio` | `0.0` |

注意:

- 这些 loss 只用于确认 profile 过程 finite 和可运行.
- profile loss 下降不能被解读为模型质量结论.

## B8,T128 profiler

命令:

```bash
cd /home/lyj/mnt/project/zoology

GPU_ID=1 \
FOX_REMOTE_READ_TOPK=2 \
FOX_GD_RESIDUAL_RANK=16 \
FOX_GD_RESIDUAL_WRITE_TOPK=4 \
NUM_CODEBOOK_VECTORS=256 \
VQ_WEIGHT_MODE=dense_softmax \
VQ_SOFTMAX_TAU=0.25 \
TRAIN_BATCH_SIZE=8 \
PROFILE_SEQ_LEN=128 \
PROFILE_MICROBATCHES=1 \
PROFILE_ENABLE_TORCH_PROFILER=1 \
PROFILE_ENABLE_GD_DIAGNOSTICS=0 \
PROFILE_OUTPUT_DIR=tmp/20260428-gd-current-t025-prof-b8-t128 \
bash zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh
```

结果文件:

- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-summary.json`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-profiler_cpu_time_total.txt`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-profiler_cuda_time_total.txt`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/prof-b8-t128-profiler_cuda_memory_usage.txt`

Profiler summary:

| 指标 | 值 |
| --- | ---: |
| `summary["profile"]["vq_softmax_tau"]` | `0.25` |
| `peak_reserved_GB` | `0.708984` |
| `peak_allocated_GB` | `0.549201` |
| `microbatch_sec` | `13.630925` |
| `forward_sec` | `5.665889` |
| `backward_sec` | `7.917611` |
| loss | `9.033282` |

Hotspot table:

| 项 | CPU total | CUDA total | calls / memory |
| --- | ---: | ---: | ---: |
| `gd_residual/grouped_chunk` | `2.784s` | `2.783s` top-level CUDA range | `4` |
| `gd_residual/event_pack` | `86.129ms` | `84.894ms` top-level CUDA range | `4` |
| `gd_residual/phase2_residual_read` | `1.299ms` | `1.220ms` top-level CUDA range | `1` |
| `aten::copy_` | `693.915ms` | `208.572ms` | `51314` calls |
| `aten::item` | `175.629ms` | `9.906ms` | `11681` calls |
| `aten::_local_scalar_dense` | `160.941ms` | `9.906ms` | `11681` calls |
| `aten::outer` | `401.823ms` | `38.248ms` | `14336` calls |
| `aten::mv` | `432.087ms` | `39.919ms` | `14336` calls |
| `aten::matmul` | `275.406ms` | `21.326ms` | `8197` calls |
| `aten::addmv_` | `265.091ms` | `39.919ms` | `14336` calls |
| `aten::gather_backward` | `151.273us` | `11.680us` | `2.00 Mb`, `2` calls |
| `aten::index_select_backward` | `465.988us` | `751.513us` | `132.19 Mb`, `5` calls |

Additional profiler evidence:

- `aten::select_backward`: `1.395s` CPU total, `160.285ms` CUDA total, `31863` calls, `20.09 Gb` CUDA memory usage.
- `torch::autograd::CopySlices`: `437.280ms` CPU total, `138.068ms` CUDA total, `4360` calls, `24.25 Gb` CUDA memory usage.
- `aten::zero_`: `43023` calls, CUDA total about `3.340s`.
- `aten::fill_`: `43150` calls, CUDA total about `3.340s`.

判断:

- `grouped_chunk_torch_ref` 仍是当前 gd_residual 主瓶颈.
- 在 gd_residual 三个命名区间中, `grouped_chunk` 的 CUDA range `2.783s` 明显大于 `event_pack` 的 `84.894ms` 和 `phase2_residual_read` 的 `1.220ms`.
- 继续优先优化 phase2 read 或 event_pack 的收益已经不如转向 `grouped_chunk_torch_ref`.

## 64x4 smoke

未执行.

跳过原因:

- 64x4 smoke 是可选 gate.
- B64,T256,8 microbatches profile 已经约 `80s/microbatch`.
- 为避免启动长任务, 本轮只保留 strict `tau=0.25` runtime/profile baseline 和 B8 profiler 热点定位.

因此本报告不能声明 current main 的 64x4 smoke 回归已通过, 只能声明 profile/profiler gate 已完成.

## 必须避免的误读

- smoke/profile 不是正式训练质量结论.
- profile loss 下降不是模型质量结论.
- 本次没有 official 4 epoch 训练曲线.
- 本次没有重跑 baseline.
- 本次没有修改 official 超参.
- `rank=8` 或 `write_topk=2` 仍只能作为 debug-only 或 bottleneck attribution.
- 本次没有改 `gd_residual_v1` 数学.
- 本次没有写 Triton/CUDA/custom backward.
- 本次没有改 legacy / `clr_v1` / `clr_delta_v1`.

## 网页 ChatGPT 阅读入口

仓库:

- `BankBro/Flash-VQG`, main 分支.
- `BankBro/zoology`, main 分支.

Flash-VQG:

- `docs/20260425-flash-vqg-gated-delta-v1-math-plan-final.md`
- `docs/20260425-flash-vqg-gated-delta-v1-codex-blueprint.md`
- `src/flash_vqg/nn/fox/gd_residual.py`
- `src/flash_vqg/nn/attn_fox.py`
- `src/flash_vqg/nn/attn.py`
- `tests/test_fox_gd_residual_v1.py`
- `tests/test_attn_fox_compat.py`

zoology:

- `docs/20260425-gd-residual-v1-4epoch-t025-cb256-report.md`
- `docs/20260425-gd-residual-v1-mqar-performance-report.md`
- `docs/20260425-gd-residual-v1-reference-path-debug-plan.md`
- `docs/20260427-gd-residual-v1-phase2-flat-gather-report.md`
- `docs/20260427-gd-residual-v1-flatgather-64x4-smoke-report.md`
- `docs/20260427-gd-residual-v1-eventpack-v1-report.md`
- `docs/20260428-gd-residual-v1-profile-tau-alignment-report.md`
- `docs/20260428-gd-residual-v1-current-t025-profile-gate-report.md`
- `docs/artifacts/20260427-gd-phase2-flat-gather/`
- `docs/artifacts/20260427-gd-flatgather-64x4/`
- `docs/artifacts/20260427-gd-eventpack-v1/`
- `docs/artifacts/20260428-gd-current-t025-profile-gate/`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/common_env.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/profile_gd_residual_v1.py`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh`
