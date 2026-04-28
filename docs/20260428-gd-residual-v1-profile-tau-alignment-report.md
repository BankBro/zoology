# gd_residual_v1 profile tau alignment report

日期: 2026-04-28

## 结论

本次只修正 `gd_residual_v1` profile 脚本的 `vq_softmax_tau` 口径. 之前 `run_smoke.sh` 和 `run_train.sh` 会通过 `VQ_SOFTMAX_TAU=0.25`, 但 `run_profile.sh -> profile_gd_residual_v1.py` 没有把 tau 传给 `build_configs`, 使 profile 结果不能严格称为 official `tau=0.25` profile.

现在 profile 入口已经对齐:

- `run_profile.sh` 显式传入 `--vq-softmax-tau "${VQ_SOFTMAX_TAU}"`.
- `profile_gd_residual_v1.py` 增加 `--vq-softmax-tau`.
- 默认值读取环境变量 `VQ_SOFTMAX_TAU`.
- 环境变量缺失时 fallback 到 `0.25`.
- `_build_config` 调用 `build_configs(..., vq_softmax_tau=float(args.vq_softmax_tau), ...)`.
- `summary["profile"]` 记录 `"vq_softmax_tau": float(args.vq_softmax_tau)`.

本次没有改模型数学, 没有优化 `grouped_chunk_torch_ref`, 没有启动 official 4 epoch, 没有重跑 baseline, 没有修改 Flash-VQG 仓库.

## 本次提交的数据文件选择

本次不新增原始 profile 数据文件. 原因是网页 ChatGPT 需要的关键实验数据已经作为小体积 artifact 被跟踪在 GitHub 仓库中. 本次只新增本说明文档, 让网页 ChatGPT 能按路径读取已有证据.

应提交和推送的本次内容:

- `docs/20260428-gd-residual-v1-profile-tau-alignment-report.md`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/profile_gd_residual_v1.py`
- `tests/test_flash_vqg_scripts.py`

不需要新增提交的数据文件:

- torch profiler trace.
- `tmp/` 目录.
- SwanLab 本地日志.
- checkpoint.
- generated launch config.
- 新的 smoke/profile 运行结果.

已有并应由网页 ChatGPT 阅读的数据文件:

- `docs/artifacts/20260427-gd-phase2-flat-gather/old-t256-rread2-b16-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-t256-rread2-b16-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/old-prof-b8-t128-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-summary.json`
- `docs/artifacts/20260427-gd-phase2-flat-gather/old-prof-b8-t128-profiler_cuda_memory_usage.txt`
- `docs/artifacts/20260427-gd-phase2-flat-gather/new-prof-b8-t128-profiler_cuda_memory_usage.txt`
- `docs/artifacts/20260427-gd-phase2-flat-gather/smoke-32x8-result.md`
- `docs/artifacts/20260427-gd-flatgather-64x4/profile-b64-t256-mb8-summary.json`
- `docs/artifacts/20260427-gd-flatgather-64x4/smoke-64x4-result.md`
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-b64-t256-mb8-summary.json`
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-summary.json`
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-profiler_cpu_time_total.txt`
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-profiler_cuda_time_total.txt`
- `docs/artifacts/20260427-gd-eventpack-v1/eventpack-v1-prof-b8-t128-profiler_cuda_memory_usage.txt`
- `docs/artifacts/20260427-gd-eventpack-v1/hotspot-summary.md`

## 背景状态

baseline 是 `dense-t025-cb256-s123-d123`, 本轮只读取 baseline, 不重跑 baseline.

baseline 前 4 epoch 很强:

| epoch | valid/loss | valid/accuracy |
| ---: | ---: | ---: |
| 1 | 1.492455 | 0.776821 |
| 2 | 0.519401 | 0.914123 |
| 3 | 0.299164 | 0.951626 |
| 4 | 0.237862 | 0.961423 |

official candidates:

- `gd-r16-wk4-mu015-t025-cb256-s123-d123`
- `gd-r16-wk4-mu01-t025-cb256-s123-d123`

official candidate 必须保持:

- `fox_gd_residual_rank=16`
- `fox_gd_residual_write_topk=4`
- `fox_remote_read_topk=2`
- `num_codebook_vectors=256`
- `vq_weight_mode=dense_softmax`
- `vq_softmax_tau=0.25`

`rank=8` 或 `write_topk=2` 只能作为 debug-only 或 bottleneck attribution, 不能作为 official candidate 或 main result.

## 已完成的工程 gate

### 初始 reference path gate

初始 reference path 暴露出显存和吞吐问题:

- `64x4` backward OOM, 需要分配 `32.00 GiB`.
- `32x8` backward OOM, 需要分配 `16.00 GiB`.
- `16x16` 能 smoke, 但 24 个 micro-batch 约 `9.7` 分钟.
- 正式 4 epoch 主训练当时没有启动.

### phase2 residual read flat gather

phase2 residual read flat gather 已完成:

- `C_sel/M_sel` 从 expanded gather 改为 compact source 上的 flat `index_select` / compact gather.
- B16,T256 `peak_reserved` 约从 `5.58 GiB` 降到 `2.15 GiB`.
- B8,T128 profiler 中 `aten::gather_backward` 约从 `1.06 GB` 降到 `2 MB`.
- `32x8` smoke 通过.
- `64x4` smoke 通过.

结论: phase2 residual read expanded gather backward materialization 基本被证实是显存热点, flat gather 应保留, 显存 gate 基本过了.

### event_pack v1

event_pack v1 已完成:

- metadata 构造放入 `no_grad` / detach 安全区.
- `rho_e` 未 detach, 保留 `zeta_pack = beta * rho_e` 的梯度.
- `logabar_pack` / `alpha_tail_pack` 改为 prefix sum 向量化.
- `M_ent_pack` group_id decode 向量化.

B64,T256,8 microbatches:

| 指标 | flat-gather baseline | eventpack-v1 |
| --- | ---: | ---: |
| avg_microbatch_sec | 197.81s | 80.82s |
| avg_backward_sec | 153.58s | 57.10s |
| peak_reserved_GB | 8.54 | 8.50 |

B8,T128 profiler 中 `gd_residual/event_pack` CPU/CUDA total 下降约 `96%`.

结论: event_pack v1 正确且有效, event_pack 已不是主要瓶颈.

## 当前仍未完成

- `gd_residual_v1` official 4 epoch 尚未启动.
- 目前只有 profile/smoke/工程优化结论, 没有 official 4 epoch 质量结论.
- smoke 不能被解读为正式训练结果.
- profile loss 下降不能被解读为模型质量结论.
- 当前主要吞吐瓶颈转向 `grouped_chunk_torch_ref` / `build_states_fox_gd_residual_grouped_chunk_torch_ref`.

eventpack-v1 后 B64,T256 仍约 `80.82s/microbatch`. 如果直接按 `64x4` full 4 epoch 线性估算:

- `704 steps/epoch * 4 microbatches/step = 2816 microbatches/epoch`.
- `2816 * 80.82s ~= 63h/epoch`.
- 4 epoch 约 `253h/run`, 即约 `10.5 days/run`.

这是线性估算, 不是正式实测. 但它足以说明不应直接把当前 profile 结果当作可以低成本跑 full 4 epoch 的证据.

## 当前 profile tau 修正的验证

搜索命令:

```bash
rg "run_profile|profile_gd_residual|vq_softmax_tau|VQ_SOFTMAX_TAU" tests zoology/experiments/flash_vqg -n
```

测试命令:

```bash
pytest tests/test_flash_vqg_scripts.py tests/test_flash_vqg_wrapper.py tests/test_flash_vqg_metrics_white_list.py -q
```

结果:

- `71 passed, 1 warning`

额外 parser 默认值检查:

- 未设置 `VQ_SOFTMAX_TAU` 时, `parse_args().vq_softmax_tau == 0.25`.

## 给网页 ChatGPT 的阅读入口

仓库:

- `BankBro/Flash-VQG`, main 分支.
- `BankBro/zoology`, main 分支.

Flash-VQG 建议阅读:

- `docs/20260425-flash-vqg-gated-delta-v1-math-plan-final.md`
- `docs/20260425-flash-vqg-gated-delta-v1-codex-blueprint.md`
- `src/flash_vqg/nn/fox/gd_residual.py`
- `src/flash_vqg/nn/attn_fox.py`
- `src/flash_vqg/nn/attn.py`
- `tests/test_fox_gd_residual_v1.py`
- `tests/test_attn_fox_compat.py`

zoology 建议阅读:

- `docs/20260425-gd-residual-v1-4epoch-t025-cb256-report.md`
- `docs/20260425-gd-residual-v1-mqar-performance-report.md`
- `docs/20260425-gd-residual-v1-reference-path-debug-plan.md`
- `docs/20260427-gd-residual-v1-phase2-flat-gather-report.md`
- `docs/20260427-gd-residual-v1-flatgather-64x4-smoke-report.md`
- `docs/20260427-gd-residual-v1-eventpack-v1-report.md`
- `docs/20260428-gd-residual-v1-profile-tau-alignment-report.md`
- `docs/artifacts/20260427-gd-phase2-flat-gather/`
- `docs/artifacts/20260427-gd-flatgather-64x4/`
- `docs/artifacts/20260427-gd-eventpack-v1/`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/common_env.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_profile.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/profile_gd_residual_v1.py`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_smoke.sh`
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh`
- `tests/test_flash_vqg_scripts.py`

## 下一步判断边界

网页 ChatGPT 做下一步建议时必须区分:

- 已实测事实.
- profile/smoke 结果.
- 线性估算.
- 尚未启动的 official 4 epoch.
- official candidate 与 debug-only 实验.

不要建议:

- 重跑 baseline.
- 改 official 超参.
- 把 `rank=8` 或 `write_topk=2` 当 official candidate.
- 把 smoke/profile 当正式质量结论.
- 重新设计 `gd_residual_v1` 数学.
- 写 Triton/CUDA/custom backward.
- 改 legacy / `clr_v1` / `clr_delta_v1` 语义.
