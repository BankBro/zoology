# Flash-VQG Gated-Delta Remote Memory v1

这组脚本用于落地 `dense-t025 + top2 read + den_aware + shared_den` 基线上的第一版 Gated-Delta Remote Memory 写入实验.

当前目录只做两件事:

- `write-mainline/`: 首波正式矩阵 `B1-B4`
- `diagnostics-read/`: 当 `B4` 不转正时, 用旧 reader 的两档放松读出做 mismatch 诊断

统一约定:

- logger 继续使用 `SwanLab`
- analysis 固定优先读本地产物, 即 `--analysis local`
- 正式训练入口统一通过 `zoology.experiments.flash_vqg.run_flash_vqg_suite`

建议执行顺序:

1. `write-mainline/run_write_smoke.sh`
2. `write-mainline/run_write_train.sh`
3. 只有当 `B1-B4` 不给正信号时, 再跑 `diagnostics-read/run_read_diagnostics.sh`

默认首波矩阵:

- `dense-t025-additive-s123-d123`
- `dense-t025-gated-only-s123-d123`
- `dense-t025-delta-only-s123-d123`
- `dense-t025-gated-delta-s123-d123`

主看指标:

- `valid/accuracy`
- `valid/loss`
- `valid/mqar_case/accuracy-512x128`
- `valid/mqar_case/accuracy-1024x256`
- `valid/attn/den_min`
- `valid/attn/nan_inf_count`
- `valid/attn/o_remote_energy_ratio`
- `valid/vq/relative_err_mean`
- `valid/vq/c_entropy`
- `valid/vq/write_entropy_mean`
