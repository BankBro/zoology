# 20260709-01 Flash-VQG default dropout R16 support-confidence screen plan

## 目标

验证 default dropout 下, 当前最强安全底座 `read_topk=16 + update_norm_softcap=0.5 + residual injection warmup 0->512 optimizer steps` 是否能通过 read-side support-confidence 控制进一步稳定。

本轮不是继续扫 read_topk, 也不做 D-hotspot damping. 重点验证 read support 低置信度时, 降低 residual 注入或平滑 read 权重是否能缓解跨机器分叉。

## 固定口径

- model: `cb64-r16`.
- fixed init checkpoint: `zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt`.
- expected init model_state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- data seed: `123`.
- canonical MQAR cache content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- training seeds: `125`, `124`.
- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- `fox_remote_read_topk=16`.
- `fox_gd_residual_write_topk=4`.
- `fox_gd_residual_update_norm_softcap=0.5`.
- `fox_gd_residual_update_norm_softcap_mode=smooth_p4`.
- residual injection warmup: optimizer step `0 -> 512`, train forward step `0 -> 2048`, eval policy `final`.
- `max_epochs=1`, `max_train_steps=704`, `gradient_accumulation_steps=4`.
- logger backend `none`.
- heavy read trace, hash probe, train inline event trace, D-geometry trace all disabled for formal runs.

## 实验矩阵

Variants:

| variant | change |
|---|---|
| `baseline-r16-joint` | current R16 joint baseline |
| `read-gate-r16` | baseline + `fox_gd_residual_read_confidence_gate_mode=margin_sigmoid` |
| `read-softmargin-r16` | baseline + `fox_gd_residual_read_softmargin_mode=topk_mass_temperature` |
| `read-gate-softmargin-r16` | baseline + both read confidence controls |

Read confidence parameters:

- `fox_gd_residual_read_confidence_margin_ref=0.5`.
- `fox_gd_residual_read_confidence_temp=0.25`.
- `fox_gd_residual_read_confidence_floor=0.25`.
- `fox_gd_residual_read_softmargin_tau_max=3.0`.
- `fox_gd_residual_read_softmargin_margin_ref=0.5`.
- `fox_gd_residual_read_softmargin_temp=0.25`.

Targets:

- `s125-baseline-r16-joint`
- `s125-read-gate-r16`
- `s125-read-softmargin-r16`
- `s125-read-gate-softmargin-r16`
- `s124-baseline-r16-joint`
- `s124-read-gate-r16`
- `s124-read-softmargin-r16`
- `s124-read-gate-softmargin-r16`

Machines:

- 2080ti: GPU1.
- 3090: GPU0.

Formal runs: `4 variants x 2 seeds x 2 machines = 16`.

## 执行流程

1. 本地生成脚本和 plan.
2. 静态检查:
   - `python -m py_compile support_confidence_screen.py batch_preflight.py`.
   - `bash -n run_queue.sh run_master.sh start_master.sh`.
3. 本地 config/preflight 检查:
   - target 的 read confidence config 正确透传.
   - fixed init hash match.
   - canonical MQAR cache hash match.
   - batch order hash可生成.
4. 提交并推送到 `flash-vqg`.
5. 3090 pull 到同一 commit.
6. 两机 smoke:
   - `MODE=smoke`, `SMOKE_TRAIN_STEPS=8`, `SMOKE_VALIDATION_BATCHES=16`.
   - smoke 覆盖全部 8 target.
   - 任一 smoke 失败, 不启动 formal.
7. smoke 成功后启动 formal:
   - 2080ti GPU1 queue 8 runs.
   - 3090 GPU0 queue 8 runs.
8. 监控到两边进入稳定训练阶段后退出会话等待。

## 收尾产物

任务结束后生成:

- `run-summary.csv`.
- `cross-machine-comparison.csv`.
- `variant-seed-summary.csv`.
- `variant-summary.csv`.
- `mechanism-metrics-summary.csv`.
- `cache-init-preflight-summary.csv`.
- `batch-order-summary.csv`.
- `queue-summary.csv`.
- `formal-ledger.csv`.
- `source-manifest.csv`.
- `metadata.json`.
- `README.md`.
- `docs/20260709-01-flash-vqg-default-dropout-r16-support-confidence-screen-report.md`.

## 判定标准

单个 seed/variant paired run 过线:

- 2080ti 和 3090 formal run 都 completed.
- 两机 final `1024x256` accuracy 都 `>=0.85`.
- paired gap `<=4pp`.
- 无 NaN/OOM/Traceback.
- cache/init/batch preflight 全部一致.

Variant 作为下一轮候选:

- `s125` 和 `s124` 两个 paired run 都过线.
- overall valid accuracy 不出现明显退化.
- read confidence metrics 显示机制确实启用.

失败策略:

- smoke 失败: 停止, 不启动 formal.
- formal 单个 run 失败: 记录 failed, 队列继续后续 target.
- GPU/NVML/torch CUDA 失败: 停止该机器队列.
- cache/init/batch mismatch: 停止 formal, 报告 mismatch.

