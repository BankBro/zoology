# 20260708-01 Flash-VQG R8/R16 fixed-init three-seed repeat plan

## 目标

验证 default dropout 下 `update_softcap0p5 + injwarm512` joint-control 方案在固定同一份初始模型 checkpoint 时的训练稳定性。

本轮不是 seed-specific init 实验。`s123/s124/s125` 表示训练 RNG seed, 不表示初始化 seed。

## 固定口径

- fixed init checkpoint: `zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt`.
- expected init model_state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- `data_seed=123`, 保持 canonical MQAR cache 和 batch order 口径一致。
- `seed=123/124/125` 只用于训练随机路径, 主要影响 dropout/RNG。
- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- model: `cb64-r16`.
- `fox_gd_residual_write_topk=4`.
- `fox_remote_read_topk=8` 或 `16`.
- `fox_gd_residual_update_norm_softcap=0.5`.
- `fox_gd_residual_update_norm_softcap_mode=smooth_p4`.
- residual injection warmup: optimizer step `0 -> 512`, train forward step `0 -> 2048`, eval policy `final`.
- `max_epochs=1`, `max_train_steps=704`, `gradient_accumulation_steps=4`.
- logger backend `none`.
- read trace, D-geometry trace, train inline event trace 均关闭。

## 实验矩阵

Formal runs 总数: `2 read_topk x 3 seeds x 2 repeats x 2 machines = 24`.

目标列表:

- `s123-r8-rep1`, `s123-r8-rep2`, `s123-r16-rep1`, `s123-r16-rep2`.
- `s124-r8-rep1`, `s124-r8-rep2`, `s124-r16-rep1`, `s124-r16-rep2`.
- `s125-r8-rep1`, `s125-r8-rep2`, `s125-r16-rep1`, `s125-r16-rep2`.

机器:

- 2080ti: 只用 GPU1.
- 3090: 用 GPU0.

每台机器一条 queue, 单 GPU 串行跑 12 个 target。两台机器并行。

## 实施文件

- script dir: `zoology/experiments/flash_vqg/scripts/20260708-01-flash-vqg-r8-r16-fixed-init-three-seed-repeat/`.
- artifact dir: `docs/artifacts/20260708-01-flash-vqg-r8-r16-fixed-init-three-seed-repeat/`.
- report: `docs/20260708-01-flash-vqg-r8-r16-fixed-init-three-seed-repeat-report.md`.

脚本:

- `fixed_init_three_seed_repeat.py`: seed-aware fixed-init wrapper.
- `batch_preflight.py`: batch order hash preflight.
- `run_queue.sh`: 单机单 GPU 串行跑 targets.
- `run_master.sh`: 同时启动 2080ti GPU1 和 3090 GPU0 queue.
- `start_master.sh`: 后台启动 master.

## 启动流程

1. 本地完成脚本和 plan 修改。
2. 静态检查:
   - `python -m py_compile fixed_init_three_seed_repeat.py batch_preflight.py`.
   - `bash -n run_queue.sh run_master.sh start_master.sh`.
3. 本地 preflight:
   - 2080ti GPU1 `nvidia-smi` 和 torch CUDA 可用。
   - fixed init hash match.
   - canonical MQAR cache hash match.
4. 提交并推送到 `flash-vqg`.
5. 3090 pull 到同一 commit。
6. 3090 preflight:
   - 3090 GPU0 `nvidia-smi` 和 conda torch CUDA 可用。
   - fixed init hash match.
   - canonical MQAR cache hash match.
7. 运行 smoke:
   - `MODE=smoke`, `SMOKE_TRAIN_STEPS=8`, `SMOKE_VALIDATION_BATCHES=16`.
   - smoke 覆盖两台机器全部 12 个 target.
   - 任一 smoke 失败, 不启动 formal.
8. smoke 成功后启动 formal:
   - 2080ti GPU1 queue 12 个 run 自动串行。
   - 3090 GPU0 queue 12 个 run 自动串行。
9. 监控到两边进入稳定训练阶段后可以退出会话等待。

## 收尾产物

任务结束后生成:

- `run-summary.csv`: 24 个 formal run 的 status, runtime, final/best metrics.
- `variant-seed-repeat-summary.csv`: 每个 seed/read_topk 的 4-run 汇总。
- `cross-machine-comparison.csv`: repeat 对齐的 2080ti vs 3090 gap。
- `within-machine-repeat-summary.csv`: 同机 repeat spread。
- `mechanism-metrics-summary.csv`: residual read/write/state 指标。
- `cache-init-preflight-summary.csv`.
- `batch-order-summary.csv`.
- `formal-ledger.csv`.
- `source-manifest.csv`.
- `metadata.json`.
- `README.md`.

正式 report 在所有 formal runs 结束后生成。

## 判定标准

单个 `seed + read_topk` 组合通过 screen:

- 4 个 formal run 全部 completed。
- 四个 `final 1024x256 accuracy >= 0.85`.
- 四个结果 max-min gap `<= 4pp`.
- 无 NaN/OOM/Traceback。
- cache/init/batch preflight 全部一致。

总体判断:

- 如果 R8 或 R16 三个 seed 全部通过, 可称为 fixed-init 下稳定候选。
- 如果 2/3 seed 通过, 只能称为有潜力但 training-seed sensitive。
- 如果 1/3 或 0/3 seed 通过, 不推进 4ep, 转向机制改进。

## 失败策略

- smoke 失败: 停止, 不启动 formal。
- formal 单个 run 失败: 记录 failed, 队列继续后续 target。
- GPU/NVML/torch CUDA 失败: 停止该机器队列。
- cache/init/batch mismatch: 停止 formal, 报告 mismatch。
