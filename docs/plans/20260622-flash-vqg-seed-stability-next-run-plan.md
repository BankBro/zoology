# Flash-VQG seed stability next 实验执行计划

updated: 2026-06-22
status: completed
branch: `flash-vqg`

## 1. 目的

本轮不重复已跑过的 fixed `read_topk=2/4` 大矩阵. 当前新增机制是 `gd_residual_v1` read-side schedule:

```text
early read_topk=4 -> late read_topk=2
release window: train forward count 200 -> 800
schedule: linear_int
eval policy: scheduled
```

目标是先验证这个机制是否能保留 `cb256-r8` 上 fixed `readk4` 的 weak-seed rescue, 同时避免 fixed `readk4` 在 `cb128-r8` 上暴露出的 rerun/boundary 风险.

## 2. 已完成前置检查

- `config-to-runtime smoke`: 覆盖 fixed readk2, fixed readk4, read schedule, write cap, bounded beta + orthogonal addr.
- smoke 检查范围: static config, generated config, runtime effective metrics.
- 关键 runtime metric: `attn/gd_residual_remote_read_topk_effective`.
- CPU smoke 不作为失败结论, 因为 Flash-VQG 局部路径仍会触发 Triton kernel; CUDA smoke 是准入检查.

## 3. 第一波, 3090

机器: `mclab-3090` 的 `Flash-VQG-tun` 容器.

并行数: 3 个实验, 满足用户给定的 3090 最多并行 3 个实验限制.

配置:

| target | GPU | seed | codebook | rank | read schedule |
|---|---:|---:|---:|---:|---|
| `cb256r8-sched-s123` | 0 | 123 | 256 | 8 | 4 -> 2, 200 -> 800 |
| `cb256r8-sched-s124` | 0 | 124 | 256 | 8 | 4 -> 2, 200 -> 800 |
| `cb256r8-sched-s125` | 0 | 125 | 256 | 8 | 4 -> 2, 200 -> 800 |

观察要求:

- 启动后观察 10 分钟.
- 确认 tmux session 未退出.
- 确认日志无 traceback/OOM.
- 确认 3090 GPU 有训练进程和显存占用.

## 4. 第二波, 2080ti

机器: `mclab-2080ti` 当前容器.

并行数: 2 个实验, 每张 2080 Ti 一路.

配置:

| target | GPU | seed | codebook | rank | read schedule |
|---|---:|---:|---:|---:|---|
| `cb128r8-sched-s124` | 0 | 124 | 128 | 8 | 4 -> 2, 200 -> 800 |
| `cb128r8-sched-s125` | 1 | 125 | 128 | 8 | 4 -> 2, 200 -> 800 |

观察要求:

- 第二波启动后观察 10 分钟.
- 退出当前会话前, 必须确认 3090 和 2080ti 两边都仍处于训练状态.
- 不写最终 report; 等用户确认训练全部完成后再收尾, 分析, 写 report.

## 5. 训练公共配置

```text
d_model=128
block_len=32
local_num_blocks=2
lr=1e-3
data_seed=123
train_batch_order=global_shuffle
train_batch_size=64
eval_batch_size=16
gradient_accumulation_steps=4
validations_per_epoch=2
max_epochs=4
early_stopping=disabled
flash_backend=torch
fox_remote_path_backend=torch
fox_remote_formula=gd_residual_v1
fox_gd_residual_write_topk=4
fox_gd_residual_builder=grouped_chunk_torch_ref
fox_gd_residual_pack_mode=semivec_ref
fox_gd_residual_chunk_size=64
fox_gd_residual_mu_min_count=0.1
vq_score_mode=codebook_dot
vq_weight_mode=dense_softmax
vq_update_mode=grad
vq_softmax_tau=0.25
vq_topk=4
```

## 6. 收尾准则

本会话只做到:

- 代码实现.
- CUDA smoke.
- 两波训练启动.
- 每波 10 分钟健康观察.
- 退出前确认两台机器都还在训练.

后续等用户确认训练完成后再做:

- 拉取 manifest, log, metric.
- 分析 final/best/repeat, effective read_topk, m_norm, lambda/inject ratio.
- 生成 `docs/20260622-flash-vqg-seed-stability-next-report.md`.
- 视结果整理轻量 artifact.

## 7. 收尾结果

已完成.

- report: `docs/20260622-flash-vqg-seed-stability-next-report.md`.
- artifact: `docs/artifacts/20260622-flash-vqg-seed-stability-next/`.
- source manifest: `docs/artifacts/20260622-flash-vqg-seed-stability-next/source-manifest.csv`.
- final table: `docs/artifacts/20260622-flash-vqg-seed-stability-next/final.csv`.
- spread summary: `docs/artifacts/20260622-flash-vqg-seed-stability-next/spread-summary.csv`.

结论: 本轮 `read_topk 4->2, 200->800, linear_int` schedule 是负结果. `cb256-r8` 三 seed final hard 为 `0.935/0.820/0.991`, spread `0.171`; `cb128-r8` 两 seed 为 `0.681/0.976`, spread `0.295`. 五条 run 均 completed, 无日志错误, best-final gap 为 `0`, `m_norm_max` 未超过红线 `8`, 但 seed spread 远超准入门槛, 因此不进入 official longer-MQAR.
