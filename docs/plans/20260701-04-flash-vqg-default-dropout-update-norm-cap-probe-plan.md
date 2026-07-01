# 20260701-04 Flash-VQG default-dropout update-norm-cap probe 计划

status: completed
ledger: not written for diagnostic/probe runs

## 目标

本轮只验证一个问题:

```text
在正常训练 dropout, canonical cache/init/batch order 都锁住的条件下,
限制 GD residual 的单步 M_state update 幅度,
是否能减少 default-r2 的跨机器 1ep 轨迹分叉?
```

`update_norm_cap` 只作为 diagnostic control. 如果有效, 后续再设计更平滑的 update warmup 或 soft cap; 不把 hard cap 直接推进为最终训练方案.

## 共同条件

- branch: `flash-vqg`.
- seed: `124`.
- data seed: `123`.
- canonical MQAR cache: 内容 hash 必须 match.
- canonical seed124 init checkpoint: state_dict tensor hash 必须 match.
- model: `cb64-r16`.
- `vq_weight_mode=dense_softmax`.
- `fox_gd_residual_write_topk=4`.
- `fox_remote_read_topk=2`.
- `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- max epochs: `1`.
- max train steps: `704`.
- trace steps: `0,16,64,128,256,384,512,704`.
- valid batch: `441`.
- machines: `2080ti` + `3090`, 均在 `Flash-VQG-tun` 容器内运行.

## Variants

| target | `fox_gd_residual_update_norm_cap` | 作用 |
|---|---:|---|
| `baseline-r2` | `None` | default-r2 失败对照 |
| `ucap0p5-r2` | `0.5` | 温和单步 update 阻尼 |
| `ucap0p25-r2` | `0.25` | 较强单步 update 阻尼 |

本轮先不跑 `read_topk=4`. 若 r2 上有正信号, 下一轮再迁移到 r4.

## 执行

1. 写入 thin wrapper 和 queue 脚本, 不改 Flash-VQG 模型实现.
2. 本地检查:
   - `python -m py_compile` 新增 Python 脚本.
   - `bash -n` 检查 shell 脚本.
   - config/preflight 确认三个 variant 的 `fox_gd_residual_update_norm_cap` 生效.
3. 提交推送, 3090 容器内拉到相同 commit.
4. 启动前硬门槛:
   - 两机容器内 `nvidia-smi` 与 `torch.cuda.is_available()` 通过.
   - 两机 branch/commit 一致.
   - cache content hash match.
   - init state hash match.
   - batch order hash match.
5. 调度:
   - 2080ti: GPU0 顺序跑 `baseline-r2`, `ucap0p25-r2`; GPU1 跑 `ucap0p5-r2`.
   - 3090: GPU0 顺序跑 3 个 target.
   - 训练进入稳定状态后显式 `sleep 900` 轮询.
6. 收尾:
   - 镜像 3090 轻量 evidence 回主工作区.
   - 生成 artifact 和 report.
   - 提交推送, 保持工作区干净.

## 判定

| 结果 | 判读 | 下一步 |
|---|---|---|
| final hard 高且两机 gap `<=4pp` | update 写入过猛是关键放大点之一 | 设计 soft cap 或 warmup |
| gap 明显缩小但 final 低 | hard cap 抓到放大点, 但有性能税 | 改成 warmup/soft cap |
| cap hit 基本为 0 且结果不变 | cap 没介入 | 降低 cap 或转向其他路径 |
| update_norm 被压住但 final 仍分叉 | 单步 update 不是唯一主因 | 转向 write support, lambda/injection 或 read support |
| 两边都低分 | cap 太强或 residual 学习被压死 | 不推进 hard cap |

## 产物

- `docs/artifacts/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe/`
- `docs/20260701-04-flash-vqg-default-dropout-update-norm-cap-probe-report.md`
- 核心 CSV:
  - `run-summary.csv`
  - `variant-gap-summary.csv`
  - `cap-metrics-summary.csv`
  - `early-window-summary.csv`
  - `read-trace-cross-machine-summary.csv`
  - `cache-init-preflight-summary.csv`
  - `queue-summary.csv`
  - `metadata.json`
