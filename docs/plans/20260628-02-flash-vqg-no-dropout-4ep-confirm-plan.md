# 20260628-02 Flash-VQG no-dropout 4ep confirm plan

status: implemented
experiment_id: `20260628-02-flash-vqg-no-dropout-4ep-confirm`

## 目标

本轮验证上一轮 no-dropout 1 epoch 发现是否能延续到 4 epoch final checkpoint:

```text
固定 canonical cache 和 canonical init,
关闭 embed_dropout/resid_dropout/drop_path,
看 2080ti 和 3090 的 final 1024x256 gap 是否仍在 4pp 内,
同时判断 no-dropout 是否有明显 ceiling tax.
```

本轮是 diagnostic / confirm screen, 不写 official MQAR ledger.

## 第一轮矩阵

共同配置:

- `seed=123`.
- `data_seed=123`.
- `cb64-r16`.
- `read_topk=2`.
- `train_batch_size=64`.
- `gradient_accumulation_steps=4`.
- `max_epochs=4`.
- `validations_per_epoch=4`.
- `init_checkpoint=20260627-02 canonical init`.
- `embed_dropout=0.0`.
- `resid_dropout=0.0`.
- `drop_path=0.0`.

| machine | queue | target |
|---|---|---|
| 2080ti | `2080ti-gpu0` | `no-dropout-4ep-s123-r1` |
| 3090 | `3090-gpu0` | `no-dropout-4ep-s123-r1` |
| 3090 | `3090-gpu0` | `no-dropout-4ep-s123-r2` |

## 前置硬门槛

每台机器都必须在对应宿主机的 `Flash-VQG-tun` 容器内检查:

- `nvidia-smi` 可用.
- `torch.cuda.is_available()` 为 true.
- zoology 和 Flash-VQG 同步到同一 commit.
- 本轮实际加载的 13 个 MQAR cache content hash 等于 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash 等于 `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.
- preflight 确认每 epoch optimizer steps 为 `704`, 4 epoch total optimizer steps 为 `2816`.

任一项失败, 停止启动训练并记录原因.

## 监控策略

启动后先确认训练进入 stable-training. 进入稳定训练后:

- runner 输出 `stable-training target=...; switching to 1200s polling`.
- 使用 `sleep 1200` 做 20 分钟轮询.
- 每次只检查进程, GPU, queue status, result 文件和错误日志.
- 错误关键词: `Traceback`, `RuntimeError`, `CUDA out of memory`, `loss=nan`, `loss=inf`.
- 不持续刷完整日志.

## 第一轮判读

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

成功条件:

- 所有 3090 run 相对 2080ti 的 final `1024x256` gap `<= 4pp`.
- 3090 r1/r2 final `1024x256` repeat gap `<= 4pp`.
- best-final gap 不明显扩大.
- final 分数不能长期停留在上一轮 1 epoch 的 `0.60-0.63` 附近.

失败条件:

- 跨机器或 3090 repeat gap `> 4pp`.
- final 分数明显低于历史 default/hard04 可用水平, 判定 no-dropout 有 ceiling tax.
- 任一 run failed/OOM/中断, 或 evidence mirror/hash 不干净.

## 第一轮结束后的 subagent 分析

第一轮收尾后自动启动只读 subagents:

1. 结果审计: 检查 run-summary, cross-machine comparison, logs, invalid runs, 4pp 判据.
2. 性能/ceiling: 对比历史 default good run, hard04/caprel 等稳定线, 判断 no-dropout 是否有 ceiling tax.
3. 二轮设计: 在不铺大矩阵的前提下给出下一轮建议.

主线程综合后在 report 中写入自动二轮判断.

## 自动二轮规则

默认最多自动启动一组第二轮, 不连续自动开第三轮.

### 分支 A: no-dropout 稳定且 final 高

自动二轮:

```text
dropout-minimal-policy-1ep-screen
```

矩阵为 `only-disable-embed-dropout`, `only-disable-resid-dropout`, `only-disable-drop-path`, 每个 variant 先跑 `2080ti x1 + 3090 x1`.

### 分支 B: no-dropout 稳定但 final 低

自动二轮:

```text
embed-dropout-only-off-1ep-screen
```

矩阵为 `disable-embed-dropout-s123-r1` on `2080ti x1 + 3090 x1 + 3090 r2`.

### 分支 C: no-dropout 仍不稳定

不自动启动长训练. 只生成 `flash-vqg-mixer-divergence-probe` plan, 聚焦 layer 1 mixer 内部 q/k/v, VQ routing, GD residual read/write 和 state update.

### 分支 D: 证据不干净或子代理结论冲突

不自动启动第二轮. 只写 failed/blocked report 和二轮建议.

## 产物

脚本:

```text
zoology/experiments/flash_vqg/scripts/20260628-02-flash-vqg-no-dropout-4ep-confirm/
```

收尾 artifact:

```text
docs/artifacts/20260628-02-flash-vqg-no-dropout-4ep-confirm/
```

报告:

```text
docs/20260628-02-flash-vqg-no-dropout-4ep-confirm-report.md
```
