# 20260628-01 Flash-VQG Stability Ablation Plan

status: implemented
experiment_id: `20260628-01-flash-vqg-stability-ablation`

## 目标

本轮回答一个低成本问题:

```text
固定 canonical MQAR cache 和 canonical init 后,
只关闭 embed_dropout,
能不能把 2080ti 和 3090 的 1024x256 准确率差距压到 4pp 以内.
```

本轮是 diagnostic / exploratory 1 epoch screen, 不写 official MQAR ledger.

## 前置硬门槛

每台机器都必须在对应宿主机的 `Flash-VQG-tun` 容器内检查:

- `nvidia-smi` 可用.
- `torch.cuda.is_available()` 为 true.
- zoology 和 Flash-VQG 同步到同一 commit.
- 本轮实际加载的 13 个 MQAR cache content hash 等于 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- canonical init state hash 等于 `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.

任一项失败, 停止启动训练.

## 实验矩阵

共同配置:

- `seed=123`.
- `data_seed=123`.
- `cb64-r16`.
- `read_topk=2`.
- `train_batch_size=64`.
- `gradient_accumulation_steps=4`.
- `max_epochs=1`.
- `validations_per_epoch=4`.
- `init_checkpoint=20260627-02 canonical init`.
- `embed_dropout=0.0`.
- `resid_dropout=0.0`.
- `drop_path=0.0`.

矩阵:

| machine | target |
|---|---|
| 2080ti | `no-embed-dropout-s123-r1` |
| 3090 | `no-embed-dropout-s123-r1` |
| 3090 | `no-embed-dropout-s123-r2` |

## 判读

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

判据:

- 若所有 3090 run 相对 2080ti 的 `1024x256` gap `<= 4pp`, dropout 扰动放大是优先解决方向.
- 若 gap 仍 `> 4pp`, dropout 只算第一扰动点, 下一步查 Flash-VQG mixer 内部.

## 监控策略

启动后前 10-15 分钟高频检查训练是否进入稳定状态.

进入稳定训练后:

- 显式输出 stable-training 标记.
- 使用 `sleep 1200` 做 20 分钟轮询.
- 每次只检查进程是否存活, log 是否出现 `Traceback`, `RuntimeError`, `CUDA out of memory`, `loss=nan`, `loss=inf`.
- 不持续刷完整日志.

## 产物

脚本:

```text
zoology/experiments/flash_vqg/scripts/20260628-01-flash-vqg-stability-ablation/
```

收尾 artifact:

```text
docs/artifacts/20260628-01-flash-vqg-stability-ablation/
```

至少包含:

- `run-summary.csv`
- `cross-machine-comparison.csv`
- `cache-init-preflight-summary.csv`
- `source-manifest.csv`
- `metadata.json`
- `README.md`

报告:

```text
docs/20260628-01-flash-vqg-stability-ablation-report.md
```
