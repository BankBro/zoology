# 20260630-03 Flash-VQG s124 fixed-r4 4ep confirm plan

status: planned
ledger: not written

## 目标

本轮验证 `seed=124` 下 `fixed-r4` 的 1 epoch 正信号是否能延续到 4 epoch。

上一轮 `20260630-02` 的严格口径结果显示:

| variant | 2080ti final 1024x256 | 3090 final 1024x256 | gap |
|---|---:|---:|---:|
| `fixed-r2-baseline` | 0.775 | 0.840 | 6.5pp |
| `fixed-r4` | 0.900 | 0.897 | 0.3pp |

因此本轮只做最小确认:

```text
seed=124
data_seed=123
cb64-r16
write_topk=4
train read_topk=4
no-dropout
max_epochs=4
2080ti x1 + 3090 x1
```

不继续扩大网格, 不做 dropout, 不做 schedule, 不做 eval topk sweep。

## 实验条件

共同配置:

- `seed=124`.
- `data_seed=123`.
- `num_codebook_vectors=64`.
- `fox_gd_residual_rank=16`.
- `fox_gd_residual_write_topk=4`.
- `fox_remote_read_topk=4`.
- `fox_remote_read_topk_initial=None`.
- `fox_remote_read_topk_final=None`.
- `fox_remote_read_topk_schedule=linear_int`.
- `fox_remote_read_topk_eval_policy=scheduled`.
- `train_batch_size=64`.
- `gradient_accumulation_steps=4`.
- `max_epochs=4`.
- `validations_per_epoch=4`.
- `embed_dropout=0.0`.
- `resid_dropout=0.0`.
- `drop_path=0.0`.

硬门槛:

- 2080ti 和 3090 的 `Flash-VQG-tun` 容器内 `nvidia-smi` / NVML 可用。
- 2080ti 和 3090 的 `Flash-VQG-tun` 容器内 `torch.cuda.is_available()` 为 true。
- 两边 zoology 同分支同 commit。
- 两边 Flash-VQG 同分支同 commit。
- 本轮实际加载的 13 个 MQAR cache 做 content hash, 必须与 canonical hash 一致。
- seed124 canonical init checkpoint 做 state_dict tensor hash, 两边必须一致。

已知 canonical hash:

```text
MQAR cache combined content sha256:
d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8

seed124 init model state sha256:
2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0
```

## 执行步骤

1. 新增实验入口:

```text
zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/
```

2. 本地提交并推送 plan/script, 3090 通过 `git pull --ff-only` 同步, 不用 `scp` 覆盖源码。

3. 在 2080ti 和 3090 分别执行:

```text
verify-init
cache-hash
preflight --max-epochs 4
train --max-epochs 4
```

4. 启动后先确认:

- GPU 有进程。
- 日志开始写入。
- 没有 immediate traceback / OOM。

5. 进入稳定训练后, 显式 `sleep 20m` 轮询:

- `queue-status.tsv`.
- `nvidia-smi`.
- 最新 validation summary.
- failed/completed 状态。

若仍在运行, 继续 `sleep 20m`。

## 判定

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

判定口径:

- `final` gap `<= 4pp`: fixed-r4 通过当前跨机器容忍线。
- `final` 高且 `best-final` drop 小: 稳定性较好。
- `best` 高但 `final` 明显回落: 需要标记 late volatility, 后续看 state/write 累积。
- gap `> 4pp`: fixed-r4 仍不能视为稳定候选。

## 收尾产物

Artifact:

```text
docs/artifacts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/
```

报告:

```text
docs/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm-report.md
```

至少包含:

- cache/init preflight summary.
- 两边 run config diff.
- 4 epoch final/best 1024x256.
- best-final drop.
- cross-machine gap.
- 运行耗时。
- source manifest.
- 明确说明本轮仍然是 no-dropout confirm, 不能回答 dropout 加回后的稳定性。

本轮是 diagnostic / exploratory confirm, 不写 official MQAR ledger。
