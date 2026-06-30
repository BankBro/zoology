# 20260630-04 Flash-VQG default-dropout fixed-r4 1ep screen 计划

## 目标

本轮先做 1 epoch screen, 不直接跑 4 epoch。核心问题是:

```text
在加回 default dropout 后, fixed-r4 是否仍保持 no-dropout 下观察到的跨机器稳定性?
```

如果 1 epoch 通过, 再进入 default-dropout fixed-r4 4 epoch confirm。

## 实验口径

实验 ID:

```text
20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen
```

共同配置:

- `seed=124`, `data_seed=123`.
- canonical MQAR cache.
- seed124 canonical init checkpoint.
- `cb64-r16`.
- `vq_weight_mode=dense_softmax`.
- `fox_gd_residual_write_topk=4`.
- train-time `fox_remote_read_topk=4`.
- `max_epochs=1`, `validations_per_epoch=4`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- 2080ti x1 + 3090 x1.

注意:

```text
G/L coarse memory 使用 dense softmax 权重.
M_state residual GD 写入仍是 top-k write, write_topk=4.
本轮变量是 default dropout 加回.
```

## 执行步骤

1. 新增实验脚本和队列脚本, 复用 `20260630-03` 的 fixed-r4/canonical cache/init/collect 流程。
2. 提交并推送代码, 在 3090 容器内 pull 到相同 commit。
3. 两边启动前检查容器内 `nvidia-smi` 和 `torch.cuda.is_available()`.
4. 两边验证 MQAR cache content hash 与 seed124 init state hash, 必须 match。
5. 两边 run preflight, 必须确认:
   - `max_epochs=1`.
   - optimizer steps per epoch `704`.
   - total optimizer steps `704`.
   - `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
   - `read_topk=4`, `write_topk=4`.
6. 同时启动 2080ti 和 3090 训练。
7. 训练进入稳定阶段后显式 `sleep 20m` 轮询日志, queue 状态和 GPU 状态。若未结束, 继续按 20 分钟粒度轮询。
8. 完成后 collect artifact, 写 report, 提交并推送轻量产物。

## 判定

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

通过标准:

- 两边训练均正常完成, 无 NaN/OOM/Traceback。
- 2080ti 和 3090 都能学起来。
- final 1024x256 gap `<= 4pp`.

如果通过, 下一步建议跑 default-dropout fixed-r4 4ep confirm。如果失败, 不继续 4ep, 转向 dropout 入口或 read candidate 稳定化拆解。

## 产物

```text
zoology/experiments/flash_vqg/scripts/20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen/
docs/artifacts/20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen/
docs/20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen-report.md
```

本轮是 diagnostic / exploratory screen, 不写 official MQAR ledger。
