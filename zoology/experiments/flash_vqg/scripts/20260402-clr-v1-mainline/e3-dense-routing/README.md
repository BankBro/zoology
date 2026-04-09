# `e3-dense-routing`

实验 3 目录, 当前只落地 `dense write + top2 read`.

本轮固定口径:

- baseline: `l2 + one-hot + ema`
- dense 组: `codebook_dot + dense_softmax + grad`
- read-side 固定 `fox_remote_read_topk=2`
- selector 固定 `den_aware`
- merge 固定 `shared_den`
- logger 固定 `swanlab`

训练矩阵:

- `baseline`
- `dense-t050`
- `dense-t100`
- `dense-t200`

smoke 只跑:

- `baseline`
- `dense-t100`

执行顺序:

1. 先跑 `run_e3_smoke.sh`, 选出 `TRAIN_BATCH_SIZE / EVAL_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS`
2. 确认 `e3_smoke.env` 已生成
3. 再跑 `run_e3_train.sh` 做正式训练

top-k 预留:

- builder 已透传 `vq_topk`
- 当前不进入正式矩阵
- 后续只需增加 `topk_softmax` 变体即可复用同一套脚本
