# 20260630-04 default-dropout fixed-r4 1ep screen

本目录运行 `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, train-time `read_topk=4`, default dropout, `max_epochs=1` 的两机 screen。

2080ti GPU1 空闲时可追加 `fixed-r2-baseline` 作为 supplemental read_topk baseline。

default dropout 口径:

```text
embed_dropout=0.1
resid_dropout=0.0
drop_path=0.0
```

本轮复用 `20260630-03` 的 seed124 canonical init checkpoint, 不在本目录提交大 checkpoint。

启动示例:

```bash
bash start_default_dropout_fixed_r4_1ep_queue.sh 2080ti-gpu0
bash start_default_dropout_fixed_r4_1ep_queue.sh 3090-gpu0
bash start_default_dropout_fixed_r4_1ep_queue.sh 2080ti-gpu1
```

收尾:

```bash
/home/lyj/miniconda3/envs/flash-vqg/bin/python default_dropout_fixed_r4_1ep.py collect
```
