# 20260628-02 Flash-VQG no-dropout 4ep confirm

本目录服务于 `docs/plans/20260628-02-flash-vqg-no-dropout-4ep-confirm-plan.md`.

目标:

- 固定 canonical MQAR cache 和 canonical init.
- 关闭 `embed_dropout`, `resid_dropout`, `drop_path`.
- 跑 `2080ti x1 + 3090 x2` 的 4 epoch confirm.
- 进入 stable-training 后使用 `POLL_SECONDS=1200` 进行 20 分钟轮询.

主要入口:

```bash
bash run_no_dropout_4ep_queue.sh 2080ti-gpu0
bash run_no_dropout_4ep_queue.sh 3090-gpu0
```

收尾:

```bash
/home/lyj/miniconda3/envs/flash-vqg/bin/python no_dropout_4ep.py collect
```

本轮是 diagnostic / confirm screen, 不写 official MQAR ledger.
