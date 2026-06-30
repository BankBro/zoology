# 20260630-03 Flash-VQG s124 fixed-r4 4ep confirm

本目录运行 `seed=124`, `data_seed=123`, `cb64-r16`, no-dropout, `write_topk=4`, train-time `read_topk=4`, `max_epochs=4` 的两机 confirm。

常用命令:

```bash
python s124_fixed_r4_4ep_confirm.py verify-init --machine-name 2080ti --output-json outputs/2080ti-preflight/init-verify.json
python s124_fixed_r4_4ep_confirm.py cache-hash --machine-name 2080ti --output-json outputs/2080ti-preflight/cache-hash.json
python s124_fixed_r4_4ep_confirm.py preflight --machine-name 2080ti --max-epochs 4 --output-json outputs/2080ti-preflight/preflight-fixed-r4.json
./start_s124_fixed_r4_4ep_queue.sh 2080ti-gpu0
./start_s124_fixed_r4_4ep_queue.sh 3090-gpu0
python s124_fixed_r4_4ep_confirm.py collect
```

`outputs/` 是本地中间产物目录, 默认不提交。收尾后只提交 `docs/artifacts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/` 下的轻量 summary 和 `docs/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm-report.md`。
