# 20260629-04 Flash-VQG eval read-topk sweep

本目录包含 checkpoint-only eval sweep 脚本.

先生成 checkpoint manifest:

```bash
python zoology/experiments/flash_vqg/scripts/20260629-04-flash-vqg-eval-read-topk-sweep/make_checkpoint_manifest.py \
  --output <manifest.csv> \
  --source-machine 2080ti \
  <checkpoint-run-dir>...
```

核心入口:

```bash
python zoology/experiments/flash_vqg/scripts/20260629-04-flash-vqg-eval-read-topk-sweep/eval_read_topk_sweep.py \
  --checkpoint-manifest <manifest.csv> \
  --output-dir <output-dir> \
  --eval-machine 2080ti \
  --topks 1,2,4,8,16,32,64 \
  --checkpoint-kinds best,last \
  --resume \
  --continue-on-error
```

汇总入口:

```bash
python zoology/experiments/flash_vqg/scripts/20260629-04-flash-vqg-eval-read-topk-sweep/collect_eval_read_topk_sweep.py \
  --artifact-dir docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep \
  --checkpoint-manifest <manifest.csv> \
  <records...>
```
