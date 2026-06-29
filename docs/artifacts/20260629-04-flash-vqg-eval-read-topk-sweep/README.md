# 20260629-04-flash-vqg-eval-read-topk-sweep

本目录保存 Flash-VQG dense-read 4ep checkpoint 的 evaluation read-topk sweep 轻量结果.

本实验不重新训练, 不保存新 checkpoint. 每条记录只加载已有 checkpoint, 覆盖评估阶段 `fox_remote_read_topk`, 然后跑完整 validation.

## 文件

- `eval-summary.csv`: 每个 checkpoint/topk/eval machine 的评估结果.
- `topk-vs-64.csv`: 同一 checkpoint 同一 eval machine 下, 各 topk 相对 topk=64 的差值.
- `cross-machine-eval-comparison.csv`: 同一 checkpoint 同一 topk 在 2080ti 与 3090 eval 的差值.
- `aggregate-by-topk.csv`: 按 eval machine 和 topk 聚合的均值/范围.
- `aggregate-by-topk-extended.csv`: 按 topk 汇总 overall accuracy, hard accuracy, loss 和 selected mass.
- `topk4-win-margins.csv`: `topk=4` 相对 `topk=64` 和相对次优 topk 的逐 checkpoint margin.
- `cache-hash-summary.csv`: 2080ti 与 3090 的 MQAR cache file/content hash 对照.
- `checkpoint-manifest.csv`: 本轮 checkpoint 输入清单.
- `source-manifest.csv`: 原始 JSONL/status 文件的来源与 sha256.
- `metadata.json`: 运行元信息.

## 汇总

```json
{
  "experiment_id": "20260629-04-flash-vqg-eval-read-topk-sweep",
  "generated_at": "2026-06-29T18:48:31+00:00",
  "record_files": [
    "zoology/experiments/flash_vqg/scripts/20260629-04-flash-vqg-eval-read-topk-sweep/outputs/full/2080ti-eval-2080ti-source/eval-records.jsonl",
    "zoology/experiments/flash_vqg/scripts/20260629-04-flash-vqg-eval-read-topk-sweep/outputs/full/2080ti-eval-3090-mirror/eval-records.jsonl",
    "zoology/experiments/flash_vqg/scripts/20260629-04-flash-vqg-eval-read-topk-sweep/outputs/full/3090-eval-3090-source/eval-records.jsonl",
    "zoology/experiments/flash_vqg/scripts/20260629-04-flash-vqg-eval-read-topk-sweep/outputs/full/3090-eval-2080ti-mirror/eval-records.jsonl"
  ],
  "raw_records": 123,
  "total_records": 112,
  "completed_records": 112,
  "failed_records": 0,
  "expected_records": 112,
  "all_expected_completed": true,
  "eval_machines": [
    "2080ti",
    "3090"
  ],
  "topks": [
    1,
    2,
    4,
    8,
    16,
    32,
    64
  ]
}
```
