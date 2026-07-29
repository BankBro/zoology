# A1训练加速候选MQAR筛选计划

## 1. 实验登记

- Experiment ID: `20260730-01-a1-acceleration-mqar-probe`.
- 状态: registered.
- 机器: RTX 2080 Ti GPU1.
- dtype: FP32.
- Zoology base: `flash-vqg@13d312a1245d56082f6102f9f1a461d6b0918a6e`.
- Flash-VQG source: `114eadbd1d2e3c9a43b927e54f6ad9a2692c40e8`.
- 上游实验: Flash-VQG `20260730-01-a1-flash-training-acceleration`.

本实验只做单seed低成本质量筛选, 不替代三seed4epoch正式门禁, 不写入canonical ledger.

## 2. 对照与唯一差异

两组共同使用seed123, data seed123, canonical init/cache, A1 `post_phase1` remat, deterministic selected backward, 1 epoch, train/eval batch `64/16`, GA4.

| Variant | block_len | write_topk | read_topk |
|---|---:|---:|---:|
| `a1-reference` | 32 | 4 | 16 |
| `a1-block256-k2r8` | 256 | 2 | 8 |

## 3. 执行流程

1. Preflight核对源码、分支、GPU、环境、cache、init、参数量和配置差异.
2. 两组各执行3 optimizer steps smoke.
3. 两组各完成1 epoch screen训练并保存best/last checkpoint.
4. 对last checkpoint读取标准`1024x256`指标, 并评估5个固定hash Longer-MQAR slice.
5. 汇总标准任务delta和4个外推slice宏平均delta.

Longer-MQAR复用已通过2080Ti Flash FP32正式batch search的配置: `1024/2048`使用B32, `4096`和两个`8190` slice使用B16.

任一阶段失败时保留现场, 分析并做可行的最小修复后重试. 不覆盖失败目录.

## 4. 晋级门槛

单seed screen同时满足以下条件才建议启动三seed4epoch正式门禁:

```text
standard 1024x256 delta >= -0.02
four extrapolation slices macro delta >= -0.05
all checkpoints, losses and accuracies finite
no Triton fallback
```

三seed正式门槛仍沿用上游计划的`-0.01/-0.02`, 本轮不启动完整自然语言训练.
