# GDN 与 Flash-VQG 公平对照 Phase 5/6 报告

updated: 2026-05-26

## 摘要

Phase 5 已完成 longer-MQAR OOD eval. 本轮只评估本方案新增的 GDN 对照: `mh-h4-k256-v128` seed123, 以及 `Banked-K shared-V` seeds 123/124/125/126. 全部使用已有 official checkpoint, 不写入 MQAR training ledger.

主要结论:

- `Banked-K` seed123/126 的 OOD 上限高于 `mh-h4` seed123.
- `Banked-K` 四 seed 均值低于 `mh-h4` seed123, 因为 seed124/125 在训练内和 OOD 上都明显退化.
- 同等 active state capacity 的 Flash `cb64-r16` seed123 在 longer-MQAR OOD 上明显强于当前所有 kernel-compatible GDN 对照.
- Phase 6 门控结论是 `go_as_separate_goal`: 如果论文需要 true per-head `K=1024,V=64` GDN baseline, 应新开独立 FLA kernel research goal, 不混入当前实验目标.

## artifact

- 逐条结果: `docs/artifacts/gdn-flash-fairness-20260526/phase5-longer-mqar/longer-mqar-phase5-detail.csv`
- 聚合结果: `docs/artifacts/gdn-flash-fairness-20260526/phase5-longer-mqar/longer-mqar-phase5-summary.csv`
- 状态表: `docs/artifacts/gdn-flash-fairness-20260526/phase5-longer-mqar/status.csv`
- 验证摘要: `docs/artifacts/gdn-flash-fairness-20260526/phase5-longer-mqar/verification.json`
- runner: `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase5/longer_mqar_eval_runner.py`

## Phase 5 结果

formal eval 共 25 条, 全部 completed. 每个 slice 的 `dataset_hash` 在所有 checkpoint 间一致.

| config | n | seeds | active cap | 1024x256 | 2048x512 | 4096x1024 | 8190x512 | 8190x2047 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Banked-K shared-V | 4 | 123,124,125,126 | 131,072 | 0.8446 +/- 0.1536 | 0.3358 +/- 0.2402 | 0.0326 +/- 0.0297 | 0.2018 +/- 0.1347 | 0.0011 +/- 0.0007 |
| mh-h4-k256-v128 | 1 | 123 | 131,072 | 0.9569 | 0.4105 | 0.0538 | 0.2548 | 0.0025 |

Banked-K 的强 seed:

| config | seed | mean over 5 slices | 1024x256 | 2048x512 | 4096x1024 | 8190x512 | 8190x2047 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Banked-K shared-V | 123 | 0.3825 | 0.9801 | 0.5325 | 0.0731 | 0.3245 | 0.0023 |
| Banked-K shared-V | 126 | 0.4006 | 0.9920 | 0.6141 | 0.0489 | 0.3469 | 0.0013 |
| mh-h4-k256-v128 | 123 | 0.3357 | 0.9569 | 0.4105 | 0.0538 | 0.2548 | 0.0025 |

因此, 如果只看 seed123 或 seed126, Banked-K 确实比增加 head 的 `mh-h4` 更有上限. 但四 seed 均值不支持“Banked-K 稳定更优”.

## 与 Flash longer-MQAR official core 对比

既有 official core 中, 同等 active state capacity 的 Flash `cb64-r16` seed123 结果如下:

| config | family | n | active cap | 1024x256 | 2048x512 | 4096x1024 | 8190x512 | 8190x2047 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cb64-r16 | Flash | 1 | 131,072 | 0.9696 | 0.8226 | 0.4689 | 0.7173 | 0.1623 |
| cb256-r4 | Flash | 1 | 131,072 | 0.8942 | 0.6667 | 0.3807 | 0.5667 | 0.1717 |
| Banked-K shared-V | GDN | 4 | 131,072 | 0.8446 | 0.3358 | 0.0326 | 0.2018 | 0.0011 |
| mh-h4-k256-v128 | GDN | 1 | 131,072 | 0.9569 | 0.4105 | 0.0538 | 0.2548 | 0.0025 |

这里有两个重要边界:

- `mh-h4` 目前只有 seed123, 不能和 Banked-K 的四 seed 稳定性直接等价比较.
- `cb64-r16` 也只有 seed123, 但它在 2048 及以上 OOD slice 上领先幅度很大, 足以说明 kernel-compatible GDN 还没有消除 fairness concern.

## Phase 6 门控结论

结论: `go_as_separate_goal`.

原因:

- Banked-K 是 kernel-compatible K-sharded approximation, 不是 true single continuous `K=1024,V=64`.
- Banked-K 的强 seed 能超过 mh-h4, 但多 seed 稳定性不足.
- Flash 在 longer-MQAR OOD 上仍明显领先当前 kernel-compatible GDN, 尤其是 2048x512, 4096x1024, 8190x512, 8190x2047.
- 如果论文主张需要公平回答 “GDN 在 true per-head `K=1024,V=64` 下是否仍落后 Flash”, 当前实验还不能替代 FLA fork.

建议的新独立目标:

```text
基于 fla-org/flash-linear-attention 的目标 commit, 新增 Gated Delta Rule chunk training 的 K-blocked state-update 路径, 支持 per-head K=512,V=128 和 K=1024,V=64, 保持 K<=256 原路径行为不变, 使用 naive_recurrent_gated_delta_rule 做 correctness oracle. 该工作作为独立 kernel research goal, 不混入当前 GDN/Flash fairness 实验目标.
```
