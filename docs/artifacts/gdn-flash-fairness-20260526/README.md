# GDN 与 Flash-VQG 公平对照 artifact, 2026-05-26

本目录用于记录 `docs/plans/20260526-gdn-flash-fairness-experiment-plan.md` 的 phase-level 状态和补充表。

## Phase 0

Phase 0 已完成, 但没有启动正式训练, 因此没有 official ledger row。

产物:

- `phase0-gdn-expanded-k-accounting.csv`: `ek4-ev4`, `ek8-ev2`, `ek16-ev1` 的 capacity/accounting 表。
- `phase-status.json`: 当前 phase 状态和下一步。

结论:

- `ek4-ev4` 在 per-head `K=256,V=256` 下通过最小 forward/backward smoke。
- `ek8-ev2` 在 per-head `K=512,V=128` 下被 FLA chunk state-update kernel 的 `K<=256` 限制阻塞。
- `ek16-ev1` 在 per-head `K=1024,V=64` 下因同一限制不启动正式训练。

这些记录只证明 Phase 0 的 blocked 结论, 不构成正式 MQAR 训练结果。
