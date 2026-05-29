# Local window fairness longer-MQAR diagnostics

This artifact is for 20260529-flash-local-window-fairness.

Rows produced by eval-time overrides are diagnostic, not formal training results.
Formal training checkpoints and ledgers must stay separate from this diagnostic artifact.

Required first-round slices:

- 1024x256
- 2048x512
- 4096x512
- 4096x1024

Distance is computed from sample-level MQAR metadata as query_pos - value_pos.
Token-value reverse lookup is not allowed.

## 阶段状态

- 阶段 1 full-only bucket eval: 已完成, 提交于 `0b5257f`.
- 阶段 2 Flash eval-time local/remote ablation: 已完成. 有 official ref 的 full rows 均通过 strict sanity, `4096x512` 为 `no_ref`. `local_only` 在第一轮 longer-MQAR 上接近 0, `local1` 和 `local4` 接近 `full`.
- 第一轮 slices 没有 `<=64` distance bucket 样本, 因此如需直接测 local window 内收益, 还需要 near-distance focused eval.
