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
