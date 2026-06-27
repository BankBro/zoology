# 20260627-02 canonical init-lock screen

本目录只服务于 `docs/plans/20260627-02-flash-vqg-canonical-init-lock-screen-plan.md`.

本轮定位是 debug hygiene:

- 复用 2080ti canonical cache.
- 从 2080ti 保存 canonical init checkpoint.
- 在 3090 加载同一 init checkpoint.
- 只跑 `3090 x2 + 2080ti x1` 的 `s123` 1 epoch screen.

不把 init-lock 当方法改进, 不写 official ledger.
