# E5-train P1 Screening 结果记录

## 1. Run Scope

- 日期: 2026-04-15
- phase: `P1 screening`
- source checkpoint: `dense-t025-s123-d123`
- seed: `s123`
- data seed: `d123`
- `row_weight_mode = uniform`
- `max_epochs = 4`
- paired control: `lambda_teacher = 0.0`
- teacher runs: `lambda_teacher in {0.02, 0.05, 0.10}`

本轮 `P1` 使用双 GPU 分片执行:

- GPU 0: `control + lambda=0.05`
- GPU 1: `lambda=0.02 + lambda=0.10`

相关结果目录:

- `flash-vqg-20260414-e5train-s123-screening-gpu0-2026-04-15-02-38-47`
- `flash-vqg-20260414-e5train-s123-screening-gpu0-2026-04-15-03-07-22`
- `flash-vqg-20260414-e5train-s123-screening-gpu1-2026-04-15-02-38-47`
- `flash-vqg-20260414-e5train-s123-screening-gpu1-2026-04-15-03-05-27`

## 2. P1 Screening Table

`delta_vs_control` 记为 `acc_total / acc_512x128 / acc_1024x256`.

| variant | lambda_teacher | max_epochs | acc_total | acc_512x128 | acc_1024x256 | delta_vs_control |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | 0.00 | 4 | 0.978078 | 0.984039 | 0.854633 | `+0.000000 / +0.000000 / +0.000000` |
| teacher | 0.02 | 4 | 0.978151 | 0.983641 | 0.854629 | `+0.000073 / -0.000398 / -0.000004` |
| teacher | 0.05 | 4 | 0.978078 | 0.984039 | 0.854633 | `+0.000000 / +0.000000 / +0.000000` |
| teacher | 0.10 | 4 | 0.978151 | 0.983641 | 0.854629 | `+0.000073 / -0.000398 / -0.000004` |

补充观察:

- `lambda=0.02` 与 `lambda=0.10` 的 end-to-end 指标几乎一致.
- `lambda=0.05` 与 control 在本轮 final valid 指标上完全持平.
- teacher 指标链路正常:
  - `valid/attn/dense_teacher_loss` 随 `lambda` 增大而单调上升.
  - `valid/attn/dense_teacher_top2_overlap` 约为 `0.415`.
  - `valid/attn/dense_teacher_valid_row_ratio` 固定为 `0.625`.
  - `valid/attn/dense_teacher_row_weight_mean` 固定为 `0.625`, 符合当前 valid 聚合口径.

## 3. Gate Decision

`P1 -> P2` 的进入条件是:

- 至少 1 个 `lambda_teacher > 0` 同时优于 paired control 的 `acc_total`
- 且 `acc_1024x256` 也优于 paired control

本轮结论:

- `lambda=0.02`: `acc_total` 略高于 control, 但 `acc_1024x256` 略低于 control
- `lambda=0.05`: 与 control 持平
- `lambda=0.10`: `acc_total` 略高于 control, 但 `acc_1024x256` 略低于 control

因此 `P1` gate **不通过**.  
按当前执行方案, 本轮应 **止损于 P1, 不进入 P2 repro**, 也不进入后续 `P3 32-epoch confirm`.

## 4. Notes

- 本轮没有出现 NaN, OOM, shape mismatch, 或 teacher 指标缺失.
- 从实现链路角度看, `dense_value teacher` 已经稳定接入训练.
- 从 `4 epoch screening` 的 end-to-end 结果看, 目前还没有看到相对 paired control 的正信号.
- 若后续仍要继续, 应视为新一轮假设修订, 而不是沿当前 `P1 -> P2 -> P3` gate 自动前进.
