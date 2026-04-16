# E5-train 基于 lr=1e-4 control 的 teacher rescreening 结果记录

## 1. Run Scope

- 日期: 2026-04-15
- phase: `rescreening on recalibrated continuation base`
- source checkpoint: `dense-t025-s123-d123`
- seed: `s123`
- data seed: `d123`
- 固定: `learning_rate = 1e-4`
- 固定: `max_epochs = 4`
- 固定: `row_weight_mode = uniform`
- paired control: `lambda_teacher = 0.0`
- teacher runs: `lambda_teacher in {0.005, 0.01, 0.02}`
- 说明: 本轮不是沿旧 `P1` 直接续跑, 而是在 continuation calibration 最优 control 底座上重新筛选 teacher signal

本轮默认使用双 GPU 分片执行:

- GPU 0: `control + lambda=0.01`
- GPU 1: `lambda=0.005 + lambda=0.02`

执行入口:

- [run_e5_train_rescreening_lr1e4_s123.sh](/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_rescreening_lr1e4_s123.sh)

## 2. Results Table

`delta_vs_control` 记为 `acc_total / acc_512x128 / acc_1024x256`.

| variant | lambda_teacher | learning_rate | max_epochs | acc_total | acc_512x128 | acc_1024x256 | delta_vs_control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 0.000 | 1e-4 | 4 | 0.980543 | 0.987180 | 0.870367 | `+0.000000000 / +0.000000000 / +0.000000000` |
| teacher | 0.005 | 1e-4 | 4 | 0.980543 | 0.987180 | 0.870367 | `+0.000000000 / +0.000000000 / +0.000000000` |
| teacher | 0.010 | 1e-4 | 4 | 0.980543 | 0.987180 | 0.870371 | `+0.000000488 / +0.000000000 / +0.000003906` |
| teacher | 0.020 | 1e-4 | 4 | 0.980543 | 0.987180 | 0.870367 | `+0.000000000 / +0.000000000 / +0.000000000` |

补充观察:

- `lambda=0.005` 与 `lambda=0.020` 在 final valid 三个主指标上都与 control 完全一致.
- `lambda=0.010` 是本轮唯一一个同时高于 control 的点, 但提升只有 `+4.88e-7 / +0 / +3.91e-6`, 量级极小.
- teacher 指标链路正常:
  - `valid/attn/dense_teacher_loss` 随 `lambda` 单调上升: `0.0 -> 0.00634 -> 0.01268 -> 0.02536`
  - `valid/attn/dense_teacher_top2_overlap` 基本稳定在 `0.39538`
  - `valid/attn/dense_teacher_valid_row_ratio` 固定为 `0.625`
  - `valid/attn/dense_teacher_row_weight_mean` 固定为 `0.625`

## 3. Relative To Source

本轮最佳 teacher 是 `lambda=0.010`, 但它仍然没有追平 source checkpoint:

| variant | acc_total | acc_512x128 | acc_1024x256 |
| --- | ---: | ---: | ---: |
| source checkpoint `dense-t025-s123-d123` | 0.981208 | 0.987250 | 0.874887 |
| rescreening control | 0.980543 | 0.987180 | 0.870367 |
| rescreening best teacher `lambda=0.010` | 0.980543 | 0.987180 | 0.870371 |

也就是说, 即便是本轮最佳 teacher, 相对 source checkpoint 的差值仍然约为:

- `acc_total`: `-0.000665039`
- `acc_512x128`: `-0.000070313`
- `acc_1024x256`: `-0.004515625`

## 4. Gate Decision

本轮仍沿用 paired control 口径:

- 至少 1 个 `lambda_teacher > 0` 同时优于 paired control 的 `acc_total`
- 且 `acc_1024x256` 也优于 paired control

按这个最小 gate 的字面定义:

- `lambda=0.010` 形式上满足 `acc_total > control` 且 `acc_1024x256 > control`

但从实验判断角度, 这还不能算稳定正信号, 原因是:

- 提升量级只有 `1e-6` 到 `1e-5`
- `lambda=0.005` 和 `lambda=0.020` 都没有复现同向提升
- 最佳 teacher 仍明显低于 source checkpoint

因此更合理的结论是:

- 本轮 **不应直接视为 teacher 已经被证实有效**
- 最多只能说: 在新 control 底座上, `lambda=0.010` 出现了一个值得记录, 但远未充分的弱正偏移

## 5. Notes

- 本轮的唯一目的, 是在 `lr=1e-4, max_epochs=4` 这一新底座上重新判断 teacher 是否优于 control.
- 这轮结果不能直接和旧 `P1` 表格混写, 因为两轮 continuation 配方不同.
- 若本轮仍没有相对 control 的正信号, 则更应把结论解释为"teacher signal 本身未形成收益", 而不是"continuation 配方拖累了 teacher".
- 如果后续还要继续推进, 更合理的动作不是直接上 `P2`.
- 一个方向是先在 `s123` 上围绕 `lambda=0.010` 做一个极小范围复查.
- 另一个方向是直接上第二个 seed 做同配方 paired repro, 专门验证这个微弱正偏移是否可复现.
