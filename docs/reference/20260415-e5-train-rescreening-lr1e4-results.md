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
| control | 0.000 | 1e-4 | 4 | pending | pending | pending | pending |
| teacher | 0.005 | 1e-4 | 4 | pending | pending | pending | pending |
| teacher | 0.010 | 1e-4 | 4 | pending | pending | pending | pending |
| teacher | 0.020 | 1e-4 | 4 | pending | pending | pending | pending |

## 3. Gate Decision

本轮仍沿用 paired control 口径:

- 至少 1 个 `lambda_teacher > 0` 同时优于 paired control 的 `acc_total`
- 且 `acc_1024x256` 也优于 paired control

只有在这两个条件同时成立时, 才说明 teacher signal 在新 continuation 底座上出现了值得继续追的正信号.

## 4. Notes

- 本轮的唯一目的, 是在 `lr=1e-4, max_epochs=4` 这一新底座上重新判断 teacher 是否优于 control.
- 这轮结果不能直接和旧 `P1` 表格混写, 因为两轮 continuation 配方不同.
- 若本轮仍没有相对 control 的正信号, 则更应把结论解释为"teacher signal 本身未形成收益", 而不是"continuation 配方拖累了 teacher".
