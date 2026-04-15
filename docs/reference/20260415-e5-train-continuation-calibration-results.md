# E5-train Continuation Calibration 结果记录

## 1. Scope

- 日期: 2026-04-15
- 目标: 排除 "`继续训练 4 epoch 且重启 optimizer` 本身导致掉点" 这一混杂因素
- source checkpoint: `dense-t025-s123-d123`
- seed: `s123`
- data seed: `d123`
- 固定: `lambda_teacher = 0.0`
- 固定: `row_weight_mode = uniform`
- 网格:
  - `learning_rate in {1e-4, 3e-4, 1e-3}`
  - `max_epochs in {1, 2, 4}`

本轮 continuation calibration 默认使用双 GPU 分片执行:

- GPU 0: `lr=1e-4@4`, `lr=3e-4@4`, `lr=1e-3@2`, `lr=1e-4@1`
- GPU 1: `lr=1e-3@4`, `lr=1e-4@2`, `lr=3e-4@2`, `lr=1e-3@1`, `lr=3e-4@1`

执行入口:

- [run_e5_train_continuation_calibration.sh](/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_continuation_calibration.sh)

## 2. Planned Table

| run_id | learning_rate | max_epochs | acc_total | acc_512x128 | acc_1024x256 | delta_vs_source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `e5cal-ctrl-e1-lr1e4-s123-d123` | 1e-4 | 1 | pending | pending | pending | pending |
| `e5cal-ctrl-e2-lr1e4-s123-d123` | 1e-4 | 2 | pending | pending | pending | pending |
| `e5cal-ctrl-e4-lr1e4-s123-d123` | 1e-4 | 4 | pending | pending | pending | pending |
| `e5cal-ctrl-e1-lr3e4-s123-d123` | 3e-4 | 1 | pending | pending | pending | pending |
| `e5cal-ctrl-e2-lr3e4-s123-d123` | 3e-4 | 2 | pending | pending | pending | pending |
| `e5cal-ctrl-e4-lr3e4-s123-d123` | 3e-4 | 4 | pending | pending | pending | pending |
| `e5cal-ctrl-e1-lr1e3-s123-d123` | 1e-3 | 1 | pending | pending | pending | pending |
| `e5cal-ctrl-e2-lr1e3-s123-d123` | 1e-3 | 2 | pending | pending | pending | pending |
| `e5cal-ctrl-e4-lr1e3-s123-d123` | 1e-3 | 4 | pending | pending | pending | pending |

## 3. Decision Rule

本轮不比较 teacher vs control.  
唯一问题是:

- 是否存在某个 `lambda=0.0` continuation 配方, 能够不明显差于 source checkpoint

优先观察:

- `valid/accuracy`
- `valid/mqar_case/accuracy-512x128`
- `valid/mqar_case/accuracy-1024x256`

若 calibration 仍普遍低于 source checkpoint, 则当前 warm-start continuation 路线应先挂起.  
若出现更稳的 control 配方, 再在该配方上重开 teacher 小网格.
