# E5-train s124 paired repro on lr=1e-4, lambda=0.010 结果记录

## 1. Run Scope

- 日期: 2026-04-16
- phase: `paired repro on second seed`
- source checkpoint: `dense-t025-s124-d123`
- seed: `s124`
- data seed: `d123`
- 固定: `learning_rate = 1e-4`
- 固定: `max_epochs = 4`
- 固定: `row_weight_mode = uniform`
- paired control: `lambda_teacher = 0.0`
- teacher run: `lambda_teacher = 0.010`
- 目的: 验证 `s123` 上那个极弱正偏移, 在第二个 seed 上是否能复现

本轮默认使用双 GPU 分片执行:

- GPU 0: control
- GPU 1: teacher `lambda=0.010`

执行入口:

- [run_e5_train_repro_lr1e4_l0010_s124.sh](/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-train-dense-teacher/run_e5_train_repro_lr1e4_l0010_s124.sh)

## 2. Results Table

`delta_vs_control` 记为 `acc_total / acc_512x128 / acc_1024x256`.

| variant | lambda_teacher | learning_rate | max_epochs | acc_total | acc_512x128 | acc_1024x256 | delta_vs_control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 0.000 | 1e-4 | 4 | 0.978829 | 0.988500 | 0.852164 | `+0.000000000 / +0.000000000 / +0.000000000` |
| teacher | 0.010 | 1e-4 | 4 | 0.978829 | 0.988500 | 0.852164 | `+0.000000000 / +0.000000000 / +0.000000000` |

补充观察:

- 本轮 teacher 和 control 在 final valid 三个主指标上完全一致.
- teacher 指标链路正常:
  - `valid/attn/dense_teacher_loss = 0.011376`
  - `valid/attn/dense_teacher_top2_overlap = 0.463408`
  - `valid/attn/dense_teacher_valid_row_ratio = 0.625`
  - `valid/attn/dense_teacher_row_weight_mean = 0.625`

## 3. Relative To Source

这轮不管是 control 还是 teacher, 都明显低于 source checkpoint:

| variant | acc_total | acc_512x128 | acc_1024x256 |
| --- | ---: | ---: | ---: |
| source checkpoint `dense-t025-s124-d123` | 0.980097 | 0.989484 | 0.860320 |
| repro control | 0.978829 | 0.988500 | 0.852164 |
| repro teacher `lambda=0.010` | 0.978829 | 0.988500 | 0.852164 |

相对 source checkpoint 的差值约为:

- `acc_total`: `-0.001268`
- `acc_512x128`: `-0.000984`
- `acc_1024x256`: `-0.008156`

## 4. Cross-Seed Interpretation

和 `s123` 的 rescreening 对比:

- `s123` 上, `lambda=0.010` 只出现了一个极弱的正偏移
- `s124` 上, 同配方 teacher 和 control 完全持平

因此:

- `s123` 上那个弱正偏移没有在第二个 seed 上复现
- 当前没有证据支持 `lambda=0.010` 在这个新 continuation 底座上带来稳定收益

## 5. Conclusion

本轮 paired repro 的结论很直接:

- teacher 路径本身是通的, 指标也正常
- 但 `lambda=0.010` 没有在 `s124` 上带来任何可见的 end-to-end 增益
- 结合 `s123` 只出现极弱正偏移这一事实, 当前更合理的判断是:
  - `dense_value teacher` 作为训练期辅助监督, 还没有拿到稳定, 可复现的收益证据
  - 这条路线如果继续, 也不应再投入大预算扫参
  - 更适合把当前结果视为"teacher signal 存在, 但现有模块结构没有把它可靠吸收成收益"
