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

## 2. Results Table

| run_id | learning_rate | max_epochs | acc_total | acc_512x128 | acc_1024x256 | delta_vs_source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `source checkpoint` | - | - | 0.981208 | 0.987250 | 0.874887 | `+0.000000 / +0.000000 / +0.000000` |
| `e5cal-ctrl-e1-lr1e4-s123-d123` | 1e-4 | 1 | 0.977729 | 0.985578 | 0.849328 | `-0.003480 / -0.001672 / -0.025559` |
| `e5cal-ctrl-e2-lr1e4-s123-d123` | 1e-4 | 2 | 0.979715 | 0.986867 | 0.864199 | `-0.001493 / -0.000383 / -0.010687` |
| `e5cal-ctrl-e4-lr1e4-s123-d123` | 1e-4 | 4 | 0.980543 | 0.987180 | 0.870371 | `-0.000665 / -0.000070 / -0.004516` |
| `e5cal-ctrl-e1-lr3e4-s123-d123` | 3e-4 | 1 | 0.969818 | 0.979586 | 0.793410 | `-0.011391 / -0.007664 / -0.081477` |
| `e5cal-ctrl-e2-lr3e4-s123-d123` | 3e-4 | 2 | 0.976810 | 0.984594 | 0.844684 | `-0.004398 / -0.002656 / -0.030203` |
| `e5cal-ctrl-e4-lr3e4-s123-d123` | 3e-4 | 4 | 0.978326 | 0.985055 | 0.855461 | `-0.002882 / -0.002195 / -0.019426` |
| `e5cal-ctrl-e1-lr1e3-s123-d123` | 1e-3 | 1 | 0.956203 | 0.959844 | 0.714668 | `-0.025006 / -0.027406 / -0.160219` |
| `e5cal-ctrl-e2-lr1e3-s123-d123` | 1e-3 | 2 | 0.969212 | 0.974977 | 0.795363 | `-0.011996 / -0.012273 / -0.079523` |
| `e5cal-ctrl-e4-lr1e3-s123-d123` | 1e-3 | 4 | 0.978151 | 0.983641 | 0.854629 | `-0.003058 / -0.003609 / -0.020258` |

## 3. Main Findings

- continuation damage 对 `learning_rate` 非常敏感, 明显大于它对 `max_epochs` 的敏感度.
- `1e-4` 是本轮唯一接近 source checkpoint 的 continuation 区间.
- 在 `1e-4` 下, `1 epoch` 已经明显掉点.
- 在 `1e-4` 下, `2 epochs` 时大部分指标回升.
- 在 `1e-4` 下, `4 epochs` 是本轮最佳 control.
- `3e-4` 和 `1e-3` 都不适合作为后续 teacher 实验底座, 尤其 `1024x256` 掉点明显.

本轮最佳 control 配方是:

- `learning_rate = 1e-4`
- `max_epochs = 4`
- 指标为 `0.980543 / 0.987180 / 0.870371`
- 相对 source checkpoint 的差值是 `-0.000665 / -0.000070 / -0.004516`

## 4. Decision Rule

本轮不比较 teacher vs control.  
唯一问题是:

- 是否存在某个 `lambda=0.0` continuation 配方, 能够不明显差于 source checkpoint

优先观察:

- `valid/accuracy`
- `valid/mqar_case/accuracy-512x128`
- `valid/mqar_case/accuracy-1024x256`

若 calibration 仍普遍低于 source checkpoint, 则当前 warm-start continuation 路线应先挂起.  
若出现更稳的 control 配方, 再在该配方上重开 teacher 小网格.

## 5. Conclusion

本轮 calibration 的结论不是"warm-start continuation 完全不可用", 而是:

- 之前的掉点主要是 continuation 配方问题, 尤其是 `learning_rate` 过高
- 将 `learning_rate` 降到 `1e-4` 后, continuation 的表现明显改善
- 但即便是本轮最佳的 `1e-4 @ 4 epochs`, 仍然没有完全追平 source checkpoint

因此更准确的判断是:

- 当前 warm-start continuation 路线 **可以继续探索**
- 但后续 teacher 实验不应再沿用 `1e-3 @ 4 epochs` 这一坏 continuation regime
- 若要重开 teacher 小网格, 唯一合理的下一起点应是 `lr=1e-4, max_epochs=4`
- 若希望更进一步缩小与 source 的差距, 可以在 teacher 之前先再补一个更小的 control 微扫.
- 建议的微扫范围是 `lr in {5e-5, 1e-4, 2e-4}`.
- 建议的 epoch 范围是 `max_epochs in {3, 4, 6}`.
