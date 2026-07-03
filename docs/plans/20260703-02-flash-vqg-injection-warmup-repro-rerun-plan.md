# 20260703-02 Flash-VQG injection warmup reproducibility rerun plan

## 背景

`20260702-03-flash-vqg-injection-warmup-screen` 中, default dropout 下两个 residual injection warmup variant 给出了接近可用但未严格过线的信号:

| variant | 2080ti final 1024x256 | 3090 final 1024x256 | gap |
| --- | ---: | ---: | ---: |
| `inj-warmup-linear512-r2` | `0.871` | `0.814` | `5.7pp` |
| `inj-warmup-silent64-linear512-r2` | `0.775` | `0.819` | `4.4pp` |

随后 `20260703-01` 只延长或调整 warmup schedule, 没有复现更稳趋势. 因此本轮不设计新机制, 只精确重跑 `linear512` 和 `silent64-linear512`, 判断上一轮信号是否能复现.

## 启动前耗时分析

旧 `20260702-03` queue 中混入了两个额外耗时项:

1. 训练后自动执行 `hash-probe`.
   - 每个 1ep train 完成后, wrapper 会再跑一次 `hash-probe --max-optimizer-steps 704`.
   - 这会额外重放训练式 forward/backward hash 诊断.
   - 历史 wall time 约为 2080ti 额外 `1h`, 3090 额外 `45min`.
   - 它发生在训练结束后, 不会影响当前 run 的 final accuracy.

2. 训练中启用 `read_trace_train_steps`.
   - 旧配置包含 `0,1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,704` 共 17 个 step.
   - 实现是在指定 optimizer step 插入 eval-mode validation-batch snapshot, 使用 `model.eval()` 和 `torch.no_grad()`, 结束后恢复 `model.train()`.
   - 它会增加训练期间前向和 trace 写入开销.
   - 为保持与旧结果可比, 本轮保留该设置.

本轮去掉训练后的 `hash-probe`, 但保留训练中的 `read_trace_train_steps`, checkpoint 和完整 final validation. 这样训练语义尽量贴近旧实验, 同时避免把复现实验扩展成额外诊断重放.

## 实验问题

本轮只回答两个问题:

1. `inj-warmup-linear512-r2` 在相同 cache/init/batch/order/default-dropout 条件下, paired 1ep 是否能复现接近上一轮的分数和 gap?
2. `inj-warmup-silent64-linear512-r2` 在相同条件下, paired 1ep 是否能复现接近上一轮的分数和 gap?

本轮不回答:

- warmup 是否是最终方案.
- 是否应该跑 4ep.
- 是否需要新机制如 `lambda cap`, `residual_scale`, `update_norm_cap`.

## 实验配置

共同配置复用 `20260702-03` 的 Python 入口:

```text
script: zoology/experiments/flash_vqg/scripts/20260702-03-flash-vqg-injection-warmup-screen/injection_warmup_screen.py
seed: 124
data_seed: 123
canonical MQAR cache: required
canonical init: cb64r16-s124-init.pt
model: cb64-r16
read_topk: 2
write_topk: 4
dropout: embed_dropout=0.1, resid_dropout=0.0, drop_path=0.0
train length: 1 epoch, max_train_steps=704
gradient_accumulation_steps: 4
logger: none
```

Variants:

| variant | residual injection warmup |
| --- | --- |
| `inj-warmup-linear512-r2` | optimizer step `0 -> 512`, factor `0 -> 1` |
| `inj-warmup-silent64-linear512-r2` | optimizer step `0-64`, factor `0`; optimizer step `64 -> 512`, factor `0 -> 1` |

## 执行安排

Machines:

| machine | GPU | queue |
| --- | --- | --- |
| `2080ti` | GPU0 | `linear512` |
| `2080ti` | GPU1 | `silent64-linear512` |
| `3090` | GPU0 | `linear512`, then `silent64-linear512` |

运行前硬门槛:

- 两边容器内 `nvidia-smi` 可用.
- 两边容器内 `torch.cuda.is_available()` 为 true.
- 两边 `zoology` 处于同一 branch/commit.
- 每个 variant 的 cache content hash 一致.
- init model state hash 一致.
- batch order hash 一致.

## 产物

Experiment id:

```text
20260703-02-flash-vqg-injection-warmup-repro-rerun
```

脚本目录:

```text
zoology/experiments/flash_vqg/scripts/20260703-02-flash-vqg-injection-warmup-repro-rerun/
```

artifact:

```text
docs/artifacts/20260703-02-flash-vqg-injection-warmup-repro-rerun/
```

report:

```text
docs/20260703-02-flash-vqg-injection-warmup-repro-rerun-report.md
```

## 判定标准

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

判定:

| 情况 | 解释 |
| --- | --- |
| 两个 variant 都接近旧结果 | 说明上一轮信号有一定复现性, 可考虑组合控制或更小范围确认 |
| 一个复现, 一个不复现 | 保留复现的 variant 作为候选, 另一个降级为偶然轨迹 |
| 两个都明显不复现 | 说明 single-factor injection warmup 不稳定, 不应继续 4ep 或多 seed 扩展 |

本轮不以单机高分作为通过条件. paired gap 和两机绝对分必须一起看.
