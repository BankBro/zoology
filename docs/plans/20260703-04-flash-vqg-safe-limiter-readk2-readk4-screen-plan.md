# 20260703-04 Flash-VQG safe limiter readk2/readk4 screen plan

## 背景

当前比较确定的是: `embed_dropout=0.1` 是 zoology 的正常训练协议, 不是 bug. 问题在于 Flash-VQG `gd_residual_v1` 后续的 residual read/write/state/injection 机制会把正常训练扰动放大, 导致跨机器或同配置重复训练结果不稳定.

已有结果给出两条启发:

- no-dropout 下 `read_topk=4` 很强, 但 default dropout 下 `fixed-r4` 会明显崩.
- `M_state update_norm_cap=0.5` 曾明显缩小 gap, 说明限制 residual GD 单次写入幅度是有价值的方向, 但固定 hard cap 复现不够稳.
- 旧的 injection smooth softcap 设计在 ratio=0.5 时出现 NaN, 所以本轮只做更保守的 safe injection limit, 不再使用旧 injection softcap 作为候选方案.

本轮目标不是马上提出最终方案, 而是做一轮低额外开销 screen:

1. 比较 read_topk=2 和 read_topk=4 在同一 limiter 下的表现.
2. 测试 safe residual injection limit 是否能稳定但不过度抑制.
3. 测试 scheduled update hard cap 是否比固定 cap 更适合 default dropout.

## 代码改动

Flash-VQG 新增默认关闭配置:

```text
fox_gd_residual_injection_softcap_mode = "safe_smooth_p4"
fox_gd_residual_update_norm_cap_final = None
fox_gd_residual_update_norm_cap_release_start_train_steps = 0
fox_gd_residual_update_norm_cap_release_end_train_steps = 0
fox_gd_residual_update_norm_cap_eval_policy = "final"
fox_gd_residual_update_norm_cap_schedule = "linear"
```

默认情况下这些配置不改变原训练语义.

### safe residual injection limit

作用位置:

```text
O_res_added 算出后, Out_f32 = O_base + O_res_added 之前.
```

形式:

```python
ratio = norm(O_res_added.detach()) / sqrt(norm(O_base.detach()) ** 2 + eps)
scale = (1 + (ratio / cap) ** 4) ** (-1 / 4)
O_res_added = O_res_added * scale.detach()
```

本轮只试:

```text
cap = 1.0
cap = 2.0
```

解释: 只限制 residual correction 相对 base output 过强的注入, 避免旧 injection softcap 中 ratio 过大时产生 NaN. `scale.detach()` 表示 limiter 只作为前向幅度控制, 不让模型通过 limiter 本身走复杂梯度.

### scheduled update hard cap

作用位置:

```text
residual GD 写入 M_state 之前.
```

形式:

```text
0.5 -> 0.8 over 512 optimizer steps, linear
0.5 -> 1.0 over 512 optimizer steps, linear
```

因为代码内部 schedule 按 train forward step 计数, 本轮 512 optimizer steps 对应:

```text
512 * gradient_accumulation_steps(4) = 2048 train forward steps
```

解释: 训练早期限制单次 residual update, 后期逐步放开, 避免固定 hard cap 长期压制模型容量.

## 实验矩阵

共同条件:

```text
seed = 124
data_seed = 123
max_epochs = 1
max_train_steps = 704 optimizer steps
gradient_accumulation_steps = 4
cb64-r16
write_topk = 4
vq_weight_mode = dense_softmax
embed_dropout = 0.1
resid_dropout = 0.0
drop_path = 0.0
canonical MQAR cache
canonical seed124 init
```

本轮明确关闭额外耗时定位:

```text
read_trace_enabled = false
read_trace_train_steps = []
read_churn_probe_enabled = false
train_inline_event_trace_enabled = false
shadow dense read = disabled
per-target hash probe = skipped
```

Variants:

| group | variant | read_topk | limiter |
|---|---|---:|---|
| r2 | `r2-baseline` | 2 | none |
| r2 | `r2-safe-inj-ratio1p0` | 2 | safe injection cap=1.0 |
| r2 | `r2-safe-inj-ratio2p0` | 2 | safe injection cap=2.0 |
| r2 | `r2-updatecap-0p5to0p8-linear512` | 2 | update hard cap 0.5->0.8 |
| r2 | `r2-updatecap-0p5to1p0-linear512` | 2 | update hard cap 0.5->1.0 |
| r4 | `r4-baseline` | 4 | none |
| r4 | `r4-safe-inj-ratio1p0` | 4 | safe injection cap=1.0 |
| r4 | `r4-safe-inj-ratio2p0` | 4 | safe injection cap=2.0 |
| r4 | `r4-updatecap-0p5to0p8-linear512` | 4 | update hard cap 0.5->0.8 |
| r4 | `r4-updatecap-0p5to1p0-linear512` | 4 | update hard cap 0.5->1.0 |

## 启动队列

2080ti 两张 GPU 并行:

```text
safe-limiter-2080ti-gpu0-r2:
  r2-baseline
  r2-safe-inj-ratio1p0
  r2-safe-inj-ratio2p0
  r2-updatecap-0p5to0p8-linear512
  r2-updatecap-0p5to1p0-linear512

safe-limiter-2080ti-gpu1-r4:
  r4-baseline
  r4-safe-inj-ratio1p0
  r4-safe-inj-ratio2p0
  r4-updatecap-0p5to0p8-linear512
  r4-updatecap-0p5to1p0-linear512
```

3090 单卡顺序跑完整 10 个:

```text
safe-limiter-3090-gpu0-all:
  all r2/r4 variants
```

队列设计:

- 每个 target 启动前做 cache hash, config preflight, batch-order preflight.
- `train` 入口也内置 cache/init hard guard. 即使绕过队列手动启动单个 variant, 也必须先通过 canonical MQAR cache 内容 hash 和 canonical seed124 init tensor hash, 否则直接退出, 不允许缺 cache 时自动生成新数据.
- 某个 target 失败时记录 `train-failed`, 但继续跑后续 target, 避免整晚 GPU 空转.
- 队列内部每 60 秒检查一次训练日志中的 `Traceback`, `RuntimeError`, `CUDA out of memory`, `loss=nan`, `loss=inf`.
- 本轮用户要求启动后退出当前会话, 因此不做长时间人工轮询.

## 启动前硬门槛

必须确认:

1. 2080ti 和 3090 容器内 `nvidia-smi` 与 `torch.cuda.is_available()` 均可用.
2. zoology 和 Flash-VQG 两仓库提交已 push, 3090 pull 到相同 commit.
3. 两机 canonical MQAR cache 内容 hash 一致.
4. 两机 canonical seed124 init tensor hash 一致.
5. `train` 入口启动前会重复执行 cache/init guard, 防止手动启动时绕过队列预检.
6. 10 个 variant 的 preflight 显示:
   - `read_trace_train_steps=[]`.
   - `train_inline_event_trace_enabled=false`.
   - `embed_dropout=0.1`.
   - `fox_remote_read_topk` 与 variant 名一致.
   - safe injection variant 使用 `safe_smooth_p4`.
   - updatecap variant 使用 scheduled hard cap 字段.

如果任一硬门槛失败, 不启动正式训练.

## 判定标准

主指标:

```text
final valid/mqar_case/accuracy-1024x256
```

通过:

```text
两机 final 1024x256 都 >= 0.82
且 2080ti/3090 gap <= 4pp
且无 NaN/Inf/OOM
```

边界:

```text
gap <= 4pp 但任一机器 < 0.80
```

这说明 limiter 可能稳定但过度抑制, 不推进 4ep.

失败:

```text
gap > 4pp
或任一机器明显低分
或 NaN/Inf/OOM
```

## 收尾报告

完成后生成:

```text
docs/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen-report.md
docs/artifacts/20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/
```

Artifact 至少包含:

```text
run-summary.csv
cross-machine-comparison.csv
limiter-metrics-summary.csv
early-window-summary.csv
cache-init-preflight-summary.csv
source-manifest.csv
metadata.json
README.md
```

报告必须回答:

1. read_topk=2 与 read_topk=4 在 default dropout 下哪个更可控.
2. safe injection limit 是否避免旧 injection softcap 的 NaN.
3. scheduled update cap 是否比 baseline 更稳定.
4. 哪个 limiter 同时保持高分和 gap <= 4pp.
5. 是否值得 repeat 1ep.
6. 是否有资格进入 4ep confirm.

如果没有任何 variant 通过, 不继续细调 cap 小数值, 下一步转向更机制化的 read margin/adaptive read, write support stability 或 M_state state-norm 控制.
