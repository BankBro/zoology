# 20260702-03 Flash-VQG residual injection warmup screen plan

## 目标

验证 default dropout 训练协议下, 只降低早期 `M_state` residual correction 注入输出的强度, 是否能缓解跨机器 1ep hard-slice gap.

本轮不改变 dropout, 不改变 `M_state` build/write/read, 不改变 `read_topk/write_topk`, 只控制:

```python
O_res_added = injection_factor * lambda_blk * residual_scale * O_res_norm
Out_f32 = O_base + O_res_added
```

## 背景

`20260702-02` 已经说明:

- default dropout 下 `baseline-r2` 仍有明显跨机器 gap.
- 真实 training minibatch 中存在较大的 residual update event.
- hard `update_norm_cap=0.5` 不是稳定解法, 会大面积且轨迹依赖地介入训练.
- step512 附近出现明显 `lambda_mean/inject_ratio/loss` 跨机器分叉.

因此本轮转向更小的干预: 不截断写入, 只让 residual correction 对输出的影响逐步放开.

## 实验设置

共同条件:

| 项 | 值 |
| --- | --- |
| seed | `124` |
| data seed | `123` |
| data/init | canonical MQAR cache + canonical seed124 init |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| residual read/write | `read_topk=2`, `write_topk=4` |
| train length | 1 epoch, `704` optimizer steps |
| machines | `2080ti` + `3090`, both in `Flash-VQG-tun` container |

Variants:

| variant | injection warmup |
| --- | --- |
| `baseline-r2` | no warmup, factor = `1` |
| `inj-warmup-linear512-r2` | optimizer step `0 -> 512`, factor linearly `0 -> 1` |
| `inj-warmup-silent64-linear512-r2` | optimizer step `0-64`, factor `0`; optimizer step `64 -> 512`, factor linearly `0 -> 1` |

Flash-VQG schedule 使用 train-forward counter. 本轮 `gradient_accumulation_steps=4`, 所以:

| optimizer step | train-forward step |
| ---: | ---: |
| `64` | `256` |
| `512` | `2048` |

## 实现要求

- 新增默认关闭配置:
  - `fox_gd_residual_injection_warmup_start_train_steps`
  - `fox_gd_residual_injection_warmup_end_train_steps`
  - `fox_gd_residual_injection_warmup_eval_policy`
- 默认 `end_train_steps=0`, 行为等价旧代码.
- 记录 metric:
  - `attn/gd_residual_injection_warmup_factor`
  - `attn/gd_residual_inject_ratio`
  - `attn/gd_residual_lambda_mean`
- 本轮 eval policy 用 `scheduled`, 让 early-window eval snapshot 反映当前训练进度的 injection factor.

## 启动前硬门槛

- 两边容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 必须通过.
- 两边 zoology 与 Flash-VQG 均同步到同一 commit.
- 训练前必须验证 cache content hash, init state hash, batch order hash 跨机器 match.
- 若 3090 SSH/GPU/preflight 失败, 暂停 paired 实验, 不改用宿主机或临时绕过路径.

## 运行安排

2080ti:

- GPU0: `baseline-r2`, 然后 `inj-warmup-linear512-r2`.
- GPU1: `inj-warmup-silent64-linear512-r2`.

3090:

- GPU0: sequential run all three variants.

进入稳定训练或 hash-probe 后, 显式 `sleep 15m` 轮询.

## 判定标准

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

通过标准:

- final hard slice 不低.
- paired 2080ti vs 3090 gap `<= 4pp`.

解释口径:

- 如果 warmup 提升分数或缩小 gap, 说明 residual injection 是有效放大环节.
- 如果 read support 仍分叉但 final gap 改善, 说明不必完全消除 read support 分叉, 降低 residual 注入强度也可能缓解后续放大.
- 如果 warmup 低分但 gap 小, 不算成功.
- 如果两种 warmup 都无效, 下一步转向 read support 稳定化或 `M_state` write/update 控制.

## 产物

- `docs/artifacts/20260702-03-flash-vqg-injection-warmup-screen/run-summary.csv`
- `docs/artifacts/20260702-03-flash-vqg-injection-warmup-screen/variant-gap-summary.csv`
- `docs/artifacts/20260702-03-flash-vqg-injection-warmup-screen/injection-warmup-summary.csv`
- `docs/artifacts/20260702-03-flash-vqg-injection-warmup-screen/read-trace-cross-machine-summary.csv`
- `docs/artifacts/20260702-03-flash-vqg-injection-warmup-screen/hash-probe-comparison-summary.csv`
- `docs/20260702-03-flash-vqg-injection-warmup-screen-report.md`
