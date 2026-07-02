# 20260703-01 Flash-VQG injection warmup refinement screen plan

## 目标

验证 default dropout 训练协议下, 更慢, 更温和的 GD residual injection warmup 是否能在 1 epoch 内同时满足:

```text
1024x256 hard slice 不低
2080ti vs 3090 paired gap <= 4pp
```

本轮不改 dropout, 不改 `read_topk/write_topk`, 不改 `M_state` build/write/read, 只改变 residual correction 注入输出的时间曲线:

```python
O_res_added = alpha_inj(t) * lambda_blk * residual_scale * O_res_norm
Out_f32 = O_base + O_res_added
```

## 背景

`20260702-03` 显示 residual injection 是明确放大环节:

| variant | 2080ti | 3090 | gap |
| --- | ---: | ---: | ---: |
| `baseline-r2` | `0.818` | `0.480` | `33.8pp` |
| `inj-warmup-linear512-r2` | `0.871` | `0.814` | `5.7pp` |
| `inj-warmup-silent64-linear512-r2` | `0.775` | `0.819` | `4.4pp` |

但两种 warmup 都没有严格通过 `<=4pp`, 且 `silent64-linear512` 在 2080ti 上绝对分下降. 因此本轮只做 refinement, 不直接进入 4ep confirm.

## 实验设置

共同条件:

| 项 | 值 |
| --- | --- |
| experiment id | `20260703-01-flash-vqg-injection-warmup-refinement-screen` |
| seed | `124` |
| data seed | `123` |
| data/init | canonical MQAR cache + canonical seed124 init |
| model | `cb64-r16` |
| dropout | `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0` |
| residual read/write | `read_topk=2`, `write_topk=4` |
| train length | 1 epoch, `704` optimizer steps |
| gradient accumulation | `4` |
| machines | `2080ti` + `3090`, both in `Flash-VQG-tun` container |

Variants:

| variant | optimizer warmup | train-forward warmup | 目的 |
| --- | --- | --- | --- |
| `inj-warmup-linear704-r2` | `0 -> 704` | `0 -> 2816` | 1ep 结束刚好完全放开, 比 `linear512` 慢 |
| `inj-warmup-linear1024-r2` | `0 -> 1024` | `0 -> 4096` | 1ep 结束 factor 约 `0.6875`, 测持续低注入 |
| `inj-warmup-silent32-linear704-r2` | `32 -> 704` | `128 -> 2816` | 前 32 step 静默, 比 `silent64-linear512` 更温和 |

## 启动前硬门槛

- 两边容器内 `nvidia-smi` 必须通过.
- 两边容器内 `torch.cuda.is_available()` 必须为 true.
- 两边 zoology 和 Flash-VQG 必须同步到记录的 commit.
- 本轮实际加载的 MQAR cache content hash 必须跨机器 match.
- canonical init state hash 必须跨机器 match.
- batch order hash 必须跨机器 match.
- 任一不匹配则暂停训练启动, 不得继续跑严格跨机器对照.

预期 hash:

| field | expected |
| --- | --- |
| cache content | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| init model state | `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0` |
| batch order | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |

## 运行安排

2080ti:

```text
GPU0: inj-warmup-linear704-r2 -> inj-warmup-linear1024-r2
GPU1: inj-warmup-silent32-linear704-r2
```

3090:

```text
GPU0: inj-warmup-linear704-r2 -> inj-warmup-linear1024-r2 -> inj-warmup-silent32-linear704-r2
```

进入稳定训练或 hash-probe 后, 使用显式 20 分钟轮询:

```text
sleep 1200
```

每轮检查 GPU, pid, log tail, result json, queue status.

## 记录指标

主指标:

```text
valid/mqar_case/accuracy-1024x256
```

诊断指标:

- `attn/gd_residual_injection_warmup_factor`
- `attn/gd_residual_inject_ratio`
- `attn/gd_residual_lambda_mean`
- `attn/gd_residual_update_norm_mean/max/p95`
- `attn/gd_residual_m_norm_mean/max`
- read support top1/top-k exact/overlap
- first mismatch/hash probe
- final valid loss/accuracy

Trace steps:

```text
0,1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,704
```

## 判定规则

通过候选:

```text
final 1024x256 on both machines >= 0.82
paired gap <= 4pp
```

解释规则:

| 结果 | 下一步 |
| --- | --- |
| `linear704` 高分且 gap <= 4pp | 进入 `linear704-r2` 4ep paired confirm |
| `linear1024` 高分且 gap <= 4pp | 进入 `linear1024-r2` 4ep paired confirm, 观察 4ep 后期完全释放后的稳定性 |
| `silent32-linear704` 过线但 `linear704` 不过 | 继续试更小 silent window 或 bounded injection, 不直接定 silent 为默认 |
| 三者高分但 gap 仍 `4-6pp` | 下一轮实现 `lambda/inject soft cap` 或暴露 `residual_scale` |
| gap 小但分数低 | 判定为过控 |
| 三者都无明显改善 | 判定只靠 injection warmup 不够, 转向 soft update control |

## 产物

计划文档:

```text
docs/plans/20260703-01-flash-vqg-injection-warmup-refinement-screen-plan.md
```

脚本目录:

```text
zoology/experiments/flash_vqg/scripts/20260703-01-flash-vqg-injection-warmup-refinement-screen/
```

Artifact:

```text
docs/artifacts/20260703-01-flash-vqg-injection-warmup-refinement-screen/
```

Report:

```text
docs/20260703-01-flash-vqg-injection-warmup-refinement-screen-report.md
```

本轮是 diagnostic screen, 不写 official MQAR ledger.
