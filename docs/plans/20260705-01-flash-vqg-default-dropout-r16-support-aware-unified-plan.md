# 20260705-01 Flash-VQG default-dropout r16 support-aware 统一实验计划

## 目的

本轮统一跑 P0-P3, 用同一套 canonical cache/init/batch order 和 default dropout 口径, 先复现 `fixed-r16`, 再筛选 r16 邻域, P2 trace, 以及第一版 read-confidence / softmargin 机制.

本轮是 exploratory 1ep screen 和 diagnostic trace, 不写 official MQAR ledger.

## 固定口径

- `seed=124`.
- `data_seed=123`.
- canonical MQAR cache.
- canonical seed124 init.
- same batch order.
- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- `cb64-r16`.
- `write_topk=4`.
- `max_epochs=1`.
- formal 训练默认 `704` optimizer steps.
- P2 trace variants 只跑 `256` optimizer steps.

## 实验变体

| 阶段 | variant | 设置 | 目的 |
|---|---|---|---|
| P0 | `p0-fixed-r16-repro` | `read_topk=16` | 复现上一轮 `0.912/0.850` |
| P1 | `fixed-r24` | `read_topk=24` | r16 邻域偏宽 |
| P1 | `fixed-r32` | `read_topk=32` | 中宽 support |
| P1 | `sched32to16-linear512` | `32 -> 16`, optimizer step `0..512` | 早期宽读, 后期回 r16 |
| P1 | `sched16to8-linear512` | `16 -> 8`, optimizer step `0..512` | 成本探索 |
| P2 | `trace-r2-readwrite-256` | r2, read trace + inline update trace | r2 失败链路 |
| P2 | `trace-r4-read-256` | r4, read trace | r4 high-risk 对照 |
| P2 | `trace-r16-readwrite-256` | r16, read trace + inline update trace | r16 好轨迹诊断 |
| P2 | `trace-r64-read-256` | r64, read trace, chunked dense read | r64 default-dropout 失败诊断 |
| P3 | `r16-injconf` | r16 + read-confidence-gated injection | 低置信 read 时降低 residual 注入 |
| P3 | `r16-softread` | r16 + top-k 内 softmargin | read 边界软化 |
| P3 | `r16-softread-injconf` | r16 + softmargin + injconf | read + injection 组合 |
| P3 | `r2-injconf` | r2 + injconf | 是否 rescue r2 |
| P3 | `r16-write-mass` | r16 + `topk_mass_scaled` | 现有 write strength 在 r16 上的效果 |
| P3 | `r16-write-mass-injconf` | r16 + write-mass + injconf | write + injection 组合 |

## 新增机制

默认关闭, 只在 P3 打开:

```text
fox_gd_residual_read_confidence_gate_mode = none | margin_sigmoid
fox_gd_residual_read_softmargin_mode = none | topk_mass_temperature
```

`injconf` 使用:

```text
read_conf = floor + (1 - floor) * sigmoid((margin - margin_ref) / temp)
O_res_added = O_res_added * read_conf
```

`softread` 只在选中的 top-k 内重分配 mass, 不改变 read_topk.

## 执行流程

1. 在 2080ti 实现代码和脚本, 本地 smoke.
2. commit/push zoology 和 Flash-VQG.
3. 3090 容器内 pull 到同一 commit.
4. 两机检查 NVML/CUDA, cache hash, init hash, batch order hash.
5. 两机对全部 variant 跑 `max_train_steps=8` smoke, smoke 只跑 `max_validation_batches=16` 用于检查启动, forward, metric 和短验证是否报错.
6. 全部 smoke 通过后, 两机启动 formal queue.
7. formal queue 自动顺序跑完所有 variant, 单个 variant 失败时记录失败并继续; GPU/CUDA/cache/init/batch 硬门槛失败时停止.
8. Codex 监控到两机第一个 formal run 进入稳定 train loop 后可退出会话.

## 判定

核心通过标准:

```text
final 1024x256 hard slice 两机都 >=0.85
paired gap <=4pp
overall accuracy 两机都高
无 NaN/OOM/Traceback
```

P2 trace 不按 final accuracy 判定, 只用于分析 read/write/injection 分叉.

## 收尾

实验结束后生成:

- `docs/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified-report.md`
- `docs/artifacts/20260705-01-flash-vqg-default-dropout-r16-support-aware-unified/`

报告必须明确区分 formal 1ep screen 和 diagnostic trace.
