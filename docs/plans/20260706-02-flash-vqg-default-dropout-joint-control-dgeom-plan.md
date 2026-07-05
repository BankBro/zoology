# 20260706-02 Flash-VQG default dropout joint control and D-geometry plan

## 目标

本轮继续推进 Flash-VQG / `gd_residual_v1` 在 default dropout 下的跨机器稳定性实验. 目标分成两部分:

1. Formal training: 复跑上一轮唯一过线配置 `r16-update-softcap0p5-injwarm512`, 并检查同一个 `update softcap + residual injection warmup` 联合控制是否能扩展到不同 `read_topk`.
2. D-direction geometry diagnostic: 只做诊断, 检查 residual write direction `D_pack = normalize((K - codebook) @ addr_proj)` 是否存在高相关, 非正交, 低 effective rank 或热点 code/head.

D-geometry 诊断不改变模型语义, 不参与 formal pass/fail 判定. 本轮不做 4ep, 不做多 seed, 不继续扩大 read_topk 网格.

## 固定条件

- machines: `2080ti` + `3090`, 均在 `Flash-VQG-tun` 容器内.
- seed: `124`.
- data_seed: `123`.
- cache: canonical MQAR cache, 启动前必须内容 hash match.
- init: canonical seed124 init checkpoint, 启动前必须 tensor hash match.
- batch order: 启动前必须 hash match.
- model: `cb64-r16`.
- `fox_gd_residual_write_topk=4`.
- default dropout:
  - `embed_dropout=0.1`.
  - `resid_dropout=0.0`.
  - `drop_path=0.0`.
- training: `max_epochs=1`, `grad_accumulation_steps=4`, `max_train_steps=704`.
- formal runs 默认关闭 heavy `read_trace`, `hash_probe`, `train_inline_event_trace`, `D-geometry trace`.

## Formal training variants

| target | read_topk | update softcap | injection warmup | 目的 |
|---|---:|---|---|---|
| `r16-update-softcap0p5-injwarm512-rerun` | 16 | `0.5`, `smooth_p4` | optimizer step `0 -> 512` | 复跑上一轮唯一过线配置 |
| `r8-update-softcap0p5-injwarm512` | 8 | `0.5`, `smooth_p4` | optimizer step `0 -> 512` | 检查联合控制能否扩展到中等 read support |
| `r4-update-softcap0p5-injwarm512` | 4 | `0.5`, `smooth_p4` | optimizer step `0 -> 512` | 检查能否救回 default dropout 下历史不稳的 read_topk=4 |
| `r2-update-softcap0p5-injwarm512` | 2 | `0.5`, `smooth_p4` | optimizer step `0 -> 512` | 检查窄 read support 下联合控制是否有效 |
| `r16-injwarm512-only` | 16 | off | optimizer step `0 -> 512` | ablation, 判断收益是否主要来自 injection warmup |

`smooth_p4` softcap 公式:

```text
scale = (1 + (update_norm / cap)^4)^(-1/4)
delta_M <- scale * delta_M
```

这里 `cap=0.5`. 它是平滑限幅, 不是 hard `min(1, cap / norm)`.

Injection warmup 作用在 residual branch 对最终输出的注入强度上. 因为 `grad_accumulation_steps=4`, optimizer step `512` 对应 train-forward step `2048`.

## D-geometry diagnostic variants

本轮只对 3 个代表性 target 额外跑 diagnostic:

| target | 对应机制 | 采样 optimizer steps |
|---|---|---|
| `dgeom-r16-update-softcap0p5-injwarm512` | r16 joint control | `0,64,256,512,703` |
| `dgeom-r16-injwarm512-only` | r16 warmup-only ablation | `0,64,256,512,703` |
| `dgeom-r4-update-softcap0p5-injwarm512` | r4 joint control | `0,64,256,512,703` |

说明: 训练总共 704 个 optimizer steps, 因此 `703` 是本 epoch 内最后一个可捕获的训练 step 采样点, 用来近似 704-step 末端窗口.

D-geometry 记录 detached scalar summary, 不保存大 tensor, 不参与 loss/grad/optimizer. 每个 layer/head/code 记录:

- event count.
- pairwise cosine mean/p50/p90/p95/max.
- signed cosine mean/std.
- effective rank.
- condition number.
- update_norm mean/p95/max.
- write_strength mean/p95.
- raw_topk_mass mean.
- projected address norm summary.

## Preflight 和 smoke

正式启动前必须完成:

1. 两仓库代码同步到同一 commit.
2. 两机容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 通过.
3. cache content hash match.
4. init checkpoint tensor hash match.
5. batch order hash match.
6. 8 个 target 在 2080ti 和 3090 上 smoke 通过.
7. smoke 不允许出现 NaN, OOM, Traceback.

只有所有 smoke 通过后, 才启动 formal queue. 队列按 target 顺序自动接续, `CONTINUE_ON_FAIL=1` 保证单个失败不会阻塞后续 target, 但任何失败都需要在 report 中如实记录.

## 判定标准

Formal screen pass:

- 两机 final `1024x256` accuracy 都 `>= 0.85`.
- paired gap `<= 4pp`.
- overall valid accuracy 两机都高.
- 无 NaN, OOM, Traceback.
- cache/init/batch order 一致.

低分但 gap 小不算成功. 单机高分不算成功. Diagnostic target 不参与 pass/fail.

## 必须生成的 artifact

- `run-summary.csv`.
- `cross-machine-comparison.csv`.
- `variant-summary.csv` 或 `variant-decision-summary.csv`.
- `mechanism-metrics-summary.csv`.
- `cache-init-preflight-summary.csv`.
- `batch-order-summary.csv`.
- `source-manifest.csv`.
- `metadata.json`.
- `README.md`.
- `d-geometry-summary.csv`.
- `d-geometry-by-code-head.csv`.
- `d-geometry-cross-machine.csv`.
- `d-geometry-hotspot-summary.csv`.
- `d-geometry-readme.md`.

## Report 必须回答的问题

Formal:

1. `r16-update-softcap0p5-injwarm512` same-seed paired rerun 是否复现.
2. 联合控制在 `r8/r4/r2` 是否仍有效.
3. 如果 r16 复现但 r8/r4/r2 失败, 是否说明联合控制仍依赖适度宽 read support.
4. `r16-injwarm512-only` 是否过线.
5. 如果 warmup-only 过线, 是否说明上一轮收益主要来自 injection warmup.
6. 如果 warmup-only 不过线而 joint 过线, 是否更支持 update softcap + warmup 联合必要.

D-geometry:

1. `D_pack` direction 是否明显非正交或低秩.
2. D-geometry hotspot 是否和 bad run, high update, high M_norm 或低 hard accuracy 相关.
3. 是否值得下一轮设计 direction-aware write damping.
4. 如果 D-geometry 不相关, 是否应转向 zeta, read/write support, lambda 或 injection 的其他控制.

## 暂不做

- 不跑 4ep.
- 不跑多 seed.
- 不扩额外 read_topk 网格.
- 不跑 fixed-r64.
- 不改 dropout.
- 不修改 MQAR cache, init 或 batch order.
- 不把 D-geometry 诊断变成训练机制.
