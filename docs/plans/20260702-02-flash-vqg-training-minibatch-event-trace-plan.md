# 20260702-02 Flash-VQG training-minibatch residual event trace plan

## 目标

本实验用于把 `20260702-01` 的 fixed validation-batch snapshot 证据推进到真实训练现场. 核心问题不是继续调 `update_norm_cap`, 而是定位 default dropout 下 residual GD 放大链路的训练现场事件:

1. `M_state` 大 update 是否真实发生在 training minibatch 内.
2. `update_norm_cap=0.5` 是拦少数尖峰, 还是大面积改变 residual GD 写入.
3. 分叉更偏 write/update, `M_state` norm, read support, 还是 residual injection.
4. 大 update 是否集中在少数 code/head/layer.
5. `cap=0.5` 为什么有帮助但还不稳定, 后续应该设计 soft cap, scheduled cap, state norm control, injection warmup, 还是 read support 稳定化.

本实验只做 diagnostic localization. `update_norm_cap=0.5` 不能被解释为最终方案.

## 实验 ID 与文件

- experiment_id: `20260702-02-flash-vqg-training-minibatch-event-trace`
- plan: `docs/plans/20260702-02-flash-vqg-training-minibatch-event-trace-plan.md`
- script: `zoology/experiments/flash_vqg/scripts/20260702-02-flash-vqg-training-minibatch-event-trace/training_minibatch_event_trace.py`
- artifact: `docs/artifacts/20260702-02-flash-vqg-training-minibatch-event-trace/`
- report: `docs/20260702-02-flash-vqg-training-minibatch-event-trace-report.md`

## 共同配置

```text
seed=124
data_seed=123
canonical MQAR cache
canonical seed124 init
cb64-r16
read_topk=2
write_topk=4
embed_dropout=0.1
resid_dropout=0.0
drop_path=0.0
max_epochs=1
max_train_steps=704
machines=2080ti + 3090
```

跨机器对照必须先验证:

- 本轮实际加载的 13 个 MQAR cache 内容 hash 一致.
- seed124 canonical init 的 state_dict tensor hash 一致.
- 两边 zoology 和 Flash-VQG 分支与 commit 一致.
- 容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 可用.

## Variants

| variant | read_topk | write_topk | update_norm_cap | hypothetical_cap | 目的 |
|---|---:|---:|---:|---:|---|
| `baseline-r2` | 2 | 4 | unset | 0.5 | 观察真实训练 batch 中如果套 cap=0.5 会命中哪些 update |
| `ucap0p5-r2` | 2 | 4 | 0.5 | 0.5 | 观察 actual cap 在真实训练 batch 中拦截了哪些 update |

## Trace 设计

本实验新增 training-minibatch inline trace. 它和旧 `read_trace_train_steps` 不同:

- 旧 trace: 在指定训练进度点额外跑 fixed validation batch 的 eval forward snapshot.
- 新 trace: 在真实 training batch forward 过程中记录 event, 该 forward 继续参与 backward 和 optimizer step.

trace optimizer steps:

```text
0,1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,704
```

每个被选中的 optimizer step 记录该梯度累积窗口里的所有 microbatch. 记录字段包括:

- `trace_phase=train_inline`
- `optimizer_step`, `train_batch_idx`, `micro_step`, `epoch_idx`
- `layer_idx`, `block_idx`, `sample_idx`, `head_idx`, `code_idx`, `token_pos`
- `update_norm_uncapped`, `err_norm`, `target_u_norm`, `addr_d_norm`
- `zeta_before_update_cap`, `zeta_after_update_cap`
- `actual_cap_hit`, `actual_cap_scale`
- `hypothetical_cap_hit`, `hypothetical_cap_scale`
- `raw_topk_mass`, `write_q_top1`, `write_q_entropy`, `write_q_raw_top1`, `write_q_raw_entropy`, `write_top1_mass`

同时保留 validation snapshot read trace, 用于和历史报告对齐 read support 分叉, 但 report 必须明确区分两类 trace.

## 判定标准

| 问题 | 判定依据 |
|---|---|
| 大 update 是否发生在真实 training batch | `baseline-r2` inline `update_norm_max/p95`, `hypothetical_cap_hit_ratio` |
| cap 是否真实生效 | `ucap0p5-r2` inline `actual_cap_hit_ratio`, `actual_cap_scale_mean/min` |
| cap 拦少数尖峰还是大面积改训练 | cap hit ratio, hit count, top event concentration, per-step timeline |
| 问题偏 write/update 还是 state/read/injection | update timeline, `M_state` scalar metrics, read support summary, lambda/inject metrics, loss timeline 的先后关系 |
| 是否 code/head 局部集中 | top event code/head histogram, top code/head share, entropy |
| cap=0.5 是否可推进 | 1ep hard slice 高, paired gap <= 4pp, 且 hit pattern 合理 |

通过线仍然使用用户接受口径:

```text
valid/mqar_case/accuracy-1024x256 gap <= 4pp
```

但如果 gap 过线而 cap hit 很大面积, 仍不能直接作为最终方案, 只能说明 hard cap 是有效 diagnostic.

## Artifact 和 report

artifact 至少生成:

- `README.md`
- `metadata.json`
- `source-manifest.csv`
- `cache-init-preflight-summary.csv`
- `execution-status-summary.csv`
- `run-summary.csv`
- `variant-gap-summary.csv`
- `train-inline-event-step-summary.csv`
- `train-inline-event-cross-machine-summary.csv`
- `train-inline-event-top.csv`
- `cap-hit-timeline.csv`
- `code-head-hotspot-summary.csv`
- `read-trace-cross-machine-summary.csv`
- `first-mismatch-summary.csv`

report 必须包含:

1. 实验目标和非目标.
2. `validation snapshot` 与 `training-minibatch inline trace` 的区别.
3. preflight cache/init/source/env 证据.
4. variant config diff.
5. 1ep final hard slice 和 paired gap.
6. inline event timeline.
7. cap hit 和 cap scale 分析.
8. code/head/layer 热点分析.
9. read support 与 update event 的时间关系.
10. 对五个未定位问题逐项给出 `支持 / 不支持 / 未证明` 判定.
11. 下一步机制设计建议.

## 执行与监控

进入稳定训练后使用显式轮询:

```text
sleep 15m
```

每轮检查:

- 训练进程是否存在.
- 日志是否有 Traceback, CUDA OOM, NaN/Inf.
- `queue-status.tsv`, `result.json`, `update_event_trace.jsonl` 是否更新.
- GPU 占用是否符合预期.

## 风险与边界

- inline trace 会增加少量 CPU I/O, 但只在 17 个 optimizer step 的 microbatch 上记录 top event, 不全量 dump tensor.
- trace runtime 默认关闭, 不影响普通训练.
- 如果任一机器失败或 preflight 不一致, 不给 paired conclusion, report 只记录失败和可用单机事实.
