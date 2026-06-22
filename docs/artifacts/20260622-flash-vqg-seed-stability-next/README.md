# 20260622 Flash-VQG seed stability next artifact

本目录保存 `gd_residual_v1` read-side schedule 首轮验证的轻量 artifact.

## 文件

- `final.csv`: 五条 completed run 的 best hard, final hard, best-final gap, final scalar telemetry, run 状态和配置摘要.
- `validation-history.csv`: 每条 run 的 validation hard curve 和关键 gd residual telemetry. 日志中同一 validation 会被进度条重复打印, 此表已做相邻去重.
- `spread-summary.csv`: 按 `config_family` 汇总 cross-seed spread, best-final gap 和 `m_norm` 红线检查.
- `source-manifest.csv`: launch id, run id, SwanLab URL, generated manifest, log, checkpoint 路径来源索引.
- `smoke-summary.csv`: config-to-runtime smoke 记录. 其中 `config-runtime-smoke-20260621T194116Z` 是本轮正式准入 smoke, CUDA 5/5 passed.
- `metadata.json`: artifact metadata, code heads, 训练上下文, 主结论和 caveats.

大型 raw logs, checkpoints 和 swanlog 不在本目录保存, 原位保留在 ignored 路径.

## 机制

本轮验证新增的 read-side schedule:

```text
fox_remote_read_topk_initial = 4
fox_remote_read_topk_final = 2
release_start_train_steps = 200
release_end_train_steps = 800
schedule = linear_int
eval_policy = scheduled
```

共同训练配置为 `data_seed=123`, `max_epochs=4`, `validations_per_epoch=2`, `train_batch_size=64`, `eval_batch_size=16`, `gradient_accumulation_steps=4`, `write_topk=4`, 无 write cap, beta hard cap 默认路径.

## 主结果

主指标是 `valid/mqar_case/accuracy-1024x256`.

| config | seed values | best hard values | final hard values | spread | verdict |
|---|---|---|---|---:|---|
| `cb256-r8` | `123,124,125` | `0.935,0.820,0.991` | `0.935,0.820,0.991` | `0.171` | fail |
| `cb128-r8` | `124,125` | `0.681,0.976` | `0.681,0.976` | `0.295` | fail |

本轮准入门槛是 `spread<=0.03`, `best-final gap<=0.01`, `m_norm_max<8`. 五条 run 都 completed 且没有日志错误, best-final gap 都是 `0`, `m_norm_max` 也都低于 `8`, 但 cross-seed spread 远超门槛.

## 结论

这版 `read_topk 4->2` schedule 不能升级为 official longer-MQAR 候选.

- `cb256-r8` 没有保住既有 fixed readk4 正证据. 先前 fixed readk4 completed spread 约 `0.010`, 本轮 schedule spread 为 `0.171`, 其中 s124 只有 `0.820`.
- `cb128-r8` 仍是 read-side 边界风险配置. 本轮 s124 只有 `0.681`, s125 是 `0.976`, spread `0.295`.
- failure 不是 late drift: 每条 run 的 best-final gap 都是 `0`.
- failure 也不像简单 state norm explosion: `max_m_norm_max_over_valid` 最大为 `5.75`, 低于红线 `8`.
- stdout validation telemetry 只看到 effective readk=`2`. 这是因为第一次 validation 已经晚于 release window, 因此 artifact 能证明最终 eval/top2 状态, 但不能单靠 stdout 证明早期训练 top4 窗口的逐步行为.

## 后续建议

停止把这版 fixed window schedule 推向 official. 下一步应转向:

- 增加 train-step early telemetry, 至少记录 release 前后的 effective readk, read margin, candidate churn, topk mass 和 inject ratio.
- 做 margin-aware read gate, 而不是固定时间窗 `4->2`.
- 将 read gate 与 mild write guard 或 beta/lambda confidence gate 组合, 因为本轮 m_norm 未爆但 basin 仍分叉.
- `cb128-r8` 继续作为 failure case, 不用本轮 schedule 支持稳定结论.
