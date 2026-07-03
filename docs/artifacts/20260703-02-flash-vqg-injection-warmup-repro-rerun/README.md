# 20260703-02-flash-vqg-injection-warmup-repro-rerun

本目录保存 `20260703-02` injection warmup reproducibility rerun 的轻量审计 artifact.

本轮只纳入 timestamp `20260703T025329Z` 的 no-trace 正式重跑. timestamp `20260703T022900Z` 的 trace-on run 已中止, 只记录在 `aborted-runs.csv`, 不进入结果解释.

关键口径:

- default dropout: `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.
- strict inputs: same canonical MQAR cache, same seed124 init checkpoint, same batch order.
- `read_trace_train_steps` 已改为显式 diagnostic 开关. 本轮未启用, 配置中 `read_trace_enabled=false`, `read_trace_train_steps=[]`.
- `results/*.json` 中 `train_result` 为空, 所以最终指标从每个 train log 的 final validation 行解析.

主要文件:

- `run-summary.csv`: 四个 run 的 final metric, config, log hash, queue 状态.
- `cross-machine-comparison.csv`: 每个 variant 的 2080ti vs 3090 gap.
- `preflight-summary.csv`: cache/init/batch/read-trace 启动前检查.
- `prelaunch-consistency-summary.csv`: 跨机器一致性聚合.
- `trace-mode-summary.csv`: read trace / hash probe / event trace 是否启用.
- `previous-comparison.csv`: 与 `20260702-03` 对照.
- `source-manifest.csv`: 本轮轻量 raw evidence 的来源, mirror path 和 sha256.
- `metadata.json`: artifact 元数据.

结论简述:

- `inj-warmup-linear512-r2`: 2080ti `0.846`, 3090 `0.771`, gap `7.5pp`.
- `inj-warmup-silent64-linear512-r2`: 2080ti `0.816`, 3090 `0.748`, gap `6.8pp`.
- 两个 variant 都没有通过 `<=4pp`; no-trace 重跑没有稳定复现上一轮接近过线的信号.
