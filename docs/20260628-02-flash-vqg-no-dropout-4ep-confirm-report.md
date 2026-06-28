# 20260628-02 Flash-VQG no-dropout 4ep confirm 报告

status: completed_diagnostic
ledger: not written

## 目标

本轮验证 `20260628-01` 的 no-dropout 1 epoch screen 是否能延续到 4 epoch final checkpoint. 口径是固定 canonical MQAR cache, 固定 canonical init checkpoint, 同时关闭 `embed_dropout`, `resid_dropout`, `drop_path`, 再看重点 hard slice `valid/mqar_case/accuracy-1024x256` 的跨机器 gap 是否在用户可接受的 4pp 内.

本轮是 diagnostic / confirm screen, 不写 official MQAR ledger.

## 执行口径

代码版本:

- zoology: `flash-vqg`, commit `eb21661`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `1e7ed33`.

共同配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`, `read_topk=2`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`.
- `max_epochs=4`, `validations_per_epoch=4`.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.
- 2080ti 上生成的 canonical init checkpoint, 两机加载同一份模型初始权重.

前置硬门槛全部通过:

- 两边容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 均可用.
- 本轮实际加载的 13 个 MQAR cache content hash 均为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- init `model_state_dict` tensor hash 均为 `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.
- preflight 确认 4 epoch 共 `2816` 个 optimizer steps.

3090 轻量 evidence 已镜像回 2080ti 主工作区. 镜像后远端/本地逐文件 sha256 为 `10/10 match=true`.

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`.

| machine | target | duration | final valid acc | final 1024x256 | best 1024x256 |
|---|---|---:|---:|---:|---:|
| 2080ti | `no-dropout-4ep-s123-r1` | 365 min | 0.972 | 0.840 | 0.841 |
| 3090 | `no-dropout-4ep-s123-r1` | 245 min | 0.962 | 0.790 | 0.803 |
| 3090 | `no-dropout-4ep-s123-r2` | 245 min | 0.958 | 0.762 | 0.798 |

以 2080ti final 为参考:

| candidate | final 1024x256 gap | within 4pp |
|---|---:|---|
| 3090 r1 | 5.0pp | false |
| 3090 r2 | 7.8pp | false |

3090 r1/r2 之间 final repeat gap 是 2.8pp, 在 4pp 内. 这说明本轮主要失败不是 3090 单机 repeat 爆掉, 而是 3090 两条都落在低于 2080ti 的带上.

## 证据审计

三条 run 都完成, `invalid_count=0`. 日志扫描未发现 `Traceback`, `RuntimeError`, OOM, `loss=nan`, `loss=inf` 或 killed 迹象. 日志中只有 `pynvml` deprecation `FutureWarning`, 不影响训练结果.

本轮 final/best 指标来自训练日志中的 validation summaries. `result_json` 里的 `train_result` 为 `null`, 所以收集脚本按日志正则解析 `valid/mqar_case/accuracy-1024x256`, `valid/accuracy`, `valid/loss`. 子代理独立复核日志后确认与 artifact 中的 `run-summary.csv` 一致. `best_*` 是日志观测到的 best validation metric, 不是重新加载 saved-best checkpoint 后复评.

`n_validation_summaries` 已对 tqdm 相邻重复 summary 去重, 每条 run 为 16; `n_validation_summary_lines` 保留原始日志匹配行数 32.

## 判读

这轮把上一轮 1 epoch 的判断收紧了.

`20260628-01` 显示, 在 cache/init 都锁住后关闭 dropout, 1 epoch 下 2080ti/3090 的 1024x256 gap 是 1.4pp 和 2.0pp. 这说明 dropout/RNG 是真实扰动源, 不是无关日志噪声.

但本轮 4 epoch final 明确没有通过 4pp 稳定线: 3090 两条相对 2080ti 分别低 5.0pp 和 7.8pp. 因此 no-dropout 不能作为当前稳定方案. 更准确的结论是:

```text
dropout/RNG 是重要扰动源之一, 但不是充分解决方案.
关闭 dropout 能消除早期 dropout 分叉, 但 4 epoch 后仍有 Flash-VQG mixer / VQ routing / GD residual read-write-state 路径放大的跨机器差异.
```

这和 `20260627-03` first-divergence probe 一致: baseline 第一处分叉在 `backbone.layers.0.dropout1`; no-dropout 后第一层完全一致, 分叉推迟到 `backbone.layers.1.sequence_mixer.mixer`.

当前 4 epoch 分数也明显低于历史可用稳定区间. 历史 cb64-r16 `hard04` 三 seed 是 `0.945039 / 0.963055 / 0.952605`, `caprel0406late` 三 seed 是 `0.949371 / 0.963004 / 0.960484`. 本轮 no-dropout 是 `0.840 / 0.790 / 0.762`. 这说明在当前 4 epoch 预算和配置下, no-dropout 有明显 performance tax. 但这不等于已经证明无限训练后的永久 ceiling 低.

## 自动二轮判断

按 plan 的自动二轮规则, 本轮应归入分支 C: no-dropout 仍不稳定.

因此不自动启动 `dropout-minimal-policy-1ep-screen`, 也不自动启动 `embed-dropout-only-off-1ep-screen`. 这两个分支都要求 no-dropout 先通过稳定性条件, 当前不满足.

下一步只生成 `flash-vqg-mixer-divergence-probe` plan, 不启动长训练. 目标不是继续比较 final accuracy, 而是定位在相同 cache, 相同 init, no-dropout 条件下, 2080ti 和 3090 的剩余分叉最早进入 Flash-VQG mixer 的哪个子路径.

## 未排除

- 2080ti 只有一条 4 epoch run, 不能完全排除 `0.840` 是单次偏高. 但 3090 两条 repeat gap 只有 2.8pp, 且都低于 2080ti, 所以当前主要信号仍是跨机器带差异.
- 本轮三种 dropout 一起关闭, 不能区分 `embed_dropout`, `resid_dropout`, `drop_path` 各自贡献.
- 本轮没有同口径 default 4 epoch cache/init-locked 对照, 所以不能精确量化 no-dropout 相对 default 缩小或扩大了多少 gap.
- 还不能断言具体根因是 VQ routing, GD residual read/write, state build, 还是更底层 GPU 数值路径. 只能说下一处分叉落在 Flash-VQG mixer 相关路径内.

## 产物

Artifact:

- `docs/artifacts/20260628-02-flash-vqg-no-dropout-4ep-confirm/run-summary.csv`
- `docs/artifacts/20260628-02-flash-vqg-no-dropout-4ep-confirm/cross-machine-comparison.csv`
- `docs/artifacts/20260628-02-flash-vqg-no-dropout-4ep-confirm/cache-init-preflight-summary.csv`
- `docs/artifacts/20260628-02-flash-vqg-no-dropout-4ep-confirm/queue-summary.csv`
- `docs/artifacts/20260628-02-flash-vqg-no-dropout-4ep-confirm/source-manifest.csv`
- `docs/artifacts/20260628-02-flash-vqg-no-dropout-4ep-confirm/metadata.json`

下一步 plan:

- `docs/plans/20260628-03-flash-vqg-mixer-divergence-probe-plan.md`
