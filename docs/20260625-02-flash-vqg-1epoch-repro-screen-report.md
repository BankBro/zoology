# 20260625-02 Flash-VQG 1-Epoch Repro Screen Report

## 1. 状态

本轮实验 `20260625-02-flash-vqg-1epoch-repro-screen` 定位为 diagnostic / exploratory, 不写 official ledger。

当前状态:

- 2080ti 和 3090 的 host-side partial runs 已停止并归档为 interrupted evidence。
- 两台机器后续均从各自持久 `Flash-VQG-tun` 容器启动 container-side queue。
- 主矩阵 `r1/r2` 与补充矩阵 `r3/r4` 均已完成。
- 3090 轻量 raw evidence 已镜像回 2080ti 主工作区, `queue-status.tsv` 做过 sha256 对账。
- artifact 已生成到 `docs/artifacts/20260625-02-flash-vqg-1epoch-repro-screen/`。

启动代码和记录代码需要分开看:

- 训练启动代码: `020c1d4`
- 报告更新提交: `07a00cd`
- 当前 collector 更新后本地工作区包含未提交变更。

## 2. 执行矩阵

正式判读使用以下 completed screen runs:

| machine | gpu | targets |
|---|---:|---|
| 2080ti | 0 | `default-s123-r1` to `default-s123-r4` |
| 2080ti | 0 | `default-s124-r1` to `default-s124-r4` |
| 3090 | 0 | `default-s123-r1` to `default-s123-r4` |
| 3090 | 0 | `default-s124-r1` to `default-s124-r4` |

公共配置:

- layout: `cb64-r16`
- `data_seed=123`
- `train_batch_order=global_shuffle`
- `train_batch_size=64`
- `eval_batch_size=16`
- `gradient_accumulation_steps=4`
- `max_epochs=1`
- `validations_per_epoch=4`
- `read_trace_train_steps=0,64,130,203,352,448,704`
- logger: `LOGGER_BACKEND=none`
- dtype policy: default torch/zoology runtime dtype, no explicit AMP, bf16, or fp16 override

## 3. Final Accuracy

Metric: `valid/mqar_case/accuracy-1024x256`.

| machine | seed | r1 | r2 | r3 | r4 | mean | gap | stable <= 0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2080ti | 123 | 0.935 | 0.937 | 0.945 | 0.950 | 0.941750 | 0.015000 | true |
| 2080ti | 124 | 0.842 | 0.716 | 0.792 | 0.871 | 0.805250 | 0.155000 | false |
| 3090 | 123 | 0.399 | 0.460 | 0.697 | 0.352 | 0.477000 | 0.345000 | false |
| 3090 | 124 | 0.917 | 0.913 | 0.912 | 0.0113 | 0.688325 | 0.905700 | false |

Per-run completed values:

| machine | target | accuracy |
|---|---|---:|
| 2080ti | `default-s123-r1` | 0.935 |
| 2080ti | `default-s123-r2` | 0.937 |
| 2080ti | `default-s123-r3` | 0.945 |
| 2080ti | `default-s123-r4` | 0.950 |
| 2080ti | `default-s124-r1` | 0.842 |
| 2080ti | `default-s124-r2` | 0.716 |
| 2080ti | `default-s124-r3` | 0.792 |
| 2080ti | `default-s124-r4` | 0.871 |
| 3090 | `default-s123-r1` | 0.399 |
| 3090 | `default-s123-r2` | 0.460 |
| 3090 | `default-s123-r3` | 0.697 |
| 3090 | `default-s123-r4` | 0.352 |
| 3090 | `default-s124-r1` | 0.917 |
| 3090 | `default-s124-r2` | 0.913 |
| 3090 | `default-s124-r3` | 0.912 |
| 3090 | `default-s124-r4` | 0.0113 |

All completed screen logs were scanned for `Traceback`, `RuntimeError`, `CUDA out of memory`, `loss=nan`, and `loss=inf`; no hits were found.

## 4. 判读

1-epoch screen 已经有明显区分度, 但本轮不能支持一个干净的 cross-machine seed/path 结论。

同机结果:

- `2080ti s123` 很稳定且高, gap `0.015`.
- `2080ti s124` 不稳定, 但整体中高, gap `0.155`.
- `3090 s123` 不稳定且偏低, gap `0.345`.
- `3090 s124` 前三次高且稳定, 但 `r4` 掉到 `0.0113`, gap `0.9057`.

最重要的审计限制:

- 13 个本轮实际加载的 cache 文件在 2080ti 与 3090 上文件级 sha256 全部不匹配。
- 对 `.pt` 加载后做 tensor 内容级 hash, 13 个文件也全部不匹配。
- 因此两台机器并未在同一训练数据内容上运行, 当前跨机器差异不能直接解释为 GPU, runtime, seed path, 或 machine effect。

所以当前结论应收紧为:

- `1 epoch` 作为 screen 是有效的, 能快速暴露 high/low 分叉和不稳定运行。
- `2080ti s123` 是本轮唯一满足 same-card repeat 稳定标准的组合。
- `2080ti s124`, `3090 s123`, `3090 s124` 都显示同机 repeat 不稳定。
- 跨机器排序差异暂时不应作为主要科学结论, 因为 cache 内容不一致是硬 confound。

## 5. 下一步建议

不建议立刻做 `4 epoch` confirm。先修数据一致性:

1. 选择一个 canonical cache 来源, 建议以 2080ti 主工作区或重新生成的 clean cache 为准。
2. 将 13 个实际使用的 `data_*.pt` 同步到 3090, 或两台机器都清理并用同一代码和配置重新生成 cache。
3. 重新跑 content-level cache hash, 要求 `cache-content-cross-machine-summary.csv` 全部 `content_match=true`。
4. 在 cache 一致后, 先重跑轻量 `1 epoch` screen, 不直接进入 4 epoch。
5. 若同机 repeat 稳定且跨机器排序仍有差异, 再进入 runtime/GPU robustness 或 4 epoch confirm。

## 6. Artifact

核心结果:

- `run-summary.csv`: per-run final metrics and status.
- `repeat-summary.csv`: per-machine, per-seed r1-r4 aggregation.
- `invalid-runs.csv`: failed or interrupted attempts retained for audit.
- `cache-content-cross-machine-summary.csv`: loaded `.pt` content-level cache comparison across 2080ti and 3090.
- `cache-cross-machine-summary.csv`: file-level cache sha256 comparison across 2080ti and 3090.

Trace and audit:

- `early-window-metrics.csv`
- `step-window-summary.csv`
- `read-trace-summary.csv`
- `preflight-summary.csv`
- `machine-summary.csv`
- `source-manifest.csv`
- `metadata.json`

Large raw files, checkpoints, and swanlog are not committed. 3090 queue/log/trace/generated lightweight evidence was mirrored back to the 2080ti main workspace under the same relative paths.
