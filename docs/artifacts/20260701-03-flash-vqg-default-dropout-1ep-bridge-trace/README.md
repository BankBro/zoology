# 20260701-03-flash-vqg-default-dropout-1ep-bridge-trace

本 artifact 收尾 default-dropout amplifier trace diagnostic. 本轮只定位放大链路, 不测试稳定化方案, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `resid_dropout=0`, `drop_path=0`.

执行说明: 6 条 run 均已完成训练和 hash probe, 对应 `result.json`, log `[done] ... train_status=0 hash_status=0`, 以及 `hash_probe.json` 均存在. `execution-status-summary.csv` 给出有效完成状态, `queue-summary.csv` 保留原始 wrapper 状态用于审计.

核心文件:

- `final-metrics-summary.csv`: 6 条 completed run 的 final hard slice 和 valid 指标简表.
- `final-gap-summary.csv`: 3 个 target 的 1024x256 cross-machine gap 简表.
- `bridge-step-scalar-summary.csv`: step 128/256/384/512/704 的 loss, lambda, inject, M/update/write scalar 简表.
- `run-summary.csv`: per-run final/best metrics.
- `variant-summary.csv`: per-variant cross-machine summary.
- `early-window-summary.csv`: train-step eval read/write scalar metrics.
- `read-trace-summary.csv`: fixed sample read trace aggregate.
- `read-trace-cross-machine-summary.csv`: 2080ti/3090 trace support match summary.
- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.
- `first-mismatch-summary.csv`: first cross-machine mismatch by target.
- `execution-status-summary.csv`: result/log/hash-probe based effective completion status.
- `variant-decision-summary.csv`: target-level completion and read-support summary.
- `preflight-effective-summary.csv`: cache/init/batch-order match summary from hash probes.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `queue-summary.csv`: queue status.
- `source-manifest.csv`: mirrored lightweight raw evidence.
