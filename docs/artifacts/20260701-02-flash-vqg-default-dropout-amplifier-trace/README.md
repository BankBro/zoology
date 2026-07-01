# 20260701-02-flash-vqg-default-dropout-amplifier-trace

本 artifact 收尾 default-dropout amplifier trace diagnostic. 本轮只定位放大链路, 不测试稳定化方案, 不写 official MQAR ledger.

共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, canonical MQAR cache, seed124 canonical init, `resid_dropout=0`, `drop_path=0`.

执行说明: 部分后台 wrapper 在目标完成后未把 `queue-status.tsv` 写到 `completed`, 但对应 `result.json`, log `[done] ... train_status=0 hash_status=0`, 以及 `hash_probe.json` 均存在. 因此本实验判断有效运行时以 `execution-status-summary.csv` 的 `effective_status` 为准, `queue-summary.csv` 保留原始 wrapper 状态用于审计.

核心文件:

- `run-summary.csv`: base collector 输出, 本轮不是主判定表; 由于部分 wrapper status 未写 `completed`, 该文件不能代表全部有效运行.
- `variant-summary.csv`: base collector 输出, 本轮不是主判定表.
- `early-window-summary.csv`: train-step eval read/write scalar metrics.
- `read-trace-summary.csv`: fixed sample read trace aggregate.
- `read-trace-cross-machine-summary.csv`: 2080ti/3090 trace support match summary.
- `hash-probe-comparison-summary.csv`: train-mode forward/backward hash comparison.
- `first-mismatch-summary.csv`: first cross-machine mismatch by target.
- `execution-status-summary.csv`: result/log/hash-probe based effective completion status.
- `variant-decision-summary.csv`: target-level completion and read-support summary.
- `preflight-effective-summary.csv`: cache/init/batch-order match summary from hash probes.
- `cache-init-preflight-summary.csv`: base collector 输出, 本轮前置一致性以 `preflight-effective-summary.csv` 为准.
- `queue-summary.csv`: queue status.
- `source-manifest.csv`: mirrored lightweight raw evidence.
