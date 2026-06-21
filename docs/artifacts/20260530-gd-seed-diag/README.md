# 20260530 GD Seed Diag Artifact

本目录保存 Flash-VQG `gd_residual_v1` seed stability 诊断的正式关键数据和来源索引.

## 文件

- `final.csv`: 早期 `cb256-r4` 诊断闭环的紧凑表.
- `gd-seed-diag-key-metrics.csv`: 早期 baseline, runtime probe, formal readk4 和 counterexample 的 run-level 指标.
- `gd-seed-diag-spread-summary.csv`: 早期 `cb256-r4` spread 对照.
- `gd-seed-diag-source-manifest.csv`: 早期指标来源索引.
- `gd-seed-diag-cross-config-final.csv`: 2026-06-03 跨 codebook/rank 复核 run-level 指标, 含 launch_id, run_id, config, seed, read_topk, codebook size, rank, data_seed, dtype policy, GPU, 日志路径和状态.
- `gd-seed-diag-cross-config-spread-summary.csv`: 跨配置 worst-case, cross-seed spread 和 readk4 rerun spread.
- `gd-seed-diag-cross-config-source-manifest.csv`: 跨配置复核的日志和 manifest 来源索引.
- `metadata.json`: artifact metadata, branches, heads, constraints, conclusions, caveats.

## 主指标

主指标为 `valid/mqar_case/accuracy-1024x256`.

## 最新结论

固定 phase2 read-side `fox_remote_read_topk=4` 不能作为跨配置全局默认方案.

- `cb256-r8`: readk2 s124/s125=`0.988/0.804`, spread=`0.184`; readk4 四条完成 run 为 `0.982/0.982/0.988/0.992`, max-min spread=`0.010`, 稳定有效.
- `cb64-r16`: readk4 s124 r1/r2=`0.831/0.849`, 明显低于 readk2 s124=`0.959`, 伤 high path 且复现.
- `cb128-r8`: readk4 main s124/s125=`0.973/0.972`, 但 s125 rerun=`0.609`, rerun spread=`0.363`, 主结果不复现.

因此 readk4 应记录为 cb256-like 配置下有效的 read-side 稳定候选和诊断开关. 更稳妥的工程方向是 early schedule, margin-aware gate, 或按配置条件启用, 而不是固定默认 readk4.

## 解释

原始 `cb256-r4` 诊断说明 `read_topk=2` 是 basin amplifier: 早期 routing margin 还不可靠时, 过窄 residual read 容易锁到坏候选. `read_topk=4` 在 `cb256-r4` 和 `cb256-r8` 能提供受控候选覆盖并救回弱 seed. 但跨配置复核表明, 这个控制面存在配置边界: 在 `cb64-r16` 会伤害原本 high path, 在 `cb128-r8` rerun 中出现强不稳定.

## Caveats

- `cross-cb64r16-readk2-s124-r1` 是 CUDA OOM, `EXIT_CODE=1`, 不计正式 spread; 有效替代 run 为 `cross-cb64r16-readk2-s124-r1b`.
- 大型 raw logs, checkpoints 和 swanlog 原位保留; 本目录只保存最终关键表和来源索引.
- 当前 artifact 未包含新的源码默认修改; 本轮没有 commit/push, 未启用 `TORCH_DETERMINISTIC=1`.
