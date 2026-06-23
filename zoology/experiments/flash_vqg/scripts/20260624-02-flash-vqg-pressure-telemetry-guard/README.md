# 20260624-02 Flash-VQG pressure telemetry guard

本目录是 `20260624-02-flash-vqg-pressure-telemetry-guard` 的脚本目录.

## 阶段 1: config-to-runtime smoke

- 检查 `hard04` 的 write cap runtime 指标.
- 检查 cap release 的 effective cap 和 release progress 指标.
- 检查 `update_norm` 分布与 `update_norm_cap` 指标.

阶段 1 不实现 guarded release, 不跑长训练, 不产生正式 MQAR ledger.

运行:

```bash
bash zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/run_config_runtime_smoke.sh
```

## 阶段 2: 最小 telemetry probe

阶段 2 跑 `cb64-r16` 的 pressure telemetry probe, 不实现 guard.

目标矩阵:

```text
3090:   default-s123, hard04-s123, cap0405-s123, caprel0406late-s123
2080ti: default-s124, hard04-s124, cap0405-s124, caprel0406late-s124
```

并发规则:

- 3090 单卡最多 3 条 run.
- 2080ti 两张卡各 1 条 run, 不在单张 2080ti 上叠两条 run.
- release 配置统一使用 `write_strength_cap_eval_policy=scheduled`, 便于观察 release 前后压力曲线.

启动:

```bash
bash zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/start_stage2_probe_queue.sh 2080ti-seed124
bash zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/start_stage2_probe_queue.sh 3090-seed123
```

直接跑单条:

```bash
GPU_ID=0 bash zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/run_stage2_probe_train.sh hard04-s123
```

输出默认写入:

```text
zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/outputs/
```

`outputs/` 是本地临时产物, 默认不提交. 收尾时只把轻量 summary/metadata 提炼到 `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/`.
