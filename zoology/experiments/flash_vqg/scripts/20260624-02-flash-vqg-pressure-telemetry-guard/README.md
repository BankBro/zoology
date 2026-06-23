# 20260624-02 Flash-VQG pressure telemetry guard

本目录是 `20260624-02-flash-vqg-pressure-telemetry-guard` 的第一阶段脚本.

当前阶段只做 config-to-runtime smoke:

- 检查 `hard04` 的 write cap runtime 指标.
- 检查 cap release 的 effective cap 和 release progress 指标.
- 检查 `update_norm` 分布与 `update_norm_cap` 指标.

本阶段不实现 guarded release, 不跑长训练, 不产生正式 MQAR ledger.

运行:

```bash
bash zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/run_config_runtime_smoke.sh
```

输出默认写入:

```text
zoology/experiments/flash_vqg/scripts/20260624-02-flash-vqg-pressure-telemetry-guard/outputs/
```

`outputs/` 是本地临时产物, 默认不提交. 收尾时只把轻量 summary/metadata 提炼到 `docs/artifacts/20260624-02-flash-vqg-pressure-telemetry-guard/`.
