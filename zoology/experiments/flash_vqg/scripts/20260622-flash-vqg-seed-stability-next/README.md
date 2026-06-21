# Flash-VQG seed stability next-step scripts

本目录承接 `docs/plans/20260622-flash-vqg-seed-stability-roadmap.md` 的 P0 readiness.

## 当前脚本

- `config_runtime_smoke.py`: 构造最小 `gd_residual_v1` 配置, 检查 static config, generated config 和 runtime metrics.
- `run_config_runtime_smoke.sh`: 从仓库根目录运行 smoke, 输出到本目录的 ignored `outputs/`.

## 产物

本地中间输出默认写入:

```text
zoology/experiments/flash_vqg/scripts/20260622-flash-vqg-seed-stability-next/outputs/config-runtime-smoke-<timestamp>/
```

`outputs/` 目录不提交. 若 smoke 支撑后续报告, 只把整理后的轻量 summary/metadata/README 提炼到 `docs/artifacts/YYYYMMDD-experiment-name/`.

## 运行

```bash
cd /home/lyj/mnt/project/zoology
bash zoology/experiments/flash_vqg/scripts/20260622-flash-vqg-seed-stability-next/run_config_runtime_smoke.sh
```
