# E5A Top2 Audit

本目录承载 E5A v1 的主审计脚本与后续扩展入口.

当前主文件:

- `e5a_audit.py`: E5A eval-only 审计主逻辑

当前调用方式:

- `zoology/experiments/flash_vqg/eval_only.py` 会按文件路径动态加载本目录下的 `e5a_audit.py`
- `run_flash_vqg_suite.py --eval-only e5a` 仍是统一入口
