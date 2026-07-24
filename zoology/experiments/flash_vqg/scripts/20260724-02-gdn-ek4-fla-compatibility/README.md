# GDN ek4-ev4 FLA 兼容性实验

本目录验证官方 FLA v0.4.2/v0.5.0 对 RTX 3090 shared-memory 启动失败的修复, 并冻结共同 benchmark 环境. 所有命令固定 FP32、`GDN_KERNEL_DTYPE=float32`、`TRITON_F32_DEFAULT=ieee` 和 TF32 off.

## 环境

候选环境分别位于:

- `/home/lyj/miniconda3/envs/flash-vqg-fla042`
- `/home/lyj/miniconda3/envs/flash-vqg-fla050`

官方源码 worktree 分别固定到 v0.4.2 `ca910f88529565b28b6e16465258f2e239a02dc7` 和 v0.5.0 `3a9ce1c83a13994d824dbb3421e2989d330bb38b`.

## 入口

- `compatibility_benchmark.py`: 复用上一轮已审计的 preflight、production shape、timing 和 memory runner, 额外记录完整依赖版本与 kernel hash.
- `equivalence_capture.py`: 分环境保存 kernel/full-model 数值结果, 再离线比较 forward、final state、梯度和一步更新.
- `environment_snapshot.py`: 保存 `pip freeze`, `pip check`, Conda package list、GPU 和源码/kernel hash.
- `run_benchmark_matrix.sh`: 单机器、单依赖版本的五重复 timing 和独立 memory 队列.
- `run_paired_benchmark.sh`: 在同一 GPU 上交替 v0.4.2/v0.5.0 顺序, 生成正式配对 timing/memory 矩阵.
- `run_formal_queue.sh`: 带 checkpoint、完整验证和状态记录的 1ep 正式质量队列.
- `collect_artifacts.py`: 汇总 compatibility、等价、五重复 timing/memory、1ep、版本选择和 source manifest.

2080 Ti 使用 `GPU=1`, 3090 使用 `GPU=0`. Raw 输出位于 `outputs/`, 默认不提交; 轻量证据在收尾时提炼至正式 artifact.
