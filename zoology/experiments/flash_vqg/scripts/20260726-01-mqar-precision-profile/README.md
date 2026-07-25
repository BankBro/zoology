# MQAR 低精度与长度泛化自动队列

本目录实现 `20260726-01-mqar-precision-profile` 的配置生成, AMP/恢复训练, 可恢复逐 batch 评估, batch 容量搜索, 全局 gate 和双机串行 formal 队列.

## 启动

2080 Ti 容器内:

```bash
cd /home/lyj/mnt/project/zoology
MQAR_PRECISION_MACHINE=2080ti bash zoology/experiments/flash_vqg/scripts/20260726-01-mqar-precision-profile/start_queue.sh
```

3090 的 `Flash-VQG-tun` 容器内:

```bash
cd /home/lyj/mnt/project/zoology
MQAR_PRECISION_MACHINE=3090 bash zoology/experiments/flash_vqg/scripts/20260726-01-mqar-precision-profile/start_queue.sh
```

两机启动后, 在 2080 Ti 容器内启动全局 gate coordinator:

```bash
tmux new-session -d -s mqar-precision-coordinator \
  '/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python /home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260726-01-mqar-precision-profile/coordinator.py'
```

## 状态与恢复

每台机器的状态位于:

```text
outputs/machines/<machine>/status.json
outputs/machines/<machine>/heartbeat.json
outputs/machines/<machine>/logs/queue.log
```

队列命令幂等. tmux 或子进程意外退出后, 重新执行对应 `start_queue.sh` 即可. file lock 防止重复 worker; training 从 `resume.pt` 的 optimizer boundary 恢复, eval 从逐 batch `progress.json` 恢复.

正式训练只在两机 `LOCAL_SMOKE_PASSED.json` 都通过并由 coordinator 生成相同的 `GLOBAL_FORMAL_GATE.json` 后启动.
