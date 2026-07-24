# 当前基线 Longer-MQAR runner

本目录实现 `20260725-01-current-baselines-longer-mqar` 的实验阶段. 正式口径以
`docs/plans/20260725-01-current-baselines-longer-mqar-plan.md` 为准.

入口:

- `experiment.py`: resolved config、preflight、六条训练 run、checkpoint manifest.
- `longer_mqar_runner.py`: manifest-driven batch search、全 source smoke、formal eval 和 repro.
- `run_queue.py`: fail-fast、可恢复的完整 smoke/formal DAG.
- `start_queue.sh`: 在 2080 Ti GPU1 的 tmux session 中启动无人值守队列.

所有本地运行输出写入本目录的 `outputs/`, 默认不提交. Formal 完成后由用户唤醒
会话, 再运行 artifact collector 并进入 Report 阶段.
