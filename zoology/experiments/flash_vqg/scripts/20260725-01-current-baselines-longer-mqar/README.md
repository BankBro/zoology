# 当前基线 Longer-MQAR runner

本目录实现 `20260725-01-current-baselines-longer-mqar` 的实验阶段. 正式口径以
`docs/plans/20260725-01-current-baselines-longer-mqar-plan.md` 为准.

入口:

- `experiment.py`: resolved config、preflight、六条训练 run、checkpoint manifest.
- `longer_mqar_runner.py`: manifest-driven batch search、全 source smoke、formal eval 和 repro.
- `run_queue.py`: fail-fast、可恢复的完整 smoke/formal DAG.
- `start_queue.sh`: 按 `2080ti|3090` 参数在对应GPU的tmux session中启动无人值守队列.
- `collect_artifacts.py`: DONE 后生成单机artifact、raw evidence镜像和ledger候选行.
- `collect_cross_machine_artifacts.py`: 验证两机artifact并生成120行combined结果和跨机器delta.
- `make_longer_mqar_figure.py`: 从combined结果生成独立的last/best跨GPU正式曲线.

启动示例:

```bash
./start_queue.sh 2080ti
./start_queue.sh 3090
```

2080 Ti保留legacy `outputs/`; 3090写入 `outputs/machines/3090/`, 两机checkpoint root和launch ID也独立.

所有本地运行输出写入本目录的 `outputs/`, 默认不提交. Formal 完成后运行机器级和
跨机器collector, 再生成两张正式图并进入 Report 阶段.
