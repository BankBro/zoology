# Flash后期退化因果诊断

本目录实现`20260801-02-flash-late-degradation-causal-diagnosis`的自适应实验队列.

```bash
export MQAR_LATE_DEGRADATION_RUN_TAG=20260801-late-degradation-01
bash zoology/experiments/flash_vqg/scripts/20260801-02-flash-late-degradation-causal-diagnosis/start_queue.sh
```

队列固定使用RTX 3090 GPU0. Raw输出位于`outputs/3090/<run_tag>/`; 正式结论以对应report和`docs/artifacts`中的轻量审计文件为准.
