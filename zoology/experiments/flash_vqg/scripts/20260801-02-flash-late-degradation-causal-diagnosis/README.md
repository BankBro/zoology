# Flash后期退化因果诊断

本目录实现`20260801-02-flash-late-degradation-causal-diagnosis`的自适应实验队列.

```bash
export MQAR_LATE_DEGRADATION_RUN_TAG=20260801-late-degradation-01
bash zoology/experiments/flash_vqg/scripts/20260801-02-flash-late-degradation-causal-diagnosis/start_queue.sh
```

队列固定使用RTX 3090 GPU0. Raw输出位于`outputs/3090/<run_tag>/`; 正式结论以对应report和`docs/artifacts`中的轻量审计文件为准.

主队列完成且满足预注册的数据暴露触发条件后, 使用同一`run_tag`运行fresh-per-epoch补充对照:

```bash
export MQAR_LATE_DEGRADATION_RUN_TAG=20260801-late-degradation-01
CUDA_VISIBLE_DEVICES=0 /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260801-02-flash-late-degradation-causal-diagnosis/fresh_data_followup.py run
```

补充对照只新增seed123的`ctrl-bridge`和`factor-block`两次4-epoch训练. Epoch0复用canonical cache, epoch1至epoch3使用独立数据seed和隔离的CPU Torch RNG生成新样本; 验证集、初始化、优化器、scheduler和更新数保持不变.
