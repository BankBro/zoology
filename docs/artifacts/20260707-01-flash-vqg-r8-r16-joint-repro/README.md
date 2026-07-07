# 20260707-01-flash-vqg-r8-r16-joint-repro

本 artifact 汇总本轮 R8/R16 joint-control 复现实验的可用结果和失败原因。

原计划是在 3090 GPU0, 2080ti GPU0, 2080ti GPU1 上分别跑 `read_topk=8` 和 `read_topk=16`。实际执行中, 2080ti GPU0 在 R8 后卡住且容器内 NVML/CUDA 失效, 因此正式 paired 复现实验中止。随后只在 3090 上补跑了 R16。

关键文件:

- `run-summary.csv`: 每个 run 的状态和最终指标。
- `variant-summary.csv`: 按 variant 汇总的可用结果和 gap。
- `mechanism-metrics-summary.csv`: 完成 run 的机制指标。
- `source-manifest.csv`: 镜像的轻量证据和源路径。
- `metadata.json`: 本轮执行口径, commit 和硬件失败说明。
- `raw-evidence/`: queue status, generated config, result json 和精简 final metric log lines。

注意: 本轮不能作为完整 cross-machine pass/fail 证明。
