# A1 Block Geometry MQAR Probe

本目录实现`20260730-02-a1-block-geometry-mqar-probe`. 它将candidate的序列长度、KV数量同时放大4倍, 将训练样本数和batch缩小4倍, 保持block数量、query密度、训练tokens和optimizer-step数量可比.

实验已完成, 两个candidate均接近随机, 终态为`quality_rejected`.

入口为`experiment.py`, 原始输出位于`outputs/2080ti/<run-tag>/`. 执行顺序为`prepare-data`, `preflight`, 各variant的`smoke`和`screen`, 最后执行`summarize`.

关联计划: [`docs/plans/20260730-02-a1-block-geometry-mqar-probe-plan.md`](../../../../../docs/plans/20260730-02-a1-block-geometry-mqar-probe-plan.md).

关联报告: [`docs/20260730-02-a1-block-geometry-mqar-probe-report.md`](../../../../../docs/20260730-02-a1-block-geometry-mqar-probe-report.md).

精简证据: [`docs/artifacts/20260730-02-a1-block-geometry-mqar-probe/`](../../../../../docs/artifacts/20260730-02-a1-block-geometry-mqar-probe/README.md).
