# MQAR 确定性 Selected-Read 回归实验

## 1. 实验登记

- Experiment ID: `20260729-02-mqar-deterministic-selected-read-regression`.
- 状态: `implementation`.
- 目标: 修复 A0/A1 共同 selected-read backward 的 `addr_proj` 非确定性归约, 重新执行三 seed MQAR 与 Longer-MQAR 回归.
- Plan: [`docs/plans/20260729-02-mqar-deterministic-selected-read-regression-plan.md`](../../../../../docs/plans/20260729-02-mqar-deterministic-selected-read-regression-plan.md).
- Report: 实验终态后生成 `docs/20260729-02-mqar-deterministic-selected-read-regression-report.md`.
- Artifact: 实验终态后生成 `docs/artifacts/20260729-02-mqar-deterministic-selected-read-regression/`.

## 2. 固定矩阵

| Variant | `fox_gd_residual_remat_mode` | Seeds |
|---|---|---|
| `a0-fixed-off` | `off` | `123/124/125` |
| `a1-fixed-post-phase1` | `post_phase1` | `123/124/125` |

两组共同使用确定性 selected-read backward, 并固定 RTX 3090 BF16、`block_len=32`、B64、validation B16、GA4、4 epochs、canonical cache/init 和 data seed 123. `last.pt` 是主结果, `best.pt` 只用于敏感性分析.

## 3. 阶段与入口

队列依次执行:

```text
preflight -> determinism -> smoke -> formal -> evaluate -> collect
```

任一前置门禁或运行阶段失败时立即停止并保留当前 run 目录. 3090 容器内启动:

```bash
cd /home/lyj/mnt/project/zoology
bash zoology/experiments/flash_vqg/scripts/20260729-02-mqar-deterministic-selected-read-regression/start_queue.sh 3090
```

原始输出位于:

```text
zoology/experiments/flash_vqg/scripts/
20260729-02-mqar-deterministic-selected-read-regression/outputs/3090/<run-tag>/
```

## 4. 裁决

A1 必须同时满足:

```text
每个 seed 的 fixed A0/A1 final model-state hash 完全一致
mean standard MQAR delta >= -0.01
mean four-slice extrapolation macro delta >= -0.02
```

质量和确定性均通过前, A1 不恢复为 300M 自然语言质量 pilot 候选. 性能与显存只记录, 不作为硬门槛.
