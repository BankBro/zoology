# A1 Block64 Remat 质量门禁报告

## 1. 实验登记

- Experiment ID: `20260730-03-a1-block64-remat-quality-canary`.
- 状态: `completed`, 终态为`passed`.
- 执行机器: RTX 2080 Ti GPU1.
- Seed和精度: seed 123, FP32, 1 epoch.
- Zoology运行commit: `2b86b133915df2e0d1ddad582c6e924f7a03d724`.
- Flash-VQG运行commit: `a18b32960b41170b5a546588b3b2ebbd0f0578b7`.
- Plan: [实验计划](plans/20260730-03-a1-block64-remat-quality-canary-plan.md).
- Artifact: [精简证据](artifacts/20260730-03-a1-block64-remat-quality-canary/README.md).

本实验在MQAR小模型中比较A0 `remat=off`与A1 `remat=post_phase1`. 两组除remat开关外配置完全相同, 并使用`block_len=64`, `write_topk=4`, `read_topk=16`和`triton_deterministic` selected backward. 目的不是替代三seed正式回归, 而是在投入C1/K1性能工程前补齐block64的低成本质量门禁.

## 2. 门禁与执行

两组均完成3-step smoke、704个optimizer step的一epoch训练、标准validation及5个锁定数据集评估. Runtime audit均通过, fallback为0. A1记录2815次selected recompute, A0为0, 说明配对确实覆盖了目标remat路径.

预注册门槛为:

- 标准`1024x256` accuracy delta不少于`-0.01`.
- 4个外推任务的accuracy宏平均delta不少于`-0.02`.
- 若fresh process出现训练轨迹或state hash分叉, 必须先核对FLA fused gate autotune配置, 不能直接归因于remat.

## 3. 主要结果

### 3.1. 质量与轨迹

| 指标 | A0 | A1 | Delta |
|---|---:|---:|---:|
| 标准validation `1024x256` | 0.966758 | 0.966758 | 0.000000 |
| 锁定eval `1024x256` | 0.966859 | 0.966859 | 0.000000 |
| `2048x512` | 0.851754 | 0.851754 | 0.000000 |
| `4096x1024` | 0.548705 | 0.548705 | 0.000000 |
| `8190x512` | 0.716687 | 0.716687 | 0.000000 |
| `8190x2047` | 0.223516 | 0.223516 | 0.000000 |

4个外推任务的宏平均delta为`0.000000`. 704个共同step的loss最大绝对差为0, 最终model state和optimizer state hash也完全相同. 本次A0/A1均由FLA选择`BT=32, num_warps=4`的fused gate backward配置.

这些证据支持一个窄而明确的结论: 在相同外部kernel配置下, 当前A1 remat在block64 MQAR路径中可以保持与A0完全一致的训练轨迹和质量.

### 3.2. 资源信号

A0和A1总训练wall time分别为270.60 s和309.26 s, A1增加约14.3%. Validation记录的peak reserved由3132 MiB降至2662 MiB, 下降470 MiB; peak allocated同为1053.93 MiB. 这些是小模型FP32辅助信号, 不能代替300M BF16的正式性能测量.

## 4. 失败现场与修复

首次run `20260730-block64-remat-01`在最长`8190x512`评估中OOM. 根因是FP32 batch16的LM head logits单次需要约4 GiB, 不是模型训练或remat失败. 修复将两个8190-token任务的eval batch降为4, 保持checkpoint、样本、数据hash、精度和指标定义不变, 并以新tag从头重跑完整流程.

首次run还记录到A0选择`BT64/warps8`, A1选择`BT32/warps4`, 最终state hash不同. 这复现了既有FLA autotune混杂. 完整通过的第二次run中两组配置相同且逐step、最终state及全部质量指标完全一致, 因此不支持把首次分叉归因于remat数学语义.

## 5. 结论与边界

Q0质量门禁通过, 因而允许继续Flash-VQG中的P0/P1观测和后继C1/K1工程探索. 当前不需要修改第三方FLA源码或永久固定warps.

本实验只有单seed、FP32、小模型和1 epoch. 它不能证明300M BF16自然语言训练在长时间尺度上必然无漂移, 也不能单独授权1B-token正式训练. 在正式预训练前仍应完成短自然语言A0/A1 paired pilot, 并继续记录FLA实际autotune配置.

## 6. 原始证据

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260730-03-a1-block64-remat-quality-canary/outputs/2080ti/
20260730-block64-remat-02
```

失败现场保存在同级`20260730-block64-remat-01`. Raw目录包含preflight、smoke、完整训练、checkpoint、704-step telemetry、5任务评估和终态summary. 大型checkpoint不进入Git.
