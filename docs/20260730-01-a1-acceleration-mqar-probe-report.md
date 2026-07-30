# A1 训练加速候选 MQAR 筛选报告

## 1. 实验登记

- Experiment ID: `20260730-01-a1-acceleration-mqar-probe`.
- 状态: `completed`, 终态为`quality_rejected`.
- 执行机器: RTX 2080 Ti GPU1.
- Seed和精度: seed 123, FP32.
- Flash-VQG运行commit: `114eadbd1d2e3c9a43b927e54f6ad9a2692c40e8`.
- Zoology分支: `20260730-055000-a1-acceleration-mqar-probe`.
- Plan: [实验计划](plans/20260730-01-a1-acceleration-mqar-probe-plan.md).
- Artifact: [精简证据](artifacts/20260730-01-a1-acceleration-mqar-probe/README.md).

本实验是单seed低成本筛选, 不替代三seed 4-epoch正式回归. Reference和候选都启用A1 `post_phase1` remat及新的`triton_deterministic` selected-read backward, 只比较大block和稀疏top-k近似.

## 2. 固定合同与执行修复

Preflight核对了RTX 2080 Ti GPU1, `flash-vqg-fla042`, canonical init和13个MQAR cache. 两个模型均为1,160,390个参数, 严格加载同一初始状态. Reference为`block32/write4/read16`, 组合候选为`block256/write2/read8`.

正式评估过程中出现5次基础设施或资源失败, 均保留现场、分析并修复后重试:

| Attempt | 失败原因 | 修复与结果 |
|---|---|---|
| 01 | Shadow checkpoint的seed类型与resume identity不一致 | 规范化identity后通过 |
| 02 | 评估batch 128在FP32下OOM | 改用已验证的shape级batch |
| 03 | 4096长度误用batch 32后OOM | 修正为Flash FP32注册batch 16 |
| 04 | Event ID未包含batch和源码commit, 与旧事件冲突 | 扩展event identity后通过 |
| 05 | 候选8190长度batch 16发生allocator碎片OOM | 启用`expandable_segments`后完成 |

这些失败均发生在评估基础设施, 不改变已经完成的训练checkpoint和最终质量结论.

## 3. 主结果

### 3.1. 标准任务与长度外推

| Variant | 标准`1024x256` | 相对reference | 四外推slice宏平均delta | 门禁 |
|---|---:|---:|---:|---|
| A1 reference | 0.959813 | 0.000000 | 0.000000 | reference |
| `block256/write2/read8` | 0.218785 | -0.741027 | -0.573613 | rejected |

组合候选在四个外推slice上分别下降`0.817949`, `0.542336`, `0.710449`和`0.223716`. 标准门槛为`-0.02`, 外推宏平均门槛为`-0.05`, 两者均明显失败.

### 3.2. 单变量诊断

| Variant | 标准`1024x256` | 相对reference | 解释 |
|---|---:|---:|---|
| `block128` | 0.060676 | -0.899137 | 严重退化 |
| `block256` | 0.199504 | -0.760309 | 严重退化 |
| `write2/read8` | 0.908125 | -0.051688 | 中等退化, 不是组合失败的主来源 |

Reference使用新的Triton deterministic selected backward仍达到`0.959813`, 说明该exact kernel至少通过了seed123、1 epoch的端到端学习canary. 这不是三seed正式非劣证明, 但可以排除“新kernel使模型完全学不起来”.

## 4. 解释与证据边界

单变量结果显示大block是组合候选退化的主要关联因素, 但上游curriculum最长训练序列只有256 tokens. `block128/256`因此只产生2个或1个block, remote path没有获得与canonical `block32`相当的训练机会. 不能仅凭本轮把退化全部归因于大block的固有能力.

为验证该混杂, 后续实验`20260730-02`同步缩放序列、KV数量、样本数和batch, 使训练tokens、更新次数及block几何匹配. 在该实验完成前, 本轮已经足以停止最快近似候选的三seed和Longer-MQAR扩展, 但不足以宣布所有大block设计在自然语言上必然失败.

## 5. 结论

`block256 + write2/read8`不得作为300M Flash的质量安全加速路径. `write2/read8`也未通过本轮`-0.02`标准门槛. 继续性能优化应优先保持逻辑`block_len`不变, 通过物理tiling或kernel融合降低launch和HBM流量.

Exact `triton_deterministic` selected backward保留为性能候选. 在自然语言正式预训练前, 仍需三seed回归或短自然语言paired pilot补齐其质量证据.

## 6. 原始证据

原始输出保留在:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260730-01-a1-acceleration-mqar-probe/outputs/2080ti/
20260730-a1-accel-mqar-01
20260730-a1-accel-mqar-02
```

其中Run01包含主训练、固定hash Longer-MQAR评估及5次失败现场. Run02包含`block128`, `block256`和`write2/read8`单变量诊断. Checkpoint和raw JSONL不进入Git.
