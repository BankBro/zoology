# MQAR seed124 remat 数值分叉因果诊断报告

## 1. 结果概览

实验已完成因果闭环, 终态为 `causal_root_identified`.

seed124 的剩余 A0/A1 分叉来自 FLA 0.4.2 `FusedRMSNormGated` backward 的 fresh-process Triton autotune. Triton 3.2 环境不保存该 autotune 结果, 每个进程会重新测速; `BT64, warps4` 和 `BT64, warps8` 使用不同 FP32 归约顺序, 导致 layer1 `output_gate_fused.weight` 梯度出现约 `1e-12` 的差异.

这说明上一轮 seed124 的最终 hash 分叉不能直接归因于 remat 改变模型数学语义. A0 和 A1 都会受到该外部 fused gate autotune 影响, 独立进程恰好选择不同 config 时才会形成不同训练轨迹.

## 2. 首次分叉时间线

初始 16-step 和 32-step A0/A1 fresh-process 恰好选择了相同归约结果, 因而完全一致. 128-step pair 选择了不同结果, 复现了正式训练的轨迹分叉.

| 阶段 | 首次差异位置 | 观测 |
|---|---|---|
| Gradient | window1, microbatch0 | 仅 layer1 `output_gate_fused.weight` 不同 |
| Adam state | window1 optimizer 后 | moment hash 不同 |
| Model state | window4 optimizer 后 | 参数 hash 首次不同 |
| Base CE loss | window10, microbatch1 | `9.0065918` vs `9.0068359` |
| 完整 window loss | window10 | `10.5231247` vs `10.5232010` |

window10 的两种完整 loss 与上一轮正式 telemetry 的两个值完全相同, 但本轮 A0/A1 方向相反. 这与“结果由独立进程选择的归约 config 决定, 而不是由 remat 标签决定”一致.

## 3. 根因证据

### 3.1. 参数级定位

8条详细 window1 轨迹只形成两个 gradient hash group. 两组之间只有一个参数不同:

```text
backbone.layers.1.sequence_mixer.mixer.attn.output_gate_fused.weight
shape = [64]
```

42/64 个 FP32 元素不同, 最大绝对差为 `1.8189894035458565e-12`, 平均绝对差为 `4.085620730620576e-13`. 该量级不会立即改变 FP32 参数, 但会从step1开始改变 Adam moment, 随后在step4改变参数并在window10改变loss.

### 3.2. Autotune config 与梯度一一对应

新增的12个默认 fresh-process run记录了实际 Triton config:

```text
BT64, warps4 -> aggregate gradient b45a9c4a...
BT64, warps8 -> aggregate gradient ccb031cb...
```

固定 `BT16/32/64` 还会产生各自稳定的 weight gradient. FLA backward 先由多个 Triton program 生成 FP32 partial `dw`, 再执行 `dw.sum(0)`. 不同 `BT/num_warps` 改变浮点加法顺序, 因而最后几位不同.

### 3.3. 真实算子 replay

从正式模型的 layer1、window1、microbatch0 保存了8.1 MiB最小 capsule, 包含 gate 输入、weight和真实 `grad_output`. 对同一 capsule 分别固定 `warps4/warps8`:

- Forward output 完全相同.
- `grad_x` 和 `grad_gate` 完全相同.
- `grad_weight` 精确复现两个训练梯度 hash.

这将根因缩小到 FLA fused RMSNorm gate backward 的 weight reduction, 排除了数据、RNG、remat forward、selected-read、grouped recurrence和AdamW本身是首个差异来源.

## 4. 单变量因果干预

诊断性干预只固定 FLA gate backward 为 `BT64, warps4`, 不改变模型配置和公式.

| 门禁 | 结果 |
|---|---:|
| A1 fixed fresh-process 重复 | 4/4逐事件一致 |
| Fixed A0/A1 2-step | 完全一致 |
| Fixed A0/A1 177-step训练事件 | 1947/1947一致 |
| 最终 model hash | 相同 |
| 最终 optimizer hash | 相同 |
| 第176步后和终止validation质量指标 | 全部相同 |

177-step验证越过了首个分叉后128步, 并执行了第176步后的正式validation生命周期. A1仍保留预期显存收益: 终止validation peak reserved为`2424 MiB`, A0为`3154 MiB`.

## 5. 实验偏差与边界

第一次 run `20260729-seed124-diag-01` 因标量hash未展平且工作目录错误而在首个microbatch forward后失败, backward和optimizer均未执行. 错误位置生成的412 MiB cache已移入该失败run的ignored outputs保留, 不作为证据.

有效 run为`20260729-seed124-diag-02`. 诊断轨迹会在microbatch边界同步GPU, 可能改变autotune获胜配置的分布, 但真实算子replay和固定config干预仍精确复现并消除了两个梯度结果, 因而不影响根因裁决.

有效run共40个短probe进程, 累计约`0.351` GPU小时. 加上失败run和replay仍远低于预注册的6 GPU小时预算.

本结论绑定 FLA 0.4.2、Triton 3.2.0和当前RTX 3090. `BT64, warps4`只是因果诊断值, 尚未经过生产吞吐选择. 本轮没有执行完整三seed 4ep质量回归, 因此A1仍不晋升.

## 6. 决策与下一步

- A0继续作为canonical实现, A1暂不用于300M自然语言正式训练.
- 下一轮不应继续修改remat或selected-read来处理该剩余分叉, 应先生产化确定性output gate backward.
- 生产实现不应依赖进程内修改第三方Autotuner. 应在Flash-VQG中增加显式、可审计的固定gate kernel/backend, 并比较`BT64, warps8`等候选吞吐.
- 修复后先重复fresh-process、177-step和validation门禁, 再重跑三seed MQAR与Longer-MQAR正式回归. 只有三seed质量和最终hash同时通过后才考虑晋升A1.

详细证据见 [artifact](artifacts/20260729-03-mqar-seed124-remat-causal-diagnosis/README.md) 和 [计划](plans/20260729-03-mqar-seed124-remat-causal-diagnosis-plan.md).
