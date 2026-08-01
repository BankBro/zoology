# Flash 后期退化低成本因果诊断计划

- `experiment_id`: `20260801-02-flash-late-degradation-causal-diagnosis`.
- Zoology base: `829ab6a960f118c65da93d62ea86463d74a7ef19`.
- Flash-VQG source: `182180fd7a0770caf72b2dec6e6d27616dfd31a3`.
- 正式机器: RTX 3090 GPU0, `Flash-VQG-tun`容器.
- 流程: `Plan -> 实验 -> Report`.

## 1. 目标与判定指标

本实验定位当前Flash在epoch1达到峰值后持续退化的原因, 不预先把现象解释为过拟合. 优先检查`block_len`, selected-read backward和selected chunk三个变化, 必要时检查交互与源码回归. Canonical路径负责因果定位, 当前fastest路径只做迁移复核.

筛选保持4-epoch scheduler不变, 在optimizer step `1232`结束. 定义:

```text
retention = accuracy(step1232) - accuracy(step704)
factor_effect = retention(candidate) - retention(bridge)
```

- 强因果: `factor_effect <= -0.05`.
- 无明显影响: `factor_effect >= -0.02`, 且候选`retention >= -0.02`.
- 中间区间为灰区.
- seed123负责筛选, seed125负责确认; 方向冲突时补seed124, 至少2/3 seeds同向且平均效应达到门槛才确认.

## 2. 固定训练合同

- `baseline-r16-joint + A1 post-phase1 remat`.
- AMP BF16, FP32 master weights和optimizer state, TF32关闭.
- Canonical init、MQAR cache、data seed `123`和固定epoch batch order保持一致.
- Train batch `64`, validation batch `16`, gradient accumulation `4`.
- 每epoch 704 optimizer updates, 4次validation, early stopping关闭.
- FLA fused-gate backward在因果实验中固定为`BT64/warps4/stages2`; 根因确认后使用默认autotune复核.
- 所有训练在单张RTX 3090上串行fresh process执行.

## 3. 第一层筛选矩阵

所有arm固定`grouped_chunk_torch_ref`, `query_w8`, `event_gemv`和`fp32_boundary`.

| Arm | block | local blocks | selected backward | chunk | 目的 |
|---|---:|---:|---|---:|---|
| `ctrl-current` | 64 | 2 | S1 | 8192 | 固定FLA后复现当前退化 |
| `ctrl-bridge` | 32 | 2 | torch | 2048 | 当前源码上的历史路径桥接 |
| `factor-block` | 64 | 2 | torch | 2048 | 只改变block |
| `factor-backend` | 32 | 2 | S1 | 2048 | 只改变selected backward |
| `factor-chunk` | 32 | 2 | torch | 8192 | 只改变chunk |

顺序为`ctrl-current -> ctrl-bridge -> factor-block -> factor-backend -> factor-chunk`. 出现强信号后立即补seed125; 确认成立则停止其余单因素. 若所有单因素稳定, 再执行三组双因素组合并检查三因素交互.

若当前源码上的`ctrl-bridge`也退化, 先在隔离worktree复跑历史Flash-VQG commit `d7dbb12`; 历史稳定而当前失败时, 对兼容且通过smoke的提交执行二分定位.

## 4. Block机制与后续确认

仅当block被确认后执行以下`2x2`矩阵:

| block | local blocks | 局部上限 | 解释 |
|---:|---:|---:|---|
| 32 | 2 | 64 | 历史控制 |
| 32 | 4 | 128 | 扩大局部与远端分界 |
| 64 | 1 | 64 | 只保留较粗逻辑边界 |
| 64 | 2 | 128 | 当前组合 |

- `32x4`单独退化支持局部/远端分界跨度解释.
- `64x1`单独退化支持逻辑记忆边界变粗解释.
- 只有`64x2`退化支持二者交互解释.
- 两者都退化则两种机制均有贡献.

确认阶段从canonical init重新训练4 epochs, 不从筛选断点晋升. 对稳定控制与退化候选的best/last checkpoint执行标准MQAR和Longer-MQAR. 随后在fastest实现上应用对应修正, 使用seeds123/125复核完整轨迹.

若退化arm表现为train loss继续下降而validation下降, 再执行seed123的`结构配置 x 数据暴露`对照: 固定cache重复与每epoch独立训练样本各一组. Fresh data消除退化时才称为数据重复交互导致的过拟合; 否则称为模型状态或优化动力学的后期退化.

## 5. 门禁、失败与输出

Preflight必须核对两个仓库源码身份、clean状态、环境版本、GPU/NVML、cache/init hash、单变量config diff和FLA实际配置. 每种新shape/backend先运行3-step smoke. Runtime fallback、NaN、OOM、checkpoint或dtype审计失败均停止其依赖分支并保留现场; runner/config bug允许最小修复后fresh-process重跑.

输出位置:

- Runner: `zoology/experiments/flash_vqg/scripts/20260801-02-flash-late-degradation-causal-diagnosis/`.
- Raw: runner旁ignored `outputs/3090/<run_tag>/`.
- Generated config: `zoology/experiments/flash_vqg/generated/<launch_id>/`.
- Artifact: `docs/artifacts/20260801-02-flash-late-degradation-causal-diagnosis/`.
- Report: `docs/20260801-02-flash-late-degradation-causal-diagnosis-report.md`.

常规block根因路径预计消耗`1.5-2.5`个3090 GPU小时; 完整交互筛选约`3-4`个GPU小时, 源码二分另计.
