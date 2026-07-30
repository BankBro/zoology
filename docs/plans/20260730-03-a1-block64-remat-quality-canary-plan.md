# A1 Block64 Remat 质量基线实验计划

## 1. 实验登记

- Experiment ID: `20260730-03-a1-block64-remat-quality-canary`.
- 状态: planned.
- 登记日期: 2026-07-30.
- 分支: `20260730-144531-a1-block64-remat-quality-canary`.
- Zoology base commit: `8e62278ab2b612f4624bb2d32df86ab4a0b96f89`.
- Flash-VQG source base: `048e89325676b9a3bdd180ff72da360ce9db741d`.
- 上游性能实验: Flash-VQG `20260730-02-a1-recompute-selective-save`.
- 机器: RTX 2080 Ti GPU1.
- dtype: FP32.

本实验建立与300M性能几何一致的block64低成本质量基线. A0与A1只切换GD `post_phase1` remat, 其他模型、数据、selected backward和训练合同完全相同. 本实验不验证C1/K1, 不替代后继三seed四epoch正式门禁.

## 2. 固定合同与矩阵

两组共同使用seed123、data seed123、canonical init/cache、`block_len=64`、`write_topk=4`、`read_topk=16`、`gd_rank=16`、deterministic selected backward、1 epoch、train/eval batch `64/16`和GA4.

| Variant | Remat | 其他差异 | 预期 |
|---|---|---|---|
| `a0-block64` | `off` | 无 | 建立同几何reference |
| `a1-block64` | `post_phase1` | 无 | 在相同FLA config下与A0轨迹和质量一致 |

正式对比只允许上述一个变量. 两组必须使用同一份内容级cache、初始化tensor、训练顺序和评估slice.

## 3. 执行流程

**(1)** Preflight核对zoology与Flash-VQG commit、canonical Python、GPU1、CUDA/FLA、cache tensor hash、init hash、参数量和resolved config diff.

**(2)** 两组各执行3 optimizer steps smoke, 验证finite loss、无fallback、checkpoint可读和remat audit符合预期.

**(3)** 使用默认FLA分别完成1 epoch正式训练, 保存best/last checkpoint、逐步loss、model/optimizer hash、时间和实际autotune config证据.

**(4)** 对last checkpoint评估标准`1024x256`任务和5个固定hash Longer-MQAR slice, 汇总4个真正外推slice宏平均.

**(5)** 若fresh-process轨迹或hash分叉, 不直接归因remat. 记录实际FLA fused-gate backward config, 使用现有capture/replay机制在同config下复验A0/A1. 不修改第三方FLA源码, 不把固定warps设为生产默认.

**(6)** 生成paired summary、source manifest、metadata、artifact和report. 本次单seed结果不写入canonical rank/seed ledger.

Longer-MQAR沿用已注册的2080ti Flash FP32 batch: `1024/2048`使用B32, `4096`和两个`8190` slice使用B16.

## 4. 双门禁

### 4.1. 轨迹门禁

- 所有loss、输出、梯度摘要和checkpoint tensor finite.
- 默认FLA主运行记录固定步骤loss和最终model/optimizer hash.
- 若默认FLA config不同导致hash分叉, matched-config capture/replay必须使A0/A1回到仓库注册数值容差内; 无法解释的分叉判为失败.
- Remat logical counter、selected backward和fallback audit必须符合预期.

### 4.2. 质量门禁

```text
standard 1024x256 delta >= -0.01
four extrapolation slices macro delta >= -0.02
```

两类门禁必须同时通过. 仅准确率相近但存在无法解释的训练轨迹分叉, 或轨迹一致但质量低于门槛, 均不得为Flash-VQG C1/K1放行.

## 5. 资源与失败策略

- 只使用2080ti GPU1, 总预算上限2 GPU-hours.
- API费用为0.
- 每次运行使用唯一run tag, 不覆盖失败目录.
- 预计超过30分钟时监控到训练稳定循环, 记录进度、GPU、loss、输出路径和ETA.
- 失败后保留现场并做有依据的最小修正; cache、环境、源码或GPU硬门禁失败时停止全部正式训练.

实验无论completed、failed或aborted都必须生成`docs/20260730-03-a1-block64-remat-quality-canary-report.md`. Completed结果生成`docs/artifacts/20260730-03-a1-block64-remat-quality-canary/`, 并追加`docs/EXPERIMENT_LOG.md`. 只有当前基线或下一步变化时更新`docs/STATUS.md`.
