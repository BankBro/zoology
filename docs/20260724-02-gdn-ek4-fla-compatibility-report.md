# GDN ek4-ev4 FLA 3090 兼容性与共同环境报告

实验 ID: `20260724-02-gdn-ek4-fla-compatibility`.

## 1. 结论

升级官方 FLA 可以解决 `gdnxk-h2-ek4-ev4-usegate0` 在 RTX 3090 上的 shared-memory kernel 启动失败. 官方 FLA v0.4.2 和 v0.5.0 都已在 3090 上通过 production `train B64/T256` forward、CE、backward, 以及 `eval B16/T1024` full batch. 本轮没有修改 FLA 源码或 GDN 数学实现.

最终共同环境选择 FLA v0.4.2, PyTorch 2.6.0+cu118, Triton 3.2.0. 不选择 v0.5.0 的主要原因是:

1. 3090 GDN train 的五重复稳态中位数比 v0.4.2 慢 9.11%, 超过预设 2% 回退线, 且 5/5 paired repeats 都更慢.
2. 3090 上 v0.5.0 相对 v0.4.2 的 kernel 单算子等价通过, 但 full-model one-step 严格门槛失败. Forward logits 和 loss 仍在门槛内, 但部分梯度 relative L2 达到 `1.23e-4`, 一次 AdamW 更新后的参数 max abs 达到 `1.72e-3`.
3. v0.5.0 还需要同时升级 PyTorch/Triton并适配移除的 API, 工程变量更多. v0.4.2 保持现有 PyTorch/Triton 主栈即可解决 3090 兼容性.

选中 v0.4.2 后, Flash-VQG 与同量级 GDN 在两机的 core train/eval 时间及 allocated memory 均已通过 `Flash/GDN <=2x`:

| 机器 | 阶段 | Flash p50 | GDN p50 | 时间比 | Flash allocated | GDN allocated | 显存比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2080 Ti GPU1 | eval | 79.967 ms | 43.915 ms | 1.821x, PASS | 1.020 GiB | 1.021 GiB | 0.999x, PASS |
| 2080 Ti GPU1 | train GA4 | 953.861 ms | 633.341 ms | 1.506x, PASS | 3.051 GiB | 1.875 GiB | 1.627x, PASS |
| 3090 GPU0 | eval | 54.372 ms | 49.993 ms | 1.088x, PASS | 1.020 GiB | 1.021 GiB | 0.999x, PASS |
| 3090 GPU0 | train GA4 | 844.080 ms | 881.370 ms | 0.958x, PASS | 3.051 GiB | 1.875 GiB | 1.627x, PASS |

双机正式 1ep 质量回归也全部通过. 最终 Flash v0.4.2 的 2080 Ti/3090 overall accuracy 为 0.985319/0.988404, `1024x256` 为 0.918789/0.939551. 相对上一轮同机 seed124 Flash 结果, overall delta 为 -0.227pp/+0.168pp, hard delta 为 -1.442pp/+0.923pp, 均满足非回归门槛.

三重复预编译完整 epoch 进一步关闭了上一轮唯一缺失的 3090 hard ratio. 2080 Ti 的 Flash/GDN total wall ratio为 1.412x, 3090为 0.839x; 两机 epoch peak allocated ratio均为 1.499x. 因此原 `20260724-01` 效率实验中因 3090 GDN不可执行而阻塞的双机 full-epoch门槛现已关闭.

## 2. 根因与上游修复

原 FLA v0.4.0 在 3090 的 train 和 eval 都在 kernel 编译阶段失败:

```text
OutOfResources: shared memory, Required: 147456, Hardware limit: 101376
```

当前容器通过 CUDA runtime 实测的 shared memory 为:

| GPU | 默认每 block | opt-in 每 block |
|---|---:|---:|
| RTX 2080 Ti, sm75 | 49152 B | 65536 B |
| RTX 3090, sm86 | 49152 B | 101376 B |

FLA 上游 commit [`bbdd5051`](https://github.com/fla-org/flash-linear-attention/commit/bbdd5051aea72021a35d7a9dde3c03dc9752ba69) 修改 `chunk_delta_h.py` 的 autotune guard. 旧实现无条件枚举 `num_stages=[2,3,4]` 和 `BV=[32,64]`; 修复后按照设备 shared-memory 能力限制 stages 和 BV. FLA [v0.4.2](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.4.2) 已包含该修复, 因而不会再选择 3090 无法启动的 147456 B 配置.

本轮两个官方 worktree 均保持 clean:

| 候选 | FLA commit | PyTorch | Triton | 安装 kernel SHA256 |
|---|---|---|---|---|
| v0.4.2 | `ca910f88529565b28b6e16465258f2e239a02dc7` | 2.6.0+cu118 | 3.2.0 | `e78c79c5889148fd471e7ec770800da4e58bce2cc40c9c1fcebf86a4dd72a2e2` |
| v0.5.0 | `3a9ce1c83a13994d824dbb3421e2989d330bb38b` | 2.7.1+cu118 | 3.3.1 | `2065d97783a27a3930494462ba2da3a23f859cc33ab649e716829516b87c48ef` |

## 3. 我们修改了什么

FLA 源码没有改动. Zoology 只做了一个模型接线兼容修改: 从三处 `chunk_gated_delta_rule`/`fused_recurrent_gated_delta_rule` 调用删除 `head_first=False`.

FLA v0.5.0 已移除 `head_first` 参数并强制输入布局为 `[B,T,H,...]`. Zoology 原本传入的 q/k/v/g/beta 本来就是该布局; FLA v0.4.x 的默认值也等价于 `head_first=False`. 因此删除该冗余 keyword 不做 transpose, 不改 state update, 不改 dtype, 不改模型参数或超参数. 2080 Ti 上 current v0.4.0 与 v0.4.2 的 kernel output/final state/loss 为 exact equal, full-model logits/loss也为 exact equal, 进一步验证该接线修改没有改变 v0.4.x 语义.

代码分支为 `20260724-gdn-ek4-fla-compatibility`. 正式质量队列使用 zoology commit `57bae2697f386139501aa0c424244f4c142a81b5`; 后续 commit只增加环境审计、warmed epoch、artifact和报告逻辑. Flash-VQG 使用上一轮已验证的 `ec770f33676036432c6514acd1ac05bd2d01f3e8`, 本轮没有修改 Flash-VQG 源码.

## 4. 固定口径

- GDN 为 `gdnxk-h2-ek4-ev4-usegate0`, `d_model=128`, 2层布局中一个 BaseConv和一个 GDN mixer, 2 heads, per-head K/V 256/256, active state capacity 131072, 参数量 1335942.
- Flash 为已优化 `baseline-r16-joint`, active state capacity 131072, 参数量 1160390.
- Seed 124, data seed 123, train `B64/T256/GA4`, eval `B16/T1024`.
- Cache content hash `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- GDN init state hash `bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6`.
- Flash init state hash `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- Batch-order hash `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.
- 全流程 FP32, `GDN_KERNEL_DTYPE=float32`, `TRITON_F32_DEFAULT=ieee`, PyTorch/cuDNN TF32 off, `NVIDIA_TF32_OVERRIDE=0`.
- 2080 Ti 使用物理 GPU1. 3090 宿主机只有一张 GPU, 使用 GPU0.

## 5. 兼容性与等价性

### 5.1. Production shape

两个候选版本在两机共 8 个正式 compatibility 单元全部成功:

| GPU | FLA | eval B16/T1024 | train B64/T256 forward+CE+backward |
|---|---|---|---|
| 2080 Ti | 0.4.2 | PASS | PASS |
| 2080 Ti | 0.5.0 | PASS | PASS |
| 3090 | 0.4.2 | PASS | PASS |
| 3090 | 0.5.0 | PASS | PASS |

### 5.2. 数值门槛

2080 Ti 可运行 legacy v0.4.0, 因而把它作为 reference:

| 比较 | kernel | full-model one-step | 结论 |
|---|---:|---:|---|
| v0.4.0 vs v0.4.2 | output/final/loss exact, worst grad max abs `3.55e-15` | logits/loss exact, worst grad rel L2 `4.07e-7`, param max abs `4.05e-6` | PASS |
| v0.4.0 vs v0.5.0 | output max abs `9.90e-10`, final state `3.73e-8`, loss abs `8.53e-14` | logits max abs `7.15e-7`, loss abs `9.54e-7`, worst grad rel L2 `6.57e-7`, param max abs `5.94e-6` | PASS |
| v0.4.2 vs v0.5.0 | PASS | PASS | PASS |

3090 上 legacy v0.4.0 无法启动, 因而直接比较两个可运行候选:

| 比较 | 主要结果 | 结论 |
|---|---|---|
| v0.4.2 vs v0.5.0 kernel | 全部 tensor 和梯度通过既定阈值 | PASS |
| v0.4.2 vs v0.5.0 full model | loss abs `9.54e-7`, logits max abs `8.96e-6`通过; worst grad rel L2 `1.23e-4`, AdamW后参数 max abs `1.72e-3`未通过 | FAIL, 记为 v0.5.0 风险 |

这项失败不能解释为 v0.4.2 相对 legacy 语义改变. v0.4.2 已在 2080 Ti 直接对 legacy reference 通过, 并在 3090 通过 production shape和正式质量. 失败说明 v0.5.0 的整套 PyTorch/Triton/FLA 环境在 sm86 上没有满足本轮要求的严格 one-step 可替换门槛.

## 6. 版本性能选择

所有 steady-state timing 为 fresh process, warmup 5, active 10, 五次交替顺序 paired repeats. Timing和 memory 分开运行. 下表的变化为 `v0.5.0/v0.4.2 - 1`:

| GPU | 模型 | 阶段 | 时间变化 | v0.5 wins | 95% bootstrap ratio区间 | allocated变化 |
|---|---|---:|---:|---:|---:|---:|
| 2080 Ti | Flash | eval | -4.05% | 5/5 | [0.946, 0.993] | 0% |
| 2080 Ti | Flash | train | -0.58% | 5/5 | [0.993, 0.996] | 0% |
| 2080 Ti | GDN | eval | -9.19% | 5/5 | [0.873, 0.948] | 0% |
| 2080 Ti | GDN | train | -2.62% | 5/5 | [0.952, 0.988] | 0% |
| 3090 | Flash | eval | +1.29% | 1/5 | [0.995, 1.040] | 0% |
| 3090 | Flash | train | -3.63% | 3/5 | [0.905, 1.035] | 0% |
| 3090 | GDN | eval | -7.08% | 4/5 | [0.755, 1.160] | 0% |
| 3090 | GDN | train | **+9.11%** | **0/5** | **[1.027, 1.177]** | 0% |

预定选择规则要求 v0.5.0 在八个 model/GPU/phase 单元均不得回退超过 2%, memory不得恶化超过 5%, 且至少一个单元有稳定收益. 3090 GDN train 单项已经明确违反门槛, 因而即使 v0.5.0 在 2080 Ti 更快, 仍选择 v0.4.2 作为跨 GPU 共同环境.

## 7. 正式 1ep 质量

完整训练均为 704 optimizer steps, 每个 run保存 best/last checkpoint及 SHA256. Smoke和失败/中断测试不写正式 ledger.

| 模型 | GPU | FLA | Overall | 1024x256 | 相对 reference | 结论 |
|---|---|---|---:|---:|---|---|
| GDN | 2080 Ti | 0.4.0 | 0.989359 | 0.916262 | legacy reference | PASS |
| GDN | 2080 Ti | 0.4.2 | 0.989359 | 0.916262 | overall/hard均 0 delta | PASS |
| GDN | 3090 | 0.4.2 | 0.989453 | 0.917094 | 对 2080 Ti +0.009pp/+0.083pp | PASS |
| GDN | 2080 Ti | 0.5.0 | 0.989359 | 0.916262 | overall/hard均 0 delta | PASS |
| GDN | 3090 | 0.5.0 | 0.989442 | 0.917008 | 对 2080 Ti +0.008pp/+0.075pp | PASS |
| Flash | 2080 Ti | 0.4.2 | 0.985319 | 0.918789 | 对上一轮 -0.227pp/-1.442pp | PASS |
| Flash | 3090 | 0.4.2 | 0.988404 | 0.939551 | 对上一轮 +0.168pp/+0.923pp | PASS |

两台 Flash hard accuracy gap为 2.076pp, 小于 4pp. 所有 run均无 OOM、NaN、Inf或 Traceback. 这些质量结果说明 v0.4.2 共同环境没有让 Flash或 GDN效果打折.

正式质量 run的 wall time包含首遇 shape编译和完整诊断, 仅用于 ledger, 不用于版本性能判定. 稳态性能使用第6节的独立 paired runner, 完整 epoch性能使用预编译 fresh-process runner.

## 8. Warmed full epoch 与 cold compile

Warmed runner先在 throwaway model上预编译 train `T64/128/256`和 eval `T64/128/256/512/1024`, 再重新加载 canonical init并计时完整 704-step epoch及4次 validation. 每项为3个 fresh processes的中位数, 执行顺序按 repeat交替.

| GPU | 模型 | Total wall | Train wall, 不含 validation | Validation wall | Epoch peak allocated |
|---|---|---:|---:|---:|---:|
| 2080 Ti | Flash | 394.972 s | 336.860 s | 58.728 s | 3.560 GiB |
| 2080 Ti | GDN | 279.641 s | 246.062 s | 33.591 s | 2.375 GiB |
| 2080 Ti | Flash/GDN | **1.412x, PASS** | 1.369x | 1.748x | **1.499x, PASS** |
| 3090 | Flash | 332.062 s | 289.851 s | 42.553 s | 3.560 GiB |
| 3090 | GDN | 395.865 s | 358.768 s | 37.097 s | 2.375 GiB |
| 3090 | Flash/GDN | **0.839x, PASS** | 0.808x | 1.147x | **1.499x, PASS** |

3090 上 Flash在完整 epoch中快于 GDN, 与 production `T256` train p50为 0.958x方向一致. Epoch比值更低是因为实际训练数据还包含较短的 `T64/T128` batches, 不能把单一 `T256` microbenchmark直接外推为全 epoch.

Cold runner为每个 GPU/variant/phase建立独立空 `TRITON_CACHE_DIR`. 下表计时从模型和固定 batch已放到 GPU之后开始, 包含首次 kernel编译/autotune和一次实际 forward或 forward+backward, 不包含 Python process、数据加载或模型构建:

| GPU | FLA | Eval首次执行 | Train首次 forward+CE+backward | 结果 |
|---|---|---:|---:|---|
| 2080 Ti | 0.4.2 | 129.311 s | 305.872 s | PASS |
| 2080 Ti | 0.5.0 | 92.572 s | 256.645 s | PASS |
| 3090 | 0.4.2 | 179.798 s | 464.401 s | PASS |
| 3090 | 0.5.0 | 136.089 s | 420.379 s | PASS |

v0.5.0 的 cold compile更快, 但 cold cost不是稳态共同环境选择门槛. 实际长期训练必须复用持久 Triton cache或在计时前预编译; 不能把数分钟的首次编译混入 steady-state p50. 作为历史背景, legacy v0.4.0在 2080 Ti 的同类 cold eval/train曾为 276.757/553.830 s, 在 3090则直接因 shared-memory配置失败.

2080 Ti v0.4.2 两个 cold JSON在一次仅影响 provenance字段的 helper缩进错误修复前启动, 因而 raw `fla_source_commit` 为 null. 它们仍记录 FLA 0.4.2、source root和安装 kernel SHA256; environment snapshot把该 clean worktree和相同 hash解析到官方 commit `ca910f88529565b28b6e16465258f2e239a02dc7`. 计算输出没有修改或重跑.

## 9. 环境与可复现性

候选环境均从现有环境克隆, 没有原位升级 `/home/lyj/miniconda3/envs/flash-vqg`. 最终 v0.4.2 两机补齐与模型无关的 `cffi==2.1.0`; 3090 侧已有 `xprof/gcsfs`, 因此把 `fsspec/gcsfs` 从 2025.10.0降到 datasets允许的 2025.9.0. 这次 housekeeping发生在正式质量矩阵之后, 相关包不在模型/FLA import链上, PyTorch/FLA/Triton和安装 kernel hash均未改变; warmed epoch和 cold compile使用整理后的最终环境. 普通和 `PYTHONNOUSERSITE=1` 两种 `pip check`均通过. 完整 package list、Conda list、GPU、源码状态和 kernel SHA256保存在 environment snapshot.

Wrapper和 dtype/shape guard测试分别在 current v0.4.0、v0.4.2、v0.5.0 三个环境执行, 每个环境均为 18 passed, 合计 54 passed. 使用 `CUDA_VISIBLE_DEVICES=''` 隐藏 GPU时, FLA/Triton在 import阶段按设计无法找到 active driver; 该次诊断失败不属于代码回归, 随后在目标 GPU环境的三次正式测试均通过.

正式配置、cache/init/order hash、checkpoint path/hash、命令和 raw source路径均由 artifact及 source manifest记录. 3090侧 201个 JSON/CSV/log轻量文件镜像后逐路径比较 SHA256, 201/201一致. 远端大型 checkpoint、equivalence capture、allocator snapshot和 empty Triton cache按规范留在 source machine, 不提交 Git; source-only tensor/snapshot另由 `large-raw-manifest.csv`记录大小和 SHA256.

## 10. Artifact 与入口

正式 artifact 位于 `docs/artifacts/20260724-02-gdn-ek4-fla-compatibility/`. 主要文件包括:

- `compatibility.csv`, `cold-compile.csv`.
- `equivalence.csv`.
- `benchmark-runs.csv`, `version-comparison.csv`, `model-comparison.csv`.
- `quality-1ep.csv`, `quality-gates.csv`.
- `warmed-epoch.csv`, `warmed-epoch-ratios.csv`.
- `environment-summary.csv`, `candidate-events.csv`, `source-manifest.csv`, `large-raw-manifest.csv`, `mirror-verification.csv`, `metadata.json`, `README.md`.

Runner位于 `zoology/experiments/flash_vqg/scripts/20260724-02-gdn-ek4-fla-compatibility/`. 环境锁见同目录 `environment-lock.md`.
