# Flash-VQG GD residual efficiency runner

本目录提供 `20260724-01-flash-vqg-gd-residual-efficiency` 的 Phase 0 公平基线和 profiler 入口.

## 1. 口径

- Flash 使用最新 `baseline-r16-joint` resolved config和 seed124 canonical init.
- GDN 使用冻结的 `gdnxk-h2-ek4-ev4-usegate0`, 自身 seed124确定性初始化, 不加载 Flash init.
- 两模型复用同一个 canonical MQAR data config和固定 `B64/T256`, `B16/T1024` batch.
- `core` 关闭 Flash layer metrics, 不做无用 `argmax`或逐标量 D2H.
- `formal` 保留当前 Flash metrics, train/eval `argmax`, train逐 microbatch loss D2H和 optimizer-boundary metrics收集.
- Timing, allocator memory和 torch profiler必须分开执行.
- Runner在导入 FLA前固定 `TRITON_F32_DEFAULT=ieee`, 防止 Ampere上的 Triton dot隐式使用 TF32.

## 2. 硬预检

首次运行先在主工作区生成一次 GDN自己的 canonical init. 不在两机分别生成, 因不同CPU上的确定性初始化state hash可能不同:

```bash
python zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  make-gdn-init
```

将生成的 `.pt`和同名 `.json`作为输入证据镜像到目标机器相同相对路径, 并校验文件 sha256及内部 model-state hash. 该checkpoint只包含 GDN state, 不包含 Flash参数.

2080ti 当前容器使用物理 GPU1:

```bash
CUDA_VISIBLE_DEVICES=1 python zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  preflight \
  --output zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/2080ti/preflight.json
```

3090 主机实际只有一张 GPU, 宿主机和容器索引均为 GPU0. 该硬件事实必须在报告中保留:

```bash
CUDA_VISIBLE_DEVICES=0 python zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  preflight \
  --output zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/3090/preflight.json
```

Preflight会重新验证 cache content hash, Flash init state hash, epoch-0 batch-order hash, 固定batch hash, 参数量和capacity.

## 3. 稳态 timing

每个 model/phase/metrics组合用 fresh process独立运行 3 次. 示例:

```bash
CUDA_VISIBLE_DEVICES=1 python zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  run --model flash --phase train --metrics-mode core --run-kind timing \
  --warmup 5 --active 10 --repeat-id 1 \
  --output-dir zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/2080ti/baseline/flash-train-core-r1
```

训练记录单位是一个 GA4 optimizer step. Eval记录单位是一个 `B16/T1024` full batch. `records.csv`同时包含 wall和CUDA分段, `summary.csv`包含 p50/p90.

## 4. Memory与profiler

Memory必须使用 fresh process:

```bash
CUDA_VISIBLE_DEVICES=1 python zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  run --model flash --phase train --metrics-mode formal --run-kind memory \
  --warmup 5 --active 1 --repeat-id 1 \
  --output-dir zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/2080ti/baseline/flash-train-formal-memory-r1
```

Profiler只用于算子和内存归因, 其时间不能写入最终性能表:

```bash
CUDA_VISIBLE_DEVICES=1 python zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  run --model flash --phase train --metrics-mode formal --run-kind profile \
  --warmup 5 --active 1 --repeat-id 1 \
  --output-dir zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/2080ti/baseline/flash-train-formal-profile-r1
```

Profile输出包括 Chrome trace, memory tables, allocator snapshot和 autograd saved-tensor账本. 原始 outputs默认不提交, 收尾时只把轻量可审计摘要整理到正式 artifact.

## 5. 等价性和轨迹

最终 backend 的 full-model forward/backward/one-step 对照:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_F32_DEFAULT=ieee python \
  zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  equivalence \
  --output zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/2080ti/equivalence/reference-vs-triton-deterministic-v2.json
```

32/128/512 optimizer-step轨迹使用同一 canonical init和实际 epoch-0 batch order:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_F32_DEFAULT=ieee python \
  zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  trajectory --checkpoints 32 128 512 \
  --output zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/3090/trajectory/reference-vs-triton-32-128-512-v1.json
```

## 6. Warmed full epoch

`epoch`先在 throwaway model上预编译 train `T64/128/256`和 eval `T64/128/256/512/1024`, 然后重新构建 canonical init并开始计时. 每个 repeat必须是独立 Python process:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_F32_DEFAULT=ieee python \
  zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  epoch --model flash --repeat-id 1 --precompile \
  --flash-grouped-chunk-backend triton \
  --flash-selected-read-backend triton_remat \
  --output zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/2080ti/epoch-gpu1/flash-r1.json
```

GDN使用相同入口, 只把 `--model`改为 `gdn`并删除两个 Flash backend参数. Hard ratio使用 3个独立 repeat的中位数.

## 7. 正式质量队列

正式 seed124/125 paired 1ep由队列脚本执行, 脚本会做 GPU, cache, init, dtype和 resolved config预检, 并生成 formal ledger:

```bash
MACHINE_NAME=2080ti GPU=1 \
EFFICIENCY_FORMAL_TIMESTAMP=20260724T0558-v3 \
bash zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/run_formal_queue.sh
```

3090机器实际只有 GPU0, 因此该机使用 `MACHINE_NAME=3090 GPU=0`. 完成后用 `summarize_formal_checkpoints.py`保存 checkpoint和 model-state hash及最终指标.

## 8. 冻结 GDN兼容性和 artifact汇总

在无法执行 GDN的机器上, 使用专用命令保存原始异常和冻结策略, 不修改第三方 GDN kernel:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_F32_DEFAULT=ieee python \
  zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/efficiency_benchmark.py \
  gdn-compatibility --phase eval \
  --output zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/outputs/3090/gdn-compatibility/eval.json
```

镜像完多机轻量证据后, 在主工作区生成正式 CSV/JSON:

```bash
python zoology/experiments/flash_vqg/scripts/20260724-01-flash-vqg-gd-residual-efficiency/collect_efficiency_artifacts.py
```

正式 artifact位于 `docs/artifacts/20260724-01-flash-vqg-gd-residual-efficiency/`. `source-manifest.csv`记录 raw source path, mirror path, bytes和 SHA256.
