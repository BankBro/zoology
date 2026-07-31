# Residual Value Bottleneck MQAR 实验计划

## 1. 实验登记

- Experiment ID: `20260731-02-residual-value-bottleneck-mqar`.
- 状态: `ready`.
- Zoology base: `flash-vqg@3e51c62de13dea73034907bb020e16fe54f1c739`.
- Flash-VQG source: `20260731-161252-residual-value-bottleneck@0ddaa2d3dd2857778a3fbacda894516a9804a675`.
- 执行机器: RTX 3090 GPU0.
- 精度: AMP BF16, FP32 master weights和optimizer state.

本实验在A1加S1 exact质量路径上比较U64, U32和U16 residual value维度. Coarse memory始终保持64维. 本实验不使用K2或W2, 不引入Flash加GDN混合模型.

## 2. 固定合同

三组共同使用`baseline-r16-joint`, block64, local2, rank16, write top-k4, read top-k16, `post_phase1` remat, grouped Triton builder, S1 exact selected backward和`fp32_boundary`.

| Variant | Residual value | 新增投影 |
|---|---:|---|
| `u64-a1-s1` | 64 | 无 |
| `u32-a1-s1` | 32 | 每层每头`[64,32]` |
| `u16-a1-s1` | 16 | 每层每头`[64,16]` |

U32和U16从同一canonical init继承全部共有参数, 仅新增列正交投影参数. 投影使用固定local RNG初始化, 不消耗共有参数的全局初始化随机流.

## 3. 阶段与门禁

### 3.1. Q0筛选

三组先执行3-update smoke, 再执行seed123一epoch训练和5个locked MQAR case评估.

- 标准`1024x256`相对U64下降不得超过10%.
- 四个外推slice宏平均相对U64下降不得超过10%.
- Loss, gradient和checkpoint必须finite.
- Grouped与selected Triton fallback必须为0.
- 三组FLA fused-gate backward config必须一致.

### 3.2. 正式矩阵

Q0通过且资源实验保留的候选进入seed123/124/125四epoch正式矩阵. 最终门禁为:

- 三seed标准均值和外推宏平均相对U64下降不超过5%.
- 任一seed标准或外推宏平均下降不超过10%.
- 任一seed, 任一外推slice下降不超过15%.

若U32和U16均通过, 优先选择资源正式吞吐更高者. 速度差低于3%或CI明显重叠时选择U32.

## 4. 执行与证据

正式运行前锁定两个仓库commit, canonical Python/FLA, CUDA/NVML, cache内容hash, canonical init, derived init, 参数量和配置差异. Q0失败只停止对应候选的正式矩阵. 所有输出保留独立run tag, 结束后生成report和artifact并更新项目日志.
