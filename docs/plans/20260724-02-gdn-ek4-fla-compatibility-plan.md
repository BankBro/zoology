# GDN ek4-ev4 FLA 3090 兼容性计划

实验 ID: `20260724-02-gdn-ek4-fla-compatibility`.

## 目标

在不修改 `gdnxk-h2-ek4-ev4-usegate0` 数学语义、模型配置和 FP32 口径的前提下, 验证官方 FLA v0.4.2 与 v0.5.0 是否修复 RTX 3090 上的 shared-memory kernel 启动失败, 并选择可供 Flash-VQG 与 GDN 公平对比的共同环境.

## 固定口径

- seed 124, data seed 123, canonical cache/init/batch order.
- train `B64/T256`, gradient accumulation 4; eval `B16/T1024`.
- outer model 和 GDN kernel 均使用 FP32, `TRITON_F32_DEFAULT=ieee`, TF32 off.
- FLA v0.4.2 保持 PyTorch 2.6.0+cu118 与 Triton 3.2.0.
- FLA v0.5.0 使用 PyTorch 2.7.1+cu118 与 Triton 3.3.1.
- 现有 `flash-vqg` 环境不原位升级.

## 验收与选择

1. 两个候选版本均执行 kernel forward/backward/final-state 和 full-model one-step 等价测试.
2. 两机均执行 production train/eval shape, 五组 fresh-process 稳态 timing 和独立 memory 测量.
3. GDN 执行 `current040@2080ti`, `v042@2080ti/3090`, `v050@2080ti/3090` 的完整 1ep 质量回归.
4. v0.5.0 只有在全部回归通过、所有共同 benchmark 单元不稳定回退超过 2%、峰值 allocated 不恶化超过 5%, 且至少一个单元存在可复现收益时才胜出; 否则选择 v0.4.2.
5. 最终环境再对 Flash-VQG 做双 GPU 兼容性、性能和 1ep 回归.

完整执行方案以本会话确认的 plan 为准. 若使用子代理, 只允许根代理一层委派, 子代理不得继续委派.
