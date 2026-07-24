# 20260724-02-gdn-ek4-fla-compatibility

本 artifact 记录 GDN `gdnxk-h2-ek4-ev4-usegate0` 的 FLA v0.4.2/v0.5.0 双 GPU兼容性、数值等价、稳态 timing/memory、正式 1ep质量、预编译完整 epoch和空 cache冷编译结果.

最终共同环境为 FLA v0.4.2 + PyTorch 2.6.0+cu118 + Triton 3.2.0. v0.5.0 虽然兼容两张 GPU, 但因 3090 GDN train 稳态回退 9.11%和 sm86 full-model one-step严格门槛失败而拒绝. 选中环境的双机 Flash/GDN core time、allocated memory和 warmed full-epoch均通过 `<=2x`.

主要文件:

- `compatibility.csv`: production train/eval及 cold run兼容性.
- `cold-compile.csv`: 独立空 `TRITON_CACHE_DIR` 的首次执行成本.
- `equivalence.csv`: kernel和 full-model tensor级误差.
- `benchmark-runs.csv`: 五重复 timing及独立 memory原始摘要.
- `version-comparison.csv`: v0.5.0/v0.4.2 paired版本比值.
- `model-comparison.csv`: 同环境 Flash/GDN core比值.
- `quality-1ep.csv`, `quality-gates.csv`: 正式质量结果和门槛.
- `warmed-epoch.csv`, `warmed-epoch-ratios.csv`: 三重复预编译完整 epoch.
- `environment-summary.csv`: 依赖、GPU、source commit和 kernel hash.
- `candidate-events.csv`: 接受/拒绝事件及失败候选.
- `source-manifest.csv`, `large-raw-manifest.csv`, `mirror-verification.csv`, `metadata.json`: 轻量镜像、大型 source-only raw的 SHA256和最终选择状态.

Raw 输出位于 `zoology/experiments/flash_vqg/scripts/20260724-02-gdn-ek4-fla-compatibility/outputs/`. 远端大型 checkpoint、equivalence `.pt` capture、allocator snapshot和 empty Triton cache保留在 source machine, 不提交 Git. Checkpoint路径和 SHA256在 `quality-1ep.csv`; 未镜像大型 raw的边界在正式报告中说明.

2080 Ti v0.4.2 cold eval/train在一次仅影响 metadata 的 helper缩进错误修复前启动, 因而 raw JSON中的 `fla_source_commit` 为 null. 其 FLA version、安装 kernel SHA256、source root均存在; `environment-summary.csv`和 clean worktree把相同 kernel hash解析到官方 commit `ca910f88529565b28b6e16465258f2e239a02dc7`. 计算结果未修改或重跑.
