# 20260627-03 Flash-VQG First-Divergence Probe

本 artifact 汇总 `20260627-03` 的短 debug probe, 不写 official ledger.

文件:

- `first-mismatch-summary.csv`: 2080ti vs 3090 的首个分叉点摘要.
- `run-summary.csv`: 每个短 probe 的 cache/init/batch/forward/step 摘要.
- `metadata.json`: 代码版本, 前置 hash 和结论.
- `source-manifest.csv`: raw evidence 路径与镜像状态.

核心结论:

- cache, init checkpoint, batch order, first input 和 first target 均一致.
- baseline 首个 forward 分叉是 `backbone.layers.0.dropout1`.
- `strict-fp32` 不改变这个首个分叉点.
- `shadow-read` 不改变训练输出, 只增加 dense shadow 指标; 第一批 shadow dense 指标为 0, 因为第一块远程 residual state 尚为零.
- `no-dropout` 后第一层完全一致, 首个分叉推迟到 `backbone.layers.1.sequence_mixer.mixer`.

解释:

- 当前问题不能再简单说成“输入或初始化不一致”.
- baseline 的第一处实际分叉是 CUDA dropout RNG 跨 GPU 不一致.
- 去掉 dropout 后, 剩余第一处数值分叉在第 1 层 Flash-VQG mixer, 不是数据/cache/init.
