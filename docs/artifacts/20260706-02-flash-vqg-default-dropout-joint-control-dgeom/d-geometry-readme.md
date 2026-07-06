# D-geometry 诊断说明

这些文件只来自 diagnostic targets, 不参与 formal pass/fail 判定. 诊断对象是 gd_residual_v1 中真正用于 M_state write 的归一化方向 `D_pack = normalize((K - codebook) @ addr_proj)`. 统计项按 layer/head/code 聚合 pairwise cosine, effective rank, condition number 和 update_norm.

- `d-geometry-summary.csv`: 原始 trace JSONL 的扁平表.
- `d-geometry-by-code-head.csv`: 按 variant/machine/step/layer/head/code 聚合.
- `d-geometry-cross-machine.csv`: 2080ti vs 3090 同组指标差异.
- `d-geometry-hotspot-summary.csv`: 高相关, 高 update_norm, 高 condition number, 低 effective rank hotspot.
