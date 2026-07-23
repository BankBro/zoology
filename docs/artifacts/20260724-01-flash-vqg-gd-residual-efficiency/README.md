# 20260724-01-flash-vqg-gd-residual-efficiency

Flash-VQG gd_residual_v1 baseline-r16-joint GPU memory and runtime audit. All hard timing rows use fixed canonical inputs, FP32/IEEE matmul policy, warmup >= 5, active >= 10, and fresh-process repeats. Formal quality runs cover seeds 124/125 on both GPUs; the trajectory artifact records exact 32/128/512-step comparisons. Formal timing and fresh empty-cache cold-start costs are reported separately from the symmetric core hard ratios.

The actual two-layer model contains one active Flash-VQG GD-residual layer and one BaseConv layer; tensor-lifetime estimates therefore use one GD layer.
