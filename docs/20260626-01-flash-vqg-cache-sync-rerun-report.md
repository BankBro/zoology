# 20260626-01-flash-vqg-cache-sync-rerun 报告

## 目的

本次收尾汇总 3090 在使用 2080ti canonical cache 后的 r1-r4 1-epoch screen, 并与 2080ti canonical r1-r4 对照, 以判断先前跨机器差异中有多少来自 cache 内容不一致, 以及 cache 一致后还剩多少训练稳定性问题。

## 数据一致性

13 个 canonical cache 已在 2080ti 与 3090 上做 content-level hash 验证, 结果为 `13/13` match=true. 3090 r1-r4 的 8 个日志均实际加载同一组 13 个 canonical cache, 未观察到 cache 重生成或旧 cache 混入。

## 3090 Canonical Run 结果

| target | wall min | mqar 1024x256 | valid accuracy | cache match |
|---|---:|---:|---:|---|
| default-s123-r1 | 73.9 | 0.865 | 0.975 | true |
| default-s123-r2 | 73.3 | 0.554 | 0.908 | true |
| default-s123-r3 | 72.8 | 0.927 | 0.985 | true |
| default-s123-r4 | 72.8 | 0.756 | 0.954 | true |
| default-s124-r1 | 53.0 | 0.71 | 0.948 | true |
| default-s124-r2 | 54.1 | 0.871 | 0.977 | true |
| default-s124-r3 | 53.9 | 0.864 | 0.976 | true |
| default-s124-r4 | 47.3 | 0.816 | 0.967 | true |

## Canonical Cross-Machine Repeat Summary

| machine | seed | mean | min | max | gap | std | r1 | r2 | r3 | r4 | stable gap<=0.05 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2080ti | 123 | 0.941750 | 0.935000 | 0.950000 | 0.015000 | 0.006057 | 0.935 | 0.937 | 0.945 | 0.95 | true |
| 2080ti | 124 | 0.805250 | 0.716000 | 0.871000 | 0.155000 | 0.058768 | 0.842 | 0.716 | 0.792 | 0.871 | false |
| 3090 | 123 | 0.775500 | 0.554000 | 0.927000 | 0.373000 | 0.141779 | 0.865 | 0.554 | 0.927 | 0.756 | false |
| 3090 | 124 | 0.815250 | 0.710000 | 0.871000 | 0.161000 | 0.064348 | 0.71 | 0.871 | 0.864 | 0.816 | false |

## Seed 级解读

- seed 123: 2080ti mean=0.941750, gap=0.015000; 3090 mean=0.775500, gap=0.373000; 3090-2080ti mean delta=-0.166250.
- seed 124: 2080ti mean=0.805250, gap=0.155000; 3090 mean=0.815250, gap=0.161000; 3090-2080ti mean delta=0.010000.

## 初步结论

cache 内容不一致已被排除为当前 canonical 对照的直接原因. 在 canonical cache 下, `s124` 两台机器的均值基本对齐, 且 gap 接近; `s123` 在 2080ti 上稳定, 但在 3090 上仍存在明显 repeat 波动和均值下移, 主要低点来自 `default-s123-r2` 和 `default-s123-r4`.

因此下一步应聚焦 3090 `s123` 的训练数值路径, 包括 TF32 flags, deterministic 配置, 初始参数 hash, step0/step1 loss/logits/parameter hash, 以及 Flash-VQG/GD residual routing 对早期浮点扰动的放大。

## Artifact

详见 `docs/artifacts/20260626-01-flash-vqg-cache-sync-rerun/`.
