# 20260622-04 Flash-VQG read candidate churn probe artifact

## 内容

这个目录保存 3090 上三条 read candidate churn probe 长训的轻量收尾产物. checkpoint `.pt`, swanlog raw 和完整大日志没有复制到 git artifact; checkpoint 只在 3090 原地读取 best/final metrics.

文件:

- `final.csv`: 每条 run 的 final checkpoint 指标.
- `final_best_metrics.csv`: 每条 run 的 best 与 final checkpoint 指标.
- `source_manifest.csv`: manifest, launch config, train config, swanlab metadata 和日志 source 的路径, 大小, sha256.
- `metadata.json`: artifact 级 metadata, 运行环境, run 清单和限制说明.
- `comparison.json`: `cb256-r8 readk4 - readk2` 的 matched 差值.
- `raw_summary.json`: 轻量原始摘要, 包含 manifest, train config, swanlab metadata, checkpoint metrics 和日志尾部.
- `summary.json`: 收集脚本摘要.

## 运行范围

三条 run 均在 `mclab-3090` 的 `Flash-VQG-tun` 容器中完成, zoology commit 为 `1dcd70acc631d395aadb6874d58d952f1b6ddbab`, branch 为 `flash-vqg`, Python `3.12.11`, CUDA runtime metadata `11.8`, GPU driver `580.159.03`.

| target | read_topk | seed | start CST | end CST | final hard 1024x256 | best hard 1024x256 |
|---|---:|---:|---|---|---:|---:|
| `cb256r8-readk4-s123-churn` | 4 | 123 | 2026-06-22 18:36:18 | 2026-06-22 21:25:54 | 0.987117 | 0.987117 |
| `cb128r8-readk4-s125-churn` | 4 | 125 | 2026-06-22 18:36:18 | 2026-06-22 21:59:45 | 0.964484 | 0.964484 |
| `cb256r8-readk2-s123-churn` | 2 | 123 | 2026-06-22 18:36:18 | 2026-06-22 23:07:56 | 0.987902 | 0.987902 |

## 主要指标

| target | churn | retention | top1 flip | margin mean | entropy mean | selected mass mean | m_norm max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cb256r8-readk4-s123-churn` | 0.000000 | 1.000000 | 0.000000 | 0.884412 | 1.444891 | 0.223174 | 1.629788 |
| `cb256r8-readk2-s123-churn` | 0.000000 | 1.000000 | 0.000000 | 0.724912 | 1.118817 | 0.277345 | 1.767193 |
| `cb128r8-readk4-s125-churn` | 0.000000 | 1.000000 | 0.000000 | 0.737360 | 0.995695 | 0.325758 | 9.533614 |

## 注意

这轮结果不能直接证明"同一个样本 token 跨训练 step 的候选稳定". 当前 probe 的 key 只包含 valid batch, layer 和 tensor shape, 不包含样本 ID 或显式 step trace; 因此 `churn=0` 应解释为本轮 instrumentation/runtime 路径生效, 且相邻 probe 状态下 candidate set 没有变化, 而不是完整的跨 step 稳定性结论.

本轮更可靠的结论是:

- 三条 run 的 best/final 一致, 没有 late drift.
- `cb256-r8 readk4` 相比 matched `readk2` 的 final hard 低 `0.000785`, 不是一次明显 rescue.
- `readk4` 的 entropy 更高, selected mass 更低, 说明扩大 topk 会摊薄 selected candidate mass; 这和 fixed readk4 不适合直接作为全局默认一致.
- `cb128-r8 readk4 s125` 这次没有 collapse, final hard 为 `0.964484`, 但 `m_norm_max=9.533614` 偏高, 仍应保留为边界配置.
