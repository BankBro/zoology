# 20260622-04 Flash-VQG read candidate churn probe report

## 结论

这轮三条 3090 长训都正常完成, 每条都是 4 epoch, best 与 final 完全一致, 日志错误匹配数为 0. 结果没有发现 late drift, 也没有复现 `cb128-r8 readk4 s125` collapse.

但是这轮没有真正回答"同一个样本 token 在 step 130, 203, 352, 448 之间候选是否频繁换"这个问题. 当前 churn probe 的实现是把 valid batch, layer 和 tensor shape 作为 key, 在 runtime 中比较上一轮保存的 `top_idx`; key 不含样本 ID, 不含显式 validation step, 也没有把固定样本的候选 trace 落盘. 因此三条 run 的 `churn=0`, `retention=1`, `top1_flip=0` 只能说明 probe 路径和指标透传生效, 以及相邻 probe 状态下 topk candidate 没有变化; 不能把它当作"跨训练过程候选完全稳定"的证据.

## 实验矩阵

| target | machine | read_topk | seed | start CST | end CST | status |
|---|---|---:|---:|---|---|---|
| `cb256r8-readk4-s123-churn` | 3090 | 4 | 123 | 2026-06-22 18:36:18 | 2026-06-22 21:25:54 | completed |
| `cb128r8-readk4-s125-churn` | 3090 | 4 | 125 | 2026-06-22 18:36:18 | 2026-06-22 21:59:45 | completed |
| `cb256r8-readk2-s123-churn` | 3090 | 2 | 123 | 2026-06-22 18:36:18 | 2026-06-22 23:07:56 | completed |

运行环境来自 SwanLab metadata: zoology branch `flash-vqg`, commit `1dcd70acc631d395aadb6874d58d952f1b6ddbab`, Python `3.12.11`, GPU `NVIDIA GeForce RTX 3090`, driver `580.159.03`, CUDA metadata `11.8`.

## Final 与 Best

| target | final hard 1024x256 | best hard 1024x256 | best-final gap | final valid acc | final loss |
|---|---:|---:|---:|---:|---:|
| `cb256r8-readk4-s123-churn` | 0.987117 | 0.987117 | 0.000000 | 0.997040 | 0.045187 |
| `cb256r8-readk2-s123-churn` | 0.987902 | 0.987902 | 0.000000 | 0.996574 | 0.039667 |
| `cb128r8-readk4-s125-churn` | 0.964484 | 0.964484 | 0.000000 | 0.992958 | 0.074539 |

matched `cb256-r8 s123` 中, `readk4 - readk2` 的 final hard delta 为 `-0.000785`, valid accuracy delta 为 `+0.000466`. 这个差异很小, 不能支持 `cb256-r8 readk4` 在这个 seed 上有明显收益; 更合理的读法是两者都在高 basin.

## Read-Side 指标

| target | read_topk | churn | retention | top1 flip | margin mean | entropy mean | selected mass mean | effective readk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cb256r8-readk4-s123-churn` | 4 | 0.000000 | 1.000000 | 0.000000 | 0.884412 | 1.444891 | 0.223174 | 4 |
| `cb256r8-readk2-s123-churn` | 2 | 0.000000 | 1.000000 | 0.000000 | 0.724912 | 1.118817 | 0.277345 | 2 |
| `cb128r8-readk4-s125-churn` | 4 | 0.000000 | 1.000000 | 0.000000 | 0.737360 | 0.995695 | 0.325758 | 4 |

`cb256-r8 readk4` 的 margin mean 高于 readk2, 但 entropy 也更高, selected mass 反而更低. 这个方向和直觉一致: 扩大 read topk 会扩大候选集合, 但把概率质量摊到更多候选上. 当前数据不支持用 churn 来解释 readk4 的正负效果.

## Write/State 指标

| target | write strength mean | write strength max | zeta mean | zeta max | lambda mean | inject ratio | m_norm max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cb256r8-readk4-s123-churn` | 0.003097 | 0.023849 | 0.012388 | 0.043954 | 0.349344 | 0.274138 | 1.629788 |
| `cb256r8-readk2-s123-churn` | 0.014824 | 0.105205 | 0.059296 | 0.244811 | 0.254927 | 0.152514 | 1.767193 |
| `cb128r8-readk4-s125-churn` | 0.008900 | 0.120171 | 0.035599 | 0.265769 | 0.347505 | 0.086197 | 9.533614 |

`cb128-r8 readk4 s125` 的 `m_norm_max=9.533614`, 明显高于两个 `cb256-r8` run. 它这次 final 很高, 但 state norm 仍然偏红线, 与"cb128-r8/readk4 是边界配置"的历史判断一致.

## 对研究路线的影响

本轮完成了 telemetry 接入的长训验证和三条 run 的轻量归档, 但没有完成"固定样本跨 step candidate trace". 因此不应该把 `churn=0` 作为根因结论. 下一步应该改成显式 trace:

- 在 validation probe 中记录样本身份或固定样本 index, `epoch_idx/global_step`, layer, head, query position 和 topk candidate ids.
- 对同一固定样本位置输出 `step -> topk ids` 的 artifact, 再离线计算 retention, churn, top1 flip 和 candidate rank changes.
- 最小复跑只需要短 probe: `cb256-r8 readk2/readk4 s123` 与 `cb128-r8 readk4 s125`, 在 step `0/130/203/352/448/705` 附近保存 trace, 不需要先跑 full grid.

## Artifact

正式 artifact 位于 `docs/artifacts/20260622-04-flash-vqg-read-candidate-churn-probe/`. 其中 `final.csv`, `final_best_metrics.csv`, `source_manifest.csv`, `metadata.json`, `comparison.json`, `raw_summary.json`, `summary.json` 已归档. checkpoint `.pt` 没有复制进 git artifact, 只在 3090 原路径读取 metrics.
