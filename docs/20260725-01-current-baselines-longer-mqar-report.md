# 当前 Flash-VQG 与 GDN 基线 Longer-MQAR 报告

实验 ID: `20260725-01-current-baselines-longer-mqar`.

## 1. 结论

本轮完整重训了当前 Flash-VQG `baseline-r16-joint` 和 GDN `gdnxk-h2-ek4-ev4-usegate0`, 每个模型使用 training seeds `123/124/125`, 固定 data seed `123` 和各自 canonical seed124 init. 6/6 正式训练均到达 epoch4, 随后的 Longer-MQAR formal eval、dataset hash 和 repro 审计全部通过.

预注册主结果使用 `last.pt`. 在训练长度端点 `1024x256`, Flash 三 seed 均值为 `0.953406`, GDN 为 `0.968518`, paired mean delta 为 `-0.015112`, 因而不支持 Flash 领先. 但在四个真正的长度外推 slice 中:

- `2048x512`: Flash `0.704091` vs GDN `0.476111`, paired delta `+0.227980`, 3/3 seeds 为正, `稳健领先`.
- `4096x1024`: Flash `0.325602` vs GDN `0.072722`, paired delta `+0.252880`, 3/3 seeds 为正, `稳健领先`.
- `8190x512`: Flash `0.556421` vs GDN `0.293578`, paired delta `+0.262842`, 2/3 seeds 为正, `混合领先`.
- `8190x2047`: Flash `0.097774` vs GDN `0.003378`, paired delta `+0.094396`, 3/3 seeds 为正, `稳健领先`.

因此, 当前证据支持 Flash-VQG 在长度外推上明显优于同 active-state-capacity 的 GDN, 但主结果不是所有 seed、所有 slice 一致领先. 主要不稳定源是 Flash seed124. 它的 epoch4 Longer-MQAR 明显低于 seeds 123/125; epoch3 `best.pt` 可以部分恢复, 但这不能用来替换预注册的 `last.pt` 主结论.

`best.pt` 敏感性结果在全部四个外推 slice 都达到 3/3 seeds `稳健领先`, 说明结论不依赖 Longer-MQAR 事后选点, 但 checkpoint 时机确实影响 Flash seed124 的外推幅度. 本轮不自动替换当前综合 baseline, 下一步应优先解释 Flash seed124 的 epoch3->epoch4 外推退化, 并考虑增加独立 training seeds.

## 2. 正式训练结果

两模型均为 4 epochs, 每 epoch 4 次 validation, `B64/GA4`, effective batch `256`, early stopping 关闭. 下表为 `last.pt` 的 epoch4 常规 validation:

| 模型 | Seed | Overall accuracy | `1024x256` | Wall time | Best epoch |
|---|---:|---:|---:|---:|---:|
| Flash | 123 | 0.995230 | 0.974562 | 26.45 min | 4 |
| Flash | 124 | 0.986959 | 0.912543 | 26.42 min | 3 |
| Flash | 125 | 0.994573 | 0.970098 | 26.41 min | 4 |
| GDN | 123 | 0.995790 | 0.967371 | 18.87 min | 4 |
| GDN | 124 | 0.995667 | 0.966355 | 18.78 min | 4 |
| GDN | 125 | 0.996248 | 0.971063 | 18.74 min | 4 |

Flash 的三 seed overall/hard 均值为 `0.992254/0.952401`; GDN 为 `0.995902/0.968263`. GDN 的训练端点方差更小. Flash seed124 在常规 validation 上已经落后于另两个 Flash seeds, 这与 Longer-MQAR 的外推退化方向一致.

6 条训练合计 wall time 为 `8140.06 s`. 这些 wall time 包含完整训练和 validation, 用于审计而非新的效率排名.

## 3. Longer-MQAR 主结果

### 3.1. `last.pt`

| Slice | Flash mean ± population std | GDN mean ± population std | Flash-GDN paired delta | Positive seeds | 分类 |
|---|---:|---:|---:|---:|---|
| `1024x256` | 0.953406 ± 0.027658 | 0.968518 ± 0.001889 | -0.015112 | 2/3 | 不支持 Flash 领先 |
| `2048x512` | 0.704091 ± 0.161179 | 0.476111 ± 0.004779 | +0.227980 | 3/3 | 稳健领先 |
| `4096x1024` | 0.325602 ± 0.165438 | 0.072722 ± 0.002686 | +0.252880 | 3/3 | 稳健领先 |
| `8190x512` | 0.556421 ± 0.200708 | 0.293578 ± 0.001806 | +0.262842 | 2/3 | 混合领先 |
| `8190x2047` | 0.097774 ± 0.065491 | 0.003378 ± 0.000215 | +0.094396 | 3/3 | 稳健领先 |

`1024x256` relative retention 在 `4096x1024` 上为 Flash `0.3367`, GDN `0.0751`; 在 `8190x2047` 上为 Flash `0.1006`, GDN `0.0035`. GDN 的跨 seed 方差很小, 但其绝对外推准确率下降更快. Flash 的均值更高, 代价是更明显的 seed 敏感性.

主结果的逐 seed 准确率如下:

| 模型 | Seed | `1024x256` | `2048x512` | `4096x1024` | `8190x512` | `8190x2047` |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 123 | 0.974406 | 0.827164 | 0.459066 | 0.714262 | 0.151590 |
| Flash | 124 | 0.914328 | 0.476398 | 0.092451 | 0.273195 | 0.005586 |
| Flash | 125 | 0.971484 | 0.808711 | 0.425289 | 0.681805 | 0.136146 |
| GDN | 123 | 0.967875 | 0.480973 | 0.075885 | 0.295758 | 0.003669 |
| GDN | 124 | 0.966594 | 0.469613 | 0.072963 | 0.291336 | 0.003308 |
| GDN | 125 | 0.971086 | 0.477746 | 0.069318 | 0.293641 | 0.003157 |

### 3.2. `best.pt` 敏感性

只有 Flash seed124 的 best 与 last 不是同一 model-state. 该 best 来自 epoch3. 其他 5 条 run 的 best 都等于 epoch4 last, 因而 12 个逻辑角色去重为 7 个物理 checkpoint.

| Slice | Flash best mean | GDN best mean | Paired delta | Positive seeds | 分类 |
|---|---:|---:|---:|---:|---|
| `1024x256` | 0.956076 | 0.968518 | -0.012443 | 2/3 | 不支持 Flash 领先 |
| `2048x512` | 0.731868 | 0.476111 | +0.255758 | 3/3 | 稳健领先 |
| `4096x1024` | 0.342249 | 0.072722 | +0.269527 | 3/3 | 稳健领先 |
| `8190x512` | 0.578517 | 0.293578 | +0.284939 | 3/3 | 稳健领先 |
| `8190x2047` | 0.100694 | 0.003378 | +0.097316 | 3/3 | 稳健领先 |

Flash seed124 的 best-last 增量从 `1024x256` 到 `8190x2047` 分别为 `+0.008008`, `+0.083332`, `+0.049941`, `+0.066289`, `+0.008762`. 这说明 epoch4 的 overall validation 退化同时伴随长度外推退化, 不像是单一 Longer-MQAR slice 噪声.

## 4. 固定口径

- Flash: `baseline-r16-joint`, codebook `64`, rank `16`, read top-k `16`, write top-k `4`, `smooth_p4` softcap `0.5`, active state capacity `131072`, 参数量 `1160390`.
- GDN: `gdnxk-h2-ek4-ev4-usegate0`, 2 heads, expand K/V `4/4`, use gate false, active state capacity `131072`, 参数量 `1335942`.
- Training seeds `123/124/125`, data seed `123`; 两模型分别固定自己的 canonical seed124 init.
- Train `B64`, validation `B16`, GA4, 4 epochs, 每 epoch 4 次 validation, early stopping 关闭.
- Longer-MQAR 固定 `eval_seed=123`, vocab `8192`, `random_non_queries=true`, `power_a=0.01`, 每 slice `500` examples.
- `last.pt` 是预注册主结果; `best.pt` 只按 4 个 epoch-end 的整体 `valid/accuracy` 选择.
- RTX 2080 Ti GPU1, full-model FP32, `TRITON_F32_DEFAULT=ieee`, TF32 off, `GDN_KERNEL_DTYPE=float32`.
- 环境为 Python 3.12.11, PyTorch 2.6.0+cu118, CUDA 11.8, Triton 3.2.0, FLA 0.4.2.
- Zoology formal commit `0dd957274ca9609d0c151aef0e9b620fcd574e79`; Flash-VQG commit `ec770f33676036432c6514acd1ac05bd2d01f3e8`.
- MQAR cache content hash和四个 epoch batch-order hash均与 plan 中预注册值一致.

## 5. 完成性与复现审计

自动队列从 `2026-07-24T19:06:22Z` 运行到 `2026-07-24T22:23:25Z`, 总历时约 `3:17:04`. 正式训练前完成了:

- 两模型 shape smoke, 覆盖 train `T64/T128/T256` 和 validation `T64/T128/T256/T512/T1024`.
- 6/6 独立训练 smoke, 均完成 optimizer step、validation、last/best 保存和 strict reload.
- 10 个 batch-search smoke, 30/30 unique-source × slice eval smoke, 30/30 formal-probe 和 6/6 repro smoke.

正式训练后, 12 个 last/best 逻辑角色经过 checkpoint 文件 SHA256、model-state hash、epoch、finite metrics 和 strict load 审计, 得到 7 个唯一 model-state. Formal 阶段完成 35/35 source smoke、35/35 500-example formal events 和 7/7 repro. 所有 repro 的 dataset hash 一致且 accuracy delta 为 `0`.

五个 formal dataset hash与历史 RNG-locked official 数据完全一致. Batch search 在两个模型的 `8190x512` 和 `8190x2047` 上各有一次 batch32 OOM, 属于 plan 唯一允许的自动降档场景; 最终均选择 batch16并完成. 其余三个 slice选择 batch32. Formal、source smoke 和 repro 无 OOM、NaN、Inf或 Traceback.

第一次启动在 preflight import 阶段失败, 原因是入口文件原名 `queue.py` 遮蔽 Python 标准库 `queue.Queue`. 当时尚未启动任何 shape smoke、训练或正式 eval. commit `0dd9572` 将入口改为 `run_queue.py`, 重新通过全部测试和 preflight后才启动实验. 该失败不进入正式 ledger, 但保留在 raw log和 artifact metadata中.

正式运行期间 GPU1 触发过 NVIDIA 软件温控 cap, 硬件 thermal slowdown保持 inactive. 这会影响 wall time, 不影响本轮质量口径; 本报告不把 wall time用于模型效率结论.

## 6. Artifact 与记录

正式 artifact 位于 `docs/artifacts/20260725-01-current-baselines-longer-mqar/`, 包含:

- `training-final.csv`.
- `longer-mqar-detail.csv`, `longer-mqar-summary.csv`, `paired-deltas.csv`.
- `checkpoint-role-comparison.csv`, `source-manifest.csv`.
- `batch-sizes.csv`, `repro-verification.csv`, `verification.json`, `metadata.json`.
- `figures/longer-mqar-accuracy-curve.{pdf,png,svg}` 和对应绘图数据.

Flash 3 条训练已追加到 `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`; GDN 3 条训练已追加到 `docs/artifacts/gdn-expanded-k/gdn-expanded-k-summary.csv`. Longer-MQAR 索引只链接本轮独立 artifact, 没有改写 2026-05 official-core 表或旧 preliminary ledger.

大型 checkpoint和 raw事件日志保留在原路径. `source-manifest.csv` 记录每个逻辑 checkpoint角色的文件 SHA256、model-state hash、大小、epoch、训练配置和来源 result.

## 7. 下一步

1. 对 Flash seed124 检查 epoch3->epoch4 的常规 validation、优化轨迹和 residual/VQ运行指标, 判断是 seed敏感性还是训练后期退化.
2. 若需要把“Flash 长度泛化更强”升级为更强结论, 增加独立 training seeds, 不复用 Longer-MQAR 结果选择 seed或 checkpoint.
3. 维持当前 Flash和 GDN综合 baseline不变. 本轮提供长度泛化证据, 不单独触发 baseline替换.
