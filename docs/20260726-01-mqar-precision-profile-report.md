# MQAR 低精度与长度泛化实验报告

## 1. 结果概览

`20260726-01-mqar-precision-profile` 已完成 30/30 个正式训练 run 和 2028 个逻辑 checkpoint-eval 事件, 其中 1066 个为物理执行, 962 个因 best/last state hash 相同而可审计去重. 2080 Ti 完成 12/12 run, RTX 3090 完成 18/18 run. 全部结果均在双机 train/validation/eval smoke, controlled resume, 全量 batch capacity, batch invariance, legacy canary 和 global commit/cache gate 通过后生成.

主结论是: 低精度训练在两种模型上均可稳定完成, 且不改变 Flash 在四个真正外推 slice 上相对 GDN 的长度泛化优势. 但 Flash 对训练机器和低精度训练轨迹比 GDN 更敏感; 不能把 2080 Ti 与 RTX 3090 合并为 `n=6`, 也不能把单次低精度差异解释为精度本身的确定性增益.

## 2. 实验口径

RTX 2080 Ti 比较 FP32 与 AMP-FP16, RTX 3090 比较 FP32, AMP-FP16 与 AMP-BF16. 每个模型和 dtype 使用 seeds `123,124,125`, 固定 B64, GA4 和 4 epochs. Flash-VQG 仅在 grouped update 与 selected-read Triton core 外建立 FP32 boundary; GDN 使用与实验 dtype 匹配的 FLA kernel dtype.

主结果使用 matching train/eval dtype. Off-diagonal 网格只用于机制分析. 两张 GPU 分别计算 3 seeds 的 mean 与 population std, 不合并为 `n=6`.

全局 gate 绑定 Zoology `e56fa9abd727`, Flash-VQG `9a8bf7074f90` 和 cache `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`. 两机 13 个 `shape x num_examples` 数据身份各自只有一个 dataset hash.

## 3. Matching dtype 主结果

下表是 last checkpoint 的 500-example longer-MQAR accuracy, 格式为 `mean ± population SD`, 每行 `n=3` seeds.

| GPU | 模型 | 精度 | 1024x256 | 2048x512 | 4096x1024 | 8190x512 | 8190x2047 |
|---|---|---|---:|---:|---:|---:|---:|
| 2080 Ti | FLASH | FP32 | 0.9602 ± 0.0219 | 0.7110 ± 0.1363 | 0.3145 ± 0.1484 | 0.5470 ± 0.1767 | 0.0932 ± 0.0613 |
| 2080 Ti | FLASH | FP16 | 0.9357 ± 0.0232 | 0.6562 ± 0.0880 | 0.2653 ± 0.1112 | 0.4966 ± 0.1286 | 0.0742 ± 0.0507 |
| 2080 Ti | GDN | FP32 | 0.9685 ± 0.0019 | 0.4761 ± 0.0048 | 0.0727 ± 0.0027 | 0.2936 ± 0.0018 | 0.0034 ± 0.0002 |
| 2080 Ti | GDN | FP16 | 0.9686 ± 0.0018 | 0.4763 ± 0.0047 | 0.0728 ± 0.0027 | 0.2936 ± 0.0018 | 0.0034 ± 0.0002 |
| RTX 3090 | FLASH | FP32 | 0.9501 ± 0.0309 | 0.7543 ± 0.0889 | 0.3889 ± 0.0895 | 0.6262 ± 0.0884 | 0.1277 ± 0.0376 |
| RTX 3090 | FLASH | FP16 | 0.9645 ± 0.0041 | 0.7938 ± 0.0187 | 0.4223 ± 0.0226 | 0.6674 ± 0.0211 | 0.1385 ± 0.0093 |
| RTX 3090 | FLASH | BF16 | 0.9572 ± 0.0082 | 0.7507 ± 0.0481 | 0.3758 ± 0.0551 | 0.6207 ± 0.0529 | 0.1216 ± 0.0206 |
| RTX 3090 | GDN | FP32 | 0.9687 ± 0.0014 | 0.4789 ± 0.0036 | 0.0750 ± 0.0011 | 0.2986 ± 0.0044 | 0.0036 ± 0.0001 |
| RTX 3090 | GDN | FP16 | 0.9686 ± 0.0014 | 0.4788 ± 0.0037 | 0.0750 ± 0.0011 | 0.2986 ± 0.0043 | 0.0036 ± 0.0001 |
| RTX 3090 | GDN | BF16 | 0.9687 ± 0.0012 | 0.4782 ± 0.0043 | 0.0747 ± 0.0014 | 0.2978 ± 0.0049 | 0.0036 ± 0.0001 |

关键观察:

- GDN 对训练精度近乎不敏感. 所有低精度相对 FP32 的五个 slice 均值变化绝对值不超过 `0.000797`.
- Flash 在 2080 Ti 上使用 FP16 后, last accuracy 相对 FP32 的五个 slice 变化为 `-0.0548 至 -0.0189`. 在 RTX 3090 上, FP16 变化为 `+0.0108 至 +0.0412`, BF16 变化为 `-0.0131 至 +0.0071`. 方向随机器改变, 说明主要是训练轨迹和 GPU 数值路径敏感性, 不是统一的“低精度提升”或“低精度退化”.
- 在排除训练端点 `1024x256` 后的四个外推 slice 上, Flash 在 `60/60` 个 `GPU x matching dtype x seed x shape` 配对中高于 GDN. 在 `1024x256` 训练端点, Flash 仅在 `3/15` 个配对中高于 GDN, 因而端点不支持 Flash 优于 GDN.
- 对固定训练 checkpoint 只改变 eval dtype, accuracy 的全网格最大跨度为 `0.002328`. 这远小于 Flash 的主要训练精度和跨 GPU 差异, 说明外围 FP32 boundary 与低精度 evaluator 本身没有造成主要质量漂移.

Best checkpoint 图支持相同的定性结论. 2080 Ti 上 best 选择可缓解部分 Flash last 波动; RTX 3090 上多数 best/last state 相同并被物理去重. 完整 best 数值见汇总 CSV.

## 4. 训练效率与数值审计

| GPU | 模型 | 训练精度 | run | wall time, min | step p50, s | peak allocated, MiB | peak reserved, MiB | scaler skips |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2080 Ti | FLASH | FP16 | 3 | 25.52 | 0.415 | 2565 | 3157 | 0 |
| 2080 Ti | FLASH | FP32 | 3 | 27.30 | 0.452 | 3131 | 3353 | 0 |
| 2080 Ti | GDN | FP16 | 3 | 5.42 | 0.076 | 1536 | 2108 | 0 |
| 2080 Ti | GDN | FP32 | 3 | 19.74 | 0.332 | 1920 | 2894 | 0 |
| RTX 3090 | FLASH | BF16 | 3 | 19.85 | 0.326 | 2566 | 3157 | 0 |
| RTX 3090 | FLASH | FP16 | 3 | 20.90 | 0.344 | 2566 | 3157 | 1 |
| RTX 3090 | FLASH | FP32 | 3 | 23.03 | 0.391 | 3131 | 3374 | 0 |
| RTX 3090 | GDN | BF16 | 3 | 5.16 | 0.075 | 1536 | 2108 | 0 |
| RTX 3090 | GDN | FP16 | 3 | 5.57 | 0.082 | 1536 | 2108 | 0 |
| RTX 3090 | GDN | FP32 | 3 | 29.02 | 0.519 | 1920 | 2894 | 0 |

- Flash-FP16 的平均 wall time 相对 FP32 为 2080 Ti `0.935x`, RTX 3090 `0.908x`; RTX 3090 Flash-BF16 为 `0.862x`. Flash 低精度 peak allocated memory 约为 FP32 的 `0.819x`.
- GDN 低精度加速更明显: 2080 Ti FP16 为 `0.274x`, RTX 3090 FP16 为 `0.192x`, BF16 为 `0.178x`. GDN 低精度 peak allocated memory 约为 FP32 的 `0.800x`.
- 30 个 run 的 model master weights 和 optimizer state 均保持 FP32. GDN kernel dtype 分别严格为 `float32`, `float16`, `bfloat16`. 全实验只记录 `1` 次 FP16 GradScaler skip, 位于 `3090-flash-s125-fp16`, 未超过预注册的每 run 上限 2, 该 run 最终正常完成 epoch 4 且指标有限.

## 5. 审计与证据

- 2080 Ti gate: 52/52 capacity profiles, 52/52 batch invariance, 312/312 eval smoke, 26/26 canary, 16/16 standard accuracy audit.
- RTX 3090 gate: 78/78 capacity profiles, 78/78 batch invariance, 702/702 eval smoke, 26/26 canary, 16/16 standard accuracy audit.
- 两机 `8190x2047` smoke 均实际执行 controlled interrupt 并从 batch cursor 恢复完成.
- 18 个 preflight/status/formal-detail/gate JSON 与30个resolved training config已镜像到artifact, 全部通过source/mirror SHA256一致性校验. 60个checkpoint大文件保留在source machine原路径, file SHA256记录于source manifest.
- 30 条正式训练记录写入独立 canonical training ledger; 780 条正式 longer-MQAR 逻辑评估写入 canonical eval ledger, 包含 source/eval dtype, 开始结束时间, wall time, GPU, batch, dataset/checkpoint hash 和物理去重状态.
- 本实验不覆盖历史 FP32 canonical ledger; 它是独立 precision profile. Matching dtype 为 official 主比较口径, off-diagonal 仅用于机制分析.

## 6. 产物

- Last 图: [matching-precision-last.pdf](artifacts/20260726-01-mqar-precision-profile/figures/matching-precision-last.pdf).
- Best 图: [matching-precision-best.pdf](artifacts/20260726-01-mqar-precision-profile/figures/matching-precision-best.pdf).
- 正式明细: [final.csv](artifacts/20260726-01-mqar-precision-profile/final.csv).
- 汇总: [precision-grid-summary.csv](artifacts/20260726-01-mqar-precision-profile/combined/precision-grid-summary.csv).
- 训练 ledger: [canonical-training-ledger.csv](artifacts/20260726-01-mqar-precision-profile/canonical-training-ledger.csv).
- Longer-MQAR ledger: [canonical-longer-mqar-ledger.csv](artifacts/20260726-01-mqar-precision-profile/canonical-longer-mqar-ledger.csv).
- Source manifest: [source-manifest.csv](artifacts/20260726-01-mqar-precision-profile/source-manifest.csv).
