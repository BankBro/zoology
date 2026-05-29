# 阶段 2 Flash local/remote eval-time ablation

本目录保存 `20260529-flash-local-window-fairness` 的阶段 2 Flash-only eval-time ablation 结果.

运行口径:

- 机器: mclab-3090, RTX 3090.
- source: `source_checkpoints.csv`, 仅包含 `cb256-r4-s123` 和 `cb64-r16-s123`.
- slices: `1024x256`, `2048x512`, `4096x512`, `4096x1024`.
- variants: `full`, `local_only`, `local1`, `local4`.
- `local_only`: `local_num_blocks=2`, `if_remote_enabled=false`.
- `local1`: `local_num_blocks=1`, remote 保持开启.
- `local4`: `local_num_blocks=4`, remote 保持开启.
- num_examples: 500.
- distance 定义: `query_pos - value_pos`, 来自 MQAR 生成时 position metadata.

完整性检查:

- `slice_summary.csv`: 32 rows.
- `distance_bucket.csv`: 288 rows.
- `eval_runs.csv`: 32 rows.
- run_status: 全部 `completed`.
- full sanity: 6 rows `passed`, 2 rows `no_ref`, 0 rows `invalid`.
- eval wall time: 5849.78 seconds.

核心观察:

- `local_only` 几乎失效, accuracy 约为 `0.00018-0.00027`, 接近随机/词表基线. 例如 `cb64-r16-s123 4096x1024`: full `0.468924`, local_only `0.000227`.
- `local1` 与 full 非常接近, 但略低. 例如 `cb64-r16-s123 4096x1024`: local1 `0.466941` vs full `0.468924`.
- `local4` 与 full 也非常接近, 多数 slice 略高. 例如 `cb64-r16-s123 4096x1024`: local4 `0.472484` vs full `0.468924`.
- 第一轮 four-slice eval 中, `<=64` distance bucket 没有样本. 因此本阶段不能直接测量 64-token exact local window 内的收益.
- 在远距离 bucket 上, full/local1/local4 基本保持一致, local_only 接近 0. 例如 `cb64-r16-s123 4096x1024`:
  - `513-1024`: full `0.399500`, local_only `0.000385`, local1 `0.398635`, local4 `0.400942`.
  - `1025-2048`: full `0.454906`, local_only `0.000221`, local1 `0.453101`, local4 `0.458563`.
  - `2049-4096`: full `0.488529`, local_only `0.000225`, local1 `0.486288`, local4 `0.492071`.

解释限制:

- 阶段 2 是 eval-time override, 不是重新训练, 因此只能说明已训练 checkpoint 的 remote 分支对当前 longer-MQAR 泛化是必要的.
- 因第一轮数据没有 `<=64` bucket 样本, 若要直接回答 local exact window 内收益, 需要增加 near-distance focused eval 或训练/eval 配置.
- 本结果支持: 当前 longer-MQAR 上 Flash 优势不是主要来自 64-token exact local window, 因为大部分样本都远超 64, 且禁用 remote 后性能坍塌.
