# 20260627-02-flash-vqg-canonical-init-lock-screen

本 artifact 收尾 2026-06-27 的 canonical cache + canonical init 1 epoch screen. 该 screen 属于 debug / hygiene 实验, 不写入正式 MQAR ledger.

## 文件

- `run-summary.csv`: 三条有效 run 的最终指标, 耗时, raw evidence hash.
- `machine-summary.csv`: 按机器聚合的 overall 与关键 `1024x256` 指标.
- `cross-machine-comparison.csv`: 以 2080ti r1 为参考的 gap. 同时给出 overall 和关键 `1024x256` 是否在 4 percentage points 容忍线内.
- `cache-actual-13-summary.csv`: 本轮实际加载 13 个 MQAR cache 的内容级一致性摘要.
- `init-verify-summary.csv`: canonical init checkpoint 在两台机器上的 state_dict hash 验证.
- `initlock-probe-summary.csv`: init-lock early-step probe 的关键 hash 对比.
- `source-manifest.csv`: raw evidence 的路径, sha256 和镜像状态.
- `metadata.json`: 机器, commit, cache/init/probe 解释和限制.

## 核心结论

- 本轮实际加载的 13 个 cache 在 2080ti 与 3090 上内容一致.
- canonical init checkpoint 在两台机器上加载后的 model state hash 一致.
- early-step probe 显示 batch order, initial model params, first inputs 和 first targets 一致, 但 first logits / first grad / optimizer step 后参数不一致. 因此现有证据只能说明差异出现在相同输入和相同 init 之后的 GPU 数值执行路径, 不能定位到具体层或 kernel.
- overall `valid/accuracy` gap 在 4 percentage points 内, 但关键 `valid/mqar_case/accuracy-1024x256` gap 为 8-10 percentage points, 超过 4 percentage points 容忍线. 如果 `1024x256` 是重点目标, 本轮结果不能按“可接受误差”处理.
