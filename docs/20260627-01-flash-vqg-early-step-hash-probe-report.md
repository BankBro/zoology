# 20260627-01 Flash-VQG early-step hash probe 报告

## 结论

本轮 debug probe 的核心结论是: canonical cache 已经统一了输入数据和 batch order, 但当前训练仍不能保证跨机器一致, 因为模型初始状态在 forward/backward 之前已经跨 2080ti 和 3090 分叉.

最直接证据:

- `cross-machine-baseline-a`: 25 个 hash 比较项中 22 个 mismatch, 首个 mismatch 是 `after_model_to_device_before_optimizer_step/model_params_sha256`.
- `cross-machine-deterministic-a`: 25 个 hash 比较项中 22 个 mismatch, 首个 mismatch 仍是 `after_model_to_device_before_optimizer_step/model_params_sha256`.
- 第一批 `inputs_sha256` 和 `targets_sha256` 跨机一致, 但 `logits_sha256` 不一致.

因此, 当前最优先的问题不是继续补 full-epoch repeat, 而是先把模型 init 锁住.

## 诊断摘要

| comparison | mismatch rows | first mismatch |
|---|---:|---|
| 2080ti baseline same-machine | 0/25 | none |
| 2080ti deterministic same-machine | 8/25 | `after_microbatch_backward`, micro 3, `grad_sha256` |
| 3090 baseline same-machine | 4/25 | `after_microbatch_backward`, micro 6, `grad_sha256` |
| 3090 deterministic same-machine | 0/25 | none |
| cross-machine baseline-a | 22/25 | initial `model_params_sha256` |
| cross-machine deterministic-a | 22/25 | initial `model_params_sha256` |

## 初始状态差异

per-key state dict hash 显示初始差异集中在 3 项:

- `backbone.embeddings.word_embeddings.weight`
- `lm_head.weight`
- `backbone.layers.1.sequence_mixer.mixer.attn.quantizer.codebook`

其中 `lm_head.weight` 与 word embedding 绑定, 所以主要差异是 embedding 初始化. codebook 差异是 1e-7 量级, 来自 Flash-VQG codebook init 中设备侧随机和归一化路径的可能性较高.

单独 `TokenEmbeddings` 初始化验证:

| machine | device | hash |
|---|---|---|
| 2080ti | cpu | `65ba3546ecf3af5739d4394d0ac82d3f310bed66cd68aa95ddf3fde1688d6134` |
| 3090 | cpu | `65ba3546ecf3af5739d4394d0ac82d3f310bed66cd68aa95ddf3fde1688d6134` |
| 2080ti | cuda | `02ca7b1c41a20f0dd2e246991f22169604d4fb5649b6dae47d51f992b187a443` |
| 3090 | cuda | `a4a74d6a4ce713bf3206add285b0507799bd85bcd94ec3cf04a678b61780b728` |

这说明同一 seed 下 CPU 初始化跨机一致, CUDA 初始化跨 GPU 架构不一致. `TokenEmbeddings` 当前默认在 `device=None` 时选择 `"cuda" if torch.cuda.is_available() else "cpu"`, 因而正式训练会在 GPU 上直接初始化 embedding.

## 解释

canonical cache 修复了 `random_non_queries=True` 导致的跨机数据不一致问题, 但没有锁定模型初始状态. 在当前路径中, 至少 embedding 初始化和 Flash-VQG codebook 初始化存在 GPU-local 随机生成/数值路径. 因此即使 cache, sampler order 和第一批输入完全一致, 两台机器也会从不同初始点开始训练.

3090 baseline 同机双跑还出现 early backward gradient hash 分叉, deterministic 开关可在 3090 上消除这个分叉. 2080ti deterministic 双跑仍有 gradient mismatch, 说明后续还可能有 backward/optimizer 层面的非确定性. 但跨机问题的第一优先级仍是初始状态已经不同.

## 下一步建议

不要马上继续补 full-epoch repeat. 建议先做 `init-lock` probe:

1. 在 2080ti 生成一个 canonical init state snapshot, 记录 state dict hash.
2. 在 3090 加载同一个 init snapshot, 再跑 early-step hash probe.
3. 如果 `after_model_to_device_before_optimizer_step` 跨机 match, 继续看 first forward, backward, optimizer step 是否分叉.
4. 若 init-lock 后早期 hash 仍分叉, 再定位 forward/backward kernel.
5. 若 init-lock 后早期 hash 对齐, 再考虑跑 1 epoch screen, 并比较 s123 是否回到接近 2080ti 的分布.

长期方案可以考虑把 embedding/codebook/address init 改为 CPU/local generator 生成后再搬到目标设备, 或把正式多机复现实验规定为必须加载同一 init checkpoint.

## Artifact

- `docs/artifacts/20260627-01-flash-vqg-early-step-hash-probe/comparison-summary.csv`
- `docs/artifacts/20260627-01-flash-vqg-early-step-hash-probe/initial-state-diff.csv`
- `docs/artifacts/20260627-01-flash-vqg-early-step-hash-probe/token-embedding-init-summary.csv`
- `docs/artifacts/20260627-01-flash-vqg-early-step-hash-probe/source-manifest.csv`
- `docs/artifacts/20260627-01-flash-vqg-early-step-hash-probe/metadata.json`

本轮是 debug probe, 不写入正式 MQAR ledger.
