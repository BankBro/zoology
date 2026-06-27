# 20260628-01 Flash-VQG no-dropout 稳定性 ablation 报告

本轮是 diagnostic / exploratory 1 epoch screen, 不写 official MQAR ledger. 目标是回答一个低成本问题: 在 2080ti 和 3090 上固定同一份 canonical MQAR cache, 固定同一份 canonical init checkpoint, 并关闭 dropout 后, `1024x256` hard slice 的跨机器差距是否能压到用户可接受的 4pp 以内.

## 执行口径

代码版本:

- zoology: `flash-vqg`, commit `f41acb0`.
- Flash-VQG: `20260428-gd-residual-v1-sync`, commit `1e7ed33`.

共同训练配置:

- `seed=123`, `data_seed=123`.
- `cb64-r16`, `read_topk=2`.
- `train_batch_size=64`, `gradient_accumulation_steps=4`, 1 epoch, 704 optimizer steps.
- 加载 2080ti 生成的 canonical init checkpoint.
- `embed_dropout=0.0`, `resid_dropout=0.0`, `drop_path=0.0`.

前置硬门槛全部通过:

- 两边容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 均可用.
- 本轮实际加载的 13 个 MQAR cache content hash 均为 `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- init `model_state_dict` tensor hash 均为 `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`.

3090 原始轻量 evidence 已镜像回 2080ti 主工作区. 镜像后按相对路径比较 sha256, 结果为 `10/10` 文件一致. 大型 checkpoint, swanlog 和 tensor trace 不纳入本轮提交, 本轮审计依赖 queue status, config JSON, result JSON, stdout log, cache hash 和 init hash.

## 结果

主指标是 `valid/mqar_case/accuracy-1024x256`.

| machine | target | valid acc | 1024x256 | time |
|---|---|---:|---:|---:|
| 2080ti | `no-embed-dropout-s123-r1` | 0.917 | 0.606 | 85 min |
| 3090 | `no-embed-dropout-s123-r1` | 0.919 | 0.620 | 65 min |
| 3090 | `no-embed-dropout-s123-r2` | 0.920 | 0.626 | 65 min |

以 2080ti run 为参考:

| candidate | 1024x256 gap | within 4pp |
|---|---:|---|
| 3090 r1 | 1.4pp | true |
| 3090 r2 | 2.0pp | true |

本轮 3 个 run 均完成, `invalid_count=0`.

## 判读

这轮结果支持一个很具体的判断: 在 cache 和 init 都锁住之后, 关闭 dropout 可以把当前 `s123, cb64-r16, 1 epoch` 的 `1024x256` 跨机器差距压到 4pp 内. 这说明之前 first-divergence probe 找到的 dropout 分叉不是无关细节, 它确实会被训练过程放大到 hard slice 指标上.

更具体地说, cache 和 init 只能保证两边从相同数据和相同权重起步, 不能保证训练轨迹相同. 当前最合理的解释是: baseline 配置里的 early dropout mask 跨 GPU 不一致, 这类早期随机路径差异进入 Flash-VQG 后, 会被 VQ routing, GD residual 的离散 read/write 选择和 state 累积过程继续放大, 最后体现为 `1024x256` hard slice 的明显差距. no-dropout 后跨机器 gap 明显缩小, 说明这个 dropout/RNG 扰动链条是真实影响因素, 不是单纯的日志或测量噪声.

但这还不是最终训练方案. 原因有三点:

1. 本轮只跑 1 epoch, 没有验证 4 epoch final checkpoint.
2. 本轮把 `embed_dropout`, `resid_dropout`, `drop_path` 全部设为 0.0, 还没有区分到底主要是 embed dropout, residual dropout, 还是 drop path 的贡献.
3. 本轮只覆盖 `s123` 和当前 canonical cache/init, 不能直接外推到所有 seed 和所有容量布局.

更稳妥的结论是: dropout/RNG 路径是当前跨 GPU 效果差异的优先处理方向. 如果下一步要找解决方案, 应优先做 dropout policy 的最小改动, 而不是继续盲目查 cache 或 init. 但 no-dropout 只是诊断成功, 还不能直接当作最终方案, 因为本轮 1 epoch 的绝对 `1024x256` 分数低于上一轮 default good run, 可能存在学习速度或 ceiling tax.

## 下一步建议

下一步不建议继续做“前几步是否 bitwise 一致”的实验, 因为用户已经明确接受跨 GPU 训练过程不会完全一致. 更有价值的是验证可用训练策略:

1. 跑一个 4 epoch confirm: 固定 canonical cache/init, 使用当前 no-dropout policy, 先做 `2080ti x1 + 3090 x2`, 看 final `1024x256` 是否仍在 4pp 内, 同时看是否有明显 ceiling tax.
2. 如果 no-dropout 4 epoch 稳定但性能下降, 再做分项 ablation: 只关 `embed_dropout`, 只关 `resid_dropout`, 或保留训练 dropout 但 eval/seed policy 固定.
3. 如果 no-dropout 4 epoch 仍漂移, 再回到 Flash-VQG mixer 内部, 查 read/write state 放大链, 而不是继续重复 cache/init 实验.

## 产物

Artifact:

- `docs/artifacts/20260628-01-flash-vqg-stability-ablation/run-summary.csv`
- `docs/artifacts/20260628-01-flash-vqg-stability-ablation/cross-machine-comparison.csv`
- `docs/artifacts/20260628-01-flash-vqg-stability-ablation/cache-init-preflight-summary.csv`
- `docs/artifacts/20260628-01-flash-vqg-stability-ablation/queue-summary.csv`
- `docs/artifacts/20260628-01-flash-vqg-stability-ablation/source-manifest.csv`
- `docs/artifacts/20260628-01-flash-vqg-stability-ablation/metadata.json`
