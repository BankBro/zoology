# 20260701-03 Flash-VQG default-dropout 1ep bridge trace plan

status: planned
ledger: not written for diagnostic/probe runs

## 目标

本轮补齐 `128 -> 704` optimizer step 的中间证据. 已知事实是:

- 128 step 时 read support / residual state 已明显跨机器分叉.
- 1 epoch 结束时 default dropout 下 `default-r2` 和 `default-r4` 都出现过 hard slice 跨机器失败.

本轮要回答:

```text
从 128 step 到 704 step 之间,
read/write/M_state/update/inject/lambda/beta 中哪条路径先出现异常增益,
并最终对应到 1ep 1024x256 hard slice gap?
```

## 共同条件

- branch: `flash-vqg`.
- seed: `124`.
- data seed: `123`.
- canonical MQAR cache: 内容 hash 必须 match.
- canonical seed124 init checkpoint: state_dict tensor hash 必须 match.
- model: `cb64-r16`.
- `vq_weight_mode=dense_softmax`.
- `fox_gd_residual_write_topk=4`.
- `resid_dropout=0.0`.
- `drop_path=0.0`.
- max epochs: `1`.
- max train steps: `704`, 等价于 1 epoch optimizer steps.
- trace steps: `0,16,64,128,256,384,512,704`.
- valid batch: `441`.
- machines: `2080ti` + `3090`, 均在 `Flash-VQG-tun` 容器内运行.

## Variants

| target | 配置 | 作用 |
|---|---|---|
| `default-r2` | `embed_dropout=0.1`, `read_topk=2` | default dropout 下 r2 1ep 失败路径桥接 |
| `default-r4` | `embed_dropout=0.1`, `read_topk=4` | default dropout 下 r4 1ep 失败路径桥接 |
| `dropout005-r4` | `embed_dropout=0.05`, `read_topk=4` | 扰动强度边界对照, 不作为最终训练协议 |

## 执行

1. Preflight:
   - 检查容器内 `nvidia-smi` 和 `torch.cuda.is_available()`.
   - 检查两机 git commit 一致.
   - 检查 13 个 MQAR cache 内容 hash 一致.
   - 检查 canonical init state hash 一致.
   - 检查 batch order hash 一致.

2. Train/eval trace:
   - 每个 target 在两机各跑 704 optimizer steps.
   - 使用 Trainer read trace 记录 fixed valid batch 的 read support, selected mass, read margin, M/update norm, lambda/inject ratio.
   - write 侧本轮先复用已有 aggregate 指标: write strength, zeta, raw topk mass, write top1 mass, write entropy.

3. Train-mode hash probe:
   - 每个 target 在两机各跑 704 optimizer steps.
   - 捕获 optimizer step `0,16,64,128,256,384,512,704` 的 train-mode forward hash.
   - 捕获 early backward grad hash 和 optimizer/model state hash.

4. 收尾:
   - 镜像 3090 轻量 evidence 回主工作区.
   - 生成 `docs/artifacts/20260701-03-flash-vqg-default-dropout-1ep-bridge-trace/`.
   - 写 `docs/20260701-03-flash-vqg-default-dropout-1ep-bridge-trace-report.md`.

长任务进入稳定运行后, 显式 `sleep 900` 轮询.

## 判读

| 现象 | 判读 | 下一步 |
|---|---|---|
| read support 继续分叉, 但 M/update/inject 平稳, final 仍崩 | 只看 support 不够, 需要查 residual contribution 或 optimizer 轨迹 | 加强 O_res/loss/grad trace |
| M_state update norm 或 M_state norm 在 256/384 后明显拉开 | M_state 写入/累积是主要放大点 | 做 update norm cap 或 M_state norm control |
| residual injection ratio 或 lambda/beta 明显拉开 | residual 注入强度是主要放大点 | 做 injection/lambda/beta warmup 或 cap |
| write aggregate 指标早于 read-side 指标恶化 | 写入侧可能先污染 memory | 下一轮补 raw write support trace 或 write margin guard |
| dropout005-r4 的同类指标明显更温和 | default dropout 强度把当前机制推过稳定边界 | 后续仍在 `embed_dropout=0.1` 下做稳定化 |

## 产物

- `execution-status-summary.csv`
- `preflight-effective-summary.csv`
- `first-mismatch-summary.csv`
- `hash-probe-comparison-summary.csv`
- `early-window-summary.csv`
- `read-trace-summary.csv`
- `read-trace-cross-machine-summary.csv`
- `variant-decision-summary.csv`
- `metadata.json`
- `README.md`

