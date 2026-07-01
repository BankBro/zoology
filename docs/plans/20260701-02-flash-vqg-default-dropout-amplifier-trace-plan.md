# 20260701-02 Flash-VQG default-dropout amplifier trace plan

status: completed
ledger: not written for diagnostic/probe runs

## 目标

本轮只做定位实验, 不做稳定化方案. 目标是在 zoology 正常训练协议 `embed_dropout=0.1` 下, 找出 dropout 训练扰动进入 Flash-VQG/GD residual 后, 最先被哪条路径放大:

```text
layer0 dropout / Flash-VQG input
VQ routing
fox gate/logf
beta/lambda
write support
M_state update
read support
O_res / loss / grad
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
- max train steps: `128`.
- trace steps: `0,1,4,16,64,128`.
- valid batch: `441`.
- machines: `2080ti` + `3090`, 均在 `Flash-VQG-tun` 容器内运行.

## Variants

| target | 配置 | 作用 |
|---|---|---|
| `default-r4` | `embed_dropout=0.1`, `read_topk=4` | default dropout 下已知失败链路的主定位对象 |
| `dropout005-r4` | `embed_dropout=0.05`, `read_topk=4` | 1ep 过线的扰动边界对照 |
| `default-r2` | `embed_dropout=0.1`, `read_topk=2` | 判断失败链路是否只和 `read_topk=4` 有关 |

## 执行

1. Preflight:
   - 检查容器内 `nvidia-smi` 和 `torch.cuda.is_available()`.
   - 检查两机 git commit 一致.
   - 检查 13 个 MQAR cache 内容 hash 一致.
   - 检查 canonical init state hash 一致.

2. Train/eval trace:
   - 每个 target 在两机各跑 128 optimizer steps.
   - 使用 Trainer read trace 记录 fixed valid batch 的 read support, selected mass, read margin, M/update norm, lambda/inject ratio.

3. Train-mode hash probe:
   - 每个 target 在两机各跑 128 optimizer steps.
   - 捕获 optimizer step `0,1,4,16,64,128` 的 train-mode forward hash.
   - 捕获 early backward grad hash 和 optimizer/model state hash.

4. 收尾:
   - 镜像 3090 轻量 evidence 回主工作区.
   - 生成 `docs/artifacts/20260701-02-flash-vqg-default-dropout-amplifier-trace/`.
   - 写 `docs/20260701-02-flash-vqg-default-dropout-amplifier-trace-report.md`.

长任务进入稳定运行后, 显式 `sleep 900` 轮询.

## 判读

本轮不以 128-step acc 作为主结论, 而以 first mismatch 和放大链路为主:

| 现象 | 判读 |
|---|---|
| `default-r4` 和 `dropout005-r4` first mismatch 相同, 但后续 M/update/inject/loss 只有 default 放大 | dropout 强度影响放大增益, 不是单纯是否分叉 |
| `default-r2` 和 `default-r4` first mismatch 相同 | 问题不只是 read_topk 数量 |
| write support 或 M_state update 早于 read support 异常 | 下一轮优先看 write/M_state 稳定化 |
| read support 早于 write/M_state 异常 | 下一轮优先看 read candidate 稳定化 |
| lambda/inject ratio 早期差异大 | 下一轮优先看 residual injection 控制 |

## 产物

- `preflight-summary.csv`
- `early-window-summary.csv`
- `read-trace-summary.csv`
- `read-trace-cross-machine-summary.csv`
- `hash-probe-comparison-summary.csv`
- `first-mismatch-summary.csv`
- `variant-decision-summary.csv`
- `metadata.json`
- `README.md`
