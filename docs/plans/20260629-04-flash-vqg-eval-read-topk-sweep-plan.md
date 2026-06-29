# 20260629-04 Flash-VQG eval read-topk sweep plan

## 目标

评估最近 dense-read 4 epoch 实验保存的 8 个 checkpoint 在不同 evaluation read topk 下的表现, 判断训练时 `read_topk=64` 的模型在评估阶段改成 `read_topk=1/2/4/8/16/32/64` 是否会明显影响 `valid/accuracy` 和重点 hard slice `valid/mqar_case/accuracy-1024x256`.

本实验不重新训练, 不更新权重, 不保存新 checkpoint.

## 输入 checkpoint

来源实验:

```text
20260629-03-flash-vqg-dense-read-confirm
```

有效 run:

```text
2080ti r1 best.pt / last.pt
2080ti r2 best.pt / last.pt
3090 r1 best.pt / last.pt
3090 r2 best.pt / last.pt
```

失败或 stale-start run 不纳入本实验.

## 评估矩阵

每个 checkpoint 评估:

```text
fox_remote_read_topk = 1,2,4,8,16,32,64
```

评估分两轮:

1. 原机器评估: 2080ti checkpoint 在 2080ti eval, 3090 checkpoint 在 3090 eval.
2. 交换机器评估: 2080ti checkpoint 复制到 3090 eval, 3090 checkpoint 复制到 2080ti eval.

总评估数:

```text
8 checkpoints * 7 topk * 2 eval_machines = 112
```

## 执行约束

- 评估前确认两台机器容器内 `nvidia-smi` 和 `torch.cuda.is_available()` 可用.
- 评估前确认两边 zoology 代码处于同一分支和 commit.
- 交叉评估前复制 checkpoint 后做 sha256 校验.
- 评估使用 checkpoint 自带 `train_config.json` 重建模型和 validation data.
- 评估只覆盖 Flash-VQG mixer/attention config 中的 `fox_remote_read_topk`.
- `fox_remote_read_topk_initial/final` 保持 `None`, 避免 schedule 干扰固定 topk eval.
- `fox_gd_residual_dense_read_chunked` 保持 checkpoint 配置, 因此 `topk=64` 会走当前 dense-read chunked 路径.
- 输出追加式 JSONL, 支持中断后 resume.

## 关键输出

正式 artifact:

```text
docs/artifacts/20260629-04-flash-vqg-eval-read-topk-sweep/
```

至少包含:

- `eval-summary.csv`
- `cross-machine-eval-comparison.csv`
- `checkpoint-manifest.csv`
- `source-manifest.csv`
- `metadata.json`
- `README.md`

人读报告:

```text
docs/20260629-04-flash-vqg-eval-read-topk-sweep-report.md
```

## 判读口径

- 如果同一 checkpoint 的小 topk eval 明显低于 `topk=64`, 说明 dense-read 训练得到的模型在评估阶段也依赖更宽 read support.
- 如果 `topk=8/16/32` 接近 `topk=64`, 说明可以把中等 topk 作为低成本推理/评估候选.
- 如果同一 checkpoint 跨机器 eval 差异很小, 说明当前主要差异更可能来自训练轨迹, 而不是纯 eval runtime.
- 如果同一 checkpoint 跨机器 eval 差异明显, 则需要把 eval runtime 数值路径单独列为风险.
- `topk=64` 是本轮参考点, 不是默认 deployment 方案.
