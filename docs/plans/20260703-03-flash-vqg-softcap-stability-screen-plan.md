# 20260703-03 Flash-VQG smooth cap stability screen plan

## 背景

当前已经排除了早期最明显的 cache/init/batch-order 混杂因素. 训练仍然会在正常 `embed_dropout=0.1` 扰动下出现跨机器效果波动. 过去几轮实验支持一个更具体的判断:

- dropout 是正常训练扰动入口, 不是 bug.
- 后续 `read/write/state/residual injection` 机制会把小扰动放大.
- hard `update_norm_cap=0.5` 曾显著缩小 gap, 但复现不稳, 因此不能作为最终方案.
- residual injection warmup 有帮助, 但单独使用仍不能稳定过 4pp gap.

本轮不继续做大范围调参, 只验证两个最小平滑控制:

1. `M_state` 写入幅度 softcap.
2. residual 读出后注入主输出的 injection softcap.

## 代码改动

Flash-VQG 新增默认关闭配置:

```text
fox_gd_residual_injection_softcap_ratio = None
fox_gd_residual_injection_softcap_mode = "none"
fox_gd_residual_update_norm_softcap = None
fox_gd_residual_update_norm_softcap_mode = "none"
```

本轮只支持 `smooth_p4`:

```python
scale = (1 + (x / cap) ** 4) ** (-1 / 4)
```

### Injection softcap

作用位置:

```text
O_res_added 算出后, Out_f32 = O_base + O_res_added 之前.
```

定义:

```python
ratio = norm(O_res_added) / (norm(O_base) + eps)
scale = (1 + (ratio / 0.5) ** 4) ** (-1 / 4)
O_res_added = O_res_added * scale
```

解释: 控制 residual correction 相对 base output 的注入强度, 避免少数 query 的 residual 读出过强.

### Update norm softcap

作用位置:

```text
residual GD 写入 M_state 之前.
```

定义:

```python
update_norm = abs(zeta) * norm(err)
scale = (1 + (update_norm / 0.5) ** 4) ** (-1 / 4)
zeta = zeta * scale.detach()
```

解释: 控制单次 residual GD update 写入 `M_state` 的幅度. `scale.detach()` 用于保持与既有 hard update cap 类似的梯度口径.

## 实验矩阵

共同条件:

```text
seed = 124
data_seed = 123
max_epochs = 1
max_train_steps = 704
cb64-r16
read_topk = 2
write_topk = 4
embed_dropout = 0.1
resid_dropout = 0.0
drop_path = 0.0
canonical MQAR cache
canonical seed124 init
read_trace_enabled = false
read_trace_train_steps = []
```

Variants:

| variant | 目的 | softcap | warmup |
|---|---|---|---|
| `baseline-r2-no-trace` | 无 trace baseline | none | none |
| `inject-softcap0p5-r2` | 只限制 residual 注入 | injection ratio 0.5 | none |
| `inject-softcap0p5-linear512-r2` | 注入限制 + 已知有帮助的 warmup | injection ratio 0.5 | 0 -> 512 optimizer steps |
| `update-softcap0p5-r2` | 只限制 `M_state` 单次写入 | update norm 0.5 | none |
| `update-softcap0p5-linear512-r2` | 写入限制 + warmup | update norm 0.5 | 0 -> 512 optimizer steps |

本轮不同时开启 injection softcap 和 update softcap, 先拆清两个路径谁更有效.

## 启动和监控

启动前必须检查:

- 2080ti/3090 容器内 `nvidia-smi` 和 `torch.cuda.is_available()`.
- 两个仓库在两台机器上同步到相同 commit.
- MQAR cache 内容 hash 一致.
- canonical init tensor hash 一致.
- batch order hash 一致.

如果任一 preflight 不一致, 停止启动训练.

GPU 队列:

```text
2080ti GPU0:
  baseline-r2-no-trace
  inject-softcap0p5-r2

2080ti GPU1:
  inject-softcap0p5-linear512-r2
  update-softcap0p5-r2
  update-softcap0p5-linear512-r2

3090 GPU0:
  all five variants sequentially
```

进入稳定训练后, 使用显式 20 分钟轮询:

```bash
sleep 20m
```

每次轮询检查:

- GPU 占用.
- queue log.
- 当前 variant log.
- `Traceback`, `RuntimeError`, `CUDA out of memory`, `loss=nan`, `loss=inf`.
- 是否完成并进入下一 variant.

## 收尾报告

生成:

```text
docs/20260703-03-flash-vqg-softcap-stability-screen-report.md
docs/artifacts/20260703-03-flash-vqg-softcap-stability-screen/
```

Artifact 至少包含:

```text
run-summary.csv
cross-machine-comparison.csv
softcap-metrics-summary.csv
preflight-summary.csv
source-manifest.csv
metadata.json
README.md
```

报告必须回答:

1. 哪个 variant 同时保持高分和低 gap.
2. injection softcap 和 update softcap 哪个更有效.
3. warmup + softcap 是否优于单独 softcap.
4. softcap hit ratio/scale 显示的是少量尖峰被压, 还是大面积压缩.
5. 是否值得 repeat.
6. 是否值得进入 4ep confirm.

判定标准:

```text
通过:
  两机 final 1024x256 hard slice 都 >= 0.82
  且 gap <= 4pp

边界:
  gap <= 4pp 但任一机器 < 0.80
  说明可能过度抑制, 不推进 4ep

失败:
  gap > 4pp
  或任一机器明显低分
```

如果有 variant 通过, 下一步先做 no-trace paired 1ep repeat. repeat 仍通过后再考虑 4ep confirm.

如果全部失败, 不继续单纯调 cap 数值, 下一步转向 read margin/adaptive read 或 write support 控制.
