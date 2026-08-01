# Flash 后期退化因果诊断 Artifact

## 1. 结论

本实验在 RTX 3090 AMP BF16 下完成 36 个主队列作业, 8 条两 seed 四 epoch 正式训练, 208 条 standard/Longer-MQAR best/last 评估和 8 条 batch-size invariance 检查. 随后按预注册条件补充 2 条 seed123 fresh-per-epoch 四 epoch 训练.

根因矩阵确认, 从 `block32/local2` 改为 `block64/local2` 的后期退化主要来自`local_num_blocks`控制的近场/远场可见跨度由 64 扩到 128 token, 而不是 64-token 记忆边界本身. 这里的`local_num_blocks`同时改变local window和remote boundary offset, 因而不能进一步归因于其中某一个算子. fixed FLA 下该跨度因素两 seed均值为 `-0.068988`, default FLA 下完整 block因素均值为 `-0.093561`.

最快效率栈改为 `block64/local1` 后, 两 seed 1024x256 validation peak均值只比 `block64/local2` 低 `0.021062`, 但 final均值提高 `0.132854`, peak-to-final drop从 `-0.157150`缩小到`-0.003234`. 四个真正外推任务的 last宏平均从`0.092636`恢复到`0.451911`, 并略高于旧配置best的`0.441406`.

Fresh data没有消除退化. `factor-block`的drop仅从`-0.247516`缓解到`-0.200363`, final提高`0.031805`; 因此终态为`persistent_window_dynamics`, 不称为重复cache导致的传统过拟合.

事后路径覆盖审计显示, `block64/local2`将按真实microbatch等权口径计算的remote-required训练信号从`30.578%`降到`16.355%`, 而1024及更长评估几乎完全依赖remote路径. 当前最一致的机制解释是: 扩大的local窗口形成短训练分布捷径, 使remote路径监督减少并延迟, 随训练表现为长程能力选择性遗忘. 该解释有路径语义和覆盖率支持, 但尚未直接测量逐epoch路径质量或梯度, 因而不写成已完全证明的内部因果链.

## 2. 文件

| 文件 | 内容 |
|---|---|
| `training-curves.csv` | 正式训练与fresh补充训练的validation peak, final, drop和wall time |
| `mechanism.csv` | block, window, boundary和default FLA的配对因果效应 |
| `evaluation-summary.csv` | 两 seed best/last的standard与Longer宏平均 |
| `evaluation-detail.csv` | 208条best/last standard及Longer-MQAR评估记录 |
| `fresh-data.csv` | fixed-repeat与fresh-per-epoch配对结果 |
| `local-path-coverage.csv` | 正式训练cache与standard/Longer-MQAR的local-visible、remote-required结构审计 |
| `metadata.json` | commit, 作业数, invariance和数据manifest审计 |

## 3. Raw 边界

3090原始输出位于:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/
20260801-02-flash-late-degradation-causal-diagnosis/outputs/3090/
20260801-late-degradation-01/
```

主队列源码为 Zoology `68d9e8e52288cf83149630825d6df37f0ebe8450`, fresh-data补充源码为`1c295200e67cffd8599183cd39965f27a7c602b2`, Flash-VQG源码为`182180fd7a0770caf72b2dec6e6d27616dfd31a3`. Checkpoint和raw日志保留在3090, 不进入Git.
