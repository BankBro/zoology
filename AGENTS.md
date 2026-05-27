# 项目协作规范

- 交互语言: 与仓库内的任何协作型 Agent 交互时, 以及与用户交互过程中, 请始终使用中文, 以保持沟通一致性.
- 输出编码: 所有需要写入文件或终端的文本, 请确保使用 UTF-8 编码, 以便正确显示中文内容并避免乱码.
- 标点符号: 文字可以用中文, 但是标点使用英文标点.
- 术语与标题: 除必要专业词汇, 代码标识符, 文件路径, 命令, 指标名外, 输出中的标题和解释性文本必须使用中文. 避免把普通说明写成英文标题, 例如用“GDN 容量和布局后续实验”而不是“GDN capacity/layout follow-up”.
- 最小适配: 允许为了完成任务进行最小化修改适配, 包括修复 bug 或增加外围开关/脚本/报告适配, 但不得改变原有语义和机制原理.
- MQAR 实验记录: 对于完整执行到预期 final checkpoint 的 MQAR 正式实验, 需要将最终 epoch-end 结果追加记录到对应实验族的 canonical ledger. 当前 gd_residual_v1 rank/seed 实验记录在 `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`; GDN 模型超参和 baseline 实验记录在 `docs/artifacts/gdn/gdn-hparam-effect-summary.csv`, 避免与 rank/seed 表混用. smoke/debug/失败/中断/未跑满预期 epoch 的不完整实验不必记录到正式结果表, 除非后续报告明确需要引用. 追加记录时必须保留 `configured_max_epochs`, `final_epoch`, `replicate_id`, `run_type`, `gpu`, `run_id`, `model_family`, `num_codebook_vectors`, `rank`, `seed`, `data_seed`, `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `effective_train_batch_size`, `batch_accum_profile` 以及对应实验族的关键超参字段, 不覆盖已有 run.
- 实验时间记录: 对所有后续 MQAR 相关正式实验统一记录时间信息, 包括完整执行到预期 final checkpoint 的 MQAR 正式训练实验, 以及正式 longer-MQAR eval. 正式记录至少包括 `started_at_utc`, `ended_at_utc`, `wall_clock_sec`, `gpu`, `gpu_name`, `status`. smoke/debug/失败/中断/未跑满预期 epoch 的实验不写入正式结果 ledger 也可以, 但必须在 artifact/status/report 中记录时间, 状态和失败原因.
- dtype 默认策略: 后续完整 MQAR 正式实验中, Flash-VQG, GDN 等模型在 RTX 2080 Ti/sm75 上默认优先使用 float32 训练口径; 在支持 bf16 的 GPU 上默认优先使用 bfloat16 训练口径. 若模型或 kernel 不支持该 dtype, 可以 fallback, 但必须在报告和 artifact 中记录 fallback 原因, 实际 dtype policy, outer model dtype, attention/mixer/kernel 输入 dtype, GPU 型号与 compute capability.
- dtype 对比口径: 只有相同 dtype 训练口径的完整实验可以作为 official 直接质量对比. `float32`, `float16`, `bfloat16`, `auto`, `input` 等 dtype policy 或实际 kernel dtype 不同时, 结果只能作为 dtype probe, hardware profile 或 ablation 解释, 不得混入同一 official rank/seed/hparam 对比结论.
- GDN dtype 记录: GatedDeltaNet 的 FLA kernel dtype 可通过 `GDN_KERNEL_DTYPE=auto|input|float32|float16|bfloat16` 控制. 当前代码的历史默认 `auto` 行为是 CUDA sm80+ 使用 bf16, sm80 以下使用 fp16, CPU 使用 fp32; `input` 表示不做 GDN 内部 kernel dtype cast. 因此在 RTX 2080 Ti/sm75 上做后续 GDN official 可比实验时, 若尚未修改运行时默认策略, 需要显式设置 `GDN_KERNEL_DTYPE=float32`. 任何非 official dtype 或 dtype 诊断实验都必须在 artifact/report/summary 中记录该字段, 并注明它是 kernel dtype 口径还是全模型 dtype 口径.

## 多机 IP 与容器路径

### 局域网机器表

| 机器名 | 局域网 IP | 别称 | 说明 |
|---|---|---|---|
| `mclab-3090` | `192.168.2.114` | `3090` | 已确认可 SSH, 3090 GPU 机器 |
| `mclab-2080ti` | `192.168.2.131` | `2080ti` | 当前 zoology 容器所在宿主机 |
| `mclab` | `192.168.2.188` | `4090` | 4090 GPU 机器 |

### Flash-VQG-tun 容器路径表

Flash-VQG 和 zoology 项目可能运行在多台机器的 `Flash-VQG-tun` 容器内. 容器内项目路径通常形如 `/home/lyj/mnt/project/...`, 但每台宿主机对应的真实宿主机路径可能不同, 不得跨机器套用.

| 宿主机 | 容器名 | 宿主机路径 | 容器路径 | 状态 |
|---|---|---|---|---|
| `mclab-3090` (`192.168.2.114`) | `Flash-VQG-tun` | `/mnt/980pro/lyj` | `/home/lyj/mnt` | 已确认 |
| `mclab-2080ti` (`192.168.2.131`) | `Flash-VQG-tun` | `/mnt/WD40EZRZ/lyj` | `/home/lyj/mnt` | 已确认 |
| `mclab` (`192.168.2.188`) | `Flash-VQG-tun` | 待确认 | 待确认 | 未记录 |

已确认的项目路径映射:

| 宿主机 | 容器内路径 | 宿主机路径 |
|---|---|---|
| `mclab-3090` | `/home/lyj/mnt/project` | `/mnt/980pro/lyj/project` |
| `mclab-3090` | `/home/lyj/mnt/project/Flash-VQG` | `/mnt/980pro/lyj/project/Flash-VQG` |
| `mclab-3090` | `/home/lyj/mnt/project/zoology` | `/mnt/980pro/lyj/project/zoology` |
| `mclab-2080ti` | `/home/lyj/mnt/project` | `/mnt/WD40EZRZ/lyj/project` |
| `mclab-2080ti` | `/home/lyj/mnt/project/Flash-VQG` | `/mnt/WD40EZRZ/lyj/project/Flash-VQG` |
| `mclab-2080ti` | `/home/lyj/mnt/project/zoology` | `/mnt/WD40EZRZ/lyj/project/zoology` |

### IP 与路径确认规则

- 宿主机 IP 识别约束: 涉及多机, 容器, 宿主机, NAT, TUN 或代理时, 不得仅凭容器内 `hostname -I`, Docker 网关, `/proc/net/route`, mihomo fake-ip/TUN 地址推断宿主机局域网 IP.
- 出口 IP 反查方法: 如需确认当前容器经宿主机访问局域网目标时的出口 IP, 必须使用 `lyj` 用户 SSH 到已知可达的局域网机器反查, 例如 `ssh lyj@192.168.2.114 'printf "SSH_CLIENT=%s\n" "$SSH_CLIENT"; hostname; hostname -I'`, 以 `SSH_CLIENT` 第一字段作为对端看到的来源 IP.
- 多出口处理: 若多台机器反查来源 IP 不一致, 必须报告多网卡, 多出口或策略路由可能性, 不得武断合并为单一宿主机 IP.
- 容器路径确认: 如需确认其他机器的 `Flash-VQG-tun` 路径映射, 必须在对应宿主机执行 `docker inspect Flash-VQG-tun` 查看 `.Mounts`, 不得凭路径名称推断.
