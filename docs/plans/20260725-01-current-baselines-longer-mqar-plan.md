# 当前基线 Longer-MQAR 三阶段执行计划

- `experiment_id`: `20260725-01-current-baselines-longer-mqar`.
- 实验分支: `20260725-01-current-baselines-longer-mqar`, 从 `flash-vqg` 派生.
- 流程: `Plan -> 实验 -> Report`.

## 1. 阶段边界

### 1.1. Plan

本文件必须先独立完成检查、提交和推送. Plan 提交不得包含 runner、smoke、checkpoint 或实验结果. Plan 提交后自动进入实验阶段, 不再等待额外人工批准.

### 1.2. 实验

实验阶段依次完成 runner 实现、静态测试、六条训练 run 的逐项 smoke、缩小版端到端队列 smoke、六条 4ep 正式训练、checkpoint 审计、全部 checkpoint 的五 slice eval smoke、formal eval 和 repro. 无人值守阶段只写 ignored raw、checkpoint、日志和状态, 不提前修改正式 ledger、artifact 或 report.

如果 smoke 迫使实验改变模型、seed、init、数据、dtype、slice、checkpoint 选择或正式门槛, 必须先修改并提交本 Plan, 再启动 formal.

### 1.3. Report

自动队列写出 `DONE.json` 后等待用户唤醒. Agent 必须先审计完成性、hash、时间和统计, 再生成正式 artifact、canonical ledger 增量、report、`EXPERIMENT_LOG` 和 `STATUS` 更新.

## 2. 正式训练口径

### 2.1. 模型

- Flash-VQG: `baseline-r16-joint`, `gd_rank=16`, `num_codebook_vectors=64`, `read_topk=16`, `write_topk=4`, `smooth_p4` update softcap `0.5`, injection warmup `0->512` optimizer steps, `triton + triton_remat` backend.
- GDN: `gdnxk-h2-ek4-ev4-usegate0`, `num_heads=2`, `expand_k=4`, `expand_v=4`, `use_gate=false`, active state capacity `131072`.
- 两模型只复用各自 canonical init, 不跨模型加载参数.

### 2.2. Run 矩阵

- Training seeds: `123`, `124`, `125`.
- Data seed: `123`.
- 每个模型 3 条, 共 6 条正式 run.
- 两个模型分别固定自己的 canonical seed124 init, 三个 training seed 只改变训练 RNG.
- `train_batch_size=64`, `eval_batch_size=16`, `gradient_accumulation_steps=4`, effective batch `256`.
- `max_epochs=4`, 每 epoch 4 次 validation, early stopping 关闭.
- 全模型 FP32, TF32 关闭, `TRITON_F32_DEFAULT=ieee`, `GDN_KERNEL_DTYPE=float32`.
- 正式机器固定 2080 Ti GPU1, 环境固定 `/home/lyj/miniconda3/envs/flash-vqg-fla042`.

### 2.3. 固定输入身份

- Flash init 文件 SHA256: `26bf2cb0b44c8a32cfd44977f609e4314ab53dfa6b6a0e8abfbffd10191ec878`.
- Flash init model-state hash: `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0`.
- GDN init 文件 SHA256: `a4e76e7776bdc83a582c2613cd7d9782100a9148aa119763ecaaeeb8273f7b71`.
- GDN init model-state hash: `bdba0c19b2530c72c3ae7dd6bd708901c2369f6d3e1da9d850ea8347d5ea60a6`.
- Canonical MQAR cache content hash: `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- Epoch 0–3 batch-order hash:
  - epoch0: `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320`.
  - epoch1: `b9d52c40883bf347d481b8d0b79141885643f4f554bbe7016acd1b1e3d69b7c4`.
  - epoch2: `5d31531aafcb4a4383a2ac711fbc9c0b2727e95c48b12d1902f0bb22cc3b6f20`.
  - epoch3: `6ae4c4584b2b365741cb9973e714825e75c138c4c8af40406333f7e612f42839`.

## 3. Checkpoint 与 Longer-MQAR 口径

- `last.pt` 是预注册主结果.
- `best.pt` 从 4 个 epoch-end 中按整体 `valid/accuracy` 选择, 作为完整敏感性结果.
- 禁止根据 Longer-MQAR slice 事后挑选 checkpoint.
- 若 last/best 的 model-state hash 相同, 物理评估一次并保留两个逻辑角色.
- Eval 固定 `eval_seed=123`, vocab `8192`, `random_non_queries=true`, `power_a=0.01`, 每 slice `500` examples.
- Formal slices:
  - `1024x256`.
  - `2048x512`.
  - `4096x1024`.
  - `8190x512`.
  - `8190x2047`.
- Batch search 候选为 `32,16,8,4,2,1`; 只有 batch-search candidate OOM 允许自动降档.
- 每个唯一 checkpoint 必须重跑一次 `1024x256`; dataset hash 必须完全一致, accuracy 差值不得超过 `1e-12`.

## 4. Formal 启动硬门槛

### 4.1. 六条训练 run 逐项预检

预先生成 6-run job manifest. 每条 resolved config 必须逐项验证 model family、核心超参、seed、data seed、4ep、batch、dtype、init、cache、四个 epoch batch order、参数量、capacity、run ID 和 checkpoint 路径. GPU/NVML、`torch.cuda.is_available()`、Python、PyTorch、CUDA、Triton 和 FLA 版本任一不符即停止.

### 4.2. 六条训练 run 逐项 smoke

每条 run 必须以自身 seed 和正式 init 在独立 fresh process 中实际完成 forward、CE、backward、optimizer step、validation、last/best checkpoint 保存和 strict reload. Shape smoke 覆盖 `T64/T128/T256` 训练 shape 及 `T64–T1024` 常规 validation shape. 6/6 均 completed 且无 OOM、NaN、Inf、Traceback 或 checkpoint 异常, 才能生成 `TRAINING_SMOKE_PASSED`.

### 4.3. 端到端 smoke DAG

Formal 前必须以截断参数自动跑通:

```text
preflight
-> 6 条训练 smoke
-> checkpoint 审计
-> source manifest
-> batch-search smoke
-> 五 slice eval smoke
-> repro smoke
-> raw summary
```

缩小队列无需人工干预到达 `SMOKE_DONE` 后, 才能启动正式训练.

### 4.4. Formal eval 前门槛

六条训练必须全部到达 epoch4. 每个 last/best 必须记录文件 hash、model-state hash、epoch 和指标并 strict load. Batch search 后, 每个唯一 checkpoint × 5 slices 必须使用选定 batch 完成至少一个完整 batch 的 fresh-process eval smoke. 全部通过并生成 `EVAL_SMOKE_PASSED` 后, 才能进入 500-example formal eval.

## 5. 自动队列与失败策略

单一 tmux 队列必须自动运行:

```text
全量预检
-> 六条训练 smoke
-> 端到端 smoke DAG
-> 六条 4ep 正式训练
-> checkpoint/source manifest 审计
-> 全 checkpoint 五 slice eval smoke
-> formal eval
-> repro
-> raw summary
-> DONE.json
```

- 队列不得包含交互确认或需要 Agent 重新调用的阶段.
- 原子状态文件记录 phase、current run、完成数、预期数、时间、退出码和日志路径.
- 队列支持幂等恢复, 但只有 config、result 和 checkpoint hash 全部验证的 completed run 才允许跳过.
- 非预期失败立即写 `FAILED.json` 并停止, 不自动重试或继续产生残缺正式矩阵.
- Agent 监控到全部 smoke 门槛通过、Flash seed123 完成 step176 validation、后续 train loop 持续推进且 GPU 稳定后即可退出会话.
- 退出前必须报告有效 run 数、状态路径、日志、checkpoint 根目录、预计剩余时间和恢复检查命令.

## 6. 统计与正式结论

- 分别汇总 last 和 best 的三 seed mean、population std、min/max、相对 `1024x256` 的绝对下降与 retention ratio.
- 计算同 seed Flash-GDN paired delta.
- 每个 slice 的定性规则:
  - 3/3 paired delta 均为正: `稳健领先`.
  - mean delta 为正但不足 3/3: `混合领先`.
  - mean delta 不为正: `不支持 Flash 领先`.
- 不把实验成功定义为 Flash 必须获胜. 完整、可复现的负结果同样属于成功实验.
- Report 只比较当前两种模型, 不把 20260526 历史模型混入排名.
- 本轮先形成长度泛化证据, 不自动替换当前综合 baseline.

## 7. 交付与记录

- 正式 artifact: `docs/artifacts/20260725-01-current-baselines-longer-mqar/`.
- Longer-MQAR 当前基线索引: `docs/artifacts/longer-mqar/` 下独立目录或索引条目, 不改写旧 official-core 表.
- Report: `docs/20260725-01-current-baselines-longer-mqar-report.md`.
- Flash 4ep训练行追加到 `docs/artifacts/gd-residual-v1/`.
- GDN 4ep训练行追加到 `docs/artifacts/gdn-expanded-k/`.
- 更新 `docs/EXPERIMENT_LOG.md`; `docs/STATUS.md` 只增加新证据和链接, 不自动更换 baseline.
- 大型 checkpoint 和 raw 日志原位保留, source manifest 记录路径、大小和 SHA256.
- Plan、runner实现、Report收尾必须形成三个可区分的提交阶段.

## 8. 时间估计

- Plan 检查与提交: `20–40` 分钟.
- Runner 实现、测试和全量 smoke: `1–1.5` 小时.
- 自动正式队列: `3.5–5` 小时.
- 用户唤醒后的 Report 审计与收尾: `30–60` 分钟.

## 9. 3090 跨 GPU 扩展

本节在2080 Ti阶段完成后追加. 扩展继续使用同一 `experiment_id`, 但机器级训练、eval、artifact和状态必须明确隔离, 不把同seed跨机器结果当作独立seed合并统计.

### 9.1. 3090正式口径

- 3090在 `mclab-3090` 的 `Flash-VQG-tun` 容器GPU0独立重训Flash/GDN × seeds `123/124/125`, 共6条4ep run, 不复用历史checkpoint.
- 模型、各自canonical seed124 init、data seed `123`、B64/GA4、每epoch 4次validation、best/last选择和五个500-example Longer-MQAR slice与2080 Ti阶段完全一致.
- 为形成直接跨GPU对照, 3090仍固定全模型FP32、TF32 off、`TRITON_F32_DEFAULT=ieee`、`GDN_KERNEL_DTYPE=float32`, 不使用sm86默认bf16策略.
- 已在扩展plan前确认两机本轮实际13个MQAR cache文件逐文件SHA256一致, 两个canonical init文件SHA256一致. 正式preflight还必须复核tensor content hash、四个epoch batch-order hash、环境、源码和resolved config.
- 两机resolved config归一化后只允许机器、路径、launch/run identity和provenance字段不同; 模型、训练、数据和数值字段必须一致.

### 9.2. 输出隔离与自动队列

- 2080 Ti既有ignored raw保留在实验脚本的legacy `outputs/`, 不移动或改写内部绝对路径.
- 3090 raw固定写入 `outputs/machines/3090/`; checkpoint root和launch ID带 `3090` 后缀, 不覆盖2080 Ti结果.
- runner提供 `--machine 2080ti|3090` 机器接口. 3090启动脚本固定GPU0、同一Conda解释器和独立tmux session.
- 3090严格复用完整自动DAG: `preflight -> shape smoke -> 6训练smoke -> eval smoke DAG -> 6 formal训练 -> checkpoint审计 -> 全source五slice smoke -> 500-example formal -> repro -> DONE`.
- 所有run使用fresh process. 非预期失败fail-fast; 仅batch-search OOM允许按 `32,16,8,4,2,1` 降档. Formal只能在3090自己的 `TRAINING_SMOKE_PASSED` 和 `SMOKE_DONE` 通过后启动.
- Agent监控到全部3090 smoke通过、首个Flash s123完成step176 validation并继续训练、GPU/NVML稳定后才退出等待.

### 9.3. Artifact与统计结构

正式artifact重构为机器级和合并级三层:

```text
docs/artifacts/20260725-01-current-baselines-longer-mqar/
├── machines/2080ti/
├── machines/3090/
├── combined/
└── figures/
```

- 两个 `machines/<machine>/` 子目录各自保存training、Longer-MQAR detail/summary、paired delta、checkpoint role、source manifest、batch size、repro、verification和metadata.
- `combined/longer-mqar-detail.csv` 固定为 `2 machines × 2 models × 3 seeds × 2 roles × 5 slices = 120` 条逻辑formal结果.
- 合并统计按机器分别计算三seed mean/population std和Flash-GDN同seed paired delta; 不把同seed在两张GPU上的结果合成 `n=6`.
- `combined/cross-machine-deltas.csv` 计算同model、同seed、同role、同slice的 `3090-2080ti` delta.
- 3090的轻量raw evidence和机器artifact在远端审核后镜像回主工作区, 逐文件SHA256验证; 大型checkpoint留在3090原位并由source manifest追踪.
- Flash/GDN canonical训练ledger分别追加3条3090行, 使用cross-GPU replicate标识, 不覆盖2080 Ti正式行.

### 9.4. 跨GPU双图

- 当前2080 Ti单机曲线由两张跨GPU图替代: `longer-mqar-accuracy-last` 和 `longer-mqar-accuracy-best`.
- 每张图包含Flash 2080 Ti、Flash 3090、GDN 2080 Ti、GDN 3090共4条曲线. 颜色和marker区分模型, 实线实心/虚线空心区分GPU, 并轻微水平错位避免误差棒遮挡.
- 图使用全部五个slice, 每条曲线为对应机器三个training seeds的mean ± population std, 固定标注checkpoint role、`n=3`、500 examples、FP32和“训练/eval均在对应GPU完成”.
- 绘图CSV必须包含machine、model、role、slice、mean/std和三个seed原始值; 图中数值与combined summary误差小于 `1e-12`.

### 9.5. 扩展验收

- 3090训练6/6到epoch4, best/last审计通过, 全source五slice smoke、formal和repro完成.
- 五个formal dataset hash在2080 Ti和3090完全一致; repro accuracy delta不超过 `1e-12`.
- 两个机器子目录各自验证通过, combined detail为120行且无重复machine/model/seed/role/slice键.
- 两图各有4条曲线, PDF字体嵌入, PNG至少300 DPI, 彩色和灰度均可区分.
- Plan扩展、runner实现和最终跨GPU Report形成可区分的提交并推送.

### 9.6. Formal eval OOM失败与恢复设计

- 3090的6/6正式训练和checkpoint审计完成后, 首次formal eval在Flash s123的`8190x512`、batch 32处OOM. 失败发生在500-example多batch执行中; 原batch-search仅使用32 examples, 因而只验证了单batch显存, 没有覆盖跨batch allocator碎片化.
- 不采用仅设置allocator环境变量的概率性缓解, 也不为单个GPU/slice硬编码batch 16. Formal模式batch-search改为使用与正式事件相同的500 examples, 并继续仅在batch-search阶段按`32,16,8,4,2,1`处理OOM降档.
- 恢复时保留旧OOM record和日志. Formal验收只统计新batch-search选中的`unique checkpoint × slice` reference矩阵, 避免旧失败尝试造成重复计数; collector仍保留所有raw尝试供审计.
- 重新启动完整自动队列后, preflight、shape smoke、训练smoke和端到端smoke必须再次通过. 已完成训练只有在config、result、checkpoint文件hash和model-state hash全部匹配时才允许跳过.
