# 当前最快 Flash 与 GDN 的 MQAR 正式对照计划

- `experiment_id`: `20260801-01-fastest-flash-vs-gdn-mqar`.
- Zoology实验分支: `20260801-121200-fastest-flash-vs-gdn-mqar`.
- Zoology base: `flash-vqg@3e51c62de13dea73034907bb020e16fe54f1c739`.
- Flash-VQG source: `20260801-103002-selected-read-native-load@396ae65b89b53aad316fbbf7daf55a92a551d684`.
- 正式机器: RTX 3090 GPU0, `Flash-VQG-tun`容器.
- 流程: `Plan -> 实验 -> Report`.

## 1. 研究问题与实验矩阵

本实验回答两个问题:

1. 当前最快Flash资源候选在标准MQAR和长度外推任务上相对GDN表现如何.
2. 当前最快Flash相对质量canonical损失了多少能力, 这些差异是否超出预注册的5个百分点容忍范围.

正式矩阵包含以下三组, 每组使用training seeds `123/124/125`, 共9条正式训练run.

| 实验组 | 核心配置 | 参数量 | 对照目的 |
|---|---|---:|---|
| `flash-fastest` | `baseline-r16-joint + A1 + K2 P8 + W2 direct + K3 VJP + G1 + F1` | `1160390` | 当前最快资源候选 |
| `flash-canonical` | `baseline-r16-joint + A1 + S1 exact` | `1160390` | 识别加速实现造成的质量变化 |
| `gdn` | `gdnxk-h2-ek4-ev4-usegate0` | `1335942` | capacity-matched外部baseline |

GDN与Flash的active state capacity均为`131072`, 但参数量不完全一致. 报告只称其为capacity-matched对照, 不表述为严格参数匹配.

## 2. 固定训练协议

- Training seeds为`123/124/125`, data seed为`123`.
- 两个Flash组共享canonical Flash init; GDN使用自己的canonical init.
- 复用canonical MQAR cache和固定epoch batch order.
- Train batch为`64`, validation batch为`16`, gradient accumulation为`4`, effective batch为`256`.
- 每组训练4 epochs, 每epoch 4次validation, early stopping关闭.
- 三组按相同examples, optimizer updates和epochs对齐, 不按wall time对齐.
- 正式dtype为AMP BF16, master weights和optimizer state保持FP32.
- TF32关闭, `TRITON_F32_DEFAULT=ieee`, `NVIDIA_TF32_OVERRIDE=0`.
- GDN显式设置`GDN_KERNEL_DTYPE=bfloat16`, 不使用`auto`.
- `last.pt`为主结果, `best.pt`为checkpoint选择敏感性结果.

300M资源配置中的`microbatch B4/GA8/T2048`不移植到小模型MQAR. MQAR使用既有`B64/GA4`质量协议, 只移植模型结构和kernel backend, 避免batch与训练暴露成为混杂变量.

### 2.1. Fastest Flash固定配置

```text
block_len=64
local_num_blocks=2
fox_gd_residual_rank=16
fox_gd_residual_write_topk=4
fox_remote_read_topk=16
fox_gd_residual_remat_mode=post_phase1
fox_gd_residual_builder=persistent_scan_triton
fox_gd_residual_persistent_tile_blocks=8
fox_gd_residual_grouped_chunk_backend=triton
fox_gd_residual_selected_read_backend=triton_remat
fox_gd_residual_selected_read_backward_backend=triton_state_owner_r1a_s1_w2
fox_gd_residual_selected_read_chunk_size=8192
fox_gd_residual_persistent_backward_backend=fixed_slot_vjp
fox_gd_residual_geometry_backend=head_grouped
fox_gd_residual_selected_read_forward_backend=hoisted_w2
fox_gd_residual_triton_input_policy=fp32_boundary
fox_gd_residual_selected_read_input_policy=fp32_boundary
fox_gd_residual_scan_read_fusion_backend=off
fox_gd_residual_persistent_host_empty_check=false
fox_gd_residual_checkpoint_preserve_rng_state=true
```

### 2.2. Canonical Flash固定配置

Canonical组保持相同block64, rank, read/write budget, remat和FP32 boundary, 但使用:

```text
fox_gd_residual_builder=grouped_chunk_torch_ref
fox_gd_residual_grouped_chunk_backend=triton
fox_gd_residual_selected_read_backward_backend=triton_deterministic_s1_head
fox_gd_residual_selected_read_chunk_size=8192
fox_gd_residual_persistent_backward_backend=autograd
fox_gd_residual_geometry_backend=event_gemv
fox_gd_residual_selected_read_forward_backend=query_w8
```

## 3. 正式评估协议

正式评估训练和eval dtype匹配, 仅使用BF16. 标准任务和长度曲线共13个逻辑case.

### 3.1. 标准MQAR

每项`1000` examples:

```text
64x4
64x8
64x16
128x32
256x64
512x64
512x128
1024x256
```

### 3.2. Longer-MQAR

每项`500` examples:

```text
1024x256
2048x512
4096x1024
8190x512
8190x2047
```

Longer中的`1024x256`保留为500-example长度曲线起点, 不与标准1000-example结果合并. 四个真正外推slice为`2048x512`, `4096x1024`, `8190x512`, `8190x2047`.

Eval batch按`128,64,32,16,8,4,2,1`搜索. Capacity search使用完整评估负载, 只有OOM允许自动降档. 选定batch必须与下一档batch完成prediction, accuracy和loss invariance检查. 五个Longer数据集必须匹配既有content hash.

`last.pt`和`best.pt`都进入逻辑评估矩阵. 如果二者model-state hash相同, 物理执行一次并保留两个逻辑角色.

## 4. 执行阶段与门禁

### 4.1. Preflight

- 核对两个仓库branch, commit和clean状态.
- 核对RTX 3090, CUDA/NVML, Python, PyTorch, Triton和FLA版本.
- 核对canonical cache, init file, init state和参数量.
- 核对三组resolved config, batch order, dtype和checkpoint路径.
- Fastest Flash必须审计A1, K2, W2, K3, G1和F1均生效.
- Canonical Flash必须审计S1生效且persistent路径未启用.
- 所有Flash runtime fallback必须为0.

### 4.2. Smoke与Q0

三组seed123分别在fresh process中完成train, backward, optimizer step, validation, checkpoint save, strict reload和fresh resume. 随后各训练1 epoch并评估`1024x256`及四个外推slice.

Q0只作为技术门禁. 相对质量差于5个百分点时写入预警, 但只要loss/gradient finite, 无fallback, checkpoint/resume正常, 仍继续三seed正式矩阵, 避免根据单seed结果选择性停止.

### 4.3. Formal

9条run在单张3090上串行执行. Run顺序按seed轮换模型顺序, 降低固定执行顺序与温度状态的系统偏差. 每条run在独立fresh process中启动. 中断后只有source/config/checkpoint identity全部匹配才允许从`resume.pt`恢复.

训练完成后执行full-load batch search, batch invariance, 全checkpoint 13-case eval和`1024x256`重复性检查.

### 4.4. 失败策略

- 非batch-search OOM, NaN, Inf, fallback, hash漂移或checkpoint错误立即fail-fast并保留现场.
- 实验runner或外围脚本bug允许在本实验分支最小修复并重跑受影响阶段.
- 若必须修改Flash模型源码, 暂停任务并先向用户报告根因和拟议修改.
- 完整可复现的负质量结果属于有效结果, 不因Flash未击败GDN而重跑或换seed.

## 5. 统计与决策口径

主结果使用`last.pt`, `best.pt`单独报告. 每项输出三个seed原始值, mean, population SD, min/max和同seed paired delta.

主要汇总包括:

- 标准8任务macro accuracy.
- 标准`1024x256` accuracy.
- 四个真正外推slice的macro accuracy.
- 五点长度曲线及相对`1024x256`的retention ratio.
- `flash-fastest - gdn`.
- `flash-fastest - flash-canonical`.
- `flash-canonical - gdn`.

Fastest Flash相对canonical的质量保留门槛为:

- 标准`1024x256`三seed均值下降不超过`0.05`.
- 标准任务宏平均和四外推宏平均下降均不超过`0.05`.
- 任一seed下降超过`0.10`时, 即使均值通过也标记为seed不稳定.

不使用`n=3`作强显著性结论. 可报告配对bootstrap区间作为描述性证据, 但必须同时展示每个seed原始值和`3/3` paired win计数.

## 6. 输出、Git与资源

- Runner: `zoology/experiments/flash_vqg/scripts/20260801-01-fastest-flash-vs-gdn-mqar/`.
- Generated configs: `zoology/experiments/flash_vqg/generated/<launch_id>/`.
- Raw outputs: 实验脚本目录下ignored `outputs/3090/<run_tag>/`.
- Artifact: `docs/artifacts/20260801-01-fastest-flash-vs-gdn-mqar/`.
- Report: `docs/20260801-01-fastest-flash-vs-gdn-mqar-report.md`.
- 正式训练行分别追加到Flash与GDN对应ledger.
- 更新Zoology和Flash-VQG的`STATUS.md`与`EXPERIMENT_LOG.md`.

Plan, runner实现和Report形成三个可区分提交并推送. 实验分支不自动合入Zoology `flash-vqg`.

预计总成本为`3–5`个RTX 3090 GPU-hours, checkpoint和raw产物低于约`1 GiB`. 大型checkpoint保留在3090, 轻量raw evidence镜像回当前工作区并校验SHA256.
