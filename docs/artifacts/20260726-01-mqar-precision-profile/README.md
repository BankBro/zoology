# 20260726-01 MQAR 低精度与长度泛化 artifact

本目录保存 `20260726-01-mqar-precision-profile` 的正式轻量可审计产物. 实验已完成 2080 Ti 12/12 和 RTX 3090 18/18 个正式训练 run, 共 30/30; 正式评估包含 2028 个逻辑 checkpoint-eval 事件, 其中 1066 个物理执行, 962 个因 best/last state hash 相同而去重.

## 结果文件

- `training.csv`: 30 个正式训练 run 的时间, dtype, scaler, kernel, 显存和 checkpoint 信息.
- `final.csv`: 标准 MQAR 与 longer-MQAR 的完整 train x eval dtype 逻辑网格.
- `combined/precision-grid-summary.csv`: 每台机器内按 3 seeds 计算的 mean 与 population SD; 不跨 GPU pooling.
- `canonical-training-ledger.csv`: 30 条正式 epoch-4 训练 canonical 记录.
- `canonical-longer-mqar-ledger.csv`: 780 条正式 longer-MQAR 逻辑评估 canonical 记录, 包含时间, GPU, batch, dtype, dataset/checkpoint hash 和物理去重状态.
- `source-manifest.csv`: 60 个 source checkpoint 的 file SHA256, 18 个双机 gate/status JSON 和30个resolved config的source/mirror SHA256.
- `metadata.json`: 收集状态, commit, cache 和计数摘要.
- `figures/matching-precision-{last,best}.{pdf,png}`: matching train/eval dtype 主图; PDF 为矢量格式, PNG 为 300 DPI.

## 证据与 raw 边界

`machines/2080ti/` 与 `machines/3090/` 镜像了各机 `preflight.json`, `status.json`, `formal-detail.json`, 全部gate JSON和30个resolved training config. 这些文件均通过source/mirror SHA256一致性校验.

大型 checkpoint, per-event progress/result, telemetry 和日志保留在各 source machine 的实验脚本 `outputs/machines/<machine>/` 下, 不提交到 Git. 它们的 source path 与 checkpoint file SHA256 记录在 manifest 和 canonical ledger 中.

## 口径与结论

本实验是独立 dtype probe, 不覆盖历史 FP32 canonical 推荐总表. Matching train/eval dtype 是主比较口径, off-diagonal 仅用于机制分析. 两台 GPU 分别统计 `n=3`, 不合并为 `n=6`.

GDN 的 matching accuracy 对训练精度近乎不敏感; Flash 的变化随 GPU 改变, 表明其训练轨迹存在更强的机器/数值路径敏感性. 在四个真正外推 slice 上, Flash 在全部 60/60 个 `GPU x dtype x seed x shape` 配对中高于 GDN; `1024x256` 训练端点不支持 Flash 优于 GDN. 详细数值和边界解释见 [正式报告](../../20260726-01-mqar-precision-profile-report.md).
