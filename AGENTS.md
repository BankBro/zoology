# 项目协作规范

- 交互语言: 与仓库内的任何协作型 Agent 交互时, 以及与用户交互过程中, 请始终使用中文, 以保持沟通一致性.
- 输出编码: 所有需要写入文件或终端的文本, 请确保使用 UTF-8 编码, 以便正确显示中文内容并避免乱码.
- 标点符号: 文字可以用中文, 但是标点使用英文标点.
- 最小适配: 允许为了完成任务进行最小化修改适配, 包括修复 bug 或增加外围开关/脚本/报告适配, 但不得改变原有语义和机制原理.
- MQAR 实验记录: 对于完整执行到预期 final checkpoint 的 MQAR 正式实验, 需要将最终 epoch-end 结果追加记录到对应实验族的 canonical ledger. 当前 gd_residual_v1 rank/seed 实验记录在 `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`; 后续 GDN 模型超参调整实验也应建立或使用独立的 GDN 超参结果表, 例如 `docs/artifacts/gdn/gdn-hparam-effect-summary.csv`, 避免与 rank/seed 表混用. smoke/debug/失败/中断/未跑满预期 epoch 的不完整实验不必记录到正式结果表, 除非后续报告明确需要引用. 追加记录时必须保留 `configured_max_epochs`, `final_epoch`, `replicate_id`, `run_type`, `gpu`, `run_id`, `model_family`, `num_codebook_vectors`, `rank`, `seed`, `data_seed` 以及对应实验族的关键超参字段, 不覆盖已有 run.
