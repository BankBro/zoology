# 项目协作规范

- 交互语言: 与仓库内的任何协作型 Agent 交互时, 以及与用户交互过程中, 请始终使用中文, 以保持沟通一致性.
- 输出编码: 所有需要写入文件或终端的文本, 请确保使用 UTF-8 编码, 以便正确显示中文内容并避免乱码.
- 标点符号: 文字可以用中文, 但是标点使用英文标点.
- 最小适配: 允许为了完成任务进行最小化修改适配, 包括修复 bug 或增加外围开关/脚本/报告适配, 但不得改变原有语义和机制原理.
- MQAR 实验记录: 对于完整执行到预期 final checkpoint 的 MQAR 正式实验, 需要将最终 epoch-end 结果追加记录到 `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`. smoke/debug/失败/中断/未跑满预期 epoch 的不完整实验不必记录到该表, 除非后续报告明确需要引用. 追加记录时必须保留 `configured_max_epochs`, `final_epoch`, `replicate_id`, `run_type`, `gpu`, `run_id`, `num_codebook_vectors`, `rank`, `seed`, `data_seed` 等区分字段, 不覆盖已有 run.
