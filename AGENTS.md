# 项目协作规范

- 交互语言: 与仓库内的任何协作型 Agent 交互时, 以及与用户交互过程中, 请始终使用中文, 以保持沟通一致性.
- 输出编码: 所有需要写入文件或终端的文本, 请确保使用 UTF-8 编码, 以便正确显示中文内容并避免乱码.
- 标点符号: 文字可以用中文, 但是标点使用英文标点.
- 最小适配: 允许为了完成任务进行最小化修改适配, 包括修复 bug 或增加外围开关/脚本/报告适配, 但不得改变原有语义和机制原理.
- MQAR 实验记录: 对于完整执行到预期 final checkpoint 的 MQAR 正式实验, 需要将最终 epoch-end 结果追加记录到对应实验族的 canonical ledger. 当前 gd_residual_v1 rank/seed 实验记录在 `docs/artifacts/gd-residual-v1/rank-seed-effect-summary.csv`; 后续 GDN 模型超参调整实验也应建立或使用独立的 GDN 超参结果表, 例如 `docs/artifacts/gdn/gdn-hparam-effect-summary.csv`, 避免与 rank/seed 表混用. smoke/debug/失败/中断/未跑满预期 epoch 的不完整实验不必记录到正式结果表, 除非后续报告明确需要引用. 追加记录时必须保留 `configured_max_epochs`, `final_epoch`, `replicate_id`, `run_type`, `gpu`, `run_id`, `model_family`, `num_codebook_vectors`, `rank`, `seed`, `data_seed`, `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `effective_train_batch_size`, `batch_accum_profile` 以及对应实验族的关键超参字段, 不覆盖已有 run.
- dtype 默认策略: 后续完整 MQAR 正式实验中, Flash-VQG, GDN 等模型在 RTX 2080 Ti/sm75 上默认优先使用 float32 训练口径; 在支持 bf16 的 GPU 上默认优先使用 bfloat16 训练口径. 若模型或 kernel 不支持该 dtype, 可以 fallback, 但必须在报告和 artifact 中记录 fallback 原因, 实际 dtype policy, outer model dtype, attention/mixer/kernel 输入 dtype, GPU 型号与 compute capability.
- dtype 对比口径: 只有相同 dtype 训练口径的完整实验可以作为 official 直接质量对比. `float32`, `float16`, `bfloat16`, `auto`, `input` 等 dtype policy 或实际 kernel dtype 不同时, 结果只能作为 dtype probe, hardware profile 或 ablation 解释, 不得混入同一 official rank/seed/hparam 对比结论.
- GDN dtype 记录: GatedDeltaNet 的 FLA kernel dtype 可通过 `GDN_KERNEL_DTYPE=auto|input|float32|float16|bfloat16` 控制. 当前代码的历史默认 `auto` 行为是 CUDA sm80+ 使用 bf16, sm80 以下使用 fp16, CPU 使用 fp32; `input` 表示不做 GDN 内部 kernel dtype cast. 因此在 RTX 2080 Ti/sm75 上做后续 GDN official 可比实验时, 若尚未修改运行时默认策略, 需要显式设置 `GDN_KERNEL_DTYPE=float32`. 任何非 official dtype 或 dtype 诊断实验都必须在 artifact/report/summary 中记录该字段, 并注明它是 kernel dtype 口径还是全模型 dtype 口径.
