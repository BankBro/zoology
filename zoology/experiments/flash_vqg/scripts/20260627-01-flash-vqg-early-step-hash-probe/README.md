# 20260627-01 Flash-VQG early-step hash probe

本目录用于定位 canonical cache 后 `s123` 跨机器训练差异的早期来源.

- `run_probe.py`: 复用已有 generated config 构造模型和数据, 只执行少量 microbatch/optimizer step, 输出 hash JSON.
- `compare_probe_results.py`: 汇总多个 probe JSON, 标出同阶段 hash 是否一致.

raw 输出默认写入 `outputs/`, 不作为正式 MQAR ledger.
